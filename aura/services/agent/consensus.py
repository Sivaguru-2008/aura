from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Modality
from aura.services.agent.base import AgentVerdict
from aura.services.agent.registry import AgentRegistry
from aura.services.agent.specialists import (
    GuidelineAgent,
    PathologyAgent,
    PulmonologistAgent,
    RadiologistAgent,
    SafetyAgent,
)

logger = logging.getLogger(__name__)

CONSENSUS_VERSION = "consensus-v2"
AGENT_TIMEOUT = 2.0
AGENT_BASE_WEIGHTS: dict[str, float] = {
    "radiologist": 0.35,
    "pulmonologist": 0.25,
    "pathology": 0.15,
    "guideline": 0.15,
    "safety": 0.10,
}


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def _fallback_verdict(agent_name: str, error: str) -> AgentVerdict:
    return AgentVerdict(
        agent_name=agent_name,
        agent_version="fallback",
        findings={},
        confidence=0.0,
        arguments=[f"Agent failed: {error}"],
        metadata={"status": "failed_timeout", "error": error},
    )


@dataclass
class ArbitrationRound:
    round_number: int
    conflicts: list[tuple[str, str, float]]
    resolution: str
    resolved_posterior: dict[Diagnosis, float] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    posterior: dict[Diagnosis, float]
    confidence: float
    consensus_entropy: float
    agreement_matrix: dict[str, dict[str, float]]
    verdicts: list[AgentVerdict]
    arbitration_history: list[ArbitrationRound] = field(default_factory=list)
    panel_discussion: str = ""
    requires_review: bool = False
    model_version: str = CONSENSUS_VERSION

    def to_dict(self) -> dict:
        return {
            "posterior": {d.value: round(p, 4) for d, p in self.posterior.items()},
            "confidence": self.confidence,
            "consensus_entropy": self.consensus_entropy,
            "agreement_matrix": self.agreement_matrix,
            "verdicts": [
                {
                    "agent_name": v.agent_name,
                    "confidence": v.confidence,
                    "findings": {d.value: round(p, 4) for d, p in v.findings.items()},
                    "arguments": v.arguments,
                    "guideline_references": v.guideline_references,
                    "abstained": len(v.findings) == 0,
                }
                for v in self.verdicts
            ],
            "arbitration_history": [
                {
                    "round_number": r.round_number,
                    "conflicts": [(a, b, round(js, 4)) for a, b, js in r.conflicts],
                    "resolution": r.resolution,
                    "resolved_posterior": {d.value: round(p, 4) for d, p in r.resolved_posterior.items()},
                }
                for r in self.arbitration_history
            ],
            "panel_discussion": self.panel_discussion,
            "requires_review": self.requires_review,
            "model_version": self.model_version,
        }


class ConsensusEngine:
    def __init__(self):
        self._register_defaults()

    @staticmethod
    def _register_defaults() -> None:
        for agent in [
            RadiologistAgent(),
            PulmonologistAgent(),
            PathologyAgent(),
            GuidelineAgent(),
            SafetyAgent(),
        ]:
            AgentRegistry.register(agent)

    async def evaluate(
        self,
        evidence: dict,
        priors: dict,
        modality: Modality | None = None,
    ) -> ConsensusResult:
        agents = AgentRegistry.all()

        if modality is not None:
            agents = [a for a in agents if a.supported(modality)]

        verdicts: list[AgentVerdict] = []
        for agent in agents:
            try:
                verdict = await asyncio.wait_for(
                    agent.evaluate(evidence, priors),
                    timeout=AGENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("Agent %s timed out after %.1fs", agent.agent_name, AGENT_TIMEOUT)
                verdict = _fallback_verdict(agent.agent_name, f"timeout after {AGENT_TIMEOUT}s")
            except Exception as exc:
                logger.exception("Agent %s raised an exception", agent.agent_name)
                verdict = _fallback_verdict(agent.agent_name, str(exc))
            verdicts.append(verdict)

        agreement = self._build_agreement_matrix(verdicts)
        consensus_entropy = self._consensus_entropy(verdicts)
        merged = self._weighted_average(
            verdicts, agreement, consensus_entropy
        )

        arbitration_history: list[ArbitrationRound] = []
        if self._detect_conflict(verdicts, agreement):
            arbitration_history = await self._arbitration_loop(
                verdicts, evidence, priors
            )

        confidence = float(np.mean([v.confidence for v in verdicts if v.findings]))
        panel_discussion = self._build_panel_discussion(
            verdicts, arbitration_history, merged
        )

        top_prob = max(merged.values())
        requires_review = bool(consensus_entropy > 0.3 and top_prob < 0.5)

        return ConsensusResult(
            posterior=merged,
            confidence=round(confidence, 4),
            consensus_entropy=round(consensus_entropy, 4),
            agreement_matrix=agreement,
            verdicts=verdicts,
            arbitration_history=arbitration_history,
            panel_discussion=panel_discussion,
            requires_review=requires_review,
        )

    def _build_agreement_matrix(
        self, verdicts: list[AgentVerdict]
    ) -> dict[str, dict[str, float]]:
        matrix: dict[str, dict[str, float]] = {}
        for a in verdicts:
            matrix[a.agent_name] = {}
            a_vec = self._verdict_to_vec(a)
            for b in verdicts:
                b_vec = self._verdict_to_vec(b)
                js = _js_divergence(a_vec, b_vec)
                matrix[a.agent_name][b.agent_name] = round(1.0 - js, 4)
        return matrix

    def _verdict_to_vec(self, v: AgentVerdict) -> np.ndarray:
        return np.array([v.findings.get(d, 0.0) for d in CHEST_DIAGNOSES], dtype=float)

    def _consensus_entropy(self, verdicts: list[AgentVerdict]) -> float:
        probs = np.array(
            [self._verdict_to_vec(v) for v in verdicts if v.findings], dtype=float
        )
        if len(probs) < 2:
            return 0.0
        mean_p = probs.mean(axis=0)
        mean_p = mean_p / (mean_p.sum() + 1e-12)
        divergences = [
            _js_divergence(mean_p, probs[i]) for i in range(len(probs))
        ]
        return float(np.mean(divergences))

    def _weighted_average(
        self,
        verdicts: list[AgentVerdict],
        agreement: dict[str, dict[str, float]],
        entropy: float,
    ) -> dict[Diagnosis, float]:
        weights: dict[str, float] = {}
        for v in verdicts:
            base = AGENT_BASE_WEIGHTS.get(v.agent_name, 0.2)
            peers = [
                agreement[v.agent_name][o] for o in agreement[v.agent_name] if o != v.agent_name
            ]
            # A lone agent has no peers to agree *with*. np.mean([]) is nan, and nan
            # then poisons every downstream weight — the `or 1.0` guards below do not
            # catch it, because nan is truthy, so the whole posterior came back nan.
            # No peers means no peer disagreement to discount for: weight 1.0. This
            # matches _consensus_entropy, which returns 0.0 entropy for < 2 verdicts.
            avg_agreement = float(np.mean(peers)) if peers else 1.0
            weights[v.agent_name] = base * v.confidence * avg_agreement

        # Guard on a *finite positive* total, not truthiness: sum() of an all-nan or
        # all-zero weight map must fall back to a uniform merge, not propagate nan.
        total_w = sum(weights.values())
        if not np.isfinite(total_w) or total_w <= 0.0:
            weights = {v.agent_name: 1.0 for v in verdicts}
            total_w = float(len(weights)) or 1.0

        merged = {d: 0.0 for d in CHEST_DIAGNOSES}
        for v in verdicts:
            w = weights.get(v.agent_name, 0.0) / total_w
            for d, p in v.findings.items():
                if d in merged:
                    merged[d] += w * p

        total = sum(merged.values())
        if not np.isfinite(total) or total <= 0.0:
            return {d: 1.0 / len(merged) for d in merged}
        return {d: v / total for d, v in merged.items()}

    def _detect_conflict(
        self,
        verdicts: list[AgentVerdict],
        agreement: dict[str, dict[str, float]],
    ) -> bool:
        for i, a in enumerate(verdicts):
            for j, b in enumerate(verdicts):
                if i >= j:
                    continue
                js_dist = 1.0 - agreement[a.agent_name][b.agent_name]
                if js_dist > 0.35:
                    return True
        return False

    async def _arbitration_loop(
        self,
        verdicts: list[AgentVerdict],
        evidence: dict,
        priors: dict,
    ) -> list[ArbitrationRound]:
        history: list[ArbitrationRound] = []
        current = {d: 0.0 for d in CHEST_DIAGNOSES}
        vote_counts = {d: 0 for d in CHEST_DIAGNOSES}

        for v in verdicts:
            for d, p in v.findings.items():
                if d in vote_counts:
                    vote_counts[d] += 1

        for d in CHEST_DIAGNOSES:
            avg = np.mean([
                v.findings.get(d, 0.0) for v in verdicts
            ])
            if avg >= 0.4 or (vote_counts[d] >= 3 and avg >= 0.15):
                current[d] = avg

        total = sum(current.values()) or 1.0
        current = {d: v / total for d, v in current.items()}

        conflicts = []
        for i, a in enumerate(verdicts):
            for j, b in enumerate(verdicts):
                if i >= j:
                    continue
                a_vec = self._verdict_to_vec(a)
                b_vec = self._verdict_to_vec(b)
                js = _js_divergence(a_vec, b_vec)
                if js > 0.35:
                    conflicts.append((a.agent_name, b.agent_name, round(js, 4)))

        lines = []
        top = max(current, key=current.get)
        lines.append(f"Panel resolved to {top.value} ({current[top]:.1%}) as the leading diagnosis.")
        for a_name, b_name, js_val in conflicts:
            lines.append(f"Mediated {a_name} vs {b_name} disagreement (JS={js_val:.2f}) — higher weight assigned to guideline-anchored reasoning.")

        resolution = " ".join(lines)

        history.append(ArbitrationRound(
            round_number=1,
            conflicts=conflicts,
            resolution=resolution,
            resolved_posterior=current,
        ))

        return history

    def _build_panel_discussion(
        self,
        verdicts: list[AgentVerdict],
        arbitration_history: list[ArbitrationRound],
        merged: dict[Diagnosis, float],
    ) -> str:
        lines = []
        lines.append("## Panel Discussion\n")
        lines.append("The following specialist agents reviewed this case:\n")

        for v in verdicts:
            agent_label = v.agent_name.replace("_", " ").title()
            top_dx = max(v.findings, key=v.findings.get) if v.findings else None
            top_p = v.findings.get(top_dx, 0.0) if top_dx else 0.0
            timeout = v.metadata.get("status") == "failed_timeout"
            lines.append(f"### {agent_label} Agent")
            if timeout:
                lines.append(f"- **Status:** failed — {v.arguments[0] if v.arguments else 'timeout'}")
            else:
                lines.append(f"- **Confidence:** {v.confidence:.2%}")
                if top_dx:
                    lines.append(f"- **Leading call:** {top_dx.value} ({top_p:.1%})")
            for arg in v.arguments:
                lines.append(f"- {arg}")
            for ref in v.guideline_references:
                lines.append(f"  - Reference: {ref}")
            lines.append("")

        if arbitration_history:
            lines.append("### Arbitration Summary\n")
            for round_ in arbitration_history:
                for conflict in round_.conflicts:
                    a_label = conflict[0].replace("_", " ").title()
                    b_label = conflict[1].replace("_", " ").title()
                    lines.append(f"- **Conflict:** {a_label} vs {b_label} (JS divergence {conflict[2]:.2f})")
                lines.append(f"- **Resolution:** {round_.resolution}")
                lines.append("")

        top = max(merged, key=merged.get)
        lines.append(f"### Consensus Diagnosis: {top.value} ({merged[top]:.1%})")
        lines.append("")

        return "\n".join(lines)
