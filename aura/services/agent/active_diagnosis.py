"""Active Diagnostic Agent — the novel capability.

Every deployed chest-X-ray AI is one-shot: image in, probabilities out. This agent
is sequential. It reasons about its OWN uncertainty and closes it:

    1. Estimate the diagnosis posterior P(Y | evidence) and its entropy H (bits).
    2. If confident (H low or top probability high) -> COMMIT.
    3. Else, score every candidate next action (extra view, lab, prior films) by
       Expected Information Gain and pick the single action that removes the most
       uncertainty per unit cost (reusing services.recommend's EVOI/EIG engine).
    4. "Acquire" it -> the resolved evidence sharpens the posterior -> repeat.
    5. If nothing informative remains and it is still uncertain -> ABSTAIN.

Quantum, honestly
-----------------
The posterior comes from the fusion model. With the quantum backend, P(Y | evidence)
is the measurement distribution of the 8-qubit variational circuit (one qubit per
evidence channel, see services.fusion.evidence). The **entropy of that quantum
measurement distribution is the uncertainty this agent minimises** — so the quantum
circuit is the uncertainty engine, not a (dis)proven accuracy claim.

Purely additive: reads the fusion model + RecommendEngine, mutates only a local copy
of the evidence vector. Nothing in the served pipeline is modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from common.mathx import entropy, softmax
from schemas.clinical import DIAGNOSES
from services.fusion.evidence import EVIDENCE_CHANNELS
from services.recommend.engine import CATALOG, RecommendEngine

# action -> the evidence channels it resolves (from the recommend catalog).
_ACTION_CHANNELS: dict[str, list[str]] = {c["action"]: c["channels"] for c in CATALOG}


@dataclass
class DiagnosisStep:
    step: int
    posterior: dict[str, float]
    entropy_bits: float
    top: list[tuple[str, float]]          # [(diagnosis, prob), ...] top-2
    confident: bool
    decision: Optional[str] = None        # COMMIT / ABSTAIN when the loop ends here
    action_display: Optional[str] = None  # test acquired to leave this step
    action_eig_bits: Optional[float] = None
    resolved: list[tuple[str, float]] = field(default_factory=list)
    rationale: Optional[str] = None


@dataclass
class DiagnosisTrajectory:
    steps: list[DiagnosisStep]
    committed: bool
    status: str                            # commit | provisional | abstain
    final_diagnosis: str
    final_probability: float
    initial_entropy: float
    final_entropy: float
    bits_resolved: float
    n_tests: int
    backend: str

    def summary(self) -> str:
        return (f"{self.status.upper()}: {self.final_diagnosis} p={self.final_probability:.2f} | "
                f"uncertainty {self.initial_entropy:.2f}->{self.final_entropy:.2f} bits "
                f"({self.bits_resolved:+.2f}) over {self.n_tests} acquired test(s) "
                f"[{self.backend}]")


class ActiveDiagnosisAgent:
    """Sequential information-gain diagnosis on the 8-channel evidence vector."""

    def __init__(self, fusion_model, recommend: RecommendEngine | None = None,
                 entropy_target_bits: float = 0.6, confidence: float = 0.85,
                 max_tests: int = 3):
        self.fusion = fusion_model                    # exposes .logits(x)
        self.recommend = recommend or RecommendEngine()
        self.entropy_target = float(entropy_target_bits)
        self.confidence = float(confidence)
        self.max_tests = int(max_tests)
        self.backend = getattr(fusion_model, "backend", getattr(fusion_model, "name", "fusion"))

    def _posterior(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.fusion.logits(x))

    def _entropy(self, x: np.ndarray) -> float:
        backend = getattr(self.fusion, "backend", None)
        model = getattr(self.fusion, "model", self.fusion)
        if backend == "quantum" and hasattr(model, "measurement_entropy"):
            return model.measurement_entropy(x)
        if hasattr(self.fusion, "measurement_entropy"):
            return self.fusion.measurement_entropy(x)
        return float(entropy(self._posterior(x)))

    def _is_confident(self, post: np.ndarray, H: float) -> bool:
        return (H <= self.entropy_target) or (float(post.max()) >= self.confidence)

    def _snapshot(self, x: np.ndarray, step: int) -> tuple[DiagnosisStep, np.ndarray, float]:
        post = self._posterior(x)
        H = self._entropy(x)
        order = np.argsort(post)[::-1]
        top = [(DIAGNOSES[i].value, round(float(post[i]), 4)) for i in order[:2]]
        s = DiagnosisStep(
            step=step,
            posterior={DIAGNOSES[i].value: round(float(post[i]), 4) for i in order},
            entropy_bits=round(H, 4),
            top=top,
            confident=self._is_confident(post, H),
        )
        return s, post, H

    def diagnose(self, x0: np.ndarray,
                 resolve_fn: Optional[Callable[[str, float], float]] = None
                 ) -> DiagnosisTrajectory:
        """Run the acquire-until-confident loop.

        ``resolve_fn(channel, current_prob) -> 0.0|1.0`` supplies the real test
        result when known; the default resolves each uncertain channel to its
        most-likely outcome (deterministic demo of the uncertainty collapse).
        """
        x = np.asarray(x0, dtype=float).copy()
        steps: list[DiagnosisStep] = []
        status = "abstain"

        for t in range(self.max_tests + 1):
            step, post, H = self._snapshot(x, t)
            dx = step.top[0][0]

            # Confident -> commit outright.
            if step.confident:
                step.decision = f"COMMIT — {dx} (confident)"
                status = "commit"
                steps.append(step)
                break

            # Out of test budget while still uncertain -> abstain to a human.
            if t == self.max_tests:
                step.decision = f"ABSTAIN — residual uncertainty after {t} test(s); refer to radiologist"
                status = "abstain"
                steps.append(step)
                break

            # Pick the single most informative next action (skip multi-test panels).
            recs = self.recommend.recommend(self.fusion, x)
            singles = [r for r in recs if not r.action.startswith("acquire_panel:")]
            best = max(singles, key=lambda r: r.expected_info_gain) if singles else None

            # "Acquire" it: resolve its uncertain channels to definite values.
            resolved: list[tuple[str, float]] = []
            if best is not None:
                for c in _ACTION_CHANNELS.get(best.action, []):
                    j = EVIDENCE_CHANNELS.index(c)
                    if 0.08 < x[j] < 0.92:                   # only genuinely uncertain channels
                        val = float(resolve_fn(c, x[j])) if resolve_fn else float(round(x[j]))
                        x[j] = val
                        resolved.append((c, val))

            # No informative test left: commit PROVISIONALLY to the leading dx,
            # flagged low-confidence — distinct from abstaining mid-workup.
            if not resolved:
                step.decision = (f"PROVISIONAL — {dx} (low confidence; no further test "
                                 f"would help — recommend radiologist review)")
                status = "provisional"
                steps.append(step)
                break

            step.action_display = best.display
            step.action_eig_bits = round(float(best.expected_info_gain), 4)
            step.resolved = resolved
            step.rationale = best.rationale
            steps.append(step)

        committed = status in ("commit", "provisional")
        return DiagnosisTrajectory(
            steps=steps,
            committed=committed,
            status=status,
            final_diagnosis=steps[-1].top[0][0],
            final_probability=steps[-1].top[0][1],
            initial_entropy=steps[0].entropy_bits,
            final_entropy=steps[-1].entropy_bits,
            bits_resolved=round(steps[0].entropy_bits - steps[-1].entropy_bits, 4),
            n_tests=sum(1 for s in steps if s.action_display is not None),
            backend=str(self.backend),
        )
