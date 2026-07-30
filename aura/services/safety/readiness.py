"""Clinical Decision Readiness Engine (CDRE) — Layer 2.

Computes a multi-dimensional Decision Readiness Profile (DRP) after reasoning
and recommendation. Each dimension quantifies one aspect of clinical readiness;
the limiting dimension determines the overall state.

Dimensions:
  * S_coverage  — evidence coverage against guideline templates
  * S_quality   — evidence freshness and image quality
  * S_consistency — supporting vs. refuting edges in the evidence graph
  * Expected Decision Value — remaining value of information from recommendations
  * S_robustness — leave-one-out stability of the reasoning posterior
  * Consensus Agreement Index — JS-divergence between classical and quantum heads
  * S_consensus — multi-agent panel consensus entropy (1.0 - normalized entropy)
"""
from __future__ import annotations

import numpy as np

from aura.common.config import get_safety_policy
from aura.common.mathx import entropy, softmax
from aura.knowledge.guidelines.templates import coverage_ratio
from aura.schemas.contracts import (
    DecisionReadinessProfile,
    EvidenceGraph,
    ReadinessDimension,
    ReadinessState,
    ReasoningTrace,
    RelationType,
)
from aura.services.agent.consensus import ConsensusResult


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


class ClinicalDecisionReadinessEngine:
    """Computes the Decision Readiness Profile for a completed case."""

    def __init__(self):
        self.policy = get_safety_policy()
        self.model_version = "cdre-v1"

    def evaluate(
        self,
        reasoning: ReasoningTrace,
        evidence_graph: EvidenceGraph | None,
        recommendations: list,
        vision_quality: float | None = None,
        fusion_model=None,
        evidence_vector: np.ndarray | None = None,
        classical_logits: np.ndarray | None = None,
        quantum_logits: np.ndarray | None = None,
        consensus_result: ConsensusResult | None = None,
    ) -> DecisionReadinessProfile:
        """Evaluate all readiness dimensions and produce the DRP."""
        dims: list[ReadinessDimension] = []

        # 1. Evidence Coverage
        s_cov = self._coverage(reasoning, evidence_graph)
        dims.append(ReadinessDimension(
            name="coverage", score=round(s_cov, 4),
            detail=f"Guideline template coverage: {s_cov:.1%}",
        ))

        # 2. Evidence Quality (image quality as proxy for freshness)
        s_qual = self._quality(vision_quality, evidence_graph)
        dims.append(ReadinessDimension(
            name="quality", score=round(s_qual, 4),
            detail=f"Evidence quality score: {s_qual:.3f}",
        ))

        # 3. Evidence Consistency
        s_cons = self._consistency(evidence_graph)
        dims.append(ReadinessDimension(
            name="consistency", score=round(s_cons, 4),
            detail=f"Graph consistency ratio: {s_cons:.3f}",
        ))

        # 4. Expected Decision Value
        edv = self._expected_decision_value(recommendations)
        dims.append(ReadinessDimension(
            name="expected_decision_value", score=round(min(edv * 10, 1.0), 4),
            detail=f"EDV from remaining recommendations: {edv:.4f}",
        ))

        # 5. Clinical Reasoning Robustness
        s_rob = self._robustness(reasoning, evidence_vector, fusion_model)
        dims.append(ReadinessDimension(
            name="robustness", score=round(s_rob, 4),
            detail=f"LOO robustness score: {s_rob:.3f}",
        ))

        # 6. Consensus Agreement Index
        cai = self._consensus_agreement(classical_logits, quantum_logits)
        dims.append(ReadinessDimension(
            name="consensus_agreement", score=round(cai, 4),
            detail=f"Classical-quantum agreement (1 - JS): {cai:.3f}",
        ))

        # 7. Multi-Agent Panel Consensus (S_consensus)
        s_consensus = self._panel_consensus(consensus_result)
        dims.append(ReadinessDimension(
            name="panel_consensus", score=round(s_consensus, 4),
            detail=f"Multi-agent panel consensus: {s_consensus:.3f}",
        ))

        # Identify limiting dimension
        if dims:
            limiting = min(dims, key=lambda d: d.score)
            limiting_name = limiting.name
            limiting_score = limiting.score
        else:
            limiting_name = ""
            limiting_score = 1.0

        # Determine state
        min_score = limiting_score
        if min_score >= self.policy.min_coverage:
            state = ReadinessState.READY
        elif min_score >= self.policy.min_coverage * 0.7:
            state = ReadinessState.CONDITIONALLY_READY
        else:
            state = ReadinessState.NOT_READY

        # Recommendation summary
        summary = self._recommendation_summary(state, limiting_name, limiting_score)

        return DecisionReadinessProfile(
            state=state,
            dimensions=dims,
            limiting_factor=limiting_name,
            limiting_score=round(limiting_score, 4),
            s_coverage=round(s_cov, 4),
            s_quality=round(s_qual, 4),
            s_consistency=round(s_cons, 4),
            s_robustness=round(s_rob, 4),
            expected_decision_value=round(edv, 4),
            consensus_agreement_index=round(cai, 4),
            recommendation_summary=summary,
            model_version=self.model_version,
        )

    # ---- Dimension evaluators ---- #

    def _coverage(self, reasoning: ReasoningTrace,
                  evidence_graph: EvidenceGraph | None) -> float:
        """Evidence coverage against guideline templates (S_coverage)."""
        if not reasoning.adjusted_posterior:
            return 0.0

        top_dx = max(reasoning.adjusted_posterior, key=reasoning.adjusted_posterior.get)

        # Collect available evidence from the graph
        available_imaging: list[str] = []
        available_labs: list[str] = []
        available_symptoms: list[str] = []

        if evidence_graph is not None:
            for nid, node in evidence_graph.nodes.items():
                if node.modality == "imaging" and node.value >= 0.4:
                    available_imaging.append(node.label)
                elif node.modality == "labs":
                    available_labs.append(node.label)
                elif node.modality == "symptoms" and node.value > 0:
                    available_symptoms.append(node.label)

        # Also include evidence from reasoning steps
        for step in reasoning.steps:
            for ev_name in step.evidence:
                parts = ev_name.split(".", 1)
                if len(parts) == 2:
                    mod, name = parts
                    if mod == "imaging" and name not in available_imaging:
                        available_imaging.append(name)
                    elif mod == "labs" and name not in available_labs:
                        available_labs.append(name)
                    elif mod == "symptoms" and name not in available_symptoms:
                        available_symptoms.append(name)

        return coverage_ratio(
            top_dx,
            available_imaging=available_imaging,
            available_labs=available_labs,
            available_symptoms=available_symptoms,
        )

    def _quality(self, vision_quality: float | None,
                 evidence_graph: EvidenceGraph | None) -> float:
        """Evidence quality combining image quality and freshness (S_quality)."""
        iq = vision_quality if vision_quality is not None else 0.8

        # Freshness penalty for evidence graph nodes (all are current pipeline output)
        freshness = 1.0
        if evidence_graph is not None and evidence_graph.nodes:
            node_values = [n.confidence for n in evidence_graph.nodes.values() if n.confidence > 0]
            if node_values:
                freshness = float(np.mean(node_values))

        return 0.6 * iq + 0.4 * freshness

    def _consistency(self, evidence_graph: EvidenceGraph | None) -> float:
        """Ratio of supporting vs. refuting edges in the evidence graph (S_consistency)."""
        if evidence_graph is None or not evidence_graph.edges:
            return 0.5  # no data = neutral

        supporting = sum(1 for e in evidence_graph.edges if e.relation == RelationType.SUPPORTS)
        refuting = sum(1 for e in evidence_graph.edges if e.relation in (
            RelationType.REFUTES, RelationType.CONTRADICTS))
        total = supporting + refuting
        if total == 0:
            return 0.5
        return supporting / total

    def _expected_decision_value(self, recommendations: list) -> float:
        """Remaining expected decision value from outstanding recommendations."""
        if not recommendations:
            return 0.0
        utilities = [getattr(r, "utility", 0) or 0 for r in recommendations]
        return float(max(utilities)) if utilities else 0.0

    def _robustness(self, reasoning: ReasoningTrace,
                    evidence_vector: np.ndarray | None,
                    fusion_model=None) -> float:
        """Leave-one-out robustness of the reasoning posterior (S_robustness).

        Uses the reasoning step effects as a proxy for LOO influence:
        if the adjusted posterior is dominated by one step, robustness is low.
        """
        if not reasoning.steps or not reasoning.adjusted_posterior:
            return 0.5

        top_dx = max(reasoning.adjusted_posterior, key=reasoning.adjusted_posterior.get)

        # Compute clinical weights from guideline step effects
        clinical_effects: dict[str, float] = {}
        for step in reasoning.steps:
            for dx, lr in step.effect.items():
                if dx.value == top_dx:
                    for ev in step.evidence:
                        clinical_effects[ev] = clinical_effects.get(ev, 0.0) + abs(lr)

        if not clinical_effects:
            return 0.5

        total_weight = sum(clinical_effects.values())
        if total_weight < 1e-9:
            return 0.5

        # Normalized clinical weights
        clinical_w = np.array([v / total_weight for v in clinical_effects.values()])

        # Simulate LOO: if one evidence is removed, how much does the posterior shift?
        # Approximate: the influence is the max single-evidence weight
        max_influence = float(np.max(clinical_w))

        # Robustness = 1 - max_influence (lower single-step dependency = more robust)
        return max(0.0, min(1.0, 1.0 - max_influence))

    def _panel_consensus(self, consensus_result: ConsensusResult | None) -> float:
        """Multi-agent panel consensus (S_consensus). 1.0 - normalized consensus entropy."""
        if consensus_result is None:
            return 1.0
        entropy = consensus_result.consensus_entropy
        s = 1.0 - min(1.0, entropy * 2.0)
        return max(0.0, min(1.0, s))

    def _consensus_agreement(self, classical_logits: np.ndarray | None,
                             quantum_logits: np.ndarray | None) -> float:
        """Consensus Agreement Index — 1 - JS(classical, quantum)."""
        if classical_logits is None or quantum_logits is None:
            return 1.0  # no quantum head = perfect agreement by default

        c_probs = softmax(np.asarray(classical_logits, dtype=float))
        q_probs = softmax(np.asarray(quantum_logits, dtype=float))
        js = _js_divergence(c_probs, q_probs)
        return max(0.0, min(1.0, 1.0 - js))

    def _recommendation_summary(self, state: ReadinessState,
                                limiting: str, score: float) -> str:
        """Human-readable summary of the readiness state."""
        labels = {
            "coverage": "Evidence Coverage",
            "quality": "Evidence Quality",
            "consistency": "Evidence Consistency",
            "expected_decision_value": "Expected Decision Value",
            "robustness": "Reasoning Robustness",
            "consensus_agreement": "Model Consensus",
            "panel_consensus": "Multi-Agent Panel Consensus",
        }
        label = labels.get(limiting, limiting)
        if state == ReadinessState.READY:
            return "All readiness dimensions above threshold. Decision is supported."
        elif state == ReadinessState.CONDITIONALLY_READY:
            return (
                f"Conditionally ready — {label} is the limiting factor "
                f"(score {score:.2f}). Proceed with caution."
            )
        else:
            return (
                f"NOT READY — {label} critically low (score {score:.2f}). "
                f"Additional evidence or review required before committing."
            )

    def evaluate_coverage(self, primary_dx, findings_map: dict, study) -> float:
        from aura.knowledge.guidelines.templates import GUIDELINE_TEMPLATES
        template = GUIDELINE_TEMPLATES.get(primary_dx.value if hasattr(primary_dx, "value") else str(primary_dx))
        if not template:
            return 1.0
        
        total = len(template.imaging) + len(template.symptoms) + len(template.labs)
        if total == 0:
            return 1.0

        known = 0
        for f in template.imaging:
            f_str = f.value if hasattr(f, "value") else str(f)
            if f_str in findings_map or f in findings_map:
                known += 1
                
        if study.multimodal is not None:
            for s in template.symptoms:
                known += 1
            for l in template.labs:
                if getattr(study.multimodal.labs, l, None) is not None:
                    known += 1

        return float(known / total)
