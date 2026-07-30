"""Tests for Step 2: Clinical SafetyController and Step 3: CDRE."""
from __future__ import annotations

import numpy as np
import pytest

from aura.common.mathx import softmax
from aura.schemas.contracts import (
    DecisionReadinessProfile,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    MitigationAction,
    ReadinessState,
    ReasoningTrace,
    RelationType,
    Recommendation,
    SafetyControllerOutput,
)
from aura.services.safety.controller import ClinicalSafetyController, compute_safety_confidence
from aura.services.safety.readiness import ClinicalDecisionReadinessEngine, _js_divergence


# --------------------------------------------------------------------------- #
# Safety Controller (Step 2)
# --------------------------------------------------------------------------- #
class TestComputeSafetyConfidence:
    def test_at_threshold(self):
        assert abs(compute_safety_confidence(2.5, 2.5) - 0.5) < 1e-6

    def test_below_threshold_high(self):
        assert compute_safety_confidence(1.0, 2.5) > 0.5

    def test_above_threshold_low(self):
        assert compute_safety_confidence(4.0, 2.5) < 0.5

    def test_output_range(self):
        for v in np.linspace(-5, 10, 50):
            val = compute_safety_confidence(v, 2.5, scale=1.0)
            assert 0.0 <= val <= 1.0


class TestClinicalSafetyController:
    def setup_method(self):
        self.controller = ClinicalSafetyController()

    def test_init_loads_policy(self):
        assert self.controller.policy is not None
        assert self.controller.policy.ood_threshold > 0

    def test_in_distribution_passes(self):
        logits = np.array([2.0, 0.5, 0.2, 0.1, -0.3, -0.5])
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=2.0,
            epistemic_std=0.05, epistemic_mi=0.01,
        )
        assert out.state in ("PASSED", "WARNING")
        assert out.safety_confidence > 0
        assert isinstance(out, SafetyControllerOutput)

    def test_ood_detection(self):
        """Very negative logits produce high energy = OOD.

        Energy = -T * logsumexp(logits/T).  When logits are very negative,
        logsumexp ≈ max(logits) + log(K), and energy becomes positive (high).
        z = (energy - ood_mean) / ood_std  > threshold triggers OOD.
        """
        logits = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=1.0,
            epistemic_std=0.03, epistemic_mi=0.005,
        )
        ood_check = [c for c in out.checks if c.name == "ood_energy"][0]
        assert not ood_check.passed
        assert out.state == "FAILED"

    def test_epistemic_failure(self):
        logits = np.array([1.0, 0.5, 0.2, 0.1, -0.3, -0.5])
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=2.0,
            epistemic_std=0.5, epistemic_mi=0.3,  # very high
        )
        epi_check = [c for c in out.checks if c.name == "epistemic"][0]
        assert not epi_check.passed
        assert out.state == "FAILED"

    def test_data_integrity_aspect_ratio(self):
        logits = np.array([1.0, 0.5, 0.2, 0.1, -0.3, -0.5])
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=2.0,
            epistemic_std=0.05, epistemic_mi=0.01,
            image_shape=(64, 512),  # aspect ratio 8.0 — extreme
            aspect_ratio=8.0,
        )
        di_check = [c for c in out.checks if c.name == "data_integrity"][0]
        assert not di_check.passed

    def test_data_quality_failure(self):
        logits = np.array([1.0, 0.5, 0.2, 0.1, -0.3, -0.5])
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=2.0,
            epistemic_std=0.05, epistemic_mi=0.01,
            image_quality=0.1,  # very low quality
        )
        qual_check = [c for c in out.checks if c.name == "data_quality"]
        assert len(qual_check) == 1
        assert not qual_check[0].passed

    def test_mitigations_generated(self):
        logits = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0])  # high energy = OOD
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=1.0,
            epistemic_std=0.5, epistemic_mi=0.3,
        )
        assert len(out.mitigations) > 0
        assert MitigationAction.ESC_HUMAN_EXPERT_REVIEW in out.mitigations

    def test_checks_structure(self):
        logits = np.array([1.0, 0.5, 0.2, 0.1, -0.3, -0.5])
        ev = np.random.default_rng(42).random(8)
        out = self.controller.check(
            evidence_vector=ev, logits=logits, temperature=1.0,
            ood_mean=-5.0, ood_std=2.0,
            epistemic_std=0.05, epistemic_mi=0.01,
        )
        assert len(out.checks) >= 3  # data_integrity, ood, epistemic
        for c in out.checks:
            assert c.name in ("data_integrity", "ood_energy", "epistemic", "data_quality")
            assert isinstance(c.passed, bool)


# --------------------------------------------------------------------------- #
# CDRE (Step 3)
# --------------------------------------------------------------------------- #
class TestJSDivergence:
    def test_identical(self):
        p = np.array([0.3, 0.5, 0.2])
        assert _js_divergence(p, p) < 1e-6

    def test_different(self):
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.05, 0.9, 0.05])
        assert _js_divergence(p, q) > 0.1

    def test_output_range(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = np.array([0.4, 0.3, 0.2, 0.1])
        js = _js_divergence(p, q)
        assert 0.0 <= js <= 1.0


class TestClinicalDecisionReadinessEngine:
    def setup_method(self):
        self.engine = ClinicalDecisionReadinessEngine()

    def _make_reasoning(self, steps=None, adjusted=None):
        adj = adjusted or {"pneumonia": 0.45, "normal": 0.25, "copd": 0.3}
        return ReasoningTrace(
            study_id="STU-001",
            prior_posterior={"pneumonia": 0.3, "normal": 0.4, "copd": 0.3},
            adjusted_posterior=adj,
            steps=steps or [],
            guideline_citations=["IDSA/ATS CAP guideline"],
        )

    def _make_graph(self):
        g = EvidenceGraph()
        g.add_node(EvidenceNode(id="ev.consolidation", kind="imaging_finding",
                                label="consolidation", value=0.7, modality="imaging", confidence=0.8))
        g.add_node(EvidenceNode(id="hx.wbc", kind="structured_prior",
                                label="wbc", value=12.5, modality="labs"))
        g.add_node(EvidenceNode(id="dx.pneumonia", kind="structured_prior",
                                label="pneumonia", value=0.45, modality="hypothesis"))
        g.add_edge(EvidenceEdge(source_id="ev.consolidation", target_id="dx.pneumonia",
                                relation=RelationType.SUPPORTS, weight=1.0))
        g.add_edge(EvidenceEdge(source_id="hx.wbc", target_id="dx.pneumonia",
                                relation=RelationType.SUPPORTS, weight=0.5))
        return g

    def test_evaluate_returns_profile(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        recs = [Recommendation(action="order_bnp_echo", display="BNP",
                               expected_info_gain=0.3, cost_tier="medium",
                               risk_tier="none", utility=0.5, rationale="Test")]
        drp = self.engine.evaluate(trace, graph, recs, vision_quality=0.85)
        assert isinstance(drp, DecisionReadinessProfile)
        assert len(drp.dimensions) >= 5

    def test_state_is_valid(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        drp = self.engine.evaluate(trace, graph, [])
        assert drp.state in (ReadinessState.READY, ReadinessState.CONDITIONALLY_READY, ReadinessState.NOT_READY)

    def test_limiting_factor_identified(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        drp = self.engine.evaluate(trace, graph, [])
        assert drp.limiting_factor != ""
        assert 0 <= drp.limiting_score <= 1

    def test_no_graph_returns_neutral(self):
        trace = self._make_reasoning()
        drp = self.engine.evaluate(trace, None, [])
        assert drp.s_consistency == 0.5  # no data = neutral

    def test_full_evidence_boosts_coverage(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        # Add more evidence
        for name in ("sym.fever", "sym.productive_cough", "hx.procalcitonin"):
            graph.add_node(EvidenceNode(id=name, kind="structured_prior",
                                        label=name.split(".")[-1], value=0.8,
                                        modality="symptoms" if "sym" in name else "labs"))
        drp = self.engine.evaluate(trace, graph, [])
        assert drp.s_coverage > 0.3  # better with more evidence

    def test_consensus_perfect_without_quantum(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        drp = self.engine.evaluate(trace, graph, [], classical_logits=None, quantum_logits=None)
        assert drp.consensus_agreement_index == 1.0

    def test_consensus_with_quantum(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        c_logits = np.array([2.0, 0.5, 0.2, 0.1, -0.3, -0.5])
        q_logits = np.array([1.8, 0.6, 0.2, 0.1, -0.3, -0.5])  # close to classical
        drp = self.engine.evaluate(trace, graph, [],
                                   classical_logits=c_logits, quantum_logits=q_logits)
        assert 0.8 < drp.consensus_agreement_index <= 1.0

    def test_recommendation_summary(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        drp = self.engine.evaluate(trace, graph, [])
        assert isinstance(drp.recommendation_summary, str)
        assert len(drp.recommendation_summary) > 0

    def test_dimension_scores_in_range(self):
        trace = self._make_reasoning()
        graph = self._make_graph()
        drp = self.engine.evaluate(trace, graph, [])
        for dim in drp.dimensions:
            assert 0.0 <= dim.score <= 1.0
