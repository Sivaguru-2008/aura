"""Tests for Decision Assurance Engine and Adaptive Conformal Inference (ACI)."""
from collections import deque
import numpy as np
import pytest

from aura.schemas.clinical import Diagnosis
from aura.schemas.contracts import (
    CaseBundle,
    CaseState,
    DecisionAssurance,
    DecisionReadinessProfile,
    DecisionVerdict,
    Prediction,
    ReadinessDimension,
    ReadinessState,
    SafetyAssessment,
    SafetyControllerOutput,
)
from aura.services.safety.aci import ACIState, AdaptiveConformalInference, MultiStreamACI
from aura.services.safety.decision import DecisionAssuranceEngine


def _dummy_safety(
    top_p: float = 0.85,
    top_dx: Diagnosis = Diagnosis.NORMAL,
    conformal_set: list[Diagnosis] | None = None,
    ood_energy: float = 0.2,
    is_ood: bool = False,
    epistemic: float = 0.05,
    abstained: bool = False,
) -> SafetyAssessment:
    if conformal_set is None:
        conformal_set = [top_dx]
    preds = [
        Prediction(diagnosis=top_dx, probability=top_p),
        Prediction(diagnosis=Diagnosis.PNEUMONIA, probability=round(1.0 - top_p, 4)),
    ]
    return SafetyAssessment(
        study_id="test-study",
        predictions=preds,
        top=top_dx,
        top_probability=top_p,
        conformal_set=conformal_set,
        conformal_coverage=0.90,
        conformal_method="marginal",
        epistemic_uncertainty=epistemic,
        ood_energy=ood_energy,
        is_ood=is_ood,
        abstained=abstained,
    )


class TestDecisionAssuranceEngine:
    def test_verdict_accept(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.88,
            conformal_set=[Diagnosis.NORMAL],
            ood_energy=0.2,
            epistemic=0.04,
        )
        readiness = DecisionReadinessProfile(
            state=ReadinessState.READY,
            limiting_factor="none",
            limiting_score=0.95,
        )
        controller = SafetyControllerOutput(state="PASSED")

        da = engine.evaluate(safety, readiness=readiness, controller=controller)
        assert da.verdict == DecisionVerdict.ACCEPT
        assert not da.requires_human_review
        assert da.limiting_factor == "none"
        assert len(da.conformal_set) == 1
        assert "routine_clinical_signoff" in da.mitigations

    def test_verdict_review_required_due_to_moderate_confidence(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.52,  # in [0.30, 0.65)
            conformal_set=[Diagnosis.NORMAL, Diagnosis.PNEUMONIA],
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.REVIEW_REQUIRED
        assert da.requires_human_review
        assert da.limiting_factor == "moderate_confidence"
        assert "secondary_radiologist_review" in da.mitigations

    def test_verdict_review_required_due_to_competing_hypotheses(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.75,
            conformal_set=[Diagnosis.NORMAL, Diagnosis.PNEUMONIA, Diagnosis.HEART_FAILURE],  # 3 candidates
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.REVIEW_REQUIRED
        assert da.requires_human_review
        assert da.limiting_factor == "competing_hypotheses"

    def test_verdict_review_required_due_to_borderline_ood(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.80,
            conformal_set=[Diagnosis.NORMAL],
            ood_energy=1.2,  # in [1.0, 1.5)
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.REVIEW_REQUIRED
        assert da.requires_human_review
        assert da.limiting_factor == "borderline_ood"

    def test_verdict_review_required_due_to_elevated_epistemic(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.80,
            conformal_set=[Diagnosis.NORMAL],
            epistemic=0.15,  # in [0.12, 0.20)
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.REVIEW_REQUIRED
        assert da.requires_human_review
        assert da.limiting_factor == "elevated_epistemic"

    def test_verdict_review_required_due_to_readiness_state(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(top_p=0.85, conformal_set=[Diagnosis.NORMAL])
        readiness = DecisionReadinessProfile(
            state=ReadinessState.CONDITIONALLY_READY,
            limiting_factor="quality",
            limiting_score=0.60,
        )
        da = engine.evaluate(safety, readiness=readiness)
        assert da.verdict == DecisionVerdict.REVIEW_REQUIRED
        assert da.requires_human_review
        assert "readiness_quality" in da.limiting_factor

    def test_verdict_abstain_due_to_severe_ood(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.85,
            ood_energy=2.8,  # > 1.5
            is_ood=True,
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.ABSTAIN
        assert da.requires_human_review
        assert da.limiting_factor == "out_of_distribution"
        assert "esc_human_expert_review" in da.mitigations

    def test_verdict_abstain_due_to_low_confidence(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.22,  # < 0.30
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.ABSTAIN
        assert da.requires_human_review
        assert da.limiting_factor == "low_confidence"

    def test_verdict_abstain_due_to_large_conformal_set(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.35,
            conformal_set=[
                Diagnosis.NORMAL,
                Diagnosis.PNEUMONIA,
                Diagnosis.HEART_FAILURE,
                Diagnosis.PNEUMOTHORAX,
                Diagnosis.MALIGNANCY,
            ],  # 5 candidates > 4
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.ABSTAIN
        assert da.requires_human_review
        assert da.limiting_factor == "large_conformal_set"

    def test_verdict_abstain_due_to_high_epistemic(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(
            top_p=0.70,
            epistemic=0.25,  # >= 0.20
        )
        da = engine.evaluate(safety)
        assert da.verdict == DecisionVerdict.ABSTAIN
        assert da.requires_human_review
        assert da.limiting_factor == "high_epistemic_uncertainty"

    def test_verdict_abstain_due_to_safety_controller_veto(self):
        engine = DecisionAssuranceEngine()
        safety = _dummy_safety(top_p=0.80)
        controller = SafetyControllerOutput(state="FAILED")
        da = engine.evaluate(safety, controller=controller)
        assert da.verdict == DecisionVerdict.ABSTAIN
        assert da.requires_human_review
        assert da.limiting_factor == "safety_controller_veto"


class TestAdaptiveConformalInference:
    def test_initialization_and_serialization(self):
        st = ACIState(qhat=0.85, alpha=0.10, gamma=0.03, t=10, window=100)
        st.recent.append(0.0)
        st.recent.append(1.0)
        row = st.to_row()
        assert row["qhat"] == 0.85
        assert row["t"] == 10
        assert row["recent"] == [0.0, 1.0]

        restored = ACIState.from_row(row)
        assert restored.qhat == 0.85
        assert restored.t == 10
        assert list(restored.recent) == [0.0, 1.0]

    def test_aci_coverage_convergence(self):
        """Verify empirical coverage converges toward 1 - alpha for stationary stream."""
        rng = np.random.default_rng(42)
        n_steps = 400
        target_coverage = 0.90
        alpha = 1.0 - target_coverage
        aci = AdaptiveConformalInference(coverage=target_coverage, gamma=0.02, window=200)

        # Generate predictions from a 4-class distribution
        for _ in range(n_steps):
            raw = rng.dirichlet(np.array([2.0, 1.0, 0.5, 0.5]))
            probs = raw / raw.sum()
            true_idx = int(rng.choice(4, p=probs))
            aci.update(probs, true_idx)

        # Over the 200-step rolling window, coverage should track around 90% (80% - 95%)
        loc_cov = 1.0 - aci.localized_miscoverage()
        assert 0.78 <= loc_cov <= 0.98

    def test_aci_drift_adaptation(self):
        """When miscoverage spikes, qhat must rise (widening sets)."""
        aci = AdaptiveConformalInference(coverage=0.90, gamma=0.04)
        initial_qhat = aci.qhat
        probs = np.array([0.7, 0.1, 0.1, 0.1])

        # Force repeated misses (true index is 3, which is not in tight set)
        for _ in range(15):
            # Pass true index 3 with low prob
            res = aci.update(probs, true_index=3)

        assert aci.qhat > initial_qhat
        assert res["qhat"] > initial_qhat

        # Force repeated hits (true index is 0, highest prob)
        peak_qhat = aci.qhat
        for _ in range(30):
            res = aci.update(probs, true_index=0)

        assert aci.qhat < peak_qhat

    def test_multistream_aci_isolation(self):
        multi = MultiStreamACI(default_coverage=0.90, gamma=0.02)
        probs = np.array([0.95, 0.05])

        # Update stream A with hits (true_index=0 has 1 - 0.95 = 0.05 <= 0.90)
        for _ in range(10):
            multi.update("densenet121", probs, true_index=0)

        # Update stream B with misses (true_index=1 has 1 - 0.05 = 0.95 > 0.90)
        for _ in range(10):
            multi.update("brain_resunet", probs, true_index=1)

        assert multi.qhat("densenet121") < multi.qhat("brain_resunet")
