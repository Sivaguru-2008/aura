"""Tests for Step 3: Uncertainty-to-mitigation mapping.

Verifies that each uncertainty dimension triggers the correct MitigationAction
in the SafetyAssessment, replacing the binary abstain flag with workflow-specific
actions.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from aura.schemas.clinical import DIAGNOSES, Diagnosis
from aura.schemas.contracts import (
    MitigationAction,
    Prediction,
    SafetyAssessment,
)
from aura.services.safety.engine import SafetyEngine


# ------------------------------------------------------------------ #
# Helper: build a SafetyAssessment with controlled uncertainty values
# ------------------------------------------------------------------ #
def _mk_safety(
    ood_energy: float = 0.0,
    is_ood: bool = False,
    predictive_entropy: float = 0.5,
    epistemic: float = 0.05,
    epistemic_mi: float = 0.0,
    conformal_set_size: int = 1,
    n_ensemble: int = 0,
) -> SafetyAssessment:
    preds = [Prediction(diagnosis=d, probability=1.0 / 6) for d in DIAGNOSES]
    return SafetyAssessment(
        study_id="STU-TEST",
        predictions=preds,
        top=Diagnosis.NORMAL,
        top_probability=0.25,
        conformal_set=[Diagnosis.NORMAL] * conformal_set_size,
        conformal_coverage=0.90,
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=0.3,
        epistemic_mi=epistemic_mi,
        predictive_entropy=predictive_entropy,
        uncertainty_method="deep_ensemble" if n_ensemble > 0 else "input_perturbation",
        n_ensemble=n_ensemble,
        ood_energy=ood_energy,
        is_ood=is_ood,
        abstained=False,
    )


# ------------------------------------------------------------------ #
# Test 1: OOD triggers ESC_HUMAN_EXPERT_REVIEW
# ------------------------------------------------------------------ #
def test_ood_triggers_esc_expert_review():
    safety = _mk_safety(is_ood=True, ood_energy=2.5)
    mitigations = _mitigations_for(safety)
    assert MitigationAction.ESC_HUMAN_EXPERT_REVIEW in mitigations


# ------------------------------------------------------------------ #
# Test 2: High entropy triggers SHOW_COMPETING_HYPOTHESES
# ------------------------------------------------------------------ #
def test_high_entropy_triggers_show_competing():
    safety = _mk_safety(predictive_entropy=1.5, conformal_set_size=4)
    mitigations = _mitigations_for(safety)
    assert MitigationAction.SHOW_COMPETING_HYPOTHESES in mitigations


# ------------------------------------------------------------------ #
# Test 3: High epistemic triggers ORDER_CONFIRMATORY_EVIDENCE
# ------------------------------------------------------------------ #
def test_high_epistemic_triggers_confirmatory():
    safety = _mk_safety(epistemic=0.25, epistemic_mi=0.15)
    mitigations = _mitigations_for(safety)
    assert MitigationAction.ORDER_CONFIRMATORY_EVIDENCE in mitigations


# ------------------------------------------------------------------ #
# Test 4: Quantum shot noise triggers RE_ACQUIRE_SHOTS
# ------------------------------------------------------------------ #
def test_shot_noise_triggers_reacquire():
    safety = _mk_safety(n_ensemble=5, epistemic=0.2)
    mitigations = _mitigations_for(safety)
    assert MitigationAction.RE_ACQUIRE_SHOTS in mitigations


# ------------------------------------------------------------------ #
# Test 5: Clean study has no mitigations
# ------------------------------------------------------------------ #
def test_clean_study_no_mitigations():
    safety = _mk_safety(
        predictive_entropy=0.3,
        epistemic=0.05,
        epistemic_mi=0.01,
        conformal_set_size=1,
    )
    mitigations = _mitigations_for(safety)
    assert len(mitigations) == 0


# ------------------------------------------------------------------ #
# Test 6: Multiple mitigations can co-occur
# ------------------------------------------------------------------ #
def test_multiple_mitigations():
    safety = _mk_safety(
        is_ood=True, ood_energy=3.0,
        predictive_entropy=1.5,
        conformal_set_size=4,
        epistemic=0.25,
    )
    mitigations = _mitigations_for(safety)
    assert MitigationAction.ESC_HUMAN_EXPERT_REVIEW in mitigations
    assert MitigationAction.SHOW_COMPETING_HYPOTHESES in mitigations
    assert MitigationAction.ORDER_CONFIRMATORY_EVIDENCE in mitigations


# ------------------------------------------------------------------ #
# Test 7: Backward compatibility — recommended_mitigations defaults to []
# ------------------------------------------------------------------ #
def test_backward_compat_default_empty():
    safety = SafetyAssessment(
        study_id="STU-BC",
        predictions=[],
        top=Diagnosis.NORMAL,
        top_probability=0.95,
        conformal_set=[Diagnosis.NORMAL],
    )
    assert safety.recommended_mitigations == []


# ------------------------------------------------------------------ #
# Test 8: SafetyEngine.assess populates mitigations for OOD input
# ------------------------------------------------------------------ #
def test_safety_engine_populates_mitigations():
    """Integration test: SafetyEngine.assess returns recommended_mitigations."""
    se = SafetyEngine()
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=8)

    fake_logits = rng.uniform(-1.0, 1.0, size=len(DIAGNOSES))
    fusion_model = MagicMock()
    fusion_model.logits.return_value = fake_logits

    with patch("aura.services.safety.engine.energy_score", return_value=10.0):
        with patch.object(se.cal, "ood_mean", 0.0):
            with patch.object(se.cal, "ood_std", 1.0):
                safety = se.assess("STU-OOD", x, fusion_model)

    assert safety.is_ood is True
    assert MitigationAction.ESC_HUMAN_EXPERT_REVIEW in safety.recommended_mitigations


# ------------------------------------------------------------------ #
# Internal: replicate the mitigation logic from SafetyEngine for
# unit-testing the threshold mapping in isolation.
# ------------------------------------------------------------------ #
def _mitigations_for(safety: SafetyAssessment) -> list[MitigationAction]:
    """Evaluate the mitigation thresholds against a pre-built SafetyAssessment.

    This mirrors the logic in ``SafetyEngine.assess`` so we can unit-test
    the threshold mapping without needing a trained fusion model.
    """
    mitigations: list[MitigationAction] = []
    z = safety.ood_energy or 0.0
    if safety.is_ood or z > 1.5:
        mitigations.append(MitigationAction.ESC_HUMAN_EXPERT_REVIEW)
    pe = safety.predictive_entropy or 0.0
    cs = len(safety.conformal_set)
    if pe > 1.2 or cs >= 3:
        mitigations.append(MitigationAction.SHOW_COMPETING_HYPOTHESES)
    ep = safety.epistemic_uncertainty or 0.0
    mi = safety.epistemic_mi or 0.0
    if ep > 0.15 or mi > 0.1:
        mitigations.append(MitigationAction.ORDER_CONFIRMATORY_EVIDENCE)
    if safety.n_ensemble > 0 and ep > 0.15:
        mitigations.append(MitigationAction.RE_ACQUIRE_SHOTS)
    return mitigations
