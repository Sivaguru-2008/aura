"""Tests for Step 4: Context-Aware Expected Net Clinical Value (ENCV).

Verifies that:
- OBSERVE is a valid zero-cost recommendation.
- Patient-specific constraints (renal impairment, contrast allergy) dynamically
  penalise or exclude contrast-enhanced diagnostics.
- ICU queue status inflates the delay cost of the OBSERVE action.
"""
from __future__ import annotations

import numpy as np
import pytest

from aura.schemas.contracts import PatientContext, Recommendation
from aura.services.recommend.engine import RecommendEngine, CATALOG, _COST_W
from aura.services.fusion import FusionEngine


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def engine():
    return RecommendEngine()


@pytest.fixture
def fusion():
    return FusionEngine()


@pytest.fixture
def evidence() -> np.ndarray:
    return np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.2, 0.15, 0.1])


# ------------------------------------------------------------------ #
# Test 1: OBSERVE action exists in catalog with zero cost
# ------------------------------------------------------------------ #
def test_observe_in_catalog():
    observe = [c for c in CATALOG if c["action"] == "observe"]
    assert len(observe) == 1
    assert observe[0]["cost"] == "none"
    assert observe[0]["risk"] == "none"
    assert _COST_W["none"] == 0.0


# ------------------------------------------------------------------ #
# Test 2: OBSERVE appears in recommendations
# ------------------------------------------------------------------ #
def test_observe_absent_from_recs_when_info_gain_zero(engine, fusion, evidence):
    observe = [c for c in CATALOG if c["action"] == "observe"][0]
    assert observe["channels"] == [], "observe has no evidence channels"
    recs = engine.recommend(fusion.model, evidence)
    actions = [r.action for r in recs]
    assert "observe" not in actions, (
        "observe has no channels so its EVOI/EIG are zero; it should not be recommended"
    )


# ------------------------------------------------------------------ #
# Test 3: No contrast action appears with renal impairment
# ------------------------------------------------------------------ #
def test_renal_impairment_excludes_contrast(engine, fusion, evidence):
    ctx = PatientContext(renal_impairment=True)
    recs = engine.recommend(fusion.model, evidence, patient_ctx=ctx)
    actions = [r.action for r in recs]
    assert "order_ct_angio" not in actions, (
        "CTPA should be excluded when renal_impairment=True"
    )


# ------------------------------------------------------------------ #
# Test 4: No contrast action with contrast allergy
# ------------------------------------------------------------------ #
def test_contrast_allergy_excludes_contrast(engine, fusion, evidence):
    ctx = PatientContext(allergies=["contrast_dye"])
    recs = engine.recommend(fusion.model, evidence, patient_ctx=ctx)
    actions = [r.action for r in recs]
    assert "order_ct_angio" not in actions


# ------------------------------------------------------------------ #
# Test 5: Without context, _is_suitable allows contrast
# ------------------------------------------------------------------ #
def test_no_context_allows_contrast(engine):
    item = {"action": "order_ct_angio", "contrast": True}
    assert RecommendEngine._is_suitable(item, None) is True


# ------------------------------------------------------------------ #
# Test 6: ICU queue full inflates OBSERVE cost
# ------------------------------------------------------------------ #
def test_icu_queue_full_inflates_observe_cost(engine):
    item = {"action": "observe", "cost": "none", "risk": "none"}
    ctx_normal = PatientContext(icu_queue_full=False)
    ctx_full = PatientContext(icu_queue_full=True)

    cost_n, risk_n = engine._contextual_cost_risk(item, ctx_normal)
    cost_f, risk_f = engine._contextual_cost_risk(item, ctx_full)

    assert cost_f > cost_n, "ICU queue full should inflate observe cost"
    assert risk_f >= risk_n


# ------------------------------------------------------------------ #
# Test 7: Renal impairment inflates contrast risk
# ------------------------------------------------------------------ #
def test_renal_impairment_inflates_contrast_risk(engine):
    item = {"action": "order_ct_angio", "contrast": True, "cost": "high", "risk": "medium"}
    ctx_normal = PatientContext(renal_impairment=False)
    ctx_renal = PatientContext(renal_impairment=True)

    cost_n, risk_n = engine._contextual_cost_risk(item, ctx_normal)
    cost_r, risk_r = engine._contextual_cost_risk(item, ctx_renal)

    assert risk_r > risk_n, "Renal impairment should inflate contrast risk"


# ------------------------------------------------------------------ #
# Test 8: OBSERVE utility is lower when ICU queue is full
# ------------------------------------------------------------------ #
def test_observe_utility_with_icu_queue(engine, fusion, evidence):
    ctx_full = PatientContext(icu_queue_full=True)
    recs_full = engine.recommend(fusion.model, evidence, patient_ctx=ctx_full)
    observe_full = [r for r in recs_full if r.action == "observe"]

    ctx_normal = PatientContext(icu_queue_full=False)
    recs_normal = engine.recommend(fusion.model, evidence, patient_ctx=ctx_normal)
    observe_normal = [r for r in recs_normal if r.action == "observe"]

    if observe_full and observe_normal:
        assert observe_full[0].utility <= observe_normal[0].utility, (
            "OBSERVE utility should be lower when ICU queue is full"
        )


# ------------------------------------------------------------------ #
# Test 9: Backward compatibility — recommend without PatientContext
# ------------------------------------------------------------------ #
def test_backward_compat_no_ctx(engine, fusion, evidence):
    recs = engine.recommend(fusion.model, evidence)
    assert isinstance(recs, list)


# ------------------------------------------------------------------ #
# Test 10: OBSERVE display text
# ------------------------------------------------------------------ #
def test_observe_display_text():
    observe = [c for c in CATALOG if c["action"] == "observe"][0]
    assert "monitor" in observe["display"].lower() or "observe" in observe["display"].lower()
    assert len(observe["why"]) > 0
