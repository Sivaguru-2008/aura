"""Tests for Step 1: Decouple conformal calibration from clinician feedback.

Clinician feedback (accept / edit / reject) must NOT update the ACI threshold.
Only verified patient outcomes submitted via ``/v1/cases/{case_id}/outcome`` may
trigger the Adaptive Conformal Inference update.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from fastapi.testclient import TestClient

from aura.gateway.app import app
from aura.gateway.storage import OutcomeRow, AuditRow
from aura.schemas.clinical import Diagnosis
from aura.schemas.contracts import (
    CaseBundle,
    CaseState,
    SafetyAssessment,
    Prediction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_case(store, case_id: str = "CASE-TEST-1") -> None:
    """Insert a minimal CaseBundle with a valid SafetyAssessment into the DB."""
    preds = [
        Prediction(diagnosis=Diagnosis.NORMAL, probability=0.40),
        Prediction(diagnosis=Diagnosis.PNEUMONIA, probability=0.35),
        Prediction(diagnosis=Diagnosis.HEART_FAILURE, probability=0.10),
        Prediction(diagnosis=Diagnosis.COPD, probability=0.05),
        Prediction(diagnosis=Diagnosis.MALIGNANCY, probability=0.05),
        Prediction(diagnosis=Diagnosis.PNEUMOTHORAX, probability=0.05),
    ]
    safety = SafetyAssessment(
        study_id=f"STU-{case_id}",
        predictions=preds,
        top=Diagnosis.NORMAL,
        top_probability=0.40,
        conformal_set=[Diagnosis.NORMAL, Diagnosis.PNEUMONIA],
        conformal_coverage=0.90,
        abstained=False,
    )
    bundle = CaseBundle(
        case_id=case_id,
        study_id=f"STU-{case_id}",
        state=CaseState.READY,
        priority_score=0.5,
        safety=safety,
    )
    store.save_case(bundle)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_and_seed(client):
    """Wipe outcomes & audit, then seed a fresh case before every test."""
    from aura.gateway.app import state as _state
    store = _state["store"]
    with Session(store.engine) as ses:
        ses.execute(text("DELETE FROM outcomes"))
        ses.execute(text("DELETE FROM audit_log"))
        ses.execute(text("DELETE FROM conformal_state"))
        ses.commit()
    _seed_case(store)


# ---------------------------------------------------------------------------
# Test 1: Feedback does NOT update conformal threshold
# ---------------------------------------------------------------------------
def test_feedback_does_not_trigger_conformal_update(client):
    """Clinician feedback (accept / edit / reject) must not touch ACI state."""
    from aura.gateway.app import state as _state
    store = _state["store"]

    resp = client.post(
        "/v1/cases/CASE-TEST-1/feedback",
        json={"verdict": "accept", "diagnosis": "normal", "correction": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # The old "conformal" key must be gone from the feedback response.
    assert "conformal" not in body

    # Verify the audit log has feedback.recorded but NOT conformal.updated.
    audit = store.recent_audit(10)
    actions = [a["action"] for a in audit]
    assert "feedback.recorded" in actions
    assert "conformal.updated" not in actions

    # Verify no ACI state was written.
    assert store.load_aci_state() is None


# ---------------------------------------------------------------------------
# Test 2: Verified outcome path updates ACI
# ---------------------------------------------------------------------------
def test_verified_outcome_updates_conformal(client):
    """A valid outcome submission must run the ACI update and write an audit row."""
    from aura.gateway.app import state as _state
    store = _state["store"]

    resp = client.post(
        "/v1/cases/CASE-TEST-1/outcome",
        json={"true_diagnosis": "normal", "source": "pcr"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["conformal"] is not None
    assert "qhat" in body["conformal"]

    # Audit must contain a conformal.updated entry.
    audit = store.recent_audit(20)
    actions = [a["action"] for a in audit]
    assert "conformal.updated" in actions

    # Outcome row must now exist.
    assert store.has_outcome("CASE-TEST-1") is True


# ---------------------------------------------------------------------------
# Test 3: Invalid source → 422
# ---------------------------------------------------------------------------
def test_invalid_source_returns_422(client):
    resp = client.post(
        "/v1/cases/CASE-TEST-1/outcome",
        json={"true_diagnosis": "normal", "source": "arbitrary_garbage"},
    )
    assert resp.status_code == 422
    body = resp.json()["detail"]
    assert body["error"] == "invalid_source"


# ---------------------------------------------------------------------------
# Test 4: Invalid diagnosis → 422
# ---------------------------------------------------------------------------
def test_invalid_diagnosis_returns_422(client):
    resp = client.post(
        "/v1/cases/CASE-TEST-1/outcome",
        json={"true_diagnosis": "made_up_condition", "source": "pcr"},
    )
    assert resp.status_code == 422
    body = resp.json()["detail"]
    assert body["error"] == "invalid_diagnosis"


# ---------------------------------------------------------------------------
# Test 5: Unknown case_id → 404
# ---------------------------------------------------------------------------
def test_unknown_case_returns_404(client):
    resp = client.post(
        "/v1/cases/CASE-NONEXISTENT/outcome",
        json={"true_diagnosis": "normal", "source": "pcr"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 6: Duplicate outcome → 409 (prevents double-calibration)
# ---------------------------------------------------------------------------
def test_duplicate_outcome_returns_409(client):
    """The second submission for the same case must be rejected."""
    from aura.gateway.app import state as _state
    store = _state["store"]

    payload = {"true_diagnosis": "normal", "source": "expert_consensus"}

    resp1 = client.post("/v1/cases/CASE-TEST-1/outcome", json=payload)
    assert resp1.status_code == 200

    resp2 = client.post("/v1/cases/CASE-TEST-1/outcome", json=payload)
    assert resp2.status_code == 409
    body = resp2.json()["detail"]
    assert body["error"] == "outcome_already_recorded"

    # The ACI must only have been updated once (by the first request).
    audit = store.recent_audit(20)
    conformal_updates = [a for a in audit if a["action"] == "conformal.updated"]
    assert len(conformal_updates) == 1
