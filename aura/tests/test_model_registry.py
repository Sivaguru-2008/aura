"""Tests for Model Registry, Local Cohort Validation, and Monitoring & Alerting APIs."""
from __future__ import annotations

import secrets

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aura.common.config import ARTIFACTS
from aura.gateway.app import app, state as app_state
from aura.gateway.storage import Store
from aura.schemas.clinical import DIAGNOSES, Diagnosis
from aura.schemas.contracts import (
    CaseBundle, FusionResult, Prediction, SafetyAssessment, StructuredPriors, VisionResult,
)

CXR_MODEL = "vision-cxr-densenet121-production"


@pytest.fixture
def model_store(tmp_path):
    """A store with an admin and a clinician; yields (store, {role: (user, pw)})."""
    store = Store(tmp_path / "registry.db")
    creds = {}
    for role, username in (("AURA_ADMIN", "reg_admin"), ("CLINICAL_USER", "reg_clinician")):
        password = secrets.token_urlsafe(16)
        store.create_user(username, password, role=role, org_id="default")
        creds[role] = (username, password)
    yield store, creds
    store.engine.dispose()


def _headers(client, creds, role):
    username, password = creds[role]
    r = client.post("/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_seeded_models_are_pending_review_with_traceable_provenance(model_store):
    store, _ = model_store
    ids = {m["model_id"] for m in store.list_models()}
    assert {CXR_MODEL, "fusion-poe-v1", "brain-resunet-v1"} <= ids

    cxr = store.get_model(CXR_MODEL)
    # Deployed is a fact; approved would be a claim nobody has made.
    assert cxr["approval_status"] == "PENDING_REVIEW"
    assert cxr["deployment_status"] == "ACTIVE"
    assert "MIMIC-CXR" in cxr["training_data_provenance"]
    assert "radiologist" not in cxr["training_data_provenance"].lower()
    if (ARTIFACTS / "evaluation_retrain_v2" / "metrics.json").exists():
        assert cxr["validation_metrics"]["macro_auroc"] == pytest.approx(0.821, abs=5e-4)
    if (ARTIFACTS / "best_model.pt").exists():
        assert len(cxr["checksum_sha256"]) == 64


def test_model_lifecycle_governance(model_store):
    store, creds = model_store
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        headers = _headers(client, creds, "AURA_ADMIN")

        payload = {
            "model_id": "pediatric-pneumonia-convnext-v2",
            "name": "Pediatric Pneumonia ConvNeXt",
            "version": "2.1.0",
            "modality": "CXR",
            "intended_use": "Pediatric inpatient triage",
            "source": "Internal fine-tuned ConvNeXt",
        }
        res_reg = client.post("/v1/models/registry", json=payload, headers=headers)
        assert res_reg.status_code == 201
        # Registration is not approval.
        assert res_reg.json()["approval_status"] == "PENDING_REVIEW"
        assert res_reg.json()["deployment_status"] == "STAGING"

        assert client.post("/v1/models/registry", json=payload, headers=headers).status_code == 409

        res_status = client.post(
            "/v1/models/registry/pediatric-pneumonia-convnext-v2/status",
            json={"approval_status": "APPROVED", "deployment_status": "ACTIVE"},
            headers=headers,
        )
        assert res_status.status_code == 200
        assert res_status.json()["approval_status"] == "APPROVED"

        res_retire = client.post(
            "/v1/models/registry/pediatric-pneumonia-convnext-v2/status",
            json={"deployment_status": "RETIRED"},
            headers=headers,
        )
        assert res_retire.status_code == 200
        assert res_retire.json()["retired_at"] is not None


def _queue(client, headers) -> str:
    r = client.post("/v1/validation-runs", headers=headers,
                    json={"model_id": CXR_MODEL, "dataset_name": "site_q3_cohort"})
    assert r.status_code == 201
    assert r.json()["status"] == "QUEUED"
    return r.json()["run_id"]


def test_client_supplied_metrics_are_stored_and_labelled_as_such(model_store):
    store, creds = model_store
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        headers = _headers(client, creds, "CLINICAL_USER")
        run_id = _queue(client, headers)

        res = client.post(f"/v1/validation-runs/{run_id}/execute", headers=headers,
                          json={"metrics": {"auroc": 0.8351, "samples_evaluated": 200}})
        assert res.status_code == 200
        done = res.json()
        assert done["status"] == "COMPLETED"
        assert done["metrics"]["auroc"] == 0.8351
        assert done["metrics"]["provenance"] == "client_supplied"
        assert "did not compute" in done["report_markdown"]
        assert "Local Cohort Validation Report" in done["report_markdown"]

        assert client.get(f"/v1/validation-runs/{run_id}", headers=headers).status_code == 200
        assert run_id in [r["run_id"] for r in client.get("/v1/validation-runs", headers=headers).json()]


def test_execute_without_outcomes_refuses_instead_of_inventing_metrics(model_store):
    """This endpoint used to return AUROC 0.8124 and an age/sex table for any empty call."""
    store, creds = model_store
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        headers = _headers(client, creds, "CLINICAL_USER")
        run_id = _queue(client, headers)

        res = client.post(f"/v1/validation-runs/{run_id}/execute", headers=headers, json={})
        assert res.status_code == 409
        assert res.json()["detail"]["error"] == "insufficient_verified_outcomes"
        run = client.get(f"/v1/validation-runs/{run_id}", headers=headers).json()
        assert run["status"] == "FAILED"
        assert run["metrics"] in ({}, None)


def _case_with_outcome(store, i: int, top: Diagnosis, truth: Diagnosis) -> None:
    probs = np.full(len(DIAGNOSES), 0.04)
    probs[DIAGNOSES.index(top)] = 0.80
    case_id, study = f"CASE-VAL-{i}", f"STU-VAL-{i}"
    store.save_case(CaseBundle(
        case_id=case_id, study_id=study, state="ready", priors=StructuredPriors(),
        vision=VisionResult(study_id=study, findings=[], embedding=[], model_version=""),
        fusion=FusionResult(study_id=study, backend="classical", posterior={}),
        safety=SafetyAssessment(
            study_id=study, top=top, top_probability=0.80, conformal_set=[top],
            predictions=[Prediction(diagnosis=d, probability=float(p))
                         for d, p in zip(DIAGNOSES, probs)]),
    ), tenant_id="default", model_version=CXR_MODEL)
    store.record_outcome(case_id, probs, DIAGNOSES.index(truth), true_diagnosis=truth.value)


def test_execute_computes_metrics_from_verified_outcomes(model_store):
    store, creds = model_store
    for i in range(20):
        _case_with_outcome(store, i, Diagnosis.PNEUMONIA, Diagnosis.PNEUMONIA)
    for i in range(20, 25):
        _case_with_outcome(store, i, Diagnosis.PNEUMONIA, Diagnosis.NORMAL)

    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        headers = _headers(client, creds, "CLINICAL_USER")
        run_id = _queue(client, headers)
        res = client.post(f"/v1/validation-runs/{run_id}/execute", headers=headers, json={})
        assert res.status_code == 200, res.text
        m = res.json()["metrics"]
        assert m["provenance"] == "computed_by_aura"
        assert m["n"] == 25
        assert m["accuracy"] == pytest.approx(0.8)
        assert res.json()["failure_analysis"]["incorrect_top1"] == 5
        assert res.json()["subgroup_analysis"] == {}


def test_monitoring_and_alerts_api(model_store):
    store, creds = model_store
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        headers = _headers(client, creds, "CLINICAL_USER")

        assert client.post("/v1/monitoring/metrics", headers=headers,
                           json={"metric_name": "inference_latency_ms", "value": 42.5,
                                 "dimensions": {"modality": "CXR"}}).status_code == 201
        res_get = client.get("/v1/monitoring/metrics?metric_name=inference_latency_ms", headers=headers)
        assert res_get.status_code == 200
        assert res_get.json()[0]["value"] == 42.5

        res_alt = client.post("/v1/monitoring/alerts", headers=headers, json={
            "category": "OOD_SPIKE", "severity": "CRITICAL",
            "message": "Spike in out-of-distribution CXRs", "model_id": CXR_MODEL})
        assert res_alt.status_code == 201
        alert_id = res_alt.json()["alert_id"]
        assert alert_id in [a["alert_id"] for a in
                            client.get("/v1/monitoring/alerts?acknowledged=false", headers=headers).json()]
        res_ack = client.post(f"/v1/monitoring/alerts/{alert_id}/ack", headers=headers)
        assert res_ack.status_code == 200
        assert res_ack.json()["acknowledged"] is True
