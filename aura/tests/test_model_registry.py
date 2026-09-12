"""Tests for Model Registry, Local Cohort Validation, and Monitoring & Alerting APIs."""
from __future__ import annotations

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from aura.gateway.storage import Store, Base
from aura.gateway.migrations import MigrationRunner
from aura.gateway.app import app, state as app_state


@pytest.fixture
def model_store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = Store(db_path=Path("model_memory.db"))
    store.engine = engine
    Base.metadata.create_all(engine)
    MigrationRunner(engine).apply_all()
    store._seed_defaults()
    yield store
    engine.dispose()


def test_seeded_models_in_registry(model_store):
    models = model_store.list_models()
    assert len(models) >= 3
    model_ids = [m["model_id"] for m in models]
    assert "vision-cxr-densenet121-production" in model_ids
    assert "fusion-poe-v1" in model_ids
    assert "brain-resunet-v1" in model_ids

    cxr = model_store.get_model("vision-cxr-densenet121-production")
    assert cxr is not None
    assert cxr["modality"] == "CXR"
    assert cxr["approval_status"] == "APPROVED"
    assert cxr["deployment_status"] == "ACTIVE"
    assert "auroc" in cxr["validation_metrics"]


def test_model_lifecycle_governance(model_store):
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = model_store

        # Login as AURA_ADMIN
        res_admin = client.post("/v1/auth/login", json={"username": "admin", "password": "AdminSecret123!"})
        tok = res_admin.json()["token"]
        headers = {"Authorization": f"Bearer {tok}"}

        # 1. Register candidate model
        new_model_payload = {
            "model_id": "pediatric-pneumonia-convnext-v2",
            "name": "Pediatric Pneumonia ConvNeXt",
            "version": "2.1.0",
            "modality": "CXR",
            "intended_use": "Pediatric inpatient triage and pneumonia probability ranking",
            "source": "Internal fine-tuned ConvNeXt",
            "training_data_provenance": "Guangzhou Women and Children's Medical Center cohort",
            "validation_dataset": "Guangzhou test split (n=624)",
            "validation_metrics": {"auroc": 0.884, "macro_f1": 0.742, "ece": 0.038},
            "checksum_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "approval_status": "PENDING_REVIEW",
            "deployment_status": "STAGING",
        }
        res_reg = client.post("/v1/models/registry", json=new_model_payload, headers=headers)
        assert res_reg.status_code == 201
        registered = res_reg.json()
        assert registered["model_id"] == "pediatric-pneumonia-convnext-v2"
        assert registered["approval_status"] == "PENDING_REVIEW"

        # 2. Duplicate registration rejected with 409 Conflict
        res_dup = client.post("/v1/models/registry", json=new_model_payload, headers=headers)
        assert res_dup.status_code == 409

        # 3. Update status: PENDING_REVIEW -> APPROVED, STAGING -> ACTIVE
        res_status = client.post(
            "/v1/models/registry/pediatric-pneumonia-convnext-v2/status",
            json={"approval_status": "APPROVED", "deployment_status": "ACTIVE"},
            headers=headers,
        )
        assert res_status.status_code == 200
        assert res_status.json()["approval_status"] == "APPROVED"
        assert res_status.json()["deployment_status"] == "ACTIVE"

        # 4. Retire model: ACTIVE -> RETIRED (sets retired_at)
        res_retire = client.post(
            "/v1/models/registry/pediatric-pneumonia-convnext-v2/status",
            json={"deployment_status": "RETIRED"},
            headers=headers,
        )
        assert res_retire.status_code == 200
        assert res_retire.json()["deployment_status"] == "RETIRED"
        assert res_retire.json()["retired_at"] is not None


def test_validation_run_workflow(model_store):
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = model_store

        # Login as clinician
        res_login = client.post("/v1/auth/login", json={"username": "clinician", "password": "Clinician123!"})
        tok = res_login.json()["token"]
        headers = {"Authorization": f"Bearer {tok}"}

        # 1. Queue local cohort validation run
        res_create = client.post(
            "/v1/validation-runs",
            json={
                "model_id": "vision-cxr-densenet121-production",
                "dataset_name": "memorial_hospital_q3_cohort",
                "config": {"sample_size": 200, "calibration_bins": 10},
            },
            headers=headers,
        )
        assert res_create.status_code == 201
        run_data = res_create.json()
        run_id = run_data["run_id"]
        assert run_data["status"] == "QUEUED"

        # 2. Execute validation run
        res_exec = client.post(
            f"/v1/validation-runs/{run_id}/execute",
            json={
                "sample_count": 200,
                "metrics": {
                    "auroc": 0.8351,
                    "ece": 0.0382,
                    "brier_score": 0.1189,
                    "macro_f1": 0.7104,
                    "conformal_coverage": 0.9085,
                    "samples_evaluated": 200,
                },
                "subgroup_analysis": {
                    "age": {"<65": {"auroc": 0.841}, ">=65": {"auroc": 0.829}},
                    "sex": {"M": {"auroc": 0.833}, "F": {"auroc": 0.837}},
                },
                "failure_analysis": {
                    "false_negatives": 5,
                    "false_positives": 8,
                    "abstention_count": 14,
                },
            },
            headers=headers,
        )
        assert res_exec.status_code == 200
        completed = res_exec.json()
        assert completed["status"] == "COMPLETED"
        assert completed["metrics"]["auroc"] == 0.8351
        assert completed["subgroup_analysis"]["age"]["<65"]["auroc"] == 0.841
        assert "Local Cohort Validation Report" in completed["report_markdown"]

        # 3. Retrieve validation run details
        res_get = client.get(f"/v1/validation-runs/{run_id}", headers=headers)
        assert res_get.status_code == 200
        assert res_get.json()["run_id"] == run_id

        # 4. List validation runs
        res_list = client.get("/v1/validation-runs", headers=headers)
        assert res_list.status_code == 200
        runs = res_list.json()
        assert len(runs) >= 1
        assert run_id in [r["run_id"] for r in runs]


def test_monitoring_and_alerts_api(model_store):
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = model_store

        # Login as clinician
        res_login = client.post("/v1/auth/login", json={"username": "clinician", "password": "Clinician123!"})
        tok = res_login.json()["token"]
        headers = {"Authorization": f"Bearer {tok}"}

        # 1. Log metrics
        res_m1 = client.post(
            "/v1/monitoring/metrics",
            json={"metric_name": "inference_latency_ms", "value": 42.5, "dimensions": {"modality": "CXR"}},
            headers=headers,
        )
        assert res_m1.status_code == 201

        res_m2 = client.post(
            "/v1/monitoring/metrics",
            json={"metric_name": "abstention_rate", "value": 0.08, "dimensions": {"window": "1h"}},
            headers=headers,
        )
        assert res_m2.status_code == 201

        # Query metrics
        res_get_m = client.get("/v1/monitoring/metrics?metric_name=inference_latency_ms", headers=headers)
        assert res_get_m.status_code == 200
        metrics = res_get_m.json()
        assert len(metrics) >= 1
        assert metrics[0]["metric_name"] == "inference_latency_ms"
        assert metrics[0]["value"] == 42.5

        # 2. Create alert
        res_alt = client.post(
            "/v1/monitoring/alerts",
            json={
                "category": "OOD_SPIKE",
                "severity": "CRITICAL",
                "message": "Spike in out-of-distribution CXRs detected in Emergency Department",
                "model_id": "vision-cxr-densenet121-production",
                "details": {"mahalanobis_distance_mean": 14.8, "threshold": 12.0},
            },
            headers=headers,
        )
        assert res_alt.status_code == 201
        alt_data = res_alt.json()
        alert_id = alt_data["alert_id"]
        assert alt_data["category"] == "OOD_SPIKE"
        assert alt_data["acknowledged"] is False

        # List unacknowledged alerts
        res_list_alts = client.get("/v1/monitoring/alerts?acknowledged=false", headers=headers)
        assert res_list_alts.status_code == 200
        assert alert_id in [a["alert_id"] for a in res_list_alts.json()]

        # Acknowledge alert
        res_ack = client.post(f"/v1/monitoring/alerts/{alert_id}/ack", headers=headers)
        assert res_ack.status_code == 200
        assert res_ack.json()["acknowledged"] is True
