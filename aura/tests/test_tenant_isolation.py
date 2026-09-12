"""Tests for multi-tenant isolation across Storage repository and API Gateway."""
from __future__ import annotations

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from aura.gateway.storage import Store, CrossTenantAccessError
from aura.schemas.contracts import CaseBundle, CaseState
from aura.gateway.app import app, state as app_state


@pytest.fixture
def memory_store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = Store(db_path=Path("memory.db"))
    store.engine = engine
    # Re-run migrations and seeds on static in-memory engine
    from aura.gateway.storage import Base
    Base.metadata.create_all(engine)
    from aura.gateway.migrations import MigrationRunner
    MigrationRunner(engine).apply_all()
    store._seed_defaults()
    yield store
    engine.dispose()


def _dummy_bundle(case_id: str, study_id: str) -> CaseBundle:
    return CaseBundle(
        case_id=case_id,
        study_id=study_id,
        image=[128, 128],
        state=CaseState.READY,
        priority_score=0.85,
    )


def test_store_case_tenant_isolation(memory_store):
    b1 = _dummy_bundle("CASE-HOSP-A-001", "STU-A-001")
    b2 = _dummy_bundle("CASE-HOSP-B-001", "STU-B-001")

    memory_store.save_case(b1, tenant_id="hospital-a")
    memory_store.save_case(b2, tenant_id="hospital-b")

    # List cases per tenant
    cases_a = memory_store.list_cases(tenant_id="hospital-a")
    assert len(cases_a) == 1
    assert cases_a[0]["case_id"] == "CASE-HOSP-A-001"
    assert cases_a[0]["tenant_id"] == "hospital-a"

    cases_b = memory_store.list_cases(tenant_id="hospital-b")
    assert len(cases_b) == 1
    assert cases_b[0]["case_id"] == "CASE-HOSP-B-001"
    assert cases_b[0]["tenant_id"] == "hospital-b"

    # Cross-tenant get_case must raise CrossTenantAccessError
    with pytest.raises(CrossTenantAccessError):
        memory_store.get_case("CASE-HOSP-A-001", tenant_id="hospital-b")

    with pytest.raises(CrossTenantAccessError):
        memory_store.get_case("CASE-HOSP-B-001", tenant_id="hospital-a")

    # Correct tenant gets the case
    case_a = memory_store.get_case("CASE-HOSP-A-001", tenant_id="hospital-a")
    assert case_a is not None
    assert case_a.case_id == "CASE-HOSP-A-001"


def test_store_alert_and_validation_tenant_isolation(memory_store):
    memory_store.create_organization("hospital-a", "Hospital A", "hosp-a")
    memory_store.create_organization("hospital-b", "Hospital B", "hosp-b")

    # Alerts
    alt_a = memory_store.create_alert("alt-a-1", "OOD_SPIKE", "High OOD in ER", org_id="hospital-a")
    alt_b = memory_store.create_alert("alt-b-1", "DRIFT", "Prior drift in ICU", org_id="hospital-b")

    alerts_a = memory_store.list_alerts(org_id="hospital-a")
    assert len(alerts_a) == 1
    assert alerts_a[0]["alert_id"] == "alt-a-1"

    alerts_b = memory_store.list_alerts(org_id="hospital-b")
    assert len(alerts_b) == 1
    assert alerts_b[0]["alert_id"] == "alt-b-1"

    with pytest.raises(CrossTenantAccessError):
        memory_store.acknowledge_alert("alt-a-1", org_id="hospital-b")

    assert memory_store.acknowledge_alert("alt-a-1", org_id="hospital-a") is True

    # Validation runs
    run_a = memory_store.create_validation_run("val-a-1", org_id="hospital-a", model_id="vision-cxr-densenet121-production", dataset_name="hosp_a_cohort")
    run_b = memory_store.create_validation_run("val-b-1", org_id="hospital-b", model_id="vision-cxr-densenet121-production", dataset_name="hosp_b_cohort")

    with pytest.raises(CrossTenantAccessError):
        memory_store.get_validation_run("val-a-1", org_id="hospital-b")

    assert memory_store.get_validation_run("val-a-1", org_id="hospital-a")["run_id"] == "val-a-1"


def test_api_tenant_isolation(memory_store):
    # Set the app store to our test memory store
    app_state["store"] = memory_store

    # Create two hospital organizations and users
    memory_store.create_organization("hospital-a", "Hospital Alpha", "alpha")
    memory_store.create_organization("hospital-b", "Hospital Beta", "beta")

    u_a = memory_store.create_user("dr_alpha", "AlphaPass123!", role="CLINICAL_USER", org_id="hospital-a")
    u_b = memory_store.create_user("dr_beta", "BetaPass123!", role="CLINICAL_USER", org_id="hospital-b")

    # Create sessions
    tok_a, _ = memory_store.create_session(u_a["user_id"], "hospital-a", "CLINICAL_USER")
    tok_b, _ = memory_store.create_session(u_b["user_id"], "hospital-b", "CLINICAL_USER")

    # Seed cases for each hospital
    b_a = _dummy_bundle("CASE-ALPHA-100", "STU-A-100")
    b_b = _dummy_bundle("CASE-BETA-200", "STU-B-200")
    memory_store.save_case(b_a, tenant_id="hospital-a")
    memory_store.save_case(b_b, tenant_id="hospital-b")

    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = memory_store
        # 1. Hospital Alpha user lists cases: should see CASE-ALPHA-100 only
        res_a = client.get("/v1/cases", headers={"Authorization": f"Bearer {tok_a}"})
        assert res_a.status_code == 200
        case_ids_a = [c["case_id"] for c in res_a.json()["cases"]]
        assert "CASE-ALPHA-100" in case_ids_a
        assert "CASE-BETA-200" not in case_ids_a

        # 2. Hospital Beta user lists cases: should see CASE-BETA-200 only
        res_b = client.get("/v1/cases", headers={"Authorization": f"Bearer {tok_b}"})
        assert res_b.status_code == 200
        case_ids_b = [c["case_id"] for c in res_b.json()["cases"]]
        assert "CASE-BETA-200" in case_ids_b
        assert "CASE-ALPHA-100" not in case_ids_b

        # 3. Hospital Alpha user tries to access Hospital Beta case: MUST BE 403 FORBIDDEN
        res_cross = client.get("/v1/cases/CASE-BETA-200", headers={"Authorization": f"Bearer {tok_a}"})
        assert res_cross.status_code == 403
        detail = res_cross.json().get("detail", {})
        assert (detail.get("error") if isinstance(detail, dict) else detail) == "cross_tenant_access_denied"

        # 4. Hospital Alpha user accesses their own case: 200 OK
        res_own = client.get("/v1/cases/CASE-ALPHA-100", headers={"Authorization": f"Bearer {tok_a}"})
        assert res_own.status_code == 200
        assert res_own.json()["case_id"] == "CASE-ALPHA-100"
