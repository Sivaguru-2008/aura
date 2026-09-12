"""Tests for Role-Based Access Control (RBAC) across all 5 clinical and administrative roles."""
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
def rbac_store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = Store(db_path=Path("rbac_memory.db"))
    store.engine = engine
    Base.metadata.create_all(engine)
    MigrationRunner(engine).apply_all()
    store._seed_defaults()
    yield store
    engine.dispose()


def test_auth_login_logout_and_revocation(rbac_store):
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = rbac_store

        # 1. Invalid login
        res_bad = client.post("/v1/auth/login", json={"username": "admin", "password": "WrongPassword!"})
        assert res_bad.status_code == 401

        # 2. Valid login for clinician
        res_login = client.post("/v1/auth/login", json={"username": "clinician", "password": "Clinician123!"})
        assert res_login.status_code == 200
        data = res_login.json()
        assert data["authenticated"] is True
        token = data["token"]
        assert token.startswith("aura_tok_")
        assert data["user"]["role"] == "CLINICAL_USER"

        # 3. GET /v1/auth/me
        res_me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_me.status_code == 200
        assert res_me.json()["user"]["username"] == "clinician"

        # 4. Logout / revocation
        res_logout = client.post("/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res_logout.status_code == 200

        # 5. Subsequent request with revoked token must fail with 401
        res_revoked = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_revoked.status_code == 401


def test_rbac_endpoint_permissions(rbac_store):
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = rbac_store

        # Log in users for all 5 roles
        tokens = {}
        for role, user, pwd in [
            ("AURA_ADMIN", "admin", "AdminSecret123!"),
            ("HOSPITAL_ADMIN", "hospital_admin", "HospitalAdmin123!"),
            ("CLINICAL_USER", "clinician", "Clinician123!"),
            ("AUDITOR", "auditor", "Auditor123!"),
            ("READ_ONLY", "observer", "Observer123!"),
        ]:
            r = client.post("/v1/auth/login", json={"username": user, "password": pwd})
            assert r.status_code == 200, f"Failed login for {user}: {r.text}"
            tokens[role] = r.json()["token"]

        # --- A. Organizations: Only AURA_ADMIN may create orgs ---
        # AURA_ADMIN: 201 Created
        res_org_admin = client.post(
            "/v1/organizations",
            headers={"Authorization": f"Bearer {tokens['AURA_ADMIN']}"},
            json={"org_id": "st-jude", "name": "St. Jude Children's", "slug": "st-jude"}
        )
        assert res_org_admin.status_code == 201

        # Other roles: 403 Forbidden
        for role in ["HOSPITAL_ADMIN", "CLINICAL_USER", "AUDITOR", "READ_ONLY"]:
            res_org_blocked = client.post(
                "/v1/organizations",
                headers={"Authorization": f"Bearer {tokens[role]}"},
                json={"org_id": f"org-{role.lower()}", "name": f"Org {role}", "slug": f"slug-{role.lower()}"}
            )
            assert res_org_blocked.status_code == 403, f"Role {role} was not blocked from creating organization"

        # --- B. Users: AURA_ADMIN & HOSPITAL_ADMIN may create users ---
        # HOSPITAL_ADMIN creates clinical user: 201 Created
        res_u_hosp = client.post(
            "/v1/users",
            headers={"Authorization": f"Bearer {tokens['HOSPITAL_ADMIN']}"},
            json={"username": "dr_new", "password": "Password123!", "role": "CLINICAL_USER"}
        )
        assert res_u_hosp.status_code == 201

        # HOSPITAL_ADMIN tries to escalate privilege to AURA_ADMIN: 403 Forbidden
        res_escalate = client.post(
            "/v1/users",
            headers={"Authorization": f"Bearer {tokens['HOSPITAL_ADMIN']}"},
            json={"username": "bad_admin", "password": "Password123!", "role": "AURA_ADMIN"}
        )
        assert res_escalate.status_code == 403

        # CLINICAL_USER, AUDITOR, READ_ONLY cannot create users: 403 Forbidden
        for role in ["CLINICAL_USER", "AUDITOR", "READ_ONLY"]:
            res_blocked = client.post(
                "/v1/users",
                headers={"Authorization": f"Bearer {tokens[role]}"},
                json={"username": f"user_by_{role.lower()}", "password": "Password123!"}
            )
            assert res_blocked.status_code == 403, f"Role {role} was not blocked from creating user"

        # --- C. Validation Runs: AURA_ADMIN, HOSPITAL_ADMIN, CLINICAL_USER may queue runs ---
        res_val_clinician = client.post(
            "/v1/validation-runs",
            headers={"Authorization": f"Bearer {tokens['CLINICAL_USER']}"},
            json={"model_id": "vision-cxr-densenet121-production", "dataset_name": "local_cxr_test"}
        )
        assert res_val_clinician.status_code == 201

        # AUDITOR and READ_ONLY cannot queue validation runs
        for role in ["AUDITOR", "READ_ONLY"]:
            res_val_blocked = client.post(
                "/v1/validation-runs",
                headers={"Authorization": f"Bearer {tokens[role]}"},
                json={"model_id": "vision-cxr-densenet121-production", "dataset_name": "local_cxr_test"}
            )
            assert res_val_blocked.status_code == 403, f"Role {role} was not blocked from queueing validation"

        # --- D. Audit Ledger: AUDITOR, HOSPITAL_ADMIN, AURA_ADMIN can view and verify ---
        res_audit_auditor = client.get(
            "/v1/audit/records",
            headers={"Authorization": f"Bearer {tokens['AUDITOR']}"}
        )
        assert res_audit_auditor.status_code == 200

        res_audit_verify = client.get(
            "/v1/audit/verify",
            headers={"Authorization": f"Bearer {tokens['AUDITOR']}"}
        )
        assert res_audit_verify.status_code == 200
        assert res_audit_verify.json()["valid"] is True

        # READ_ONLY cannot access audit records
        res_audit_ro = client.get(
            "/v1/audit/records",
            headers={"Authorization": f"Bearer {tokens['READ_ONLY']}"}
        )
        assert res_audit_ro.status_code == 403


def test_user_disablement_revokes_active_sessions(rbac_store):
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = rbac_store

        # 1. Login user
        res_login = client.post("/v1/auth/login", json={"username": "clinician", "password": "Clinician123!"})
        tok = res_login.json()["token"]
        uid = res_login.json()["user"]["user_id"]

        # 2. Confirm token works
        res_ok = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
        assert res_ok.status_code == 200

        # 3. Admin disables user
        res_admin = client.post("/v1/auth/login", json={"username": "admin", "password": "AdminSecret123!"})
        admin_tok = res_admin.json()["token"]

        res_disable = client.delete(f"/v1/users/{uid}", headers={"Authorization": f"Bearer {admin_tok}"})
        assert res_disable.status_code == 200
        assert res_disable.json()["disabled"] is True

        # 4. User's active session is now rejected (401)
        res_disabled_call = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
        assert res_disabled_call.status_code == 401

        # 5. User cannot log in again
        res_relogin = client.post("/v1/auth/login", json={"username": "clinician", "password": "Clinician123!"})
        assert res_relogin.status_code == 401
