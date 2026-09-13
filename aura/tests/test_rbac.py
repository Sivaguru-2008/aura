"""Tests for Role-Based Access Control (RBAC) across all 5 clinical and administrative roles.

Accounts are created per test with random passwords. Nothing here relies on seeded
accounts: a fresh database has none (see test_accounts_and_validation.py).
"""
from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from aura.gateway.app import app, state as app_state
from aura.gateway.security import ROLES
from aura.gateway.storage import Store


@pytest.fixture
def rbac(tmp_path):
    """A store with one account per role; yields (store, {role: (username, password)})."""
    store = Store(tmp_path / "rbac.db")
    creds: dict[str, tuple[str, str]] = {}
    for role in ROLES:
        username, password = f"user_{role.lower()}", secrets.token_urlsafe(16)
        store.create_user(username, password, role=role, org_id="default")
        creds[role] = (username, password)
    yield store, creds
    store.engine.dispose()


def _login(client, creds, role) -> str:
    username, password = creds[role]
    r = client.post("/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, f"login failed for {role}: {r.text}"
    return r.json()["token"]


def test_auth_login_logout_and_revocation(rbac):
    store, creds = rbac
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store

        res_bad = client.post("/v1/auth/login",
                              json={"username": creds["AURA_ADMIN"][0], "password": "WrongPassword!"})
        assert res_bad.status_code == 401

        username, password = creds["CLINICAL_USER"]
        res_login = client.post("/v1/auth/login", json={"username": username, "password": password})
        assert res_login.status_code == 200
        data = res_login.json()
        assert data["authenticated"] is True
        token = data["token"]
        assert token.startswith("aura_tok_")
        assert data["user"]["role"] == "CLINICAL_USER"

        res_me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_me.status_code == 200
        assert res_me.json()["user"]["username"] == username

        res_logout = client.post("/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res_logout.status_code == 200

        res_revoked = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_revoked.status_code == 401


def test_rbac_endpoint_permissions(rbac):
    store, creds = rbac
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        tokens = {role: _login(client, creds, role) for role in ROLES}

        def auth(role):
            return {"Authorization": f"Bearer {tokens[role]}"}

        # --- A. Organizations: only AURA_ADMIN may create orgs ---
        r = client.post("/v1/organizations", headers=auth("AURA_ADMIN"),
                        json={"org_id": "hospital-x", "name": "Hospital X", "slug": "hospital-x"})
        assert r.status_code == 201
        for role in ["HOSPITAL_ADMIN", "CLINICAL_USER", "AUDITOR", "READ_ONLY"]:
            r = client.post("/v1/organizations", headers=auth(role),
                            json={"org_id": f"org-{role.lower()}", "name": f"Org {role}",
                                  "slug": f"slug-{role.lower()}"})
            assert r.status_code == 403, f"Role {role} was not blocked from creating organization"

        # --- B. Users: AURA_ADMIN & HOSPITAL_ADMIN may create users ---
        r = client.post("/v1/users", headers=auth("HOSPITAL_ADMIN"),
                        json={"username": "dr_new", "password": "Password123!", "role": "CLINICAL_USER"})
        assert r.status_code == 201
        r = client.post("/v1/users", headers=auth("HOSPITAL_ADMIN"),
                        json={"username": "bad_admin", "password": "Password123!", "role": "AURA_ADMIN"})
        assert r.status_code == 403
        for role in ["CLINICAL_USER", "AUDITOR", "READ_ONLY"]:
            r = client.post("/v1/users", headers=auth(role),
                            json={"username": f"user_by_{role.lower()}", "password": "Password123!"})
            assert r.status_code == 403, f"Role {role} was not blocked from creating user"

        # --- C. Validation runs: AURA_ADMIN, HOSPITAL_ADMIN, CLINICAL_USER may queue ---
        r = client.post("/v1/validation-runs", headers=auth("CLINICAL_USER"),
                        json={"model_id": "vision-cxr-densenet121-production",
                              "dataset_name": "local_cxr_test"})
        assert r.status_code == 201
        for role in ["AUDITOR", "READ_ONLY"]:
            r = client.post("/v1/validation-runs", headers=auth(role),
                            json={"model_id": "vision-cxr-densenet121-production",
                                  "dataset_name": "local_cxr_test"})
            assert r.status_code == 403, f"Role {role} was not blocked from queueing validation"

        # --- D. Monitoring: read-only roles may read but not write ---
        for role in ["AUDITOR", "READ_ONLY"]:
            r = client.post("/v1/monitoring/alerts", headers=auth(role),
                            json={"category": "SYSTEM", "message": "should be refused"})
            assert r.status_code == 403, f"Role {role} was allowed to raise an alert"

        # --- E. Audit ledger: AUDITOR, HOSPITAL_ADMIN, AURA_ADMIN can view and verify ---
        assert client.get("/v1/audit/records", headers=auth("AUDITOR")).status_code == 200
        r = client.get("/v1/audit/verify", headers=auth("AUDITOR"))
        assert r.status_code == 200
        assert r.json()["valid"] is True
        assert client.get("/v1/audit/records", headers=auth("READ_ONLY")).status_code == 403


def test_user_disablement_revokes_active_sessions(rbac):
    store, creds = rbac
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store

        username, password = creds["CLINICAL_USER"]
        res_login = client.post("/v1/auth/login", json={"username": username, "password": password})
        tok = res_login.json()["token"]
        uid = res_login.json()["user"]["user_id"]
        assert client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"}).status_code == 200

        admin_tok = _login(client, creds, "AURA_ADMIN")
        res_disable = client.delete(f"/v1/users/{uid}", headers={"Authorization": f"Bearer {admin_tok}"})
        assert res_disable.status_code == 200
        assert res_disable.json()["disabled"] is True

        assert client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"}).status_code == 401
        res_relogin = client.post("/v1/auth/login", json={"username": username, "password": password})
        assert res_relogin.status_code == 401
