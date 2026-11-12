"""No default accounts, real password hashing, bounded open access, honest validation.

Builds before 2026-09-13 seeded five accounts whose passwords were written in
``storage.py`` (``admin`` / ``AdminSecret123!`` and four more), hashed passwords with a
single round of salted SHA-256, let any open-access caller claim ``AURA_ADMIN`` with a
header, and filled validation runs with fixed metrics when none were supplied. These
tests pin each fix.
"""
from __future__ import annotations

import dataclasses
import hashlib
import secrets

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from aura.common.config import get_settings
from aura.gateway import security
from aura.gateway.app import app, state as app_state
from aura.gateway.storage import Store, UserRow
from aura.services.assurance import local_validation


def _legacy_user(store: Store, username: str, password: str, role: str = "CLINICAL_USER") -> None:
    """Insert an account the way pre-2026-09-13 builds stored it (salted SHA-256)."""
    salt = secrets.token_hex(16)
    with Session(store.engine) as ses:
        ses.add(UserRow(user_id=f"usr-{username}", org_id="default", username=username,
                        password_hash=hashlib.sha256((password + salt).encode()).hexdigest(),
                        salt=salt, role=role, is_active=True))
        ses.commit()


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #
def test_a_fresh_database_has_no_user_accounts(tmp_path):
    assert Store(tmp_path / "a.db").list_users() == []


def test_passwords_are_stored_with_scrypt(tmp_path):
    store = Store(tmp_path / "a.db")
    store.create_user("dr.rao", "correct horse battery", role="CLINICAL_USER")
    row = store.get_user_by_username("dr.rao")
    assert row["password_hash"].startswith("scrypt$")
    assert store.verify_password("correct horse battery", row["salt"], row["password_hash"])
    assert not store.verify_password("wrong", row["salt"], row["password_hash"])


def test_a_legacy_hash_still_signs_in_and_is_upgraded(tmp_path):
    store = Store(tmp_path / "a.db")
    _legacy_user(store, "dr.old", "an-old-password")
    with TestClient(app, raise_server_exceptions=False) as client:
        app_state["store"] = store
        r = client.post("/v1/auth/login", json={"username": "dr.old", "password": "an-old-password"})
    assert r.status_code == 200
    assert store.get_user_by_username("dr.old")["password_hash"].startswith("scrypt$")


def test_accounts_still_using_a_published_default_password_are_disabled(tmp_path):
    db = tmp_path / "a.db"
    store = Store(db)
    _legacy_user(store, "admin", "AdminSecret123!", role="AURA_ADMIN")
    # The quarantined pilot script gave every account it created one published password.
    _legacy_user(store, "sysadmin_some_hospital", "HospitalPilotSecure2026!", role="AURA_ADMIN")
    _legacy_user(store, "clinician", "changed-after-install")
    reopened = Store(db)
    assert reopened.get_user_by_username("admin")["is_active"] is False
    assert reopened.get_user_by_username("sysadmin_some_hospital")["is_active"] is False
    assert reopened.get_user_by_username("clinician")["is_active"] is True
    actions = [r["action"] for r in reopened.recent_audit(20)]
    assert "security.default_account_disabled" in actions


def test_the_first_administrator_comes_from_the_bootstrap_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_BOOTSTRAP_ADMIN_PASSWORD", "a-long-bootstrap-password")
    db = tmp_path / "a.db"
    Store(db)
    Store(db)                                         # second start creates nothing more
    users = Store(db).list_users()
    assert [(u["username"], u["role"]) for u in users] == [("admin", "AURA_ADMIN")]


def test_a_short_bootstrap_password_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_BOOTSTRAP_ADMIN_PASSWORD", "short")
    with pytest.raises(RuntimeError, match="at least 12"):
        Store(tmp_path / "a.db")


# --------------------------------------------------------------------------- #
# Open access is bounded
# --------------------------------------------------------------------------- #
@pytest.fixture
def open_network(monkeypatch):
    s = dataclasses.replace(get_settings(), auth_token="", allow_open_access=True)
    monkeypatch.setattr(security, "get_settings", lambda: s)


@pytest.mark.parametrize("path, body", [
    ("/v1/organizations", {"org_id": "x", "name": "X"}),
    ("/v1/users", {"username": "evil", "password": "Password123!", "role": "AURA_ADMIN"}),
])
def test_an_open_access_network_caller_cannot_administer(open_network, path, body):
    with TestClient(app, client=("10.9.8.7", 50000), raise_server_exceptions=False) as client:
        r = client.post(path, json=body, headers={"x-aura-role": "AURA_ADMIN"})
    assert r.status_code == 403


def test_an_unknown_role_header_is_refused(open_network):
    with TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False) as client:
        r = client.get("/v1/cases", headers={"x-aura-role": "SUPERUSER"})
    assert r.status_code == 403


# --------------------------------------------------------------------------- #
# Validation metrics
# --------------------------------------------------------------------------- #
def _pair(top: str, truth: str, p: float = 0.8) -> dict:
    other = "normal" if top != "normal" else "pneumonia"
    return {"posterior": {top: p, other: 1 - p}, "truth": truth, "covered": True,
            "abstained": False}


def test_validation_refuses_below_the_minimum():
    with pytest.raises(local_validation.InsufficientOutcomes):
        local_validation.compute([_pair("pneumonia", "pneumonia")] * 5)


def test_validation_metrics_are_computed_not_assumed():
    pairs = ([_pair("pneumonia", "pneumonia")] * 15 + [_pair("normal", "normal")] * 10
             + [_pair("pneumonia", "normal")] * 5)
    out = local_validation.compute(pairs)
    m = out["metrics"]
    assert m["n"] == 30
    assert m["accuracy"] == pytest.approx(25 / 30, abs=1e-4)
    lo, hi = m["accuracy_ci95"]
    assert lo < m["accuracy"] < hi
    assert out["failure_analysis"]["incorrect_top1"] == 5
    assert 0.0 <= m["macro_auroc"] <= 1.0


def test_auroc_matches_the_definition():
    assert local_validation._auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert local_validation._auroc([0.1], [0.9]) == 0.0
    assert local_validation._auroc([0.5], [0.5]) == 0.5
