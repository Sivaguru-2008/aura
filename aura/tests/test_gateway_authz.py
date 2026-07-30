"""The gate must cover reads, not just mutations.

Authentication used to be applied only to POST/PUT/DELETE. That left every GET
open — including the two endpoints that exist specifically to emit patient data to
other systems (``/export/fhir``, ``/export/hl7``), the full case bundle, and
``/v1/admin/safety`` — with no token and no rate limit even when auth was fully
configured. Case ids were enumerable from the unauthenticated ``/v1/cases``.

These tests assert the property, not the implementation: *no route that can return
case data is reachable without a principal*. If someone adds an endpoint and forgets
to think about auth, ``test_no_case_route_is_public`` fails.
"""
from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from aura.common.config import get_settings
from aura.gateway import security
from aura.gateway.app import app

TOKEN = "test-token-do-not-ship"          # nosec B105 — fixture value, not a secret


@pytest.fixture
def secured(monkeypatch):
    """Turn auth on for the duration of a test.

    Settings is a frozen dataclass and get_settings() is cached, so build a replaced
    copy and patch the resolver that security.enforce actually calls. That keeps the
    real settings singleton untouched for every other test in the session.
    """
    secured_settings = dataclasses.replace(
        get_settings(), auth_token=TOKEN, auth_header="x-aura-token"
    )
    monkeypatch.setattr(security, "get_settings", lambda: secured_settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _auth() -> dict[str, str]:
    return {"x-aura-token": TOKEN, "x-aura-user": "dr.who"}


# --------------------------------------------------------------------------- #
# The regression that motivated this file
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [
    "/v1/cases",
    "/v1/cases/CASE-1",
    "/v1/cases/CASE-1/geometry",
    "/v1/cases/CASE-1/neuroview",
    "/v1/cases/CASE-1/chat",
    "/v1/cases/CASE-1/discussion",
    "/v1/cases/CASE-1/similar",
    "/v1/cases/CASE-1/export/fhir",
    "/v1/cases/CASE-1/export/hl7",
    "/v1/studies",
    "/v1/admin/safety",
    "/v1/models",
    "/v1/model-card",
])
def test_reads_require_a_token(secured, path):
    """Every GET that can surface case or policy state answers 401 without a token."""
    assert secured.get(path).status_code == 401


@pytest.mark.parametrize("path", ["/v1/cases/CASE-1/export/fhir",
                                  "/v1/cases/CASE-1/export/hl7"])
def test_phi_export_requires_a_token(secured, path):
    """Called out separately: these emit records designed for downstream ingestion."""
    assert secured.get(path).status_code == 401


def test_authenticated_read_is_not_401(secured):
    """A correct token gets past the gate (404 is fine — the case does not exist)."""
    assert secured.get("/v1/cases/CASE-1", headers=_auth()).status_code != 401


def test_token_without_principal_is_403(secured):
    """An authenticated call must still name its actor, so reads stay attributable."""
    r = secured.get("/v1/cases", headers={"x-aura-token": TOKEN})
    assert r.status_code == 403


def test_health_stays_public(secured):
    """Container/k8s probes run before any credential is injected."""
    assert secured.get("/v1/health").status_code == 200


def test_mutations_still_gated(secured):
    """The original behaviour must not regress while widening the gate."""
    r = secured.post("/v1/cases/CASE-1/feedback", json={"verdict": "accept"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# Structural guard — fails when a *new* sensitive route is added
# --------------------------------------------------------------------------- #
def test_no_case_route_is_public():
    """Nothing under /v1/cases, /v1/studies or /v1/admin may be on the allowlist.

    Cheap insurance: the allowlist is a frozenset someone could append to without
    thinking about what the path returns.
    """
    for path in security.PUBLIC_PATHS:
        assert not path.startswith(("/v1/cases", "/v1/studies", "/v1/admin")), path
    for prefix in security.PUBLIC_PREFIXES:
        assert not prefix.startswith("/v1"), prefix


def test_every_v1_route_is_gated_unless_explicitly_public():
    """Enumerate the live app and assert the gate covers all of /v1 except health.

    This is the test that catches the next endpoint someone adds. If a new /v1 route
    is genuinely meant to be public it has to be named in PUBLIC_PATHS, which is a
    deliberate act rather than an oversight.
    """
    ungated = [
        r.path for r in app.routes
        if getattr(r, "path", "").startswith("/v1") and security.is_public_path(r.path)
    ]
    assert ungated == ["/v1/health"], f"unexpectedly public /v1 routes: {ungated}"


def test_gate_is_inert_when_no_token_is_configured():
    """The offline demo must keep working untouched: no token configured, no 401.

    This is the property that lets the gate default to on without breaking the P0
    single-box demo the whole design is built around.
    """
    assert not get_settings().auth_token, "test env should ship without a token"
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/v1/cases").status_code == 200
