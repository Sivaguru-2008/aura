"""Secure by default: no token means *this machine only*, and browsers get sessions.

Until 2026-09-12 an AURA with no ``AURA_AUTH_TOKEN`` answered anyone who could reach
its port, including ``GET /v1/cases`` and the FHIR/HL7 exporters. The preflight warned
about it, but a warning is not a control. Now a gateway with no token serves only
direct, un-proxied connections from the loopback interface. The offline single-box
demo keeps working, and nothing on a network can read a case until someone sets a
token. ``AURA_ALLOW_OPEN_ACCESS=1`` is the explicit, logged opt-out.

Browsers exchange the token for a signed, HttpOnly, SameSite=Strict cookie that
carries the signed-in name, so every browser action lands in the audit log under a
person rather than under ``"clinician"``.

The suite as a whole runs with open access on (see conftest.py); every test here
patches the settings it needs.
"""
from __future__ import annotations

import dataclasses
import time

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from aura.common.config import get_settings
from aura.gateway import security
from aura.gateway.app import app

TOKEN = "test-token-do-not-ship"          # nosec B105 — fixture value, not a secret
REMOTE = ("10.1.2.3", 50000)
LOCAL = ("127.0.0.1", 50000)


def _settings(monkeypatch, **overrides):
    s = dataclasses.replace(get_settings(), **overrides)
    monkeypatch.setattr(security, "get_settings", lambda: s)
    return s


# --------------------------------------------------------------------------- #
# No token configured
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_token(monkeypatch):
    _settings(monkeypatch, auth_token="", allow_open_access=False)


def test_no_token_refuses_network_clients(no_token):
    with TestClient(app, client=REMOTE, raise_server_exceptions=False) as c:
        r = c.get("/v1/cases")
    assert r.status_code == 403
    assert r.json()["error"] == "auth_not_configured"


@pytest.mark.parametrize("path", ["/v1/cases/CASE-1/export/fhir",
                                  "/v1/cases/CASE-1/export/hl7",
                                  "/v1/admin/safety"])
def test_no_token_refuses_phi_exports_to_the_network(no_token, path):
    with TestClient(app, client=REMOTE, raise_server_exceptions=False) as c:
        assert c.get(path).status_code == 403


def test_no_token_still_serves_this_machine(no_token):
    """The offline demo: browser and server on one box, no configuration."""
    with TestClient(app, client=LOCAL) as c:
        assert c.get("/v1/cases").status_code == 200


def test_ipv6_loopback_counts_as_local(no_token):
    with TestClient(app, client=("::1", 50000)) as c:
        assert c.get("/v1/cases").status_code == 200


@pytest.mark.parametrize("header", ["x-forwarded-for", "forwarded", "x-real-ip"])
def test_a_proxied_loopback_request_is_remote(no_token, header):
    """nginx on the same host connects from 127.0.0.1 too.

    Without this rule, putting AURA behind a TLS proxy would silently publish the
    open API to the whole network.
    """
    with TestClient(app, client=LOCAL, raise_server_exceptions=False) as c:
        r = c.get("/v1/cases", headers={header: "203.0.113.7"})
    assert r.status_code == 403


def test_probes_and_shells_stay_reachable_from_the_network(no_token):
    with TestClient(app, client=REMOTE) as c:
        assert c.get("/v1/health").status_code == 200
        assert c.get("/").status_code == 200
        status = c.get("/v1/auth/status").json()
    assert status["blocked"] is True
    assert status["auth_required"] is False
    assert "AURA_AUTH_TOKEN" in status["reason"]


def test_open_access_flag_restores_the_old_behaviour(monkeypatch):
    _settings(monkeypatch, auth_token="", allow_open_access=True)
    with TestClient(app, client=REMOTE) as c:
        assert c.get("/v1/cases").status_code == 200


# --------------------------------------------------------------------------- #
# Browser sessions
# --------------------------------------------------------------------------- #
@pytest.fixture
def secured(monkeypatch):
    _settings(monkeypatch, auth_token=TOKEN, allow_open_access=False)
    with TestClient(app, client=REMOTE, raise_server_exceptions=False) as c:
        yield c


def test_sign_in_sets_a_hardened_cookie_that_authenticates_reads(secured):
    r = secured.post("/v1/auth/session", json={"token": TOKEN, "user": "Dr Rao"})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    # No header at all: the cookie alone carries the credential and the principal.
    assert secured.get("/v1/cases").status_code == 200
    status = secured.get("/v1/auth/status").json()
    assert status["authenticated"] is True and status["principal"] == "Dr Rao"


def test_wrong_token_does_not_sign_in(secured):
    r = secured.post("/v1/auth/session", json={"token": "guess", "user": "mallory"})
    assert r.status_code == 401
    assert "set-cookie" not in r.headers
    assert secured.get("/v1/cases").status_code == 401


@pytest.mark.parametrize("user", ["", " ", "<script>", "x" * 65])
def test_sign_in_rejects_names_the_audit_log_cannot_hold(secured, user):
    r = secured.post("/v1/auth/session", json={"token": TOKEN, "user": user})
    assert r.status_code == 422


def test_sign_out_ends_the_session(secured):
    secured.post("/v1/auth/session", json={"token": TOKEN, "user": "Dr Rao"})
    assert secured.delete("/v1/auth/session").status_code == 200
    assert secured.get("/v1/cases").status_code == 401


def test_a_cookie_signed_with_another_token_is_rejected(secured):
    forged = security.issue_session("dr.a", "some-other-token", 3600)
    secured.cookies.set(security.SESSION_COOKIE, forged)
    assert secured.get("/v1/cases").status_code == 401


def test_a_cookie_with_an_edited_name_is_rejected():
    good = security.issue_session("dr.a", TOKEN, 3600)
    version, _name, expiry, mac = good.split(".")
    import base64
    other = base64.urlsafe_b64encode(b"dr.b").decode().rstrip("=")
    assert security.read_session(f"{version}.{other}.{expiry}.{mac}", TOKEN) is None


def test_an_expired_session_is_rejected():
    stale = security.issue_session("dr.a", TOKEN, ttl_s=60, now=time.time() - 3600)
    assert security.read_session(stale, TOKEN) is None


def test_rotating_the_token_ends_every_session():
    cookie = security.issue_session("dr.a", TOKEN, 3600)
    assert security.read_session(cookie, TOKEN) == "dr.a"
    assert security.read_session(cookie, TOKEN + "-rotated") is None


@pytest.mark.parametrize("junk", ["", "v1", "v1.a.b", "v2.a.1.b", "....", "v1.@@.1.x"])
def test_malformed_cookies_are_rejected_not_crashed_on(junk):
    assert security.read_session(junk, TOKEN) is None


def _request(headers: dict[str, str], client=REMOTE) -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/v1/cases", "root_path": "",
        "scheme": "http", "server": ("testserver", 80), "query_string": b"",
        "client": client,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    })


def test_the_signed_in_name_wins_over_a_self_asserted_header(monkeypatch):
    """The audit log must name who signed in, not whatever the page claims."""
    _settings(monkeypatch, auth_token=TOKEN, allow_open_access=False)
    cookie = security.issue_session("dr.rao", TOKEN, 3600)
    req = _request({"cookie": f"{security.SESSION_COOKIE}={cookie}",
                    "x-aura-user": "someone-else"})
    assert security.enforce(req) == "dr.rao"


def test_a_non_ascii_token_is_a_401_not_a_500(monkeypatch):
    """hmac.compare_digest raises TypeError on non-ASCII str; compare bytes instead."""
    _settings(monkeypatch, auth_token=TOKEN, allow_open_access=False)
    req = _request({"x-aura-token": "tökén", "x-aura-user": "dr.who"})
    with pytest.raises(Exception) as exc:
        security.enforce(req)
    assert getattr(exc.value, "status_code", None) == 401
