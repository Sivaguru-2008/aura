"""Gateway security controls — authentication, authorization, rate limiting, and
upload validation.

Design constraint: AURA's differentiator is that it runs **entirely offline** on an
edge box, so security must never require an external identity provider or network
call. Every control here is therefore local. Since 2026-09-12 the default is closed:
with no ``auth_token`` the gateway answers only direct connections from this machine,
so the offline single-box demo still works and nothing on a network can read a case;
``allow_open_access`` is the explicit opt-out. Setting the corresponding ``AURA_*``
env vars (or ``[tool.aura]`` keys) configures a deployment without touching any
endpoint signature.

  * Sessions        — browsers cannot send a custom header on a navigation or a
                      download, so they trade the token for a signed, HttpOnly,
                      SameSite=Strict cookie at ``POST /v1/auth/session``. The name
                      given at sign-in is the principal on every audit row.

  * Authentication  — a shared bearer token compared in constant time. When
                      ``auth_token`` is set, every request to a non-public path
                      requires the ``auth_header`` to carry it, **reads included**.
                      This gate used to be keyed on the HTTP method, which left
                      ``GET /v1/cases/{id}``, ``/export/fhir``, ``/export/hl7`` and
                      ``/v1/admin/safety`` reachable with no token at all — the read
                      surface is where the patient data actually leaves the box, so
                      the method is the wrong axis to gate on. Paths that must stay
                      open (health, the dashboard shell, static assets) are named in
                      ``PUBLIC_PATHS`` / ``PUBLIC_PREFIXES`` below.
  * Authorization   — the ``x-aura-user`` principal is recorded per request and
                      required (non-anonymous) once auth is enabled, so every
                      call is attributable in the audit log.
  * Rate limiting   — a per-principal in-process token bucket (no Redis needed),
                      enabled by ``rate_limit_rpm > 0``.
  * Upload guards   — a hard byte cap enforced while streaming the body (so a large
                      file can never be buffered whole) plus an extension/MIME
                      allowlist, layered *before* the content-based X-ray gate.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request, Response

from aura.common.config import get_settings

# Accepted upload types — layered in front of the content-based xray_gate. DICOM
# often has no extension, so the empty suffix is permitted and content-sniffed.
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom", ".zip", ".nii", ".gz", ".nrrd", ".nhdr", ""}
_ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/tiff", "application/dicom",
    "application/zip", "application/x-zip-compressed",
    "application/octet-stream", "", None,
    "application/x-gzip", "application/gzip",
    "application/x-nifti", "application/nifti", "application/nii",
}


# Paths reachable without a token even when auth is enabled. Deliberately a short
# allowlist rather than a denylist: a new endpoint is protected by default, and
# forgetting to register it fails closed (401) instead of silently exposing a case.
#   /v1/health  — liveness/readiness probes run before any credential is injected
#                 (docker-compose healthcheck, k8s probes). Returns no case data.
#   /v1/auth/status,
#   /v1/auth/session
#               — the sign-in handshake. Status says only whether this caller must
#                 sign in and whether it has; session trades the shared token for a
#                 signed, expiring cookie. Neither returns case data, and a browser
#                 cannot get past the gate without reaching them first.
#   /, /app,
#   /history,
#   /quantum    — HTML shells. They carry no patient data themselves; every panel is
#                 filled by a /v1/* fetch that IS gated. /quantum renders model-level
#                 evidence only and reads /v1/quantum/evidence, which is gated.
PUBLIC_PATHS = frozenset({"/v1/health", "/v1/auth/status", "/v1/auth/session",
                          "/v1/auth/login", "/v1/auth/logout",
                          "/", "/app", "/history", "/quantum",
                          "/docs", "/redoc", "/openapi.json", "/favicon.ico"})
PUBLIC_PREFIXES = ("/static", "/assets", "/css", "/js")


def is_public_path(path: str) -> bool:
    """True when *path* may be served without authentication.

    Kept as a function (not an inline check) so tests can enumerate the public
    surface and assert nothing under /v1/cases, /v1/studies or /v1/admin is in it.
    """
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


class RateLimiter:
    """Per-key sliding-window limiter. Process-local and thread-safe.

    A deque of recent request timestamps per key; a request is admitted iff fewer
    than ``rpm`` fall inside the trailing 60 s window. O(1) amortised.
    """

    def __init__(self, rpm: int):
        self.rpm = int(rpm)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.rpm <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.rpm:
                return False
            dq.append(now)
            return True


# Module-level limiter, sized from settings at import; rebuilt if settings change.
_LIMITER = RateLimiter(get_settings().rate_limit_rpm)


def _principal(request: Request) -> str:
    return request.headers.get("x-aura-user", "anonymous")


def _presented_token(request: Request, header: str) -> str:
    presented = request.headers.get(header, "")
    if not presented:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:]
    return presented


def _token_matches(presented: str, token: str) -> bool:
    # Compare bytes. hmac.compare_digest raises TypeError on a non-ASCII str, which
    # used to turn a garbage header into a 500 instead of a 401.
    return hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Secure default: with no token, answer this machine only
# --------------------------------------------------------------------------- #
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# Any of these means a proxy relayed the request, whatever address it arrived from.
_FORWARDING_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")

NOT_CONFIGURED_REASON = (
    "No AURA_AUTH_TOKEN is set, so this server only answers requests from the machine "
    "it runs on. Set AURA_AUTH_TOKEN to allow network access (clients then sign in "
    "with it), or AURA_ALLOW_OPEN_ACCESS=1 on an isolated demo network."
)


def is_local_request(request: Request) -> bool:
    """True only for a direct, un-proxied connection from this machine.

    A reverse proxy on the same host (nginx terminating TLS, say) also connects from
    127.0.0.1, so a forwarding header marks the request as remote. Without that rule,
    putting AURA behind a proxy would publish the open API to the whole network.
    """
    host = (request.client.host if request.client else "") or ""
    if host.startswith("::ffff:"):                   # IPv4-mapped IPv6
        host = host[len("::ffff:"):]
    if not (host in _LOOPBACK_HOSTS or host.startswith("127.")):
        return False
    return not any(h in request.headers for h in _FORWARDING_HEADERS)


# --------------------------------------------------------------------------- #
# Browser sessions
# --------------------------------------------------------------------------- #
# A browser cannot attach a custom header to a navigation or a download, and a token
# kept in JavaScript-readable storage is one XSS bug away from leaking. So browsers
# trade the shared token for a cookie that is HttpOnly (scripts cannot read it),
# SameSite=Strict (other sites cannot ride it) and signed (the name it carries cannot
# be edited). There is no server-side session store: the signature and the expiry are
# the whole state, so sessions survive a restart and need no database.
SESSION_COOKIE = "aura_session"
# 1-64 characters, no leading/trailing space. The name lands in every audit row.
_PRINCIPAL_RE = re.compile(r"^[A-Za-z0-9._@-](?:[A-Za-z0-9 ._@-]{0,62}[A-Za-z0-9._@-])?$")


def _session_key(token: str) -> bytes:
    # Derived, so the cookie MAC never keys directly on the API credential; derived
    # from the token, so rotating AURA_AUTH_TOKEN ends every session.
    return hashlib.sha256(b"aura-session-v1|" + token.encode("utf-8")).digest()


def _mac(token: str, message: str) -> str:
    return hmac.new(_session_key(token), message.encode("ascii"), hashlib.sha256).hexdigest()


def issue_session(principal: str, token: str, ttl_s: int, now: float | None = None) -> str:
    """Signed ``v1.<name>.<expiry>.<mac>`` cookie value for *principal*."""
    expires = int((time.time() if now is None else now) + ttl_s)
    name = base64.urlsafe_b64encode(principal.encode("utf-8")).decode("ascii").rstrip("=")
    message = f"v1.{name}.{expires}"
    return f"{message}.{_mac(token, message)}"


def read_session(cookie: str | None, token: str, now: float | None = None) -> str | None:
    """The principal of a valid, unexpired session cookie; ``None`` for anything else."""
    if not cookie or not token:
        return None
    parts = cookie.split(".")
    if len(parts) != 4 or parts[0] != "v1":
        return None
    _, name, expires, mac = parts
    try:
        good = _mac(token, f"v1.{name}.{expires}")
        if not hmac.compare_digest(mac.encode("ascii"), good.encode("ascii")):
            return None
        if int(expires) < (time.time() if now is None else now):
            return None
        principal = base64.urlsafe_b64decode(name + "=" * (-len(name) % 4)).decode("utf-8")
    except (ValueError, UnicodeError):
        return None
    return principal if _PRINCIPAL_RE.match(principal) else None


ROLES = ("AURA_ADMIN", "HOSPITAL_ADMIN", "CLINICAL_USER", "AUDITOR", "READ_ONLY")

ROLE_HIERARCHY: dict[str, set[str]] = {
    "AURA_ADMIN": {"AURA_ADMIN", "HOSPITAL_ADMIN", "CLINICAL_USER", "AUDITOR", "READ_ONLY"},
    "HOSPITAL_ADMIN": {"HOSPITAL_ADMIN", "CLINICAL_USER", "AUDITOR", "READ_ONLY"},
    "CLINICAL_USER": {"CLINICAL_USER", "READ_ONLY"},
    "AUDITOR": {"AUDITOR", "READ_ONLY"},
    "READ_ONLY": {"READ_ONLY"},
}


def require_role(*allowed_roles: str):
    """FastAPI dependency to require one of the specified roles or higher authority."""
    def _role_checker(request: Request):
        user_role = getattr(request.state, "role", "READ_ONLY")
        user_allowed = ROLE_HIERARCHY.get(user_role, {user_role})
        if not any(r in user_allowed for r in allowed_roles):
            raise HTTPException(
                403,
                {
                    "error": "forbidden",
                    "reason": f"Role '{user_role}' lacks required permissions; requires one of {list(allowed_roles)}",
                },
            )
        return user_role
    return _role_checker


def get_current_user(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if not user:
        return {
            "user_id": "usr-anonymous",
            "username": getattr(request.state, "principal", "anonymous"),
            "org_id": getattr(request.state, "tenant_id", "default"),
            "role": getattr(request.state, "role", "READ_ONLY"),
        }
    return user


def get_tenant_id(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if user and user.get("org_id"):
        # If AURA_ADMIN specifies x-aura-tenant header, allow tenant context override for cross-tenant operations
        if user.get("role") == "AURA_ADMIN" and request.headers.get("x-aura-tenant"):
            return request.headers.get("x-aura-tenant")
        return user["org_id"]
    return getattr(request.state, "tenant_id", "default")


def enforce(request: Request) -> str:
    """Authenticate + authorize + rate-limit one request. Returns the principal.

    Called from the gateway middleware for **every** request whose path is not
    public, so every endpoint is covered without per-handler wiring — reads as well
    as mutations. Raises 401/403/429 as appropriate.

    Supports database-backed multi-tenant user sessions (``aura_tok_*`` tokens or
    session cookies), legacy shared tokens (``AURA_AUTH_TOKEN``), and loopback demo access.
    """
    s = get_settings()
    principal = _principal(request)

    # Initialize request state defaults
    request.state.principal = principal
    request.state.tenant_id = "default"
    request.state.role = "READ_ONLY"
    request.state.user = None

    # Public paths skip the gate entirely
    if is_public_path(request.url.path):
        return principal

    presented = _presented_token(request, s.auth_header)
    cookie = request.cookies.get(SESSION_COOKIE)

    # 1. Try resolving a database-backed user session from presented token or cookie
    token_candidate = presented or cookie
    if token_candidate:
        st = None
        try:
            from aura.gateway.app import state as app_state
            st = app_state.get("store")
        except Exception:
            pass
        if st is not None:
            sess = st.validate_session(token_candidate)
            if sess is not None:
                principal = sess["username"]
                request.state.user = sess
                request.state.principal = principal
                request.state.tenant_id = sess["org_id"]
                request.state.role = sess["role"]
                if s.rate_limit_rpm > 0:
                    if _LIMITER.rpm != s.rate_limit_rpm:
                        _LIMITER.rpm = s.rate_limit_rpm
                    if not _LIMITER.allow(principal):
                        raise HTTPException(429, {"error": "rate_limited",
                                                  "reason": f"exceeded {s.rate_limit_rpm} req/min"})
                return principal
        if token_candidate.startswith("aura_tok_"):
            raise HTTPException(401, {"error": "unauthenticated",
                                      "reason": "Session token invalid, expired, or revoked"})

    # 2. Legacy / Shared Token Authentication
    if s.auth_token:
        signed_in = read_session(cookie, s.auth_token)
        if signed_in is not None:
            principal = signed_in
            request.state.user = {
                "user_id": f"usr-{principal}",
                "username": principal,
                "org_id": "default",
                "role": "CLINICAL_USER",
            }
            request.state.principal = principal
            request.state.tenant_id = "default"
            request.state.role = "CLINICAL_USER"
        else:
            if not _token_matches(presented, s.auth_token):
                raise HTTPException(401, {"error": "unauthenticated",
                                          "reason": "missing or invalid API token"})
            if principal == "anonymous":
                raise HTTPException(403, {"error": "forbidden",
                                          "reason": "x-aura-user principal required"})
            role = request.headers.get("x-aura-role", "AURA_ADMIN")
            tenant = request.headers.get("x-aura-tenant", "default")
            request.state.user = {
                "user_id": f"usr-{principal}",
                "username": principal,
                "org_id": tenant,
                "role": role,
            }
            request.state.principal = principal
            request.state.tenant_id = tenant
            request.state.role = role
    elif not s.allow_open_access and not is_local_request(request):
        raise HTTPException(403, {"error": "auth_not_configured",
                                  "reason": NOT_CONFIGURED_REASON})
    else:
        # Local request or open access: default administrative access for offline demo
        principal = request.headers.get("x-aura-user", "local")
        tenant = request.headers.get("x-aura-tenant", "default")
        role = request.headers.get("x-aura-role", "AURA_ADMIN")
        request.state.user = {
            "user_id": f"usr-{principal}",
            "username": principal,
            "org_id": tenant,
            "role": role,
        }
        request.state.principal = principal
        request.state.tenant_id = tenant
        request.state.role = role

    # 3. Rate limiting (per principal)
    if s.rate_limit_rpm > 0:
        if _LIMITER.rpm != s.rate_limit_rpm:
            _LIMITER.rpm = s.rate_limit_rpm
        if not _LIMITER.allow(principal):
            raise HTTPException(429, {"error": "rate_limited",
                                      "reason": f"exceeded {s.rate_limit_rpm} req/min"})
    return principal


# --------------------------------------------------------------------------- #
# Sign-in endpoints (mounted by gateway/app.py)
# --------------------------------------------------------------------------- #
router = APIRouter()

#: Failed sign-ins allowed per client address per minute. A 32-byte token cannot be
#: guessed at any rate; this keeps a misconfigured script from flooding the audit log.
_SIGNIN_LIMITER = RateLimiter(10)


def _is_https(request: Request) -> bool:
    return (request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https")


@router.get("/v1/auth/status")
def auth_status(request: Request) -> dict:
    """Whether this caller must sign in, and whether it has. Returns no case data."""
    s = get_settings()
    signed_in = (read_session(request.cookies.get(SESSION_COOKIE), s.auth_token)
                 if s.auth_token else None)
    blocked = bool(not s.auth_token and not s.allow_open_access
                   and not is_local_request(request))
    return {
        "auth_required": bool(s.auth_token),
        "authenticated": signed_in is not None or getattr(request.state, "user", None) is not None,
        "principal": signed_in or getattr(request.state, "principal", None),
        "open_access": bool(not s.auth_token and s.allow_open_access),
        "blocked": blocked,
        "reason": NOT_CONFIGURED_REASON if blocked else "",
    }


@router.post("/v1/auth/session")
def sign_in(request: Request, response: Response, body: dict = Body(...)) -> dict:
    """Exchange the shared token plus a name for a signed session cookie."""
    s = get_settings()
    if not s.auth_token:
        raise HTTPException(400, {"error": "auth_disabled",
                                  "reason": "no AURA_AUTH_TOKEN is configured, so there is "
                                            "nothing to sign in to"})
    user = str(body.get("user", "")).strip()
    if not _PRINCIPAL_RE.match(user):
        raise HTTPException(422, {"error": "invalid_user",
                                  "reason": "name must be 1-64 characters: letters, digits, "
                                            "spaces and . _ @ -"})
    if not _token_matches(str(body.get("token", "")), s.auth_token):
        client = request.client.host if request.client else "unknown"
        if not _SIGNIN_LIMITER.allow(client):
            raise HTTPException(429, {"error": "rate_limited",
                                      "reason": "too many failed sign-ins; wait a minute"})
        raise HTTPException(401, {"error": "unauthenticated",
                                  "reason": "that access token is not valid"})
    ttl = int(max(0.25, float(s.session_ttl_hours)) * 3600)
    response.set_cookie(SESSION_COOKIE, issue_session(user, s.auth_token, ttl),
                        max_age=ttl, httponly=True, samesite="strict",
                        secure=_is_https(request), path="/")
    return {"authenticated": True, "principal": user, "expires_in_s": ttl}


@router.delete("/v1/auth/session")
def sign_out(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return {"authenticated": False}


@router.post("/v1/auth/login")
def login(request: Request, response: Response, body: dict = Body(...)) -> dict:
    """Authenticate with username + password and receive an active session token."""
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    if not username or not password:
        raise HTTPException(400, {"error": "bad_request", "reason": "username and password required"})

    client = request.client.host if request.client else "unknown"
    from aura.gateway.app import state as app_state
    st = app_state.get("store")
    if not st:
        raise HTTPException(503, {"error": "service_unavailable", "reason": "Storage not initialized"})

    user = st.get_user_by_username(username)
    if not user or not user.get("is_active"):
        if not _SIGNIN_LIMITER.allow(client):
            raise HTTPException(429, {"error": "rate_limited", "reason": "Too many sign-in attempts; wait a minute"})
        raise HTTPException(401, {"error": "unauthenticated", "reason": "Invalid credentials"})

    if not st.verify_password(password, user["salt"], user["password_hash"]):
        if not _SIGNIN_LIMITER.allow(client):
            raise HTTPException(429, {"error": "rate_limited", "reason": "Too many sign-in attempts; wait a minute"})
        raise HTTPException(401, {"error": "unauthenticated", "reason": "Invalid credentials"})

    s = get_settings()
    ttl_hours = int(max(0.25, float(s.session_ttl_hours)))
    token, session_info = st.create_session(
        user_id=user["user_id"],
        org_id=user["org_id"],
        role=user["role"],
        duration_hours=ttl_hours,
    )

    ttl_s = ttl_hours * 3600
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=ttl_s,
        httponly=True,
        samesite="strict",
        secure=_is_https(request),
        path="/",
    )

    st.audit(
        action="auth.login",
        entity_type="user",
        entity_id=user["user_id"],
        actor=user["username"],
        tenant_id=user["org_id"],
        detail={"role": user["role"], "org_id": user["org_id"]},
    )

    return {
        "authenticated": True,
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "org_id": user["org_id"],
        },
        "expires_in_s": ttl_s,
    }


@router.post("/v1/auth/logout")
def logout(request: Request, response: Response) -> dict:
    """Revoke session token and clear authentication cookie."""
    from aura.gateway.app import state as app_state
    st = app_state.get("store")
    token = _presented_token(request, "x-aura-token") or request.cookies.get(SESSION_COOKIE)
    if st and token:
        st.revoke_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return {"authenticated": False, "message": "Logged out successfully"}


@router.get("/v1/auth/me")
def me(request: Request) -> dict:
    """Return the authenticated user profile and active permissions."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(401, {"error": "unauthenticated", "reason": "Not authenticated"})
    return {"user": user}


def validate_upload_name(filename: str | None, content_type: str | None) -> None:
    """Extension + declared-MIME allowlist. Raises 415 on a disallowed type."""
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(415, {"error": "unsupported_media_type",
                                  "reason": f"extension '{suffix}' not allowed"})
    if content_type is not None and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, {"error": "unsupported_media_type",
                                  "reason": f"content-type '{content_type}' not allowed"})


async def read_capped(file, max_bytes: int) -> bytes:
    """Stream an UploadFile into memory, aborting past ``max_bytes`` (DoS guard).

    Reads in chunks so a hostile ``Content-Length`` can never force the whole body
    into RAM — the previous ``await file.read()` was an unbounded allocation.
    """
    chunks: list[bytes] = []
    total = 0
    size = 1 << 20                                       # 1 MiB chunks
    while True:
        chunk = await file.read(size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, {"error": "payload_too_large",
                                      "reason": f"upload exceeds {max_bytes} bytes"})
        chunks.append(chunk)
    return b"".join(chunks)


def parse_nifti_header(source: Path | str | bytes) -> tuple[int, ...] | None:
    """Parse NIfTI-1 or NIfTI-2 header to extract image shape.
    
    Supports gzipped and raw NIfTI files, as well as direct byte payloads.
    """
    import struct
    import gzip

    try:
        if isinstance(source, bytes):
            payload = source
            if payload.startswith(b"\x1f\x8b"):
                try:
                    payload = gzip.decompress(payload)
                except Exception:
                    return None
            header = payload[:540]
        else:
            path = Path(source)
            if not path.exists():
                return None
            is_gz = path.suffix.lower() == ".gz" or path.name.lower().endswith(".nii.gz")
            if is_gz:
                try:
                    with gzip.open(path, "rb") as f:
                        header = f.read(540)
                except Exception:
                    try:
                        with open(path, "rb") as f:
                            header = f.read(540)
                    except Exception:
                        return None
            else:
                try:
                    with open(path, "rb") as f:
                        header = f.read(540)
                except Exception:
                    return None
                        
        if len(header) < 348:
            return None
            
        sizeof_hdr_le = struct.unpack("<i", header[:4])[0]
        sizeof_hdr_be = struct.unpack(">i", header[:4])[0]
        
        endian = None
        if sizeof_hdr_le == 348:
            endian = "<"
        elif sizeof_hdr_be == 348:
            endian = ">"
            
        if endian:
            dim = struct.unpack(f"{endian}8h", header[40:56])
            ndim = dim[0]
            if 0 < ndim <= 7:
                return tuple(int(x) for x in dim[1:1+ndim])
                
        if len(header) >= 12:
            magic2 = header[4:12]
            if magic2 in (b"n+2\x00\r\n\x1a\n", b"ni2\x00\r\n\x1a\n"):
                sizeof_hdr2_le = struct.unpack("<i", header[:4])[0]
                sizeof_hdr2_be = struct.unpack(">i", header[:4])[0]
                endian2 = None
                if sizeof_hdr2_le == 540:
                    endian2 = "<"
                elif sizeof_hdr2_be == 540:
                    endian2 = ">"
                if endian2:
                    if len(header) >= 80:
                        dim = struct.unpack(f"{endian2}8q", header[16:80])
                        ndim = dim[0]
                        if 0 < ndim <= 7:
                            return tuple(int(x) for x in dim[1:1+ndim])
    except Exception:
        pass
    return None


def validate_mri_content(payload: bytes, filename: str | None) -> None:
    """Perform content-based signature and integrity checks on MRI/NIfTI uploads.
    
    Validates that:
    - .gz / .nii.gz files are valid gzip archives that decompress into valid NIfTI volumes.
    - .nii files are valid NIfTI volumes.
    - Raises HTTPException(422, ...) on validation failures.
    """
    import tempfile
    import gzip
    
    filename_lower = filename.lower() if filename else ""
    is_nii = filename_lower.endswith(".nii")
    is_gz = filename_lower.endswith(".nii.gz") or filename_lower.endswith(".gz")
    
    if not (is_nii or is_gz):
        return

    # 1. Inspect file signature (magic bytes check)
    if is_gz:
        # Gzip magic is \x1f\x8b
        if not payload.startswith(b"\x1f\x8b"):
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Corrupted MRI archive"})
        
        # Test decompression
        try:
            decompressed = gzip.decompress(payload)
        except Exception:
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Corrupted MRI archive"})
            
        # Inspect decompressed magic bytes for NIfTI
        if len(decompressed) < 348:
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Invalid NIfTI file"})
            
        magic_nifti1 = decompressed[344:348]
        magic_nifti2 = decompressed[4:12]
        
        is_nifti1 = magic_nifti1 in (b"n+1\x00", b"ni1\x00", b"n+1\x00\x00", b"ni1\x00\x00")
        is_nifti2 = magic_nifti2 in (b"n+2\x00\r\n\x1a\n", b"ni2\x00\r\n\x1a\n")
        
        if not (is_nifti1 or is_nifti2):
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Unsupported MRI format"})
            
    elif is_nii:
        # Inspect magic bytes directly
        if len(payload) < 348:
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Invalid NIfTI file"})
            
        magic_nifti1 = payload[344:348]
        magic_nifti2 = payload[4:12]
        
        is_nifti1 = magic_nifti1 in (b"n+1\x00", b"ni1\x00", b"n+1\x00\x00", b"ni1\x00\x00")
        is_nifti2 = magic_nifti2 in (b"n+2\x00\r\n\x1a\n", b"ni2\x00\r\n\x1a\n")
        
        if not (is_nifti1 or is_nifti2):
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Unsupported MRI format"})

    # 2. Validate using nibabel if available, otherwise fallback to direct header parsing
    try:
        import nibabel as nib
        from nibabel.filebasedimages import ImageFileError
        nib_available = True
    except ImportError:
        nib_available = False

    if nib_available:
        suffix = ".nii.gz" if filename_lower.endswith(".nii.gz") else Path(filename_lower).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            img = nib.load(tmp_path)
            if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
                raise HTTPException(422, {"error": "study_validation_failed", "reason": "Unsupported MRI format"})
            shape = img.shape
            if not shape or len(shape) < 3 or any(s <= 0 for s in shape):
                raise HTTPException(422, {"error": "study_validation_failed", "reason": "Invalid NIfTI file"})
        except HTTPException:
            raise
        except ImageFileError as exc:
            msg = str(exc)
            if "not a gzip file" in msg:
                raise HTTPException(422, {"error": "study_validation_failed", "reason": "Corrupted MRI archive"})
            elif "Cannot work out file type" in msg or "Unsupported" in msg:
                raise HTTPException(422, {"error": "study_validation_failed", "reason": "Unsupported MRI format"})
            else:
                raise HTTPException(422, {"error": "study_validation_failed", "reason": "Invalid NIfTI file"})
        except Exception:
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Invalid NIfTI file"})
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
    else:
        # Fallback security check using direct header parsing when nibabel is absent
        shape = parse_nifti_header(payload)
        if shape is None or len(shape) < 3 or any(s <= 0 for s in shape):
            raise HTTPException(422, {"error": "study_validation_failed", "reason": "Invalid NIfTI file"})
