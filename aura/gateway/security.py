"""Gateway security controls — authentication, authorization, rate limiting, and
upload validation.

Design constraint: AURA's differentiator is that it runs **entirely offline** on an
edge box, so security must never require an external identity provider or network
call. Every control here is therefore local and, crucially, **opt-in**: with the
default configuration (no ``auth_token``, ``rate_limit_rpm=0``) the controls are
inert and the P0 offline demo behaves exactly as before. Setting the corresponding
``AURA_*`` env vars (or ``[tool.aura]`` keys) switches them on for a deployment
without touching any endpoint signature.

  * Authentication  — a shared bearer token compared in constant time. When
                      ``auth_token`` is set, mutating endpoints (POST/PUT/DELETE)
                      require the ``auth_header`` to carry it.
  * Authorization   — the ``x-aura-user`` principal is recorded per request and
                      required (non-anonymous) once auth is enabled, so every
                      mutation is attributable in the audit log.
  * Rate limiting   — a per-principal in-process token bucket (no Redis needed),
                      enabled by ``rate_limit_rpm > 0``.
  * Upload guards   — a hard byte cap enforced while streaming the body (so a large
                      file can never be buffered whole) plus an extension/MIME
                      allowlist, layered *before* the content-based X-ray gate.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException, Request

from common.config import get_settings

# Accepted upload types — layered in front of the content-based xray_gate. DICOM
# often has no extension, so the empty suffix is permitted and content-sniffed.
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom", ""}
_ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/tiff", "application/dicom",
    "application/octet-stream", "", None,
}


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


def enforce(request: Request) -> str:
    """Authenticate + authorize + rate-limit one request. Returns the principal.

    Inert unless configured. Raises 401/403/429 as appropriate. Called from the
    gateway middleware for mutating methods so every endpoint is covered without
    per-handler wiring.
    """
    s = get_settings()
    principal = _principal(request)

    # --- authentication (shared bearer token, constant-time compare) ---
    if s.auth_token:
        presented = request.headers.get(s.auth_header, "")
        if not presented:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                presented = auth[7:]
        if not hmac.compare_digest(str(presented), str(s.auth_token)):
            raise HTTPException(401, {"error": "unauthenticated",
                                      "reason": "missing or invalid API token"})
        # --- authorization: an authenticated call must name its actor ---
        if principal == "anonymous":
            raise HTTPException(403, {"error": "forbidden",
                                      "reason": "x-aura-user principal required"})

    # --- rate limiting (per principal) ---
    if s.rate_limit_rpm > 0:
        if _LIMITER.rpm != s.rate_limit_rpm:            # settings changed at runtime
            _LIMITER.rpm = s.rate_limit_rpm
        if not _LIMITER.allow(principal):
            raise HTTPException(429, {"error": "rate_limited",
                                      "reason": f"exceeded {s.rate_limit_rpm} req/min"})
    return principal


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
