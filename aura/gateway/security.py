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

from aura.common.config import get_settings

# Accepted upload types — layered in front of the content-based xray_gate. DICOM
# often has no extension, so the empty suffix is permitted and content-sniffed.
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom", ".zip", ".nii", ".gz", ""}
_ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/tiff", "application/dicom",
    "application/zip", "application/x-zip-compressed",
    "application/octet-stream", "", None,
    "application/x-gzip", "application/gzip",
    "application/x-nifti", "application/nifti", "application/nii",
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
