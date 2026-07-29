"""Upload intake — the single door every image enters through.

Responsibilities, in order:

1. **Type allowlist and size cap.** Delegated to ``gateway.security``, the module the
   legacy upload endpoint already uses. Reused rather than reimplemented so the two
   paths cannot drift apart: whatever the deployment allows through
   ``POST /v1/studies/upload``, it allows through the router, and no more.
2. **Staging to a temp file.** The decoders (pydicom, OpenCV, PIL) all want a path,
   and so does the existing CXR gate. One staged file is read by intake, the router,
   and the engine — the bytes are never re-uploaded or copied between stages.
3. **Content hashing.** The SHA-256 is the upload's identity in audit records. It ties
   a routing decision to exact bytes without retaining the image itself, which matters
   when the image is a patient radiograph.
4. **Cleanup.** Always, including on every error path. Medical images must not
   accumulate in a temp directory.

The context manager is the only supported entry point, precisely so that cleanup
cannot be forgotten::

    async with UploadIntake().receive(file) as asset:
        decision = router.route(asset)
    # temp file is gone here, whatever happened inside
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from ..shared.errors import UploadRejected
from ..shared.logging import get_logger
from ..shared.types import ImageAsset

log = get_logger("upload")

#: Suffix used when the upload has none. DICOM commonly arrives extensionless, and
#: the decoders sniff content rather than trusting the name, so this is only a
#: filesystem convenience.
_DEFAULT_SUFFIX = ".bin"


def _read_dicom_tags(path: Path) -> dict[str, str]:
    """Best-effort DICOM header read for the audit record.

    Deliberately header-only (``stop_before_pixels``): intake must stay cheap, and
    the router does the full decode once, later. Returns ``{}`` for anything that is
    not DICOM — including when pydicom is not installed.
    """
    try:
        import pydicom
    except ImportError:
        return {}
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception:
        return {}
    tags = {
        "modality": str(getattr(ds, "Modality", "") or ""),
        "body_part": str(getattr(ds, "BodyPartExamined", "") or ""),
        "study_description": str(getattr(ds, "StudyDescription", "") or ""),
        "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
    }
    return {k: v for k, v in tags.items() if v}


@contextlib.contextmanager
def stage_bytes(payload: bytes, filename: str | None = None,
                content_type: str | None = None) -> Iterator[ImageAsset]:
    """Stage an in-memory image and yield it as an :class:`ImageAsset`.

    Separate from :class:`UploadIntake` so tests, CLI tools, and batch jobs can route
    a file without constructing a FastAPI ``UploadFile``. The transport guards are
    *not* applied here — this is the trusted, in-process entry point; anything
    arriving over HTTP must go through :class:`UploadIntake`.
    """
    name = filename or "upload"
    suffix = Path(name).suffix or _DEFAULT_SUFFIX
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="aura-intake-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        path = Path(tmp_path)
        asset = ImageAsset(
            path=path,
            original_filename=name,
            content_type=content_type,
            size_bytes=len(payload),
            sha256=ImageAsset.digest(payload),
            dicom_tags=_read_dicom_tags(path),
        )
        log.info(
            "upload staged",
            extra={"context": {"filename": name, "bytes": len(payload),
                               "sha256": asset.sha256[:12],
                               "dicom": bool(asset.dicom_tags)}},
        )
        yield asset
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        except OSError as exc:                        # pragma: no cover - fs failure
            # Never fatal, but never silent: a leaked patient image is a real problem.
            log.error(f"failed to remove staged upload {tmp_path}: {exc}")


class UploadIntake:
    """HTTP upload intake with the deployment's transport guards applied."""

    def __init__(self, max_upload_mb: float | None = None) -> None:
        self._max_upload_mb = max_upload_mb

    @property
    def max_bytes(self) -> int:
        """Hard size cap, read from settings unless explicitly overridden."""
        if self._max_upload_mb is not None:
            return int(self._max_upload_mb * 1024 * 1024)
        from aura.common.config import get_settings

        return int(get_settings().max_upload_mb * 1024 * 1024)

    @contextlib.asynccontextmanager
    async def receive(self, file: Any) -> AsyncIterator[ImageAsset]:
        """Validate, stream, stage, and yield one uploaded file.

        ``file`` is a Starlette ``UploadFile``. Raises :class:`UploadRejected` before
        any bytes are staged when the type is not allowed, and mid-stream if the size
        cap is exceeded — so an oversized body is never buffered whole.
        """
        from fastapi import HTTPException

        from aura.gateway.security import read_capped, validate_upload_name, validate_mri_content

        filename = getattr(file, "filename", None) or "upload"
        content_type = getattr(file, "content_type", None)

        try:
            validate_upload_name(filename, content_type)
            payload = await read_capped(file, self.max_bytes)
            validate_mri_content(payload, filename)
        except HTTPException as exc:
            # Translate the gateway's HTTP-shaped guard into the backend taxonomy, so
            # the API layer has exactly one error type to render.
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            reason = detail.get("reason") or str(exc.detail)
            log.warning(
                "upload rejected by transport guard",
                extra={"context": {"filename": filename, "status": exc.status_code,
                                   "reason": reason}},
            )
            raise UploadRejected(reason, http_status=exc.status_code,
                                 detail={"filename": filename}) from exc

        if not payload:
            raise UploadRejected("the uploaded file is empty", http_status=422,
                                 detail={"filename": filename})

        with stage_bytes(payload, filename, content_type) as asset:
            yield asset

    @contextlib.asynccontextmanager
    async def receive_multiple(self, files: list[Any]) -> AsyncIterator[ImageAsset]:
        """Validate, stream, stage, and yield multiple uploaded files in a temp directory."""
        from fastapi import HTTPException
        import hashlib
        import shutil
        from aura.gateway.security import read_capped, validate_upload_name, validate_mri_content

        if not files:
            raise UploadRejected("no files uploaded", http_status=400)

        temp_dir = tempfile.mkdtemp(prefix="aura-intake-multi-")
        temp_path = Path(temp_dir)
        
        try:
            total_bytes = 0
            hashes = []
            
            for file in files:
                filename = getattr(file, "filename", None) or "upload"
                content_type = getattr(file, "content_type", None)
                
                try:
                    validate_upload_name(filename, content_type)
                    payload = await read_capped(file, self.max_bytes)
                    validate_mri_content(payload, filename)
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, dict) else {}
                    reason = detail.get("reason") or str(exc.detail)
                    log.warning(
                        "upload rejected by transport guard",
                        extra={"context": {"filename": filename, "status": exc.status_code, "reason": reason}},
                    )
                    raise UploadRejected(reason, http_status=exc.status_code, detail={"filename": filename}) from exc

                if not payload:
                    continue
                
                total_bytes += len(payload)
                file_hash = ImageAsset.digest(payload)
                hashes.append((filename, file_hash))
                
                target_file_path = temp_path / filename
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_file_path, "wb") as f:
                    f.write(payload)

            if total_bytes == 0:
                raise UploadRejected("the uploaded files are empty", http_status=422)

            hashes.sort()
            combined_string = "".join(h[1] for h in hashes)
            combined_hash = hashlib.sha256(combined_string.encode("utf-8")).hexdigest()

            asset = ImageAsset(
                path=temp_path,
                original_filename="multiple_files",
                content_type="application/x-directory",
                size_bytes=total_bytes,
                sha256=combined_hash,
                dicom_tags={},
            )
            
            log.info(
                "multiple upload staged",
                extra={"context": {"files": len(files), "bytes": total_bytes, "sha256": asset.sha256[:12]}},
            )
            
            yield asset

        finally:
            try:
                shutil.rmtree(temp_dir)
            except OSError as exc:
                log.error(f"failed to remove staged temp directory {temp_dir}: {exc}")

