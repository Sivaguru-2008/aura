"""Error taxonomy for the routing layer.

Every failure the backend can produce is one of these, and every one carries a
machine-readable ``code`` plus an HTTP status. The API layer converts them to
responses mechanically, so no handler has to invent an error shape and no internal
exception text ever reaches a client.

Two rules inherited from the existing gateway (audit §10.9 / §11.5):

* the client sees ``{"error": <code>, "reason": <safe message>}`` and nothing else —
  never a stack trace, filesystem path, or raw exception string;
* the *server* log keeps the full detail, so a failure stays diagnosable.

``detail`` is for structured, client-safe context (which checks failed, what was
detected). Anything sensitive belongs in the log, not here.
"""
from __future__ import annotations

from typing import Any


class AuraBackendError(Exception):
    """Base class for every routing-layer failure.

    Subclasses set ``code`` (stable, machine-readable) and ``http_status``. The
    message passed in is the client-safe reason.
    """

    code: str = "backend_error"
    http_status: int = 500

    def __init__(self, reason: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail: dict[str, Any] = detail or {}

    def to_payload(self) -> dict[str, Any]:
        """Client-facing body. Stable across versions — clients switch on ``error``."""
        body: dict[str, Any] = {"error": self.code, "reason": self.reason}
        if self.detail:
            body["detail"] = self.detail
        return body

    def __repr__(self) -> str:                                # pragma: no cover - debug aid
        return f"{type(self).__name__}(code={self.code!r}, reason={self.reason!r})"


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #
class UploadRejected(AuraBackendError):
    """The upload failed a transport-level guard (type allowlist or size cap).

    Raised before a single pixel is decoded, so a hostile file cannot reach the
    decoders at all. ``http_status`` is overridden per instance because the two
    causes map to different codes (415 vs 413).
    """

    code = "upload_rejected"
    http_status = 415

    def __init__(self, reason: str, *, http_status: int = 415,
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason, detail=detail)
        self.http_status = http_status


class UnreadableImage(AuraBackendError):
    """The bytes arrived intact but no decoder could turn them into an image."""

    code = "unreadable_image"
    http_status = 422


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
class ModalityUndetermined(AuraBackendError):
    """No modality signature scored high enough to commit to a route.

    This is the *safe* failure: the router declines rather than guessing, which is
    the same posture the safety engine takes when it abstains. Detection evidence
    travels in ``detail`` so a human can see why nothing matched.
    """

    code = "modality_undetermined"
    http_status = 422


class UnsupportedModality(AuraBackendError):
    """The modality was identified confidently, but no engine claims it.

    Distinct from :class:`ModalityUndetermined`: here AURA knows what it is looking
    at (e.g. an abdominal ultrasound) and knows it has nothing to offer yet.
    """

    code = "unsupported_modality"
    http_status = 422


class ModalityConflict(AuraBackendError):
    """The caller declared a modality the detector confidently contradicts.

    Raised only when the detector *committed* to a different modality on a calibrated
    signature — a genuine contradiction, not merely a weak or absent detection. An
    undetermined image does not conflict with anything, so the declaration stands.

    The declaration is honoured everywhere else because it is real evidence: a user
    clicking "upload brain MRI" knows something the pixels do not carry. What it must
    never do is *override measurement*. Running the chest model because a form field
    said so is how a brain MRI comes back reported as pneumonia; running the brain
    model on a chest film because the other button was clicked is the same failure
    with the labels swapped.
    """

    code = "modality_conflict"
    http_status = 422


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
class EngineNotAvailable(AuraBackendError):
    """The selected engine is registered but could not be constructed.

    Typically a missing model artifact or an optional dependency that is not
    installed. Routing succeeded; execution could not start.
    """

    code = "engine_unavailable"
    http_status = 503


class EngineNotImplemented(AuraBackendError):
    """Routing resolved to an engine that is a declared placeholder.

    This is a normal, expected outcome while an engine is under construction — it
    proves the route is wired correctly. The dispatch service converts it into a
    structured ``not_implemented`` result rather than an error response, so callers
    still receive the full routing metadata.
    """

    code = "engine_not_implemented"
    http_status = 501


class EngineExecutionError(AuraBackendError):
    """The engine ran and failed. The underlying exception is logged, never returned."""

    code = "engine_failed"
    http_status = 500
