"""Errors raised by the Brain Vision Engine.

Same shape as the foundation layer's error family (``code`` / ``reason`` / ``detail``)
so a caller that already handles :class:`~aura.backend.foundation.mri.errors.MRIFoundationError`
handles these the same way, and so a failure that crosses the engine boundary arrives
as a structured payload rather than a string.
"""
from __future__ import annotations

from typing import Any


class BrainVisionError(Exception):
    """Base class. ``code`` is the stable machine-readable identifier."""

    code = "brain_vision_error"

    def __init__(self, reason: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail: dict[str, Any] = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "reason": self.reason, "detail": dict(self.detail)}


class CorpusNotFound(BrainVisionError):
    """The training corpus is not where configuration says it is."""

    code = "corpus_not_found"


class CorpusIntegrityError(BrainVisionError):
    """The corpus is present but does not have the structure this module requires.

    Raised for the failures that would otherwise train a model on silently wrong data:
    a volume missing slices, a label with values outside the declared space, a channel
    assignment that fails its own verification.
    """

    code = "corpus_integrity_error"


class CacheUnavailable(BrainVisionError):
    """The ingest cache is absent, incomplete, or written by an incompatible version."""

    code = "cache_unavailable"


class ConfigurationError(BrainVisionError):
    """A configuration combination that cannot be satisfied.

    Preferred over silently correcting the caller: a curriculum whose stages exceed the
    epoch budget, or a loss with every weight at zero, is a mistake worth surfacing
    before a training run consumes an afternoon.
    """

    code = "configuration_error"


class ArchitectureUnavailable(BrainVisionError):
    """A requested encoder or decoder is not registered in this deployment.

    The message names what *is* registered. Declaring an architecture the package does
    not implement — a 3D U-Net, SwinUNETR — and failing loudly is the honest posture;
    quietly substituting the 2D network would produce a model card that lies.
    """

    code = "architecture_unavailable"


class CheckpointError(BrainVisionError):
    """A checkpoint could not be written, read, or reconciled with the current model."""

    code = "checkpoint_error"


class ModelNotTrained(BrainVisionError):
    """Inference was requested and no trained weights are available.

    Deliberately fatal rather than falling back to a randomly initialised network:
    a segmentation produced by untrained weights looks exactly like one produced by
    trained weights, and there is no downstream check that would catch it.
    """

    code = "model_not_trained"


__all__ = [
    "ArchitectureUnavailable", "BrainVisionError", "CacheUnavailable",
    "CheckpointError", "ConfigurationError", "CorpusIntegrityError", "CorpusNotFound",
    "ModelNotTrained",
]
