"""Cross-cutting primitives: error taxonomy, structured logging, shared types."""

from backend.core.shared.errors import (
    AuraBackendError,
    EngineExecutionError,
    EngineNotAvailable,
    EngineNotImplemented,
    ModalityUndetermined,
    UnreadableImage,
    UnsupportedModality,
    UploadRejected,
)
from backend.core.shared.logging import (
    correlation_id,
    get_logger,
    new_correlation_id,
    use_correlation_id,
)
from backend.core.shared.types import (
    MODALITY_LABELS,
    EngineStatus,
    ImageAsset,
    ImagingModality,
    to_clinical_modality,
)

__all__ = [
    "AuraBackendError",
    "EngineExecutionError",
    "EngineNotAvailable",
    "EngineNotImplemented",
    "ModalityUndetermined",
    "UnreadableImage",
    "UnsupportedModality",
    "UploadRejected",
    "correlation_id",
    "get_logger",
    "new_correlation_id",
    "use_correlation_id",
    "MODALITY_LABELS",
    "EngineStatus",
    "ImageAsset",
    "ImagingModality",
    "to_clinical_modality",
]
