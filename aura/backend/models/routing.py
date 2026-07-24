"""Routing + analysis response contracts.

:class:`RoutingMetadata` is the required router return shape::

    {modality, selected_engine, confidence, supported, reason}

Those five fields are first-class and their meaning is frozen. Everything after them
is additive context — a client that only reads the five keys keeps working as the
platform grows.

The extra fields exist because a bare confidence number is not auditable. A clinician
(or a reviewer six months later) needs to know *how* the modality was decided: from a
DICOM ``Modality`` tag, or from pixel statistics; and whether the thresholds behind
that decision were calibrated against real images or are provisional. ``evidence``,
``detector``, and ``calibrated`` carry that, and ``requires_review`` is the actionable
summary of it.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModalityCandidate(BaseModel):
    """One modality the detector scored, with the evidence behind the score."""

    modality: str = Field(..., description="ImagingModality value, e.g. 'chest_xray'.")
    label: str = Field("", description="Human-readable modality name.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated: bool = Field(
        ...,
        description="True when this score's thresholds were fitted on a real image "
                    "corpus; False when they are provisional and confidence is capped.",
    )
    source: str = Field(
        ...,
        description="How the score was produced: 'dicom_metadata' | 'pixel_signature' "
                    "| 'pixel_signature+dicom_metadata'.",
    )
    reason: str = Field("", description="One line explaining this candidate's score.")
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Measured values the score came from (tags read, features computed).",
    )


class RoutingMetadata(BaseModel):
    """The router's decision for one upload."""

    # ---- required contract (requirement 6) --------------------------------- #
    modality: str = Field(..., description="Detected ImagingModality value.")
    selected_engine: str | None = Field(
        ..., description="Engine id that will analyse the study; null when unsupported."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    supported: bool = Field(
        ..., description="True when a registered engine claims this modality."
    )
    reason: str = Field(..., description="Human-readable explanation of the decision.")

    # ---- audit / diagnostics ------------------------------------------------ #
    modality_label: str = Field("", description="Human-readable modality name.")
    request_id: str = Field("", description="Correlation id shared with the server log.")
    detector: str = Field("", description="Detector implementation that decided this.")
    calibrated: bool = Field(
        True,
        description="False when the winning signature's thresholds are provisional. "
                    "Such decisions carry capped confidence and require review.",
    )
    requires_review: bool = Field(
        False,
        description="True when the decision is low-confidence, uncalibrated, or "
                    "contested by a close runner-up. A human should confirm the route.",
    )
    engine_status: str | None = Field(
        None, description="Lifecycle state of the selected engine (see EngineStatus)."
    )
    candidates: list[ModalityCandidate] = Field(
        default_factory=list,
        description="All scored modalities, highest first — including the losers.",
    )
    asset_sha256: str = Field("", description="Content hash of the routed bytes.")


class ResultStatus(str, Enum):
    """Outcome of running an engine (distinct from HTTP status)."""

    COMPLETED = "completed"
    #: Routing resolved to a declared placeholder engine. Not an error: the route is
    #: correct and the engine is honest about not being built yet.
    NOT_IMPLEMENTED = "not_implemented"
    #: Routing succeeded but no engine claims the modality.
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class EngineOutcome(BaseModel):
    """What an engine produced for one study."""

    status: ResultStatus
    engine: str = Field(..., description="Engine id that ran (or would have run).")
    engine_version: str = ""
    message: str = Field("", description="Human-readable summary or reason.")
    case_id: str | None = Field(
        None, description="Persisted case id, when the engine created one."
    )
    #: Engine-specific clinical payload. Shape is owned by the engine and documented
    #: in its module — the routing layer never interprets it.
    payload: dict[str, Any] = Field(default_factory=dict)
    #: Rendered report sections, when the engine produced a report.
    report: dict[str, Any] | None = None


class AnalysisEnvelope(BaseModel):
    """Full response for an analyse request: how it routed, and what came back."""

    request_id: str
    routing: RoutingMetadata
    result: EngineOutcome | None = None
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Wall-clock per stage: intake, detection, analysis, total.",
    )


class EngineDescriptorModel(BaseModel):
    """Registry entry as exposed by ``GET /v1/engines``."""

    engine_id: str
    display_name: str
    version: str
    status: str
    modalities: list[str]
    capabilities: list[str] = Field(default_factory=list)
    description: str = ""
