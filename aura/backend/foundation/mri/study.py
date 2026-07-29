"""Foundation Data Model — module 7: the object every NeuroMind module consumes.

:class:`StandardizedSeries` is the single object the specification asks for: volume,
metadata, quality report, orientation, spacing, brain-mask slot, registration slot,
processing history. :class:`FoundationStudy` is the container, because a brain MRI
study is *several* of those — T1, T2, FLAIR, DWI — and a model that fuses sequences
needs them together with the study-level findings that only exist across series.

Why the processing history is not optional
------------------------------------------
Two volumes that look identical can have had completely different things done to them,
and the difference decides whether a model's output means anything. Did bias correction
run, or was it unavailable? Was that mask a brain mask or a head mask? Which frame of
a 4D series is this? Every stage records a :class:`ProcessingStep` with its status, its
parameters, and its duration, so the answer is always in the object rather than in
somebody's memory of how the pipeline was configured that week.

:meth:`ProcessingHistory.was_applied` is the query a downstream module should actually
use — "did N4 run on this volume?" — instead of assuming from configuration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from .masking import BrainMaskSlot
from .metadata import MRIMetadata
from .quality import QualityReport
from .registration import RegistrationPlan
from .sequence import SequenceAssignment
from .types import (
    FOUNDATION_VERSION,
    FileFormat,
    QualityVerdict,
    SequenceType,
    StepStatus,
)
from .volume import MRIVolume


@dataclass(frozen=True)
class ProcessingStep:
    """One stage that ran, or was meant to."""

    name: str
    status: StepStatus
    duration_ms: float = 0.0
    message: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    #: What actually did the work, when it matters (``SimpleITK N4``, ``scipy
    #: affine_transform``). Two deployments differing here can produce different
    #: numbers from the same configuration.
    implementation: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def ran(self) -> bool:
        return self.status in (StepStatus.APPLIED, StepStatus.NO_OP)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 3),
            "message": self.message,
            "parameters": dict(self.parameters),
            "implementation": self.implementation,
            "timestamp": self.timestamp,
        }


@dataclass
class ProcessingHistory:
    """Ordered record of everything that happened to a volume."""

    steps: list[ProcessingStep] = field(default_factory=list)

    def record(self, step: ProcessingStep) -> ProcessingStep:
        self.steps.append(step)
        return step

    def __iter__(self) -> Iterator[ProcessingStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def step(self, name: str) -> ProcessingStep | None:
        return next((s for s in self.steps if s.name == name), None)

    def was_applied(self, name: str) -> bool:
        """Did ``name`` actually transform the volume?

        The query a consumer should use before assuming a property of its input. A
        ``no_op`` counts as applied — the stage ran and correctly found nothing to do,
        so its guarantee holds.
        """
        step = self.step(name)
        return step is not None and step.ran

    @property
    def unavailable_stages(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps if s.status is StepStatus.UNAVAILABLE)

    @property
    def failed_stages(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps if s.status is StepStatus.FAILED)

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_ms": round(self.total_ms, 3),
            "unavailable": list(self.unavailable_stages),
            "failed": list(self.failed_stages),
        }


class StepTimer:
    """Wall-clock timer for one step. Small enough to be obvious, used everywhere."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


@dataclass
class StandardizedSeries:
    """One standardised MR series — the foundation object downstream models consume.

    Everything the specification listed, in one place: the volume, its metadata, its
    quality report, its orientation and spacing (through the volume's geometry), the
    brain-mask slot, the registration slot, and the processing history.
    """

    series_id: str
    volume: MRIVolume
    metadata: MRIMetadata
    sequence: SequenceAssignment
    quality: QualityReport
    brain_mask: BrainMaskSlot = field(default_factory=BrainMaskSlot)
    registration: RegistrationPlan = field(default_factory=RegistrationPlan)
    history: ProcessingHistory = field(default_factory=ProcessingHistory)
    #: Pre-standardisation volume, kept only when the config asks for it.
    source_volume: MRIVolume | None = None

    # -- the convenience accessors the specification named explicitly -------- #
    @property
    def orientation(self) -> str:
        return self.volume.geometry.orientation

    @property
    def spacing(self) -> tuple[float, float, float]:
        return self.volume.spacing

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.volume.shape

    @property
    def affine(self):
        return self.volume.affine

    @property
    def sequence_type(self) -> SequenceType:
        return self.sequence.sequence

    @property
    def usable(self) -> bool:
        """Quality-acceptable and structurally sound. The one-line gate a model uses."""
        return self.quality.acceptable and not self.history.failed_stages

    def to_dict(self) -> dict[str, Any]:
        """Full JSON description. Voxels are excluded — deliberately.

        A foundation study is passed in memory. Serialising the array here would make
        every log line and every API response carry hundreds of megabytes, and would
        put patient image data into places that were only ever meant to hold metadata.
        """
        return {
            "series_id": self.series_id,
            "sequence": self.sequence.model_dump(mode="json"),
            "volume": self.volume.to_dict(),
            "metadata": self.metadata.model_dump(mode="json"),
            "quality": self.quality.model_dump(mode="json"),
            "brain_mask": self.brain_mask.to_dict(),
            "registration": self.registration.to_dict(),
            "history": self.history.to_dict(),
            "usable": self.usable,
        }


@dataclass
class FoundationStudy:
    """A complete standardised MRI study: one or more :class:`StandardizedSeries`.

    This is what :class:`~aura.backend.foundation.mri.pipeline.MRIFoundationPipeline`
    returns and what every future NeuroMind module takes as input.
    """

    study_id: str
    series: tuple[StandardizedSeries, ...]
    source_format: FileFormat = FileFormat.UNKNOWN
    study_instance_uid: str | None = None
    #: Study-level findings from the loader — mixed studies, mixed formats, series
    #: that do not share a frame of reference.
    warnings: tuple[str, ...] = ()
    #: Steps that apply to the study rather than to one series (loading, validation).
    history: ProcessingHistory = field(default_factory=ProcessingHistory)
    #: Series that were read but could not be standardised, with the reason. Present
    #: so a partial success never looks like a complete one.
    rejected_series: tuple[dict[str, Any], ...] = ()
    foundation_version: str = FOUNDATION_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # -- access ------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self) -> Iterator[StandardizedSeries]:
        return iter(self.series)

    def by_sequence(self, sequence: SequenceType) -> tuple[StandardizedSeries, ...]:
        """Every series classified as ``sequence``, best-confidence first.

        The accessor a multi-sequence model uses. Returns a tuple rather than one
        series because a study legitimately holds two FLAIRs (different planes), and
        silently picking one would hide that.
        """
        matches = [s for s in self.series if s.sequence.sequence is sequence]
        return tuple(sorted(matches, key=lambda s: s.sequence.confidence, reverse=True))

    def first(self, sequence: SequenceType) -> StandardizedSeries | None:
        matches = self.by_sequence(sequence)
        return matches[0] if matches else None

    @property
    def primary(self) -> StandardizedSeries | None:
        """The series most likely to be the study's main volume.

        Highest quality among the usable series, with an identified sequence
        preferred. A convenience for single-series studies and for logging — a model
        that needs a specific contrast must ask for it by name.
        """
        usable = [s for s in self.series if s.usable]
        pool = usable or list(self.series)
        if not pool:
            return None
        return max(pool, key=lambda s: (s.sequence.sequence is not SequenceType.UNKNOWN,
                                        s.quality.quality_score))

    @property
    def sequences_present(self) -> tuple[SequenceType, ...]:
        seen: list[SequenceType] = []
        for series in self.series:
            if series.sequence.sequence not in seen:
                seen.append(series.sequence.sequence)
        return tuple(seen)

    @property
    def usable_series(self) -> tuple[StandardizedSeries, ...]:
        return tuple(s for s in self.series if s.usable)

    @property
    def verdict(self) -> QualityVerdict:
        """Study-level quality: the best any single series achieved.

        "Best", not "worst": a study holding a clean T1 and a motion-corrupted DWI is
        not a rejected study — it is a study with one bad series, and the per-series
        reports say which. Taking the worst would discard usable data.
        """
        if not self.series:
            return QualityVerdict.REJECTED
        order = [QualityVerdict.REJECTED, QualityVerdict.ACCEPTABLE_WITH_WARNINGS,
                 QualityVerdict.ACCEPTABLE]
        return max((s.quality.verdict for s in self.series), key=order.index)

    def summary(self) -> str:
        """One line for a log or a report header."""
        sequences = ", ".join(s.value for s in self.sequences_present) or "none"
        return (f"{self.study_id}: {len(self.series)} series [{sequences}], "
                f"quality {self.verdict.value}, format {self.source_format.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "foundation_version": self.foundation_version,
            "created_at": self.created_at,
            "source_format": self.source_format.value,
            "study_instance_uid": self.study_instance_uid,
            "series_count": len(self.series),
            "sequences_present": [s.value for s in self.sequences_present],
            "verdict": self.verdict.value,
            "usable_series": [s.series_id for s in self.usable_series],
            "warnings": list(self.warnings),
            "rejected_series": [dict(r) for r in self.rejected_series],
            "history": self.history.to_dict(),
            "series": [s.to_dict() for s in self.series],
        }


def step(name: str, status: StepStatus, timer: StepTimer | None = None, *,
         message: str = "", implementation: str = "",
         parameters: Sequence[tuple[str, Any]] | dict[str, Any] | None = None
         ) -> ProcessingStep:
    """Construct a :class:`ProcessingStep`, filling the duration from ``timer``."""
    return ProcessingStep(
        name=name,
        status=status,
        duration_ms=timer.elapsed_ms if timer is not None else 0.0,
        message=message,
        parameters=dict(parameters or {}),
        implementation=implementation,
    )


__all__ = [
    "FoundationStudy", "ProcessingHistory", "ProcessingStep", "StandardizedSeries",
    "StepTimer", "step",
]
