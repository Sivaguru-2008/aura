"""Error taxonomy for the MRI Foundation Layer.

Every failure below extends :class:`~aura.backend.core.shared.errors.AuraBackendError`,
so the API layer renders foundation failures exactly like routing failures — one
error shape for the whole backend, no handler inventing its own.

The split follows *who has to act*:

* :class:`StudyNotFound`, :class:`UnsupportedStudyFormat`, :class:`CorruptStudy` —
  the caller sent something the layer cannot work with. Fixable by re-uploading.
* :class:`StudyValidationError` — the study is readable but internally inconsistent
  (mixed geometry, no complete series). Fixable at the scanner or by re-exporting.
* :class:`StudyRejected` — the study loaded and was measured, and quality control
  refused it. Carries the :class:`~aura.backend.foundation.mri.quality.QualityReport`
  in ``detail`` so the refusal is always accompanied by the numbers behind it.
* :class:`StageUnavailable` — a standardisation stage is declared but has no working
  backend in this deployment (N4 without SimpleITK, skull stripping without a
  stripper). **Not fatal by default**: the pipeline records it in the processing
  history and continues, because a study that is un-bias-corrected and honest about
  it is far more useful than no study at all. ``strict=True`` turns it fatal.
* :class:`StageFailed` — a stage that *was* available ran and broke. Always fatal for
  that series: silently continuing would leave a half-transformed volume whose
  affine no longer describes its voxels.

``detail`` is client-safe structured context only. Filesystem paths are logged, never
returned — a temp path leaks deployment layout, and a study path can identify a
patient's export directory.
"""
from __future__ import annotations

from typing import Any

from aura.backend.core.shared.errors import AuraBackendError


class MRIFoundationError(AuraBackendError):
    """Base class for every MRI Foundation Layer failure."""

    code = "mri_foundation_error"
    http_status = 500


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class StudyNotFound(MRIFoundationError):
    """The requested study path does not exist or holds no readable files."""

    code = "study_not_found"
    http_status = 404


class UnsupportedStudyFormat(MRIFoundationError):
    """Nothing in the study matched a registered reader.

    Names what *was* found so the caller can tell "wrong format" from "empty
    directory" without a second request.
    """

    code = "unsupported_study_format"
    http_status = 415


class CorruptStudy(MRIFoundationError):
    """Files were found and claimed a known format, but could not be decoded.

    Distinct from :class:`UnsupportedStudyFormat`: the header parsed far enough to
    identify the format, then the content contradicted it — truncated pixel data, a
    declared size that does not match the payload, an unreadable transfer syntax.
    """

    code = "corrupt_study"
    http_status = 422


class StudyValidationError(MRIFoundationError):
    """The study decoded but is not internally consistent enough to build a volume.

    Mixed image dimensions within a series, contradictory orientations, or fewer
    slices than a volume requires. This is a study-construction failure, not a
    quality judgement — quality control never runs on a volume that cannot be built.
    """

    code = "study_validation_failed"
    http_status = 422


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
class StudyRejected(MRIFoundationError):
    """Quality control refused the study.

    Raised only when the pipeline is configured to reject (``reject_on_quality``);
    the default is to attach the report and let the caller decide, because "usable
    for triage, not for volumetry" is a real and common verdict that a hard reject
    cannot express.
    """

    code = "study_rejected_by_quality_control"
    http_status = 422

    def __init__(self, reason: str, *, quality_report: Any = None,
                 detail: dict[str, Any] | None = None) -> None:
        payload = dict(detail or {})
        if quality_report is not None:
            payload["quality"] = (
                quality_report.model_dump()
                if hasattr(quality_report, "model_dump") else quality_report
            )
        super().__init__(reason, detail=payload)
        self.quality_report = quality_report


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
class StageUnavailable(MRIFoundationError):
    """A standardisation stage has no working backend in this deployment.

    Raised by the stage's constructor or by :meth:`apply`. The pipeline catches it,
    records ``status=unavailable`` in the processing history with the reason, and
    continues — unless the configuration is strict.
    """

    code = "standardisation_stage_unavailable"
    http_status = 501

    def __init__(self, stage: str, reason: str, *,
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason, detail={"stage": stage, **(detail or {})})
        self.stage = stage


class StageFailed(MRIFoundationError):
    """An available standardisation stage ran and failed."""

    code = "standardisation_stage_failed"
    http_status = 500

    def __init__(self, stage: str, reason: str, *,
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason, detail={"stage": stage, **(detail or {})})
        self.stage = stage
