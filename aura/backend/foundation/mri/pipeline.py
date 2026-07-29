"""The MRI Foundation Pipeline — the one entry point downstream code calls.

    study path -> FoundationStudy

Everything else in this package is a component this class composes. Every component
is injected and every one has a default, so the common case is one line and the
research case is a constructor argument::

    study = MRIFoundationPipeline().run("/data/BRATS/patient-004")
    study = MRIFoundationPipeline(sequence_detector=LearnedDetector()).run(path)

Order, and why quality control sits where it does
-------------------------------------------------
::

    load -> validate -> per series:
        build volume -> extract metadata -> identify sequence
                     -> QUALITY CONTROL           <- on the *source* volume
                     -> standardisation stages
                     -> registration preparation

Quality is measured before standardisation, not after. Resampling, normalisation, and
cropping change every intensity statistic the inspector looks at: a volume z-scored
over its mask has an SNR that describes the normalisation, not the acquisition, and a
cropped volume has no air left to estimate noise from. Inspecting afterwards would
produce a report about the pipeline instead of about the scan.

Failure policy
--------------
Per series, so one bad series never costs a study:

* A stage with no backend (``StageUnavailable``) is recorded as ``unavailable`` and
  the pipeline continues — unless ``strict``. An un-bias-corrected volume that says so
  is useful; no volume at all is not.
* A stage that runs and fails (``StageFailed``) is fatal *for that series*. Continuing
  would leave a volume whose affine no longer describes its voxels.
* A series that cannot be built or is rejected by quality control (when configured to
  reject) is moved to ``rejected_series`` with its reason.
* Only when *every* series fails does the study raise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from aura.backend.core.shared.logging import get_logger
from .config import FoundationConfig
from .errors import (
    MRIFoundationError,
    StageFailed,
    StageUnavailable,
    StudyRejected,
    StudyValidationError,
)
from .io.base import RawSeries
from .loader import LoadedStudy, MRIStudyLoader
from .metadata import MetadataExtractor, MRIMetadata
from .quality import MRIQualityInspector, QualityReport
from .registration import RegistrationPreparer
from .sequence import RuleBasedSequenceDetector, SequenceDetector
from .standardize import (
    StandardizationContext,
    VolumeTransform,
    default_pipeline_stages,
)
from .study import (
    FoundationStudy,
    ProcessingHistory,
    StandardizedSeries,
    StepTimer,
    step,
)
from .types import FileFormat, QualityVerdict, StepStatus
from .volume import MRIVolume, VolumeBuilder

log = get_logger("foundation.mri.pipeline")


class MRIFoundationPipeline:
    """Raw MRI study to standardised, quality-assessed, model-ready output."""

    def __init__(
        self,
        config: FoundationConfig | None = None,
        *,
        loader: MRIStudyLoader | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        sequence_detector: SequenceDetector | None = None,
        volume_builder: VolumeBuilder | None = None,
        quality_inspector: MRIQualityInspector | None = None,
        stages: Sequence[VolumeTransform] | None = None,
        registration_preparer: RegistrationPreparer | None = None,
    ) -> None:
        self.config = config or FoundationConfig()
        self.loader = loader or MRIStudyLoader(config=self.config.loader)
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.sequence_detector = sequence_detector or RuleBasedSequenceDetector()
        self.volume_builder = volume_builder or VolumeBuilder(
            min_slices=self.config.loader.min_slices_for_volume)
        self.quality_inspector = quality_inspector or MRIQualityInspector(
            self.config.quality)
        self.registration_preparer = registration_preparer or RegistrationPreparer()
        self.stages: tuple[VolumeTransform, ...] = tuple(
            stages if stages is not None
            else default_pipeline_stages(self.config.standardization))

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    def run(self, source: Path | str, *, study_id: str | None = None,
            display_name: str | None = None, frame: int = 0) -> FoundationStudy:
        """Process a complete study.

        Args:
            source: study directory, or a single volume file.
            study_id: identifier for the result. Defaults to the path's name.
            display_name: the name the study arrived under, when ``source`` is a
                staged copy. A staged upload has a randomly generated filename, and a
                random name is not evidence — ``aura-intake-kt2ln441.nii`` contains
                the token ``T2``. Passing the original name keeps the sequence
                detector's weakest channel fed with something a human actually wrote.
            frame: which volume to take from any 4D series.

        Raises:
            MRIFoundationError: the study could not be loaded, or no series survived.
        """
        history = ProcessingHistory()
        timer = StepTimer()
        loaded = self.loader.load(source, study_id=study_id)
        history.record(step(
            "study_load", StepStatus.APPLIED, timer,
            message=(f"read {loaded.series_count} series from "
                     f"{loaded.files_examined} file(s) as {loaded.source_format.value}"),
            implementation=type(self.loader).__name__,
            parameters={"files_examined": loaded.files_examined,
                        "series_found": loaded.series_count,
                        "complete": loaded.complete}))

        validation = StepTimer()
        history.record(step(
            "study_validation", StepStatus.APPLIED, validation,
            message=("; ".join(loaded.warnings) if loaded.warnings
                     else "no study-level inconsistencies found"),
            parameters={"warnings": len(loaded.warnings),
                        "study_uids": list(loaded.study_instance_uids)}))

        standardised: list[StandardizedSeries] = []
        # Series the reader could not decode start the rejected list, so the study's
        # own record covers everything that was found — not only what got as far as
        # standardisation.
        rejected: list[dict[str, Any]] = [dict(s) for s in loaded.skipped_series]
        for raw in loaded.series:
            try:
                series = self.run_series(raw, loaded=loaded, frame=frame,
                                         display_name=display_name)
            except MRIFoundationError as exc:
                log.warning("series could not be standardised",
                            extra={"context": {"series": raw.series_key,
                                               "code": exc.code,
                                               "reason": exc.reason}})
                rejected.append({"series_key": raw.series_key,
                                 "source_name": raw.source_name,
                                 "error": exc.code, "reason": exc.reason,
                                 **({"detail": exc.detail} if exc.detail else {})})
                continue
            if (self.config.reject_on_quality
                    and series.quality.verdict is QualityVerdict.REJECTED):
                rejected.append({"series_key": raw.series_key,
                                 "source_name": raw.source_name,
                                 "error": "study_rejected_by_quality_control",
                                 "reason": series.quality.reject_reason or "",
                                 "quality": series.quality.model_dump(mode="json")})
                continue
            standardised.append(series)

        if not standardised:
            self._raise_all_failed(loaded, rejected)

        study = FoundationStudy(
            study_id=loaded.study_id,
            series=tuple(standardised),
            source_format=loaded.source_format,
            study_instance_uid=(loaded.study_instance_uids[0]
                                if loaded.study_instance_uids else None),
            warnings=loaded.warnings,
            history=history,
            rejected_series=tuple(rejected),
        )
        log.info("foundation study ready", extra={"context": {
            "study_id": study.study_id, "series": len(study.series),
            "rejected": len(rejected), "verdict": study.verdict.value,
            "sequences": [s.value for s in study.sequences_present]}})
        return study

    def run_series(self, raw: RawSeries, *, loaded: LoadedStudy | None = None,
                   frame: int = 0,
                   display_name: str | None = None) -> StandardizedSeries:
        """Process one series end to end. Public so a caller can drive series alone."""
        history = ProcessingHistory()

        volume, metadata = self._build(raw, history, frame, display_name)
        sequence = self._identify_sequence(metadata, history)
        quality = self._inspect(volume, metadata, raw, history)

        source_volume = volume if self.config.retain_source_volume else None
        context = StandardizationContext(config=self.config.standardization,
                                         metadata=metadata)
        volume = self._standardize(volume, context, history)

        registration = self._prepare_registration(volume, context, raw, loaded, history)

        for note in context.notes:
            history.record(step("standardisation_note", StepStatus.APPLIED,
                                message=note))

        return StandardizedSeries(
            series_id=raw.series_key,
            volume=volume,
            metadata=metadata,
            sequence=sequence,
            quality=quality,
            brain_mask=context.mask,
            registration=registration,
            history=history,
            source_volume=source_volume,
        )

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def _build(self, raw: RawSeries, history: ProcessingHistory, frame: int,
               display_name: str | None = None) -> tuple[MRIVolume, MRIMetadata]:
        timer = StepTimer()
        volume, warnings = self.volume_builder.build(raw, frame=frame)
        history.record(step(
            "volume_reconstruction", StepStatus.APPLIED, timer,
            message="; ".join(warnings) if warnings else
                    f"assembled a {volume.shape} volume at "
                    f"{tuple(round(v, 3) for v in volume.spacing)} mm",
            implementation=type(self.volume_builder).__name__,
            parameters={"shape": list(volume.shape), "frames": raw.frames,
                        "frame_selected": frame if raw.is_multiframe else None,
                        "integrity": raw.integrity.to_dict()}))

        # A DICOM series' source name is its SeriesDescription — real protocol text
        # written by a human, and worth keeping. A NIfTI's or NRRD's is only a
        # filename, so an override supplied by the caller is strictly better evidence.
        source_name = raw.source_name
        if display_name and raw.source_format in (FileFormat.NIFTI, FileFormat.NRRD):
            source_name = display_name

        meta_timer = StepTimer()
        metadata = self.metadata_extractor.extract(
            raw.header, source_format=raw.source_format, geometry=volume.geometry,
            slice_count=int(volume.shape[2]), source_name=source_name)
        history.record(step(
            "metadata_extraction", StepStatus.APPLIED, meta_timer,
            message=metadata.describe(),
            implementation=type(self.metadata_extractor).__name__,
            parameters={"warnings": list(metadata.extraction_warnings),
                        "acquisition_parameters_available":
                            metadata.has_acquisition_parameters}))
        return volume, metadata

    def _identify_sequence(self, metadata: MRIMetadata,
                           history: ProcessingHistory):
        timer = StepTimer()
        assignment = self.sequence_detector.detect(metadata)
        history.record(step(
            "sequence_identification", StepStatus.APPLIED, timer,
            message=f"{assignment.label} ({assignment.confidence:.2f}, "
                    f"{assignment.source})",
            implementation=type(self.sequence_detector).__name__,
            parameters={"sequence": assignment.sequence.value,
                        "confidence": assignment.confidence,
                        "metadata_available": assignment.metadata_available,
                        "requires_review": assignment.requires_review}))
        return assignment

    def _inspect(self, volume: MRIVolume, metadata: MRIMetadata, raw: RawSeries,
                 history: ProcessingHistory) -> QualityReport:
        timer = StepTimer()
        report = self.quality_inspector.inspect(volume, metadata=metadata,
                                                integrity=raw.integrity)
        history.record(step(
            "quality_assessment", StepStatus.APPLIED, timer,
            message=report.summary(),
            implementation=type(self.quality_inspector).__name__,
            parameters={"quality_score": report.quality_score,
                        "verdict": report.verdict.value,
                        "reject_reason": report.reject_reason,
                        "assessed_on": "source volume, before standardisation"}))
        return report

    def _standardize(self, volume: MRIVolume, context: StandardizationContext,
                     history: ProcessingHistory) -> MRIVolume:
        for stage in self.stages:
            timer = StepTimer()
            name = getattr(stage, "name", type(stage).__name__)
            try:
                result = stage.apply(volume, context)
            except StageUnavailable as exc:
                history.record(step(
                    name, StepStatus.UNAVAILABLE, timer, message=exc.reason,
                    implementation=type(stage).__name__, parameters=exc.detail))
                if self.config.standardization.strict:
                    raise
                log.info("standardisation stage unavailable; continuing",
                         extra={"context": {"stage": name}})
                continue
            except StageFailed as exc:
                history.record(step(name, StepStatus.FAILED, timer,
                                    message=exc.reason,
                                    implementation=type(stage).__name__,
                                    parameters=exc.detail))
                raise
            except Exception as exc:
                history.record(step(name, StepStatus.FAILED, timer,
                                    message=f"unexpected failure: {type(exc).__name__}",
                                    implementation=type(stage).__name__))
                log.exception("standardisation stage raised unexpectedly",
                              extra={"context": {"stage": name}})
                raise StageFailed(name, "the stage failed unexpectedly") from exc

            history.record(step(
                name, StepStatus.APPLIED if result.changed else StepStatus.NO_OP,
                timer, message=result.message, implementation=type(stage).__name__,
                parameters=result.parameters))
            volume = result.volume
        return volume

    def _prepare_registration(self, volume: MRIVolume,
                              context: StandardizationContext, raw: RawSeries,
                              loaded: LoadedStudy | None,
                              history: ProcessingHistory):
        timer = StepTimer()
        frame_uid = raw.header.get("FrameOfReferenceUID")
        aligned = ()
        if loaded is not None and frame_uid:
            aligned = tuple(
                other.series_key for other in loaded.series
                if other.series_key != raw.series_key
                and other.header.get("FrameOfReferenceUID") == frame_uid)

        plan = self.registration_preparer.prepare(
            volume, mask=context.mask,
            frame_of_reference_uid=str(frame_uid) if frame_uid else None,
            aligned_series=aligned,
            target_spacing_mm=self.config.standardization.target_spacing_mm)
        history.record(step(
            "registration_preparation", StepStatus.APPLIED, timer,
            message=("registration inputs computed; no transform was applied "
                     "(registration is out of scope for the foundation layer)"),
            implementation=type(self.registration_preparer).__name__,
            parameters={"aligned_series": list(aligned),
                        "transform": None,
                        "notes": list(plan.notes)}))
        return plan

    # ------------------------------------------------------------------ #
    @staticmethod
    def _raise_all_failed(loaded: LoadedStudy,
                          rejected: list[dict[str, Any]]) -> None:
        """Every series failed. Raise the most specific error available."""
        codes = {r.get("error") for r in rejected}
        detail = {"series_attempted": loaded.series_count,
                  "rejected": rejected[:10]}
        if codes == {"study_rejected_by_quality_control"}:
            raise StudyRejected(
                "every series in this study was rejected by quality control",
                detail=detail)
        raise StudyValidationError(
            "no series in this study could be standardised", detail=detail)


def build_foundation_pipeline(config: FoundationConfig | None = None
                              ) -> MRIFoundationPipeline:
    """Default pipeline factory. The seam a service layer wires through DI."""
    return MRIFoundationPipeline(config=config)


__all__ = ["MRIFoundationPipeline", "build_foundation_pipeline"]
