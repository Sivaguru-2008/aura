"""AURA NeuroMind — brain MRI engine.

Runs the trained Brain Vision network (``backend/vision/brain``) on a standardised MR
study and reports what it measures. Four stages, matching the engine contract:

    validate_input   -> is this a volumetric MR study this model can read?
    preprocess       -> MRI Foundation Layer: load, identify sequences, QC, standardise
    analyze          -> BrainVisionEngine inference + calibration -> CaseBundle
    generate_report  -> project the grounded draft the bundle already carries

Two refusals are load-bearing, and both were measured rather than assumed.

**A 2D image export is refused.** The network takes four co-registered sequences
(FLAIR, T1, T1ce, T2), all declared required, and was trained with no modality dropout.
A PNG or JPEG is one slice of one sequence, and it does not say which. Measured on
held-out BraTS slices, feeding a single sequence and zeroing the rest gives whole-tumour
Dice of 0.52 (FLAIR), 0.28 (T2), 0.02 (T1) and 0.00 (T1ce) against 0.58 with all four —
and the presence head keeps emitting a confident-looking number regardless (0.42 on
T1-only, 0.32 on T1ce-only). Identifying the sequence from pixels was tried and reaches
85.1% accuracy patient-disjoint, so roughly one PNG in seven would be assigned to a
channel where the model produces nothing. There is no operating point here that is
honest, so the format is refused with the numbers attached.

**An uncalibrated presence probability is refused.** See
:mod:`backend.engines.neuro.calibration`.

The engine that this replaced accepted any PNG and derived its diagnosis from
``int(sha256[:12], 16) % 5``, with a fixed 0.85 posterior, a ``random.random()``
embedding and a rectangle for saliency. No output was a function of the image.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.core.shared.errors import EngineExecutionError, UnreadableImage
from backend.core.shared.logging import get_logger
from backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
from backend.engines.base.contract import (
    AnalysisEngine,
    AnalysisResult,
    EngineDescriptor,
    EngineReport,
    PreparedStudy,
    ValidationOutcome,
)
from backend.engines.base.registry import EngineRegistry, default_registry
from backend.engines.neuro.multisequence import (
    ChannelOrderRejected,
    MultiSequenceStudy,
    load_multisequence,
    looks_multisequence,
)
from schemas.contracts import StructuredPriors

log = get_logger("engine.neuromind")

#: Formats the MRI Foundation Layer can reconstruct a volume from.
_VOLUMETRIC_SUFFIXES = (".nii", ".nii.gz", ".gz", ".nrrd", ".nhdr", ".dcm", ".dicom", "")

#: Measured single-sequence whole-tumour Dice, against 0.58 with all four sequences.
#: Quoted in the refusal so it reads as a finding rather than an unimplemented feature.
_SINGLE_SEQUENCE_DICE = {"FLAIR": 0.52, "T2": 0.28, "T1": 0.02, "T1ce": 0.00}

#: Quality score below which the engine abstains rather than reporting.
_ABSTAIN_QUALITY_FLOOR = 0.50


class NeuroMindEngine(AnalysisEngine):
    """Brain MRI engine backed by the trained BraTS segmentation network."""

    descriptor = EngineDescriptor(
        engine_id="neuromind",
        display_name="AURA NeuroMind",
        version="2.0.0",
        modalities=(ImagingModality.BRAIN_MRI,),
        status=EngineStatus.AVAILABLE,
        capabilities=(
            "study_loading",              # DICOM series / NIfTI-1,2 / NRRD
            "metadata_extraction",
            "sequence_identification",    # T1 / T1ce / T2 / FLAIR / DWI / ADC / SWI
            "quality_assessment",
            "volume_standardisation",     # canonical RAS, isotropic, normalised
            "tumor_segmentation",         # BraTS: necrotic core / oedema / enhancing
            "tumor_presence_detection",   # calibrated whole-tumour probability
            "volumetric_burden",
            "learned_embedding",
            "grounded_report",
            "neuroview_mri_visualization",
        ),
        description=(
            "BraTS2020 segmentation network (residual U-Net, epoch 24, composite Dice "
            "0.875) with a Platt-calibrated whole-tumour presence head. Segments "
            "necrotic core, peritumoural oedema and enhancing tumour, and reports "
            "volumetric burden. Requires a volumetric MR study with all four "
            "sequences; it does not classify tumour subtype."
        ),
    )

    #: Named so the capability listing does not imply these are available.
    PLANNED_CAPABILITIES = (
        "tumor_subtype_classification",
        "intracranial_hemorrhage",
        "midline_shift_quantification",
        "white_matter_burden",
        "longitudinal_comparison",
    )

    def __init__(
        self,
        pipeline=None,
        store=None,
        on_case_created: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._pipeline_handle = pipeline
        self._store = store
        self._on_case_created = on_case_created
        self._foundation: Any | None = None
        self._vision: Any | None = None
        self._calibrator: Any | None = None
        self._qkl: Any | None = None

    # ------------------------------------------------------------------ #
    # Lazily-constructed collaborators
    # ------------------------------------------------------------------ #
    def _qkl_classifier(self):
        from backend.engines.neuro.qkl import QKLClassifier
        if self._qkl is None:
            self._qkl = QKLClassifier.load()
        return self._qkl

    def _foundation_pipeline(self):
        from backend.foundation.mri import FoundationConfig, MRIFoundationPipeline

        if self._foundation is None:
            self._foundation = MRIFoundationPipeline(
                FoundationConfig(reject_on_quality=False))
        return self._foundation

    def _vision_engine(self):
        """The trained network. Constructed once, on first real study."""
        from backend.vision.brain.inference import BrainVisionEngine

        if self._vision is None:
            self._vision = BrainVisionEngine.load()
            log.info("brain vision engine bound",
                     extra={"context": {"epoch": self._vision.meta.epoch,
                                        "monitor": self._vision.meta.monitor,
                                        "value": self._vision.meta.monitor_value}})
        return self._vision

    def _presence_calibrator(self):
        from backend.engines.neuro.calibration import load_calibrator

        if self._calibrator is None:
            self._calibrator = load_calibrator()
        return self._calibrator

    # ------------------------------------------------------------------ #
    # 1. validate_input
    # ------------------------------------------------------------------ #
    def validate_input(self, asset: ImageAsset) -> ValidationOutcome:
        """Accept volumetric MR studies. Refuse everything else, with the reason."""
        from backend.foundation.mri.io.dicom_reader import DicomSeriesReader
        from backend.foundation.mri.io.nifti_reader import NiftiReader
        from backend.foundation.mri.io.nrrd_reader import NrrdReader
        import zipfile

        readers = (DicomSeriesReader(), NiftiReader(), NrrdReader())
        claimed_by = next((r.file_format.value for r in readers
                           if _safe_can_read(r, asset.path)), None)
        
        is_zip = zipfile.is_zipfile(asset.path)
        is_dir = asset.path.is_dir()

        if claimed_by is None and is_zip:
            try:
                with zipfile.ZipFile(asset.path, 'r') as zf:
                    for name in zf.namelist():
                        name_lower = name.lower()
                        if name_lower.endswith(('.dcm', '.dicom')) or 'dicom' in name_lower:
                            claimed_by = "DICOM"
                            break
                        elif name_lower.endswith(('.nii', '.nii.gz')):
                            claimed_by = "NIfTI"
                            break
                        elif name_lower.endswith('.nrrd'):
                            claimed_by = "NRRD"
                            break
            except Exception:
                pass

        if claimed_by is None and is_dir:
            nii_files = [f for f in asset.path.rglob("*.nii*") if f.is_file() and not f.name.startswith(".")]
            if nii_files:
                claimed_by = "NIfTI"

        from backend.core.router.router import ModalityRouter
        from backend.core.shared.types import ImagingModality
        
        router = ModalityRouter()
        metadata = router.route(asset)
        claimed_by_lower = str(claimed_by).lower() if claimed_by else None
        if metadata.modality != ImagingModality.BRAIN_MRI.value and claimed_by_lower not in ("nifti", "nrrd"):
            reason = f"Unsupported modality entered NeuroMind engine: detected {metadata.modality_label} ({metadata.confidence:.2f})"
            if self._store:
                try:
                    self._store.audit(
                        action="modality.rejected",
                        entity_type="upload",
                        entity_id=asset.sha256[:12],
                        detail={
                            "reason": reason,
                            "confidence": metadata.confidence,
                            "triggered_rule": metadata.reason,
                            "detected": metadata.modality,
                            "asset_sha256": asset.sha256,
                        }
                    )
                except Exception:
                    pass
            return ValidationOutcome(
                accepted=False,
                reason=reason,
                checks={
                    "confidence": metadata.confidence,
                    "triggered_rule": metadata.reason,
                    "detected": metadata.modality,
                }
            )

        declared_modality = (asset.dicom_tags.get("modality") or "").upper()
        suffix = Path(asset.original_filename).suffix.lower()
        checks: dict[str, object] = {
            "readable_format": claimed_by or "(none)",
            "dicom_modality": declared_modality or "(none)",
            "file_suffix": suffix or "(none)",
            "supported_formats": [r.file_format.value for r in readers],
        }

        if declared_modality and declared_modality != "MR":
            return ValidationOutcome(
                False,
                f"DICOM header declares modality {declared_modality!r}; NeuroMind "
                "analyses MR studies only",
                checks)

        if claimed_by is not None:
            channels = len(self._sequence_keys())
            if looks_multisequence(asset.path, channels):
                checks["multisequence_channels"] = channels
                return ValidationOutcome(
                    True,
                    f"accepted as a complete {channels}-sequence MR study "
                    f"(4D {claimed_by}); channel order will be verified against the "
                    f"pixels before analysis",
                    checks)
            if is_dir:
                return ValidationOutcome(
                    True, f"accepted as a candidate brain MR study directory ({claimed_by})", checks)
            return ValidationOutcome(
                True, f"accepted as a candidate brain MR study ({claimed_by})", checks)

        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            dice = ", ".join(f"{seq} {d:.2f}" for seq, d in _SINGLE_SEQUENCE_DICE.items())
            return ValidationOutcome(
                False,
                "a 2D image export cannot be analysed by this model. The network reads "
                "four co-registered sequences (FLAIR, T1, T1ce, T2) and was trained "
                "with all four required. A single exported slice carries one sequence "
                "and does not identify which — and single-sequence whole-tumour Dice "
                f"measures {dice} against 0.58 with all four, while the presence head "
                "keeps returning a confident-looking probability either way. Upload "
                "the study as a DICOM series, a NIfTI (.nii/.nii.gz), or an NRRD "
                "volume",
                checks)

        return ValidationOutcome(
            False,
            "the upload is not a volumetric MR study. NeuroMind requires a DICOM "
            "series, a NIfTI (.nii/.nii.gz), or an NRRD volume — no volume can be "
            "reconstructed from this file",
            checks)

    # ------------------------------------------------------------------ #
    # 2. preprocess
    # ------------------------------------------------------------------ #
    def _sequence_keys(self) -> tuple[str, ...]:
        """The network's declared input channels, in order. Read from the checkpoint."""
        try:
            return tuple(spec.key for spec in self._vision_engine().network.modalities)
        except Exception:
            # Falls back to the shipped spec only so validation can answer before the
            # network is constructed; analysis always uses the checkpoint's own order.
            from backend.vision.brain.types import DEFAULT_MODALITIES

            return tuple(spec.key for spec in DEFAULT_MODALITIES)

    def preprocess(self, asset: ImageAsset) -> PreparedStudy:
        """Standardise the study through the MRI Foundation Layer."""
        from backend.foundation.mri.intake_manager import MRIIntakeManager
        from backend.foundation.mri.errors import StudyValidationError
        from backend.engines.neuro.multisequence import looks_multisequence
        import zipfile
        import shutil
        import tempfile

        # Check format/nature of the upload
        is_zip = zipfile.is_zipfile(asset.path)
        is_dir = asset.path.is_dir()
        
        # Determine if we should treat it as a multi-sequence study upload
        is_multisequence = False
        
        if is_zip:
            temp_inspect = Path(tempfile.mkdtemp(prefix="aura-inspect-"))
            try:
                with zipfile.ZipFile(asset.path, 'r') as zf:
                    zf.extractall(temp_inspect)
                nii_files = [f for f in temp_inspect.rglob("*.nii*") if f.is_file() and not f.name.startswith(".")]
                if len(nii_files) > 1 or (len(nii_files) == 1 and looks_multisequence(nii_files[0], 4)):
                    is_multisequence = True
            except Exception:
                pass
            finally:
                shutil.rmtree(temp_inspect)
        elif is_dir:
            nii_files = [f for f in asset.path.rglob("*.nii*") if f.is_file() and not f.name.startswith(".")]
            if len(nii_files) > 1 or (len(nii_files) == 1 and looks_multisequence(nii_files[0], 4)):
                is_multisequence = True
        else:
            # Single file
            if looks_multisequence(asset.path, 4):
                is_multisequence = True

        if is_multisequence:
            manager = MRIIntakeManager()
            try:
                study = manager.process(asset.path)
            except StudyValidationError:
                raise
            except Exception as exc:
                raise StudyValidationError(
                    f"Validation failed for MR study: {exc}",
                    detail={"filename": asset.original_filename}
                ) from exc

            index = self._store.count() + 1 if self._store else 1
            is_4d = "4D" in study.order_source or study.order_source == "load_multisequence"
            source_format = "nifti-4d" if is_4d else "folder/zip stacked"
            
            return PreparedStudy(
                study_id=f"STU-MR-{asset.sha256[:12]}",
                modality=ImagingModality.BRAIN_MRI,
                payload=study,
                metadata={
                    "case_id": f"CASE-MR-{index}",
                    "sha256": asset.sha256,
                    "foundation": {
                        "source_format": source_format,
                        "series_count": len(study.sequence_keys),
                        "sequences_identified": list(study.sequence_keys),
                        "spacing_mm": list(study.spacing_mm or ()),
                        "channel_order_source": study.order_source,
                        "channel_order_check": study.order_endorsement,
                    }
                },
            )
        
        # Fallback to single-series foundation pipeline (keeps existing tests passing!)
        temp_dir_path = None
        study_path = asset.path

        if is_zip:
            temp_dir_path = Path(tempfile.mkdtemp(prefix="aura-unzipped-", dir=str(Path.cwd())))
            with zipfile.ZipFile(asset.path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir_path)
            study_path = temp_dir_path

        try:
            from backend.foundation.mri.errors import MRIFoundationError
            from backend.core.shared.errors import UnreadableImage
            try:
                study = self._foundation_pipeline().run(
                    study_path,
                    study_id=f"STU-MR-{asset.sha256[:12]}",
                    display_name=asset.original_filename)
            except MRIFoundationError as exc:
                log.warning(
                    "the MRI foundation layer could not standardise this upload",
                    extra={"context": {"code": exc.code, "reason": exc.reason,
                                       "sha256": asset.sha256[:12]}},
                )
                raise UnreadableImage(
                    f"the MR study could not be prepared for analysis: {exc.reason}",
                    detail={"filename": asset.original_filename,
                            "foundation_error": exc.code, **exc.detail},
                ) from exc

            index = self._store.count() + 1 if self._store else 1
            return PreparedStudy(
                study_id=study.study_id,
                modality=ImagingModality.BRAIN_MRI,
                payload=study,
                metadata={"case_id": f"CASE-MR-{index}",
                          "sha256": asset.sha256,
                          "foundation": self._describe(study, asset)},
            )
        finally:
            if temp_dir_path and temp_dir_path.exists():
                try:
                    shutil.rmtree(temp_dir_path)
                except Exception:
                    pass

    @staticmethod
    def _describe(study, asset: ImageAsset) -> dict[str, Any]:
        return {
            "sha256": asset.sha256,
            "foundation_version": study.foundation_version,
            "source_format": study.source_format.value,
            "series_count": len(study),
            "sequences_identified": [s.value for s in study.sequences_present],
            "quality_verdict": study.verdict.value,
            "study_warnings": list(study.warnings),
            "rejected_series": [dict(r) for r in study.rejected_series],
            "series": [
                {
                    "series_id": series.series_id,
                    "sequence": series.sequence.sequence.value,
                    "sequence_confidence": series.sequence.confidence,
                    "sequence_requires_review": series.sequence.requires_review,
                    "shape": list(series.shape),
                    "spacing_mm": [round(v, 4) for v in series.spacing],
                    "orientation": series.orientation,
                    "quality_score": series.quality.quality_score,
                    "quality_verdict": series.quality.verdict.value,
                    "quality_warnings": list(series.quality.warnings),
                    "recommendations": list(series.quality.recommendations),
                    "brain_mask": series.brain_mask.to_dict(),
                    "registration": series.registration.to_dict(),
                    "processing_history": series.history.to_dict(),
                    "usable": series.usable,
                }
                for series in study.series
            ],
        }

    # ------------------------------------------------------------------ #
    # 3. analyze
    # ------------------------------------------------------------------ #
    async def analyze(self, prepared: PreparedStudy) -> AnalysisResult:
        """Run the trained network, calibrate, and assemble the case bundle."""
        from backend.engines.neuro.bundle import build_case_bundle
        from backend.foundation.mri.errors import StudyValidationError
        from backend.foundation.mri.types import SequenceType

        case_id = prepared.metadata["case_id"]
        study = prepared.payload

        # Validate sequence presence and duplicates for directory uploads
        if not isinstance(study, MultiSequenceStudy):
            present_seqs = {}
            for s in study.series:
                seq = s.sequence.sequence
                if seq != SequenceType.UNKNOWN:
                    present_seqs.setdefault(seq, []).append(s.series_id)

            required = [SequenceType.FLAIR, SequenceType.T1, SequenceType.T1CE, SequenceType.T2]
            missing = [req for req in required if req not in present_seqs]
            if missing:
                raise StudyValidationError(
                    f"Incomplete study: missing required sequences (missing: {', '.join(m.value.upper() for m in missing)}).",
                    detail={
                        "missing_sequences": [m.value for m in missing],
                        "present_sequences": [s.value for s in study.sequences_present]
                    }
                )

            duplicates = [seq for seq, series_list in present_seqs.items() if len(series_list) > 1 and seq in required]
            if duplicates:
                raise StudyValidationError(
                    f"Duplicate sequence detected: sequence {duplicates[0].value.upper()} is present in multiple series ({', '.join(present_seqs[duplicates[0]])}).",
                    detail={
                        "duplicate_sequences": [d.value for d in duplicates],
                        "sequence_mapping": {k.value: v for k, v in present_seqs.items()}
                    }
                )

        started = time.perf_counter()

        output = self._run_vision(study, prepared.study_id)
        calibrator = self._presence_calibrator()

        raw = output.tumor_probability
        if raw is None:
            raise EngineExecutionError(
                "the loaded checkpoint has no presence head, so no tumour probability "
                "can be reported",
                detail={"case_id": case_id, "model": output.model_version})
        presence = calibrator(raw)

        from common.config import get_settings
        qkl_res = None
        if get_settings().neuro_qkl_enabled and getattr(output, "features", None) is not None and getattr(output.features, "pooled", None) is not None:
            try:
                classifier = self._qkl_classifier()
                qkl_res = classifier.predict_subtype(output.features.pooled)
            except Exception as e:
                log.warning(f"QKL subtype classification failed: {e}")

        caveats, abstain_reason, abstain_detail = self._disclosures(output, calibrator)
        bundle = build_case_bundle(
            output,
            case_id=case_id,
            study_id=prepared.study_id,
            image=self._display_slice(output, study),
            presence_probability=presence,
            priors=StructuredPriors(),
            caveats=caveats,
            abstain_reason=abstain_reason,
            abstain_detail=abstain_detail,
            clinical_context=prepared.metadata.get("clinical_context"),
            tumor_subtypes=qkl_res,
        )
        inference_seconds = time.perf_counter() - started

        if self._store:
            self._store.save_case(bundle)
            self._store.save_neuroview(
                case_id,
                prepared.study_id,
                self._build_neuroview(case_id, study, output),
            )
            if self._on_case_created is not None:
                try:
                    self._on_case_created(case_id)
                except Exception:
                    self.log.exception("case-created hook failed; case is saved regardless")
            self._audit(case_id, bundle)

        meta_dict = {
            "inference_time_s": round(inference_seconds, 4),
            "top_diagnosis": bundle.safety.top.value,
            "top_probability": round(float(bundle.safety.top_probability), 4),
            "abstained": bool(bundle.safety.abstained),
            "fusion_backend": bundle.fusion.backend,
            "presence_probability_raw": round(float(raw), 4),
            "presence_probability_calibrated": round(float(presence), 4),
            "sequences_used": list(getattr(output.processing, "sequences_used", ()) or ()),
            "sequences_missing": list(getattr(output.processing, "sequences_missing", ()) or ()),
            "model_version": output.model_version,
            "state": bundle.state.value,
        }
        if qkl_res:
            meta_dict.update({
                "tumor_subtype_glioma": round(qkl_res["glioma"], 4),
                "tumor_subtype_meningioma": round(qkl_res["meningioma"], 4),
                "tumor_subtype_metastasis": round(qkl_res["metastasis"], 4),
                "tumor_subtype_qkl_enabled": True,
            })

        return AnalysisResult(
            study_id=prepared.study_id,
            payload=bundle,
            case_id=case_id,
            metadata=meta_dict,
        )

    def _build_neuroview(self, case_id: str, study, output) -> dict[str, Any]:
        """Create the MRI-only NeuroView artifact from NeuroMind inputs."""
        from backend.engines.neuro.neuroview import build_neuroview_payload

        engine = self._vision_engine()
        return build_neuroview_payload(
            case_id=case_id,
            study=study,
            output=output,
            modalities=tuple(engine.network.modalities),
            min_brain_fraction=float(engine.config.ingest.min_brain_fraction),
        )

    def _run_vision(self, study, study_id: str):
        """Dispatch to the right inference entry point for this study's shape."""
        engine = self._vision_engine()
        if not isinstance(study, MultiSequenceStudy):
            return engine.analyze_study(study)

        # Score the same slices the ingest would have kept: those with brain in them.
        volumes = study.volumes
        brain = np.any(volumes > 0, axis=0)
        per_slice = brain.reshape(-1, brain.shape[2]).mean(axis=0)
        keep = np.flatnonzero(per_slice >= engine.config.ingest.min_brain_fraction)
        if keep.size == 0:
            keep = np.arange(volumes.shape[3])
        slices = [np.ascontiguousarray(volumes[..., int(z)]) for z in keep]
        return engine.analyze_slices(
            slices, study_id=study_id,
            sequences_used=list(study.sequence_keys), sequences_missing=[],
            spacing_mm=study.spacing_mm)

    def _disclosures(self, output, calibrator):
        """Caveats that must travel with the result, and any reason to abstain.

        Abstention here is not a low-confidence rule — the presence head is calibrated,
        so a mid-range probability is a legitimate answer and not a failure. It fires
        when a *condition of validity* is broken: sequences the model requires are
        absent, or measured quality is below the floor. In both cases the number would
        be precise and meaningless, which is worse than a wide one.

        Returns:
            ``(caveats, abstention_reason, abstention_detail)``.
        """
        from schemas.contracts import AbstentionReason
        from common.config import get_settings
        import math

        settings = get_settings()

        caveats: list[str] = [calibrator.summary + "."]
        reason: AbstentionReason | None = None
        detail: str | None = None

        missing = list(getattr(output.processing, "sequences_missing", ()) or ())
        if missing:
            dice = ", ".join(f"{s} {d:.2f}" for s, d in _SINGLE_SEQUENCE_DICE.items())
            caveats.append(
                f"Sequences missing from this study: {', '.join(missing)}. The network "
                f"requires all four; measured single-sequence whole-tumour Dice is "
                f"{dice} against 0.58 complete.")
            reason = AbstentionReason.INCOMPLETE_STUDY
            detail = (f"the study is missing required sequences "
                      f"({', '.join(missing)}); the calibration and the reported Dice "
                      f"both assume a complete four-sequence study")

        quality = getattr(output.quality, "score", None)
        if quality is not None and float(quality) < _ABSTAIN_QUALITY_FLOOR:
            caveats.append(f"Measured study quality {float(quality):.2f} is below the "
                           f"{_ABSTAIN_QUALITY_FLOOR:.2f} floor.")
            if reason is None:
                reason = AbstentionReason.LOW_QUALITY
                detail = (f"measured study quality {float(quality):.2f} is below the "
                          f"{_ABSTAIN_QUALITY_FLOOR:.2f} floor for a reportable result")

        # OOD Check
        raw_prob = output.tumor_probability
        if raw_prob is not None and 0.0 < raw_prob < 1.0:
            raw_logit = math.log(raw_prob / (1.0 - raw_prob))
            energy = -math.log(math.exp(raw_logit) + 1.0)
            z = (energy - (-2.0)) / 1.0
            if z > settings.ood_energy_threshold:
                caveats.append(f"Out of distribution study detected (OOD energy z-score {z:.2f} exceeds threshold {settings.ood_energy_threshold:.2f}).")
                if reason is None:
                    reason = AbstentionReason.OUT_OF_DISTRIBUTION
                    detail = f"the study is out of distribution (OOD z-score: {z:.2f})"

        # Low Confidence (Uncertainty) check on presence probability
        calibrated_prob = calibrator(raw_prob) if raw_prob is not None else 0.5
        top_prob = max(calibrated_prob, 1.0 - calibrated_prob)
        confidence_threshold = max(0.70, 1.0 - settings.low_confidence_threshold)
        if top_prob < confidence_threshold:
            caveats.append(f"Calibrated tumor presence probability {calibrated_prob:.2f} is in the low-confidence range.")
            if reason is None:
                reason = AbstentionReason.LOW_CONFIDENCE
                detail = f"tumor presence probability {calibrated_prob:.2f} is in the low-confidence range (threshold: {confidence_threshold:.2f})"

        for caveat in getattr(output, "caveats", ()) or ():
            caveats.append(str(caveat))
        return caveats, reason, detail

    @staticmethod
    def _display_slice(output, study) -> np.ndarray:
        """The slice the console shows, taken from the same volume the model scored.

        Falls back to a mid-volume slice only when the segmentation is empty, so the
        image and the saliency overlay always describe the same plane. For a complete
        study the first declared sequence is shown, because the overlay is drawn from a
        segmentation the network produced from all of them and any single channel is an
        equally arbitrary backdrop.
        """
        from backend.engines.neuro.bundle import _representative_index

        try:
            if isinstance(study, MultiSequenceStudy):
                volume = study.volumes[0]
            else:
                volume = study.series[0].volume.array
        except Exception:
            log.warning("no display volume available; console image will be blank")
            return np.zeros((224, 224), dtype=np.float32)

        index = _representative_index(output)
        depth = volume.shape[2] if volume.ndim == 3 else 1
        index = min(index, max(0, depth - 1))
        plane = volume[:, :, index] if volume.ndim == 3 else volume
        plane = np.asarray(plane, dtype=np.float32)
        lo, hi = float(plane.min()), float(plane.max())
        return (plane - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(plane)

    def _audit(self, case_id: str, bundle) -> None:
        try:
            self._store.audit(
                "case.uploaded", "case", case_id,
                detail={
                    "top": bundle.safety.top.value,
                    "abstained": bool(bundle.safety.abstained),
                    "engine": self.descriptor.engine_id,
                    "via": "modality_router",
                },
            )
        except Exception:
            self.log.exception(f"FAILED to write audit row for {case_id}")

    # ------------------------------------------------------------------ #
    # 4. generate_report
    # ------------------------------------------------------------------ #
    def generate_report(self, result: AnalysisResult) -> EngineReport:
        bundle = result.payload
        draft = getattr(bundle, "report", None)
        if draft is None:
            raise EngineExecutionError(
                "analysis completed but produced no report",
                detail={"case_id": result.case_id},
            )

        sections = {
            "findings": draft.findings_text,
            "impression": draft.impression_text,
            "differential": draft.differential_text,
            "confidence": draft.confidence_text,
            "recommendations": draft.recommendation_text,
        }
        return EngineReport(
            sections={k: v for k, v in sections.items() if v},
            summary=draft.impression_text or "Brain MR study analysed.",
            grounding=dict(draft.grounding or {}),
        )


def _safe_can_read(reader, path) -> bool:
    from pathlib import Path
    try:
        return bool(reader.can_read(Path(path)))
    except Exception:
        log.debug("a foundation reader raised while probing an upload",
                  extra={"context": {"reader": type(reader).__name__}})
        return False


def register_neuromind_engine(
    pipeline,
    store,
    on_case_created: Callable[[str], None] | None = None,
    registry: EngineRegistry | None = None,
) -> None:
    """Bind the NeuroMind engine to live gateway handles and register it."""
    (registry or default_registry).register(
        NeuroMindEngine.descriptor,
        lambda: NeuroMindEngine(pipeline, store, on_case_created),
        replace=True,
    )
