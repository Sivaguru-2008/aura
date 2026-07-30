"""Neuro adapter — brain MRI pipeline behind the ModalityAdapter interface.

Wraps the MRI Foundation Layer standardization and the NeuroMindEngine
inference into the three-phase adapter contract.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aura.gateway.adapters.base import (
    EngineOutput,
    InspectionResult,
    ModalityAdapter,
    StandardizedAsset,
)


class NeuroAdapter(ModalityAdapter):
    """Brain MRI intake, standardization, and analysis adapter."""

    modality = "brain_mri"
    display_name = "AURA Neuro Adapter"

    def inspect(self, asset_path: str, asset_meta: dict[str, Any] | None = None,
                **kwargs) -> InspectionResult:
        """Accept volumetric MR studies; refuse everything else."""
        from aura.backend.engines.neuro.engine import NeuroMindEngine, _safe_can_read
        from aura.backend.foundation.mri.io.dicom_reader import DicomSeriesReader
        from aura.backend.foundation.mri.io.nifti_reader import NiftiReader
        from aura.backend.foundation.mri.io.nrrd_reader import NrrdReader

        meta = asset_meta or {}
        asset_path_obj = Path(asset_path)

        readers = (DicomSeriesReader(), NiftiReader(), NrrdReader())
        claimed_by = next(
            (r.file_format.value for r in readers if _safe_can_read(r, asset_path_obj)),
            None,
        )

        import zipfile
        is_zip = zipfile.is_zipfile(asset_path)
        if claimed_by is None and is_zip:
            try:
                with zipfile.ZipFile(asset_path, 'r') as zf:
                    for name in zf.namelist():
                        nl = name.lower()
                        if nl.endswith(('.dcm', '.dicom')) or 'dicom' in nl:
                            claimed_by = "DICOM"
                            break
                        elif nl.endswith(('.nii', '.nii.gz')):
                            claimed_by = "NIfTI"
                            break
                        elif nl.endswith('.nrrd'):
                            claimed_by = "NRRD"
                            break
            except Exception:
                pass

        suffix = asset_path_obj.suffix.lower()
        checks: dict[str, Any] = {
            "readable_format": claimed_by or "(none)",
            "file_suffix": suffix or "(none)",
        }

        if claimed_by is not None:
            from aura.backend.engines.neuro.multisequence import looks_multisequence
            channels = 4
            if looks_multisequence(asset_path_obj, channels):
                checks["multisequence_channels"] = channels
            return InspectionResult(accepted=True, reason=f"MR study ({claimed_by})", checks=checks)

        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            return InspectionResult(
                accepted=False,
                reason="2D image export cannot be analysed by the neuro adapter; "
                       "upload a DICOM series, NIfTI, or NRRD volume",
                checks=checks,
            )

        return InspectionResult(
            accepted=False,
            reason="not a volumetric MR study",
            checks=checks,
        )

    def standardize(self, asset_path: str, asset_meta: dict[str, Any] | None = None,
                    **kwargs) -> StandardizedAsset:
        """Standardise the MR study through the MRI Foundation Layer."""
        from aura.backend.engines.neuro.engine import NeuroMindEngine
        from aura.backend.engines.neuro.multisequence import looks_multisequence, load_multisequence, MultiSequenceStudy
        from aura.backend.foundation.mri.errors import MRIFoundationError
        from aura.backend.engines.neuro.engine import _safe_can_read
        from aura.backend.foundation.mri.io.dicom_reader import DicomSeriesReader
        from aura.backend.foundation.mri.io.nifti_reader import NiftiReader
        from aura.backend.foundation.mri.io.nrrd_reader import NrrdReader

        meta = asset_meta or {}
        store = kwargs.get("store")
        asset_path_obj = Path(asset_path)

        keys = (
            "FLAIR", "T1", "T1ce", "T2"
        )

        import zipfile
        import shutil
        import tempfile
        temp_dir_path = None
        study_path = asset_path_obj

        is_zip = zipfile.is_zipfile(asset_path)
        if is_zip:
            temp_dir_path = Path(tempfile.mkdtemp(prefix="aura-unzipped-", dir=str(Path.cwd())))
            with zipfile.ZipFile(asset_path, 'r') as zf:
                zf.extractall(temp_dir_path)
            study_path = temp_dir_path

        try:
            if looks_multisequence(study_path, len(keys)):
                return self._preprocess_multisequence(
                    study_path, keys, asset_path, meta, store)

            from aura.backend.foundation.mri import FoundationConfig, MRIFoundationPipeline
            foundation = MRIFoundationPipeline(FoundationConfig(reject_on_quality=False))
            study = foundation.run(
                study_path,
                study_id=f"STU-MR-{meta.get('sha256', 'unknown')[:12]}",
                display_name=meta.get("filename", "upload"),
            )

            index = (store.count() + 1) if store else 1
            case_id = meta.get("case_id", f"CASE-MR-{index}")
            return StandardizedAsset(
                study_id=study.study_id,
                case_id=case_id,
                payload=study,
                metadata={
                    "sha256": meta.get("sha256", ""),
                    "foundation": {
                        "sha256": meta.get("sha256", ""),
                        "foundation_version": study.foundation_version,
                        "source_format": study.source_format.value,
                        "series_count": len(study),
                        "sequences_identified": [s.value for s in study.sequences_present],
                        "quality_verdict": study.verdict.value,
                    },
                },
            )
        finally:
            if temp_dir_path and temp_dir_path.exists():
                try:
                    shutil.rmtree(temp_dir_path)
                except Exception:
                    pass

    def _preprocess_multisequence(self, study_path, keys, asset_path, meta, store):
        """Load a complete 4D study."""
        from aura.backend.engines.neuro.multisequence import load_multisequence

        study = load_multisequence(study_path, keys)
        index = (store.count() + 1) if store else 1
        return StandardizedAsset(
            study_id=f"STU-MR-{meta.get('sha256', 'unknown')[:12]}",
            case_id=meta.get("case_id", f"CASE-MR-{index}"),
            payload=study,
            metadata={
                "sha256": meta.get("sha256", ""),
                "foundation": {
                    "source_format": "nifti-4d",
                    "series_count": len(keys),
                    "sequences_identified": list(keys),
                    "spacing_mm": list(study.spacing_mm or ()),
                    "channel_order_source": study.order_source,
                    "channel_order_check": study.order_endorsement,
                },
            },
        )

    async def analyze(self, standardized: StandardizedAsset,
                      pipeline: Any, store: Any,
                      on_case_created: Any | None = None,
                      **kwargs) -> EngineOutput:
        """Run NeuroMindEngine analysis and persist results."""
        from aura.backend.engines.neuro.engine import NeuroMindEngine
        from aura.backend.engines.neuro.bundle import build_case_bundle
        from aura.backend.engines.neuro.calibration import load_calibrator
        from aura.backend.engines.neuro.multisequence import MultiSequenceStudy
        from aura.backend.engines.neuro.neuroview import build_neuroview_payload
        from aura.backend.vision.brain.inference import BrainVisionEngine
        from aura.backend.engines.neuro.engine import _representative_index
        from aura.schemas.contracts import StructuredPriors
        import numpy as np

        case_id = standardized.case_id
        study = standardized.payload

        vision_engine = BrainVisionEngine.load()
        calibrator = load_calibrator()

        if isinstance(study, MultiSequenceStudy):
            volumes = study.volumes
            brain = np.any(volumes > 0, axis=0)
            per_slice = brain.reshape(-1, brain.shape[2]).mean(axis=0)
            keep = np.flatnonzero(per_slice >= 0.10)
            if keep.size == 0:
                keep = np.arange(volumes.shape[3])
            slices = [np.ascontiguousarray(volumes[..., int(z)]) for z in keep]
            output = vision_engine.analyze_slices(
                slices, study_id=standardized.study_id,
                sequences_used=list(study.sequence_keys), sequences_missing=[],
                spacing_mm=study.spacing_mm)
        else:
            output = vision_engine.analyze_study(study)

        raw = output.tumor_probability
        presence = calibrator(raw) if raw is not None else 0.5

        caveats = [calibrator.summary + "."]

        from aura.backend.engines.neuro.bundle import _representative_index
        display_volume = study.volumes[0] if isinstance(study, MultiSequenceStudy) else study.series[0].volume.array
        index = _representative_index(output)
        depth = display_volume.shape[2] if display_volume.ndim == 3 else 1
        index = min(index, max(0, depth - 1))
        plane = display_volume[:, :, index] if display_volume.ndim == 3 else display_volume
        plane = np.asarray(plane, dtype=np.float32)
        lo, hi = float(plane.min()), float(plane.max())
        image = (plane - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(plane)

        bundle = build_case_bundle(
            output,
            case_id=case_id,
            study_id=standardized.study_id,
            image=image,
            presence_probability=presence,
            priors=StructuredPriors(),
            caveats=caveats,
        )

        t0 = time.perf_counter()
        if store:
            store.save_case(bundle)
            store.save_neuroview(
                case_id,
                standardized.study_id,
                self._build_neuroview(case_id, study, output, vision_engine),
            )
        if on_case_created is not None:
            try:
                on_case_created(case_id)
            except Exception:
                pass
        if store:
            try:
                store.audit(
                    "case.uploaded", "case", case_id,
                    detail={
                        "top": bundle.safety.top.value,
                        "abstained": bool(bundle.safety.abstained),
                        "engine": "neuromind",
                        "via": "modality_adapter",
                    },
                )
            except Exception:
                pass
        inference_s = time.perf_counter() - t0

        return EngineOutput(
            case_id=case_id,
            study_id=standardized.study_id,
            bundle=bundle,
            metadata={
                "inference_time_s": round(inference_s, 4),
                "top_diagnosis": bundle.safety.top.value,
                "top_probability": round(float(bundle.safety.top_probability), 4),
                "abstained": bool(bundle.safety.abstained),
                "fusion_backend": bundle.fusion.backend,
                "presence_probability_calibrated": round(float(presence), 4),
                "model_version": output.model_version,
            },
        )

    def _build_neuroview(self, case_id, study, output, vision_engine):
        from aura.backend.engines.neuro.neuroview import build_neuroview_payload
        return build_neuroview_payload(
            case_id=case_id,
            study=study,
            output=output,
            modalities=tuple(vision_engine.network.modalities),
            min_brain_fraction=float(vision_engine.config.ingest.min_brain_fraction),
        )
