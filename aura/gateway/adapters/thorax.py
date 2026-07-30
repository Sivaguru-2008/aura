"""Thorax adapter — chest radiography pipeline behind the ModalityAdapter interface.

Wraps the existing ``validate_cxr`` gate, ``study_from_cxr`` standardization,
and ``Pipeline.run`` analysis into the three-phase adapter contract.
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


class ThoraxAdapter(ModalityAdapter):
    """Chest radiograph intake, standardization, and analysis adapter."""

    modality = "chest_xray"
    display_name = "AURA Thorax Adapter"

    def inspect(self, asset_path: str, asset_meta: dict[str, Any] | None = None,
                **kwargs) -> InspectionResult:
        """Run the chest-radiograph intake gate (cheap, no model load)."""
        from aura.services.vision.xray_gate import validate_cxr

        gate = validate_cxr(asset_path)
        return InspectionResult(
            accepted=bool(gate.ok),
            reason=gate.reason or "",
            checks=dict(gate.checks or {}),
        )

    def standardize(self, asset_path: str, asset_meta: dict[str, Any] | None = None,
                    **kwargs) -> StandardizedAsset:
        """Decode the film to a 224x224 StudyInput."""
        from aura.services.vision.io import study_from_cxr

        meta = asset_meta or {}
        store = kwargs.get("store")
        grid = kwargs.get("grid", 224)

        study = study_from_cxr(asset_path, grid=grid)

        index = (store.count() + 1) if store else 1
        case_id = meta.get("case_id", f"CASE-UPLOAD-{index}")
        study.study_id = meta.get("study_id", f"STU-UPLOAD-{index}")

        return StandardizedAsset(
            study_id=study.study_id,
            case_id=case_id,
            payload=study,
            metadata={
                "grid": grid,
                "sha256": meta.get("sha256", ""),
                "source_path": asset_path,
            },
        )

    async def analyze(self, standardized: StandardizedAsset,
                      pipeline: Any, store: Any,
                      on_case_created: Any | None = None,
                      **kwargs) -> EngineOutput:
        """Run the full chest-X-ray pipeline, persist the case, write audit."""
        study = standardized.payload
        case_id = standardized.case_id

        t0 = time.perf_counter()
        bundle = await pipeline.run(study, case_id=case_id)
        inference_s = time.perf_counter() - t0

        if store:
            store.save_case(bundle)
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
                        "top": bundle.safety.top.value if bundle.safety else None,
                        "abstained": bool(bundle.safety.abstained) if bundle.safety else None,
                        "via": "modality_adapter",
                    },
                )
            except Exception:
                pass

        try:
            from aura.services.inference.audit_log import log_inference
            log_inference(
                bundle,
                standardized.metadata.get("source_path", ""),
                inference_s,
                backbone=getattr(pipeline.vision, "backbone", None),
            )
        except Exception:
            pass

        return EngineOutput(
            case_id=case_id,
            study_id=standardized.study_id,
            bundle=bundle,
            metadata={
                "inference_time_s": round(inference_s, 4),
                "top_diagnosis": bundle.safety.top.value if bundle.safety else None,
                "top_probability": (round(float(bundle.safety.top_probability), 4)
                                    if bundle.safety else None),
                "abstained": bool(bundle.safety.abstained) if bundle.safety else None,
                "fusion_backend": bundle.fusion.backend if bundle.fusion else None,
            },
        )
