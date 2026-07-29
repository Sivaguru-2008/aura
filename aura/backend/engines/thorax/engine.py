"""Thorax engine adapter — the existing chest-X-ray pipeline behind the new contract.

**Nothing in the chest-X-ray stack is modified by this file.** Every stage calls the
same public functions ``POST /v1/studies/upload`` has always called, in the same
order, with the same arguments:

    validate_input   -> services.vision.xray_gate.validate_cxr
    preprocess       -> services.vision.io.study_from_cxr(grid=224)
    analyze          -> gateway.pipeline.Pipeline.run  +  Store.save_case
    generate_report  -> the ReportDraft the pipeline already composed

That correspondence is the whole point: the router becomes a second door into an
unchanged room. The legacy endpoint keeps working byte-for-byte, and a case created
through the router is indistinguishable from one created through the legacy path —
same case-id scheme, same persistence, same audit rows, same inference log — so the
console worklist, history portal, and report signing all keep working untouched.

The engine holds *injected* handles to the live ``Pipeline`` and ``Store`` rather than
constructing its own. Building a second ``Pipeline`` would load a second copy of the
DenseNet backbone and the fusion model into memory, and — worse — a second
``MemoryEngine``, so similarity search would silently see only half the cases.
"""
from __future__ import annotations

import time
from typing import Callable

from aura.backend.core.shared.errors import EngineExecutionError, UnreadableImage
from aura.backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
from ..base.contract import (
    AnalysisEngine,
    AnalysisResult,
    EngineDescriptor,
    EngineReport,
    PreparedStudy,
    ValidationOutcome,
)
from ..base.registry import EngineRegistry, default_registry

#: Resolution the film reaches the CNN at. 224 is the DenseNet-121 input size, and
#: matching the legacy endpoint here is not cosmetic: a smaller grid erases thin
#: structures (a pneumothorax pleural line, a small nodule) before the model sees
#: them — the fidelity repair recorded as audit F5.
_GRID = 224


class ThoraxEngine(AnalysisEngine):
    """Chest radiograph analysis via the pre-existing AURA pipeline."""

    descriptor = EngineDescriptor(
        engine_id="thorax",
        display_name="AURA Thorax",
        version="1.0.0",
        modalities=(ImagingModality.CHEST_XRAY,),
        status=EngineStatus.AVAILABLE,
        capabilities=(
            "finding_detection",
            "calibrated_diagnosis",
            "conformal_abstention",
            "grad_cam_localization",
            "grounded_report",
            "clinical_reasoning",
        ),
        description=(
            "DenseNet-121 findings, Platt-calibrated per finding, fused with "
            "structured priors and validated by the conformal safety engine."
        ),
    )

    def __init__(
        self,
        pipeline,
        store,
        on_case_created: Callable[[str], None] | None = None,
    ) -> None:
        """
        Args:
            pipeline: the live ``gateway.pipeline.Pipeline``.
            store: the live ``gateway.storage.Store``.
            on_case_created: optional hook fired with each new case id. The gateway
                uses it to add router-created cases to the active session worklist,
                so they appear in the console exactly like legacy uploads.
        """
        self._pipeline = pipeline
        self._store = store
        self._on_case_created = on_case_created
        super().__init__()

    # ------------------------------------------------------------------ #
    # 1. validate_input
    # ------------------------------------------------------------------ #
    def validate_input(self, asset: ImageAsset) -> ValidationOutcome:
        """Run the production chest-radiograph intake gate.

        This repeats the router's chest check on purpose. The router decides *where*
        an image goes; this decides whether *this model* is willing to look at it.
        Keeping the engine's own refusal intact means swapping the detector — for a
        learned classifier, say — can never expose the chest model to a knee film.
        """
        from aura.backend.core.router.router import ModalityRouter
        from aura.backend.core.shared.types import ImagingModality
        
        router = ModalityRouter()
        metadata = router.route(asset)
        if metadata.modality != ImagingModality.CHEST_XRAY.value:
            reason = f"Unsupported modality entered Thorax engine: detected {metadata.modality_label} ({metadata.confidence:.2f})"
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

        from aura.services.vision.xray_gate import validate_cxr

        gate = validate_cxr(asset.path)
        return ValidationOutcome(
            accepted=bool(gate.ok),
            reason=gate.reason or "",
            checks=dict(gate.checks or {}),
        )

    # ------------------------------------------------------------------ #
    # 2. preprocess
    # ------------------------------------------------------------------ #
    def preprocess(self, asset: ImageAsset) -> PreparedStudy:
        """Decode the film to a 224x224 ``StudyInput`` — identical to the legacy path."""
        from aura.services.vision.io import study_from_cxr

        try:
            study = study_from_cxr(asset.path, grid=_GRID)
        except Exception as exc:
            raise UnreadableImage(
                "the radiograph could not be decoded",
                detail={"filename": asset.original_filename},
            ) from exc

        # Case numbering matches the legacy endpoint so both doors produce one
        # continuous, non-colliding sequence in the store.
        index = self._store.count() + 1
        study.study_id = f"STU-UPLOAD-{index}"
        return PreparedStudy(
            study_id=study.study_id,
            modality=ImagingModality.CHEST_XRAY,
            payload=study,
            metadata={"case_id": f"CASE-UPLOAD-{index}", "grid": _GRID,
                      "sha256": asset.sha256, "source_path": str(asset.path)},
        )

    # ------------------------------------------------------------------ #
    # 3. analyze
    # ------------------------------------------------------------------ #
    async def analyze(self, prepared: PreparedStudy) -> AnalysisResult:
        """Run the full pipeline, persist the case, and write the audit trail."""
        study = prepared.payload
        case_id = prepared.metadata["case_id"]

        started = time.perf_counter()
        bundle = await self._pipeline.run(study, case_id=case_id)
        inference_seconds = time.perf_counter() - started

        self._store.save_case(bundle)
        if self._on_case_created is not None:
            try:
                self._on_case_created(case_id)
            except Exception:
                # A worklist hook must never lose a persisted case.
                self.log.exception("case-created hook failed; case is saved regardless")

        self._audit(case_id, bundle)
        self._log_inference(bundle, prepared, inference_seconds)

        return AnalysisResult(
            study_id=prepared.study_id,
            payload=bundle,
            case_id=case_id,
            metadata={
                "inference_time_s": round(inference_seconds, 4),
                "top_diagnosis": bundle.safety.top.value if bundle.safety else None,
                "top_probability": (round(float(bundle.safety.top_probability), 4)
                                    if bundle.safety else None),
                "abstained": bool(bundle.safety.abstained) if bundle.safety else None,
                "fusion_backend": bundle.fusion.backend if bundle.fusion else None,
                "state": bundle.state.value if hasattr(bundle.state, "value")
                         else str(bundle.state),
            },
        )

    def _audit(self, case_id: str, bundle) -> None:
        """Audit row for the routed upload. Logged, never swallowed (audit §10.9)."""
        try:
            self._store.audit(
                "case.uploaded", "case", case_id,
                detail={
                    "top": bundle.safety.top.value if bundle.safety else None,
                    "abstained": bool(bundle.safety.abstained) if bundle.safety else None,
                    "via": "modality_router",
                },
            )
        except Exception:
            self.log.exception(f"FAILED to write audit row for {case_id}")

    def _log_inference(self, bundle, prepared: PreparedStudy, seconds: float) -> None:
        """Full-provenance inference record — same writer the legacy endpoint uses."""
        try:
            from aura.services.inference.audit_log import log_inference

            log_inference(
                bundle,
                prepared.metadata["source_path"],
                seconds,
                backbone=getattr(self._pipeline.vision, "backbone", None),
            )
        except Exception:
            self.log.exception("inference logging failed; analysis result is unaffected")

    # ------------------------------------------------------------------ #
    # 4. generate_report
    # ------------------------------------------------------------------ #
    def generate_report(self, result: AnalysisResult) -> EngineReport:
        """Surface the grounded report the pipeline already composed.

        Pure projection — no new inference, no new text. The pipeline's
        ``ReportEngine`` produced this draft from the same findings and safety
        assessment ``analyze`` returned, and its ``grounding`` map ties every section
        back to those evidence nodes.
        """
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
            summary=draft.impression_text or "Chest radiograph analysed.",
            grounding=dict(draft.grounding or {}),
        )


def register_thorax_engine(
    pipeline,
    store,
    on_case_created: Callable[[str], None] | None = None,
    registry: EngineRegistry | None = None,
) -> None:
    """Bind the Thorax engine to live gateway handles and register it.

    Called from the gateway's startup once the ``Pipeline`` and ``Store`` exist. The
    factory closure defers construction until the first chest upload actually
    arrives, so registering costs nothing.
    """
    (registry or default_registry).register(
        ThoraxEngine.descriptor,
        lambda: ThoraxEngine(pipeline, store, on_case_created),
        replace=True,
    )
