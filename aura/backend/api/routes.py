"""Routing-layer HTTP endpoints.

    POST /v1/studies/route     detect modality + select engine, no analysis
    POST /v1/studies/analyze   detect, select, and run the engine
    GET  /v1/engines           registered engines, modalities, detector configuration

``POST /v1/studies/upload`` — the legacy chest-X-ray endpoint — is untouched and
keeps its exact behaviour. Existing clients need no change; new clients use
``/analyze`` and get modality independence.

Why ``/route`` exists as its own endpoint: a client should be able to ask "what is
this and would you analyse it?" before committing to an inference run. It is cheap
(no model load, no engine construction) and it is the endpoint to reach for when
diagnosing a rejected or misrouted upload, since it returns every candidate the
detector scored, with the evidence behind each one.
"""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from backend.core.shared.errors import AuraBackendError
from backend.core.shared.logging import get_logger, new_correlation_id, use_correlation_id
from backend.models.routing import AnalysisEnvelope, RoutingMetadata
from backend.services.dispatch import DispatchService

log = get_logger("api")


def build_router(dispatch: DispatchService) -> APIRouter:
    """Build the API router bound to a dispatch service.

    Takes the service as an argument rather than importing a singleton so tests can
    mount the routes against a stub, and so a deployment can run more than one
    configuration in one process.
    """
    api = APIRouter(tags=["modality-router"])

    def _error(exc: AuraBackendError) -> JSONResponse:
        """Uniform error rendering. Internal detail stays in the log."""
        log.info(
            "request rejected",
            extra={"context": {"code": exc.code, "reason": exc.reason}},
        )
        return JSONResponse(exc.to_payload(), status_code=exc.http_status)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/route", response_model=RoutingMetadata,
              summary="Identify a study's modality and the engine that would analyse it")
    async def route_study(file: UploadFile | None = File(None),
                          files: list[UploadFile] | None = File(None)):
        """Inspect an upload without analysing it.

        Returns the full :class:`RoutingMetadata`, including every scored candidate.
        Unroutable and unsupported uploads still return **200** here — "I could not
        identify this" is a successful inspection, and the body explains why. Only a
        transport-level rejection (disallowed type, oversized) is an error status.
        """
        from backend.core.upload import UploadIntake

        request_id = new_correlation_id()
        with use_correlation_id(request_id):
            try:
                if files:
                    async with UploadIntake().receive_multiple(files) as asset:
                        return dispatch.inspect(asset)
                elif file:
                    async with UploadIntake().receive(file) as asset:
                        return dispatch.inspect(asset)
                else:
                    from backend.core.shared.errors import UploadRejected
                    raise UploadRejected("No file or files uploaded.", http_status=400)
            except AuraBackendError as exc:
                return _error(exc)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/analyze", response_model=AnalysisEnvelope,
              summary="Route a study to its engine and analyse it")
    async def analyze_study(file: UploadFile | None = File(None),
                            files: list[UploadFile] | None = File(None),
                            declared_modality: str | None = None,
                            force_modality: str | None = None,
                            age: int | None = None,
                            sex: str | None = None,
                            symptoms: str | None = None,
                            previous_diagnosis: str | None = None,
                            previous_surgery: str | None = None,
                            radiotherapy: bool | None = None,
                            chemotherapy: bool | None = None,
                            clinical_notes: str | None = None):
        """Route and analyse in one call — the modality-agnostic upload endpoint."""
        from backend.core.upload import UploadIntake

        request_id = new_correlation_id()
        with use_correlation_id(request_id):
            try:
                clinical_context = {
                    "age": age,
                    "sex": sex,
                    "symptoms": symptoms,
                    "previous_diagnosis": previous_diagnosis,
                    "previous_surgery": previous_surgery,
                    "radiotherapy": radiotherapy,
                    "chemotherapy": chemotherapy,
                    "clinical_notes": clinical_notes,
                }
                # Filter out None values
                clinical_context = {k: v for k, v in clinical_context.items() if v is not None}

                if files:
                    async with UploadIntake().receive_multiple(files) as asset:
                        return await dispatch.dispatch(
                            asset, request_id=request_id,
                            declared_modality=declared_modality or force_modality,
                            clinical_context=clinical_context)
                elif file:
                    async with UploadIntake().receive(file) as asset:
                        return await dispatch.dispatch(
                            asset, request_id=request_id,
                            declared_modality=declared_modality or force_modality,
                            clinical_context=clinical_context)
                else:
                    from backend.core.shared.errors import UploadRejected
                    raise UploadRejected("No file or files uploaded.", http_status=400)
            except AuraBackendError as exc:
                return _error(exc)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/preview",
              summary="Parse and extract metadata and preview thumbnails of an upload without running inference")
    async def preview_study(file: UploadFile | None = File(None),
                            files: list[UploadFile] | None = File(None)):
        """Parse and validate the upload, generating preview thumbnails and metadata without running AI inference."""
        from backend.core.upload import UploadIntake
        from backend.foundation.mri.intake_manager import MRIIntakeManager
        from PIL import Image
        import io
        import base64
        import numpy as np

        async def _do_preview(asset) -> dict[str, Any]:
            # Process using MRIIntakeManager
            manager = MRIIntakeManager()
            study = manager.process(asset.path)

            # Extract shape and spacing
            H, W, Z = study.volumes.shape[1:4]
            spacing = list(study.spacing_mm)

            # Generate 2D thumbnails for each sequence
            thumbnails = {}
            order = ["flair", "t1", "t1ce", "t2"]
            for c, seq_key in enumerate(order):
                vol = study.volumes[c] # Shape (H, W, Z)
                slice_2d = vol[:, :, Z // 2] # Get middle slice
                # Normalize to 0-255 uint8
                lo, hi = float(slice_2d.min()), float(slice_2d.max())
                if hi - lo > 1e-6:
                    slice_uint8 = np.clip((slice_2d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                else:
                    slice_uint8 = np.zeros(slice_2d.shape, dtype=np.uint8)

                # Flip vertically so it's not upside down (matches drawing logic)
                slice_uint8 = np.flipud(slice_uint8)

                # Encode to base64 PNG
                img = Image.fromarray(slice_uint8)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
                thumbnails[seq_key] = f"data:image/png;base64,{b64_str}"

            affine_repr = f"Diagonal spacing affine matrix [{spacing[0]}mm, {spacing[1]}mm, {spacing[2]}mm]"

            # Everything reported here is read off the parsed study. Fields the
            # intake layer does not carry (``MultiSequenceStudy`` holds volumes,
            # sequence keys, spacing and channel-order provenance — no DICOM
            # demographics, no scanner tags, no affine orientation) are returned as
            # null so the console renders "—". A preview that invents a patient id,
            # an orientation or a scanner make is indistinguishable from one that
            # read them off the file, which is exactly the confusion a clinical
            # intake screen must not create.
            seq_keys = list(getattr(study, "sequence_keys", ()) or [])
            n_seq = len(seq_keys) or study.volumes.shape[0]

            return {
                "status": "success",
                "patient_id": None,               # not carried by the intake layer
                "study_id": f"STU-MR-{asset.sha256[:12]}",
                "detected_modalities": [s.upper() for s in seq_keys],
                "voxel_spacing": spacing,
                "orientation": None,              # no affine is retained upstream
                "original_dimensions": [H, W, Z],
                "number_of_slices": Z,
                "affine_matrix_summary": affine_repr,
                "sequence_type": f"Multi-sequence 3D MRI study ({n_seq} sequences)",
                "channel_order_source": getattr(study, "order_source", None),
                "scanner_metadata": None,         # no DICOM tags retained upstream
                "thumbnails": thumbnails
            }

        request_id = new_correlation_id()
        with use_correlation_id(request_id):
            try:
                if files:
                    async with UploadIntake().receive_multiple(files) as asset:
                        return await _do_preview(asset)
                elif file:
                    async with UploadIntake().receive(file) as asset:
                        return await _do_preview(asset)
                else:
                    from backend.core.shared.errors import UploadRejected
                    raise UploadRejected("No file or files uploaded.", http_status=400)
            except AuraBackendError as exc:
                return _error(exc)
            except Exception as exc:
                from backend.foundation.mri.errors import StudyValidationError
                return _error(StudyValidationError(f"Preview failed: {exc}"))

    # ------------------------------------------------------------------ #
    @api.post("/v1/cases/progression",
              summary="Compare a previous MRI with a current MRI to track progression")
    def compare_cases(
        previous_case_id: str,
        current_case_id: str,
        growth_threshold: float | None = None,
        regression_threshold: float | None = None,
    ):
        from gateway.app import store
        from backend.services.reasoning.progression import LongitudinalAnalyzer
        from fastapi import HTTPException

        prev = store().get_case(previous_case_id)
        curr = store().get_case(current_case_id)
        if prev is None:
            raise HTTPException(status_code=404, detail=f"Previous case {previous_case_id} not found.")
        if curr is None:
            raise HTTPException(status_code=404, detail=f"Current case {current_case_id} not found.")

        report = LongitudinalAnalyzer.compare(
            prev, curr,
            growth_threshold=growth_threshold,
            regression_threshold=regression_threshold
        )
        return report

    # ------------------------------------------------------------------ #
    @api.get("/v1/cases/{case_id}/tracking",
             summary="Get historical tumor tracking timeline for a case")
    def track_case(case_id: str):
        from gateway.app import store
        from backend.services.reasoning.tracking import TumorTracker
        from fastapi import HTTPException

        case = store().get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

        timeline = TumorTracker.get_timeline(store(), case_id)
        return timeline

    # ------------------------------------------------------------------ #
    @api.get("/v1/engines",
             summary="Registered analysis engines, served modalities, and detector config")
    def list_engines():
        """Platform capability listing.

        `status` distinguishes `available` from `placeholder` (routes resolve,
        analysis pending) and `unavailable` (registered but failed to construct on
        this deployment), so a client can tell what will happen before uploading.
        """
        return dispatch.describe()

    return api
