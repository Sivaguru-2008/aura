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
    async def route_study(file: UploadFile = File(...)):
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
                async with UploadIntake().receive(file) as asset:
                    return dispatch.inspect(asset)
            except AuraBackendError as exc:
                return _error(exc)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/analyze", response_model=AnalysisEnvelope,
              summary="Route a study to its engine and analyse it")
    async def analyze_study(file: UploadFile = File(...),
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

                async with UploadIntake().receive(file) as asset:
                    return await dispatch.dispatch(
                        asset, request_id=request_id,
                        declared_modality=declared_modality or force_modality,
                        clinical_context=clinical_context)
            except AuraBackendError as exc:
                return _error(exc)

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
