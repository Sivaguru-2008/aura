"""FastAPI gateway: /v1 API + dashboard hosting + audit middleware.

Auth is stubbed for the P0 demo (a header-based principal); the RBAC/OIDC seam is
marked so production auth drops in at the same boundary. See docs/ARCHITECTURE.md
sections 13 & 15.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError

from aura.common.config import DB_PATH, ensure_dirs, get_settings
from aura.ml.data import IMG, make_multimodal, make_sample
from aura.schemas.clinical import DIAGNOSES, Diagnosis, Modality
from aura.schemas.contracts import (
    CaseState, ClinicalChatRequest, ClinicalChatResponse, StudyInput, StructuredPriors,
)
from aura.services.models import ModelRegistry
from .intake import IntakeRequest, symptoms_text, triage_priority
from .pipeline import Pipeline
from .seed import seed
from .storage import Store

from aura.services.enterprise.fhir import export_fhir_diagnostic_report, export_fhir_observations
from aura.services.enterprise.hl7 import export_hl7_oru_r01
from aura.services.agent.discussion import SpecialistDiscussionEngine


WEB_DIR = Path(__file__).resolve().parent.parent / "apps" / "web"

#: Local hour at which the nightly recalibration runs. Deployments in other
#: timezones (or with a different maintenance window) override it rather than
#: patching the scheduler.
RECALIBRATION_HOUR = max(0, min(23, int(os.environ.get("AURA_RECALIBRATION_HOUR", "2"))))

state: dict = {}
session_case_ids: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    settings = get_settings()
    # A commercial deployment must not serve research-licensed weights, so it refuses to
    # start (aura/common/data_governance.py). Research and pilot deployments skip this.
    if settings.commercial_mode:
        from aura.common.data_governance import get_governance_manager
        get_governance_manager().enforce_commercial_gate(commercial_mode=True)
    # AURA_DATABASE_URL moves the case store to PostgreSQL; AURA_SITE_ID binds it to
    # this hospital. A site mismatch raises here and the server refuses to start,
    # which is the point: see storage.SiteBindingRow.
    store = Store(DB_PATH, url=settings.database_url or None,
                  site_id=settings.site_id or None)
    pipeline = Pipeline(store=store)          # store handle enables online ACI (F9)
    state["store"] = store
    state["pipeline"] = pipeline
    state["chest_registry"] = ModelRegistry("chest_registry.json")
    state["brain_registry"] = ModelRegistry("brain_registry.json")
    state["registry"] = state["chest_registry"]

    # Intelligent Modality Router (backend/): mounts /v1/studies/route,
    # /v1/studies/analyze and /v1/engines, and binds the Thorax engine adapter to the
    # pipeline + store built above. This is the ONLY integration point — the chest-X-ray
    # pipeline and the legacy /v1/studies/upload endpoint are unchanged. A router
    # failure must not prevent the chest-X-ray service from starting.
    try:
        from aura.backend.bootstrap import install_router

        state["dispatch"] = install_router(
            app, pipeline, store, on_case_created=session_case_ids.append
        )
        
        # Start mock PACS DICOM listener on port 11112
        try:
            from aura.services.enterprise.dicom_listener import DicomListener
            listener = DicomListener(state["dispatch"], store, port=11112)
            listener.start()
            state["dicom_listener"] = listener
        except Exception as dle:
            print(f"[gateway] WARNING: mock DICOM listener failed to start: {dle!r}")
            
    except Exception as e:                        # pragma: no cover - startup resilience
        print(f"[gateway] WARNING: modality router failed to install: {e!r}")

    if not pipeline.fusion.is_trained():
        print("[gateway] WARNING: fusion model not trained; run `aura_cli train`.")

    # Nightly recalibration. It must land at a real off-hours time, not 30s after
    # every boot: the run rewrites safety_<backend>.npz, which carries the
    # temperature and conformal threshold the abstention gate depends on. A
    # restart-triggered rewrite would silently move the safety envelope, and a
    # development restart loop would do it repeatedly.
    import asyncio
    from datetime import datetime, timedelta

    async def _sleep_until_next_run() -> None:
        now = datetime.now()
        target = now.replace(hour=RECALIBRATION_HOUR, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())

    async def _nightly_recalibrate_loop():
        while True:
            await _sleep_until_next_run()
            try:
                from aura.services.training.recalibrate import recalibrate_from_feedback
                print(f"[recalibrate] Nightly recalibration starting ({datetime.now():%Y-%m-%d %H:%M}).")
                res = recalibrate_from_feedback(store)
                print(f"[recalibrate] Nightly recalibration finished: {res}")
            except asyncio.CancelledError:
                raise
            except Exception as ree:
                print(f"[recalibrate] Scheduled recalibration failed: {ree!r}")

    state["recalibrate_task"] = asyncio.create_task(_nightly_recalibrate_loop())
    print(f"[recalibrate] Nightly recalibration scheduled for {RECALIBRATION_HOUR:02d}:00 local.")

    # PRODUCTION: Seeding has been disabled. The worklist starts empty.
    # Only newly uploaded images appear in the active session.
    import os
    source = os.environ.get("AURA_DATA_SOURCE", "mimic").lower()
    print(f"[gateway] ready — 0 cases in active session (worklist empty on startup, fusion backend: {pipeline.fusion.backend}).")
    yield
    
    # Lifespan shutdown: stop the DICOM listener & recalibrate task
    if "recalibrate_task" in state and state["recalibrate_task"]:
        try:
            state["recalibrate_task"].cancel()
            print("[recalibrate] Nightly recalibration background task stopped")
        except Exception:
            pass

    if "dicom_listener" in state and state["dicom_listener"]:
        try:
            state["dicom_listener"].stop()
        except Exception as dle:
            print(f"[gateway] WARNING: mock DICOM listener failed to stop: {dle!r}")


app = FastAPI(title="AURA Clinical Intelligence Copilot", version="0.1.0",
              lifespan=lifespan)

# Sign-in endpoints (/v1/auth/status, /v1/auth/session). Registered on the app rather
# than in lifespan, so they exist even when model loading fails and an operator has to
# sign in to find out why.
from .security import router as _auth_router  # noqa: E402
from .enterprise import router as _enterprise_router  # noqa: E402

app.include_router(_auth_router)
app.include_router(_enterprise_router)


@app.middleware("http")
async def audit_mw(request: Request, call_next):
    # Security gate runs *before* the handler on EVERY non-public path: opt-in auth,
    # authorization, and rate limiting (all inert unless configured). A rejection
    # here is itself audited, then returned, so blocked calls are attributable.
    #
    # This used to be gated on `request.method in ("POST","PUT","DELETE")`, which is
    # the wrong axis. Reads are where patient data leaves the box: GET /v1/cases/{id}
    # returns the full bundle, /export/fhir and /export/hl7 emit standards-conformant
    # clinical records built for downstream ingestion, and /v1/admin/safety exposes
    # policy state — all of which were reachable with no token and no rate limit even
    # when auth was fully configured. Case ids were enumerable from GET /v1/cases.
    # enforce() decides what is public via security.is_public_path.
    from .security import enforce
    try:
        principal = enforce(request)
    except HTTPException as exc:
        _safe_audit(action=f"blocked {request.method} {request.url.path}",
                    actor=request.headers.get("x-aura-user", "anonymous"),
                    entity_type="http", detail={"status": exc.status_code})
        return JSONResponse(exc.detail if isinstance(exc.detail, dict)
                            else {"error": exc.detail}, status_code=exc.status_code)

    resp = await call_next(request)
    # Dashboard assets must revalidate on every load — stale cached JS leaves
    # buttons rendered by fresh HTML with no handlers bound.
    if request.url.path == "/" or request.url.path.startswith(("/app", "/static")):
        resp.headers["Cache-Control"] = "no-cache"
    if request.method in ("POST", "PUT", "DELETE") and "store" in state:
        tenant_id = getattr(request.state, "tenant_id", "default")
        _safe_audit(action=f"{request.method} {request.url.path}",
                    actor=principal, entity_type="http", tenant_id=tenant_id)
    return resp


def _safe_audit(**kw) -> None:
    """Write an audit row, logging (never swallowing) a failure.

    An audit log that can silently fail to write is not an audit log (audit
    §10.9). We still must not let an audit failure sink the request, so the write
    is guarded — but the failure is surfaced to the server log instead of a bare
    ``except: pass``.
    """
    if "store" not in state:
        return
    try:
        state["store"].audit(**kw)
    except Exception as e:                       # pragma: no cover - storage failure
        print(f"[audit] FAILED to write audit row {kw!r}: {e}")


def store() -> Store:
    return state["store"]


def pipeline() -> Pipeline:
    return state["pipeline"]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/v1/health")
def health():
    # Public (probes run before any credential), so operational facts only. site_id
    # says which hospital's database this server has open.
    return {"status": "ok", "backend": pipeline().fusion.backend,
            "trained": pipeline().fusion.is_trained(),
            "cases": store().count(),
            "site_id": getattr(store(), "site_id", "") or None}


@app.get("/v1/studies")
def list_studies():
    out = []
    from aura.schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
    for cid in session_case_ids:
        b = store().get_case(cid)
        if b:
            s = b.safety
            try:
                dx_enum = Diagnosis(s.top.value if (s and s.top) else "")
                label = DIAGNOSIS_LABELS.get(dx_enum, s.top.value if (s and s.top) else "")
            except ValueError:
                label = s.top.value if (s and s.top) else ""
            out.append({
                "case_id": b.case_id,
                "study_id": b.study_id,
                "state": b.state.value if hasattr(b.state, "value") else b.state,
                "priority_score": b.priority_score,
                "top_diagnosis": s.top.value if (s and s.top) else "",
                "top_diagnosis_label": label,
                "top_probability": s.top_probability if s else 0.0,
                "abstained": bool(s.abstained) if s else False,
                "backend": (b.fusion.backend if b.fusion else ""),
                "conformal_set": (s.conformal_set if s else []),
                "priors": b.priors or {},
                "created_at": b.created_at.isoformat() if hasattr(b.created_at, "isoformat") else b.created_at,
            })
    from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
    dx_lbls = {d.value: l for d, l in DIAGNOSIS_LABELS.items()}
    ev_lbls = {f.value: l for f, l in FINDING_LABELS.items()}
    return {
        "cases": out,
        "dx_labels": dx_lbls,
        "ev_labels": ev_lbls
    }


@app.get("/v1/cases")
def list_cases(request: Request, state_filter: str | None = None):
    from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
    from .security import get_tenant_id
    tenant_id = get_tenant_id(request)
    dx_lbls = {d.value: l for d, l in DIAGNOSIS_LABELS.items()}
    ev_lbls = {f.value: l for f, l in FINDING_LABELS.items()}
    return {
        "cases": store().list_cases(state=state_filter, tenant_id=tenant_id),
        "dx_labels": dx_lbls,
        "ev_labels": ev_lbls
    }


@app.post("/v1/cases", status_code=201)
def create_intake_case(payload: IntakeRequest = Body(...)):
    """Record a clinical intake taken *before* the study arrives.

    The dashboard's clinical-first flow files the history, labs and symptoms
    first, then uploads the scan; ``POST /v1/cases/{case_id}/attach`` merges the
    two once the study has been analysed. The baseline is deliberately not
    written to the ``cases`` table — see ``storage.IntakeBaselineRow`` for why an
    imaging-less record must stay out of the worklist and out of ``count()``.
    """
    priority, drivers = triage_priority(payload)
    record = {
        "request": json.loads(payload.model_dump_json()),
        "priors": json.loads(payload.priors().model_dump_json()),
        "multimodal": json.loads(payload.multimodal().model_dump_json()),
        "clinical_context": json.loads(payload.clinical_context().model_dump_json()),
        "symptoms_text": symptoms_text(payload.symptoms),
        "clinical_notes": payload.clinical_notes or "",
        "priority_score": priority,
        "priority_drivers": drivers,
    }
    case_id = store().next_intake_id()
    store().save_intake(case_id, record, priority)
    store().audit("intake.created", "intake", case_id,
                  detail={"priority_score": priority, "drivers": drivers})
    return {"case_id": case_id, "state": "awaiting_imaging", **record}


@app.post("/v1/cases/{case_id}/attach")
async def attach_intake_case(case_id: str, payload: dict = Body(...)):
    """Merge a pre-imaging intake baseline into the case that was just analysed.

    The analyse call already carries sex/symptoms/notes as query parameters, but
    not the labs, the symptom flags or the structured history — those arrive
    here. Because those facts are exactly what the reasoning rules fire on (BNP,
    procalcitonin, SpO2 ...), the merge is followed by a re-run of the reasoning
    chain so the posterior reflects them. A re-run failure is reported, not
    swallowed: the merged context is persisted either way.
    """
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    baseline_id = str(payload.get("baseline_case_id") or "").strip()
    if not baseline_id:
        raise HTTPException(422, {"error": "invalid_request",
                                  "reason": "baseline_case_id is required"})
    record = store().get_intake(baseline_id)
    if record is None:
        raise HTTPException(404, "intake baseline not found")

    req = IntakeRequest.model_validate(record.get("request") or {})
    b.priors = req.priors()

    intake_mm = req.multimodal()
    if b.multimodal is None:
        b.multimodal = intake_mm
    else:
        # The intake is the only source of these fields, but merge rather than
        # replace so anything an EHR lookup already resolved is not dropped.
        for name in type(intake_mm.labs).model_fields:
            v = getattr(intake_mm.labs, name)
            if v is not None:
                setattr(b.multimodal.labs, name, v)
        for name in type(intake_mm.symptoms).model_fields:
            if getattr(intake_mm.symptoms, name):
                setattr(b.multimodal.symptoms, name, True)
        for name in type(intake_mm.history).model_fields:
            v = getattr(intake_mm.history, name)
            if name == "smoking_pack_years":
                b.multimodal.history.smoking_pack_years = max(
                    float(b.multimodal.history.smoking_pack_years or 0.0), float(v or 0.0))
            elif v:
                setattr(b.multimodal.history, name, True)

    intake_ctx = req.clinical_context()
    if b.clinical_context is None:
        b.clinical_context = intake_ctx
    else:
        for name in type(intake_ctx).model_fields:
            v = getattr(intake_ctx, name)
            if v is not None and getattr(b.clinical_context, name) in (None, ""):
                setattr(b.clinical_context, name, v)

    store().save_case(b)
    store().mark_intake_attached(baseline_id, case_id)
    store().audit("intake.attached", "case", case_id,
                  detail={"baseline_case_id": baseline_id})

    reasoning_refreshed, reason = True, ""
    try:
        b = await recompute_case(case_id, {"priors": {}, "clinical_context": {}})
    except Exception as exc:                      # pragma: no cover - defensive
        reasoning_refreshed, reason = False, f"{type(exc).__name__}: {exc}"
        print(f"[intake] attach {baseline_id} -> {case_id}: re-run failed: {reason}")

    return {
        "case_id": case_id,
        "baseline_case_id": baseline_id,
        "priors": json.loads(b.priors.model_dump_json()),
        "multimodal": json.loads(b.multimodal.model_dump_json()) if b.multimodal else None,
        "priority_score": b.priority_score,
        "reasoning_refreshed": reasoning_refreshed,
        "reason": reason,
    }


@app.delete("/v1/cases/{case_id}")
def discard_intake_case(case_id: str):
    """Discard a pre-imaging intake baseline that never received a study.

    Scoped to baselines on purpose. An analysed case is a clinical record with an
    audit trail behind it, and its id feeds ``CASE-UPLOAD-{count()+1}``, so
    deleting one would both destroy evidence and let the next upload reuse the
    id — this endpoint refuses rather than doing that.
    """
    if store().delete_intake(case_id):
        store().audit("intake.discarded", "intake", case_id)
        return {"deleted": case_id}
    if store().get_case(case_id) is not None:
        raise HTTPException(409, {"error": "not_deletable",
                                  "reason": "analysed cases are immutable clinical records"})
    raise HTTPException(404, "intake baseline not found")


@app.get("/v1/cases/{case_id}")
def get_case(request: Request, case_id: str):
    from .security import get_tenant_id
    from .storage import CrossTenantAccessError
    tenant_id = get_tenant_id(request)
    try:
        b = store().get_case(case_id, tenant_id=tenant_id)
    except CrossTenantAccessError as exc:
        raise HTTPException(403, {"error": "cross_tenant_access_denied", "reason": str(exc)})
    if b is None:
        raise HTTPException(404, "case not found")
    d = json.loads(b.model_dump_json())
    # Attach the *calibrated* per-finding operating point + present flag so the
    # dashboard shows exactly the findings the model asserts — the same F1-optimal
    # thresholds (0.13–0.29) the grounded report uses (services/report/engine.py,
    # services/inference/predict.py), not a hardcoded 0.5. Without this the console
    # hid genuine detections between the calibrated threshold and 0.5, so a diagnosis
    # could render with its supporting findings invisible (audit H1). The threshold
    # is data-derived (vision_serving_calibration.json), never a static default.
    from aura.common.config import finding_present_threshold
    vis = d.get("vision") or {}
    for f in (vis.get("findings") or []):
        try:
            thr = finding_present_threshold(f["finding"])
            f["threshold"] = round(float(thr), 4)
            f["present"] = bool(float(f["probability"]) >= thr)
        except Exception:
            pass
    return d


@app.get("/v1/cases/{case_id}/geometry")
def get_case_geometry(
    case_id: str,
    method: str | None = None,
    threshold: float = 0.5,
    otsu: bool = False,
    levels: str = "0.5,0.7,0.9",
    max_regions: int = 5,
    overlay: bool = True,
    overlay_size: int | None = None,
):
    """Drawable geometry for a stored case's saliency map.

    Recomputed on request so the frontend can re-threshold, switch attribution
    method, or ask for a larger overlay without re-running inference. `method`
    selects one of `explanation.saliency_methods` (grad_cam++, grad_cam,
    integrated_gradients, smoothgrad, occlusion); omitted, the primary map is
    used.

    Returns normalized region polygons, bounding boxes, intensity-weighted
    centroids, iso-level contours, and an RGBA overlay PNG data URI.
    """
    import numpy as np

    from aura.services.explain.geometry import heatmap_geometry

    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    ex = getattr(b, "explanation", None)
    if ex is None or not ex.saliency:
        raise HTTPException(404, {"error": "no_saliency",
                                  "reason": "this case carries no saliency map"})

    flat = ex.saliency
    if method:
        if method not in (ex.saliency_methods or {}):
            raise HTTPException(404, {
                "error": "unknown_method",
                "reason": f"method {method!r} not computed for this case",
                "available": sorted((ex.saliency_methods or {}).keys()),
            })
        flat = ex.saliency_methods[method]

    shape = tuple(ex.saliency_shape or (64, 64))
    try:
        heat = np.asarray(flat, dtype=float).reshape(shape)
    except ValueError:
        raise HTTPException(500, {"error": "malformed_saliency",
                                  "reason": f"{len(flat)} values do not fit shape {shape}"})

    try:
        level_tuple = tuple(float(v) for v in levels.split(",") if v.strip())
    except ValueError:
        raise HTTPException(422, {"error": "bad_levels",
                                  "reason": "levels must be comma-separated floats"})

    # The probability that weights the overlay is the target finding's own
    # calibrated probability, so intensity tracks evidence rather than the
    # per-finding normalization of the raw map.
    probability = None
    for f in (getattr(b.vision, "findings", None) or []):
        if getattr(f, "finding", None) == ex.saliency_target:
            probability = float(f.probability)
            break

    payload = heatmap_geometry(
        heat,
        probability=probability,
        finding=ex.saliency_target,
        levels=level_tuple,
        thresh_rel=threshold,
        use_otsu=otsu,
        include_overlay=overlay,
        overlay_size=(overlay_size, overlay_size) if overlay_size else None,
        max_regions=max_regions,
    )
    payload["case_id"] = case_id
    payload["method"] = method or ex.method
    payload["available_methods"] = sorted((ex.saliency_methods or {}).keys())
    return payload


@app.get("/v1/cases/{case_id}/neuroview")
def get_neuroview(case_id: str):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    is_brain_case = (
        str(b.study_id).startswith("STU-MR")
        or ((b.fusion.backend if b.fusion else "") == "brain-vision-presence-head")
    )
    if not is_brain_case:
        raise HTTPException(404, "NeuroView is available only for Brain MRI cases.")
    payload = store().get_neuroview(case_id)
    if payload is None:
        raise HTTPException(404, "NeuroView data is not available for this Brain MRI case.")
    return payload


@app.post("/v1/cases/{case_id}/feedback")
def feedback(case_id: str, payload: dict = Body(...)):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    verdict = payload.get("verdict", "accept")
    correction = payload.get("correction", "")
    diagnosis = payload.get("diagnosis", b.safety.top.value if b.safety else "")
    store().add_feedback(case_id, diagnosis, verdict, correction)
    store().audit("feedback.recorded", "case", case_id,
                  detail={"verdict": verdict, "correction": correction})

    return {"ok": True, "stats": store().feedback_stats()}


@app.post("/v1/recalibrate")
def trigger_recalibration():
    """Trigger automated calibration update based on clinician feedback."""
    from aura.services.training.recalibrate import recalibrate_from_feedback
    try:
        res = recalibrate_from_feedback(store())
        return res
    except Exception as e:
        raise HTTPException(500, f"Recalibration failed: {e}")


VALID_OUTCOME_SOURCES = {"pcr", "expert_consensus", "biopsy", "pathology", "clinical_course"}


@app.post("/v1/cases/{case_id}/outcome")
def record_outcome(case_id: str, payload: dict = Body(...)):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")

    true_diagnosis = payload.get("true_diagnosis", "")
    source = payload.get("source", "")

    if source not in VALID_OUTCOME_SOURCES:
        raise HTTPException(422, detail={"error": "invalid_source", "source": source})

    try:
        Diagnosis(true_diagnosis)
    except ValueError:
        raise HTTPException(422, detail={"error": "invalid_diagnosis",
                                         "diagnosis": true_diagnosis})

    if store().has_outcome(case_id):
        raise HTTPException(409, detail={"error": "outcome_already_recorded",
                                         "case_id": case_id})

    aci_info = None
    if get_settings().aci_enabled and b.safety is not None:
        # The has_outcome() check above is a courtesy, not the guarantee: it is a
        # read followed by a write, so two concurrent submissions for the same case
        # can both pass it. outcomes.case_id is UNIQUE, so the loser hits an
        # IntegrityError — which used to escape as a 500. Translate it to the same
        # 409 the pre-check returns, so the race and the ordinary duplicate look
        # identical to a caller. Note the ACI step has already run for this request;
        # it is idempotent in effect because the losing row is never committed.
        try:
            aci_info = _record_conformal_outcome(b, true_diagnosis)
        except IntegrityError:
            raise HTTPException(409, detail={"error": "outcome_already_recorded",
                                             "case_id": case_id})

    store().audit("outcome.recorded", "case", case_id,
                  detail={"true_diagnosis": true_diagnosis, "source": source})
    return {"ok": True, "conformal": aci_info}


@app.post("/v1/cases/{case_id}/chat")
async def clinical_chat(case_id: str, payload: ClinicalChatRequest):
    from aura.services.copilot.ollama_client import (
        OllamaCopilotClient,
        OllamaConnectionError,
    )
    from aura.services.copilot.prompt_builder import build_grounding_prompt

    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")

    system_prompt = build_grounding_prompt(b)

    history_dicts = [{"role": m.role, "content": m.content} for m in payload.history]

    client = OllamaCopilotClient()
    try:
        answer, correlation_id = await client.ask_copilot(
            system_prompt=system_prompt,
            user_message=payload.question,
            history=history_dicts,
        )
    except OllamaConnectionError as e:
        raise HTTPException(503, {"error": "ollama_unavailable", "reason": str(e)})

    store().save_chat_message(case_id, "user", payload.question)
    store().save_chat_message(case_id, "assistant", answer)
    store().audit("copilot.chat", "case", case_id,
                  detail={"question": payload.question, "correlation_id": correlation_id})

    sources = _extract_sources(system_prompt)
    return ClinicalChatResponse(answer=answer, sources=sources, correlation_id=correlation_id)


def _extract_sources(prompt: str) -> list[str]:
    sources: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            idx = line.index("]")
            candidate = line[1:idx].strip()
            if candidate not in sources:
                sources.append(candidate)
    return sources


@app.get("/v1/cases/{case_id}/chat")
def get_chat_history(case_id: str):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    return {"case_id": case_id, "messages": store().get_chat_history(case_id)}


def _record_conformal_outcome(bundle, confirmed_diagnosis: str) -> dict | None:
    """Map a confirmed diagnosis to its calibrated posterior + index, run ACI."""
    try:
        true_idx = [d.value for d in DIAGNOSES].index(confirmed_diagnosis)
    except ValueError:
        return None                       # unknown label — skip the update
    # Calibrated posterior AURA emitted for this case, aligned to DIAGNOSES order.
    by_dx = {p.diagnosis: p.probability for p in bundle.safety.predictions}
    probs = [float(by_dx.get(d, 0.0)) for d in DIAGNOSES]
    info = store().record_outcome(
        bundle.case_id, probs, true_idx, true_diagnosis=confirmed_diagnosis
    )
    store().audit("conformal.updated", "case", bundle.case_id,
                  detail={"qhat": info["qhat"], "covered": info["covered"],
                          "localized_coverage": info["localized_coverage"]})
    return info


@app.post("/v1/cases/{case_id}/report/sign")
def sign_report(case_id: str, payload: dict = Body(default={})):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    from aura.schemas.contracts import CaseState
    b.state = CaseState.SIGNED
    
    # Calculate provenance hash
    from .storage import compute_provenance_hash
    p_hash = compute_provenance_hash(b)
    
    store().save_case(b)
    store().audit("report.signed", "case", case_id,
                  actor=payload.get("signed_by", "clinician"),
                  detail={"provenance_hash": p_hash})
    return {"ok": True, "state": b.state.value}


@app.post("/v1/cases/{case_id}/recompute")
async def recompute_case(case_id: str, payload: dict = Body(...)):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
        
    priors_payload = payload.get("priors", {})
    context_payload = payload.get("clinical_context", {})
    
    # 1. Update priors (StructuredPriors)
    for k, v in priors_payload.items():
        if hasattr(b.priors, k):
            setattr(b.priors, k, v)
            
    # 2. Update clinical context
    if b.clinical_context:
        for k, v in context_payload.items():
            if hasattr(b.clinical_context, k):
                setattr(b.clinical_context, k, v)
    else:
        from aura.schemas.contracts import ClinicalContext
        b.clinical_context = ClinicalContext(**context_payload)
        
    # Update multimodal symptoms/history/labs from the priors/context payload if chest case
    if b.multimodal:
        m = b.multimodal
        m.symptoms.fever = b.priors.fever
        if b.priors.smoker:
            m.history.smoking_pack_years = max(m.history.smoking_pack_years or 0.0, 20.0)
        # An unticked "smoker" box means *not currently smoking*, not "never
        # smoked": zeroing the count here erased a pack-year history the intake
        # form had actually recorded for an ex-smoker, and with it the exposure
        # the malignancy rules read.
        m.history.prior_cancer = b.priors.prior_cancer
        m.history.immunosuppression = b.priors.immunocompromised
        
        # Check custom sidebar fields in priors_payload
        if "cough" in priors_payload:
            m.symptoms.productive_cough = priors_payload["cough"]
        if "high_bnp" in priors_payload:
            m.labs.bnp = 450.0 if priors_payload["high_bnp"] else None
        if "crp" in priors_payload or "elevated_crp" in priors_payload:
            val = priors_payload.get("crp", priors_payload.get("elevated_crp", False))
            m.labs.crp = 75.0 if val else None
        if "prior_cardiopathy" in priors_payload:
            m.history.heart_failure = priors_payload["prior_cardiopathy"]
        if "diabetes" in priors_payload:
            m.history.diabetes = priors_payload["diabetes"]
        if "hypertension" in priors_payload:
            m.history.hypertension = priors_payload["hypertension"]
            
    is_brain_case = (
        str(b.study_id).startswith("STU-MR")
        or ((b.fusion.backend if b.fusion else "") == "brain-vision-presence-head")
    )
    
    if is_brain_case:
        # Brain MRI has no clinical reasoning, but we update prior risk score and regenerate report
        # Let's regenerate the report with the new priors
        from aura.backend.engines.neuro.bundle import _findings_text, _impression_text, _recommendation_text, _confidence_text, _recommendations, _priority
        # Determine top/probs
        presence_probability = b.safety.top_probability if (b.safety.top == Diagnosis.BRAIN_TUMOR) else (1.0 - b.safety.top_probability)
        caveats = []
        if b.priors.age_band == "65+":
            caveats.append("degraded sequence details due to microvascular ischemic burden (age 65+)")
            
        b.report.findings_text = _findings_text(None, b.vision.findings, b.volumes, presence_probability, b.safety.abstained)
        b.report.impression_text = _impression_text(b.safety.top, presence_probability, caveats, b.vision.findings, abstained=b.safety.abstained)
        b.report.recommendation_text = _recommendation_text(b.safety.top, b.safety.abstention_reason.value if b.safety.abstained else None)
        b.report.confidence_text = _confidence_text(presence_probability, caveats)
        
        # update recommendations
        b.recommendations = _recommendations(b.safety.top, None, caveats)
        b.priority_score = _priority(b.safety.top, presence_probability, b.safety.abstention_reason if b.safety.abstained else None)
        
    else:
        # Chest case: full re-run of pipeline downstream steps!
        from aura.services.fusion.evidence import encode, to_evidence_items
        img = np.array(b.image, dtype=float).reshape(b.image_shape)
        
        x = encode(b.vision, b.priors)
        b.evidence = to_evidence_items(x, b.priors)
        b.fusion = pipeline().fusion.fuse_vector(x, study_id=b.study_id)
        
        resolved_logits = pipeline().fusion.resolved_logits(x, b.fusion)
        
        # Clinical reasoning
        imaging_probs = pipeline().safety.calibrated_posterior(resolved_logits)
        imaging_prior = {d: float(imaging_probs[i]) for i, d in enumerate(DIAGNOSES)}
        findings_map = {fs.finding: fs.probability for fs in b.vision.findings}
        # reason() returns (trace, graph) — see Pipeline.run step 3. Assigning the
        # tuple straight to b.reasoning made every chest recompute die on
        # `.steps` one line later, so this branch had never completed.
        b.reasoning, b.evidence_graph = pipeline().reasoner.reason(
            b.study_id, findings_map, imaging_prior, b.priors, b.multimodal
        )

        final_posterior = None
        if b.reasoning.steps:
            final_posterior = np.array(
                [b.reasoning.adjusted_posterior.get(d, 0.0) for d in DIAGNOSES], dtype=float
            )
            
        # Safety
        b.safety = pipeline().safety.assess(
            b.study_id, x, pipeline().fusion.model,
            resolved_logits=resolved_logits, aci_qhat=pipeline()._aci_qhat(),
            final_posterior=final_posterior,
        )
        
        # Explainability (wires SHAP/Shapley)
        b.explanation = pipeline().explain.explain(
            b.study_id, pipeline().vision, img, pipeline().fusion.model, x, b.safety.top
        )
        
        # Recommendations
        b.recommendations = pipeline().recommend.recommend(pipeline().fusion.model, x)
        
        # Multi-agent consensus panel — re-run rather than reused, because the
        # specialists read the labs and symptoms that just changed.
        mm = b.multimodal
        agent_evidence = {
            "findings": findings_map,
            "embedding": (b.vision.embedding.tolist()
                          if hasattr(b.vision.embedding, "tolist") else list(b.vision.embedding or [])),
            "labs": mm.labs.model_dump() if mm and mm.labs else {},
            "symptoms": mm.symptoms.model_dump() if mm and mm.symptoms else {},
        }
        consensus_result = await pipeline().consensus.evaluate(
            evidence=agent_evidence,
            priors=b.priors.model_dump(),
            modality=Modality.CXR,
        )
        b.consensus_result = consensus_result.to_dict() if consensus_result else None

        # CDRE. The engine's entry point is evaluate(); the assess() called here
        # does not exist on ClinicalDecisionReadinessEngine, and the report was
        # handed b.fusion where compose() expects the ConsensusResult it reads
        # `.verdicts` off — both raised on every chest recompute.
        classical_logits = pipeline().fusion.model.logits(x) if hasattr(pipeline().fusion, "model") else None
        quantum_logits = None
        if getattr(pipeline().fusion, "_quantum_model", None) is not None:
            quantum_logits = pipeline().fusion._quantum_model.logits(x)
        b.decision_readiness = pipeline().cdre.evaluate(
            reasoning=b.reasoning,
            evidence_graph=b.evidence_graph,
            recommendations=b.recommendations,
            vision_quality=None,
            fusion_model=pipeline().fusion.model,
            evidence_vector=x,
            classical_logits=classical_logits,
            quantum_logits=quantum_logits,
            consensus_result=consensus_result,
        )
        b.drp = b.decision_readiness

        # Report
        b.report = pipeline().report.compose(b.vision, b.safety, b.recommendations,
                                             b.reasoning, consensus_result)
        b.priority_score = pipeline()._priority(b.safety.top, b.safety)
        b.state = CaseState.ABSTAINED if b.safety.abstained else CaseState.READY
        
    store().save_case(b)
    store().audit("case.recomputed", "case", case_id, detail={"priors": priors_payload})
    return b


@app.get("/v1/cases/{case_id}/discussion")
def get_case_discussion(case_id: str):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    disc = SpecialistDiscussionEngine.discuss(b)
    store().audit("case.discussion_generated", "case", case_id)
    return disc


@app.get("/v1/cases/{case_id}/export/fhir")
def get_fhir_export(case_id: str):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    report = export_fhir_diagnostic_report(b)
    observations = export_fhir_observations(b)
    store().audit("case.exported_fhir", "case", case_id)
    return {"diagnostic_report": report, "observations": observations}


@app.get("/v1/cases/{case_id}/export/hl7")
def get_hl7_export(case_id: str):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    hl7_msg = export_hl7_oru_r01(b)
    store().audit("case.exported_hl7", "case", case_id)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(hl7_msg, media_type="text/plain")


@app.post("/v1/studies/simulate")
async def simulate_study(payload: dict = Body(default={})):
    """Disabled in production. AURA analyzes only real uploaded radiographs — it never
    fabricates synthetic patient studies, findings, or diagnoses. Use the upload path.
    """
    raise HTTPException(410, {
        "error": "synthetic_studies_disabled",
        "reason": "AURA runs real inference on uploaded chest radiographs only; "
                  "synthetic study generation has been removed. Use POST /v1/studies/upload.",
    })


@app.post("/v1/studies/upload")
async def upload_study(request: Request, file: UploadFile = File(...)):
    """Upload a chest radiograph (PNG/JPG/DICOM) and analyze it live.

    The X-ray intake gate runs first: anything that is not a chest radiograph is
    rejected with 422 and a named reason — no case is created for junk uploads.
    Valid films go through the full trained pipeline; if the film sits outside
    the training distribution the safety engine abstains rather than guessing.
    """
    import tempfile
    import os
    from .security import validate_upload_name, read_capped, validate_mri_content, get_tenant_id

    # Type allowlist (extension + declared MIME) and a hard size cap enforced while
    # streaming, so a hostile upload can neither smuggle a non-image type nor
    # exhaust memory (audit §11.5). These layer in front of the content-based gate.
    validate_upload_name(file.filename, file.content_type)
    max_bytes = int(get_settings().max_upload_mb * 1024 * 1024)
    payload = await read_capped(file, max_bytes)
    validate_mri_content(payload, file.filename)

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    try:
        from aura.services.vision.xray_gate import validate_cxr
        gate = validate_cxr(tmp_path)
        if not gate.ok:
            store().audit("case.upload_rejected", "study", file.filename or "upload",
                          detail={"reason": gate.reason})
            raise HTTPException(422, {"error": "not_a_cxr", "reason": gate.reason,
                                      "checks": gate.checks})

        from aura.services.vision.io import study_from_cxr
        # Full 224 fidelity through the pipeline (same resolution the DenseNet trains at),
        # not the 64-px thumbnail default.
        study = study_from_cxr(tmp_path, grid=224)

        idx = store().count() + 1
        case_id = f"CASE-UPLOAD-{idx}"
        study.study_id = f"STU-UPLOAD-{idx}"

        import time as _time
        _t0 = _time.perf_counter()
        from aura.services.safety import ClinicalSafetyException
        try:
            bundle = await pipeline().run(study, case_id=case_id)
        except ClinicalSafetyException as exc:
            store().audit("case.upload_rejected", "study", file.filename or "upload",
                          detail={"reason": exc.reason})
            raise HTTPException(422, exc.to_payload())
        infer_s = _time.perf_counter() - _t0
        tenant_id = get_tenant_id(request)
        store().save_case(bundle, tenant_id=tenant_id)
        session_case_ids.append(case_id)
        store().audit("case.uploaded", "case", case_id,
                      actor=getattr(request.state, "principal", "system"),
                      tenant_id=tenant_id,
                      detail={"top": bundle.safety.top.value,
                              "abstained": bundle.safety.abstained,
                              "gate_checks": gate.checks})
        # Production audit trail: full-provenance record per real upload (req 8).
        try:
            from aura.services.inference.audit_log import log_inference
            log_inference(bundle, tmp_path, infer_s,
                          backbone=getattr(pipeline().vision, "backbone", None))
        except Exception as _e:
            print(f"[upload] inference logging failed: {_e!r}")

        return {"case_id": case_id, "inference_time_s": round(infer_s, 4)}
    except HTTPException:
        raise
    except Exception as e:
        # Never echo internal exception text (it can leak filesystem paths). Log
        # server-side; return an opaque error id to the client (audit §10.9/11.5).
        print(f"[upload] processing failed for {file.filename!r}: {e!r}")
        raise HTTPException(500, {"error": "processing_failed",
                                  "reason": "the uploaded image could not be processed"})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/v1/studies/agent")
async def run_agent_study(
    file: UploadFile = File(...),
    entropy_target: float = 0.6,
    confidence: float = 0.85,
    max_tests: int = 3
):
    """Upload a chest radiograph (PNG/JPG/DICOM) and run the sequential Active Diagnosis Agent.

    The X-ray intake gate runs first: non-radiographs are rejected.
    Returns the step-by-step diagnostic trajectory showing information gain and decisions.
    """
    import tempfile
    import os
    from .security import validate_upload_name, read_capped, validate_mri_content

    validate_upload_name(file.filename, file.content_type)
    max_bytes = int(get_settings().max_upload_mb * 1024 * 1024)
    payload = await read_capped(file, max_bytes)
    validate_mri_content(payload, file.filename)

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    try:
        from aura.services.vision.xray_gate import validate_cxr
        gate = validate_cxr(tmp_path)
        if not gate.ok:
            store().audit("agent.upload_rejected", "study", file.filename or "upload",
                          detail={"reason": gate.reason})
            raise HTTPException(422, {"error": "not_a_cxr", "reason": gate.reason,
                                       "checks": gate.checks})

        from aura.services.vision.io import study_from_cxr
        # Full 224 fidelity through the pipeline
        study = study_from_cxr(tmp_path, grid=224)

        img = np.array(study.image, dtype=float).reshape(study.image_shape)
        
        # Run Vision
        vision_result = pipeline().vision.analyze(study.study_id, img)
        
        # Encode evidence
        from aura.services.fusion.evidence import encode
        x = encode(vision_result, study.priors)
        
        # Run agent
        from aura.services.agent.active_diagnosis import ActiveDiagnosisAgent
        agent = ActiveDiagnosisAgent(
            fusion_model=pipeline().fusion,
            entropy_target_bits=entropy_target,
            confidence=confidence,
            max_tests=max_tests
        )
        trajectory = agent.diagnose(x)
        
        # Audit
        store().audit("agent.run", "study", study.study_id,
                      detail={"status": trajectory.status, "final_dx": trajectory.final_diagnosis})
        
        # Serialize and return
        return {
            "committed": trajectory.committed,
            "status": trajectory.status,
            "final_diagnosis": trajectory.final_diagnosis,
            "final_probability": float(trajectory.final_probability),
            "initial_entropy": float(trajectory.initial_entropy),
            "final_entropy": float(trajectory.final_entropy),
            "bits_resolved": float(trajectory.bits_resolved),
            "n_tests": int(trajectory.n_tests),
            "backend": str(trajectory.backend),
            "steps": [
                {
                    "step": int(s.step),
                    "posterior": {dx.value if hasattr(dx, "value") else str(dx): float(p) for dx, p in s.posterior.items()},
                    "entropy_bits": float(s.entropy_bits),
                    "top": [[dx.value if hasattr(dx, "value") else str(dx), float(p)] for dx, p in s.top],
                    "confident": bool(s.confident),
                    "decision": s.decision,
                    "action_display": s.action_display,
                    "action_eig_bits": float(s.action_eig_bits) if s.action_eig_bits is not None else None,
                    "resolved": [[c, float(v)] for c, v in s.resolved],
                    "rationale": s.rationale,
                }
                for s in trajectory.steps
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[agent endpoint] processing failed for {file.filename!r}: {e!r}")
        raise HTTPException(500, {"error": "processing_failed",
                                  "reason": "the uploaded image could not be processed"})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass



@app.get("/v1/cases/{case_id}/similar")
def similar(case_id: str):
    b = store().get_case(case_id)
    if b is None or b.vision is None:
        raise HTTPException(404, "case not found")
    sims = pipeline().memory.similar(b.vision.embedding, k=3, exclude=case_id)
    return {"similar": sims}


@app.get("/v1/models")
def models():
    chest_versions = state["chest_registry"].list_versions()
    brain_versions = state["brain_registry"].list_versions()
    return {
        "chest": chest_versions,
        "brain": brain_versions,
        "versions": chest_versions + brain_versions
    }


@app.get("/v1/model-card")
def model_card():
    """Real, measured model metadata — read straight from artifacts. Never fabricated:
    any source that hasn't been produced returns null so the UI shows '—', not a
    placeholder number. Evaluation numbers appear only after `aura_cli evaluate` runs."""
    import hashlib
    from aura.common.config import ARTIFACTS
    from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
    labels = {}
    for d, l in DIAGNOSIS_LABELS.items():
        labels[d.value] = l
    for f, l in FINDING_LABELS.items():
        labels[f.value] = l

    card: dict = {
        "model_version": None,
        "model_file": "artifacts/best_model.pt",
        "dataset": "MIMIC-CXR validation split",
        "calibration": None,
        "evaluation": None,
        "inference": None,
        "labels": labels,
    }
    try:
        bb = getattr(pipeline().vision, "backbone", None)
        card["model_version"] = (bb.model_version if bb is not None
                                 else pipeline().vision.model_version)
    except Exception:
        pass
    try:
        bm = ARTIFACTS / "best_model.pt"
        if bm.exists():
            import torch
            ck = torch.load(bm, map_location="cpu", weights_only=False)
            card["model_epoch"] = ck.get("epoch")
            card["model_val_metric"] = ck.get("best_metric")
    except Exception:
        pass

    calp = ARTIFACTS / "vision_serving_calibration.json"
    if calp.exists():
        try:
            raw = calp.read_bytes(); d = json.loads(raw)
            card["calibration"] = {
                "method": d.get("method"),
                "n_images_fit": d.get("n_images"),
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()[:16],
                "mean_ece_after": d.get("mean_ece_after"),
            }
        except Exception:
            pass

    evp = ARTIFACTS / "evaluation" / "metrics.json"
    if evp.exists():
        try:
            m = json.loads(evp.read_text())
            card["evaluation"] = {
                "evaluated_model": m.get("model_path"),
                "n_images": m.get("n_images"),
                "macro_auroc": m.get("macro", {}).get("auroc"),
                "macro_auroc_ci95": m.get("bootstrap_ci", {}).get("macro_auroc_ci95"),
                "macro_ece": m.get("macro", {}).get("ece"),
                "macro_f1": m.get("macro", {}).get("f1"),
                "per_label_auroc": {k: v.get("auroc") for k, v in m.get("per_label", {}).items()},
            }
        except Exception:
            pass

    logp = ARTIFACTS / "inference_log.jsonl"
    if logp.exists():
        try:
            lines = [l for l in logp.read_text(encoding="utf-8").splitlines() if l.strip()]
            times = []
            for l in lines[-500:]:
                try:
                    t = json.loads(l).get("inference_time_s")
                    if t is not None:
                        times.append(float(t))
                except Exception:
                    pass
            card["inference"] = {
                "count": len(lines),
                "mean_time_s": round(sum(times) / len(times), 3) if times else None,
            }
        except Exception:
            pass

    # ----- Label provenance: how faithful are the report-derived labels the AUROC is
    # measured against? Pre-empts the circularity objection with two measured artifacts
    # (labeler_validation.json, kappa_crosscheck.json). Null when a source is absent —
    # never a placeholder. Produced by scratchpad scoring scripts / aura_cli.
    prov: dict = {}
    lv = ARTIFACTS / "labeler_validation.json"
    if lv.exists():
        try:
            d = json.loads(lv.read_text())
            gm = d.get("v2_macro_average", {})
            gp = d.get("gold_provenance", {})
            prov["labeler"] = "rule-based report labeler (mimic/labeling_v2.py)"
            prov["is_official_chexpert_labeler"] = False
            prov["gold_n_reports"] = gp.get("n_reports")
            prov["gold_annotator"] = gp.get("annotator")
            prov["gold_convention"] = gp.get("label_convention")
            prov["labeler_vs_gold"] = {
                "macro_precision": gm.get("macro_precision"),
                "macro_recall": gm.get("macro_recall"),
                "macro_f1": gm.get("macro_f1"),
                "macro_cohen_kappa": gm.get("macro_kappa"),
                "per_finding": {k: {"precision": v.get("precision"), "recall": v.get("recall"),
                                    "f1": v.get("f1_score"), "kappa": v.get("cohen_kappa"),
                                    "gold_positives": v.get("gold_positives")}
                                for k, v in d.get("v2_metrics_per_finding", {}).items()},
            }
        except Exception:
            pass
    kc = ARTIFACTS / "kappa_crosscheck.json"
    if kc.exists():
        try:
            d = json.loads(kc.read_text())
            mac = d.get("macro", {})
            prov["cross_model_check"] = {
                "reference_model": "torchxrayvision densenet121-res224-mimic_ch (independently labelled)",
                "n_images": d.get("provenance", {}).get("n_images"),
                "mean_cross_auroc": mac.get("mean_auroc_aura_vs_xrv"),
                "mean_spearman_rho": mac.get("mean_spearman_rho"),
                "macro_kappa_prevalence_matched": mac.get("mean_cohen_kappa"),
                "per_finding": {k: {"cross_auroc": v.get("auroc_aura_score_vs_xrv_label"),
                                    "kappa": v.get("cohen_kappa_prevalence_matched")}
                                for k, v in d.get("per_finding", {}).items()},
                "caveat": "nodule shows below-chance cross-model agreement — corroborates its disclosed unreliability.",
            }
        except Exception:
            pass
    card["label_provenance"] = prov or None
    return card


@app.get("/v1/admin/safety")
def admin_safety():
    reg = state["registry"].list_versions()
    bench = {}
    bpath = Path(get_settings_artifacts()) / "benchmark.json"
    if bpath.exists():
        bench = json.loads(bpath.read_text())
    return {
        "registry": reg,
        "benchmark": bench,
        "feedback": store().feedback_stats(),
        "abstention_rate": _abstention_rate(),
        "recent_audit": store().recent_audit(20),
    }


def get_settings_artifacts() -> str:
    from aura.common.config import ARTIFACTS
    return str(ARTIFACTS)


def _abstention_rate() -> float:
    rows = store().list_cases()
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r["abstained"]) / len(rows), 4)


# --------------------------------------------------------------------------- #
# Dashboard (static SPA)
# --------------------------------------------------------------------------- #
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    idx = WEB_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "AURA gateway up. Dashboard not built."})


@app.get("/app")
def console_route():
    """Deep link into the console — same SPA, which boots straight into /app."""
    return index()


@app.get("/history")
def history_route():
    """Deep link into the history & report portal."""
    idx = WEB_DIR / "history.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "History page not found."}, status_code=404)


@app.get("/quantum")
def quantum_route():
    """Quantum evidence view.

    System-level, not per-case: it renders what the quantum layer is, what it costs
    on real hardware, and what it does not buy. Deliberately a separate view rather
    than a fourth console tab, because the console pager is three views over *one
    case* and this data belongs to the model, not to a study.

    The shell carries no patient data — every figure comes from a subsequent fetch of
    /v1/quantum/evidence, which is gated like every other /v1 route.
    """
    idx = WEB_DIR / "quantum.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "Quantum evidence page not found."}, status_code=404)
