"""FastAPI gateway: /v1 API + dashboard hosting + audit middleware.

Auth is stubbed for the P0 demo (a header-based principal); the RBAC/OIDC seam is
marked so production auth drops in at the same boundary. See docs/ARCHITECTURE.md
sections 13 & 15.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from common.config import DB_PATH, ensure_dirs, get_settings
from ml.data import IMG, make_multimodal, make_sample
from schemas.clinical import DIAGNOSES, Diagnosis
from schemas.contracts import StudyInput, StructuredPriors
from services.models import ModelRegistry
from gateway.pipeline import Pipeline
from gateway.seed import seed
from gateway.storage import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "apps" / "web"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    store = Store(DB_PATH)
    pipeline = Pipeline(store=store)          # store handle enables online ACI (F9)
    state["store"] = store
    state["pipeline"] = pipeline
    state["registry"] = ModelRegistry()
    if not pipeline.fusion.is_trained():
        print("[gateway] WARNING: fusion model not trained; run `aura_cli train`.")
    # Data source: real MIMIC-CXR by default; set AURA_DATA_SOURCE=synthetic to
    # fall back to the legacy synthetic seeder (kept for offline/dev use).
    import os
    source = os.environ.get("AURA_DATA_SOURCE", "mimic").lower()
    if source == "mimic":
        try:
            from mimic.seed import seed_mimic
            n = await seed_mimic(store, pipeline)
            if n == 0:                       # corpus absent -> graceful fallback
                print("[gateway] MIMIC-CXR unavailable; falling back to synthetic seed.")
                n = await seed(store, pipeline)
        except Exception as e:               # never block startup on data issues
            print(f"[gateway] MIMIC seed failed ({e}); falling back to synthetic.")
            n = await seed(store, pipeline)
    else:
        n = await seed(store, pipeline)
    print(f"[gateway] ready — {store.count()} cases in worklist "
          f"(source: {source}, fusion backend: {pipeline.fusion.backend}).")
    yield


app = FastAPI(title="AURA Clinical Intelligence Copilot", version="0.1.0",
              lifespan=lifespan)


@app.middleware("http")
async def audit_mw(request: Request, call_next):
    # Security gate runs *before* the handler for mutating methods: opt-in auth,
    # authorization, and rate limiting (all inert unless configured). A rejection
    # here is itself audited, then returned, so blocked calls are attributable.
    if request.method in ("POST", "PUT", "DELETE"):
        from gateway.security import enforce
        try:
            enforce(request)
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
        _safe_audit(action=f"{request.method} {request.url.path}",
                    actor=request.headers.get("x-aura-user", "anonymous"),
                    entity_type="http")
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
    return {"status": "ok", "backend": pipeline().fusion.backend,
            "trained": pipeline().fusion.is_trained(),
            "cases": store().count()}


@app.get("/v1/cases")
def list_cases(state_filter: str | None = None):
    return {"cases": store().list_cases(state=state_filter)}


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str):
    b = store().get_case(case_id)
    if b is None:
        raise HTTPException(404, "case not found")
    return json.loads(b.model_dump_json())


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

    # Module 8: fold the confirmed outcome into the online conformal threshold so
    # coverage self-corrects under covariate shift. Runs on the local SQLite log.
    aci_info = None
    if get_settings().aci_enabled and b.safety is not None:
        aci_info = _record_conformal_outcome(b, diagnosis)

    return {"ok": True, "stats": store().feedback_stats(), "conformal": aci_info}


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
    from schemas.contracts import CaseState
    b.state = CaseState.SIGNED
    store().save_case(b)
    store().audit("report.signed", "case", case_id,
                  actor=payload.get("signed_by", "clinician"))
    return {"ok": True, "state": b.state.value}


@app.post("/v1/studies/simulate")
async def simulate_study(payload: dict = Body(default={})):
    """Generate a fresh synthetic study of a chosen diagnosis and analyze it live —
    powers the dashboard's 'new study' button so the pipeline can be demoed on demand.
    """
    dx_name = payload.get("diagnosis", "random")
    rng = np.random.default_rng()
    if dx_name == "random":
        dx = DIAGNOSES[int(rng.integers(len(DIAGNOSES)))]
    else:
        try:
            dx = Diagnosis(dx_name)
        except ValueError:
            raise HTTPException(400, f"unknown diagnosis {dx_name}")
    s = make_sample(dx, rng)
    idx = store().count() + 1
    study = StudyInput(
        study_id=f"STU-LIVE-{idx}", image=[float(v) for v in s.image.flatten()],
        image_shape=(IMG, IMG), priors=s.priors,
        multimodal=make_multimodal(s.diagnosis, rng), ground_truth=s.diagnosis,
    )
    case_id = f"CASE-LIVE-{idx}"
    bundle = await pipeline().run(study, case_id=case_id)
    store().save_case(bundle)
    store().audit("case.analyzed", "case", case_id,
                  detail={"top": bundle.safety.top.value})
    return {"case_id": case_id}


@app.post("/v1/studies/upload")
async def upload_study(file: UploadFile = File(...)):
    """Upload a chest radiograph (PNG/JPG/DICOM) and analyze it live.

    The X-ray intake gate runs first: anything that is not a chest radiograph is
    rejected with 422 and a named reason — no case is created for junk uploads.
    Valid films go through the full trained pipeline; if the film sits outside
    the training distribution the safety engine abstains rather than guessing.
    """
    import tempfile
    import os
    from gateway.security import validate_upload_name, read_capped

    # Type allowlist (extension + declared MIME) and a hard size cap enforced while
    # streaming, so a hostile upload can neither smuggle a non-image type nor
    # exhaust memory (audit §11.5). These layer in front of the content-based gate.
    validate_upload_name(file.filename, file.content_type)
    max_bytes = int(get_settings().max_upload_mb * 1024 * 1024)
    payload = await read_capped(file, max_bytes)

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    try:
        from services.vision.xray_gate import validate_cxr
        gate = validate_cxr(tmp_path)
        if not gate.ok:
            store().audit("case.upload_rejected", "study", file.filename or "upload",
                          detail={"reason": gate.reason})
            raise HTTPException(422, {"error": "not_a_cxr", "reason": gate.reason,
                                      "checks": gate.checks})

        from services.vision.io import study_from_cxr
        study = study_from_cxr(tmp_path)

        idx = store().count() + 1
        case_id = f"CASE-UPLOAD-{idx}"
        study.study_id = f"STU-UPLOAD-{idx}"

        bundle = await pipeline().run(study, case_id=case_id)
        store().save_case(bundle)
        store().audit("case.uploaded", "case", case_id,
                      detail={"top": bundle.safety.top.value,
                              "abstained": bundle.safety.abstained,
                              "gate_checks": gate.checks})

        return {"case_id": case_id}
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



@app.get("/v1/cases/{case_id}/similar")
def similar(case_id: str):
    b = store().get_case(case_id)
    if b is None or b.vision is None:
        raise HTTPException(404, "case not found")
    sims = pipeline().memory.similar(b.vision.embedding, k=3, exclude=case_id)
    return {"similar": sims}


@app.get("/v1/models")
def models():
    return {"versions": state["registry"].list_versions()}


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
    from common.config import ARTIFACTS
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

