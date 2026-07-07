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
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from common.config import DB_PATH, ensure_dirs, get_settings
from ml.data import IMG, make_sample
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
    pipeline = Pipeline()
    state["store"] = store
    state["pipeline"] = pipeline
    state["registry"] = ModelRegistry()
    if not pipeline.fusion.is_trained():
        print("[gateway] WARNING: fusion model not trained; run `aura_cli train`.")
    n = await seed(store, pipeline)
    print(f"[gateway] ready — {store.count()} cases in worklist "
          f"(fusion backend: {pipeline.fusion.backend}).")
    yield


app = FastAPI(title="AURA Clinical Intelligence Copilot", version="0.1.0",
              lifespan=lifespan)


@app.middleware("http")
async def audit_mw(request: Request, call_next):
    resp = await call_next(request)
    if request.method in ("POST", "PUT", "DELETE") and "store" in state:
        try:
            state["store"].audit(
                action=f"{request.method} {request.url.path}",
                actor=request.headers.get("x-aura-user", "anonymous"),
                entity_type="http",
            )
        except Exception:
            pass
    return resp


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
    return {"ok": True, "stats": store().feedback_stats()}


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
        image_shape=(IMG, IMG), priors=s.priors, ground_truth=s.diagnosis,
    )
    case_id = f"CASE-LIVE-{idx}"
    bundle = await pipeline().run(study, case_id=case_id)
    store().save_case(bundle)
    store().audit("case.analyzed", "case", case_id,
                  detail={"top": bundle.safety.top.value})
    return {"case_id": case_id}


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
