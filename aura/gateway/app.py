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
session_case_ids: list[str] = []


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
    # PRODUCTION: Seeding has been disabled. The worklist starts empty.
    # Only newly uploaded images appear in the active session.
    import os
    source = os.environ.get("AURA_DATA_SOURCE", "mimic").lower()
    print(f"[gateway] ready — 0 cases in active session (worklist empty on startup, fusion backend: {pipeline.fusion.backend}).")
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


@app.get("/v1/studies")
def list_studies():
    out = []
    from schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
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
    from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
    dx_lbls = {d.value: l for d, l in DIAGNOSIS_LABELS.items()}
    ev_lbls = {f.value: l for f, l in FINDING_LABELS.items()}
    return {
        "cases": out,
        "dx_labels": dx_lbls,
        "ev_labels": ev_lbls
    }


@app.get("/v1/cases")
def list_cases(state_filter: str | None = None):
    from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
    dx_lbls = {d.value: l for d, l in DIAGNOSIS_LABELS.items()}
    ev_lbls = {f.value: l for f, l in FINDING_LABELS.items()}
    return {
        "cases": store().list_cases(state=state_filter),
        "dx_labels": dx_lbls,
        "ev_labels": ev_lbls
    }


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str):
    b = store().get_case(case_id)
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
    from common.config import finding_present_threshold
    vis = d.get("vision") or {}
    for f in (vis.get("findings") or []):
        try:
            thr = finding_present_threshold(f["finding"])
            f["threshold"] = round(float(thr), 4)
            f["present"] = bool(float(f["probability"]) >= thr)
        except Exception:
            pass
    return d


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
    """Disabled in production. AURA analyzes only real uploaded radiographs — it never
    fabricates synthetic patient studies, findings, or diagnoses. Use the upload path.
    """
    raise HTTPException(410, {
        "error": "synthetic_studies_disabled",
        "reason": "AURA runs real inference on uploaded chest radiographs only; "
                  "synthetic study generation has been removed. Use POST /v1/studies/upload.",
    })


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
        # Full 224 fidelity through the pipeline (same resolution the DenseNet trains at),
        # not the 64-px thumbnail default.
        study = study_from_cxr(tmp_path, grid=224)

        idx = store().count() + 1
        case_id = f"CASE-UPLOAD-{idx}"
        study.study_id = f"STU-UPLOAD-{idx}"

        import time as _time
        _t0 = _time.perf_counter()
        bundle = await pipeline().run(study, case_id=case_id)
        infer_s = _time.perf_counter() - _t0
        store().save_case(bundle)
        session_case_ids.append(case_id)
        store().audit("case.uploaded", "case", case_id,
                      detail={"top": bundle.safety.top.value,
                              "abstained": bundle.safety.abstained,
                              "gate_checks": gate.checks})
        # Production audit trail: full-provenance record per real upload (req 8).
        try:
            from services.inference.audit_log import log_inference
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
    from gateway.security import validate_upload_name, read_capped

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
            store().audit("agent.upload_rejected", "study", file.filename or "upload",
                          detail={"reason": gate.reason})
            raise HTTPException(422, {"error": "not_a_cxr", "reason": gate.reason,
                                       "checks": gate.checks})

        from services.vision.io import study_from_cxr
        # Full 224 fidelity through the pipeline
        study = study_from_cxr(tmp_path, grid=224)

        img = np.array(study.image, dtype=float).reshape(study.image_shape)
        
        # Run Vision
        vision_result = pipeline().vision.analyze(study.study_id, img)
        
        # Encode evidence
        from services.fusion.evidence import encode
        x = encode(vision_result, study.priors)
        
        # Run agent
        from services.agent.active_diagnosis import ActiveDiagnosisAgent
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
    return {"versions": state["registry"].list_versions()}


@app.get("/v1/model-card")
def model_card():
    """Real, measured model metadata — read straight from artifacts. Never fabricated:
    any source that hasn't been produced returns null so the UI shows '—', not a
    placeholder number. Evaluation numbers appear only after `aura_cli evaluate` runs."""
    import hashlib
    from common.config import ARTIFACTS
    from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
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

