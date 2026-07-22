"""End-to-end single-image prediction (Step 7/10 ``predict`` and ``explain``).

Loads a real radiograph, runs it through the *existing* pipeline (vision → fusion →
safety → explain → recommend → reasoning → report), times the run, renders the full
clinical report, and writes high-resolution saliency overlays + bounding boxes + an
HTML explainability report. Nothing here changes an engine or a schema — it is an
orchestration layer over what already exists, used by the CLI.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np

from common.config import ARTIFACTS
from schemas.clinical import FINDING_LABELS, Finding
from schemas.contracts import StructuredPriors, StudyInput
from services.report.clinical_report import build_clinical_report, save_report
from services.vision.io import load_cxr, study_from_cxr

# The CNN resizes internally to 224; feeding a 224 grid preserves image fidelity
# through the pipeline while keeping the stored bundle image reasonably small.
_PREDICT_GRID = 224


def _run_pipeline_sync(pipeline, study: StudyInput, case_id: str):
    """Run the async pipeline from sync code, whether or not a loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:                       # pragma: no cover - not hit from CLI
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(lambda: asyncio.run(pipeline.run(study, case_id))).result()
    return asyncio.run(pipeline.run(study, case_id))


def _hi_res_size(full: np.ndarray, cap: int = 512) -> int:
    return int(min(cap, max(128, min(full.shape[:2]))))


def predict_image(
    image_path: str | Path,
    pipeline=None,
    priors: Optional[StructuredPriors] = None,
    out_dir: Optional[str | Path] = None,
    save_explain: bool = True,
    save_reports: bool = True,
    include_scorecam: bool = True,
    study_id: Optional[str] = None,
) -> dict:
    """Predict on one radiograph and return a structured result dict.

    The returned dict has: ``findings``, ``clinical_report`` (full structured report),
    ``inference_time_s``, and ``artifacts`` (paths to overlays / html / json written).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    if pipeline is None:
        from gateway.pipeline import Pipeline

        pipeline = Pipeline()

    stem = study_id or image_path.stem
    out_dir = Path(out_dir) if out_dir else (ARTIFACTS / "predictions" / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the study at full CNN fidelity and time the whole pipeline.
    study = study_from_cxr(image_path, study_id=stem, priors=priors, grid=_PREDICT_GRID)
    t0 = time.perf_counter()
    bundle = _run_pipeline_sync(pipeline, study, case_id=f"CASE-{stem}")
    inference_time = time.perf_counter() - t0

    # Full structured clinical report (Step 5), grounded in the finished bundle.
    calibration = getattr(pipeline.safety, "cal", None)
    report = build_clinical_report(bundle, inference_time_s=inference_time, calibration=calibration)

    artifacts: dict[str, str] = {}
    if save_reports:
        paths = save_report(report, out_dir, stem="clinical_report")
        artifacts.update({f"report_{k}": str(v) for k, v in paths.items()})

    if save_explain:
        try:
            artifacts.update(
                _write_explanations(pipeline, bundle, image_path, out_dir, include_scorecam)
            )
        except Exception as e:  # explanation is best-effort; never sink a prediction
            print(f"[predict] explanation artifacts failed: {e}")

    # Production audit trail: one immutable record per real inference (req 8).
    try:
        from services.inference.audit_log import log_inference
        log_inference(bundle, image_path, inference_time,
                      backbone=getattr(pipeline.vision, "backbone", None))
    except Exception as e:
        print(f"[predict] inference logging failed: {e!r}")

    from common.config import finding_present_threshold
    findings = [
        {"finding": FINDING_LABELS.get(fs.finding, fs.finding.value),
         "key": fs.finding.value, "probability": round(float(fs.probability), 4),
         "threshold": round(finding_present_threshold(fs.finding.value), 3),
         "present": bool(fs.probability >= finding_present_threshold(fs.finding.value))}
        for fs in (bundle.vision.findings if bundle.vision else [])
    ]

    return {
        "study_id": stem,
        "findings": findings,
        "top_diagnosis": report["confidence"].get("top_diagnosis"),
        "top_probability": report["confidence"].get("top_probability"),
        "risk_level": report["risk_level"],
        "clinical_report": report,
        "inference_time_s": round(inference_time, 4),
        "artifacts": artifacts,
        "bundle": bundle,
    }


def _write_explanations(pipeline, bundle, image_path, out_dir, include_scorecam) -> dict[str, str]:
    """Compute high-resolution saliency (incl. Score-CAM) and write PNG/HTML overlays."""
    from services.explain import methods as M
    from services.explain import overlays as O

    full = load_cxr(image_path)
    size = _hi_res_size(full)
    backbone = getattr(pipeline.vision, "backbone", None)

    # Choose the target finding (highest-probability positive, else highest).
    scores = {fs.finding: fs.probability for fs in bundle.vision.findings}
    target = max(scores, key=scores.get)

    if backbone is not None:
        maps = M.all_methods(backbone, full, target, out_size=size,
                             include_scorecam=include_scorecam)
        primary = "grad_cam++" if "grad_cam++" in maps else next(iter(maps), None)
    else:
        eng = pipeline.explain
        sal = eng.occlusion_saliency(pipeline.vision, full)
        maps = {"occlusion": sal}
        primary = "occlusion"

    if not maps:
        return {}

    boxes = O.heatmap_bboxes(maps[primary], thresh_rel=0.5)
    findings_pairs = [
        (FINDING_LABELS.get(fs.finding, fs.finding.value), float(fs.probability))
        for fs in bundle.vision.findings
    ]
    ev_attr = bundle.explanation.evidence_attribution if bundle.explanation else {}

    written: dict[str, str] = {}
    # Primary overlay PNG + high-res overlay.
    written["overlay_png"] = str(O.save_overlay_png(
        out_dir / "overlay.png", full, maps[primary], boxes, f"{primary} · {target.value}"))
    written["overlay_hires_png"] = str(O.save_high_res_overlay(
        out_dir / "overlay_hires.png", full, maps[primary], boxes))
    written["heatmap_png"] = str(O.save_raw_heatmap_png(out_dir / "heatmap.png", maps[primary]))
    if ev_attr:
        written["evidence_png"] = str(O.save_evidence_bar_png(out_dir / "evidence.png", ev_attr))
    # Per-method overlays.
    for name, m in maps.items():
        written[f"method_{name}"] = str(O.save_overlay_png(
            out_dir / f"method_{name}.png", full, m,
            boxes if name == primary else None, name))
    # HTML report bundling everything.
    written["explanation_html"] = str(O.save_explanation_html(
        out_dir / "explanation.html", full, maps, primary, boxes, target.value,
        findings_pairs, evidence_attribution=ev_attr,
        meta={"study_id": bundle.study_id, "model": bundle.vision.model_version,
              "target_finding": target.value, "n_boxes": len(boxes)}))
    return written
