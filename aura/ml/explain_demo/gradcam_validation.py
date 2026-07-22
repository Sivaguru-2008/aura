"""Task 1 — Visual Explainability validation for the served retrain_v2 model.

What this does (all on REAL MIMIC-CXR validation data, v2 labels):

  1. Loads the served DenseNet121 (artifacts/best_model.pt) via the same
     ``VisionModel`` the app serves, Grad-CAM++ target = densenet.features.norm5.
  2. Scores the full per-study validation set (calibrated probs) and picks real
     True-Positive / False-Positive / False-Negative cases for effusion,
     cardiomegaly, pneumothorax.
  3. Renders Grad-CAM++ heatmap overlays for those cases (PNG + HTML contact sheet),
     reusing services/explain (methods.grad_cam) and services/explain/overlays.
  4. Quantifies localization with two ANNOTATION-FREE metrics, honestly:
       * pointing-game accuracy  — does the Grad-CAM++ peak fall inside the
         finding's anatomical region prior?
       * concentration ratio     — heat-mass inside region / region area
         (>1 = better than a uniform map).
     NOTE: MIMIC-CXR ships no lesion bounding boxes. The "region" is the
     anatomical prior in services/vision/engine._FINDING_REGION (costophrenic
     band for effusion, cardiac silhouette for cardiomegaly, lung field for
     pneumothorax) — a clinically-motivated prior, NOT a radiologist annotation.
     We report the metric for what it is and never dress it up as expert IoU.

Outputs (isolated, never clobbers served artifacts): artifacts/explain_demo/
"""
from __future__ import annotations

import os
os.environ.setdefault("AURA_LABELER", "v2")

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common.config import ARTIFACTS
from mimic.config import get_mimic_paths
from ml.vision_cxr.dataset import load_mimic_samples
from ml.vision_cxr.inference import VisionModel
from schemas.clinical import Finding, FINDINGS
from services.explain import methods as M
from services.explain import overlays as OV
from services.vision.engine import _FINDING_REGION

OUT = ARTIFACTS / "explain_demo"
TARGETS = [Finding.EFFUSION, Finding.CARDIOMEGALY, Finding.PNEUMOTHORAX]
# Region priors that are anatomically SPECIFIC (worth a pointing game). The
# full-lung-field findings (opacity/consolidation/nodule/pneumothorax) share the
# same broad box, so their pointing game is near-trivial — reported but flagged.
SPECIFIC = {Finding.EFFUSION, Finding.CARDIOMEGALY}

# Broader, clinically-motivated zone per finding — the region where the sign
# actually manifests on a frontal film, distinct from the TIGHT anatomical prior
# used for the strict pointing game. Effusion/basal disease is a lower-zone
# phenomenon (fluid layers up the hemithorax on portable/AP films, so it is NOT
# confined to the bottom-20% costophrenic band); cardiomegaly is the central
# cardiac shadow; pneumothorax/opacity/nodule are lung-field. We report heat-mass
# inside this zone as a softer, honestly-labelled complement to the strict test.
_CLINICAL_ZONE: dict[Finding, tuple[float, float, float, float]] = {
    Finding.OPACITY: (0.10, 0.05, 0.95, 0.95),
    Finding.CONSOLIDATION: (0.10, 0.05, 0.95, 0.95),
    Finding.EFFUSION: (0.50, 0.05, 1.00, 0.95),        # lower hemithorax
    Finding.CARDIOMEGALY: (0.38, 0.28, 0.85, 0.72),    # cardiac silhouette
    Finding.NODULE: (0.10, 0.05, 0.95, 0.95),
    Finding.PNEUMOTHORAX: (0.05, 0.02, 0.75, 0.98),    # apices + periphery
    Finding.HYPERINFLATION: (0.05, 0.05, 0.98, 0.95),
}


def _load_thresholds() -> dict[Finding, float]:
    path = ARTIFACTS / "vision_serving_calibration.json"
    d = json.loads(Path(path).read_text())
    thr = d.get("per_finding_threshold", {})
    return {f: float(thr.get(f.value, 0.5)) for f in FINDINGS}


def _read_gray(path: Path) -> np.ndarray:
    """Grayscale 0-255 float, exactly as the dataset/VisionModel expects."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((224, 224), dtype=np.float32)
    return img.astype(np.float32)


def _box_mask(box: tuple[float, float, float, float], size: int) -> np.ndarray:
    r0, c0, r1, c1 = box
    m = np.zeros((size, size), dtype=bool)
    a, b = int(r0 * size), int(np.ceil(r1 * size))
    c, d = int(c0 * size), int(np.ceil(c1 * size))
    m[a:b, c:d] = True
    return m


def _region_mask(finding: Finding, size: int) -> np.ndarray:
    return _box_mask(_FINDING_REGION[finding], size)


def _zone_mass(heat: np.ndarray, finding: Finding) -> tuple[float, float]:
    """(mass frac, concentration ratio) of heat inside the broad clinical zone."""
    mask = _box_mask(_CLINICAL_ZONE[finding], heat.shape[0])
    total = float(heat.sum())
    mass_frac = float(heat[mask].sum()) / total if total > 1e-9 else 0.0
    area_frac = float(mask.mean())
    return mass_frac, (mass_frac / area_frac if area_frac > 1e-9 else 0.0)


def _localization(heat: np.ndarray, finding: Finding) -> tuple[bool, float, float]:
    """Return (peak_in_region, mass_frac_in_region, concentration_ratio)."""
    size = heat.shape[0]
    mask = _region_mask(finding, size)
    peak = np.unravel_index(int(heat.argmax()), heat.shape)
    peak_in = bool(mask[peak])
    total = float(heat.sum())
    mass_in = float(heat[mask].sum())
    mass_frac = mass_in / total if total > 1e-9 else 0.0
    area_frac = float(mask.mean())
    conc = mass_frac / area_frac if area_frac > 1e-9 else 0.0
    return peak_in, mass_frac, conc


def score_all(model: VisionModel, paths: list[Path]) -> np.ndarray:
    """Calibrated probs (N, 7) for the whole val set."""
    probs = np.zeros((len(paths), len(FINDINGS)), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        s = model.score_findings(_read_gray(p))
        probs[i] = [s[f] for f in FINDINGS]
        if (i + 1) % 250 == 0:
            print(f"  scored {i+1}/{len(paths)}  ({(i+1)/(time.time()-t0):.0f} img/s)")
    return probs


def pointing_game(model, paths, labels, probs, thr, max_n=250, seed=7):
    """Pointing game + concentration over true positives, per finding."""
    rng = np.random.default_rng(seed)
    fi = {f: i for i, f in enumerate(FINDINGS)}
    results = {}
    for f in FINDINGS:
        j = fi[f]
        tp_idx = np.where((labels[:, j] == 1) & (probs[:, j] >= thr[f]))[0]
        if len(tp_idx) == 0:
            results[f.value] = {"n_tp": 0, "note": "no true positives at serving threshold"}
            continue
        if len(tp_idx) > max_n:
            tp_idx = rng.choice(tp_idx, size=max_n, replace=False)
        hits, masses, concs, zmass, zconc = [], [], [], [], []
        for idx in tp_idx:
            heat = M.grad_cam(model, _read_gray(paths[idx]), f, plusplus=True, out_size=64)
            pin, mf, cc = _localization(heat, f)
            zmf, zcc = _zone_mass(heat, f)
            hits.append(pin); masses.append(mf); concs.append(cc)
            zmass.append(zmf); zconc.append(zcc)
        results[f.value] = {
            "n_tp": int(len(tp_idx)),
            "pointing_game_acc": round(float(np.mean(hits)), 4),
            "mean_mass_in_region": round(float(np.mean(masses)), 4),
            "region_area_frac": round(float(_region_mask(f, 64).mean()), 4),
            "concentration_ratio": round(float(np.mean(concs)), 4),
            "clinical_zone_mass": round(float(np.mean(zmass)), 4),
            "clinical_zone_conc": round(float(np.mean(zconc)), 4),
            "clinical_zone_area_frac": round(float(_box_mask(_CLINICAL_ZONE[f], 64).mean()), 4),
            "region_specific": f in SPECIFIC,
        }
        print(f"  [{f.value}] n_tp={len(tp_idx)} PG={results[f.value]['pointing_game_acc']} "
              f"conc={results[f.value]['concentration_ratio']} "
              f"zone_mass={results[f.value]['clinical_zone_mass']} "
              f"zone_conc={results[f.value]['clinical_zone_conc']}")
    return results


def _pick(labels, probs, thr, f, kind, exclude=()):
    """Pick one illustrative case of the requested kind for finding f.

    Selection is threshold-FREE (ground-truth label + calibrated-confidence rank),
    so every slot fills even where the serving threshold is degenerate (effusion
    thr=0.0 → no formal FN; pneumothorax Platt → nothing crosses 0.5). Each pick is
    later annotated with its calibrated prob and whether it crosses the serving
    threshold, so nothing is mislabelled.
      TP = positive label, highest confidence (best detection)
      FP = negative label, highest confidence (worst false alarm)
      FN = positive label, lowest  confidence (most-missed)
    """
    j = FINDINGS.index(f)
    y, p = labels[:, j], probs[:, j]
    if kind == "TP":
        cand = np.where(y == 1)[0]; order = cand[np.argsort(-p[cand])]
    elif kind == "FP":
        cand = np.where(y == 0)[0]; order = cand[np.argsort(-p[cand])]
    elif kind == "FN":
        cand = np.where(y == 1)[0]; order = cand[np.argsort(p[cand])]
    else:
        return None
    for idx in order:
        if idx not in exclude:
            return int(idx)
    return None


def make_demo_cases(model, paths, labels, probs, thr):
    """Render Grad-CAM++ overlays for TP/FP/FN of each target finding."""
    ov_dir = OUT / "overlays"
    ov_dir.mkdir(parents=True, exist_ok=True)
    fi = {f: i for i, f in enumerate(FINDINGS)}
    cases = []
    used = set()
    for f in TARGETS:
        for kind in ("TP", "FP", "FN"):
            idx = _pick(labels, probs, thr, f, kind, exclude=used)
            if idx is None:
                cases.append({"finding": f.value, "kind": kind, "note": "no such case"})
                continue
            used.add(idx)
            gray = _read_gray(paths[idx])
            disp = gray / 255.0
            heat = M.grad_cam(model, gray, f, plusplus=True, out_size=64)
            boxes = OV.heatmap_bboxes(heat, thresh_rel=0.5, max_boxes=2)
            pin, mf, cc = _localization(heat, f)
            prob = float(probs[idx, fi[f]])
            crosses = prob >= thr[f]
            # Honest label: the requested TP/FP/FN, plus whether it actually crosses
            # the serving threshold (degenerate operating points are flagged here).
            note = ""
            if kind == "TP" and not crosses:
                note = "positive label but BELOW serving threshold (missed at op-point)"
            elif kind == "FN" and crosses:
                note = f"lowest-confidence positive; still ≥ threshold ({thr[f]:.2f})"
            elif kind == "FP" and not crosses:
                note = "highest-confidence negative; still below threshold (no false alarm at op-point)"
            fname = f"{f.value}_{kind}.png"
            title = f"{f.value}  {kind}   p={prob:.2f}  thr={thr[f]:.2f}   peak_in_region={pin}"
            OV.save_overlay_png(ov_dir / fname, disp, heat, boxes=boxes, title=title)
            cases.append({
                "finding": f.value, "kind": kind, "prob": round(prob, 4),
                "threshold": round(thr[f], 4), "crosses_threshold": crosses,
                "label_note": note, "peak_in_region": pin,
                "mass_in_region": round(mf, 4), "concentration_ratio": round(cc, 4),
                "image": str(paths[idx]), "png": f"overlays/{fname}",
            })
            print(f"  demo {f.value:16s} {kind}  p={prob:.2f} crosses={crosses}  "
                  f"peak_in_region={pin}  -> {fname}")
    return cases


def write_html(summary: dict):
    """Self-contained contact sheet of the TP/FP/FN overlays + localization table."""
    rows = []
    for c in summary["demo_cases"]:
        if "png" not in c:
            continue
        img = OUT / c["png"]
        uri = OV._png_data_uri(img.read_bytes())
        note = f'<br><span style="color:#e0a458">{c["label_note"]}</span>' if c.get("label_note") else ""
        rows.append(
            f'<div class="card"><img src="{uri}"/>'
            f'<div class="cap">{c["finding"]} · <b>{c["kind"]}</b> · p={c["prob"]:.2f} '
            f'(thr {c["threshold"]:.2f}) · peak_in_region={c["peak_in_region"]} · '
            f'conc={c["concentration_ratio"]}{note}</div></div>')
    loc = summary["localization"]
    lrows = "".join(
        f"<tr><td>{k}</td><td>{v.get('n_tp','-')}</td>"
        f"<td>{v.get('pointing_game_acc','-')}</td>"
        f"<td>{v.get('concentration_ratio','-')}</td>"
        f"<td>{v.get('clinical_zone_conc','-')}</td>"
        f"<td>{v.get('region_area_frac','-')}</td>"
        f"<td>{'specific' if v.get('region_specific') else 'lung-field'}</td></tr>"
        for k, v in loc.items())
    html = f"""<!doctype html><meta charset=utf-8>
<title>AURA — Grad-CAM++ validation (retrain_v2)</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;background:#0b0f14;color:#e6edf3;margin:0;padding:26px}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:24px;border-bottom:1px solid #223;padding-bottom:6px}}
.grid{{display:flex;flex-wrap:wrap;gap:14px}}
.card{{background:#111823;border:1px solid #223;border-radius:10px;padding:8px}}
.card img{{width:300px;border-radius:6px;display:block}}
.cap{{font-size:12px;margin-top:6px;color:#9fb2c8;max-width:300px}}
table{{border-collapse:collapse;font-size:13px;margin-top:8px}}
td,th{{border-bottom:1px solid #223;padding:6px 10px;text-align:left}}
.note{{color:#9fb2c8;font-size:12.5px;max-width:820px;line-height:1.5}}
</style>
<h1>AURA Grad-CAM++ Validation — served model <span style=color:#22d3ee>retrain_v2</span> (DenseNet121)</h1>
<p class=note>Target layer <code>densenet.features.norm5</code>. All cases are real MIMIC-CXR
validation images (per-study v2 labels). Metrics below are annotation-free: MIMIC-CXR ships no
lesion boxes, so localization is scored against <b>anatomical region priors</b> (costophrenic band
for effusion, cardiac silhouette for cardiomegaly, lung field otherwise) — a clinically-motivated
prior, not a radiologist annotation.</p>
<h2>Localization over true positives (n per finding shown)</h2>
<table><tr><th>finding</th><th>n_TP</th><th>pointing-game</th><th>concentration&nbsp;ratio</th>
<th>clinical-zone&nbsp;conc</th><th>region area frac</th><th>region type</th></tr>{lrows}</table>
<p class=note>Pointing-game = fraction of TP cases whose Grad-CAM++ peak lands inside the <b>tight</b>
anatomical prior. Concentration ratio = (heat mass in tight region)/(region area); &gt;1 = more focused
than a uniform map. Clinical-zone conc = the same ratio against a <b>broader</b> clinically-motivated
zone (lower hemithorax for effusion, cardiac shadow for cardiomegaly, lung field otherwise) — this
separates "peak not in the narrow band" from "map is in the wrong place". Lung-field findings share a
broad box, so their pointing game is near-trivial and flagged as such.</p>
<h2>Case overlays — True Positive / False Positive / False Negative</h2>
<div class=grid>{''.join(rows)}</div>
"""
    (OUT / "gradcam_validation.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap val images (smoke)")
    ap.add_argument("--max-pg", type=int, default=250, help="max TP per finding for pointing game")
    ap.add_argument("--cases-only", action="store_true",
                    help="reuse cached probs + localization, only re-render demo overlays")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[gradcam] labeler = {os.environ['AURA_LABELER']}   out = {OUT}")

    model = VisionModel(str(ARTIFACTS / "best_model.pt"))
    thr = _load_thresholds()
    print("[gradcam] thresholds:", {f.value: round(t, 3) for f, t in thr.items()})

    mp = get_mimic_paths()
    print("[gradcam] loading per-study val labels (v2)...")
    paths, labels = load_mimic_samples(mp.validate_csv, mp.images_root,
                                       limit=args.limit, per_study=True)
    print(f"[gradcam] val images: {len(paths)}  positives/finding: "
          + str({f.value: int(labels[:, i].sum()) for i, f in enumerate(FINDINGS)}))

    cache = OUT / "scores_cache.npz"
    if args.cases_only and cache.exists():
        print("[gradcam] --cases-only: loading cached probs + localization")
        z = np.load(cache, allow_pickle=True)
        probs = z["probs"]
        localization = json.loads(str(z["localization"]))
    else:
        print("[gradcam] scoring val set (calibrated)...")
        probs = score_all(model, paths)
        print("[gradcam] pointing game over true positives...")
        localization = pointing_game(model, paths, labels, probs, thr, max_n=args.max_pg)
        np.savez(cache, probs=probs, localization=json.dumps(localization))

    print("[gradcam] rendering TP/FP/FN demo overlays...")
    demo = make_demo_cases(model, paths, labels, probs, thr)

    summary = {
        "model": "retrain_v2 (best_model.pt)",
        "target_layer": "densenet.features.norm5",
        "method": "grad_cam++",
        "n_val": len(paths),
        "labeler": os.environ["AURA_LABELER"],
        "localization": localization,
        "demo_cases": demo,
        "localization_note": (
            "Anatomical region priors, not radiologist boxes; MIMIC-CXR has no lesion "
            "annotations. Pointing-game/concentration measured against those priors."),
    }
    (OUT / "gradcam_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(summary)
    print(f"[gradcam] DONE. summary -> {OUT/'gradcam_validation.json'}")
    print(f"[gradcam]        report  -> {OUT/'gradcam_validation.html'}")


if __name__ == "__main__":
    main()
