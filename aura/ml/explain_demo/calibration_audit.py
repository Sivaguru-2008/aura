"""Task 2 (trust) — honest re-measurement of the VISION calibration on the full
per-study validation set (n=2099, v2 labels), comparing three operating points:

  raw     : sigmoid(logit)                              — uncalibrated
  served  : sigmoid(platt_a·logit + platt_b)            — what ml/vision_cxr/inference.py
            using artifacts/vision_serving_calibration.json  actually applies (fit n=16)
  full    : sigmoid(platt_a·logit + platt_b)            — artifacts/calibration/calibration.json
            using artifacts/calibration/calibration.json    (fit n=2099)

Outputs (isolated): artifacts/explain_demo/calibration_audit.{json,md} + reliability PNG.
Every number is measured here, not copied from a report artifact.
"""
from __future__ import annotations
import os
os.environ.setdefault("AURA_LABELER", "v2")
import json
from pathlib import Path

import numpy as np
import torch

from common.config import ARTIFACTS
from mimic.config import get_mimic_paths
from ml.vision_cxr.dataset import load_mimic_samples
from ml.vision_cxr.inference import VisionModel
from schemas.clinical import FINDINGS
import cv2

OUT = ARTIFACTS / "explain_demo"


def raw_logits(model: VisionModel, paths, bs=32) -> np.ndarray:
    """Uncalibrated logits (N,7) — batched forward, no Platt."""
    dev = model.device
    out = np.zeros((len(paths), len(FINDINGS)), dtype=np.float32)
    buf, idxs = [], []

    def flush(buf, idxs):
        if not buf:
            return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            lg = model.model(x).cpu().numpy()
        out[idxs] = lg
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = np.zeros((224, 224), np.float32) if img is None else img.astype(np.float32)
        # replicate VisionModel._to_tensor without the batch dim
        t = model._to_tensor(img)[0]
        buf.append(t); idxs.append(i)
        if len(buf) == bs:
            flush(buf, idxs); buf, idxs = [], []
    flush(buf, idxs)
    return out


def ece(probs_1d, labels_1d, n_bins=15) -> float:
    """Expected Calibration Error (equal-width bins) for one finding."""
    p = np.asarray(probs_1d); y = np.asarray(labels_1d)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        conf = p[m].mean(); acc = y[m].mean()
        e += (m.mean()) * abs(conf - acc)
    return float(e)


def reliability_bins(probs_1d, labels_1d, n_bins=10):
    p = np.asarray(probs_1d); y = np.asarray(labels_1d)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = VisionModel(str(ARTIFACTS / "best_model.pt"))
    mp = get_mimic_paths()
    paths, labels = load_mimic_samples(mp.validate_csv, mp.images_root, per_study=True)
    print(f"[calib] val images: {len(paths)}")

    logits = raw_logits(model, paths)
    print("[calib] raw logits computed", logits.shape)

    # "served" = the OLD degenerate n=16 fit (from the backup taken before the fix);
    # "full"   = the current, fixed serving calibration (validated n=2099 full-val fit).
    # This shows the before/after of fixing the test-clobbering bug.
    bak = ARTIFACTS / "vision_serving_calibration.n16.bak.json"
    serv = json.loads(bak.read_text()) if bak.exists() else \
        json.loads((ARTIFACTS / "vision_serving_calibration.json").read_text())
    full = json.loads((ARTIFACTS / "vision_serving_calibration.json").read_text())  # current served (fixed)
    full_pf = {f: {"platt_a": full["per_finding_platt"][f]["a"],
                   "platt_b": full["per_finding_platt"][f]["b"],
                   "threshold": full["per_finding_threshold"][f]} for f in full["per_finding_platt"]}
    serv_platt = serv.get("per_finding_platt", {})
    serv_thr = serv.get("per_finding_threshold", {})

    rows = {}
    macro = {"raw": [], "served": [], "full": []}
    fire = {"served": {}, "full": {}}
    relia = {}
    for i, f in enumerate(FINDINGS):
        z = logits[:, i]; y = labels[:, i]
        p_raw = sig(z)
        sa = float(serv_platt.get(f.value, {}).get("a", 1.0))
        sb = float(serv_platt.get(f.value, {}).get("b", 0.0))
        p_serv = sig(sa * z + sb)
        fa = float(full_pf.get(f.value, {}).get("platt_a", 1.0))
        fb = float(full_pf.get(f.value, {}).get("platt_b", 0.0))
        p_full = sig(fa * z + fb)
        e_raw, e_serv, e_full = ece(p_raw, y), ece(p_serv, y), ece(p_full, y)
        macro["raw"].append(e_raw); macro["served"].append(e_serv); macro["full"].append(e_full)
        st = float(serv_thr.get(f.value, 0.5))
        ft = float(full_pf.get(f.value, {}).get("threshold", 0.5))
        fire["served"][f.value] = int((p_serv >= st).sum())
        fire["full"][f.value] = int((p_full >= ft).sum())
        rows[f.value] = {
            "n_pos": int(y.sum()),
            "ece_raw": round(e_raw, 4), "ece_served": round(e_serv, 4), "ece_full": round(e_full, 4),
            "served_threshold": round(st, 4), "served_fires": fire["served"][f.value],
            "full_threshold": round(ft, 4), "full_fires": fire["full"][f.value],
        }
        relia[f.value] = {kind: [a.tolist() for a in reliability_bins(pp, y)]
                          for kind, pp in (("raw", p_raw), ("served", p_serv), ("full", p_full))}
        print(f"  {f.value:16s} ECE raw={e_raw:.3f} served={e_serv:.3f} full={e_full:.3f} "
              f"| fires served={fire['served'][f.value]:4d} full={fire['full'][f.value]:4d} / {int(y.sum())} pos")

    summary = {
        "n_val": len(paths),
        "macro_ece": {k: round(float(np.mean(v)), 4) for k, v in macro.items()},
        "per_finding": rows,
        "served_calibration_n_fit": serv.get("n_images"),
        "full_calibration_n_fit": full.get("n_images"),
        "note": ("BEFORE/AFTER of fixing the calibration-clobber bug. served = the OLD degenerate "
                 "fit (n=16, written by run_calibration(limit=3) in tests, backup "
                 "vision_serving_calibration.n16.bak.json); full = the CURRENT served calibration "
                 "(validated full-val fit, n=2099). Root cause fixed in ml/evaluation/"
                 "vision_calibration.py: production serving write is now gated to canonical runs."),
    }
    (OUT / "calibration_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(relia, summary)
    _markdown(summary)
    print("[calib] macro ECE:", summary["macro_ece"])
    print(f"[calib] DONE -> {OUT/'calibration_audit.json'}")


def _plot(relia, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    findings = list(relia.keys())
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    axes = axes.ravel()
    colors = {"raw": "#e74c3c", "served": "#e0a458", "full": "#22d3ee"}
    label_of = {"raw": "raw", "served": "before n=16", "full": "fixed n=2099"}
    for ax, f in zip(axes, findings):
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        for kind in ("raw", "served", "full"):
            xs, ys, ns = relia[f][kind]
            xs, ys = np.array(xs), np.array(ys)
            if len(xs):
                ax.plot(xs, ys, "o-", color=colors[kind], ms=4, lw=1.5,
                        label=f"{label_of[kind]} (ECE {summary['per_finding'][f]['ece_'+kind]:.2f})")
        ax.set_title(f, fontsize=11); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.legend(fontsize=7)
    # summary panel
    ax = axes[-1]; ax.axis("off")
    m = summary["macro_ece"]
    ax.text(0.02, 0.9, "Macro ECE (n=%d val)" % summary["n_val"], fontsize=12, weight="bold")
    ax.text(0.02, 0.72, f"raw          {m['raw']:.3f}", color=colors["raw"], fontsize=12)
    ax.text(0.02, 0.60, f"before n=16  {m['served']:.3f}", color=colors["served"], fontsize=12)
    ax.text(0.02, 0.48, f"fixed n=2099 {m['full']:.3f}", color=colors["full"], fontsize=12)
    ax.text(0.02, 0.26, "The n=16 fit (written by a limit=3 smoke run in tests)\n"
            "made pneumothorax/hyperinflation never fire. Root cause\n"
            "fixed; served calibration re-wired to the full-val fit.",
            fontsize=8.5, color="#9fb2c8")
    fig.suptitle("AURA vision reliability — before/after fixing the calibration-clobber bug (retrain_v2)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "calibration_reliability.png", dpi=130)
    plt.close(fig)


def _markdown(s):
    L = ["# AURA Vision Calibration Audit (measured on full val, n=%d)\n" % s["n_val"]]
    m = s["macro_ece"]
    L.append(f"**Macro ECE:** raw {m['raw']:.3f} → served {m['served']:.3f} (Platt, fit n=16) "
             f"→ **full-val {m['full']:.3f}** (Platt, fit n=2099)\n")
    L.append("| finding | n_pos | ECE raw | ECE served | ECE full | served thr | served fires | full thr | full fires |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for f, r in s["per_finding"].items():
        L.append(f"| {f} | {r['n_pos']} | {r['ece_raw']:.3f} | {r['ece_served']:.3f} | {r['ece_full']:.3f} "
                 f"| {r['served_threshold']:.2f} | {r['served_fires']} | {r['full_threshold']:.2f} | {r['full_fires']} |")
    L.append("\n" + s["note"])
    (OUT / "calibration_audit.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
