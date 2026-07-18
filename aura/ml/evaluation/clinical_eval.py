"""Inference validation suite on the unseen MIMIC-CXR validation set (Step 2).

Runs the production DenseNet-121 over the held-out MIMIC-CXR validation split and
computes the full clinical metric battery for the multi-label finding task, plus
the plots and bootstrap confidence intervals a model card needs. Every artifact is
written under ``artifacts/evaluation/``.

Reuse, not reinvention:
    * labels come from ``ml/vision_cxr/dataset.load_mimic_samples`` — the exact
      report-derived convention the model was trained/validated against;
    * per-label / macro / micro AUROC-AUPRC-F1 reuse ``mimic.evaluation`` and
      ``services.safety.uncertainty`` primitives;
    * this module adds ROC/PR/calibration curves, per-label operating-point rates,
      Brier, ECE, 2×2 confusion matrices, and bootstrap CIs on top.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from common.config import ARTIFACTS
from schemas.clinical import FINDINGS

FINDING_NAMES = [f.value for f in FINDINGS]
EVAL_DIR = ARTIFACTS / "evaluation"


# --------------------------------------------------------------------------- #
# Data + inference
# --------------------------------------------------------------------------- #
def load_validation(limit: Optional[int] = None):
    """Return (image_paths, labels[N,7]) for the MIMIC validation split."""
    from ml.vision_cxr.config import TrainConfig
    from ml.vision_cxr.dataset import load_mimic_samples

    cfg = TrainConfig()
    paths = cfg.mimic_paths
    return load_mimic_samples(paths.validate_csv, paths.images_root, limit=limit)


def run_inference(image_paths, labels, model_path: Optional[str] = None,
                  batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward the DenseNet over all images. Returns (probs[N,7], y[N,7], seconds)."""
    import torch
    from torch.utils.data import DataLoader

    from ml.vision_cxr.dataset import ChestXrayDataset, get_transforms
    from ml.vision_cxr.inference import VisionModel

    model_path = model_path or str(ARTIFACTS / "best_model.pt")
    vm = VisionModel(model_path)
    ds = ChestXrayDataset(image_paths, labels, transform=get_transforms(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(vm.device == "cuda"))
    probs, ys = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(vm.device, non_blocking=True)
            p = torch.sigmoid(vm.model(x)).cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
    dt = time.perf_counter() - t0
    return np.concatenate(probs), np.concatenate(ys), dt


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _binary_rates(scores: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    eps = 1e-12
    row = {
        "accuracy": round((tp + tn) / (tp + fp + tn + fn + eps), 4),
        "precision": round(tp / (tp + fp + eps), 4),
        "recall": round(tp / (tp + fn + eps), 4),
        "sensitivity": round(tp / (tp + fn + eps), 4),
        "specificity": round(tn / (tn + fp + eps), 4),
        "f1": round(2 * tp / (2 * tp + fp + fn + eps), 4),
        "npv": round(tn / (tn + fn + eps), 4),
        "brier": round(float(np.mean((scores - y) ** 2)), 4),
        "ece": round(binary_ece(scores, y), 4),
        "support": int(y.sum()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    if 0 < y.sum() < len(y):
        row["auroc"] = round(float(roc_auc_score(y, scores)), 4)
        row["auprc"] = round(float(average_precision_score(y, scores)), 4)
    else:
        row["auroc"] = float("nan")
        row["auprc"] = float("nan")
    return row


def binary_ece(scores: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for a single binary probability channel."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(bins):
        m = (scores > edges[i]) & (scores <= edges[i + 1])
        if m.sum():
            ece += (m.sum() / n) * abs(y[m].mean() - scores[m].mean())
    return float(ece)


def _bootstrap_ci(probs: np.ndarray, Y: np.ndarray, n_boot: int, seed: int = 7) -> dict:
    """Percentile bootstrap CIs for per-label AUROC and macro AUROC/AUPRC."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    n, C = Y.shape
    per_auroc = {c: [] for c in range(C)}
    macro_auroc, macro_auprc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Yb, Pb = Y[idx], probs[idx]
        aucs, aps = [], []
        for c in range(C):
            yc = Yb[:, c]
            if 0 < yc.sum() < len(yc):
                a = float(roc_auc_score(yc, Pb[:, c]))
                per_auroc[c].append(a)
                aucs.append(a)
                aps.append(float(average_precision_score(yc, Pb[:, c])))
        if aucs:
            macro_auroc.append(float(np.mean(aucs)))
            macro_auprc.append(float(np.mean(aps)))

    def ci(vals):
        if not vals:
            return [float("nan"), float("nan")]
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    return {
        "n_bootstrap": n_boot,
        "macro_auroc_ci95": ci(macro_auroc),
        "macro_auprc_ci95": ci(macro_auprc),
        "per_label_auroc_ci95": {FINDING_NAMES[c]: ci(per_auroc[c]) for c in range(C)},
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plots(probs: np.ndarray, Y: np.ndarray, plots_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    C = Y.shape[1]
    present = [c for c in range(C) if 0 < Y[:, c].sum() < len(Y)]
    written = {}

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in present:
        fpr, tpr, _ = roc_curve(Y[:, c], probs[:, c])
        ax.plot(fpr, tpr, lw=1.4, label=f"{FINDING_NAMES[c]} ({roc_auc_score(Y[:,c], probs[:,c]):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.set_title("ROC — per finding"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); p = plots_dir / "roc_curves.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written["roc_curves"] = str(p)

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in present:
        prec, rec, _ = precision_recall_curve(Y[:, c], probs[:, c])
        ax.plot(rec, prec, lw=1.4, label=f"{FINDING_NAMES[c]} ({average_precision_score(Y[:,c], probs[:,c]):.2f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision–Recall — per finding"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); p = plots_dir / "pr_curves.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written["pr_curves"] = str(p)

    # Calibration (reliability) grid
    ncol = 4
    nrow = int(np.ceil(C / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(C):
        ax = axes[c]
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (probs[:, c] > edges[i]) & (probs[:, c] <= edges[i + 1])
            if m.sum():
                xs.append(probs[m, c].mean()); ys.append(Y[m, c].mean())
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.plot(xs, ys, "o-", color="#0ea5e9", ms=3, lw=1.2)
        ax.set_title(FINDING_NAMES[c], fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for k in range(C, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Calibration (reliability) per finding", fontsize=11)
    fig.tight_layout(); p = plots_dir / "calibration.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written["calibration"] = str(p)

    # Confidence histograms
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(C):
        axes[c].hist(probs[:, c], bins=20, color="#22d3ee", alpha=0.85)
        axes[c].set_title(FINDING_NAMES[c], fontsize=8)
    for k in range(C, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Predicted-probability histograms", fontsize=11)
    fig.tight_layout(); p = plots_dir / "confidence_histograms.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written["confidence_histograms"] = str(p)

    # Confusion matrices (2×2 per label at 0.5)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(C):
        pred = (probs[:, c] >= 0.5).astype(int)
        cm = np.array([
            [int(((pred == 0) & (Y[:, c] == 0)).sum()), int(((pred == 1) & (Y[:, c] == 0)).sum())],
            [int(((pred == 0) & (Y[:, c] == 1)).sum()), int(((pred == 1) & (Y[:, c] == 1)).sum())],
        ])
        ax = axes[c]
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_title(FINDING_NAMES[c], fontsize=8)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["N", "P"], fontsize=7); ax.set_yticklabels(["N", "P"], fontsize=7)
    for k in range(C, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Confusion matrices (threshold 0.5)", fontsize=11)
    fig.tight_layout(); p = plots_dir / "confusion_matrices.png"; fig.savefig(p, dpi=140); plt.close(fig)
    written["confusion_matrices"] = str(p)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_validation(limit: Optional[int] = None, batch_size: int = 32,
                        n_bootstrap: int = 1000, make_plots: bool = True,
                        model_path: Optional[str] = None,
                        out_dir: Optional[Path] = None) -> dict:
    """Full evaluation on the MIMIC validation split → metrics + plots + CIs on disk."""
    from sklearn.metrics import f1_score

    out_dir = Path(out_dir) if out_dir else EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = load_validation(limit=limit)
    if len(image_paths) == 0:
        raise RuntimeError("No validation images found on disk (check MIMIC paths).")
    probs, Y, infer_s = run_inference(image_paths, labels, model_path=model_path,
                                      batch_size=batch_size)
    C = Y.shape[1]

    per_label = {}
    aurocs, auprcs, f1s = [], [], []
    for c in range(C):
        row = _binary_rates(probs[:, c], Y[:, c])
        per_label[FINDING_NAMES[c]] = row
        if not np.isnan(row["auroc"]):
            aurocs.append(row["auroc"]); auprcs.append(row["auprc"])
        f1s.append(row["f1"])

    pred = (probs >= 0.5).astype(int)
    macro = {
        "auroc": round(float(np.mean(aurocs)), 4) if aurocs else float("nan"),
        "auprc": round(float(np.mean(auprcs)), 4) if auprcs else float("nan"),
        "f1": round(float(np.mean(f1s)), 4),
        "sensitivity": round(float(np.mean([per_label[n]["sensitivity"] for n in FINDING_NAMES])), 4),
        "specificity": round(float(np.mean([per_label[n]["specificity"] for n in FINDING_NAMES])), 4),
        "precision": round(float(np.mean([per_label[n]["precision"] for n in FINDING_NAMES])), 4),
        "accuracy": round(float(np.mean([per_label[n]["accuracy"] for n in FINDING_NAMES])), 4),
        "brier": round(float(np.mean([per_label[n]["brier"] for n in FINDING_NAMES])), 4),
        "ece": round(float(np.mean([per_label[n]["ece"] for n in FINDING_NAMES])), 4),
    }
    micro = {
        "f1": round(float(f1_score(Y.ravel(), pred.ravel(), zero_division=0)), 4),
        "accuracy": round(float((pred == Y).mean()), 4),
        "brier": round(float(np.mean((probs - Y) ** 2)), 4),
    }
    ci = _bootstrap_ci(probs, Y, n_bootstrap) if n_bootstrap > 0 else {}

    plots = {}
    if make_plots:
        plots = _plots(probs, Y, out_dir / "plots")

    report = {
        "task": "multilabel_findings",
        "model_path": model_path or str(ARTIFACTS / "best_model.pt"),
        "n_images": int(len(Y)),
        "labels": FINDING_NAMES,
        "prevalence": {FINDING_NAMES[c]: int(Y[:, c].sum()) for c in range(C)},
        "inference_seconds": round(infer_s, 3),
        "throughput_img_per_s": round(len(Y) / infer_s, 2) if infer_s else None,
        "macro": macro,
        "micro": micro,
        "per_label": per_label,
        "bootstrap_ci": ci,
        "plots": plots,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_summary_md(report, out_dir / "EVALUATION_SUMMARY.md")
    return report


def _write_summary_md(rep: dict, path: Path) -> None:
    L = [f"# MIMIC-CXR Validation — Inference Metrics", "",
         f"- **Model:** `{rep['model_path']}`",
         f"- **Images:** {rep['n_images']}",
         f"- **Inference:** {rep['inference_seconds']}s "
         f"({rep['throughput_img_per_s']} img/s)", ""]
    m = rep["macro"]
    L += ["## Headline (macro over 7 findings)", "",
          f"| AUROC | AUPRC | F1 | Sens | Spec | Prec | Brier | ECE |",
          f"|---|---|---|---|---|---|---|---|",
          f"| {m['auroc']} | {m['auprc']} | {m['f1']} | {m['sensitivity']} | "
          f"{m['specificity']} | {m['precision']} | {m['brier']} | {m['ece']} |", ""]
    if rep.get("bootstrap_ci"):
        ci = rep["bootstrap_ci"]
        L += [f"- **Macro AUROC 95% CI:** {ci['macro_auroc_ci95']} "
              f"({ci['n_bootstrap']} bootstraps)",
              f"- **Macro AUPRC 95% CI:** {ci['macro_auprc_ci95']}", ""]
    L += ["## Per-finding", "",
          "| finding | AUROC | AUPRC | sens | spec | F1 | Brier | ECE | support |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in rep["per_label"].items():
        L.append(f"| {n} | {r['auroc']} | {r['auprc']} | {r['sensitivity']} | "
                 f"{r['specificity']} | {r['f1']} | {r['brier']} | {r['ece']} | {r['support']} |")
    L += ["", f"_Micro F1 {rep['micro']['f1']} · micro accuracy {rep['micro']['accuracy']}._"]
    path.write_text("\n".join(L), encoding="utf-8")


def run(limit: Optional[int] = None) -> dict:
    """CLI entry point."""
    print("[evaluate] running MIMIC-CXR validation inference ...")
    rep = evaluate_validation(limit=limit)
    m = rep["macro"]
    print(f"[evaluate] n={rep['n_images']}  macro AUROC={m['auroc']}  AUPRC={m['auprc']}  "
          f"F1={m['f1']}  ECE={m['ece']}")
    print(f"[evaluate] artifacts -> {EVAL_DIR}")
    return rep


if __name__ == "__main__":
    run()
