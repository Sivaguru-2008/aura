"""Step 10 — Evaluation suite for the MIMIC-CXR tasks.

The multiclass diagnosis case is already fully covered by
``ml.evaluation.metrics.evaluate`` (accuracy, NLL, Brier, ECE, macro AUROC/AUPRC,
per-class sensitivity/specificity/PPV/NPV/F1, reliability curve). This module:

    * reuses that suite for diagnosis and adds an explicit **confusion matrix**;
    * adds a **binary** evaluator (all operating-point rates + AUROC/AUPRC/ECE/Brier);
    * adds a **multi-label** evaluator (per-label + macro/micro AUROC/AUPRC/F1);
    * writes a markdown **model card**.

Everything is numpy + scikit-learn, matching the existing code's dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.evaluation.metrics import evaluate as evaluate_diagnosis_core
from services.safety.uncertainty import brier_score


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, cols = predicted class."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        m[t, p] += 1
    return m


def evaluate_multiclass(proba: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    """Full diagnosis report + confusion matrix."""
    rep = dict(evaluate_diagnosis_core(proba, y))
    pred = np.asarray(proba).argmax(1)
    rep["confusion_matrix"] = confusion_matrix(y, pred, len(classes)).tolist()
    rep["classes"] = classes
    return rep


def _rates(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    eps = 1e-12
    return {
        "accuracy": (tp + tn) / (tp + fp + tn + fn + eps),
        "precision": tp / (tp + fp + eps),          # == PPV
        "recall": tp / (tp + fn + eps),             # == sensitivity
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "ppv": tp / (tp + fp + eps),
        "npv": tn / (tn + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
    }


def evaluate_binary(scores: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary task report: AUROC/AUPRC + all operating-point rates + confusion + calibration.

    ``scores`` is P(positive) — either a 1-D array or the positive column of a
    2-column proba matrix.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, float)
    if scores.ndim == 2:
        scores = scores[:, 1]
    y = np.asarray(y, int)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out = {k: round(v, 4) for k, v in _rates(tp, fp, tn, fn).items()}
    out["confusion_matrix"] = [[tn, fp], [fn, tp]]
    out["support_positive"] = int(y.sum())
    out["n"] = int(len(y))
    if 0 < y.sum() < len(y):
        out["auroc"] = round(float(roc_auc_score(y, scores)), 4)
        out["auprc"] = round(float(average_precision_score(y, scores)), 4)
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    # Brier for the positive class
    out["brier"] = round(float(np.mean((scores - y) ** 2)), 4)
    return out


def evaluate_multilabel(proba: np.ndarray, Y: np.ndarray, labels: list[str]) -> dict:
    """Per-label AUROC/AUPRC/F1 (@0.5) + macro & micro averages."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    proba = np.asarray(proba, float)
    Y = np.asarray(Y, int)
    pred = (proba >= 0.5).astype(int)
    per: dict[str, dict] = {}
    aurocs, auprcs, f1s = [], [], []
    for i, lab in enumerate(labels):
        yi = Y[:, i]
        row: dict[str, float] = {"support": int(yi.sum())}
        if 0 < yi.sum() < len(yi):
            row["auroc"] = round(float(roc_auc_score(yi, proba[:, i])), 4)
            row["auprc"] = round(float(average_precision_score(yi, proba[:, i])), 4)
            aurocs.append(row["auroc"]); auprcs.append(row["auprc"])
        else:
            row["auroc"] = float("nan"); row["auprc"] = float("nan")
        row["f1"] = round(float(f1_score(yi, pred[:, i], zero_division=0)), 4)
        f1s.append(row["f1"])
        per[lab] = row
    return {
        "n": int(len(Y)),
        "macro_auroc": round(float(np.mean(aurocs)), 4) if aurocs else float("nan"),
        "macro_auprc": round(float(np.mean(auprcs)), 4) if auprcs else float("nan"),
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else float("nan"),
        "micro_f1": round(float(f1_score(Y, pred, average="micro", zero_division=0)), 4),
        "per_label": per,
    }


def write_model_card(report: dict, path: Path, title: str = "Model Card") -> Path:
    """Serialize an evaluation report to a readable markdown model card + JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for k, v in report.items():
        if isinstance(v, (int, float, str)):
            lines.append(f"- **{k}**: {v}")
    if "confusion_matrix" in report:
        lines += ["", "## Confusion matrix", "", "```",
                  "\n".join(str(r) for r in report["confusion_matrix"]), "```"]
    if "per_class" in report:
        lines += ["", "## Per-class", "", "| class | sens | spec | ppv | npv | f1 | auroc | n |",
                  "|---|---|---|---|---|---|---|---|"]
        for c, m in report["per_class"].items():
            lines.append(
                f"| {c} | {m['sensitivity']} | {m['specificity']} | {m['ppv']} | "
                f"{m['npv']} | {m['f1']} | {m['auroc']} | {m['support']} |"
            )
    if "per_label" in report:
        lines += ["", "## Per-label", "", "| label | auroc | auprc | f1 | support |",
                  "|---|---|---|---|---|"]
        for c, m in report["per_label"].items():
            lines.append(f"| {c} | {m['auroc']} | {m['auprc']} | {m['f1']} | {m['support']} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
