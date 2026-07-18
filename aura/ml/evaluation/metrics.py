"""Full clinical evaluation suite.

Why this exists
---------------
Training/benchmarking reported accuracy, NLL, ECE, and conformal stats — useful,
but a clinician cares about the operating-point numbers: does it *catch* disease
(sensitivity), does it avoid false alarms (specificity), what does a positive read
mean (PPV/NPV), and is the ranking any good regardless of threshold (AUROC/AUPRC)?
This module computes all of them, per class (one-vs-rest) and macro-averaged, plus
proper scores (Brier) and calibration curves — everything needed for a model card.

Pure numpy + scikit-learn.
"""
from __future__ import annotations

import numpy as np

from schemas.clinical import DIAGNOSES
from services.safety.uncertainty import brier_score, reliability_curve


def _binary_rates(tp, fp, tn, fn) -> dict[str, float]:
    eps = 1e-12
    return {
        "sensitivity": tp / (tp + fn + eps),   # recall / TPR
        "specificity": tn / (tn + fp + eps),   # TNR
        "ppv": tp / (tp + fp + eps),           # precision
        "npv": tn / (tn + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "support": int(tp + fn),
    }


def per_class_metrics(P: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """One-vs-rest sensitivity/specificity/PPV/NPV/F1 + AUROC/AUPRC per class."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = P.argmax(axis=1)
    out: dict[str, dict] = {}
    for c, dx in enumerate(DIAGNOSES):
        yc = (y == c).astype(int)
        pc = (pred == c).astype(int)
        tp = int(((pc == 1) & (yc == 1)).sum())
        fp = int(((pc == 1) & (yc == 0)).sum())
        tn = int(((pc == 0) & (yc == 0)).sum())
        fn = int(((pc == 0) & (yc == 1)).sum())
        row = _binary_rates(tp, fp, tn, fn)
        # AUROC/AUPRC need both classes present.
        if 0 < yc.sum() < len(yc):
            row["auroc"] = float(roc_auc_score(yc, P[:, c]))
            row["auprc"] = float(average_precision_score(yc, P[:, c]))
        else:
            row["auroc"] = float("nan")
            row["auprc"] = float("nan")
        out[dx.value] = {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in row.items()}
    return out


def _macro(per_class: dict[str, dict], key: str) -> float:
    vals = [m[key] for m in per_class.values()
            if isinstance(m.get(key), (int, float)) and not np.isnan(m[key])]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(P: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Complete report: accuracy, NLL, Brier, ECE, macro AUROC/AUPRC/sens/spec/PPV/
    NPV/F1, per-class table, and a reliability curve for the calibration diagram."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    acc = float((P.argmax(1) == y).mean())
    nll = float(-np.log(np.clip(P[np.arange(n), y], 1e-12, 1)).mean())
    rel = reliability_curve(P, y, bins=bins)
    pcm = per_class_metrics(P, y)
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "nll": round(nll, 4),
        "brier": round(brier_score(P, y), 4),
        "ece": round(rel["ece"], 4),
        "macro_auroc": round(_macro(pcm, "auroc"), 4),
        "macro_auprc": round(_macro(pcm, "auprc"), 4),
        "macro_sensitivity": round(_macro(pcm, "sensitivity"), 4),
        "macro_specificity": round(_macro(pcm, "specificity"), 4),
        "macro_ppv": round(_macro(pcm, "ppv"), 4),
        "macro_npv": round(_macro(pcm, "npv"), 4),
        "macro_f1": round(_macro(pcm, "f1"), 4),
        "per_class": pcm,
        "reliability_curve": {
            "bin_confidence": [round(v, 4) for v in rel["bin_confidence"]],
            "bin_accuracy": [round(v, 4) for v in rel["bin_accuracy"]],
            "bin_weight": [round(v, 4) for v in rel["bin_weight"]],
        },
    }


def print_report(report: dict, title: str = "Evaluation") -> None:
    print(f"\n  {title}  (n={report['n']})\n")
    top = [("accuracy", "higher"), ("macro_auroc", "higher"), ("macro_auprc", "higher"),
           ("macro_sensitivity", "higher"), ("macro_specificity", "higher"),
           ("macro_f1", "higher"), ("ece", "lower"), ("brier", "lower"), ("nll", "lower")]
    for k, better in top:
        print(f"  {k:<20}{report[k]:>10}   ({better})")
    print(f"\n  {'class':<16}{'sens':>7}{'spec':>7}{'ppv':>7}{'npv':>7}{'f1':>7}"
          f"{'auroc':>8}{'auprc':>8}{'n':>6}")
    print("  " + "-" * 74)
    for cls, m in report["per_class"].items():
        print(f"  {cls:<16}{m['sensitivity']:>7}{m['specificity']:>7}{m['ppv']:>7}"
              f"{m['npv']:>7}{m['f1']:>7}{m['auroc']:>8}{m['auprc']:>8}{m['support']:>6}")
    print()
