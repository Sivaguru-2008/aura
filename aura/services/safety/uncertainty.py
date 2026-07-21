"""Uncertainty quantification primitives shared by serving and evaluation.

Why this exists
---------------
Epistemic uncertainty shipped as an *input-perturbation proxy*: jitter the
evidence vector, measure posterior spread. That conflates model uncertainty with
input sensitivity and has no theory behind it. This module provides the real
tools:

  * Deep-ensemble decomposition — total predictive uncertainty splits into
    aleatoric (expected member entropy) and epistemic (mutual information / BALD).
  * Monte-Carlo dropout — stochastic forward passes through a torch module.
  * Brier score, reliability diagrams — proper scoring + calibration curves.
  * Mondrian (class-conditional) conformal — coverage *per class*, not just on
    average, which is what matters when a rare-but-dangerous label is present.

Pure numpy except the MC-dropout helper (lazy torch). Entropies are in bits to
match ``common.mathx.entropy``.
"""
from __future__ import annotations

import numpy as np

from common.mathx import entropy


# --------------------------------------------------------------------------- #
# Deep-ensemble uncertainty decomposition
# --------------------------------------------------------------------------- #
def ensemble_decomposition(member_probs: np.ndarray) -> dict[str, float]:
    """Split predictive uncertainty for one case given K member posteriors.

    member_probs: (K, C). Returns predictive_entropy = aleatoric + epistemic (BALD),
    plus the ensemble mean posterior's top-class std (a [0,1]-scaled epistemic that
    is drop-in comparable with the old proxy's threshold).
    """
    P = np.asarray(member_probs, dtype=float)
    mean = P.mean(axis=0)
    predictive = entropy(mean)                          # H[E[p]]
    aleatoric = float(np.mean([entropy(p) for p in P]))  # E[H[p]]
    epistemic_mi = max(0.0, predictive - aleatoric)      # mutual information
    top = int(mean.argmax())
    top_std = float(P[:, top].std())
    return {
        "mean": mean,
        "predictive_entropy": float(predictive),
        "aleatoric_entropy": float(aleatoric),
        "epistemic_mi": float(epistemic_mi),
        "epistemic_std": top_std,
    }


# --------------------------------------------------------------------------- #
# Monte-Carlo dropout
# --------------------------------------------------------------------------- #
def enable_dropout(module) -> None:
    """Put only dropout layers into train mode (keep BN/etc. in eval)."""
    for m in module.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def mc_dropout_logits(module, x, k: int = 20):
    """K stochastic forward passes with dropout active. Returns (K, ...) logits np array."""
    import torch

    was_training = module.training
    module.eval()
    enable_dropout(module)
    outs = []
    with torch.no_grad():
        for _ in range(k):
            outs.append(module(x).detach().cpu().numpy())
    if was_training:
        module.train()
    else:
        module.eval()
    return np.stack(outs, axis=0)


# --------------------------------------------------------------------------- #
# Proper scoring + calibration curves
# --------------------------------------------------------------------------- #
def brier_score(P: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error against one-hot)."""
    P = np.asarray(P, dtype=float)
    Y = np.eye(P.shape[1])[np.asarray(y, dtype=int)]
    return float(((P - Y) ** 2).sum(axis=1).mean())


def reliability_curve(P: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Reliability diagram data: per-bin confidence, accuracy, and weight.

    Returns arrays suitable for plotting a calibration curve and for reporting ECE.
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    conf_b, acc_b, w_b = [], [], []
    n = len(y)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            conf_b.append(0.0); acc_b.append(0.0); w_b.append(0.0)
            continue
        c, a, w = float(conf[m].mean()), float(correct[m].mean()), float(m.sum() / n)
        conf_b.append(c); acc_b.append(a); w_b.append(w)
        ece += w * abs(a - c)
    return {
        "bin_confidence": conf_b,
        "bin_accuracy": acc_b,
        "bin_weight": w_b,
        "ece": float(ece),
        "edges": edges.tolist(),
    }


# --------------------------------------------------------------------------- #
# Mondrian (class-conditional) conformal prediction
# --------------------------------------------------------------------------- #
def min_calibration_count(coverage: float) -> int:
    """Smallest per-class calibration count for a *non-degenerate* conformal q̂.

    The split-conformal quantile level is ``ceil((n+1)·coverage)/n``. When that is
    ``≥ 1`` the "higher" quantile collapses onto the **maximum** nonconformity score
    — a degenerate threshold that makes the class appear in almost every prediction
    set (audit F7: malignancy q̂ = 0.9889 → malignancy in ~78 % of sets). This
    returns the smallest ``n`` for which the level is strictly ``< 1`` (e.g. 19 at
    90 % coverage), so below it we must defer to the pooled marginal threshold.
    """
    n = 1
    while np.ceil((n + 1) * coverage) / n >= 1.0:
        n += 1
        if n > 10_000:                          # coverage → 1 pathological guard
            break
    return n


def mondrian_qhats(probs: np.ndarray, labels: np.ndarray, coverage: float,
                   n_classes: int) -> np.ndarray:
    """Per-class split-conformal thresholds on nonconformity s = 1 - p[true].

    Guarantees ~coverage *within each class* (conditional), unlike the single
    marginal quantile. A class only gets its own threshold when it has enough
    calibration points for a **non-degenerate** quantile (``>= min_calibration_count``);
    otherwise — and for classes unseen in calibration — it falls back to the pooled
    marginal threshold, so no class saturates to its maximum score (audit F7).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    scores = 1.0 - probs[np.arange(len(labels)), labels]
    marginal = _quantile_hi(scores, coverage)
    if marginal is None:                        # even the pooled set is tiny
        marginal = float(scores.max()) if len(scores) else 1.0
    min_n = min_calibration_count(coverage)
    qhats = np.full(n_classes, marginal, dtype=float)
    for c in range(n_classes):
        m = labels == c
        if m.sum() >= min_n:                    # enough for a valid class quantile
            q = _quantile_hi(scores[m], coverage)
            if q is not None:
                qhats[c] = q
    return qhats


def _quantile_hi(scores: np.ndarray, coverage: float) -> float | None:
    """Upper split-conformal quantile of ``scores``, or ``None`` if degenerate.

    Returns ``None`` when there are too few points for the target coverage level to
    be a genuine order statistic (the level would be ``>= 1`` and the quantile would
    collapse onto the maximum score) — the caller then falls back to a pooled
    threshold instead of a saturated, class-inflating one (audit F7).
    """
    n = len(scores)
    if n == 0:
        return None
    level = np.ceil((n + 1) * coverage) / n
    if level >= 1.0:
        return None
    return float(np.quantile(scores, level, method="higher"))


def mondrian_set(probs: np.ndarray, qhats: np.ndarray) -> list[int]:
    """Class-conditional conformal set: keep class c if (1 - p[c]) <= qhat[c]."""
    probs = np.asarray(probs, dtype=float)
    keep = [c for c in range(len(probs)) if (1.0 - probs[c]) <= qhats[c]]
    return keep or [int(probs.argmax())]
