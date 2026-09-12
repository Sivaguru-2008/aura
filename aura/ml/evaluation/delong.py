"""DeLong's test for two correlated ROC curves.

Why this file exists
--------------------
Section 5.3 of the manuscript states that "AUROC comparisons used DeLong [70]".
No implementation of DeLong existed anywhere in this repository when that sentence
was written; AUROC differences were reported with bootstrap intervals instead. The
method statement therefore described a procedure the artifacts could not have used.
This module supplies it, so that the statement becomes true rather than being
softened.

The implementation is the fast midrank algorithm of Sun and Xu (2014), which
computes the same statistic as DeLong, DeLong and Clarke-Pearson (1988) in
O(N log N) rather than O(N^2).

References
----------
DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric approach.
*Biometrics*. 1988;44(3):837-845.

Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas
under correlated receiver operating characteristic curves. *IEEE Signal Processing
Letters*. 2014;21(11):1389-1393.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["auc_variance", "delong_test", "midrank"]


def midrank(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x`` with ties assigned their average rank (1-based)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks_sorted[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _structural_components(scores: np.ndarray, m: int):
    """V10, V01 and AUC for each row of ``scores``.

    ``scores`` is ``(k_models, m + n)`` with the ``m`` positive cases first.
    """
    k, total = scores.shape
    n = total - m
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative case")

    v10 = np.empty((k, m), dtype=float)
    v01 = np.empty((k, n), dtype=float)
    auc = np.empty(k, dtype=float)

    for r in range(k):
        pos, neg = scores[r, :m], scores[r, m:]
        tz = midrank(scores[r])
        tx = midrank(pos)
        ty = midrank(neg)
        auc[r] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v10[r] = (tz[:m] - tx) / n
        v01[r] = 1.0 - (tz[m:] - ty) / m

    return v10, v01, auc


def auc_variance(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """AUROC and its DeLong variance for one model."""
    label = np.asarray(label).astype(bool)
    score = np.asarray(score, dtype=float)
    ordered = np.concatenate([score[label], score[~label]])
    m = int(label.sum())
    v10, v01, auc = _structural_components(ordered[None, :], m)
    n = len(score) - m
    var = float(np.var(v10[0], ddof=1) / m + np.var(v01[0], ddof=1) / n)
    return float(auc[0]), var


def delong_test(score_a: np.ndarray, score_b: np.ndarray,
                label: np.ndarray) -> dict:
    """Paired comparison of two AUROCs on the same cases.

    Returns the two areas, their difference, the standard error of the difference,
    a two-sided p-value, and a 95% confidence interval for the difference. The
    interval is the normal-theory one implied by the same covariance matrix, so it
    agrees with the p-value by construction — a bootstrap interval reported beside
    a DeLong p-value need not.
    """
    label = np.asarray(label).astype(bool)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if not (len(a) == len(b) == len(label)):
        raise ValueError("scores and labels must be the same length")

    m = int(label.sum())
    n = int((~label).sum())
    stacked = np.vstack([
        np.concatenate([a[label], a[~label]]),
        np.concatenate([b[label], b[~label]]),
    ])
    v10, v01, auc = _structural_components(stacked, m)

    s10 = np.cov(v10, ddof=1)
    s01 = np.cov(v01, ddof=1)
    cov = s10 / m + s01 / n

    contrast = np.array([1.0, -1.0])
    var_diff = float(contrast @ cov @ contrast)
    delta = float(auc[0] - auc[1])
    se = float(np.sqrt(var_diff)) if var_diff > 0 else 0.0

    if se == 0.0:
        p = 1.0 if delta == 0.0 else 0.0
        lo = hi = delta
    else:
        z = delta / se
        p = float(2.0 * stats.norm.sf(abs(z)))
        lo, hi = delta - 1.96 * se, delta + 1.96 * se

    return {
        "auroc_a": round(float(auc[0]), 4),
        "auroc_b": round(float(auc[1]), 4),
        "delta_auroc": round(delta, 4),
        "se": round(se, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "z": round(float(delta / se), 4) if se else None,
        "p": float(f"{p:.4e}"),
        "n_positive": m,
        "n_negative": n,
        "method": "DeLong (Sun & Xu 2014 fast midrank implementation)",
    }
