"""Murphy / DeGroot-Fienberg decomposition of the multiclass Brier score.

Replaces ``murphy_decomp_seed7.py``, which was wrong in two ways:

  1. It binned on ``probs.max(axis=1)``. The decomposition identity
     ``Brier = Reliability - Resolution + Uncertainty`` holds for groups of
     *identical forecast vectors*. Binning on the max probability pools
     forecasts that are different vectors into one group, so the identity does
     not hold and the reported terms do not reconstruct the score.
  2. It computed ``delta_brier`` from ``actual_brier`` while reporting
     reliability and resolution from the binned path. The two are not the same
     quantity, and the mismatch was invisible because ``binned_brier`` was
     computed and never printed.

Here forecasts are grouped by k-means on the probability simplex, and the
residual ``actual - (REL - RES + UNC)`` is computed and reported rather than
assumed to be zero. With continuous forecasts the residual is never exactly
zero: it collects the within-group forecast variance and the within-group
forecast/outcome covariance. Reporting it is the honest alternative to
pretending the identity closes. The decomposition is run at three group counts
so that a reader can see whether the conclusion depends on the binning.

Writes ``artifacts/murphy_decomposition.json``.
"""
from __future__ import annotations

import json
import time

import numpy as np

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.services.fusion.device import make_qnode
from aura.services.safety.calibration import fit_temperature
from aura.ml.evaluation.quantum_study import load_evidence
from aura.ml.training.train_fusion import train_quantum

OUTPUT = ARTIFACTS / "murphy_decomposition.json"

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
N_LAYERS = 3
EPOCHS = 30
GROUP_COUNTS = (5, 10, 20)
WIDTHS = (8, 7)          # 8 qubits reproduces Table 8; 7 drops the dead channel


def _quantum_logits(theta, W, b, X, n_qubits, n_layers, entangler):
    circuit = make_qnode(n_qubits, n_layers, interface="numpy", entangler=entangler)
    z = np.stack([np.asarray(v) for v in circuit(np.asarray(X), theta)], axis=-1)
    return z @ W.T + b


def _kmeans_groups(P: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """Group forecast vectors by k-means on the simplex. Returns a label per row."""
    rng = np.random.default_rng(seed)
    n = len(P)
    k = min(k, n)
    centres = P[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((P[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                centres[j] = P[m].mean(axis=0)
    return labels


def murphy_decomposition(probs: np.ndarray, targets: np.ndarray, n_groups: int) -> dict:
    """Multiclass Murphy decomposition with the reconstruction residual reported.

    Brier = REL - RES + UNC + residual, where the residual is what continuous
    forecasts leave behind after grouping. It is returned, not discarded.
    """
    N, C = probs.shape
    y = np.zeros((N, C))
    y[np.arange(N), targets] = 1.0

    y_bar = y.mean(axis=0)
    uncertainty = float(np.sum(y_bar * (1.0 - y_bar)))

    labels = _kmeans_groups(probs, n_groups)

    reliability = 0.0
    resolution = 0.0
    used = 0
    for g in np.unique(labels):
        m = labels == g
        n_g = int(m.sum())
        if n_g == 0:
            continue
        used += 1
        p_g = probs[m].mean(axis=0)
        y_g = y[m].mean(axis=0)
        reliability += n_g * float(np.sum((p_g - y_g) ** 2))
        resolution += n_g * float(np.sum((y_g - y_bar) ** 2))
    reliability /= N
    resolution /= N

    actual = float(np.mean(np.sum((probs - y) ** 2, axis=1)))
    reconstructed = reliability - resolution + uncertainty
    return {
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "reconstructed_brier": round(reconstructed, 6),
        "actual_brier": round(actual, 6),
        "residual": round(actual - reconstructed, 6),
        "n_groups_requested": n_groups,
        "n_groups_used": used,
    }


def _train_arm(Xtr, ytr, Xcal, ycal, Xte, n_qubits, entangler, seed):
    theta, W, b = train_quantum(
        Xtr, ytr, n_qubits, N_LAYERS, epochs=EPOCHS, seed=seed, entangler=entangler
    )
    cal = _quantum_logits(theta, W, b, Xcal, n_qubits, N_LAYERS, entangler)
    test = _quantum_logits(theta, W, b, Xte, n_qubits, N_LAYERS, entangler)
    T = fit_temperature(cal, ycal)
    return np.array([softmax(r / T) for r in test]), float(T)


def _summarise(values: list[float]) -> dict:
    from scipy import stats

    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_p = float(stats.ttest_1samp(a, 0.0).pvalue)
    try:
        w_p = float(stats.wilcoxon(a).pvalue)
    except ValueError:
        w_p = 1.0
    return {
        "mean": round(mean, 6),
        "sd_across_seeds": round(sd, 6),
        "ci95_mean": [round(mean - tcrit * se, 6), round(mean + tcrit * se, 6)],
        "excludes_zero": bool((mean - tcrit * se) * (mean + tcrit * se) > 0),
        "t_p_two_sided": round(t_p, 6),
        "wilcoxon_p": round(w_p, 6),
        "n_seeds_negative": int((a < 0).sum()),
        "n_seeds_positive": int((a > 0).sum()),
    }


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for r, (k, p) in enumerate(items):
        adj = max(min(1.0, p * (m - r)), prev)
        prev = adj
        out[k] = round(adj, 6)
    return out


def main() -> None:
    ensure_dirs()
    started = time.time()
    Xtr8, ytr, Xcal8, ycal, Xte8, yte, source = load_evidence(n=1338, seed=7)

    result = {
        "purpose": (
            "Murphy/DeGroot-Fienberg decomposition of the multiclass Brier score for the "
            "entanglement ablation. Answers whether the ring trades refinement for "
            "calibration, which Section 8.3 previously inferred from the sign of ECE "
            "rather than measured. Forecasts are grouped by k-means on the simplex and "
            "the reconstruction residual is reported, because the decomposition identity "
            "is exact only for identical forecast vectors."
        ),
        "protocol": {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "n_layers": N_LAYERS,
            "widths": list(WIDTHS),
            "group_counts": list(GROUP_COUNTS),
            "grouping": "k-means on the predicted probability vector",
            "data": "cached quantum_study_evidence.npz",
            "n_train": int(len(ytr)),
            "n_cal": int(len(ycal)),
            "n_test": int(len(yte)),
            "source": source,
            "note": (
                "7-qubit runs drop the constant prior_risk channel (channel 8), which is "
                "identically zero across all splits."
            ),
        },
        "by_width": {},
    }

    def checkpoint() -> None:
        """Write after every seed. A 50-minute run that loses everything to a
        killed shell is a run that gets skipped next time."""
        OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for width in WIDTHS:
        Xtr, Xcal, Xte = Xtr8[:, :width], Xcal8[:, :width], Xte8[:, :width]
        per_seed = []
        result["by_width"][f"{width}q"] = {"per_seed": per_seed, "across_seeds": {},
                                           "complete": False}
        for seed in SEEDS:
            t0 = time.time()
            probs_r, T_r = _train_arm(Xtr, ytr, Xcal, ycal, Xte, width, "ring", seed)
            probs_p, T_p = _train_arm(Xtr, ytr, Xcal, ycal, Xte, width, "none", seed)

            entry = {"seed": seed, "temperature_ring": round(T_r, 4),
                     "temperature_product": round(T_p, 4), "by_group_count": {}}
            for k in GROUP_COUNTS:
                dr = murphy_decomposition(probs_r, yte, k)
                dp = murphy_decomposition(probs_p, yte, k)
                entry["by_group_count"][str(k)] = {
                    "ring": dr,
                    "product": dp,
                    "delta_reliability": round(dr["reliability"] - dp["reliability"], 6),
                    "delta_resolution": round(dr["resolution"] - dp["resolution"], 6),
                    "delta_brier_actual": round(dr["actual_brier"] - dp["actual_brier"], 6),
                    "delta_residual": round(dr["residual"] - dp["residual"], 6),
                }
            per_seed.append(entry)
            print(f"[{width}q seed {seed}] done in {time.time() - t0:.1f}s "
                  f"dBrier={entry['by_group_count']['10']['delta_brier_actual']:+.4f} "
                  f"dREL={entry['by_group_count']['10']['delta_reliability']:+.4f} "
                  f"dRES={entry['by_group_count']['10']['delta_resolution']:+.4f}",
                  flush=True)
            checkpoint()

        across = {}
        for k in GROUP_COUNTS:
            kk = str(k)
            summ = {
                "delta_reliability": _summarise([s["by_group_count"][kk]["delta_reliability"] for s in per_seed]),
                "delta_resolution": _summarise([s["by_group_count"][kk]["delta_resolution"] for s in per_seed]),
                "delta_brier_actual": _summarise([s["by_group_count"][kk]["delta_brier_actual"] for s in per_seed]),
            }
            holm = _holm({m: summ[m]["wilcoxon_p"] for m in summ})
            for m in summ:
                summ[m]["wilcoxon_p_holm"] = holm[m]
            summ["mean_abs_residual_ring"] = round(float(np.mean(
                [abs(s["by_group_count"][kk]["ring"]["residual"]) for s in per_seed])), 6)
            summ["mean_abs_residual_product"] = round(float(np.mean(
                [abs(s["by_group_count"][kk]["product"]["residual"]) for s in per_seed])), 6)
            across[kk] = summ

        result["by_width"][f"{width}q"]["across_seeds"] = across
        result["by_width"][f"{width}q"]["complete"] = True
        checkpoint()

    checkpoint()
    print(f"\n[done] wrote {OUTPUT} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
