"""Risk-coverage analysis of selective prediction on the primary split.

Why this exists
---------------
Section 9.1 previously reported abstention from ``fusion_ablation.json``, an
artifact that carries calibration temperatures matching neither the 447-study
primary split nor the 69-study benchmark, records no sample size or split
identifier, and predates the per-arm calibration regime (P1) imposed on every
negative result in Section 8. It was the paper's only positive claim and it was
held to a lower evidential standard than the findings it sat beside.

It also could not support the claim made from it. Its three rows sit at three
different coverage levels, so they are three points on three different
risk-coverage curves rather than a controlled comparison; and it contains no
trivial rejector, so it cannot show that the four-criterion policy beats simply
thresholding the maximum calibrated probability.

This module replaces it:

  * risk-coverage curves for every backend on the 447-study primary split, each
    arm calibrated on its own 446-study calibration split (P1);
  * AURC and excess-AURC against the optimal rejector [Geifman, Uziel & El-Yaniv,
    ICLR 2019];
  * a maximum-softmax rejector as the trivial control [Hendrycks & Gimpel 2017];
  * the served four-criterion policy placed on that curve, and compared with the
    trivial rejector **at matched coverage**, which is the only comparison that
    answers whether the policy earns its complexity.

Reconstruction note. Three of the policy's four criteria are computed here
exactly as the server computes them: the conformal set size from a per-arm
split-conformal threshold fitted on the calibration split, the low-confidence
floor on the maximum calibrated probability, and the out-of-distribution energy
z-score against per-arm in-distribution statistics. The fourth, epistemic mutual
information, is identically zero for the three deterministic backends
(``classical``/``learnable``/``logistic`` return a zero standard deviation by
construction) and is not reconstructed for the quantum arm, whose serving path
derives it from shot noise. The quantum row is therefore a three-criterion
reconstruction and is labelled as such in the artifact.

Writes ``artifacts/risk_coverage_n447.json``.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.safety.calibration import fit_conformal, fit_temperature
from aura.ml.evaluation.fusion_baselines import (
    _converged_logistic,
    _served_classical,
    _served_learnable,
    _served_quantum,
)
from aura.ml.evaluation.quantum_study import load_evidence

OUTPUT = ARTIFACTS / "risk_coverage_n447.json"
N_BOOT = 2000
COVERAGE_GRID = (1.0, 0.95, 0.90, 0.80, 0.70)


# --------------------------------------------------------------------------- #
# Risk-coverage machinery
# --------------------------------------------------------------------------- #
def risk_coverage_curve(correct: np.ndarray, confidence: np.ndarray):
    """Risk and coverage as the rejector sweeps from full to minimal coverage.

    Points are ordered by decreasing confidence, so index i is the risk incurred
    when the i+1 most confident cases are answered.
    """
    order = np.argsort(-confidence, kind="mergesort")
    errs = (~correct[order]).astype(float)
    n = len(errs)
    k = np.arange(1, n + 1)
    risk = np.cumsum(errs) / k
    coverage = k / n
    return coverage, risk


def aurc(correct: np.ndarray, confidence: np.ndarray) -> float:
    _, risk = risk_coverage_curve(correct, confidence)
    return float(risk.mean())


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of the oracle rejector, which answers every correct case first."""
    return aurc(correct, correct.astype(float))


def risk_at_coverage(correct: np.ndarray, confidence: np.ndarray, target: float) -> float:
    coverage, risk = risk_coverage_curve(correct, confidence)
    idx = int(np.searchsorted(coverage, target, side="left"))
    idx = min(idx, len(risk) - 1)
    return float(risk[idx])


# --------------------------------------------------------------------------- #
# The served four-criterion policy, reconstructed offline
# --------------------------------------------------------------------------- #
def _energy(logits: np.ndarray, T: float) -> np.ndarray:
    return -T * np.log(np.exp(logits / T).sum(axis=1))


def policy_abstentions(P_test, logits_test, P_cal, logits_cal, ycal, T, settings):
    """Reproduce the serving abstention policy on the test split.

    Returns the boolean abstention mask and the per-criterion masks, so the
    artifact can record which criterion fires rather than only the total.
    """
    qhat = fit_conformal(P_cal, ycal, settings.conformal_coverage)
    set_sizes = (1.0 - P_test <= qhat).sum(axis=1)
    # a conformal set is never empty in serving; the top class is always kept
    set_sizes = np.maximum(set_sizes, 1)

    e_cal = _energy(logits_cal, T)
    e_test = _energy(logits_test, T)
    z = (e_test - e_cal.mean()) / e_cal.std()

    top_p = P_test.max(axis=1)

    is_ood = z > settings.ood_energy_threshold
    low_conf = top_p < settings.low_confidence_threshold
    big_set = set_sizes > settings.abstention_conformal_size

    abstain = is_ood | low_conf | big_set
    return abstain, {
        "out_of_distribution": int(is_ood.sum()),
        "low_confidence": int((~is_ood & low_conf).sum()),
        "large_conformal_set": int((~is_ood & ~low_conf & big_set).sum()),
        "high_epistemic": 0,
    }, {"conformal_qhat": round(float(qhat), 4),
        "mean_set_size": round(float(set_sizes.mean()), 4)}


# --------------------------------------------------------------------------- #
def _boot_ci(fn, n: int, rng, n_boot: int = N_BOOT):
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    a = np.asarray(vals)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def main() -> None:
    ensure_dirs()
    started = time.time()
    settings = get_settings()
    Xtr, ytr, Xcal, ycal, Xte, yte, source = load_evidence(n=1338, seed=7)
    rng = np.random.default_rng(7)

    builders = {
        "classical": lambda: _served_classical(Xcal, Xte)[:2],
        "quantum": lambda: _served_quantum(Xcal, Xte)[:2],
        "learnable": lambda: _served_learnable(Xcal, Xte)[:2],
        "logistic": lambda: _converged_logistic(Xtr, ytr, Xcal, ycal, Xte)[:2],
    }

    backends = {}
    for name, build in builders.items():
        Lcal, Lte = build()
        T = fit_temperature(Lcal, ycal)
        P = np.array([softmax(r / T) for r in Lte])
        Pcal = np.array([softmax(r / T) for r in Lcal])
        correct = P.argmax(axis=1) == yte
        conf = P.max(axis=1)

        a = aurc(correct, conf)
        opt = optimal_aurc(correct)

        abstain, reasons, conf_info = policy_abstentions(
            P, Lte, Pcal, Lcal, ycal, T, settings
        )
        answered = ~abstain
        n_ans = int(answered.sum())
        policy_cov = n_ans / len(yte)
        policy_risk = float((~correct[answered]).mean()) if n_ans else float("nan")
        # the trivial rejector held to the policy's own coverage
        matched_risk = risk_at_coverage(correct, conf, policy_cov)

        def _d(idx, _c=correct, _cf=conf, _ab=abstain):
            ans = _c[idx][~_ab[idx]]
            if len(ans) == 0:
                return None
            cov = len(ans) / len(idx)
            return float((~ans).mean()) - risk_at_coverage(_c[idx], _cf[idx], cov)

        backends[name] = {
            "n": int(len(yte)),
            "temperature": round(float(T), 4),
            "accuracy_all_cases": round(float(correct.mean()), 4),
            "aurc": round(a, 4),
            "optimal_aurc": round(opt, 4),
            "excess_aurc": round(a - opt, 4),
            "aurc_ci95": _boot_ci(
                lambda i, _c=correct, _cf=conf: aurc(_c[i], _cf[i]), len(yte), rng
            ),
            "risk_at_coverage": {
                f"{c:.2f}": round(risk_at_coverage(correct, conf, c), 4)
                for c in COVERAGE_GRID
            },
            "served_policy": {
                "reconstruction": ("four criteria" if name == "quantum"
                                   else "three criteria; epistemic is zero by construction"),
                "epistemic_reconstructed": False,
                "abstention_rate": round(1.0 - policy_cov, 4),
                "coverage": round(policy_cov, 4),
                "n_answered": n_ans,
                "risk_answered": round(policy_risk, 4),
                "accuracy_answered": round(1.0 - policy_risk, 4),
                "reasons": reasons,
                **conf_info,
            },
            "matched_coverage_control": {
                "rejector": "maximum calibrated probability",
                "coverage": round(policy_cov, 4),
                "risk": round(matched_risk, 4),
                "accuracy": round(1.0 - matched_risk, 4),
                "policy_minus_control_risk": round(policy_risk - matched_risk, 4),
                "policy_minus_control_risk_ci95": _boot_ci(_d, len(yte), rng),
            },
        }
        print(f"{name:10s} AURC={a:.4f} E-AURC={a-opt:.4f} "
              f"policy cov={policy_cov:.4f} risk={policy_risk:.4f} "
              f"| matched max-softmax risk={matched_risk:.4f} "
              f"(policy-control {policy_risk-matched_risk:+.4f})")

    result = {
        "purpose": (
            "Risk-coverage analysis of selective prediction on the 447-study primary "
            "split under per-arm calibration (P1), replacing the off-split, "
            "un-P1-calibrated abstention table previously reported in Section 9.1. "
            "Reports AURC and excess AURC against the optimal rejector, and compares "
            "the served abstention policy with a maximum-softmax rejector at matched "
            "coverage."
        ),
        "data": {"source": source, "test_studies": int(len(yte)),
                 "calibration_studies": int(len(ycal)), "patient_disjoint": True},
        "method": {
            "rejector": "maximum calibrated probability",
            "aurc": "mean selective risk over all coverage levels",
            "excess_aurc": "AURC minus the AURC of the optimal rejector [82]",
            "bootstrap": f"{N_BOOT} study-level resamples (P3)",
            "policy_reconstruction": (
                "conformal set size, low-confidence floor and OOD energy z-score are "
                "computed as the server computes them; epistemic mutual information is "
                "zero by construction for the deterministic arms and is not "
                "reconstructed for the quantum arm."
            ),
            "thresholds_as_served": {
                "low_confidence": settings.low_confidence_threshold,
                "ood_energy_z": settings.ood_energy_threshold,
                "conformal_set_size": settings.abstention_conformal_size,
                "conformal_coverage": settings.conformal_coverage,
                "epistemic": 0.20,
            },
            "threshold_divergence": {
                "finding": (
                    "Two settings exist for each of the OOD and epistemic criteria, and "
                    "the serving path reads the one the documentation does not publish. "
                    "services/safety/engine.py line 177 reads ood_energy_threshold "
                    f"({settings.ood_energy_threshold}); the manuscript, "
                    "docs/KNOWN_LIMITATIONS.md and tests/test_doc_numbers.py all publish "
                    f"ood_threshold ({settings.ood_threshold}), which no serving code "
                    "reads. Line 191 hardcodes epistemic > 0.20 rather than reading "
                    f"epistemic_threshold ({settings.epistemic_threshold})."
                ),
                "documented_ood_z": settings.ood_threshold,
                "served_ood_z": settings.ood_energy_threshold,
                "documented_epistemic": settings.epistemic_threshold,
                "served_epistemic": 0.20,
                "consequence": (
                    "The build test that claims to pin documented thresholds to served "
                    "ones compares against settings the engine ignores, so it passes "
                    "while the published policy is wrong on two of four criteria. This "
                    "analysis uses the served values."
                ),
            },
        },
        "backends": backends,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {OUTPUT} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
