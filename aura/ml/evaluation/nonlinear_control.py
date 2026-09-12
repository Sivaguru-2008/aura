"""Positive control: does the comparison apparatus detect a nonlinear advantage
when one exists?

Why this exists
---------------
Section 5.1.2 establishes that the six-way clinical target is an argmax over a
fixed *weighted sum* of the evidence concepts, so the Bayes-optimal rule for it
is linear in the evidence vector. Section 11.2 concedes the consequence: a task
with no inter-channel structure gives an entangling ansatz nothing to exploit,
which makes the null of Sections 8.2 to 8.4 a *predicted* outcome. A reviewer is
entitled to sharpen that concession into an objection:

    "You have shown that a nonlinear model does not beat a linear model on a task
     you have proved is linear. That is a theorem restated, not an experiment."

Conceding the objection is not answering it. This script answers it.

The design
----------
We hold **everything** fixed except the structure of the target:

* the same evidence vectors — real MIMIC-CXR model outputs, not simulated inputs;
* the same patient-disjoint splits, the same sizes, the same seed;
* the same four arms;
* the same matched optimiser protocol of ``matched_optimiser.py`` — same optimiser,
  batch size, epoch cap, learning-rate grid, early stopping, weight decay;
* the same per-arm temperature calibration (P1) and the same tests.

and we swap the target for one that *does* carry higher-order structure among the
evidence channels. Two are used:

**T2a, conjunctive.** Clinically shaped rules in which a category requires a
*combination* of findings rather than a weighted sum of them — heart failure needs
oedema **and** cardiomegaly; pneumonia needs consolidation **and** opacity **and
not** effusion; and so on. Conjunction and negated conjunction are the structures
a weighted sum cannot represent, and they are the structures a clinician would
describe when asked why a constellation, rather than any single finding, makes a
diagnosis.

**T2b, generic pairwise.** ``score_k(x) = a_k·x + x' M_k x`` with a fixed random
``M_k``, so that the conclusion does not depend on the particular rules chosen
for T2a.

The interpretation
------------------
For each target we first measure the **linear ceiling**: the accuracy a
multinomial logistic regression achieves against the noise-free target on the
whole corpus. On the real clinical target that ceiling is 1.0 by construction —
that *is* Section 5.1.2's claim, and measuring it here is the first empirical
check of it this project has run. If the ceiling on T2a and T2b is well below
1.0 while an explicit interaction-feature model reaches it, the target genuinely
requires higher-order structure.

Then the four arms are fitted and compared. The outcome decides how Section 8's
null should be read:

* If the nonlinear arms beat the linear arms on T2a/T2b, the apparatus **has**
  the sensitivity to detect a nonlinear advantage, and the null on the clinical
  target is an informative negative about that target rather than an artifact of
  the measurement.
* If they do not, the apparatus lacks that sensitivity at this scale, and every
  null in Section 8 must be reported as uninformative. That outcome would be
  worse for the paper, and it is reported either way (P5).

What this does **not** establish: T2a and T2b are semi-synthetic targets. Real
inputs, constructed labels. They say what the *comparison* can detect. They say
nothing about clinical performance and are not offered as evidence of it.

Writes ``artifacts/nonlinear_control.json``.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.safety.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from aura.ml.evaluation.quantum_study import load_evidence
from aura.ml.evaluation.matched_optimiser import (
    LR_GRID,
    MAX_EPOCHS,
    BATCH,
    PATIENCE,
    WEIGHT_DECAY,
    SEED,
    _build_gated,
    _build_linear,
    _build_quantum,
    _holm,
    _lbfgs_anchor,
    _mcnemar,
    _per_study_nll,
    _search,
)

OUTPUT = ARTIFACTS / "nonlinear_control.json"

N_DX = len(DIAGNOSES)

#: Channel order of the evidence vector, from ``quantum_study_evidence.npz``.
CHANNELS = ("opacity", "consolidation", "effusion", "cardiomegaly",
            "nodule", "hyperinflation", "pneumothorax", "prior_risk")
IDX = {name: i for i, name in enumerate(CHANNELS)}


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def target_linear(X: np.ndarray) -> np.ndarray:
    """T1: the clinical target's generating rule, applied to the evidence vector.

    This reproduces ``aura/mimic/labeling.py::_map_diagnosis`` as a function of the
    evidence channels — the weighted sum of Section 5.1.2 — so that the linear
    ceiling reported for it is a measurement of that section's claim rather than a
    restatement of it.
    """
    g = lambda n: X[:, IDX[n]]  # noqa: E731
    scores = np.stack([
        g("pneumothorax") * 1.0,                                        # Pneumothorax
        g("nodule") * 1.0,                                              # Malignancy
        np.zeros(len(X)) + 0.5 * g("cardiomegaly") + 0.2 * g("effusion"),  # Heart failure
        0.6 * g("consolidation") + 0.3 * g("opacity"),                  # Pneumonia
        0.9 * g("hyperinflation"),                                      # COPD
        np.full(len(X), 0.15),                                          # Normal floor
    ], axis=1)
    return scores.argmax(axis=1)


#: Per-channel binarisation thresholds for T2a, as quantiles of the corpus.
#: A fixed absolute threshold of 0.5 was tried first and produced a degenerate
#: target: the vision model's channel outputs sit mostly below 0.5, so every
#: conjunction fired rarely and 92% of studies fell through to the "normal"
#: fallback. Quantile thresholds make each term fire at a known rate, so the
#: conjunctions are actually exercised. The threshold is a property of the target's
#: *definition*, computed once over the whole corpus, not a fitted parameter — no
#: model sees it and it is identical across splits.
T2A_QUANTILE = 0.5


def target_conjunctive(X: np.ndarray, thresholds: np.ndarray | None = None
                       ) -> np.ndarray:
    """T2a: categories that require a *combination* of findings.

    Each rule is a conjunction, or a conjunction with a negated term. Neither is
    representable by a weighted sum of the same channels, which is the property
    this control needs.
    """
    if thresholds is None:
        thresholds = np.quantile(X, T2A_QUANTILE, axis=0)
    b = (X >= thresholds[None, :]).astype(float)
    g = lambda n: b[:, IDX[n]]  # noqa: E731
    scores = np.stack([
        # Pneumothorax: the one genuinely single-finding category, kept linear so
        # the target is not uniformly nonlinear.
        g("pneumothorax"),
        # Malignancy: a nodule that is NOT accompanied by consolidation (a nodule
        # with consolidation reads as infection, not tumour).
        g("nodule") * (1.0 - g("consolidation")),
        # Heart failure: cardiomegaly AND effusion.
        g("cardiomegaly") * g("effusion"),
        # Pneumonia: consolidation AND opacity AND NOT effusion.
        g("consolidation") * g("opacity") * (1.0 - g("effusion")),
        # COPD: hyperinflation AND NOT opacity.
        g("hyperinflation") * (1.0 - g("opacity")),
        # Normal: everything below threshold.
        np.full(len(X), 0.5),
    ], axis=1)
    return scores.argmax(axis=1)


def target_pairwise(X: np.ndarray, seed: int = 11) -> np.ndarray:
    """T2b: ``score_k(x) = a_k . x + x' M_k x`` with fixed random coefficients.

    Generic higher-order structure, so that the conclusion does not rest on the
    particular rules of T2a. ``M_k`` is symmetric with a zero diagonal, so every
    nonlinear term is a genuine *interaction* between two distinct channels and
    not a squared single channel (which a monotone model could absorb).
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    a = rng.normal(0, 0.5, size=(N_DX, d))
    M = rng.normal(0, 1.5, size=(N_DX, d, d))
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    for k in range(N_DX):
        np.fill_diagonal(M[k], 0.0)
    lin = X @ a.T
    quad = np.einsum("ni,kij,nj->nk", X, M, X)
    return (lin + quad).argmax(axis=1)


# --------------------------------------------------------------------------- #
# Ceilings — is the target actually nonlinear?
# --------------------------------------------------------------------------- #
def _ceilings(X: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy of a linear and an interaction model fitted to the *noise-free*
    target on the whole corpus.

    This is deliberately fitted without a held-out split: the question is not
    generalisation but representability — can this model class express this map at
    all? A gap between the two ceilings is the definition of higher-order
    structure, measured rather than asserted.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures

    lin = LogisticRegression(C=1e4, solver="lbfgs", max_iter=20000, tol=1e-10)
    lin.fit(X, y)
    lin_acc = float((lin.predict(X) == y).mean())

    Z = PolynomialFeatures(degree=2, interaction_only=True,
                           include_bias=False).fit_transform(X)
    quad = LogisticRegression(C=1e4, solver="lbfgs", max_iter=20000, tol=1e-10)
    quad.fit(Z, y)
    quad_acc = float((quad.predict(Z) == y).mean())

    counts = np.bincount(y, minlength=N_DX)
    return {
        "linear_ceiling": round(lin_acc, 4),
        "interaction_ceiling": round(quad_acc, 4),
        "gap": round(quad_acc - lin_acc, 4),
        "majority_class": round(float(counts.max() / len(y)), 4),
        "class_support": {DIAGNOSES[i]: int(counts[i]) for i in range(N_DX)},
        "note": ("Fitted on the noise-free target over the whole corpus. This "
                 "measures representability, not generalisation: a linear ceiling "
                 "below the interaction ceiling is higher-order structure the "
                 "linear model provably cannot express."),
    }


# --------------------------------------------------------------------------- #
def _evaluate(logits_cal, logits_test, ycal, yte) -> dict:
    T = fit_temperature(logits_cal, ycal)
    P = np.array([softmax(r / T) for r in logits_test])
    correct = P.argmax(axis=1) == yte
    nll = _per_study_nll(P, yte)
    return {
        "n": int(len(yte)),
        "temperature": round(float(T), 4),
        "accuracy": round(float(correct.mean()), 4),
        "correct": int(correct.sum()),
        "nll": round(float(nll.mean()), 4),
        "ece": round(float(expected_calibration_error(P, yte)), 4),
        "_correct": correct,
        "_nll": nll,
    }


def _run_target(name: str, description: str, Xtr, ytr, Xcal, ycal, Xte, yte,
                X_all, y_all) -> dict:
    print(f"\n=== {name}: {description} ===", flush=True)
    ceilings = _ceilings(X_all, y_all)
    print(f"  linear ceiling {ceilings['linear_ceiling']:.4f}  "
          f"interaction ceiling {ceilings['interaction_ceiling']:.4f}  "
          f"gap {ceilings['gap']:+.4f}", flush=True)

    d = int(Xtr.shape[1])
    arms, params, fitting = {}, {}, {}

    fit, Lcal, Lte, p = _search(_build_linear(d), Xtr, ytr, Xcal, ycal, Xte,
                                f"{name}/linear")
    arms["classical"] = _evaluate(Lcal, Lte, ycal, yte)
    params["classical"], fitting["classical"] = p, fit

    fit, Lcal, Lte, p = _search(_build_gated(d), Xtr, ytr, Xcal, ycal, Xte,
                                f"{name}/gated")
    arms["learnable"] = _evaluate(Lcal, Lte, ycal, yte)
    params["learnable"], fitting["learnable"] = p, fit

    fit, Lcal, Lte, p = _search(_build_quantum(), Xtr, ytr, Xcal, ycal, Xte,
                                f"{name}/quantum")
    arms["quantum"] = _evaluate(Lcal, Lte, ycal, yte)
    params["quantum"], fitting["quantum"] = p, fit

    Lcal, Lte, p, anchor = _lbfgs_anchor(Xtr, ytr, Xcal, ycal, Xte)
    arms["logistic"] = _evaluate(Lcal, Lte, ycal, yte)
    params["logistic"], fitting["logistic"] = p, anchor

    names = ["classical", "quantum", "learnable", "logistic"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pairwise, acc_p, nll_p = {}, {}, {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        A, B = arms[a], arms[b]
        mc = _mcnemar(A["_correct"], B["_correct"])
        diff = A["_nll"] - B["_nll"]
        try:
            w = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            w = 1.0
        pairwise[key] = {
            "delta_accuracy": round(float(A["accuracy"] - B["accuracy"]), 4),
            "mcnemar": mc,
            "delta_nll": round(float(A["nll"] - B["nll"]), 4),
            "nll_wilcoxon_p": float(f"{w:.3e}"),
        }
        acc_p[key], nll_p[key] = mc["p"], w
    for key, adj in _holm(acc_p).items():
        pairwise[key]["mcnemar_p_holm"] = adj
    for key, adj in _holm(nll_p).items():
        pairwise[key]["nll_wilcoxon_p_holm"] = float(f"{adj:.3e}")

    # The question this control exists to answer, reduced to two booleans.
    nonlinear_best_acc = max(arms["quantum"]["accuracy"], arms["learnable"]["accuracy"])
    linear_best_acc = max(arms["classical"]["accuracy"], arms["logistic"]["accuracy"])
    q_vs_log = pairwise["quantum_vs_logistic"]
    l_vs_log = pairwise["learnable_vs_logistic"]
    verdict = {
        "nonlinear_arms_lead_on_accuracy": bool(nonlinear_best_acc > linear_best_acc),
        "accuracy_margin": round(float(nonlinear_best_acc - linear_best_acc), 4),
        "quantum_beats_logistic_accuracy_holm": bool(
            q_vs_log["delta_accuracy"] > 0 and q_vs_log["mcnemar_p_holm"] < 0.05),
        "gated_beats_logistic_accuracy_holm": bool(
            l_vs_log["delta_accuracy"] > 0 and l_vs_log["mcnemar_p_holm"] < 0.05),
    }

    print(f"  {'arm':10s} {'params':>6s} {'acc':>7s} {'nll':>7s} {'ece':>7s}", flush=True)
    for k in names:
        a = arms[k]
        print(f"  {k:10s} {params[k]:6d} {a['accuracy']:7.4f} {a['nll']:7.4f} "
              f"{a['ece']:7.4f}", flush=True)

    return {
        "description": description,
        "ceilings": ceilings,
        "trainable_parameters": params,
        "fitting": fitting,
        "backends": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in arms.items()},
        "pairwise": pairwise,
        "verdict": verdict,
    }


def main() -> None:
    ensure_dirs()
    started = time.time()
    Xtr, ytr, Xcal, ycal, Xte, yte, source = load_evidence(n=1338, seed=7)
    X_all = np.concatenate([Xtr, Xcal, Xte])

    # The T2a thresholds are part of the target's definition and must be identical
    # across splits. Computing them inside the target function would silently give
    # each split its own definition.
    t2a_thresholds = np.quantile(X_all, T2A_QUANTILE, axis=0)

    targets = {
        "T1_clinical_linear": (
            "The clinical target's generating rule (Section 5.1.2): argmax over a "
            "weighted sum of evidence channels. Included to measure the linear "
            "ceiling that Section 5.1.2 asserts is 1.0.",
            target_linear),
        "T2a_conjunctive": (
            "Categories requiring a combination of findings — conjunction and "
            "negated conjunction — which a weighted sum cannot represent. Channels "
            f"are binarised at their corpus median (quantile {T2A_QUANTILE}).",
            lambda X: target_conjunctive(X, t2a_thresholds)),
        "T2b_pairwise": (
            "Generic pairwise interaction target: a_k.x + x' M_k x with fixed "
            "random symmetric M_k, zero diagonal.",
            target_pairwise),
    }

    results = {}
    for name, (desc, fn) in targets.items():
        ytr_t, ycal_t, yte_t = fn(Xtr), fn(Xcal), fn(Xte)
        y_all = fn(X_all)
        # A target with a class absent from a split cannot be compared on that
        # split; report it rather than silently dropping the class.
        missing = [DIAGNOSES[i] for i in range(N_DX)
                   if not (ytr_t == i).any() or not (yte_t == i).any()]
        res = _run_target(name, desc, Xtr, ytr_t, Xcal, ycal_t, Xte, yte_t,
                          X_all, y_all)
        res["classes_absent_from_a_split"] = missing
        results[name] = res

    # The cross-target comparison is the actual finding.
    summary = {
        "question": ("Does the comparison apparatus detect a nonlinear advantage "
                     "when the target contains one?"),
        "by_target": {
            name: {
                "linear_ceiling": r["ceilings"]["linear_ceiling"],
                "interaction_ceiling": r["ceilings"]["interaction_ceiling"],
                "nonlinear_arms_lead": r["verdict"]["nonlinear_arms_lead_on_accuracy"],
                "accuracy_margin": r["verdict"]["accuracy_margin"],
                "quantum_beats_logistic_holm":
                    r["verdict"]["quantum_beats_logistic_accuracy_holm"],
                "gated_beats_logistic_holm":
                    r["verdict"]["gated_beats_logistic_accuracy_holm"],
            } for name, r in results.items()
        },
    }

    result = {
        "purpose": (
            "Positive control for the comparison protocol. Holds the evidence "
            "vectors, splits, arms, optimiser protocol and tests fixed, and varies "
            "only whether the target carries higher-order structure among the "
            "evidence channels. Establishes whether the null of Sections 8.2 to 8.4 "
            "is a property of the task or a limit of the measurement. Addresses "
            "Section 11.2."
        ),
        "scope_limitation": (
            "T2a and T2b are semi-synthetic: real evidence vectors, constructed "
            "labels. They characterise the sensitivity of the comparison and carry "
            "no clinical interpretation whatsoever."
        ),
        "protocol": {
            "optimiser": "Adam", "learning_rate_grid": list(LR_GRID),
            "batch_size": BATCH, "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE, "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "identical_to": "aura/ml/evaluation/matched_optimiser.py",
        },
        "data": {"source": source, "train_studies": int(len(ytr)),
                 "calibration_studies": int(len(ycal)),
                 "test_studies": int(len(yte)),
                 "evidence_channels": int(Xtr.shape[1]),
                 "channel_names": list(CHANNELS)},
        "summary": summary,
        "targets": results,
        "runtime_seconds": round(time.time() - started, 1),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for name, s in summary["by_target"].items():
        print(f"{name:22s} linear_ceiling={s['linear_ceiling']:.4f} "
              f"interaction_ceiling={s['interaction_ceiling']:.4f} "
              f"nonlinear_lead={str(s['nonlinear_arms_lead']):5s} "
              f"margin={s['accuracy_margin']:+.4f}")
    print(f"\n[done] wrote {OUTPUT} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
