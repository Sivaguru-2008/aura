"""Quantum kernel against a classical kernel, on the full subject corpus.

Why this exists
---------------
Section 8.5 compares a 10-qubit fidelity kernel against an RBF kernel for glioma
grading and reports that the two do not differ. Section 11.4 then states that the
comparison "establishes nothing in either direction", for three reasons that are all
properties of the experiment rather than of the kernels:

1. **Eleven test subjects.** The converged embedding dump covers 55 of the corpus's
   185 subjects, so a 60/20/20 subject split leaves 11 subjects to test on. The
   minimum detectable AUROC difference at that size is roughly 0.10; the measured
   difference was 0.045. The point estimate changed sign between two runs.
2. **A single split.** Hyperparameters were selected on one calibration fold and
   evaluated on one test fold, so the selection is not separated from the estimate.
3. **Asymmetric tuning.** Section 11.5: the grid gave the RBF arm a bandwidth
   (``gamma``) and gave the quantum arm nothing comparable, then moved the RBF arm to
   the extreme of both ranges. Identical *procedure* was applied to arms that did not
   have identical *parameters to search*, which is not symmetry.

This script removes all three.

**Coverage.** It reads ``epoch_001.npz``, which holds all 25,268 slices from all 185
subjects at the same ``best_brain_model.pt`` checkpoint the 55-subject dump was taken
from. Nothing is retrained; this is the re-export Section 11.9 item 2 asked for, and
it was already on disk.

**Nested cross-validation.** Five outer subject-level folds, stratified by grade.
Within each fold's training subjects, an inner calibration split selects
hyperparameters and fits Platt scaling. No subject contributes to both the selection
and the estimate of any fold, and every one of the 185 subjects appears in exactly one
test fold. The reported figure is therefore an out-of-fold estimate over the whole
corpus rather than a one-split point estimate.

**Symmetric tuning.** Both arms search a 5 x 5 grid of the same shape: five values of
the SVM regularisation ``C``, and five values of a kernel bandwidth. For the RBF arm
the bandwidth is ``gamma``. For the quantum arm it is a scaling of the projection that
feeds the feature map — the quantum-kernel bandwidth of Shaydulin and Wild, which is
precisely the parameter whose absence they identify as the usual cause of an
apparent quantum deficit. Giving the quantum arm no bandwidth was not neutrality; it
was handicapping it, and Section 8.5's null was measured under that handicap.

**Unit of analysis.** Grade is a property of a subject, not of a slice, so the primary
estimate aggregates each subject's slice probabilities to one subject-level
probability. Slice-level figures are reported as secondary and should not be read as
independent observations. AUROC differences are tested by DeLong on the paired
subject-level predictions.

Writes ``artifacts/brain/qkl_full_coverage.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from aura.backend.engines.neuro.qkl import QKLClassifier
from aura.common.config import ARTIFACTS
from aura.ml.evaluation.delong import delong_test
from aura.ml.training.train_qkl import (
    GRADE_CLASSES,
    binary_metrics,
    expected_calibration_error,
    fit_platt,
    load_embeddings,
)

EMBED_DIR = ARTIFACTS / "brain" / "embeddings"
OUTPUT = ARTIFACTS / "brain" / "qkl_full_coverage.json"

#: Both arms search a grid of this shape. Five regularisation values, five
#: bandwidths. Same cardinality, same selection split, same selection metric.
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
RBF_GAMMA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
QUANTUM_BANDWIDTH_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)

N_QUBITS = 10
N_FOLDS = 5
FIT_SLICES = 1200          # balanced, from the fold's inner-training subjects
CAL_SLICES = 800           # balanced, from the fold's inner-calibration subjects
TEST_SLICES_PER_SUBJECT = 30
SEED = 7


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def stratified_subject_folds(subjects: np.ndarray, labels: np.ndarray,
                             k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Partition subject ids into ``k`` folds, stratified by grade.

    Stratification matters here: LGG is roughly a quarter of subjects, and an
    unstratified partition of 185 subjects into five folds can leave a fold with
    very few positives, which makes that fold's AUROC unstable for reasons that
    have nothing to do with either kernel.
    """
    subj_ids = np.unique(subjects)
    subj_label = np.asarray([labels[subjects == s][0] for s in subj_ids])

    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(subj_label):
        pool = subj_ids[subj_label == cls]
        pool = pool[rng.permutation(len(pool))]
        for i, s in enumerate(pool):
            folds[i % k].append(int(s))
    return [np.asarray(sorted(f)) for f in folds]


def balanced_slice_sample(idx: np.ndarray, labels: np.ndarray, n_total: int,
                          rng: np.random.Generator) -> np.ndarray:
    classes = np.unique(labels[idx])
    per_class = max(1, n_total // max(len(classes), 1))
    picked = []
    for cls in classes:
        pool = idx[labels[idx] == cls]
        picked.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    out = np.concatenate(picked)
    return out[rng.permutation(len(out))]


def capped_per_subject(idx: np.ndarray, subjects: np.ndarray, cap: int,
                       rng: np.random.Generator) -> np.ndarray:
    """At most ``cap`` slices per subject, so no large subject dominates."""
    picked = []
    for s in np.unique(subjects[idx]):
        pool = idx[subjects[idx] == s]
        picked.append(rng.choice(pool, size=min(cap, len(pool)), replace=False))
    return np.concatenate(picked)


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def _make_projection(X_fit: np.ndarray, n_qubits: int, seed: int, bandwidth: float):
    """Standardise, PCA to ``n_qubits``, rescale, then apply the bandwidth.

    The bandwidth multiplies the projection matrix, so it scales the argument of
    the feature map's ``tanh`` — compressing the angles toward the centre of their
    range at small values and pushing them toward saturation at large ones. This is
    the quantum-kernel bandwidth of Shaydulin and Wild, applied at the only place in
    this pipeline where an input scale exists.
    """
    from sklearn.decomposition import PCA

    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0)
    std[std < 1e-9] = 1.0
    Z = (X_fit - mean) / std
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Z)
    comp = pca.components_
    scale = np.percentile(np.abs(Z @ comp.T), 95, axis=0)
    scale[scale < 1e-9] = 1.0
    W = (comp / scale[:, None]) * float(bandwidth)
    return QKLClassifier(
        W_proj=W, b_proj=np.zeros(n_qubits), n_qubits=n_qubits,
        classes=GRADE_CLASSES, task="glioma_grade",
        feature_mean=mean, feature_scale=std,
    )


def _svm_scores(K_fit, y_fit, K_eval_list, C, seed):
    from sklearn.svm import SVC

    svm = SVC(kernel="precomputed", C=C, random_state=seed)
    svm.fit(K_fit, y_fit)
    return [svm.decision_function(K) for K in K_eval_list]


def _run_quantum_fold(X, y, fit_idx, cal_idx, te_idx, seed) -> dict:
    """Select (C, bandwidth) on the inner calibration split, then score the test."""
    y_fit, y_cal = y[fit_idx], y[cal_idx]
    best = None
    encode_seconds = 0.0

    for bw in QUANTUM_BANDWIDTH_GRID:
        clf = _make_projection(X[fit_idx], N_QUBITS, seed, bw)
        t0 = time.perf_counter()
        S_fit = clf.feature_states(clf.project_batch(X[fit_idx]))
        S_cal = clf.feature_states(clf.project_batch(X[cal_idx]))
        encode_seconds += time.perf_counter() - t0
        K_fit = np.abs(S_fit @ S_fit.conj().T) ** 2
        K_cal = np.abs(S_cal @ S_fit.conj().T) ** 2

        for C in C_GRID:
            (f_cal,) = _svm_scores(K_fit, y_fit, [K_cal], C, seed)
            score = _safe_auroc(y_cal, f_cal)
            if best is None or score > best["calibration_auroc"]:
                best = {"C": C, "bandwidth": bw, "calibration_auroc": score}

    # Refit at the selection and score the held-out fold.
    clf = _make_projection(X[fit_idx], N_QUBITS, seed, best["bandwidth"])
    t0 = time.perf_counter()
    S_fit = clf.feature_states(clf.project_batch(X[fit_idx]))
    S_cal = clf.feature_states(clf.project_batch(X[cal_idx]))
    S_te = clf.feature_states(clf.project_batch(X[te_idx]))
    encode_seconds += time.perf_counter() - t0
    K_fit = np.abs(S_fit @ S_fit.conj().T) ** 2
    K_cal = np.abs(S_cal @ S_fit.conj().T) ** 2
    K_te = np.abs(S_te @ S_fit.conj().T) ** 2

    f_cal, f_te = _svm_scores(K_fit, y_fit, [K_cal, K_te], best["C"], seed)
    p_te = _platt(f_cal, y_cal, f_te)
    return {"selection": best, "p_test": p_te,
            "encode_seconds": round(encode_seconds, 2)}


def _run_rbf_fold(X, y, fit_idx, cal_idx, te_idx, seed) -> dict:
    """Identical protocol, RBF kernel, gamma in place of the quantum bandwidth."""
    from sklearn.svm import SVC

    Xf, Xc, Xt = X[fit_idx], X[cal_idx], X[te_idx]
    y_fit, y_cal = y[fit_idx], y[cal_idx]

    mean, std = Xf.mean(axis=0), Xf.std(axis=0)
    std[std < 1e-9] = 1.0
    Zf, Zc, Zt = (Xf - mean) / std, (Xc - mean) / std, (Xt - mean) / std

    best = None
    for gamma in RBF_GAMMA_GRID:
        for C in C_GRID:
            svm = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)
            svm.fit(Zf, y_fit)
            score = _safe_auroc(y_cal, svm.decision_function(Zc))
            if best is None or score > best["calibration_auroc"]:
                best = {"C": C, "gamma": gamma, "calibration_auroc": score}

    svm = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], random_state=seed)
    svm.fit(Zf, y_fit)
    p_te = _platt(svm.decision_function(Zc), y_cal, svm.decision_function(Zt))
    return {"selection": best, "p_test": p_te}


def _platt(f_cal: np.ndarray, y_cal: np.ndarray, f_eval: np.ndarray) -> np.ndarray:
    """Fit Platt scaling on the calibration fold and apply it to ``f_eval``.

    ``train_qkl.fit_platt`` returns ``(a, b)`` for the convention that
    ``QKLClassifier`` evaluates, ``p = 1 / (1 + exp(a*f + b))`` — Platt's original
    parameterisation, in which a correctly oriented score yields a *negative* ``a``
    and the exponent is not negated again. Writing the more familiar
    ``sigma(a*f + b)`` here inverts every probability, which is a silent bug:
    the numbers stay in [0, 1], the calibration looks plausible, and only the AUROC
    betrays it by landing symmetrically below 0.5. It did exactly that on the first
    run of this script. One helper, used by both arms, so the two cannot disagree.
    """
    a, b = fit_platt(f_cal, y_cal)
    return 1.0 / (1.0 + np.exp(a * np.asarray(f_eval, dtype=float) + b))


def _safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    pos = y.astype(bool)
    r = stats.rankdata(score)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------- #
def _subject_level(p: np.ndarray, y: np.ndarray, subj: np.ndarray):
    """Mean predicted probability per subject, with that subject's grade."""
    ids = np.unique(subj)
    return (np.asarray([p[subj == s].mean() for s in ids]),
            np.asarray([y[subj == s][0] for s in ids]),
            ids)


def _bootstrap_delta(pa, pb, y, n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(_safe_auroc(y[idx], pa[idx]) - _safe_auroc(y[idx], pb[idx]))
    deltas = np.asarray(deltas)
    return {"median": round(float(np.median(deltas)), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                     round(float(np.percentile(deltas, 97.5)), 4)],
            "n_resamples": int(len(deltas))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=EMBED_DIR / "epoch_001.npz")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    started = time.time()
    rng = np.random.default_rng(args.seed)

    data = load_embeddings(args.embeddings)
    X, y, subj = data["embedding"], data["grade"], data["subject"]
    sidecar = json.loads(args.embeddings.with_suffix(".json").read_text(encoding="utf-8"))
    n_subjects = int(len(np.unique(subj)))
    print(f"[data] {args.embeddings.name}: {len(y)} slices, {n_subjects} subjects, "
          f"checkpoint {sidecar.get('checkpoint')}", flush=True)
    if n_subjects < 100:
        raise SystemExit(
            f"{args.embeddings.name} covers only {n_subjects} subjects. This script "
            f"exists to use the full-coverage dump; point --embeddings at it."
        )

    folds = stratified_subject_folds(subj, y, args.folds, rng)
    per_fold, oof = [], {"quantum": [], "rbf": [], "y": [], "subject": []}

    for f, test_subjects in enumerate(folds):
        t_fold = time.time()
        te_mask = np.isin(subj, test_subjects)
        tr_subjects = np.setdiff1d(np.unique(subj), test_subjects)

        # Inner split of the training subjects: 75% fit, 25% calibration.
        inner = stratified_subject_folds(
            subj[np.isin(subj, tr_subjects)], y[np.isin(subj, tr_subjects)], 4, rng)
        cal_subjects = inner[0]
        fit_subjects = np.concatenate(inner[1:])

        fit_pool = np.flatnonzero(np.isin(subj, fit_subjects))
        cal_pool = np.flatnonzero(np.isin(subj, cal_subjects))
        te_pool = np.flatnonzero(te_mask)

        fit_idx = balanced_slice_sample(fit_pool, y, FIT_SLICES, rng)
        cal_idx = balanced_slice_sample(cal_pool, y, CAL_SLICES, rng)
        te_idx = capped_per_subject(te_pool, subj, TEST_SLICES_PER_SUBJECT, rng)

        print(f"[fold {f}] test_subjects={len(test_subjects)} "
              f"fit={len(fit_idx)} cal={len(cal_idx)} test={len(te_idx)}", flush=True)

        q = _run_quantum_fold(X, y, fit_idx, cal_idx, te_idx, args.seed)
        r = _run_rbf_fold(X, y, fit_idx, cal_idx, te_idx, args.seed)

        yq = y[te_idx]
        qs, ys_, ids = _subject_level(q["p_test"], yq, subj[te_idx])
        rs, _, _ = _subject_level(r["p_test"], yq, subj[te_idx])

        oof["quantum"].append(qs)
        oof["rbf"].append(rs)
        oof["y"].append(ys_)
        oof["subject"].append(ids)

        rec = {
            "fold": f,
            "test_subjects": int(len(test_subjects)),
            "test_slices": int(len(te_idx)),
            "positives_subject_level": int(ys_.sum()),
            "quantum": {"selection": q["selection"],
                        "subject_auroc": round(_safe_auroc(ys_, qs), 4),
                        "slice_auroc": round(_safe_auroc(yq, q["p_test"]), 4)},
            "rbf": {"selection": r["selection"],
                    "subject_auroc": round(_safe_auroc(ys_, rs), 4),
                    "slice_auroc": round(_safe_auroc(yq, r["p_test"]), 4)},
            "seconds": round(time.time() - t_fold, 1),
        }
        rec["delta_subject_auroc"] = round(
            rec["quantum"]["subject_auroc"] - rec["rbf"]["subject_auroc"], 4)
        per_fold.append(rec)
        print(f"[fold {f}] quantum subj AUROC {rec['quantum']['subject_auroc']:.4f} "
              f"(C={q['selection']['C']}, bw={q['selection']['bandwidth']})  "
              f"rbf {rec['rbf']['subject_auroc']:.4f} "
              f"(C={r['selection']['C']}, gamma={r['selection']['gamma']})  "
              f"delta {rec['delta_subject_auroc']:+.4f}  [{rec['seconds']}s]", flush=True)

    pq = np.concatenate(oof["quantum"])
    pr = np.concatenate(oof["rbf"])
    ys = np.concatenate(oof["y"])
    ids = np.concatenate(oof["subject"])

    # Orientation guard. Two independent models cannot both land well below chance
    # by accident; if they do, the probabilities are inverted somewhere shared —
    # the Platt convention, or the label polarity. Refusing to write the artifact is
    # the point: an inverted run produces numbers that look publishable.
    q_auroc = _safe_auroc(ys, pq)
    r_auroc = _safe_auroc(ys, pr)
    if q_auroc < 0.45 and r_auroc < 0.45:
        raise SystemExit(
            f"both arms scored below chance (quantum {q_auroc:.4f}, rbf {r_auroc:.4f}). "
            f"That is an orientation bug, not a result — check the Platt convention "
            f"in _platt() and the polarity of the grade column. No artifact written."
        )

    dl = delong_test(pq, pr, ys)
    boot = _bootstrap_delta(pq, pr, ys, seed=args.seed)
    fold_deltas = [r["delta_subject_auroc"] for r in per_fold]
    try:
        fold_p = float(stats.wilcoxon(fold_deltas).pvalue)
    except ValueError:
        fold_p = 1.0

    q_metrics = binary_metrics(ys, pq)
    r_metrics = binary_metrics(ys, pr)
    q_metrics["ece"] = round(expected_calibration_error(ys, pq), 4)
    r_metrics["ece"] = round(expected_calibration_error(ys, pr), 4)

    # Minimum detectable difference at this sample size, for comparison with the
    # 11-subject experiment this supersedes.
    n_pos, n_neg = int(ys.sum()), int((~ys.astype(bool)).sum())
    result = {
        "purpose": (
            "Quantum fidelity kernel against an RBF kernel for glioma grading, on all "
            "185 subjects, under nested subject-level cross-validation and symmetric "
            "hyperparameter grids. Supersedes the 11-test-subject single-split "
            "comparison of Section 8.5, which Section 11.4 reports as establishing "
            "nothing in either direction."
        ),
        "supersedes": "artifacts/brain/qkl_training_report.json",
        "embedding_source": str(args.embeddings),
        "embedding_checkpoint": sidecar.get("checkpoint"),
        "embedding_note": (
            "Same checkpoint as the superseded run. The earlier dump covered 55 of "
            "185 subjects; this one covers all of them. No model was retrained."
        ),
        "design": {
            "outer_folds": args.folds,
            "outer_split": "subject-level, stratified by grade",
            "inner_split": "75/25 of each fold's training subjects, stratified",
            "selection_metric": "AUROC on the inner calibration split",
            "calibration": "Platt, fitted on the inner calibration split only",
            "primary_unit": "subject (grade is a subject-level property)",
            "slices_per_test_subject_cap": TEST_SLICES_PER_SUBJECT,
            "fit_slices_per_fold": FIT_SLICES,
            "calibration_slices_per_fold": CAL_SLICES,
            "seed": args.seed,
        },
        "symmetric_tuning": {
            "quantum": {"C": list(C_GRID), "bandwidth": list(QUANTUM_BANDWIDTH_GRID)},
            "rbf": {"C": list(C_GRID), "gamma": list(RBF_GAMMA_GRID)},
            "note": (
                "Both arms search a 5 x 5 grid on the same split by the same metric. "
                "The superseded run gave the RBF arm a bandwidth and the quantum arm "
                "none, which is not symmetry (Section 11.5). The quantum bandwidth "
                "scales the projection feeding the feature map, following Shaydulin "
                "and Wild."
            ),
        },
        "corpus": {
            "slices": int(len(y)),
            "subjects": n_subjects,
            "subject_positives_LGG": n_pos,
            "subject_negatives_HGG": n_neg,
        },
        "per_fold": per_fold,
        "pooled_out_of_fold": {
            "n_subjects": int(len(ys)),
            "quantum": q_metrics,
            "rbf": r_metrics,
            "delong": dl,
            "bootstrap_delta_auroc": boot,
            "fold_level_wilcoxon_p": round(fold_p, 4),
            "fold_deltas": fold_deltas,
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(OUTPUT.with_suffix(".npz"), p_quantum=pq, p_rbf=pr, y=ys, subject=ids)

    print("\n=== pooled out-of-fold, subject level ===")
    print(f"n subjects        {len(ys)}  ({n_pos} LGG / {n_neg} HGG)")
    print(f"quantum AUROC     {dl['auroc_a']:.4f}")
    print(f"rbf AUROC         {dl['auroc_b']:.4f}")
    print(f"delta             {dl['delta_auroc']:+.4f}  "
          f"95% CI [{dl['ci95'][0]:+.4f}, {dl['ci95'][1]:+.4f}]  "
          f"DeLong p = {dl['p']:.4g}")
    print(f"bootstrap delta   {boot['median']:+.4f} CI {boot['ci95']}")
    print(f"fold-level Wilcoxon p = {fold_p:.4f}   fold deltas {fold_deltas}")
    print(f"\n[done] wrote {OUTPUT} in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
