"""Fit the quantum-kernel (QSVM) brain classifier and write its serving artefact.

What this trains
----------------
AURA's brain corpus is BraTS-2020. The **only** tumour label axis it carries is
glioma grade — ``name_mapping.csv`` gives HGG/LGG, and the encoder embedding dump
(``artifacts/brain/embeddings/epoch_*.npz``) propagates it as the ``grade`` column.
There is no meningioma or metastasis imaging anywhere in the repository, so the
three-class subtype problem the QKL module was originally sketched against
**cannot be honestly trained**. Training it anyway would emit fabricated subtype
probabilities, which is precisely the failure mode the label-provenance work was
built to prevent.

So this trains the real task: ``glioma_grade`` (LGG vs HGG), on real embeddings,
with a real held-out evaluation. The three-class surface remains available and
keeps returning a uniform prior with ``abstained=True``.

Methodology
-----------
* **Subject-level splitting.** 7 407 slices come from 55 subjects. Splitting by
  slice would leak the same tumour across train and test and inflate every metric;
  every split here is grouped by ``subject_index``.
* **Fidelity kernel.** ``K(x,x') = |<phi(x)|phi(x')>|^2`` for the 6-qubit IQP
  feature map, computed exactly via statevectors (one encode per sample, then a
  single matmul) rather than N^2 circuit executions.
* **Calibration.** Platt scaling fitted on a calibration split disjoint from both
  train and test — never on the test fold.
* **Classical control.** An RBF-kernel SVM is fitted on the identical split and
  projection so the quantum/classical comparison is like-for-like. The result is
  reported whichever way it falls.
* **Bootstrap CIs.** 1 000 subject-level resamples, because with 55 subjects a
  point estimate alone is not meaningful.

Usage::

    python -m aura.ml.training.train_qkl
    python -m aura.ml.training.train_qkl --qubits 6 --train-slices 700
    python -m aura.ml.training.train_qkl --dry-run        # report only, no write
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aura.backend.engines.neuro.qkl import DEFAULT_WEIGHTS, QKLClassifier
from aura.common.config import ARTIFACTS

EMBED_DIR = ARTIFACTS / "brain" / "embeddings"
REPORT_PATH = ARTIFACTS / "brain" / "qkl_training_report.json"

#: BraTS grade encoding in the embedding dump: 0 = HGG (majority), 1 = LGG.
GRADE_CLASSES = ("HGG", "LGG")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def latest_embedding_dump(directory: Path = EMBED_DIR) -> Path:
    """Newest ``epoch_*.npz`` in the embedding directory."""
    files = sorted(directory.glob("epoch_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no embedding dumps in {directory}; run the brain trainer first "
            "(python -m aura.backend.vision.brain.cli train)"
        )
    return files[-1]


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    missing = {"embedding", "grade", "subject_index"} - set(d.files)
    if missing:
        raise KeyError(f"{path.name} lacks required columns: {sorted(missing)}")
    return {
        "embedding": np.asarray(d["embedding"], dtype=float),
        "grade": np.asarray(d["grade"], dtype=int),
        "subject": np.asarray(d["subject_index"], dtype=int),
        "tumor_area": np.asarray(d["tumor_area"], dtype=float) if "tumor_area" in d.files else None,
    }


def subject_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified *subject-level* train/calibration/test split.

    Stratifying by subject grade keeps the rare LGG class present in all three
    folds — with only 11 LGG subjects an unstratified draw routinely produces an
    empty positive test fold.
    """
    subj_ids = np.unique(subjects)
    subj_label = np.asarray([labels[subjects == s][0] for s in subj_ids])

    train, cal, test = [], [], []
    for cls in np.unique(subj_label):
        pool = subj_ids[subj_label == cls]
        pool = pool[rng.permutation(len(pool))]
        n = len(pool)
        n_tr = max(1, int(round(fractions[0] * n)))
        n_cal = max(1, int(round(fractions[1] * n))) if n - n_tr >= 2 else 0
        train.extend(pool[:n_tr])
        cal.extend(pool[n_tr : n_tr + n_cal])
        test.extend(pool[n_tr + n_cal :])

    def mask(group: list) -> np.ndarray:
        return np.isin(subjects, np.asarray(group, dtype=subjects.dtype))

    return mask(train), mask(cal), mask(test)


def balanced_subsample(
    idx: np.ndarray, labels: np.ndarray, n_total: int, rng: np.random.Generator
) -> np.ndarray:
    """Class-balanced subsample of slice indices.

    The Gram matrix is O(n^2); 7 407 slices is 27 M kernel entries. Balancing also
    counteracts the 4:1 HGG:LGG slice imbalance.
    """
    per_class = max(1, n_total // max(len(np.unique(labels[idx])), 1))
    picked: list[np.ndarray] = []
    for cls in np.unique(labels[idx]):
        pool = idx[labels[idx] == cls]
        take = min(per_class, len(pool))
        picked.append(rng.choice(pool, size=take, replace=False))
    out = np.concatenate(picked)
    return out[rng.permutation(len(out))]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    both = len(np.unique(y)) > 1
    return {
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "accuracy": float(accuracy_score(y, yhat)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "ece": expected_calibration_error(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 7,
) -> dict[str, float]:
    """Subject-level bootstrap CI — resample subjects, not slices."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    vals: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in drawn])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(binary_metrics(y[idx], p[idx])[metric])
        except Exception:
            continue
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "median": float("nan"), "n_boot": 0}
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "n_boot": int(len(arr)),
    }


def fit_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling ``p = sigma(-(A*f + B))`` by logistic regression on scores."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return -1.0, 0.0
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y)
    # QKLClassifier evaluates p = 1 / (1 + exp(a*f + b)), i.e. a = -coef, b = -intercept.
    return float(-lr.coef_[0][0]), float(-lr.intercept_[0])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    n_qubits: int = 6,
    train_slices: int = 600,
    eval_slices: int = 800,
    C: float = 1.0,
    seed: int = 7,
    n_boot: int = 1000,
    embedding_path: Path | None = None,
) -> tuple[QKLClassifier, dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    rng = np.random.default_rng(seed)
    src = Path(embedding_path) if embedding_path else latest_embedding_dump()
    data = load_embeddings(src)
    X, y, subj = data["embedding"], data["grade"], data["subject"]

    tr_m, cal_m, te_m = subject_split(subj, y, rng)
    tr_idx, cal_idx, te_idx = map(np.flatnonzero, (tr_m, cal_m, te_m))

    # --- projection fitted on train only ---------------------------------- #
    mean = X[tr_idx].mean(axis=0)
    std = X[tr_idx].std(axis=0)
    std[std < 1e-9] = 1.0
    Z_tr = (X[tr_idx] - mean) / std
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Z_tr)
    # Rescale rows so tanh() lands in its responsive range rather than saturating.
    comp = pca.components_
    proj_scale = np.percentile(np.abs(Z_tr @ comp.T), 95, axis=0)
    proj_scale[proj_scale < 1e-9] = 1.0
    W = comp / proj_scale[:, None]

    clf = QKLClassifier(
        W_proj=W,
        b_proj=np.zeros(n_qubits),
        n_qubits=n_qubits,
        classes=GRADE_CLASSES,
        task="glioma_grade",
        feature_mean=mean,
        feature_scale=std,
    )

    fit_idx = balanced_subsample(tr_idx, y, train_slices, rng)
    cal_use = balanced_subsample(cal_idx, y, eval_slices, rng) if len(cal_idx) else fit_idx
    te_use = balanced_subsample(te_idx, y, eval_slices, rng) if len(te_idx) else np.array([], int)

    A_fit = clf.project_batch(X[fit_idx])
    A_cal = clf.project_batch(X[cal_use])
    A_te = clf.project_batch(X[te_use]) if len(te_use) else np.zeros((0, n_qubits))

    t0 = time.perf_counter()
    S_fit = clf.feature_states(A_fit)
    K_fit = np.abs(S_fit @ S_fit.conj().T) ** 2
    encode_seconds = time.perf_counter() - t0

    y_fit = y[fit_idx]
    svm = SVC(kernel="precomputed", C=C, random_state=seed)
    svm.fit(K_fit, y_fit)

    sv_local = svm.support_
    clf.support_vectors = A_fit[sv_local]
    clf.support_labels = y_fit[sv_local].astype(float)
    clf.alpha = svm.dual_coef_[0].astype(float)
    clf.b_svm = float(svm.intercept_[0])
    clf._sv_states = S_fit[sv_local]

    def raw_scores(A: np.ndarray) -> np.ndarray:
        if len(A) == 0:
            return np.zeros(0)
        S = clf.feature_states(A)
        K = np.abs(S @ clf._sv_states.conj().T) ** 2
        return K @ clf.alpha + clf.b_svm

    f_cal = raw_scores(A_cal)
    clf.platt_a, clf.platt_b = fit_platt(f_cal, y[cal_use])

    def platt(f: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(clf.platt_a * f + clf.platt_b))

    p_cal = platt(f_cal)
    f_te = raw_scores(A_te)
    p_te = platt(f_te) if len(f_te) else np.zeros(0)

    # --- classical control on the identical split/projection --------------- #
    # Platt-scaled on the same calibration fold as the quantum arm, so the two
    # differ only in the kernel — not in how their scores become probabilities.
    rbf = SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)
    rbf.fit(A_fit, y_fit)
    rbf_a, rbf_b = fit_platt(rbf.decision_function(A_cal), y[cal_use]) if len(A_cal) else (-1.0, 0.0)
    p_te_rbf = (
        1.0 / (1.0 + np.exp(rbf_a * rbf.decision_function(A_te) + rbf_b))
        if len(A_te)
        else np.zeros(0)
    )

    quantum_test = binary_metrics(y[te_use], p_te) if len(te_use) else {}
    classical_test = binary_metrics(y[te_use], p_te_rbf) if len(te_use) else {}

    ci = (
        {
            m: bootstrap_ci(y[te_use], p_te, subj[te_use], m, n_boot=n_boot, seed=seed)
            for m in ("auroc", "accuracy", "f1")
        }
        if len(te_use)
        else {}
    )
    ci_classical = (
        {
            m: bootstrap_ci(y[te_use], p_te_rbf, subj[te_use], m, n_boot=n_boot, seed=seed)
            for m in ("auroc",)
        }
        if len(te_use)
        else {}
    )

    # Two paired deltas, because they answer different questions and disagree here.
    # AUROC: does the quantum kernel discriminate better? (No — significantly worse.)
    # ECE:   is it better calibrated? Reported whatever the answer, so that publishing
    #        the discrimination loss and the calibration result is one decision, not
    #        two — a benchmark you only run when it flatters you is not a benchmark.
    delta = (paired_bootstrap_delta(y[te_use], p_te, p_te_rbf, subj[te_use], n_boot, seed,
                                    metric="auroc") if len(te_use) else {})
    delta_ece = (paired_bootstrap_delta(y[te_use], p_te, p_te_rbf, subj[te_use], n_boot, seed,
                                        metric="ece") if len(te_use) else {})

    report: dict[str, Any] = {
        "task": "glioma_grade",
        "classes": list(GRADE_CLASSES),
        "label_axis_note": (
            "BraTS-2020 carries glioma grade (HGG/LGG) only. AURA holds no meningioma "
            "or metastasis imaging, so the three-class subtype problem is NOT trained "
            "and the legacy predict_subtype() surface abstains."
        ),
        "embedding_source": str(src),
        "n_slices_total": int(len(y)),
        "n_subjects_total": int(len(np.unique(subj))),
        "split": {
            "level": "subject (grouped, stratified by grade)",
            "train_subjects": int(len(np.unique(subj[tr_idx]))),
            "cal_subjects": int(len(np.unique(subj[cal_idx]))) if len(cal_idx) else 0,
            "test_subjects": int(len(np.unique(subj[te_idx]))) if len(te_idx) else 0,
            "train_slices_used": int(len(fit_idx)),
            "cal_slices_used": int(len(cal_use)),
            "test_slices_used": int(len(te_use)),
        },
        "quantum": {
            "feature_map": "IQP (Hadamard + RZ + ring ZZ)",
            "kernel": "fidelity |<phi(x)|phi(x')>|^2",
            "n_qubits": n_qubits,
            "hilbert_dim": 2**n_qubits,
            "support_vectors": int(len(clf.support_vectors)),
            "encode_seconds_train": round(encode_seconds, 3),
        },
        "calibration": {"platt_a": clf.platt_a, "platt_b": clf.platt_b,
                        "cal_metrics": binary_metrics(y[cal_use], p_cal) if len(cal_use) else {}},
        "test_quantum": quantum_test,
        "test_classical_rbf": classical_test,
        "bootstrap_ci_quantum": ci,
        "bootstrap_ci_classical": ci_classical,
        "quantum_minus_classical_auroc": delta,
        # Lower ECE is better, so a NEGATIVE delta favours the quantum kernel — the
        # opposite sign convention to the AUROC delta above. Stated in the artifact
        # itself because a reader comparing two "delta" keys will otherwise assume
        # one direction means the same thing in both.
        "quantum_minus_classical_ece": {**delta_ece,
                                        "lower_is_better": True,
                                        "negative_favours": "quantum"},
        "seed": seed,
        "C": C,
    }

    clf.provenance = {
        "task": "glioma_grade",
        "trained_on": src.name,
        "train_subjects": report["split"]["train_subjects"],
        "test_subjects": report["split"]["test_subjects"],
        "test_auroc": quantum_test.get("auroc"),
        "test_auroc_ci95": [ci.get("auroc", {}).get("lo"), ci.get("auroc", {}).get("hi")] if ci else None,
        "classical_rbf_auroc": classical_test.get("auroc"),
        "untrained_classes": ["meningioma", "metastasis"],
        "note": report["label_axis_note"],
    }
    return clf, report


def paired_bootstrap_delta(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
    n_boot: int = 1000, seed: int = 7, metric: str = "auroc",
) -> dict[str, float]:
    """Paired subject-level bootstrap of metric(quantum) - metric(classical).

    ``metric="auroc"`` answers "does the quantum kernel *discriminate* better?" and
    the answer is no, significantly. ``metric="ece"`` answers a different and, for
    this system, more load-bearing question: "is it better *calibrated*?" AURA's whole
    product is calibrated doubt, so a model that trades a little ranking power for a
    materially honest probability is trading in the right direction — but only if the
    trade is real, which is what this measures rather than asserts.

    Note the sign convention differs by metric: for AUROC higher is better, for ECE
    lower is better, so a *negative* ECE delta favours the quantum kernel.
    """
    from sklearn.metrics import roc_auc_score

    if metric == "auroc":
        def _score(yy, pp):
            return roc_auc_score(yy, pp)
    elif metric == "ece":
        def _score(yy, pp):
            return expected_calibration_error(yy, pp)
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'auroc' or 'ece'")

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    deltas: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in drawn])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            deltas.append(_score(y[idx], p_a[idx]) - _score(y[idx], p_b[idx]))
        except Exception:
            continue
    if not deltas:
        return {"n_boot": 0}
    arr = np.asarray(deltas)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "mean": float(arr.mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "p_two_sided": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
        "significant_at_95": bool(lo > 0 or hi < 0),
        "n_boot": int(len(arr)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the QKL brain classifier.")
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--train-slices", type=int, default=600)
    ap.add_argument("--eval-slices", type=int, default=800)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)

    clf, report = train(
        n_qubits=args.qubits,
        train_slices=args.train_slices,
        eval_slices=args.eval_slices,
        C=args.C,
        seed=args.seed,
        n_boot=args.bootstrap,
        embedding_path=args.embeddings,
    )

    q, c = report["test_quantum"], report["test_classical_rbf"]
    print(f"\n  task            : {report['task']}  {report['classes']}")
    print(f"  split           : {report['split']['train_subjects']}/"
          f"{report['split']['cal_subjects']}/{report['split']['test_subjects']} subjects "
          f"(train/cal/test), grouped")
    print(f"  support vectors : {report['quantum']['support_vectors']}")
    if q:
        ci = report["bootstrap_ci_quantum"].get("auroc", {})
        print(f"  QUANTUM  AUROC  : {q['auroc']:.4f}  "
              f"[{ci.get('lo', float('nan')):.4f}, {ci.get('hi', float('nan')):.4f}]  "
              f"acc {q['accuracy']:.4f}  F1 {q['f1']:.4f}  ECE {q['ece']:.4f}")
        print(f"  CLASSICAL RBF   : {c['auroc']:.4f}  acc {c['accuracy']:.4f}  "
              f"F1 {c['f1']:.4f}  ECE {c['ece']:.4f}")
        d = report["quantum_minus_classical_auroc"]
        if d.get("n_boot"):
            print(f"  delta AUROC     : {d['mean']:+.4f} "
                  f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}]  "
                  f"p={d['p_two_sided']:.3f}  "
                  f"{'SIGNIFICANT' if d['significant_at_95'] else 'not significant'}")
        # Printed next to the AUROC delta on purpose. The ECE *point* estimates look
        # like a 30%-ish calibration win for the quantum kernel, which is tempting to
        # quote; with only 11 test subjects the paired bootstrap says otherwise. Show
        # both or the favourable-looking half gets quoted on its own.
        de = report.get("quantum_minus_classical_ece", {})
        if de.get("n_boot"):
            print(f"  delta ECE       : {de['mean']:+.4f} "
                  f"[{de['ci95_lo']:+.4f}, {de['ci95_hi']:+.4f}]  "
                  f"p={de['p_two_sided']:.3f}  "
                  f"{'SIGNIFICANT' if de['significant_at_95'] else 'not significant'}"
                  f"   (lower is better; negative favours quantum)")

    if args.dry_run:
        print("\n  --dry-run: no artefacts written")
        return 0

    path = clf.save(args.out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  weights -> {path}")
    print(f"  report  -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
