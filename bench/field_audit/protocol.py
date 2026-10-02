"""The symmetric arm of the field audit (docs/FIELD_AUDIT_PROTOCOL.md §4.2-4.3).

One function, ``run_symmetric``, takes a study's data, its feature pipeline and its
quantum model family, and returns fold-level metrics for three arms on identical folds:

* ``quantum``   — the study's quantum model, hyperparameters chosen by inner CV;
* ``classical`` — nested-selected classical: inner CV chooses among five families;
* ``named``     — the study's own named classical comparator, tuned the same way.

The verdict compares ``quantum`` with ``classical`` by the Nadeau-Bengio corrected
resampled t-test on the primary metric. Nothing here knows which study it is running.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED = 7
N_SPLITS, N_REPEATS, INNER_SPLITS = 5, 5, 3

#: Metrics a model's probabilities are scored on, every fold, every arm.
def fold_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, zero_division=0),
        "sensitivity": recall_score(y, yhat, pos_label=1, zero_division=0),
        "specificity": recall_score(y, yhat, pos_label=0, zero_division=0),
        "nll": log_loss(y, p, labels=[0, 1]),
    }
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return out


SINGLE_CLASS_RATES = {"sensitivity", "specificity", "recall"}


def selection_metric(primary: str) -> str:
    return "balanced_accuracy" if primary in SINGLE_CLASS_RATES else primary


# --------------------------------------------------------------------------- #
# Model families. Every entry is (family, params, estimator factory).
# --------------------------------------------------------------------------- #
class BalancedMLP:
    """MLP trained on a class-balanced (minority-oversampled) copy of its training fold;
    MLPClassifier takes no class weights, and this is the symmetric substitute."""

    def __init__(self, hidden, seed):
        self.m = MLPClassifier(hidden, alpha=1e-3, max_iter=2000, random_state=seed)
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        cls, cnt = np.unique(y, return_counts=True)
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), cnt.max(), replace=True)
                              for c in cls])
        self.m.fit(X[idx], y[idx])
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def classical_grid(seed: int = SEED, balanced: bool = False
                   ) -> list[tuple[str, dict, Callable[[], Any]]]:
    cw = "balanced" if balanced else None
    g: list[tuple[str, dict, Callable[[], Any]]] = []
    for C in (0.1, 1.0, 10.0):
        g.append(("logreg", {"C": C},
                  lambda C=C: LogisticRegression(C=C, max_iter=5000, class_weight=cw)))
    for C in (1.0, 10.0):
        for gamma in ("scale", 0.1):
            g.append(("svm_rbf", {"C": C, "gamma": gamma},
                      lambda C=C, gamma=gamma: SVC(C=C, gamma=gamma, probability=True,
                                                   class_weight=cw, random_state=seed)))
    for depth in (None, 5, 10):
        g.append(("random_forest", {"max_depth": depth},
                  lambda d=depth: RandomForestClassifier(300, max_depth=d, class_weight=cw,
                                                         random_state=seed, n_jobs=-1)))
    for lr in (0.05, 0.1, 0.2):
        g.append(("hist_gb", {"learning_rate": lr},
                  lambda lr=lr: HistGradientBoostingClassifier(learning_rate=lr,
                                                               class_weight=cw,
                                                               random_state=seed)))
    for hidden in ((32,), (64,), (64, 32)):
        if balanced:
            g.append(("mlp", {"hidden": hidden}, lambda h=hidden: BalancedMLP(h, seed)))
        else:
            g.append(("mlp", {"hidden": hidden},
                      lambda h=hidden: MLPClassifier(h, alpha=1e-3, max_iter=2000,
                                                     random_state=seed)))
    assert len(g) == 16
    return g


def classical_arm(name: str = "classical", families: Sequence[str] | None = None,
                  seed: int = SEED, balanced: bool = False) -> "Arm":
    """The nested-selected classical arm, or one named family of it."""
    cfg = [(fam, params, (lambda f=factory: SklearnModel(f)))
           for fam, params, factory in classical_grid(seed, balanced)
           if families is None or fam in families]
    return Arm(name, cfg)


class SklearnModel:
    """Adapter so a sklearn estimator looks like a study's quantum model."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory

    def fit(self, X, y):
        self.m = self.factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """A family of configurations; inner CV chooses one per outer fold."""
    name: str
    configs: list[tuple[str, dict, Callable[[], Any]]]   # (label, params, model factory)


def _inner_select(arm: Arm, X, y, pipeline_factory, metric: str, seed: int,
                  groups=None) -> int:
    if len(arm.configs) == 1:
        return 0
    if groups is None:
        splits = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed).split(X, y)
    else:
        splits = StratifiedGroupKFold(INNER_SPLITS, shuffle=True,
                                      random_state=seed).split(X, y, groups)
    scores = np.zeros(len(arm.configs))
    for tr, va in splits:
        pipe = pipeline_factory()
        Xtr = pipe.fit_transform(X[tr], y[tr])
        Xva = pipe.transform(X[va])
        for k, (_, _, make) in enumerate(arm.configs):
            m = make()
            m.fit(Xtr, y[tr])
            scores[k] += fold_metrics(y[va], m.predict_proba(Xva))[metric]
    return int(np.argmax(scores))


def outer_splits(X, y, groups=None, seed: int = SEED):
    if groups is None:
        yield from RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                           random_state=seed).split(X, y)
        return
    for r in range(N_REPEATS):
        yield from StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                        random_state=seed + r).split(X, y, groups)


def run_symmetric(X: np.ndarray, y: np.ndarray, *, arms: Sequence[Arm],
                  pipeline_factory: Callable[[], Any], primary: str,
                  seed: int = SEED, groups=None, train_fraction: float | None = None,
                  n_jobs: int = 12, log: Callable[[str], None] = print) -> dict:
    """Run every arm on identical outer folds. Returns per-fold metrics and choices.

    ``groups``: subject ids; outer and inner folds never split a subject.
    ``train_fraction``: each outer training fold is first reduced to a stratified
    random subset of this size (the same subset for every arm), for sample-size claims.
    """
    from joblib import Parallel, delayed

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    sel = selection_metric(primary)
    folds = []
    for f, (tr, te) in enumerate(outer_splits(X, y, groups, seed)):
        if train_fraction is not None:
            rng = np.random.default_rng(seed + 1000 + f)
            tr = np.concatenate([
                rng.choice(tr[y[tr] == c],
                           max(2, int(round(train_fraction * (y[tr] == c).sum()))),
                           replace=False)
                for c in (0, 1)])
        folds.append((f, tr, te))
    total = len(folds)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_one_fold)(f, tr, te, X, y, arms, pipeline_factory, sel, primary,
                           seed, groups, total)
        for f, tr, te in folds)
    per_arm = {a.name: [r["metrics"][a.name] for r in results] for a in arms}
    chosen = {a.name: [r["chosen"][a.name] for r in results] for a in arms}
    for r in results:
        log(r["line"])
    return {"per_fold": per_arm, "chosen": chosen,
            "n_train": results[-1]["n_train"], "n_test": results[-1]["n_test"],
            "primary": primary, "selection_metric": sel,
            "train_fraction": train_fraction, "grouped": groups is not None}


def _one_fold(f, tr, te, X, y, arms, pipeline_factory, sel, primary, seed, groups,
              total):
    """One outer fold, every arm. Runs in a worker process."""
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    g_tr = None if groups is None else np.asarray(groups)[tr]
    metrics, chosen = {}, {}
    for arm in arms:
        k = _inner_select(arm, X[tr], y[tr], pipeline_factory, sel, seed + f, g_tr)
        label, params, make = arm.configs[k]
        pipe = pipeline_factory()
        Xtr = pipe.fit_transform(X[tr], y[tr])
        Xte = pipe.transform(X[te])
        m = make()
        m.fit(Xtr, y[tr])
        metrics[arm.name] = fold_metrics(y[te], m.predict_proba(Xte))
        chosen[arm.name] = f"{label}{params}"
    line = (f"  fold {f + 1:2d}/{total} "
            + "  ".join(f"{a.name}={metrics[a.name][primary]:.3f}" for a in arms)
            + f"  [{time.time() - t0:.1f}s]")
    return {"metrics": metrics, "chosen": chosen, "line": line,
            "n_train": len(tr), "n_test": len(te)}


def nadeau_bengio(d: np.ndarray, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio 2003) on fold-level differences."""
    d = np.asarray(d, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean": float(d.mean()), "t": float("nan"), "p": 1.0 if d.mean() == 0 else 0.0}
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    half = stats.t.ppf(0.975, J - 1) * se
    return {"mean": float(d.mean()), "t": float(t), "p": float(p),
            "ci95": [float(d.mean() - half), float(d.mean() + half)]}


def verdict(result: dict, quantum: str = "quantum", classical: str = "classical") -> dict:
    prim = result["primary"]
    q = np.array([m[prim] for m in result["per_fold"][quantum]])
    c = np.array([m[prim] for m in result["per_fold"][classical]])
    nb = nadeau_bengio(q - c, result["n_train"], result["n_test"])
    if nb["p"] < 0.05 and nb["mean"] > 0:
        v = "SURVIVES"
    elif nb["p"] < 0.05 and nb["mean"] < 0:
        v = "REVERSED"
    else:
        v = "NOT SUPPORTED"
    summary = {}
    for arm, rows in result["per_fold"].items():
        summary[arm] = {k: [float(np.mean([r[k] for r in rows])),
                            float(np.std([r[k] for r in rows]))] for k in rows[0]}
    return {"primary": prim, "difference_quantum_minus_classical": nb,
            "verdict": v, "arm_means_sd": summary}
