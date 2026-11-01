"""Audit-factor separation, under ``docs/PREREGISTRATION_v6_2026-10-03.md`` §5.

    python -m bench.field_audit.factors parkinsons   # F1: 2x2x2, specificity
    python -m bench.field_audit.factors pima         # F2: 2x2, sensitivity
    python -m bench.field_audit.factors eeg          # F3: epoch-level vs subject-level splits

The quantum model in F1/F2 is the paper-configured ``KernelStack`` (C=1, bandwidth=1) in every
cell. Factors: PCA/scaler scope (global vs inside training folds), CV unit (recording vs
subject; Parkinson's only), comparator (untuned RBF-SVM on the stack's PCA-8 features, as in
the paper, vs the nested-selected classical arm of ``protocol.classical_arm`` on standardised
features, as in the symmetric arm). Outer scheme: 10-fold, repeated with seeds 0-4.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from . import datasets as D
from . import qmodels as Q
from . import qsim
from .protocol import _inner_select, classical_arm, fold_metrics, nadeau_bengio

warnings.filterwarnings("ignore")
SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# F1 / F2
# --------------------------------------------------------------------------- #
def _global_stack(Z_all):
    """A KernelStack whose scaler/PCA/MinMax are fixed on the full data (paper protocol)."""
    ks = Q.KernelStack()
    ks.sc = StandardScaler().fit(Z_all)
    ks.pca = PCA(8, random_state=7).fit(ks.sc.transform(Z_all))
    ks.mm = MinMaxScaler((0, math.pi)).fit(ks.pca.transform(ks.sc.transform(Z_all)))
    ks._prep_fit = lambda X_: None
    return ks


def _one_cell_fold(X, y, groups, tr, te, pca_scope, comparator, metric, seed):
    # --- quantum: paper-configured stack
    if pca_scope == "global":
        ks = _global_stack(X)
    else:
        ks = Q.KernelStack()
    ks.fit(X[tr], y[tr])
    pq = ks.predict_proba(X[te])

    # --- comparator
    if comparator == "untuned_rbf":
        P = ks.pca.transform(ks.sc.transform(X))         # the stack's own PCA-8 features
        rbf = SVC(probability=True, random_state=7).fit(P[tr], y[tr])
        pc = rbf.predict_proba(P[te])[:, 1]
    else:
        arm = classical_arm("classical", balanced=True)
        sc = StandardScaler().fit(X if pca_scope == "global" else X[tr])

        class _Pipe:                                     # scaler already fixed above
            def fit_transform(self, A, b=None):
                return sc.transform(A)

            def transform(self, A):
                return sc.transform(A)
        k = _inner_select(arm, X[tr], y[tr], _Pipe, "balanced_accuracy", seed,
                          groups[tr] if groups is not None else None)
        m = arm.configs[k][2]()
        m.fit(sc.transform(X[tr]), y[tr])
        pc = m.predict_proba(sc.transform(X[te]))
    return (fold_metrics(y[te], pq)[metric], fold_metrics(y[te], pc)[metric],
            len(tr), len(te))


def _cell(X, y, groups, pca_scope, cv_unit, comparator, metric):
    from joblib import Parallel, delayed

    folds = []
    for s in SEEDS:
        if cv_unit == "subject":
            it = StratifiedGroupKFold(10, shuffle=True, random_state=s).split(X, y, groups)
        else:
            it = StratifiedKFold(10, shuffle=True, random_state=s).split(X, y)
        folds += [(s, tr, te) for tr, te in it]
    g = groups if cv_unit == "subject" else None
    res = Parallel(n_jobs=int(__import__("os").environ.get("FA_JOBS", "4")), backend="loky")(
        delayed(_one_cell_fold)(X, y, g, tr, te, pca_scope, comparator, metric, s)
        for s, tr, te in folds)
    q = np.array([r[0] for r in res])
    c = np.array([r[1] for r in res])
    ntr, nte = int(np.mean([r[2] for r in res])), int(np.mean([r[3] for r in res]))
    nb = nadeau_bengio(q - c, ntr, nte)
    return {"pca_scope": pca_scope, "cv_unit": cv_unit, "comparator": comparator,
            "quantum_mean": float(q.mean()), "comparator_mean": float(c.mean()),
            "delta": nb, "folds": len(q),
            "advantage_reproduces": bool(nb["mean"] > 0 and nb["p"] < 0.05)}


def _attribution(cells: list[dict], factors: dict) -> dict:
    """Main effects and the single corrections applied to the paper cell."""
    def key(c):
        return tuple(c[f] for f in factors)
    by = {key(c): c for c in cells}
    paper = tuple(v[0] for v in factors.values())
    main = {}
    for f, levels in factors.items():
        a = [c["delta"]["mean"] for c in cells if c[f] == levels[0]]
        b = [c["delta"]["mean"] for c in cells if c[f] == levels[1]]
        main[f] = {"paper_level": levels[0], "corrected_level": levels[1],
                   "mean_delta_paper_level": float(np.mean(a)),
                   "mean_delta_corrected_level": float(np.mean(b)),
                   "main_effect": float(np.mean(b) - np.mean(a))}
    singles = {}
    names = list(factors)
    for i, f in enumerate(names):
        k = list(paper)
        k[i] = factors[f][1]
        c = by[tuple(k)]
        singles[f] = {"delta": c["delta"], "removes_advantage":
                      bool(c["delta"]["p"] >= 0.05 or c["delta"]["mean"] <= 0)}
    return {"paper_cell": by[paper], "main_effects": main,
            "single_corrections_from_paper_cell": singles}


def run_stack(which: str) -> dict:
    t0 = time.time()
    if which == "parkinsons":
        X, y, _, m = D.parkinsons()
        groups = np.asarray(m["groups"])
        metric = "specificity"
        factors = {"pca_scope": ("global", "in_fold"), "cv_unit": ("recording", "subject"),
                   "comparator": ("untuned_rbf", "nested_selected")}
    else:
        X, y, _, m = D.pima()
        groups = None
        metric = "sensitivity"
        factors = {"pca_scope": ("global", "in_fold"), "cv_unit": ("recording",),
                   "comparator": ("untuned_rbf", "nested_selected")}
    X, y = np.asarray(X, float), np.asarray(y, int)
    cells = []
    for p in factors["pca_scope"]:
        for u in factors["cv_unit"]:
            for c in factors["comparator"]:
                cell = _cell(X, y, groups, p, u, c, metric)
                cells.append(cell)
                print(f"[{which}] pca={p:8s} unit={u:9s} comp={c:15s} "
                      f"q={cell['quantum_mean']:.3f} c={cell['comparator_mean']:.3f} "
                      f"d={cell['delta']['mean']:+.3f} p={cell['delta']['p']:.3g}", flush=True)
    fac = {k: v for k, v in factors.items() if len(v) == 2}
    out = {"analysis": f"F{'1' if which == 'parkinsons' else '2'} — audit factor separation",
           "preregistration": "docs/PREREGISTRATION_v6_2026-10-03.md §5 (commit 61e777b)",
           "metric": metric, "quantum_model": "KernelStack(C=1, bandwidth=1), paper config",
           "outer": "10-fold x seeds 0-4", "cells": cells,
           "attribution": _attribution(cells, fac), "runtime_seconds": round(time.time() - t0)}
    return out


# --------------------------------------------------------------------------- #
# F3 — EEG epoch-level splitting
# --------------------------------------------------------------------------- #
def _z_states_product(A: np.ndarray) -> np.ndarray:
    """Per-qubit 2-amplitude states of PauliFeatureMap(['Z'], reps=2): (H, P(2x)) x 2.
    The map has no entangler, so the state is a product and the fidelity kernel is the
    product of per-qubit overlaps. Verified against ``qsim`` in ``_check_fast_kernel``."""
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    out = np.zeros(A.shape + (2,), complex)
    out[..., 0] = 1.0
    for _ in range(2):
        out = out @ h.T
        out[..., 1] *= np.exp(2j * A)
    return out


def _fast_z_kernel(A, B):
    sa, sb = _z_states_product(A), _z_states_product(B)
    ov = np.einsum("aqi,bqi->abq", sa.conj(), sb)
    return np.prod(np.abs(ov) ** 2, axis=-1)


class FastPauliZQSVM(Q.PauliQSVM):
    def fit(self, X, y):
        self.pca = PCA(self.k, random_state=self.seed).fit(X)
        self.mm = MinMaxScaler((0, math.pi)).fit(self.pca.transform(X))
        self.A = self._A(X).numpy()
        self.svm = SVC(kernel="precomputed", C=self.C, probability=True,
                       random_state=self.seed).fit(_fast_z_kernel(self.A, self.A), y)
        return self

    def predict_proba(self, X):
        return self.svm.predict_proba(_fast_z_kernel(self._A(X).numpy(), self.A))[:, 1]


def _check_fast_kernel() -> float:
    rng = np.random.default_rng(0)
    A = rng.uniform(0, math.pi, (40, 6))
    S = qsim.pauli_feature_map_single(A, "Z", reps=2)
    ref = qsim.fidelity_kernel(S, S)
    return float(np.abs(ref - _fast_z_kernel(A, A)).max())


def _eeg_epochs(seconds: int, channel: str = "Cz"):
    ch = D.EEG_CHANNELS.index(channel)
    L = 128 * seconds
    X, y, subj = [], [], []
    for label, folder in ((0, "norm"), (1, "sch")):
        for f in sorted((D.DATA / "eeg" / folder).glob("*.eea")):
            sig = np.loadtxt(f).reshape(16, 7680)[ch]
            for e in range(7680 // L):
                feats = []
                for band in D.wavedec_db2(sig[e * L:(e + 1) * L], 4):
                    feats += D._band_stats(band)
                X.append(feats)
                y.append(label)
                subj.append(f.stem)
    return np.array(X), np.array(y), np.array(subj)


def _eeg_split_run(X, y, subj, scheme, seed):
    if scheme == "epoch_random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y,
                                  random_state=seed)
    else:
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                      .split(X, y, subj))
    sc = StandardScaler().fit(X[tr])
    A, B = sc.transform(X[tr]), sc.transform(X[te])
    out = {}
    for k in (5, 15):
        m = FastPauliZQSVM(k=k, pauli="Z").fit(A, y[tr])
        out[f"QSVM_Z_k{k}"] = accuracy_score(y[te], m.predict_proba(B) >= 0.5)
    for nm, mk in (("LR", lambda: LogisticRegression(max_iter=5000)),
                   ("RF", lambda: RandomForestClassifier(random_state=7)),
                   ("SVM", lambda: SVC(random_state=7))):
        out[nm] = accuracy_score(y[te], mk().fit(A, y[tr]).predict(B))
    return out


def run_eeg() -> dict:
    from joblib import Parallel, delayed

    t0 = time.time()
    check = _check_fast_kernel()
    assert check < 1e-10, check
    res = {"analysis": "F3 — EEG epoch-level vs subject-level splitting",
           "preregistration": "docs/PREREGISTRATION_v6_2026-10-03.md §5 (commit 61e777b)",
           "kernel_check_max_abs_diff_vs_qsim": check, "by_epoch_length": {}}
    for seconds in (5, 1, 2):
        X, y, subj = _eeg_epochs(seconds)
        block = {"epochs": int(len(y)), "subjects": int(len(set(subj))),
                 "epochs_per_subject": 60 // seconds}
        for scheme in ("epoch_random", "subject_grouped"):
            runs = Parallel(n_jobs=int(__import__("os").environ.get("FA_JOBS", "4")), backend="loky")(
                delayed(_eeg_split_run)(X, y, subj, scheme, s) for s in range(200))
            block[scheme] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                                 "sd": float(np.std([r[k] for r in runs])),
                                 "share_100pct": float(np.mean([r[k] == 1.0 for r in runs]))}
                             for k in runs[0]}
        a, b = block["epoch_random"], block["subject_grouped"]
        block["reading"] = {
            m: bool(a[m]["mean"] >= 0.95 and b[m]["mean"] <= 0.75) for m in a}
        block["epoch_splitting_sufficient_for_near_perfect"] = any(block["reading"].values())
        res["by_epoch_length"][f"{seconds}s"] = block
        print(f"[eeg {seconds}s] " + " | ".join(
            f"{m}: {a[m]['mean']:.3f}/{b[m]['mean']:.3f}" for m in a), flush=True)
    res["runtime_seconds"] = round(time.time() - t0)
    return res


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "results"
    which = sys.argv[1]
    out = run_eeg() if which == "eeg" else run_stack(which)
    (out_dir / f"factors_{which}.json").write_text(json.dumps(out, indent=2, default=float),
                                                   encoding="utf-8")
    print(f"wrote factors_{which}.json")
