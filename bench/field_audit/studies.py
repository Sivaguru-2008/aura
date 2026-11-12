"""The included studies: what each claims, how each is reproduced, and its symmetric arm.

Run one study:   python -m bench.field_audit.studies ckd
Run all:         python -m bench.field_audit.studies all

Writes ``results/<study>.json``: the paper's reported numbers, the reproduction arm
under the paper's own protocol, the symmetric arm (25 outer folds, three arms), and the
verdict. The verdict rule, grids and fold scheme are fixed in
``docs/FIELD_AUDIT_PROTOCOL.md``; nothing here changes after a result is seen.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import datasets as D
from . import qmodels as Q
from .protocol import (Arm, SklearnModel, classical_arm, fold_metrics, run_symmetric,
                       verdict)

RESULTS = Path(__file__).parent / "results"
warnings.filterwarnings("ignore")


class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, idx):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx]


def _grid(model_cls, **axes):
    """Cartesian grid of a quantum model's hyperparameters, as Arm configs."""
    keys = list(axes)
    out = [{}]
    for k in keys:
        out = [dict(o, **{k: v}) for o in out for v in axes[k]]
    return [(model_cls.__name__, p, (lambda p=p: model_cls(**p))) for p in out]


@dataclass
class Study:
    key: str
    pmid: str
    doi: str
    citation: str
    claim: str
    primary: str
    reported: dict
    load: Callable[[], tuple]
    pipeline: Callable[[], Any]
    quantum: Callable[[], list]
    named: Callable[[], Arm]
    reproduce: Callable[[], dict]
    balanced: bool = False
    train_fractions: tuple = (None,)
    notes: list = field(default_factory=list)


# =========================================================================== #
# 1. CKD — 40567645
# =========================================================================== #
CKD_FEATURES = ["age", "bp", "al", "su", "ba", "bu", "sc", "sod", "pot", "hemo", "pcv",
                "wbcc", "rbcc"]


def _ckd_load():
    X, y, f, m = D.ckd()
    return X, y, None, {"features": f, **m}


def _ckd_pipe():
    _, _, f, _ = D.ckd()
    return make_pipeline(Columns([f.index(n) for n in CKD_FEATURES]),
                         SimpleImputer(strategy="median"), MinMaxScaler())


def _ckd_reproduce():
    """Paper protocol as far as stated: imputation and min-max on the full data, the
    paper's 13 LASSO features, 10-fold CV. Sparse-autoencoder augmentation and the
    Kalman-filter imputation are not reproduced (not specified enough)."""
    X, y, f, _ = D.ckd()
    Xs = _ckd_pipe().fit_transform(X, y)          # fitted on all rows, as the paper did
    models = {"VQC(reps=3,maxiter=400)": lambda: Q.CKDVQC(reps=3, maxiter=400),
              "SVM": lambda: SklearnModel(lambda: SVC(probability=True, random_state=7)),
              "RF": lambda: SklearnModel(lambda: RandomForestClassifier(random_state=7)),
              "LR": lambda: SklearnModel(lambda: LogisticRegression(max_iter=5000)),
              "MLP": lambda: SklearnModel(lambda: MLPClassifier(max_iter=2000, random_state=7))}
    return _cv_accuracy(models, Xs, y, 10)


# =========================================================================== #
# 2. Glioma — 40676161
# =========================================================================== #
GLIOMA_FEATURES = ["IDH1", "Age_at_diagnosis", "PTEN", "EGFR", "ATRX"]


def _glioma_load():
    X, y, f, m = D.glioma()
    return X, y, None, {"features": f, **m}


def _glioma_pipe():
    _, _, f, _ = D.glioma()
    return make_pipeline(Columns([f.index(n) for n in GLIOMA_FEATURES]), MinMaxScaler())


def _glioma_reproduce():
    """Paper protocol: min-max on the full data, the paper's 5 selected features
    (selected on the full data), 5-fold CV, default classical models."""
    X, y, _, _ = D.glioma()
    Xs = _glioma_pipe().fit_transform(X, y)
    sk = lambda make: (lambda: SklearnModel(make))  # noqa: E731
    models = {"VQC-1(reps=2,lr=1.0,iters=60)": lambda: Q.GliomaVQC(reps=2, lr=1.0, iters=60),
              "KNN": sk(lambda: KNeighborsClassifier()),
              "SVC": sk(lambda: SVC(probability=True, random_state=7)),
              "DTC": sk(lambda: DecisionTreeClassifier(random_state=7)),
              "RFC": sk(lambda: RandomForestClassifier(random_state=7)),
              "HistGB(XGBoost stand-in)": sk(lambda: HistGradientBoostingClassifier(random_state=7)),
              "GBM": sk(lambda: GradientBoostingClassifier(random_state=7))}
    return _cv_accuracy(models, Xs, y, 5)


# =========================================================================== #
# 3 & 4. Quantum-kernel stacking — 42286086 (Parkinson's specificity, Pima recall)
# =========================================================================== #
def _pd_load():
    X, y, f, m = D.parkinsons()
    return X, y, m["groups"], {"features": f, **{k: v for k, v in m.items() if k != "groups"}}


def _pima_load():
    X, y, f, m = D.pima()
    return X, y, None, {"features": f, **m}


def _std_pipe():
    return make_pipeline(StandardScaler())


def _stack_reproduce(loader, metric):
    """Paper protocol: z-score + PCA-8 fitted on the full data (the paper: 'PCA was
    applied globally before cross-validation'), 10-fold stratified CV by recording,
    base kernels + 5-fold OOF stacking with a balanced LR; baseline = untuned RBF-SVM
    (sklearn defaults, no class weights) on the same PCA-8 features."""
    X, y, _, _ = loader()
    Z = StandardScaler().fit_transform(X)
    outer = StratifiedKFold(10, shuffle=True, random_state=42)
    p_stack, p_rbf = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in outer.split(Z, y):
        # KernelStack refits its own scaler/PCA; feeding it globally-scaled data and
        # fixing PCA on the full set reproduces the paper's leakage.
        ks = Q.KernelStack()
        ks.sc = StandardScaler().fit(Z)
        from sklearn.decomposition import PCA
        ks.pca = PCA(8, random_state=7).fit(ks.sc.transform(Z))
        ks.mm = MinMaxScaler((0, math.pi)).fit(ks.pca.transform(ks.sc.transform(Z)))
        ks._prep_fit = lambda X_: None
        ks.fit(Z[tr], y[tr])
        p_stack[te] = ks.predict_proba(Z[te])
        P = ks.pca.transform(ks.sc.transform(Z))
        rbf = SVC(probability=True, random_state=7).fit(P[tr], y[tr])
        p_rbf[te] = rbf.predict_proba(P[te])[:, 1]
    return {"protocol": "paper (global PCA, 10-fold by recording)",
            "stack": fold_metrics(y, p_stack), "rbf_svm_untuned": fold_metrics(y, p_rbf),
            "claimed_metric": metric}


# =========================================================================== #
# 5. Heart QNN vs NN at small training sizes — 41513819
# =========================================================================== #
def _heart_load():
    X, y, f, m = D.heart918()
    return X, y, None, {"features": f, **m}


def _heart_pipe():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def _heart_named():
    cfg = [("mlp_3x52", {"alpha": a},
            (lambda a=a: SklearnModel(lambda: MLPClassifier((52, 52, 52), alpha=a,
                                                            max_iter=2000, random_state=7))))
           for a in (1e-4, 1e-3, 1e-2)]
    return Arm("named", cfg)


def _heart_reproduce():
    """Paper protocol: 70/30 split, training subset of 10%, 1-layer QNN with dropout,
    stopping on the test cost (as the paper does), 10 initialisations; NN 3x52 and a
    perceptron on the same subset. Test-cost stopping is emulated by passing the test
    split as the QNN's stopping set."""
    X, y, _, _ = D.heart918()
    X = SimpleImputer(strategy="median").fit_transform(X)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=7)
    rng = np.random.default_rng(7)
    sub = np.concatenate([rng.choice(tr[y[tr] == c], int(round(0.1 * (y[tr] == c).sum())),
                                     replace=False) for c in (0, 1)])
    acc_q = []
    for s in range(10):
        m = Q.HeartQNN(layers=1, dropout=True, seed=s)
        m.fit_with_stopping_set(X[sub], y[sub], X[te], y[te])
        acc_q.append(accuracy_score(y[te], m.predict_proba(X[te]) >= 0.5))
    sc = StandardScaler().fit(X[sub])
    acc_nn = [accuracy_score(y[te], MLPClassifier((52, 52, 52), max_iter=2000,
                                                  random_state=s)
                             .fit(sc.transform(X[sub]), y[sub]).predict(sc.transform(X[te])))
              for s in range(10)]
    acc_p = [accuracy_score(y[te], Perceptron(random_state=s)
                            .fit(sc.transform(X[sub]), y[sub]).predict(sc.transform(X[te])))
             for s in range(10)]
    return {"protocol": "paper (70/30 split, 10% training subset, stop on test cost)",
            "n_train_subset": int(len(sub)), "n_test": int(len(te)),
            "qnn_1layer_dropout": [float(np.mean(acc_q)), float(np.std(acc_q))],
            "nn_3x52": [float(np.mean(acc_nn)), float(np.std(acc_nn))],
            "perceptron": [float(np.mean(acc_p)), float(np.std(acc_p))]}


# =========================================================================== #
# 6. EEG schizophrenia QSVM — 38441727
# =========================================================================== #
def _eeg_load():
    X, y, f, m = D.eeg_schizophrenia("Cz")
    return X, y, None, {"features": f, **{k: v for k, v in m.items() if k != "ids"}}


def _eeg_reproduce():
    """Paper protocol: one random 70/30 split (seed unreported), PCA-k, QSVM with the
    Pauli-Z map; classical LR/RF/SVM defaults. The split seed is unknown, so 200 random
    splits are run and the distribution reported, with the share reaching 100%."""
    X, y, _, _ = D.eeg_schizophrenia("Cz")
    out = {k: [] for k in ("QSVM_Z_k5", "QSVM_Z_k15", "LR", "RF", "SVM")}
    for s in range(200):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y,
                                  random_state=s)
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        for k in (5, 15):
            m = Q.PauliQSVM(k=k, pauli="Z").fit(A, y[tr])
            out[f"QSVM_Z_k{k}"].append(accuracy_score(y[te], m.predict_proba(B) >= 0.5))
        for nm, mk in (("LR", lambda: LogisticRegression(max_iter=5000)),
                       ("RF", lambda: RandomForestClassifier(random_state=7)),
                       ("SVM", lambda: SVC(random_state=7))):
            out[nm].append(accuracy_score(y[te], mk().fit(A, y[tr]).predict(B)))
    return {"protocol": "paper (random 70/30, 200 seeds)",
            **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                   "share_100pct": float(np.mean(np.asarray(v) == 1.0))}
               for k, v in out.items()}}


# =========================================================================== #
def _cv_accuracy(models: dict, X, y, k: int) -> dict:
    skf = StratifiedKFold(k, shuffle=True, random_state=7)
    out = {}
    for name, make in models.items():
        accs = []
        for tr, te in skf.split(X, y):
            m = make().fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict_proba(X[te]) >= 0.5))
        out[name] = [float(np.mean(accs)), float(np.std(accs))]
    return {"protocol": f"paper ({k}-fold CV, preprocessing on full data)", "accuracy": out}


STUDIES: dict[str, Study] = {
    "ckd": Study(
        "ckd", "40567645", "10.7717/peerj-cs.2789",
        "PeerJ Comput Sci 2025 — VQC for chronic kidney disease",
        "VQC reaches 99.2% accuracy and outperforms SVM, RF, LR, MLP and RBF networks",
        "accuracy",
        {"VQC": 0.992, "RBF network": 0.963, "MLP": 0.948, "LR": 0.939, "SVM": 0.928,
         "RF": 0.907},
        _ckd_load, _ckd_pipe,
        lambda: _grid(Q.CKDVQC, reps=(1, 2, 3, 4), maxiter=(100, 200, 400, 800)),
        lambda: classical_arm("named", ["mlp"]),
        _ckd_reproduce),
    "glioma": Study(
        "glioma", "40676161", "10.1038/s41598-025-97067-3",
        "Sci Rep 2025 — hybrid classical/quantum glioma grading (TCGA)",
        "VQC-1 (CV accuracy 0.83) outperforms KNN, SVC, DTC and RFC",
        "accuracy",
        {"VQC-1": 0.83, "KNN": 0.82, "SVC": 0.75, "DTC": 0.80, "RFC": 0.82,
         "XGBoost": 0.84, "GBM": 0.83},
        _glioma_load, _glioma_pipe,
        lambda: _grid(Q.GliomaVQC, reps=(1, 2, 3, 4), lr=(0.05, 0.1, 0.5, 1.0)),
        lambda: classical_arm("named", ["random_forest"]),
        _glioma_reproduce),
    "stack_parkinsons": Study(
        "stack_parkinsons", "42286086", "10.1038/s41598-026-56928-1",
        "Sci Rep 2026 — adaptive quantum-kernel stacking (Parkinson's)",
        "Quantum-kernel stack raises Parkinson's specificity from 0.585 (RBF-SVM) to 0.813",
        "specificity",
        {"stack": 0.813, "RBF-SVM": 0.585},
        _pd_load, _std_pipe,
        lambda: _grid(Q.KernelStack, C=(0.1, 1.0, 10.0, 100.0),
                      bandwidth=(0.25, 0.5, 1.0, 2.0)),
        lambda: classical_arm("named", ["svm_rbf"], balanced=True),
        lambda: _stack_reproduce(_pd_load, "specificity"),
        balanced=True),
    "stack_pima": Study(
        "stack_pima", "42286086", "10.1038/s41598-026-56928-1",
        "Sci Rep 2026 — adaptive quantum-kernel stacking (Pima diabetes)",
        "Quantum-kernel stack raises diabetes recall from 0.553 (RBF-SVM) to 0.621",
        "sensitivity",
        {"stack": 0.621, "RBF-SVM": 0.553},
        _pima_load, _std_pipe,
        lambda: _grid(Q.KernelStack, C=(0.1, 1.0, 10.0, 100.0),
                      bandwidth=(0.25, 0.5, 1.0, 2.0)),
        lambda: classical_arm("named", ["svm_rbf"], balanced=True),
        lambda: _stack_reproduce(_pima_load, "sensitivity"),
        balanced=True),
    "heart_qnn": Study(
        "heart_qnn", "41513819", "10.1038/s41598-025-33825-7",
        "Sci Rep 2026 — QNNs vs NNs for heart disease, low-data regime",
        "1-layer QNN with dropout outperforms the NN and perceptron at 10-50% of the "
        "training data (10%: 82.72% vs 80.51%)",
        "accuracy",
        {"QNN_10pct": 0.8272, "NN_10pct": 0.8051, "perceptron_10pct": 0.8094},
        _heart_load, _heart_pipe,
        lambda: _grid(Q.HeartQNN, angle_range=(math.pi / 2, math.pi, 1.5 * math.pi,
                                               2 * math.pi),
                      lr=(0.003, 0.01, 0.03, 0.1)),
        _heart_named,
        _heart_reproduce,
        train_fractions=(0.1, 0.3, 0.5)),
    "eeg_qsvm": Study(
        "eeg_qsvm", "38441727", "10.1007/s10916-024-02048-0",
        "J Med Syst 2024 — QSVM for adolescent schizophrenia EEG",
        "QSVM with Pauli X/Z maps reaches 100% and shows superior performance",
        "accuracy",
        {"QSVM": 1.00, "LR (Cz)": 1.00, "RF (Cz)": 1.00, "SVM (Cz)": 1.00},
        _eeg_load, _std_pipe,
        lambda: _grid(Q.PauliQSVM, k=(3, 5, 8, 15), bandwidth=(0.25, 0.5, 1.0, 2.0)),
        lambda: classical_arm("named", ["svm_rbf"]),
        _eeg_reproduce),
}


def run(key: str, n_jobs: int = 12) -> dict:
    st = STUDIES[key]
    t0 = time.time()
    X, y, groups, meta = st.load()
    print(f"\n=== {key}: PMID {st.pmid} | n={len(y)} | primary={st.primary} "
          f"| grouped={groups is not None} | balanced={st.balanced}", flush=True)
    out: dict[str, Any] = {
        "study": key, "pmid": st.pmid, "doi": st.doi, "citation": st.citation,
        "claim": st.claim, "primary_metric": st.primary, "reported": st.reported,
        "data": {k: v for k, v in meta.items() if k != "features"},
        "features_in_pipeline": meta.get("features"),
    }
    print("[reproduction arm]", flush=True)
    out["reproduction"] = st.reproduce()
    print(json.dumps(out["reproduction"], indent=1)[:1500], flush=True)
    out["symmetric"] = {}
    for frac in st.train_fractions:
        tag = "full" if frac is None else f"train_{int(frac * 100)}pct"
        print(f"[symmetric arm: {tag}]", flush=True)
        arms = [Arm("quantum", st.quantum()),
                classical_arm("classical", balanced=st.balanced),
                st.named()]
        res = run_symmetric(X, y, arms=arms, pipeline_factory=st.pipeline,
                            primary=st.primary, groups=groups, train_fraction=frac,
                            n_jobs=n_jobs)
        v = verdict(res)
        out["symmetric"][tag] = {"verdict": v, "chosen": res["chosen"],
                                 "per_fold": res["per_fold"],
                                 "n_train_per_fold": res["n_train"],
                                 "n_test_per_fold": res["n_test"]}
        d = v["difference_quantum_minus_classical"]
        print(f"  => {tag}: {v['verdict']}  quantum-classical {st.primary} "
              f"{d['mean']:+.4f}  p={d['p']:.4g}", flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{key}.json").write_text(json.dumps(out, indent=2, default=float),
                                         encoding="utf-8")
    return out


if __name__ == "__main__":
    keys = list(STUDIES) if sys.argv[1] == "all" else sys.argv[1:]
    for k in keys:
        run(k)
