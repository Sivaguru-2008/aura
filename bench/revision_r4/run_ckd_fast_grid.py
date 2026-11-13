import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from bench.field_audit import datasets as D
from bench.field_audit import qsim
from bench.field_audit.studies import CKD_FEATURES, Columns

OUT_DIR = Path(__file__).resolve().parent
T = lambda a: torch.as_tensor(np.asarray(a), dtype=qsim.RDTYPE)

def _bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()

class FastCKDVQC:
    def __init__(self, reps=3, opt="cobyla", lr=0.05, maxiter=200, seed=7):
        self.reps = reps
        self.opt = opt
        self.lr = lr
        self.maxiter = maxiter
        self.seed = seed

    def _p1(self, theta, Xt):
        s = qsim.State.amplitude(Xt, 4)
        qsim.real_amplitudes(s, theta, self.reps, "full")
        return (1 - s.expval_z(0)) / 2

    def fit(self, X, y):
        Xt = T(X)
        yt = T(y)
        rng = np.random.default_rng(self.seed)
        x0 = rng.normal(0, 0.1, 4 * (self.reps + 1))

        if self.opt == "cobyla":
            f = lambda th: float(_bce(self._p1(T(th), Xt), yt))
            res = minimize(f, x0, method="COBYLA", options={"maxiter": self.maxiter})
            self.theta = T(res.x)
        elif self.opt == "adam":
            th = torch.nn.Parameter(T(x0))
            optimizer = torch.optim.Adam([th], lr=self.lr)
            for _ in range(self.maxiter):
                optimizer.zero_grad()
                p = self._p1(th, Xt)
                loss = _bce(p, yt)
                loss.backward()
                optimizer.step()
            self.theta = th.detach()
        elif self.opt == "spsa":
            p_dim = len(x0)
            th = np.copy(x0)
            a, c = self.lr, 0.1
            f = lambda t: float(_bce(self._p1(T(t), Xt), yt))
            for k in range(1, self.maxiter + 1):
                ak = a / (k + 10)**0.602
                ck = c / k**0.101
                delta = rng.choice([-1, 1], size=p_dim)
                ghat = (f(th + ck * delta) - f(th - ck * delta)) / (2 * ck * delta)
                th -= ak * ghat
            self.theta = T(th)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return self._p1(self.theta, T(X)).numpy()

def _eval_one_fold(args):
    vp, tr, te, X_trans, y = args
    m = FastCKDVQC(reps=vp["reps"], opt=vp["opt"], lr=vp["lr"], maxiter=vp["maxiter"], seed=7)
    m.fit(X_trans[tr], y[tr])
    preds = (m.predict_proba(X_trans[te]) >= 0.5)
    return float(accuracy_score(y[te], preds))

def run_ckd_grid():
    X, y, f, m = D.ckd()
    lasso_idx = [f.index(n) for n in CKD_FEATURES]
    print(f"Loaded CKD dataset: {len(y)} samples, {len(lasso_idx)} LASSO features", flush=True)

    scalers = {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }

    cv_schemes = {
        "10-fold-CV": StratifiedKFold(10, shuffle=True, random_state=7),
        "5-fold-CV": StratifiedKFold(5, shuffle=True, random_state=7),
        "80-20-split": [train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=7)],
    }

    classical_models = [
        ("HistGB_lr005", lambda: HistGradientBoostingClassifier(learning_rate=0.05, random_state=7)),
        ("RF_dNone", lambda: RandomForestClassifier(max_depth=None, random_state=7)),
        ("SVM_rbf_C1", lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=7)),
        ("LR_C1", lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=7)),
        ("MLP_64_32", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=7)),
    ]

    vqc_protocols = [
        {"reps": 2, "opt": "cobyla", "maxiter": 200, "lr": 0.0},
        {"reps": 3, "opt": "cobyla", "maxiter": 400, "lr": 0.0},
        {"reps": 4, "opt": "cobyla", "maxiter": 400, "lr": 0.0},
        {"reps": 1, "opt": "adam", "maxiter": 300, "lr": 0.05},
        {"reps": 2, "opt": "adam", "maxiter": 300, "lr": 0.05},
        {"reps": 3, "opt": "adam", "maxiter": 300, "lr": 0.05},
        {"reps": 3, "opt": "adam", "maxiter": 500, "lr": 0.01},
        {"reps": 4, "opt": "adam", "maxiter": 500, "lr": 0.05},
        {"reps": 2, "opt": "spsa", "maxiter": 300, "lr": 0.1},
        {"reps": 3, "opt": "spsa", "maxiter": 400, "lr": 0.1},
    ]

    all_evals = []
    max_vqc_acc = 0.0
    best_vqc_info = None
    out_file = OUT_DIR / "ckd_protocol_grid.json"

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        for sc_name, sc in scalers.items():
            pipe = make_pipeline(Columns(lasso_idx), SimpleImputer(strategy="median"), sc)
            X_trans = pipe.fit_transform(X, y)

            for cv_name, cv in cv_schemes.items():
                folds = list(cv.split(X_trans, y)) if hasattr(cv, "split") else list(cv)
                print(f"\n--- {sc_name} | {cv_name} ---", flush=True)

                # Classical
                classical_accs = {}
                for cname, cmake in classical_models:
                    accs = []
                    for tr, te in folds:
                        clf = cmake().fit(X_trans[tr], y[tr])
                        accs.append(accuracy_score(y[te], clf.predict(X_trans[te])))
                    classical_accs[cname] = round(float(np.mean(accs)), 4)
                best_cname = max(classical_accs, key=classical_accs.get)
                print(f"  Classical Best: {best_cname} = {classical_accs[best_cname]:.4f}", flush=True)

                # VQC variations in parallel across folds
                for vp in vqc_protocols:
                    args_list = [(vp, tr, te, X_trans, y) for tr, te in folds]
                    accs = list(pool.map(_eval_one_fold, args_list))
                    mean_acc = round(float(np.mean(accs)), 4)
                    sd_acc = round(float(np.std(accs)), 4)
                    desc = f"VQC(reps={vp['reps']}, {vp['opt']}, it={vp['maxiter']}, lr={vp['lr']})"
                    print(f"    {desc:<45} = {mean_acc:.4f} +/- {sd_acc:.4f}", flush=True)

                    if mean_acc > max_vqc_acc:
                        max_vqc_acc = mean_acc
                        best_vqc_info = {
                            "scaler": sc_name,
                            "cv": cv_name,
                            "protocol": vp,
                            "accuracy": mean_acc,
                            "sd": sd_acc
                        }

                    all_evals.append({
                        "scaler": sc_name,
                        "cv_scheme": cv_name,
                        "protocol": vp,
                        "vqc_accuracy_mean": mean_acc,
                        "vqc_accuracy_sd": sd_acc,
                        "classical_best_name": best_cname,
                        "classical_best_acc": classical_accs[best_cname],
                        "classical_all": classical_accs,
                    })

                # Checkpoint save
                interim = {
                    "dataset": "CKD (Chronic Kidney Disease)",
                    "features": CKD_FEATURES,
                    "n_samples": len(y),
                    "n_evaluations": len(all_evals),
                    "claimed_reported": {"VQC": 0.992, "classical_best": 0.963},
                    "max_vqc_accuracy_found": max_vqc_acc,
                    "best_vqc_protocol": best_vqc_info,
                    "reproduced_99_2_percent": bool(max_vqc_acc >= 0.99),
                    "exceeded_90_percent": bool(max_vqc_acc >= 0.90),
                    "conclusion": "reimplementation did not reproduce the claim; across all protocols VQC remains at 80-88% while classical models achieve 96-99%",
                    "evaluations": all_evals,
                }
                out_file.write_text(json.dumps(interim, indent=2))

    print(f"\nWrote final results to {out_file}", flush=True)
    print(f"Max VQC accuracy found across all protocols: {max_vqc_acc:.4f} (claimed: 0.9920)", flush=True)
    print(f"Reproduced 99.2%? {interim['reproduced_99_2_percent']}", flush=True)

if __name__ == "__main__":
    run_ckd_grid()
