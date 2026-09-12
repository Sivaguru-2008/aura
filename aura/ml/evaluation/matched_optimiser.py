"""Fusion backend comparison under a single, shared optimiser protocol.

Why this exists
---------------
Section 11.5 of the manuscript concedes an exposure that Section 8.2 had already
demonstrated is not hypothetical: the four arms of Table 6 differ in *optimiser*
as well as in model class. The product-of-experts is fitted by 400 steps of
fixed-rate full-batch gradient descent, the gated and quantum arms by Adam at
different epoch counts, and the logistic regression by L-BFGS to optimality.
Section 8.2 then showed that converging one arm moved it further than the entire
quantum effect the paper opened with.

That makes every ordering in Table 6 uninterpretable as a statement about model
class, which is the ordering the paper's central claim rests on. This script
removes the confound by fitting **every arm under one protocol**:

* the same optimiser (Adam),
* the same mini-batch size,
* the same maximum epoch budget,
* the same learning-rate grid, searched symmetrically per arm,
* the same early-stopping rule on the same calibration split,
* the same seed,
* the same weight decay (zero, for all arms, so that regularisation is not a
  hidden per-arm difference either).

The learning-rate grid is searched *per arm* rather than fixed, because a single
shared learning rate is not symmetry — it is a choice that happens to suit
whichever arm it was tuned on. Symmetric treatment means every arm gets the same
search, on the same split, by the same criterion. This is the operationalisation
of P2 that Section 11.5 says the paper owes the reader.

The multinomial logistic regression is retained as a **convergence anchor**: it is
the same model class as the product-of-experts, fitted to optimality by L-BFGS.
The gap between the Adam-fitted linear arm and the L-BFGS optimum is a direct
measurement of whether the shared budget was sufficient, on the one arm whose
optimum is knowable. If that gap is negligible, the budget is adequate for the
arms whose optimum is not knowable.

Writes ``artifacts/fusion_matched_optimiser.json``.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.fusion.device import make_qnode
from aura.services.safety.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from aura.ml.evaluation.quantum_study import load_evidence

OUTPUT = ARTIFACTS / "fusion_matched_optimiser.json"
PREDICTIONS = ARTIFACTS / "fusion_matched_optimiser_predictions.npz"

N_DX = len(DIAGNOSES)

# ---- the shared protocol. One definition, applied to every arm. ------------ #
LR_GRID = (0.01, 0.05, 0.1)
MAX_EPOCHS = 150
BATCH = 50
PATIENCE = 20          # epochs without calibration-NLL improvement
WEIGHT_DECAY = 0.0
SEED = 7

N_QUBITS = 8
N_LAYERS = 3
ENTANGLER = "ring"


# --------------------------------------------------------------------------- #
# Scoring helpers (identical to fusion_baselines.py, so the tables are readable
# against each other)
# --------------------------------------------------------------------------- #
def _jeffreys(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 0.5, n - k + 0.5))
    return [round(lo, 4), round(hi, 4)]


def _auroc(score: np.ndarray, pos: np.ndarray) -> float:
    r = stats.rankdata(score)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _macro_auroc(P: np.ndarray, y: np.ndarray) -> float:
    aucs = []
    for c in range(P.shape[1]):
        pos = y == c
        if pos.all() or not pos.any():
            continue
        aucs.append(_auroc(P[:, c], pos))
    return float(np.mean(aucs)) if aucs else float("nan")


def _per_study_nll(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))


def _evaluate(logits_cal, logits_test, ycal, yte) -> dict:
    T = fit_temperature(logits_cal, ycal)
    P = np.array([softmax(r / T) for r in logits_test])
    correct = P.argmax(axis=1) == yte
    nll = _per_study_nll(P, yte)
    n = len(yte)
    return {
        "n": n,
        "temperature": round(float(T), 4),
        "accuracy": round(float(correct.mean()), 4),
        "correct": int(correct.sum()),
        "accuracy_ci95_jeffreys": _jeffreys(int(correct.sum()), n),
        "nll": round(float(nll.mean()), 4),
        "ece": round(float(expected_calibration_error(P, yte)), 4),
        "macro_auroc": round(_macro_auroc(P, yte), 4),
        "_probs": P,
        "_correct": correct,
        "_nll": nll,
    }


# --------------------------------------------------------------------------- #
# The shared trainer. Every arm goes through this function and no other.
# --------------------------------------------------------------------------- #
def _train_under_protocol(build, Xtr, ytr, Xcal, ycal, lr: float):
    """Fit one arm at one learning rate under the shared protocol.

    ``build`` returns ``(params, forward)`` where ``forward(Xtensor)`` produces
    logits. Everything else — optimiser, batching, epoch cap, early stopping,
    seed — is fixed here so that no arm can differ in it.
    """
    import torch

    torch.manual_seed(SEED)
    params, forward = build()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xc = torch.tensor(Xcal, dtype=torch.float64)
    yc = torch.tensor(ycal, dtype=torch.long)
    n = len(ytr)

    best_nll, best_state, best_epoch, since = np.inf, None, -1, 0
    history = []
    for ep in range(MAX_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(forward(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            cal_nll = float(loss_fn(forward(Xc), yc).item())
            tr_nll = float(loss_fn(forward(Xt), yt).item())
        history.append({"epoch": ep, "train_nll": round(tr_nll, 6),
                        "calibration_nll": round(cal_nll, 6)})
        if cal_nll < best_nll - 1e-6:
            best_nll, best_epoch, since = cal_nll, ep, 0
            best_state = [p.detach().clone() for p in params]
        else:
            since += 1
            if since >= PATIENCE:
                break

    for p, s in zip(params, best_state):
        p.data.copy_(s)
    stopped_early = since >= PATIENCE
    return {
        "lr": lr,
        "best_calibration_nll": round(float(best_nll), 6),
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "stopped_early": bool(stopped_early),
        "final_train_nll": history[-1]["train_nll"],
        "history": history,
    }, params, forward


def _search(build, Xtr, ytr, Xcal, ycal, Xte, label: str):
    """Symmetric learning-rate search: same grid, same split, same criterion."""
    import torch

    runs, best = [], None
    for lr in LR_GRID:
        t0 = time.time()
        rec, params, forward = _train_under_protocol(build, Xtr, ytr, Xcal, ycal, lr)
        rec["seconds"] = round(time.time() - t0, 1)
        runs.append({k: v for k, v in rec.items() if k != "history"})
        print(f"    [{label}] lr={lr:<5} best_cal_nll={rec['best_calibration_nll']:.4f} "
              f"best_epoch={rec['best_epoch']:3d} epochs={rec['epochs_run']:3d} "
              f"early={rec['stopped_early']} ({rec['seconds']}s)", flush=True)
        if best is None or rec["best_calibration_nll"] < best[0]["best_calibration_nll"]:
            with torch.no_grad():
                Lcal = forward(torch.tensor(Xcal, dtype=torch.float64)).numpy()
                Lte = forward(torch.tensor(Xte, dtype=torch.float64)).numpy()
            n_params = int(sum(int(p.numel()) for p in params))
            best = (rec, Lcal, Lte, n_params)

    rec, Lcal, Lte, n_params = best
    return {
        "selected_lr": rec["lr"],
        "selected_on": "calibration split, by cross-entropy",
        "best_epoch": rec["best_epoch"],
        "epochs_run": rec["epochs_run"],
        "converged_by_early_stopping": rec["stopped_early"],
        "final_train_nll": rec["final_train_nll"],
        "grid": runs,
    }, Lcal, Lte, n_params


# --------------------------------------------------------------------------- #
# Arm builders. Each returns (trainable params, forward fn).
# --------------------------------------------------------------------------- #
def _build_linear(d: int):
    import torch

    def build():
        W = torch.nn.Parameter(0.1 * torch.randn(N_DX, d, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
        return [W, b], lambda X: X @ W.T + b
    return build


def _build_gated(d: int):
    import torch

    def build():
        A = torch.nn.Parameter(0.05 * torch.randn(d, d, dtype=torch.float64))
        c = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
        W = torch.nn.Parameter(0.1 * torch.randn(N_DX, d, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))

        def forward(X):
            return (torch.sigmoid(X @ A.T + c) * X) @ W.T + b
        return [A, c, W, b], forward
    return build


def _build_quantum(n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
                   entangler: str = ENTANGLER):
    import torch

    circuit = make_qnode(n_qubits, n_layers, interface="torch", entangler=entangler)

    def build():
        theta = torch.nn.Parameter(
            0.1 * torch.randn(n_layers, n_qubits, 2, dtype=torch.float64))
        W = torch.nn.Parameter(0.1 * torch.randn(N_DX, n_qubits, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))

        def forward(X):
            z = torch.stack([zi.to(torch.float64) for zi in circuit(X, theta)], dim=-1)
            return z @ W.T + b
        return [theta, W, b], forward
    return build


def _full_width_logits(model, X) -> np.ndarray:
    """Decision function widened to the full ``N_DX`` diagnosis space.

    scikit-learn emits one column per class *present in the training split*, and a
    single column in the binary case. The torch arms always emit ``N_DX`` logits, so
    a comparison would otherwise be between differently shaped outputs — and any
    target whose rare class is absent from a split would raise rather than report.
    Absent classes are given a large negative logit: never predicted, but present in
    the softmax denominator, which is the correct behaviour for a class the model
    was given no examples of.
    """
    d = model.decision_function(X)
    if d.ndim == 1:                      # binary case: sklearn returns one margin
        d = np.column_stack([-d, d])
    out = np.full((d.shape[0], N_DX), -30.0, dtype=float)
    out[:, np.asarray(model.classes_, dtype=int)] = d
    return out


def _lbfgs_anchor(Xtr, ytr, Xcal, ycal, Xte):
    """The convergence anchor: same model class as the linear arm, fitted to
    optimality. Its gap to the Adam-fitted linear arm measures budget adequacy."""
    from sklearn.linear_model import LogisticRegression

    best, best_nll, search = None, np.inf, []
    for C in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=5000, tol=1e-8)
        m.fit(Xtr, ytr)
        Lcal = _full_width_logits(m, Xcal)
        T = fit_temperature(Lcal, ycal)
        Pcal = np.array([softmax(r / T) for r in Lcal])
        v = float(_per_study_nll(Pcal, ycal).mean())
        search.append({"C": C, "calibration_nll": round(v, 4)})
        if v < best_nll:
            best, best_nll = m, v
    return (_full_width_logits(best, Xcal), _full_width_logits(best, Xte),
            int(best.coef_.size + best.intercept_.size),
            {"selected_C": float(best.C), "lbfgs_iterations": int(best.n_iter_.max()),
             "converged": bool(best.n_iter_.max() < 5000),
             "classes_fitted": [int(c) for c in best.classes_], "grid": search})


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _mcnemar(a_correct, b_correct) -> dict:
    b01 = int((~a_correct & b_correct).sum())
    b10 = int((a_correct & ~b_correct).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else float(stats.binomtest(b01, n, 0.5).pvalue)
    return {"discordant": n, "a_only": b10, "b_only": b01, "p": round(p, 6)}


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
    Xtr, ytr, Xcal, ycal, Xte, yte, source = load_evidence(n=1338, seed=7)
    d = int(Xtr.shape[1])

    arms: dict[str, dict] = {}
    params: dict[str, int] = {}
    fitting: dict[str, dict] = {}

    print("[matched] product-of-experts (linear softmax)", flush=True)
    fit, Lcal, Lte, p = _search(_build_linear(d), Xtr, ytr, Xcal, ycal, Xte, "linear")
    arms["classical"] = _evaluate(Lcal, Lte, ycal, yte)
    params["classical"], fitting["classical"] = p, fit

    print("[matched] gated head", flush=True)
    fit, Lcal, Lte, p = _search(_build_gated(d), Xtr, ytr, Xcal, ycal, Xte, "gated")
    arms["learnable"] = _evaluate(Lcal, Lte, ycal, yte)
    params["learnable"], fitting["learnable"] = p, fit

    print("[matched] quantum variational circuit", flush=True)
    fit, Lcal, Lte, p = _search(_build_quantum(), Xtr, ytr, Xcal, ycal, Xte, "quantum")
    arms["quantum"] = _evaluate(Lcal, Lte, ycal, yte)
    params["quantum"], fitting["quantum"] = p, fit

    print("[matched] L-BFGS convergence anchor", flush=True)
    Lcal, Lte, p, anchor = _lbfgs_anchor(Xtr, ytr, Xcal, ycal, Xte)
    arms["logistic"] = _evaluate(Lcal, Lte, ycal, yte)
    params["logistic"], fitting["logistic"] = p, anchor

    counts = np.bincount(yte, minlength=N_DX)
    floor = {"accuracy": round(float(counts.max() / len(yte)), 4),
             "class": DIAGNOSES[int(counts.argmax())],
             "support": {DIAGNOSES[i]: int(counts[i]) for i in range(N_DX)}}

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

    # The budget-adequacy check: Adam-fitted linear arm against its own optimum.
    budget_check = {
        "question": ("Is the shared budget sufficient? Measured on the one arm "
                     "whose optimum is knowable — the linear model, fitted both "
                     "under the shared Adam protocol and to optimality by L-BFGS."),
        "adam_linear_nll": arms["classical"]["nll"],
        "lbfgs_optimum_nll": arms["logistic"]["nll"],
        "gap_nll": round(float(arms["classical"]["nll"] - arms["logistic"]["nll"]), 4),
        "adam_linear_accuracy": arms["classical"]["accuracy"],
        "lbfgs_optimum_accuracy": arms["logistic"]["accuracy"],
        "gap_accuracy": round(
            float(arms["classical"]["accuracy"] - arms["logistic"]["accuracy"]), 4),
    }

    result = {
        "purpose": (
            "Refits every fusion arm under one optimiser protocol — same optimiser, "
            "batch size, epoch cap, learning-rate grid, early-stopping rule, seed and "
            "weight decay — so that any ordering between arms is attributable to model "
            "class rather than to optimisation. Addresses Section 11.5."
        ),
        "protocol": {
            "optimiser": "Adam",
            "learning_rate_grid": list(LR_GRID),
            "learning_rate_selected_per_arm_on": "calibration split, by cross-entropy",
            "batch_size": BATCH,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": f"calibration cross-entropy, patience {PATIENCE} epochs, "
                              "best-epoch weights restored",
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "temperature": "one per arm, fitted on the calibration split (P1)",
            "note": ("The learning rate is searched per arm rather than fixed across "
                     "arms. A single shared rate is not symmetry; it is a choice that "
                     "suits whichever arm it was tuned on. Symmetry is the same grid, "
                     "the same split and the same criterion for every arm."),
        },
        "data": {"source": source, "train_studies": int(len(ytr)),
                 "calibration_studies": int(len(ycal)),
                 "test_studies": int(len(yte)), "patient_disjoint": True,
                 "evidence_channels": d,
                 "constant_channels": [int(i) for i in range(d)
                                       if np.all(Xte[:, i] == 0)]},
        "trainable_parameters": params,
        "fitting": fitting,
        "budget_adequacy": budget_check,
        "majority_class_floor": floor,
        "backends": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in arms.items()},
        "pairwise": pairwise,
        "runtime_seconds": round(time.time() - started, 1),
    }

    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(PREDICTIONS, yte=yte,
             **{f"probs_{k}": arms[k]["_probs"] for k in names})

    print()
    for k in names:
        a = arms[k]
        print(f"{k:10s} params={params[k]:4d} T={a['temperature']:.4f} "
              f"acc={a['accuracy']:.4f} nll={a['nll']:.4f} ece={a['ece']:.4f} "
              f"auroc={a['macro_auroc']:.4f}")
    print()
    for key, v in pairwise.items():
        print(f"{key:24s} dAcc={v['delta_accuracy']:+.4f} "
              f"McNemar Holm={v['mcnemar_p_holm']:.4f}  "
              f"dNLL={v['delta_nll']:+.4f} Wilcoxon Holm={v['nll_wilcoxon_p_holm']:.3e}")
    print(f"\n[budget] Adam-linear NLL {budget_check['adam_linear_nll']:.4f} vs "
          f"L-BFGS optimum {budget_check['lbfgs_optimum_nll']:.4f} "
          f"(gap {budget_check['gap_nll']:+.4f})")
    print(f"[done] wrote {OUTPUT} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
