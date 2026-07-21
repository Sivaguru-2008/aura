"""Train both fusion backends + fit calibration, then register versions.

  1. Build evidence dataset (vision over synthetic images).
  2. Train classical product-of-experts (multinomial softmax regression).
  3. Train the quantum VQC (PennyLane + torch backprop, parameter-shift-capable).
  4. Fit temperature scaling, conformal threshold, OOD stats on the calibration split.
  5. Write artifacts + registry.json.

Designed to finish in seconds-to-a-minute on CPU.
"""
from __future__ import annotations

import json
import time

import numpy as np

from common.config import ARTIFACTS, ensure_dirs, get_settings
from common.mathx import softmax
from schemas.clinical import DIAGNOSES
from services.fusion.device import make_qnode
from services.safety.calibration import (
    Calibration,
    expected_calibration_error,
    fit_conformal,
    fit_temperature,
    ood_stats,
)
from ml.training.dataset import (
    build_evidence_dataset,
    make_splits,
    real_evidence_splits,
)

N_DX = len(DIAGNOSES)


# --------------------------------------------------------------------------- #
# Classical product-of-experts (softmax regression) — pure numpy.
# --------------------------------------------------------------------------- #
def train_classical(X, y, epochs=400, lr=0.3, l2=1e-3, seed=7):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.1, size=(N_DX, d))
    b = np.zeros(N_DX)
    Y = np.eye(N_DX)[y]
    for _ in range(epochs):
        logits = X @ W.T + b
        P = np.array([softmax(row) for row in logits])
        gW = (P - Y).T @ X / n + l2 * W
        gb = (P - Y).mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    return W, b


def train_ensemble(X, y, k=5, seed=7):
    """Deep ensemble: k heads, each on a bootstrap resample with its own seed.

    Bootstrap + seed variation is what makes the members disagree off-distribution,
    which is exactly the signal `SafetyEngine` reads as epistemic uncertainty.
    """
    n = len(y)
    Ws, bs = [], []
    for m in range(k):
        rng = np.random.default_rng(seed + 1000 * (m + 1))
        idx = rng.integers(0, n, size=n)               # bootstrap resample
        Wm, bm = train_classical(X[idx], y[idx], seed=seed + m + 1)
        Ws.append(Wm)
        bs.append(bm)
    return np.stack(Ws), np.stack(bs)


# --------------------------------------------------------------------------- #
# Quantum variational circuit — PennyLane + torch backprop.
# --------------------------------------------------------------------------- #
def train_quantum(X, y, n_qubits, n_layers, epochs=30, lr=0.05, batch=50, seed=7):
    import torch

    torch.manual_seed(seed)
    circuit = make_qnode(n_qubits, n_layers, interface="torch")

    theta = torch.nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2, dtype=torch.float64))
    W = torch.nn.Parameter(0.1 * torch.randn(N_DX, n_qubits, dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
    opt = torch.optim.Adam([theta, W, b], lr=lr)

    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(y)

    def forward_batch(xb):
        # broadcast the whole batch through the circuit in one simulated pass
        z = circuit(xb, theta)                    # list of n_qubits expvals, each (batch,)
        z = torch.stack([zi.to(torch.float64) for zi in z], dim=-1)   # (batch, n_qubits)
        return z @ W.T + b

    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = forward_batch(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    [quantum] epoch {ep:2d}  loss={total / n:.4f}")
    return (theta.detach().numpy(), W.detach().numpy(), b.detach().numpy())


# --------------------------------------------------------------------------- #
# Serving-side numpy evaluators (match services/fusion behaviour).
# --------------------------------------------------------------------------- #
def _classical_logits(W, b, X):
    return X @ W.T + b


def _quantum_logits(theta, W, b, X, n_qubits, n_layers):
    circuit = make_qnode(n_qubits, n_layers, interface="numpy")
    z = np.stack([np.asarray(v) for v in circuit(np.asarray(X), theta)], axis=-1)
    return z @ W.T + b


def run(n_samples: int = 700) -> dict:
    ensure_dirs()
    s = get_settings()
    t0 = time.time()

    # Prefer REAL MIMIC evidence (real DenseNet over real films + report diagnoses)
    # so the fusion model trains on the exact distribution it serves (audit F1).
    # Falls back to the synthetic world — still through the trained backbone — when
    # the corpus is absent (CI / offline demo).
    data_source = "synthetic"
    real = None
    if s.fusion_train_source == "mimic":
        # Cap the common classes so the naturally-skewed real distribution still
        # teaches fusion the rare-but-dangerous labels (pneumothorax, malignancy).
        cap = max(60, s.fusion_train_n // N_DX)
        print(f"[train] building fusion evidence from real MIMIC-CXR studies "
              f"(target n={s.fusion_train_n}, per-class cap {cap}) ...")
        real = real_evidence_splits(n=s.fusion_train_n, split="train", seed=s.seed,
                                    per_class_cap=cap)
    if real is not None:
        Xtr, ytr, Xcal, ycal, Xte, yte = real
        data_source = "mimic-cxr"
        print(f"[train] real evidence set: train={len(ytr)} cal={len(ycal)} test={len(yte)}")
    else:
        print(f"[train] generating {n_samples} synthetic studies + vision features ...")
        tr, cal, te = make_splits(n_samples, seed=s.seed)
        Xtr, ytr = build_evidence_dataset(tr)
        Xcal, ycal = build_evidence_dataset(cal)
        Xte, yte = build_evidence_dataset(te)

    print("[train] fitting classical product-of-experts fusion ...")
    Wc, bc = train_classical(Xtr, ytr)
    np.savez(ARTIFACTS / "fusion_classical.npz", W=Wc, b=bc)

    print(f"[train] fitting deep ensemble ({s.ensemble_size} members) ...")
    Ws, bs = train_ensemble(Xtr, ytr, k=s.ensemble_size, seed=s.seed)
    np.savez(ARTIFACTS / "fusion_ensemble.npz", Ws=Ws, bs=bs)

    print("[train] fitting learnable attention-gated fusion ...")
    from services.fusion.learnable import train_learnable
    A, cvec, Wl, bl = train_learnable(Xtr, ytr, seed=s.seed)
    np.savez(ARTIFACTS / "fusion_learnable.npz", A=A, c=cvec, W=Wl, b=bl)

    print(f"[train] training quantum VQC ({s.n_qubits} qubits, {s.n_layers} layers) ...")
    theta, Wq, bq = train_quantum(Xtr, ytr, s.n_qubits, s.n_layers)
    np.savez(ARTIFACTS / "fusion_quantum.npz", theta=theta, W=Wq, b=bq,
             n_qubits=s.n_qubits, n_layers=s.n_layers)

    # ---- Calibrate the configured (default) backend on the calibration split. ----
    backend = s.fusion_backend
    if backend == "quantum":
        cal_logits = _quantum_logits(theta, Wq, bq, Xcal, s.n_qubits, s.n_layers)
    else:
        cal_logits = _classical_logits(Wc, bc, Xcal)

    T = fit_temperature(cal_logits, ycal)
    cal_probs = np.array([softmax(r / T) for r in cal_logits])
    qhat = fit_conformal(cal_probs, ycal, s.conformal_coverage)
    ece = expected_calibration_error(cal_probs, ycal)
    om, osd = ood_stats(cal_logits, T)
    Calibration(temperature=T, conformal_qhat=qhat, coverage=s.conformal_coverage,
                ood_mean=om, ood_std=osd, ece=ece).save()

    # Class-conditional (Mondrian) conformal thresholds on the calibration split.
    from services.safety.uncertainty import mondrian_qhats
    mq = mondrian_qhats(cal_probs, ycal, s.conformal_coverage, N_DX)
    np.save(ARTIFACTS / "conformal_mondrian.npy", mq)

    # ---- Held-out metrics for the registry. ----
    # Each backend is scored at *its own* fitted temperature. Applying the quantum
    # backend's T to the classical logits (the previous behaviour) understated the
    # classical calibration and produced the inflated headline ECE gap (audit F6).
    T_q = fit_temperature(_quantum_logits(theta, Wq, bq, Xcal, s.n_qubits, s.n_layers), ycal)
    T_c = fit_temperature(_classical_logits(Wc, bc, Xcal), ycal)

    def metrics(logits, y, temperature):
        P = np.array([softmax(r / temperature) for r in logits])
        acc = float((P.argmax(1) == y).mean())
        nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())
        return {"accuracy": round(acc, 4), "nll": round(nll, 4),
                "ece": round(expected_calibration_error(P, y), 4),
                "temperature": round(float(temperature), 4)}

    q_metrics = metrics(_quantum_logits(theta, Wq, bq, Xte, s.n_qubits, s.n_layers), yte, T_q)
    c_metrics = metrics(_classical_logits(Wc, bc, Xte), yte, T_c)

    registry = [
        {"service": "vision", "name": "vision-cxr-region-v1", "version": "1.0",
         "status": "active", "metrics": {"note": "region-feature detectors"}},
        {"service": "fusion", "name": "fusion-vqc-v1", "version": "1.0",
         "status": "active" if backend == "quantum" else "canary",
         "backend": "quantum", "metrics": q_metrics, "train_data": data_source},
        {"service": "fusion", "name": "fusion-poe-v1", "version": "1.0",
         "status": "active" if backend == "classical" else "canary",
         "backend": "classical", "metrics": c_metrics, "train_data": data_source},
        {"service": "safety", "name": "safety-v1", "version": "1.0", "status": "active",
         "calibration": {"temperature": round(T, 4), "conformal_qhat": round(qhat, 4),
                         "coverage": s.conformal_coverage, "ece": round(ece, 4),
                         "train_data": data_source}},
    ]
    (ARTIFACTS / "registry.json").write_text(json.dumps(registry, indent=2))

    dt = time.time() - t0
    print(f"[train] done in {dt:.1f}s  |  source={data_source}  temperature={T:.3f}  "
          f"ECE={ece:.3f}  qhat={qhat:.3f}")
    print(f"[train] quantum test  : {q_metrics}")
    print(f"[train] classical test: {c_metrics}")
    return {"quantum": q_metrics, "classical": c_metrics, "temperature": T,
            "ece": ece, "qhat": qhat, "seconds": round(dt, 1), "data_source": data_source}


if __name__ == "__main__":
    run()
