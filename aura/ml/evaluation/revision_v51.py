"""Revision analyses for manuscript v5.1, executed under
``docs/PREREGISTRATION_ADDENDUM_2026-10.md``.

    python -m aura.ml.evaluation.revision_v51 tost        # A — instant
    python -m aura.ml.evaluation.revision_v51 ablation    # B — re-run 7q ablation, save per-study
    python -m aura.ml.evaluation.revision_v51 resplit     # C — K=5 patient-disjoint re-splits
    python -m aura.ml.evaluation.revision_v51 reupload    # D — data re-uploading ablation
    python -m aura.ml.evaluation.revision_v51 summarize   # bootstraps + merge -> revision_v51.json

B, C and D are slow (minutes each) and independent, so they write their own part files and
can run in parallel; ``summarize`` is the only step that reads across them. Every decision
rule lives in the addendum, not here — this file computes, it does not choose.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.ml.evaluation.quantum_study import load_evidence
from aura.services.safety.calibration import expected_calibration_error, fit_temperature

OUT = ARTIFACTS / "revision_v51.json"
PART = ARTIFACTS / "revision_v51"
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
EPOCHS, N_QUBITS, N_LAYERS = 30, 7, 3
RESPLIT_SEEDS = [101, 102, 103, 104, 105]
MARGIN = 0.010
N_BOOT = 10_000


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _calibrated(logits_cal, y_cal, logits_test, y_test) -> tuple[dict, np.ndarray]:
    """Same scoring as ``quantum_study.score`` but also returns the test probabilities."""
    T = fit_temperature(logits_cal, y_cal)
    P = np.array([softmax(r / T) for r in logits_test])
    nll = -np.log(np.clip(P[np.arange(len(y_test)), y_test], 1e-12, 1))
    m = {
        "accuracy": round(float((P.argmax(1) == y_test).mean()), 4),
        "nll": round(float(nll.mean()), 4),
        "ece": round(float(expected_calibration_error(P, y_test)), 4),
        "temperature": round(float(T), 4),
    }
    return m, P


def _per_study_nll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def _boot_p(b):
    """Two-sided percentile-bootstrap p for H0: mean = 0."""
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


def _boot_summary(b, point):
    return {
        "point": round(float(point), 5),
        "bootstrap_se": round(float(b.std(ddof=1)), 5),
        "ci95": [round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)],
        "bootstrap_p_two_sided": round(_boot_p(b), 5),
        "n_resamples": len(b),
    }


def _seed_stats(deltas):
    d = np.asarray(deltas, float)
    t = stats.ttest_1samp(d, 0.0)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    try:
        w = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        w = float("nan")
    return {
        "mean": round(float(d.mean()), 5), "sd": round(float(d.std(ddof=1)), 5),
        "ci95_t": [round(float(ci[0]), 5), round(float(ci[1]), 5)],
        "t_p": round(float(t.pvalue), 5), "wilcoxon_p": round(w, 5),
        "n_positive": int((d > 0).sum()), "n": len(d),
    }


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"[revision] wrote {path}")


# --------------------------------------------------------------------------- #
# A — TOST
# --------------------------------------------------------------------------- #
def tost(d: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    n, m, se = len(d), float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    t_lo, t_hi = (m + margin) / se, (margin - m) / se
    p_lo, p_hi = float(stats.t.sf(t_lo, n - 1)), float(stats.t.sf(t_hi, n - 1))
    tc = stats.t.ppf(1 - alpha, n - 1)
    return {
        "n": n, "mean_difference": round(m, 5), "se": round(se, 5), "margin": margin,
        "t_lower": round(t_lo, 3), "p_lower": round(p_lo, 5),
        "t_upper": round(t_hi, 3), "p_upper": round(p_hi, 5),
        "p_tost": round(max(p_lo, p_hi), 5),
        "ci90": [round(m - tc * se, 5), round(m + tc * se, 5)],
        "equivalent": bool(max(p_lo, p_hi) < alpha),
    }


def run_tost() -> dict:
    d = np.load(ARTIFACTS / "fusion_matched_optimiser_predictions.npz")
    y = d["yte"]
    diff = _per_study_nll(d["probs_quantum"], y) - _per_study_nll(d["probs_classical"], y)
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [5, 95])
    out = {
        "analysis": "A — TOST quantum VQC vs product-of-experts, matched optimiser",
        "blind": False,
        "contrast": "NLL_quantum - NLL_poe (positive = PoE better)",
        "parametric": tost(diff, MARGIN),
        "bootstrap_sensitivity": {
            "ci90": [round(float(lo), 5), round(float(hi), 5)],
            "inside_margin": bool(lo > -MARGIN and hi < MARGIN),
            "n_resamples": N_BOOT,
        },
    }
    _dump(PART / "A_tost.json", out)
    return out


# --------------------------------------------------------------------------- #
# Circuits
# --------------------------------------------------------------------------- #
def _reupload_qnode(n_qubits, n_layers, entangler, interface):
    import pennylane as qml
    from aura.services.fusion.device import apply_entangler

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def circuit(x, theta, w):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(w[layer][i] * np.pi * x[..., i], wires=i)
                qml.RY(theta[layer][i][0], wires=i)
                qml.RZ(theta[layer][i][1], wires=i)
            apply_entangler(entangler, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def train_reupload(X, y, n_qubits, n_layers, epochs, seed, entangler, lr=0.05, batch=50):
    """Mirror of ``train_fusion.train_quantum``'s loop; only the circuit differs.

    Same seed call, same theta/W/b initialisation and draw order (w is deterministic, =1, so it
    consumes no RNG), same Adam, batch and shuffling — so B and D differ in ansatz alone.
    """
    import torch
    from aura.ml.training.train_fusion import N_DX

    torch.manual_seed(seed)
    circuit = _reupload_qnode(n_qubits, n_layers, entangler, "torch")
    theta = torch.nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2, dtype=torch.float64))
    W = torch.nn.Parameter(0.1 * torch.randn(N_DX, n_qubits, dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(N_DX, dtype=torch.float64))
    w = torch.nn.Parameter(torch.ones(n_layers, n_qubits, dtype=torch.float64))
    opt = torch.optim.Adam([theta, W, b, w], lr=lr)
    Xt, yt, n = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.long), len(y)
    loss_fn = torch.nn.CrossEntropyLoss()
    final = float("nan")
    for ep in range(epochs):
        perm, total = torch.randperm(n), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            z = torch.stack([zi.to(torch.float64) for zi in circuit(Xt[idx], theta, w)], -1)
            loss = loss_fn(z @ W.T + b, yt[idx])
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        final = total / n
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    [reupload/{entangler}] epoch {ep:2d} loss={final:.4f}", flush=True)
    n_par = sum(p.numel() for p in (theta, W, b, w))
    return (theta.detach().numpy(), W.detach().numpy(), b.detach().numpy(),
            w.detach().numpy(), final, int(n_par))


def _reupload_logits(theta, W, b, w, X, entangler):
    c = _reupload_qnode(N_QUBITS, N_LAYERS, entangler, "numpy")
    z = np.stack([np.asarray(v) for v in c(np.asarray(X), theta, w)], -1)
    return z @ W.T + b


def _plain_fit(Xtr, ytr, Xcal, ycal, Xte, yte, seed, entangler):
    from aura.ml.evaluation.quantum_study_7q import _quantum_logits
    from aura.ml.training.train_fusion import train_quantum

    th, W, b = train_quantum(Xtr, ytr, N_QUBITS, N_LAYERS, epochs=EPOCHS, seed=seed,
                             entangler=entangler)
    m, P = _calibrated(_quantum_logits(th, W, b, Xcal, N_QUBITS, N_LAYERS, entangler), ycal,
                       _quantum_logits(th, W, b, Xte, N_QUBITS, N_LAYERS, entangler), yte)
    return m, P


def _data7():
    Xtr, ytr, Xcal, ycal, Xte, yte, src = load_evidence(n=900, seed=7)
    assert src == "mimic-cxr", src
    return (Xtr[:, :N_QUBITS], ytr, Xcal[:, :N_QUBITS], ycal, Xte[:, :N_QUBITS], yte)


# --------------------------------------------------------------------------- #
# B — re-run 7q ablation saving per-study probabilities
# --------------------------------------------------------------------------- #
def run_ablation() -> dict:
    Xtr, ytr, Xcal, ycal, Xte, yte = _data7()
    published = {s["seed"]: s for s in
                 json.loads((ARTIFACTS / "entanglement_multiseed_7q.json").read_text())["per_seed"]}
    rows, Pr, Pp = [], [], []
    for seed in SEEDS:
        print(f"[B] seed {seed}", flush=True)
        mr, P_r = _plain_fit(Xtr, ytr, Xcal, ycal, Xte, yte, seed, "ring")
        mp, P_p = _plain_fit(Xtr, ytr, Xcal, ycal, Xte, yte, seed, "none")
        dn = round(mr["nll"] - mp["nll"], 4)
        pub = published[seed]
        repro = (abs(mr["nll"] - pub["entangled"]["nll"]) < 5e-5
                 and abs(mp["nll"] - pub["product"]["nll"]) < 5e-5)
        print(f"    ring {mr['nll']:.4f} prod {mp['nll']:.4f} d={dn:+.4f} "
              f"(published {pub['delta_nll']:+.4f}) reproduced={repro}", flush=True)
        rows.append({"seed": seed, "entangled": mr, "product": mp, "delta_nll": dn,
                     "published_delta_nll": pub["delta_nll"], "reproduced": repro})
        Pr.append(P_r); Pp.append(P_p)
    np.savez(PART / "B_probs.npz", yte=yte, ring=np.array(Pr), none=np.array(Pp),
             seeds=np.array(SEEDS))
    out = {"analysis": "B — 7q ablation re-run with per-study predictions", "blind": False,
           "all_reproduced": all(r["reproduced"] for r in rows), "per_seed": rows,
           "seed_level": _seed_stats([r["delta_nll"] for r in rows])}
    _dump(PART / "B_ablation.json", out)
    return out


# --------------------------------------------------------------------------- #
# C — K re-splits
# --------------------------------------------------------------------------- #
def run_resplit() -> dict:
    e = np.load(ARTIFACTS / "quantum_study_evidence.npz")
    X = np.vstack([e["Xtr"], e["Xcal"], e["Xte"]])[:, :N_QUBITS]
    y = np.concatenate([e["ytr"], e["ycal"], e["yte"]])
    n_tr, n_cal = int(0.6 * len(X)), int(0.2 * len(X))
    rows = []
    for k in RESPLIT_SEEDS:
        idx = np.random.default_rng(k).permutation(len(X))
        tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]
        print(f"[C] split {k}: {len(tr)}/{len(cal)}/{len(te)}", flush=True)
        mr, _ = _plain_fit(X[tr], y[tr], X[cal], y[cal], X[te], y[te], 7, "ring")
        mp, _ = _plain_fit(X[tr], y[tr], X[cal], y[cal], X[te], y[te], 7, "none")
        dn = round(mr["nll"] - mp["nll"], 4)
        print(f"    ring {mr['nll']:.4f} prod {mp['nll']:.4f} d={dn:+.4f}", flush=True)
        rows.append({"split_seed": k, "n_test": len(te), "entangled": mr, "product": mp,
                     "delta_nll": dn})
    out = {"analysis": "C — K=5 patient-disjoint 60/20/20 re-splits, training seed 7",
           "blind": True, "per_split": rows,
           "split_level": _seed_stats([r["delta_nll"] for r in rows])}
    _dump(PART / "C_resplit.json", out)
    return out


# --------------------------------------------------------------------------- #
# D — data re-uploading ablation
# --------------------------------------------------------------------------- #
def run_reupload() -> dict:
    Xtr, ytr, Xcal, ycal, Xte, yte = _data7()
    rows, Pr, Pp = [], [], []
    for seed in SEEDS:
        print(f"[D] seed {seed}", flush=True)
        arms, probs = {}, {}
        for ent in ("ring", "none"):
            t0 = time.time()
            th, W, b, w, tr_nll, n_par = train_reupload(Xtr, ytr, N_QUBITS, N_LAYERS, EPOCHS,
                                                        seed, ent)
            m, P = _calibrated(_reupload_logits(th, W, b, w, Xcal, ent), ycal,
                               _reupload_logits(th, W, b, w, Xte, ent), yte)
            m.update(final_train_nll=round(tr_nll, 4), trainable_parameters=n_par,
                     train_seconds=round(time.time() - t0, 1))
            arms[ent], probs[ent] = m, P
        dn = round(arms["ring"]["nll"] - arms["none"]["nll"], 4)
        print(f"    ring {arms['ring']['nll']:.4f} prod {arms['none']['nll']:.4f} d={dn:+.4f}",
              flush=True)
        rows.append({"seed": seed, "entangled": arms["ring"], "product": arms["none"],
                     "delta_nll": dn})
        Pr.append(probs["ring"]); Pp.append(probs["none"])
    np.savez(PART / "D_probs.npz", yte=yte, ring=np.array(Pr), none=np.array(Pp),
             seeds=np.array(SEEDS))
    out = {"analysis": "D — data re-uploading ablation, 7 qubits, L=3", "blind": True,
           "trainable_parameters": {"entangled": rows[0]["entangled"]["trainable_parameters"],
                                    "product": rows[0]["product"]["trainable_parameters"]},
           "per_seed": rows, "seed_level": _seed_stats([r["delta_nll"] for r in rows])}
    _dump(PART / "D_reupload.json", out)
    return out


# --------------------------------------------------------------------------- #
# summarize — bootstraps over studies (= patients) and merge
# --------------------------------------------------------------------------- #
def _study_bootstrap(npz) -> dict:
    d = np.load(npz)
    y = d["yte"]
    nr = np.stack([_per_study_nll(P, y) for P in d["ring"]])     # (seeds, studies)
    nn = np.stack([_per_study_nll(P, y) for P in d["none"]])
    D = nr - nn
    per_study = D.mean(0)
    rng = np.random.default_rng(0)
    S, N = D.shape
    primary = np.array([per_study[rng.integers(0, N, N)].mean() for _ in range(N_BOOT)])
    hier = np.array([D[rng.integers(0, S, S)][:, rng.integers(0, N, N)].mean()
                     for _ in range(N_BOOT)])
    return {
        "n_studies": int(N), "n_seeds": int(S),
        "primary_study_bootstrap": _boot_summary(primary, per_study.mean()),
        "hierarchical_seed_x_study_bootstrap": _boot_summary(hier, D.mean()),
        "seed_ensemble_mean_nll": {"entangled": round(float(nr.mean()), 5),
                                   "product": round(float(nn.mean()), 5)},
    }


def summarize() -> dict:
    parts = {k: json.loads((PART / f).read_text()) for k, f in
             (("A", "A_tost.json"), ("B", "B_ablation.json"),
              ("C", "C_resplit.json"), ("D", "D_reupload.json")) if (PART / f).exists()}
    if "B" in parts:
        parts["B"]["study_bootstrap"] = _study_bootstrap(PART / "B_probs.npz")
    if "D" in parts:
        parts["D"]["study_bootstrap"] = _study_bootstrap(PART / "D_probs.npz")
    out = {"preregistration": "docs/PREREGISTRATION_ADDENDUM_2026-10.md",
           "resampling_unit_note": "one study per patient (build_real_evidence_dataset), so "
                                   "study-level resampling is patient-level resampling",
           **parts}
    if "B" in parts and "C" in parts:
        out["variance_components"] = {
            "sigma_init_seed_sd": parts["B"]["seed_level"]["sd"],
            "sigma_data_split_sd": parts["C"]["split_level"]["sd"],
        }
    _dump(OUT, out)
    return out


if __name__ == "__main__":
    ensure_dirs()
    PART.mkdir(parents=True, exist_ok=True)
    {"tost": run_tost, "ablation": run_ablation, "resplit": run_resplit,
     "reupload": run_reupload, "summarize": summarize}[sys.argv[1]]()
