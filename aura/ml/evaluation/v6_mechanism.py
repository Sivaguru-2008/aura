"""Mechanism measurements M1–M3 of ``docs/PREREGISTRATION_v6_2026-10-03.md`` §4.

    python -m aura.ml.evaluation.v6_mechanism m1     # interaction-order spectrum of trained circuits
    python -m aura.ml.evaluation.v6_mechanism m2     # matched trigonometric surrogate (needs test set)
    python -m aura.ml.evaluation.v6_mechanism m3     # kernel diagnostics (fusion encoding, glioma QKL)

Explanatory only: nothing here is a hypothesis test of the primary claim.

M1 rests on one identity. With a single RY(pi x_i) encoding per qubit, the encoded state is
rho(x) = prod_i (I + cos(pi x_i) Z + sin(pi x_i) X) / 2, so every readout of the circuit is
<Z_j>(x) = sum_P c_P prod_i f_{P_i}(x_i) over P in {I, X, Z}^n with f_I = 1, f_Z = cos(pi x),
f_X = sin(pi x), and c_P = Tr(U^dag Z_j U P) / 2^n. A term's interaction order is the number of
non-identity factors in P. The coefficients are exact, not fitted.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.ml.evaluation import v6_confirmatory as v6
from aura.services.safety.calibration import fit_temperature

PART = ARTIFACTS / "v6"
OUT = ARTIFACTS / "v6_mechanism.json"

_P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1.0, -1.0])}


# --------------------------------------------------------------------------- #
# M1
# --------------------------------------------------------------------------- #
def _unitary(theta: np.ndarray, n: int, entangler: str) -> np.ndarray:
    """The trainable part of the circuit (after encoding) as a 2^n x 2^n matrix, built by
    running the verified simulator's gate sequence on the computational basis."""
    import torch

    from aura.services.fusion.device import entangler_pairs

    L = theta.shape[0]
    dim = 2 ** n
    psi = torch.eye(dim, dtype=torch.complex128).reshape((dim,) + (2,) * n)

    def apply1(psi, q, m):
        psi = psi.movedim(q + 1, -1)
        psi = psi @ m.T
        return psi.movedim(-1, q + 1)

    def cnot(psi, c, t):
        psi = psi.clone()
        idx = [slice(None)] * (n + 1)
        idx[c + 1] = 1
        tt = t if t < c else t - 1
        psi[tuple(idx)] = psi[tuple(idx)].flip(tt + 1)
        return psi

    th = torch.tensor(theta)
    for layer in range(L):
        for i in range(n):
            a = th[layer, i, 0]
            ry = torch.tensor([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]],
                              dtype=torch.complex128)
            b = complex(th[layer, i, 1])
            rz = torch.tensor([[np.exp(-0.5j * b), 0], [0, np.exp(0.5j * b)]],
                              dtype=torch.complex128)
            psi = apply1(psi, i, ry)
            psi = apply1(psi, i, rz)
        for c, t in entangler_pairs(entangler, n):
            psi = cnot(psi, c, t)
    # row k of psi is U|k>; U[:, k] = psi[k]
    return psi.reshape(dim, dim).T.numpy()


def _pauli_coeffs(O: np.ndarray, n: int) -> np.ndarray:
    """c_P = Tr(O P)/2^n for P in {I, X, Z}^n, as an array of shape (3,)*n."""
    T = O.reshape((2,) * (2 * n))
    basis = np.stack([_P["I"], _P["X"], _P["Z"]])          # [3,2,2]
    # contract each (row_i, col_i) index pair with P_i^T: Tr(O P) = sum O[r,c] P[c,r]
    for i in range(n):
        # current tensor: [3]*i + [2]*(2(n-i)) with remaining pairs (r_i..., c_i...)
        k = n - i                                          # remaining qubit pairs
        T = np.moveaxis(T, [i, i + k], [-2, -1])           # bring r_i, c_i to the end
        T = np.einsum("...rc,pcr->...p", T, basis)
        T = np.moveaxis(T, -1, i)
    return np.real(T) / 2 ** n


def _order_mask(n: int) -> np.ndarray:
    idx = np.array(list(itertools.product(range(3), repeat=n)))
    return (idx > 0).sum(1).reshape((3,) * n)


def _features(X: np.ndarray, n: int, max_order: int | None = None):
    """Rows of prod_i f_{P_i}(x_i) for all P (optionally only order <= max_order)."""
    c, s = np.cos(np.pi * X[:, :n]), np.sin(np.pi * X[:, :n])
    per = np.stack([np.ones_like(c), s, c], axis=-1)      # I, X(sin), Z(cos)
    Ps = [P for P in itertools.product(range(3), repeat=n)
          if max_order is None or sum(p > 0 for p in P) <= max_order]
    F = np.ones((len(X), len(Ps)))
    for k, P in enumerate(Ps):
        for i, p in enumerate(P):
            if p:
                F[:, k] *= per[:, i, p]
    return F, Ps


def spectrum(theta, W, b, n, entangler, X) -> dict:
    U = _unitary(theta, n, entangler)
    order = _order_mask(n)
    coeffs = []
    for j in range(n):
        Zj = np.diag([1.0 if ((k >> (n - 1 - j)) & 1) == 0 else -1.0 for k in range(2 ** n)])
        O = U.conj().T @ Zj @ U
        coeffs.append(_pauli_coeffs(O, n))
    coeffs = np.stack(coeffs)                              # [n_readout, 3,...,3]
    energy = {int(o): float((coeffs[:, order == o] ** 2).sum()) for o in range(n + 1)}
    tot = sum(energy.values())
    # logit decomposition on data: logit_k = sum_j W_kj <Z_j> + b_k
    flat = coeffs.reshape(n, -1)                           # [n, 3^n]
    lin = W @ flat                                         # [K, 3^n]
    ord_flat = order.ravel()
    low = np.where(ord_flat <= 1)[0]
    r2 = []
    for i in range(0, len(X), 4000):
        F, _ = _features(X[i:i + 4000], n)
        full = F @ lin.T
        trunc = F[:, low] @ lin[:, low].T
        r2.append((full, trunc))
    full = np.concatenate([a for a, _ in r2])
    trunc = np.concatenate([b_ for _, b_ in r2])
    var_full = full.var(0)
    resid = (full - trunc).var(0)
    share_ge2 = float(np.mean(np.where(var_full > 1e-12, resid / np.maximum(var_full, 1e-12),
                                       0.0)))
    return {"qubits": n, "entangler": entangler,
            "coefficient_energy_share_by_order": {o: round(e / tot, 6) for o, e in energy.items()},
            "logit_variance_share_order_ge2": round(share_ge2, 6),
            "sanity_identity_holds": bool(np.allclose(
                (_features(X[:50], n)[0] @ flat.T), _readout(theta, n, entangler, X[:50]),
                atol=1e-8))}


def _readout(theta, n, entangler, X):
    import torch

    f = v6.torch_circuit(n, theta.shape[0], entangler)
    with torch.no_grad():
        return f(torch.tensor(X[:, :n]), torch.tensor(theta)).numpy()


def run_m1() -> dict:
    Xte = v6.test_set()[0] if v6.FRESH.exists() else v6.dev_splits()[2]
    out = {"evaluated_on": "C1 test set" if v6.FRESH.exists() else "development calibration split"}
    z = np.load(PART / "C1_y_u0_quantum_params.npz")
    out["C1_quantum_8q_ring"] = spectrum(z["p0"], z["p1"], z["p2"], 8, "ring", Xte)
    print("C1", out["C1_quantum_8q_ring"], flush=True)
    rows = []
    from aura.ml.evaluation.revision_v51 import SEEDS
    for seed in SEEDS:
        r = {"seed": seed}
        for ent in ("ring", "none"):
            p = PART / f"C1b_seed{seed}_{ent}_params.npz"
            if p.exists():
                q = np.load(p)
                r[ent] = spectrum(q["theta"], q["W"], q["b"], 7, ent, Xte)
        rows.append(r)
        print(seed, {k: v.get("logit_variance_share_order_ge2") for k, v in r.items()
                     if isinstance(v, dict)}, flush=True)
    out["C1b_7q"] = rows
    prod_ok = all(r["none"]["coefficient_energy_share_by_order"].get(o, 0) < 1e-12
                  for r in rows if "none" in r for o in range(2, 8))
    out["sanity_product_has_no_order_ge2"] = bool(prod_ok)
    return out


# --------------------------------------------------------------------------- #
# M2 — matched trigonometric surrogate
# --------------------------------------------------------------------------- #
def run_m2() -> dict:
    from sklearn.linear_model import LogisticRegression

    from aura.ml.evaluation.matched_optimiser import _full_width_logits

    Xtr, ytr, Xcal, ycal = v6.dev_splits()
    Xte, yte = v6.test_set()
    n = 8
    qP = np.load(PART / "C1_y_u0_probs.npz")["probs_quantum"]
    q_nll = -np.log(np.clip(qP[np.arange(len(yte)), yte], 1e-12, 1))
    out = {}
    for K in (1, 2):
        Ftr, Ps = _features(Xtr, n, K)
        keep = Ftr.std(0) > 1e-12                          # constant channel -> constant cols
        keep[0] = False                                    # intercept is fitted separately
        Fc, Ft = _features(Xcal, n, K)[0][:, keep], _features(Xte, n, K)[0][:, keep]
        Ftr = Ftr[:, keep]
        best = None
        for C in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = LogisticRegression(C=C, max_iter=10000).fit(Ftr, ytr)
            Lc = _full_width_logits(m, Fc)
            T = fit_temperature(Lc, ycal)
            Pc = np.array([softmax(r / T) for r in Lc])
            v = float(-np.log(np.clip(Pc[np.arange(len(ycal)), ycal], 1e-12, 1)).mean())
            if best is None or v < best[0]:
                best = (v, C, m, T)
        _, C, m, T = best
        P = np.array([softmax(r / T) for r in _full_width_logits(m, Ft)])
        s_nll = -np.log(np.clip(P[np.arange(len(yte)), yte], 1e-12, 1))
        d = q_nll - s_nll
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[f"order_le_{K}"] = {
            "features": int(keep.sum()), "selected_C": C,
            "surrogate_nll": round(float(s_nll.mean()), 5),
            "surrogate_accuracy": round(float((P.argmax(1) == yte).mean()), 5),
            "quantum_minus_surrogate_nll": round(float(d.mean()), 5),
            "ci95": [round(float(d.mean() - 1.96 * se), 5), round(float(d.mean() + 1.96 * se), 5)]}
        print(K, out[f"order_le_{K}"], flush=True)
    return out


# --------------------------------------------------------------------------- #
# M3 — kernel diagnostics
# --------------------------------------------------------------------------- #
def _diagnostics(KQ: np.ndarray, KC: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> dict:
    N = len(KQ)
    off = ~np.eye(N, dtype=bool)

    def tn(K):
        return K * N / np.trace(K)

    def deff(K):
        ev = np.clip(np.linalg.eigvalsh(K), 0, None)
        return float(ev.sum() ** 2 / (ev ** 2).sum())

    Y = np.eye(int(y.max()) + 1)[y]
    Yk = Y @ Y.T

    def kta(K):
        return float((K * Yk).sum() / np.sqrt((K * K).sum() * (Yk * Yk).sum()))

    Q, C = tn(KQ), tn(KC)
    w, V = np.linalg.eigh(Q)
    sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sq @ np.linalg.solve(C + lam * np.eye(N), sq)
    g = float(np.sqrt(np.max(np.abs(np.linalg.eigvalsh((M + M.T) / 2)))))
    return {"n": N,
            "quantum_offdiag_mean": float(KQ[off].mean()), "quantum_offdiag_var": float(KQ[off].var()),
            "classical_offdiag_mean": float(KC[off].mean()), "classical_offdiag_var": float(KC[off].var()),
            "effective_dimension_quantum": deff(Q), "effective_dimension_classical": deff(C),
            "kta_quantum": kta(KQ), "kta_classical": kta(KC),
            "geometric_difference_g_CQ": g, "g_bound_sqrt_n": float(np.sqrt(N)),
            "regulariser": lam}


def run_m3() -> dict:
    import torch
    from sklearn.metrics.pairwise import rbf_kernel

    out = {}
    # (a) fusion encoding: product of RY(pi x) states -> closed-form fidelity kernel
    X = np.load(v6.DEV)["X"]
    y = np.load(v6.DEV)["y_u0"]
    idx = np.random.default_rng(7).choice(len(X), 2000, replace=False)
    Xs, ys = X[idx], y[idx]
    closed = np.prod(np.cos(np.pi * (Xs[:, None, :] - Xs[None, :, :]) / 2) ** 2, axis=-1)
    # statevector check on 200 points
    f = v6.torch_circuit(8, 0, "none")

    def states(A):
        n = 8
        c, s = np.cos(np.pi * A / 2), np.sin(np.pi * A / 2)
        st = np.ones((len(A), 1))
        for i in range(n):
            st = np.einsum("ba,bq->baq", st, np.stack([c[:, i], s[:, i]], 1)).reshape(len(A), -1)
        return st
    S = states(Xs[:200])
    sv = np.abs(S @ S.T) ** 2
    # and the readout identity from the simulator with zero layers: <Z_i> = cos(pi x_i)
    with torch.no_grad():
        z = f(torch.tensor(Xs[:200]), torch.zeros(0, 8, 2, dtype=torch.float64)).numpy()
    med = np.median(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
    gamma = 1.0 / med
    out["fusion_encoding"] = {
        "closed_form": "prod_i cos^2(pi (x_i - x'_i) / 2)",
        "closed_form_max_abs_diff_vs_statevector": float(np.abs(sv - closed[:200, :200]).max()),
        "readout_identity_max_abs_diff": float(np.abs(z - np.cos(np.pi * Xs[:200])).max()),
        "rbf_gamma_rule": "median heuristic 1/median(||x - x'||^2) (deviation D1: no RBF arm exists in fusion)",
        "rbf_gamma": float(gamma),
        **_diagnostics(closed, rbf_kernel(Xs, gamma=gamma), ys)}
    print("fusion", out["fusion_encoding"], flush=True)

    # (b) glioma QKL, 111 held-out subjects, bandwidth / gamma selected in every fold
    from aura.ml.evaluation.qkl_full_coverage import _make_projection
    from aura.ml.training.train_qkl import load_embeddings

    emb = ARTIFACTS / "brain" / "embeddings_official" / "brats2020_369.npz"
    data = load_embeddings(emb)
    Xe, ye, subj = data["embedding"], data["grade"], data["subject"]
    side = json.loads(emb.with_suffix(".json").read_text(encoding="utf-8"))
    keep_ids = [s["subject_index"] for s in side["subjects"] if s["split"] != "train"]
    m = np.isin(subj, keep_ids)
    Xe, ye, subj = Xe[m], ye[m], subj[m]
    rng = np.random.default_rng(7)
    pick = np.concatenate([rng.choice(np.where(subj == s)[0], min(10, (subj == s).sum()),
                                      replace=False) for s in np.unique(subj)])
    Xe, ye = Xe[pick], ye[pick]
    clf = _make_projection(Xe, 10, 7, 0.25)
    St = clf.feature_states(clf.project_batch(Xe))
    KQ = np.abs(St @ St.conj().T) ** 2
    mu, sd = Xe.mean(0), Xe.std(0)
    sd[sd < 1e-9] = 1.0
    KC = rbf_kernel((Xe - mu) / sd, gamma=0.01)
    out["glioma_qkl"] = {"subjects": int(len(np.unique(subj))), "slices": int(len(ye)),
                         "slices_per_subject_cap": 10,
                         "quantum_bandwidth": 0.25, "rbf_gamma": 0.01,
                         "selection_note": "bandwidth 0.25 and gamma 0.01 were selected in all 5 folds of the published run; 0.25 is the lowest point of the quantum grid",
                         **_diagnostics(KQ, KC, ye.astype(int))}
    # bandwidth sweep of concentration, to show where the selection sits
    sweep = {}
    for bw in (0.25, 0.5, 1.0, 2.0, 4.0):
        c = _make_projection(Xe, 10, 7, bw)
        S2 = c.feature_states(c.project_batch(Xe))
        K2 = np.abs(S2 @ S2.conj().T) ** 2
        d = _diagnostics(K2, KC, ye.astype(int))
        sweep[str(bw)] = {k: d[k] for k in ("quantum_offdiag_mean", "quantum_offdiag_var",
                                             "effective_dimension_quantum", "kta_quantum",
                                             "geometric_difference_g_CQ")}
    out["glioma_qkl"]["bandwidth_sweep"] = sweep
    print("glioma", out["glioma_qkl"], flush=True)
    return out


if __name__ == "__main__":
    ensure_dirs()
    cmd = sys.argv[1]
    res = json.loads(OUT.read_text()) if OUT.exists() else {
        "preregistration": "docs/PREREGISTRATION_v6_2026-10-03.md §4 (commit 61e777b)"}
    res[cmd.upper()] = {"m1": run_m1, "m2": run_m2, "m3": run_m3}[cmd]()
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")
