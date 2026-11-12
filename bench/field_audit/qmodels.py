"""The quantum models of the included studies, as fit/predict_proba estimators.

Each class implements what its paper specifies; where a paper is silent the choice is
named in the docstring and in the study's result file. Every circuit is built from
``qsim`` primitives that ``test_qsim.py`` checks against Qiskit.

All models take features already scaled by the study's shared pipeline and do their
own quantum-specific reduction (PCA to the qubit count, angle rescaling) inside
``fit`` — on training data only.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from . import qsim

T = lambda a: torch.as_tensor(np.asarray(a), dtype=qsim.RDTYPE)  # noqa: E731


def _bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


# --------------------------------------------------------------------------- #
# 40567645 — CKD: RawFeatureVector + RealAmplitudes(full) + COBYLA
# --------------------------------------------------------------------------- #
class CKDVQC:
    """Qiskit-ML VQC as the paper describes it.

    Encoding: ``RawFeatureVector`` (amplitude) of the 13 LASSO features into 4 qubits.
    Ansatz: ``RealAmplitudes(4, reps, entanglement='full')``. Loss: cross-entropy.
    Optimiser: COBYLA. Readout: Qiskit-ML VQC's default two-class interpretation,
    ``class = bitstring % 2``, i.e. P(class 1) = P(qubit 0 measured 1).
    Paper-silent: COBYLA iteration cap (grid), initial point (seeded normal, sd 0.1).
    """

    def __init__(self, reps=3, maxiter=200, seed=7):
        self.reps, self.maxiter, self.seed = reps, maxiter, seed

    def _p1(self, theta, X):
        s = qsim.State.amplitude(X, 4)
        qsim.real_amplitudes(s, theta, self.reps, "full")
        return (1 - s.expval_z(0)) / 2

    def fit(self, X, y):
        Xt, yt = T(X), T(y)
        rng = np.random.default_rng(self.seed)
        x0 = rng.normal(0, 0.1, 4 * (self.reps + 1))
        f = lambda th: float(_bce(self._p1(T(th), Xt), yt))  # noqa: E731
        res = minimize(f, x0, method="COBYLA", options={"maxiter": self.maxiter})
        self.theta = T(res.x)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return self._p1(self.theta, T(X)).numpy()


# --------------------------------------------------------------------------- #
# 40676161 — glioma: Rx+CX feature map, Ry/Rz/CY ansatz, AQGD
# --------------------------------------------------------------------------- #
class GliomaVQC:
    """VQC-1 of the paper: feature map of RX(x_i) then a CX chain; ansatz of RY, RZ on
    every qubit then a CY chain, repeated ``reps`` times; readout as Qiskit-ML VQC.

    AQGD (analytic gradients, momentum 0.25) is reproduced as full-batch gradient
    descent with momentum 0.25 on exact gradients. Paper-silent: reps and learning
    rate (grid), iterations (100; the paper's loss curve flattens by ~60), init.
    """

    def __init__(self, reps=2, lr=0.5, iters=100, seed=7):
        self.reps, self.lr, self.iters, self.seed = reps, lr, iters, seed

    def _p1(self, theta, X):
        n = X.shape[1]
        s = qsim.State.zeros(X.shape[0], n)
        for q in range(n):
            s.rx(q, X[:, q])
        for q in range(n - 1):
            s.cx(q, q + 1)
        k = 0
        for _ in range(self.reps):
            for q in range(n):
                s.ry(q, theta[k]); s.rz(q, theta[k + 1]); k += 2
            for q in range(n - 1):
                s.cy(q, q + 1)
        return (1 - s.expval_z(0)) / 2

    def fit(self, X, y):
        Xt, yt = T(X), T(y)
        torch.manual_seed(self.seed)
        n = X.shape[1]
        theta = torch.nn.Parameter(0.1 * torch.randn(2 * n * self.reps, dtype=qsim.RDTYPE))
        opt = torch.optim.SGD([theta], lr=self.lr, momentum=0.25)
        for _ in range(self.iters):
            opt.zero_grad()
            loss = _bce(self._p1(theta, Xt), yt)
            loss.backward()
            opt.step()
        self.theta = theta.detach()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return self._p1(self.theta, T(X)).numpy()


# --------------------------------------------------------------------------- #
# 42286086 — fidelity kernels (angle / amplitude / ZZ), QSVM + QKNN, LR stacking
# --------------------------------------------------------------------------- #
def _angle_states(A):
    s = qsim.State.zeros(A.shape[0], A.shape[1])
    for q in range(A.shape[1]):
        s.ry(q, A[:, q])
    return s


def _zz_states(A):
    # "Hadamard + RZ layer, then ZZ entangling gates on adjacent pairs" — one layer,
    # linear entanglement, Qiskit's (pi - x)(pi - y) pair angle.
    return qsim.zz_feature_map(A, reps=1, entanglement="linear")


def _amp_states(Z):
    n = int(math.ceil(math.log2(Z.shape[1])))
    return qsim.State.amplitude(Z, n)


class QKNN:
    """k-NN on the quantum distance sqrt(1 - fidelity); probability = neighbour vote."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, K_train, y):
        self.y = np.asarray(y)
        return self

    def predict_proba_from(self, K_eval):
        D = np.sqrt(np.clip(1 - K_eval, 0, None))
        nn = np.argsort(D, axis=1)[:, :self.k]
        return self.y[nn].mean(axis=1)


class KernelStack:
    """The paper's adaptive stack: six base learners {QSVM, QKNN} x {angle, amplitude,
    ZZ} on PCA-8 features, combined by a class-balanced logistic regression trained on
    5-fold out-of-fold base predictions.

    ``bandwidth`` scales the [0, pi] angles (Shaydulin & Wild); ``C`` is the QSVM
    regularisation. Paper-silent: C (sklearn default 1 in the paper), bandwidth (1).
    """

    def __init__(self, C=1.0, bandwidth=1.0, n_components=8, k=5, seed=7):
        self.C, self.bw, self.nc, self.k, self.seed = C, bandwidth, n_components, k, seed

    def _prep_fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(self.nc, random_state=self.seed).fit(self.sc.transform(X))
        Z = self.pca.transform(self.sc.transform(X))
        self.mm = MinMaxScaler((0, math.pi)).fit(Z)
        return Z

    def _states(self, X):
        Z = self.pca.transform(self.sc.transform(X))
        A = np.clip(self.mm.transform(Z), 0, math.pi) * self.bw
        return {"angle": _angle_states(T(A)), "amp": _amp_states(T(Z)),
                "zz": _zz_states(T(A))}

    def _base_fit_predict(self, S_tr, y_tr, S_ev_list):
        """Fit the six base learners on S_tr; return their P(1) on each eval set."""
        outs = [[] for _ in S_ev_list]
        for enc in ("angle", "amp", "zz"):
            K_tr = qsim.fidelity_kernel(S_tr[enc], S_tr[enc])
            svm = SVC(kernel="precomputed", C=self.C, probability=True,
                      random_state=self.seed).fit(K_tr, y_tr)
            knn = QKNN(self.k).fit(K_tr, y_tr)
            for i, S_ev in enumerate(S_ev_list):
                K_ev = qsim.fidelity_kernel(S_ev[enc], S_tr[enc])
                outs[i].append(svm.predict_proba(K_ev)[:, 1])
                outs[i].append(knn.predict_proba_from(K_ev))
        return [np.column_stack(o) for o in outs]

    @staticmethod
    def _subset(S, idx):
        return {k: qsim.State(v.psi[idx], v.n) for k, v in S.items()}

    def fit(self, X, y):
        y = np.asarray(y)
        self._prep_fit(X)
        S = self._states(X)
        oof = np.zeros((len(y), 6))
        inner = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        for tr, va in inner.split(X, y):
            (oof[va],) = self._base_fit_predict(self._subset(S, tr), y[tr],
                                                [self._subset(S, va)])
        self.meta = LogisticRegression(class_weight="balanced", max_iter=5000).fit(oof, y)
        self.S_train, self.y_train = S, y
        return self

    def predict_proba(self, X):
        (B,) = self._base_fit_predict(self.S_train, self.y_train, [self._states(X)])
        return self.meta.predict_proba(B)[:, 1]


# --------------------------------------------------------------------------- #
# 38441727 — EEG: PCA-k + PauliFeatureMap(['Z' or 'X'], reps=2) QSVM
# --------------------------------------------------------------------------- #
class PauliQSVM:
    """Qiskit QSVC with a single-Pauli feature map on PCA-k features.

    Paper-silent: the input scaling before the feature map (MinMax to [0, pi] x
    bandwidth, grid), C (sklearn default 1, as the paper gives none).
    """

    def __init__(self, k=5, pauli="Z", bandwidth=1.0, C=1.0, seed=7):
        self.k, self.pauli, self.bw, self.C, self.seed = k, pauli, bandwidth, C, seed

    def _A(self, X):
        Z = self.pca.transform(X)
        return T(np.clip(self.mm.transform(Z), 0, math.pi) * self.bw)

    def fit(self, X, y):
        self.pca = PCA(self.k, random_state=self.seed).fit(X)
        self.mm = MinMaxScaler((0, math.pi)).fit(self.pca.transform(X))
        self.S = qsim.pauli_feature_map_single(self._A(X), self.pauli, reps=2)
        K = qsim.fidelity_kernel(self.S, self.S)
        self.svm = SVC(kernel="precomputed", C=self.C, probability=True,
                       random_state=self.seed).fit(K, y)
        return self

    def predict_proba(self, X):
        S = qsim.pauli_feature_map_single(self._A(X), self.pauli, reps=2)
        return self.svm.predict_proba(qsim.fidelity_kernel(S, self.S))[:, 1]


# --------------------------------------------------------------------------- #
# 41513819 — heart: angle encoding, double-CZ ansatz, sum-Z sigmoid, quantum dropout
# --------------------------------------------------------------------------- #
class HeartQNN:
    """The paper's QNN. Encoding RY(x_i) on 11 qubits (x scaled to [0, angle_range]);
    per layer: RY on all qubits, CZ on (0,1),(2,3),..., RY on qubits 1..n-2, CZ on
    (1,2),(3,4),... within 1..n-2; output sigmoid(sum_i <Z_i>); cross-entropy; Adam
    lr0 halved after 50 non-improving epochs, stop after 100; init N(0,1) x 0.01;
    quantum dropout (Scala et al.): per step, each layer is chosen with p=0.05 and in a
    chosen layer each rotation is dropped (angle zeroed) with p=0.2.

    The paper stops on the *test* cost. Here the stopping cost is a stratified 20%
    validation split of the training data. Paper-silent: the rotation axis (RY), the
    encoding range (grid), maximum epochs (1000).
    """

    def __init__(self, layers=1, dropout=True, angle_range=2 * math.pi, lr=0.01,
                 max_epochs=1000, seed=7, device=None):
        self.layers, self.dropout, self.range = layers, dropout, angle_range
        self.lr, self.max_epochs, self.seed = lr, max_epochs, seed
        self.device = device or "cpu"

    # Every gate in this circuit (RY, CZ) is real and the input state is real, so the
    # simulation runs in real arithmetic: half the memory and no complex einsums.
    # ``test_heart_qnn_matches_qsim`` checks it against the generic complex simulator.
    @staticmethod
    def _ry_real(psi, n, q, th):
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** (n - q - 1), 2, 2 ** q)
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        if th.dim() == 1 and th.shape[0] == B:
            c, s = c.view(B, 1, 1), s.view(B, 1, 1)
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], dim=2).reshape(B, -1)

    def _signs(self, n, device):
        idx = torch.arange(2 ** n, device=device)
        bit = lambda q: ((idx >> q) & 1)  # noqa: E731
        cz1 = torch.ones(2 ** n, dtype=qsim.RDTYPE, device=device)
        for q in range(0, n - 1, 2):
            cz1 = cz1 * torch.where((bit(q) & bit(q + 1)).bool(), -1.0, 1.0)
        cz2 = torch.ones(2 ** n, dtype=qsim.RDTYPE, device=device)
        for q in range(1, n - 2, 2):
            cz2 = cz2 * torch.where((bit(q) & bit(q + 1)).bool(), -1.0, 1.0)
        sumz = sum((1.0 - 2.0 * bit(q)).to(qsim.RDTYPE) for q in range(n))
        return cz1.to(qsim.RDTYPE), cz2.to(qsim.RDTYPE), sumz

    def _encode(self, X):
        B, n = X.shape
        psi = torch.zeros(B, 2 ** n, dtype=qsim.RDTYPE, device=X.device)
        psi[:, 0] = 1.0
        for q in range(n):
            psi = self._ry_real(psi, n, q, X[:, q])
        return psi

    def _out(self, theta, X, mask=None, encoded=None):
        n = X.shape[1]
        if (not hasattr(self, "_sg") or self._sg[2].shape[0] != 2 ** n
                or self._sg[2].device != X.device):
            self._sg = self._signs(n, X.device)
        cz1, cz2, sumz = self._sg
        psi = self._encode(X) if encoded is None else encoded
        k = 0
        for layer in range(self.layers):
            for q in range(n):
                th = theta[k] if mask is None else theta[k] * mask[k]
                psi = self._ry_real(psi, n, q, th); k += 1
            psi = psi * cz1
            for q in range(1, n - 1):
                th = theta[k] if mask is None else theta[k] * mask[k]
                psi = self._ry_real(psi, n, q, th); k += 1
            psi = psi * cz2
        return torch.sigmoid((psi ** 2) @ sumz)

    def _nparams(self, n):
        return self.layers * (n + (n - 2))

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mm = MinMaxScaler((0, self.range)).fit(X)
        A = self.mm.transform(X)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        # stratified 80/20 split for the stopping criterion
        va = np.concatenate([idx[y[idx] == c][: max(1, int(0.2 * (y == c).sum()))]
                             for c in (0, 1)])
        tr = np.setdiff1d(idx, va)
        return self._train(A[tr], y[tr], A[va], y[va], rng)

    def fit_with_stopping_set(self, X, y, X_stop, y_stop):
        """Train on all of (X, y) and stop on (X_stop, y_stop) — the paper stops on its
        test set; this exists only so the reproduction arm can do the same."""
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mm = MinMaxScaler((0, self.range)).fit(X)
        A_stop = np.clip(self.mm.transform(np.asarray(X_stop, float)), 0, self.range)
        return self._train(self.mm.transform(X), y, A_stop, np.asarray(y_stop, float),
                           np.random.default_rng(self.seed))

    def _train(self, A_tr, y_tr, A_va, y_va, rng):
        dev = self.device
        A = A_tr
        Xtr, ytr, Xva, yva = (T(A_tr).to(dev), T(y_tr).to(dev),
                              T(A_va).to(dev), T(y_va).to(dev))
        torch.manual_seed(self.seed)
        n = A.shape[1]
        P = self._nparams(n)
        theta = torch.nn.Parameter(0.01 * torch.randn(P, dtype=qsim.RDTYPE).to(dev))
        opt = torch.optim.Adam([theta], lr=self.lr)
        best, best_theta, since, since_lr = float("inf"), theta.detach().clone(), 0, 0
        per_layer = n + (n - 2)
        Etr, Eva = self._encode(Xtr), self._encode(Xva)
        for _ in range(self.max_epochs):
            mask = None
            if self.dropout:
                mask = torch.ones(P, dtype=qsim.RDTYPE, device=dev)
                for l in range(self.layers):
                    if rng.random() < 0.05:
                        drop = torch.as_tensor(rng.random(per_layer) < 0.2, device=dev)
                        mask[l * per_layer:(l + 1) * per_layer][drop] = 0.0
            opt.zero_grad()
            loss = _bce(self._out(theta, Xtr, mask, encoded=Etr), ytr)
            loss.backward()
            opt.step()
            with torch.no_grad():
                v = float(_bce(self._out(theta, Xva, encoded=Eva), yva))
            if v < best - 1e-6:
                best, best_theta, since, since_lr = v, theta.detach().clone(), 0, 0
            else:
                since += 1; since_lr += 1
                if since_lr >= 50:
                    for g in opt.param_groups:
                        g["lr"] *= 0.5
                    since_lr = 0
                if since >= 100:
                    break
        self.theta = best_theta
        return self

    def predict_proba(self, X):
        A = np.clip(self.mm.transform(np.asarray(X, float)), 0, self.range)
        with torch.no_grad():
            return self._out(self.theta, T(A).to(self.device)).cpu().numpy()
