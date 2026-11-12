"""A small batched statevector simulator, for training variational circuits fast.

Why not PennyLane or Qiskit directly: the audit trains each study's circuit inside
25 outer folds x an inner grid, and the published circuits are 3-13 qubits. Per-sample
circuit objects make that hours of Python overhead; a batch of statevectors as one
``(B, 2**n)`` tensor makes it seconds. Correctness is not assumed — ``test_qsim.py``
checks every gate and every feature map here against ``qiskit.quantum_info.Statevector``
on random inputs.

Conventions follow Qiskit: qubit 0 is the least-significant bit of the basis index, and
``P(theta) = diag(1, e^{i theta})``. Parameters may be scalars (shared across the batch)
or ``(B,)`` tensors (data-dependent angles).
"""
from __future__ import annotations

import math

import numpy as np
import torch

CDTYPE = torch.complex128
RDTYPE = torch.float64


class State:
    """A batch of ``n``-qubit statevectors, shape ``(B, 2**n)``."""

    def __init__(self, psi: torch.Tensor, n: int) -> None:
        self.psi = psi
        self.n = n

    @classmethod
    def zeros(cls, batch: int, n: int, device="cpu") -> "State":
        psi = torch.zeros(batch, 2 ** n, dtype=CDTYPE, device=device)
        psi[:, 0] = 1.0
        return cls(psi, n)

    @classmethod
    def amplitude(cls, x: torch.Tensor, n: int) -> "State":
        """Amplitude embedding: pad to ``2**n`` and L2-normalise (Qiskit RawFeatureVector)."""
        x = torch.as_tensor(x, dtype=RDTYPE)
        B, d = x.shape
        if d > 2 ** n:
            raise ValueError(f"{d} features do not fit in {n} qubits")
        v = torch.zeros(B, 2 ** n, dtype=RDTYPE, device=x.device)
        v[:, :d] = x
        norm = v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return cls((v / norm).to(CDTYPE), n)

    # ------------------------------------------------------------------ #
    def _view(self, q: int) -> torch.Tensor:
        # index = high * 2**(q+1) + bit * 2**q + low
        B = self.psi.shape[0]
        return self.psi.reshape(B, 2 ** (self.n - q - 1), 2, 2 ** q)

    def apply1(self, q: int, m: torch.Tensor) -> "State":
        """Apply a single-qubit matrix ``m`` of shape ``(2,2)`` or ``(B,2,2)``."""
        v = self._view(q)
        if m.dim() == 2:
            out = torch.einsum("ab,xhbl->xhal", m, v)
        else:
            out = torch.einsum("xab,xhbl->xhal", m, v)
        self.psi = out.reshape(self.psi.shape)
        return self

    def controlled(self, c: int, t: int, m: torch.Tensor) -> "State":
        """Apply ``m`` to target ``t`` where control ``c`` is 1."""
        B = self.psi.shape[0]
        idx = torch.arange(2 ** self.n, device=self.psi.device)
        on = ((idx >> c) & 1).bool()
        sub = State(self.psi.clone(), self.n).apply1(t, m).psi
        self.psi = torch.where(on.unsqueeze(0), sub, self.psi)
        return self

    # -- named gates ---------------------------------------------------- #
    def h(self, q):
        return self.apply1(q, _const([[1, 1], [1, -1]], self.psi.device) / math.sqrt(2))

    def x(self, q):
        return self.apply1(q, _const([[0, 1], [1, 0]], self.psi.device))

    def rx(self, q, th):
        return self.apply1(q, rx_mat(th, self.psi.device))

    def ry(self, q, th):
        return self.apply1(q, ry_mat(th, self.psi.device))

    def rz(self, q, th):
        return self.apply1(q, rz_mat(th, self.psi.device))

    def p(self, q, th):
        return self.apply1(q, p_mat(th, self.psi.device))

    def u3(self, q, th, ph, lam):
        return self.apply1(q, u3_mat(th, ph, lam, self.psi.device))

    def cx(self, c, t):
        # a permutation: flip bit t wherever bit c is 1
        idx = torch.arange(2 ** self.n, device=self.psi.device)
        perm = torch.where(((idx >> c) & 1).bool(), idx ^ (1 << t), idx)
        self.psi = self.psi[:, perm]
        return self

    def cy(self, c, t):
        return self.controlled(c, t, torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE,
                                                  device=self.psi.device))

    def cz(self, c, t):
        # diagonal: -1 where both bits are 1
        idx = torch.arange(2 ** self.n, device=self.psi.device)
        both = (((idx >> c) & 1) & ((idx >> t) & 1)).bool()
        sign = torch.where(both, -1.0, 1.0).to(CDTYPE)
        self.psi = self.psi * sign
        return self

    # -- readout ------------------------------------------------------- #
    def probs(self) -> torch.Tensor:
        return (self.psi.abs() ** 2).real

    def expval_z(self, q: int) -> torch.Tensor:
        idx = torch.arange(2 ** self.n, device=self.psi.device)
        sign = 1.0 - 2.0 * ((idx >> q) & 1).to(RDTYPE)
        return self.probs() @ sign

    def parity_prob(self) -> torch.Tensor:
        """P(odd parity) — Qiskit ML's default binary interpretation of a bitstring."""
        idx = torch.arange(2 ** self.n, device=self.psi.device)
        odd = torch.tensor([bin(i).count("1") % 2 for i in range(2 ** self.n)],
                           dtype=RDTYPE, device=self.psi.device)
        return self.probs() @ odd


# --------------------------------------------------------------------------- #
def _const(m, device):
    return torch.tensor(m, dtype=CDTYPE, device=device)


def _angle(th, device):
    th = torch.as_tensor(th, dtype=RDTYPE, device=device)
    return th


def _stack(a, b, c, d):
    top = torch.stack([a, b], dim=-1)
    bot = torch.stack([c, d], dim=-1)
    return torch.stack([top, bot], dim=-2)


def rx_mat(th, device="cpu"):
    th = _angle(th, device)
    c, s = torch.cos(th / 2).to(CDTYPE), torch.sin(th / 2).to(CDTYPE)
    return _stack(c, -1j * s, -1j * s, c)


def ry_mat(th, device="cpu"):
    th = _angle(th, device)
    c, s = torch.cos(th / 2).to(CDTYPE), torch.sin(th / 2).to(CDTYPE)
    return _stack(c, -s, s, c)


def rz_mat(th, device="cpu"):
    th = _angle(th, device)
    e = torch.exp(-0.5j * th.to(CDTYPE))
    z = torch.zeros_like(e)
    return _stack(e, z, z, e.conj())


def p_mat(th, device="cpu"):
    th = _angle(th, device)
    one = torch.ones_like(th).to(CDTYPE)
    z = torch.zeros_like(one)
    return _stack(one, z, z, torch.exp(1j * th.to(CDTYPE)))


def u3_mat(th, ph, lam, device="cpu"):
    th, ph, lam = (_angle(v, device) for v in (th, ph, lam))
    c, s = torch.cos(th / 2).to(CDTYPE), torch.sin(th / 2).to(CDTYPE)
    eph, elam = torch.exp(1j * ph.to(CDTYPE)), torch.exp(1j * lam.to(CDTYPE))
    return _stack(c, -elam * s, eph * s, eph * elam * c)


# --------------------------------------------------------------------------- #
# Qiskit circuit-library equivalents
# --------------------------------------------------------------------------- #
def pairs(n: int, entanglement: str):
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "reverse_linear":
        return [(i, i + 1) for i in reversed(range(n - 1))]
    if entanglement == "circular":
        return ([(n - 1, 0)] if n > 2 else []) + [(i, i + 1) for i in range(n - 1)]
    raise ValueError(entanglement)


def zz_feature_map(x: torch.Tensor, reps: int = 2, entanglement: str = "full") -> State:
    """``qiskit.circuit.library.ZZFeatureMap`` (alpha=2, phi(x)=x, phi(x,y)=(pi-x)(pi-y))."""
    x = torch.as_tensor(x, dtype=RDTYPE)
    B, n = x.shape
    s = State.zeros(B, n, x.device)
    for _ in range(reps):
        for q in range(n):
            s.h(q)
        for q in range(n):
            s.p(q, 2.0 * x[:, q])
        for i, j in pairs(n, entanglement):
            s.cx(i, j)
            s.p(j, 2.0 * (math.pi - x[:, i]) * (math.pi - x[:, j]))
            s.cx(i, j)
    return s


def pauli_feature_map_single(x: torch.Tensor, pauli: str, reps: int = 2) -> State:
    """``PauliFeatureMap(paulis=[pauli])`` for a single-qubit Pauli 'X', 'Y' or 'Z'."""
    x = torch.as_tensor(x, dtype=RDTYPE)
    B, n = x.shape
    s = State.zeros(B, n, x.device)
    for _ in range(reps):
        for q in range(n):
            s.h(q)
        for q in range(n):
            if pauli == "X":
                s.h(q)
            elif pauli == "Y":
                s.rx(q, math.pi / 2)
            s.p(q, 2.0 * x[:, q])
            if pauli == "X":
                s.h(q)
            elif pauli == "Y":
                s.rx(q, -math.pi / 2)
    return s


def real_amplitudes(s: State, theta: torch.Tensor, reps: int, entanglement="reverse_linear"):
    """``RealAmplitudes``: RY layer, then (entangle, RY) x reps. ``theta`` has n*(reps+1)."""
    n = s.n
    k = 0
    for q in range(n):
        s.ry(q, theta[k]); k += 1
    for _ in range(reps):
        for i, j in pairs(n, entanglement):
            s.cx(i, j)
        for q in range(n):
            s.ry(q, theta[k]); k += 1
    return s


def fidelity_kernel(sa: State, sb: State) -> np.ndarray:
    """``|<a_i|b_j>|^2`` as a numpy matrix."""
    return (torch.abs(sa.psi.conj() @ sb.psi.T) ** 2).real.cpu().numpy()
