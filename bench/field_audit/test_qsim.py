"""Every circuit in qsim.py against Qiskit's own Statevector, on random inputs."""
import math

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import Statevector

from bench.field_audit import qsim

RNG = np.random.default_rng(0)


def _close(ours: torch.Tensor, circ: QuantumCircuit):
    ref = Statevector(circ).data
    got = ours.detach().cpu().numpy()
    # global phase is physical-irrelevant; compare up to it
    k = np.argmax(np.abs(ref))
    phase = got[k] / ref[k]
    assert abs(abs(phase) - 1) < 1e-9
    np.testing.assert_allclose(got, ref * phase, atol=1e-9)


@pytest.mark.parametrize("gate", ["h", "x", "rx", "ry", "rz", "p", "u3", "cx", "cy", "cz"])
def test_gates(gate):
    n = 3
    th = RNG.uniform(-3, 3, size=3)
    pre = RNG.uniform(-3, 3, size=n)
    qc = QuantumCircuit(n)
    s = qsim.State.zeros(1, n)
    for q in range(n):                     # a non-trivial input state
        qc.ry(pre[q], q); s.ry(q, pre[q])
    qc.cx(0, 2); s.cx(0, 2)
    if gate in ("h", "x"):
        getattr(qc, gate)(1); getattr(s, gate)(1)
    elif gate in ("rx", "ry", "rz", "p"):
        getattr(qc, gate)(th[0], 1); getattr(s, gate)(1, th[0])
    elif gate == "u3":
        qc.u(th[0], th[1], th[2], 1); s.u3(1, *th)
    else:
        getattr(qc, gate)(2, 1); getattr(s, gate)(2, 1)
    _close(s.psi[0], qc)


@pytest.mark.parametrize("ent", ["full", "linear"])
def test_zz_feature_map(ent):
    n, reps = 4, 2
    x = RNG.uniform(0, math.pi, size=(3, n))
    s = qsim.zz_feature_map(torch.tensor(x), reps=reps, entanglement=ent)
    fm = ZZFeatureMap(n, reps=reps, entanglement=ent)
    for b in range(3):
        _close(s.psi[b], fm.assign_parameters(x[b]))


@pytest.mark.parametrize("pauli", ["X", "Z", "Y"])
def test_pauli_feature_map(pauli):
    n = 3
    x = RNG.uniform(0, math.pi, size=(2, n))
    s = qsim.pauli_feature_map_single(torch.tensor(x), pauli, reps=2)
    fm = PauliFeatureMap(n, reps=2, paulis=[pauli])
    for b in range(2):
        _close(s.psi[b], fm.assign_parameters(x[b]))


def test_real_amplitudes_full():
    n, reps = 4, 3
    th = RNG.uniform(-3, 3, size=n * (reps + 1))
    s = qsim.real_amplitudes(qsim.State.zeros(1, n), torch.tensor(th), reps, "full")
    ra = RealAmplitudes(n, reps=reps, entanglement="full")
    _close(s.psi[0], ra.assign_parameters(th))


def test_amplitude_embedding():
    x = RNG.uniform(0, 1, size=(2, 13))
    s = qsim.State.amplitude(torch.tensor(x), 4)
    for b in range(2):
        v = np.zeros(16); v[:13] = x[b]; v /= np.linalg.norm(v)
        qc = QuantumCircuit(4); qc.initialize(v, range(4))
        _close(s.psi[b], qc)


def test_parity_and_expval():
    th = RNG.uniform(-3, 3, size=3)
    s = qsim.State.zeros(1, 3)
    for q in range(3):
        s.ry(q, th[q])
    p1 = [math.sin(t / 2) ** 2 for t in th]
    odd = (p1[0] * (1 - p1[1]) * (1 - p1[2]) + (1 - p1[0]) * p1[1] * (1 - p1[2])
           + (1 - p1[0]) * (1 - p1[1]) * p1[2] + p1[0] * p1[1] * p1[2])
    assert abs(float(s.parity_prob()[0]) - odd) < 1e-12
    assert abs(float(s.expval_z(1)[0]) - math.cos(th[1])) < 1e-12


def test_heart_qnn_matches_qsim():
    """The real-arithmetic HeartQNN circuit equals the generic simulator's."""
    from bench.field_audit.qmodels import HeartQNN
    n, layers = 5, 2
    X = torch.tensor(RNG.uniform(0, 2 * math.pi, size=(3, n)))
    m = HeartQNN(layers=layers)
    theta = torch.tensor(RNG.uniform(-2, 2, size=m._nparams(n)))
    fast = m._out(theta, X)
    s = qsim.State.zeros(3, n)
    for q in range(n):
        s.ry(q, X[:, q])
    k = 0
    for _ in range(layers):
        for q in range(n):
            s.ry(q, theta[k]); k += 1
        for q in range(0, n - 1, 2):
            s.cz(q, q + 1)
        for q in range(1, n - 1):
            s.ry(q, theta[k]); k += 1
        for q in range(1, n - 2, 2):
            s.cz(q, q + 1)
    ref = torch.sigmoid(sum(s.expval_z(q) for q in range(n)))
    np.testing.assert_allclose(fast.numpy(), ref.numpy(), atol=1e-12)
