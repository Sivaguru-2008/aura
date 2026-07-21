"""Quantum backend abstraction + the variational circuit definition.

Backend-agnostic by design: `default.qubit` (simulator) today; the same
`QuantumDevice` surface accepts `lightning.qubit`, `qiskit.ibmq`, or a Braket
adapter later with no change to the fusion engine. The circuit is defined once
and reused by both training (torch interface) and serving (numpy/analytic).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def make_qnode(n_qubits: int, n_layers: int, device_name: str = "default.qubit",
               shots: int | None = None, interface: str = "numpy"):
    """Build the fusion QNode.

    Encoding: angle-encode each evidence channel as RY(pi * x_i) on its own qubit.
    Ansatz : n_layers of (trainable RY, RZ per qubit) + a ring of CNOTs. The
             entangling ring is what lets the model represent higher-order
             interactions between evidence sources in a 2**n Hilbert space.
    Readout: <Z_i> per qubit -> a classical linear head maps to diagnosis logits.
    """
    dev = qml.device(device_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def circuit(x, theta):
        # x may be a single sample (n_qubits,) or a broadcast batch (batch, n_qubits);
        # PennyLane parameter broadcasting simulates the whole batch in one pass.
        for i in range(n_qubits):
            qml.RY(np.pi * x[..., i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(theta[layer][i][0], wires=i)
                qml.RZ(theta[layer][i][1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def n_params(n_qubits: int, n_layers: int) -> int:
    return n_layers * n_qubits * 2


# --------------------------------------------------------------------------- #
# Data re-uploading ansatz (barren-plateau-aware).
# --------------------------------------------------------------------------- #
def make_reuploading_qnode(n_qubits: int, n_layers: int,
                           device_name: str = "default.qubit",
                           shots: int | None = None, interface: str = "numpy"):
    """Hardware-efficient **data re-uploading** circuit.

    STATUS — EXPERIMENTAL, NOT WIRED: the serving VQC uses ``make_qnode`` (single
    angle-encoding). This re-uploading ansatz has no importer in the running
    pipeline; it is the designed pair for ``projection.JointProjection`` when the
    high-dimensional embedding path is enabled. Retained as an extension point, not
    active code (audit §3.5 / §11.1).

    Difference from ``make_qnode``: the *data* ``x`` is re-encoded at the start of
    every layer rather than only once. Each layer is

        [ RX(π x)  RY(π x)  RZ(π x) ]        <- data re-upload (fixed, not trainable)
        [ RX(θ)    RY(θ)    RZ(θ)   ]        <- trainable rotations, per qubit
        [ ring of CNOTs ]                    <- entangler

    Why this shape mitigates barren plateaus
    ----------------------------------------
    Three levers, all of which the design pins down:

      1. **Small width.** ``x`` is the output of ``JointProjection`` — exactly
         ``n_qubits`` features — so ``n`` never grows with the input dimension.
         Gradient variance for a 2-design scales like ``2**(-n)``; capping ``n``
         is the only lever that attacks the exponent directly.
      2. **Local cost.** Readout is single-qubit ``⟨Z_i⟩`` (a *local* observable).
         Cerezo et al. (2021) show local cost functions on shallow
         (``O(log n)`` depth) circuits have gradient variance vanishing only
         *polynomially*, not exponentially — the practical escape hatch.
      3. **Data re-uploading.** Re-injecting ``x`` each layer raises the circuit's
         expressivity without deepening the trainable block, and empirically keeps
         ``Var[∂θ]`` off the floor (Pérez-Salinas et al. 2020). This is a
         *mitigation*, not a theorem — see ``docs/ARCHITECTURE_REFACTOR.md``.

    Trainable parameters: ``theta`` has shape ``(n_layers, n_qubits, 3)`` for the
    three axis rotations per qubit per layer.
    """
    dev = qml.device(device_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def circuit(x, theta):
        for layer in range(n_layers):
            # (a) data re-upload — fixed encoding of the projected features
            for i in range(n_qubits):
                qml.RX(np.pi * x[..., i], wires=i)
                qml.RY(np.pi * x[..., i], wires=i)
                qml.RZ(np.pi * x[..., i], wires=i)
            # (b) trainable rotations
            for i in range(n_qubits):
                qml.RX(theta[layer][i][0], wires=i)
                qml.RY(theta[layer][i][1], wires=i)
                qml.RZ(theta[layer][i][2], wires=i)
            # (c) entangling ring
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def n_params_reuploading(n_qubits: int, n_layers: int) -> int:
    return n_layers * n_qubits * 3
