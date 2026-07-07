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
