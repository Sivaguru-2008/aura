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
               shots: int | None = None, interface: str = "numpy",
               entangler: str = "ring"):
    """Build the fusion QNode.

    Encoding: angle-encode each evidence channel as RY(pi * x_i) on its own qubit.
    Ansatz : n_layers of (trainable RY, RZ per qubit) + an entangling block. The
             entangler is what lets the model represent higher-order interactions
             between evidence sources in a 2**n Hilbert space.
    Readout: <Z_i> per qubit -> a classical linear head maps to diagnosis logits.

    ``entangler`` selects the two-qubit block. All four keep the *same* trainable
    parameter count, so a difference between them is attributable to entangling
    topology alone and never to capacity:

    * ``"ring"`` — CNOT ring (i -> i+1 mod n), the shipped ansatz. ``n`` two-qubit
      gates per layer, depth O(n) as scheduled, and every qubit is coupled.
    * ``"linear"`` — open chain (i -> i+1, no wrap). ``n-1`` gates per layer. On real
      hardware this is the cheapest topology that still connects the whole register,
      because it maps onto a heavy-hex coupling map without a SWAP to close the ring.
    * ``"full"`` — all-to-all, ``n(n-1)/2`` gates per layer. Maximal expressivity per
      layer and the worst transpiled depth on any device that is not fully connected;
      included so the cost of expressivity is measured rather than assumed.
    * ``"none"`` — no two-qubit gates at all. The register stays a product state, so
      every qubit evolves independently and ``<Z_i Z_j> = <Z_i><Z_j>`` exactly. This
      is the **ablation control**: same qubit count, same layer count, same trainable
      parameter count, same encoding, same readout, same optimiser — entanglement is
      the only thing removed. Any performance difference is therefore attributable to
      entanglement and to nothing else, which is the only way to answer "is the
      quantum part doing work?" with a number instead of an opinion.

    A product-state VQC is not a trivial model: it is still a trained non-linear map
    (each qubit performs its own rotation chain and contributes ``<Z_i>``), so the
    control is a fair one rather than a straw man.
    """
    _check_entangler(entangler)
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
            apply_entangler(entangler, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


#: Two-qubit blocks, keyed by name. Each returns the (control, target) pairs for one
#: layer. Kept as data so the design-space sweep can enumerate topologies and report
#: the gate count each one costs, instead of the choice being a constant in a file.
ENTANGLERS = ("ring", "linear", "full", "none")


def _check_entangler(entangler: str) -> None:
    if entangler not in ENTANGLERS:
        raise ValueError(f"unknown entangler {entangler!r}; expected one of {ENTANGLERS}")


def entangler_pairs(entangler: str, n_qubits: int) -> list[tuple[int, int]]:
    """(control, target) pairs for one layer of *entangler* on *n_qubits* wires."""
    _check_entangler(entangler)
    if entangler == "none":
        return []
    if entangler == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entangler == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]      # ring


def apply_entangler(entangler: str, n_qubits: int) -> None:
    """Emit one layer's two-qubit block onto the active PennyLane tape."""
    for control, target in entangler_pairs(entangler, n_qubits):
        qml.CNOT(wires=[control, target])


def two_qubit_gate_count(entangler: str, n_qubits: int, n_layers: int) -> int:
    """Two-qubit gates in the whole circuit — the cost that matters on hardware.

    Single-qubit rotations are cheap and high-fidelity; CNOTs dominate both the
    error budget and the transpiled depth, so this is the number to compare
    topologies on.
    """
    return len(entangler_pairs(entangler, n_qubits)) * n_layers


def n_params(n_qubits: int, n_layers: int) -> int:
    return n_layers * n_qubits * 2


def make_probs_qnode(n_qubits: int, n_layers: int, device_name: str = "default.qubit",
                     shots: int | None = None, interface: str = "numpy",
                     entangler: str = "ring"):
    """Build a QNode returning the joint probability distribution of all 2**n_qubits basis states.

    ``entangler`` must match the circuit the parameters were trained on — see
    :func:`make_qnode`. Measuring a product-trained ``theta`` through the ring ansatz
    evaluates a different unitary and returns a distribution that is well-formed and
    wrong.
    """
    _check_entangler(entangler)
    dev = qml.device(device_name, wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def circuit(x, theta):
        for i in range(n_qubits):
            qml.RY(np.pi * x[..., i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(theta[layer][i][0], wires=i)
                qml.RZ(theta[layer][i][1], wires=i)
            apply_entangler(entangler, n_qubits)
        return qml.probs(wires=range(n_qubits))

    return circuit


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
         *mitigation*, not a theorem — see ``docs/ARCHITECTURE.md``.

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
