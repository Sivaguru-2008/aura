"""Quantum Autoencoder (QAE) feature compressor.

.. admonition:: STATUS — IMPORTED BY THE FUSION ENGINE, BUT UNTRAINED AND INERT

   ``FusionEngine`` calls ``QuantumAutoencoder.load()`` at construction, which
   returns ``None`` because no trained weights ship. ``qae_enabled`` also defaults
   to ``False``. So the compression branch in ``FusionEngine.fuse`` is not taken and
   evidence is built by the plain ``encode()`` path.

   ``FusionResult.qae_applied`` reports whether the branch was **actually** taken —
   deliberately not a copy of the ``qae_enabled`` setting, because the branch also
   requires a vision embedding and loaded weights. Setting ``AURA_QAE_ENABLED=1``
   without training weights first changes nothing except, previously, a console row
   that claimed a quantum autoencoder had run.

Compresses 1024-dimensional classical DenseNet feature maps down to an 8-qubit space
while preserving non-linear covariance.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pennylane as qml

from aura.common.config import ARTIFACTS


class QuantumAutoencoder:
    def __init__(
        self,
        theta: np.ndarray | None = None,
        n_qubits: int = 10,
        n_latent: int = 8,
        n_layers: int = 3,
    ):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers

        if theta is None:
            # 3 rotation parameters (RX, RY, RZ) per qubit per layer
            self.theta = np.zeros((n_layers, n_qubits, 3), dtype=float)
        else:
            self.theta = np.asarray(theta, dtype=float)

        self._dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self._dev, interface="numpy")
        def compression_circuit(features, weights):
            # 1. State preparation using Amplitude Embedding
            qml.AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)

            # 2. Variational Unitary layers U(theta)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer][i][0], wires=i)
                    qml.RY(weights[layer][i][1], wires=i)
                    qml.RZ(weights[layer][i][2], wires=i)
                # Ring of CNOTs
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = compression_circuit

    def compress(self, features: np.ndarray) -> np.ndarray:
        """Compress 1024-dimensional feature vector to 8-dimensional space."""
        # Clean input
        a = np.asarray(features, dtype=float).flatten()
        if len(a) != 1024:
            # Pad or truncate to 1024
            b = np.zeros(1024)
            b[:min(1024, len(a))] = a[:min(1024, len(a))]
            a = b

        norm = np.linalg.norm(a)
        if norm > 1e-9:
            a = a / norm
        else:
            a = np.zeros(1024)
            a[0] = 1.0

        res = self._circuit(a, self.theta)
        # First 8 represent system/latent qubit expectation values.
        # Map expectation values in [-1, 1] to VQC angle inputs in [0, 1].
        latent = np.array(res[:self.n_latent], dtype=float)
        return (latent + 1.0) / 2.0

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS / "qae_params.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=self.theta, n_qubits=self.n_qubits, n_latent=self.n_latent, n_layers=self.n_layers)

    @classmethod
    def load(cls, path: Path | None = None) -> QuantumAutoencoder | None:
        path = path or (ARTIFACTS / "qae_params.npz")
        if not path.exists():
            return None
        try:
            d = np.load(path)
            return cls(d["theta"], int(d["n_qubits"]), int(d["n_latent"]), int(d["n_layers"]))
        except Exception:
            return None
