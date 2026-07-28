"""Quantum Kernel Learning (QKL) / QSVM for rare tumor subtype classification.

Projects ResU-Net 128-dimensional bottleneck features into a 6-qubit Hilbert space
using an IQP feature map to classify glioma, meningioma, and metastasis.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pennylane as qml

from common.config import ARTIFACTS


class QKLClassifier:
    def __init__(
        self,
        W_proj: np.ndarray | None = None,
        b_proj: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        b_svm: float = 0.0,
        support_vectors: np.ndarray | None = None,
        support_labels: np.ndarray | None = None,
        n_qubits: int = 6,
        n_layers: int = 2,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Projection from 128-d to 6-d
        if W_proj is None:
            self.W_proj = np.zeros((n_qubits, 128), dtype=float)
        else:
            self.W_proj = np.asarray(W_proj, dtype=float)

        if b_proj is None:
            self.b_proj = np.zeros(n_qubits, dtype=float)
        else:
            self.b_proj = np.asarray(b_proj, dtype=float)

        self.alpha = alpha if alpha is not None else np.array([], dtype=float)
        self.b_svm = float(b_svm)
        self.support_vectors = support_vectors if support_vectors is not None else np.zeros((0, n_qubits), dtype=float)
        self.support_labels = support_labels if support_labels is not None else np.array([], dtype=float)

        self._dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self._dev, interface="numpy")
        def iqp_kernel_circuit(x1, x2):
            # Encode first vector x1
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(np.pi * x1[i], wires=i)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                qml.RZ(np.pi * x1[i] * x1[(i + 1) % self.n_qubits], wires=(i + 1) % self.n_qubits)
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # Invert encoding for second vector x2 (adjoint)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                qml.RZ(-np.pi * x2[i] * x2[(i + 1) % self.n_qubits], wires=(i + 1) % self.n_qubits)
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            for i in range(self.n_qubits):
                qml.RZ(-np.pi * x2[i], wires=i)
                qml.Hadamard(wires=i)

            return qml.probs(wires=range(self.n_qubits))

        self._kernel_circuit = iqp_kernel_circuit

    def _project(self, embedding: np.ndarray) -> np.ndarray:
        # Scale to (-1, 1) and normalize to (0, 1)
        z = self.W_proj @ embedding + self.b_proj
        return (np.tanh(z) + 1.0) / 2.0

    def kernel_eval(self, x1: np.ndarray, x2: np.ndarray) -> float:
        probs = self._kernel_circuit(x1, x2)
        return float(probs[0])

    def predict_subtype(self, embedding: np.ndarray) -> dict[str, float]:
        """Predict tumor subtype probability mapping."""
        x = self._project(embedding)

        if len(self.support_vectors) == 0:
            return {"glioma": 0.3333, "meningioma": 0.3333, "metastasis": 0.3333}

        scores = np.zeros(3)
        for i, sv in enumerate(self.support_vectors):
            k_val = self.kernel_eval(sv, x)
            lbl = int(self.support_labels[i])
            scores[lbl] += self.alpha[i] * k_val

        scores += self.b_svm

        # Softmax decision values to yield calibrated probabilities
        from common.mathx import softmax
        probs = softmax(scores)

        return {
            "glioma": float(probs[0]),
            "meningioma": float(probs[1]),
            "metastasis": float(probs[2]),
        }

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS / "brain" / "qkl_classifier.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_proj=self.W_proj,
            b_proj=self.b_proj,
            alpha=self.alpha,
            b_svm=self.b_svm,
            support_vectors=self.support_vectors,
            support_labels=self.support_labels,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> QKLClassifier:
        path = path or (ARTIFACTS / "brain" / "qkl_classifier.npz")
        if not path.exists():
            return cls()
        try:
            d = np.load(path)
            return cls(
                d["W_proj"],
                d["b_proj"],
                d["alpha"],
                float(d["b_svm"]),
                d["support_vectors"],
                d["support_labels"],
                int(d["n_qubits"]),
                int(d["n_layers"]),
            )
        except Exception:
            return cls()
