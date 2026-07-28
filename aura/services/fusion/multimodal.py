"""Quantum Multi-modal Fusion (QMMF).

Coherently fuses chest features (evidence vectors) and brain features (ResU-Net features)
in a joint 8-qubit variational circuit to estimate patient systemic conditions.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pennylane as qml

from common.config import ARTIFACTS


class UnifiedFusionEngine:
    def __init__(
        self,
        theta: np.ndarray | None = None,
        n_qubits: int = 8,
        n_layers: int = 3,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if theta is None:
            self.theta = np.zeros((n_layers, n_qubits, 3), dtype=float)
        else:
            self.theta = np.asarray(theta, dtype=float)

        self._dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self._dev, interface="numpy")
        def joint_fusion_circuit(cxr_feats, brain_feats, weights):
            # Encode chest features on first 4 qubits
            for i in range(4):
                qml.RY(np.pi * cxr_feats[i], wires=i)
            # Encode brain features on remaining 4 qubits
            for i in range(4):
                qml.RY(np.pi * brain_feats[i], wires=4 + i)

            # Apply variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer][i][0], wires=i)
                    qml.RY(weights[layer][i][1], wires=i)
                    qml.RZ(weights[layer][i][2], wires=i)
                # Entangling CNOT ring
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = joint_fusion_circuit

    def fuse_multimodal(self, cxr_vector: np.ndarray, brain_vector: np.ndarray) -> np.ndarray:
        """Fuse multimodal vectors into a 12-class joint posterior."""
        cxr_4d = cxr_vector[:4]
        if len(cxr_4d) < 4:
            cxr_4d = np.pad(cxr_4d, (0, 4 - len(cxr_4d)))

        brain_4d = np.zeros(4)
        if len(brain_vector) >= 128:
            brain_4d[0] = float(np.mean(brain_vector[:32]))
            brain_4d[1] = float(np.mean(brain_vector[32:64]))
            brain_4d[2] = float(np.mean(brain_vector[64:96]))
            brain_4d[3] = float(np.mean(brain_vector[96:]))
        else:
            brain_4d[:min(4, len(brain_vector))] = brain_vector[:min(4, len(brain_vector))]

        res = self._circuit(cxr_4d, brain_4d, self.theta)

        joint_logits = np.zeros(12)
        for i in range(8):
            joint_logits[i] = res[i]
        joint_logits[8:] = 0.1 * np.array(res[:4])

        from common.mathx import softmax
        return softmax(joint_logits)

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS / "joint_qmmf_params.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=self.theta, n_qubits=self.n_qubits, n_layers=self.n_layers)

    @classmethod
    def load(cls, path: Path | None = None) -> UnifiedFusionEngine:
        path = path or (ARTIFACTS / "joint_qmmf_params.npz")
        if not path.exists():
            return cls()
        try:
            d = np.load(path)
            return cls(d["theta"], int(d["n_qubits"]), int(d["n_layers"]))
        except Exception:
            return cls()
