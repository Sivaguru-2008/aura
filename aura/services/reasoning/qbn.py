"""Quantum Bayesian Network (QBN) for clinical reasoning.

Models diagnostic joint probabilities and guideline conditions natively
using complex probability amplitudes and quantum entanglement.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pennylane as qml

from aura.common.config import ARTIFACTS
from aura.schemas.clinical import DIAGNOSES, Diagnosis


class QuantumBayesianNetwork:
    def __init__(self, theta: np.ndarray | None = None, n_qubits: int = 6):
        self.n_qubits = n_qubits

        # Variational conditional transition parameters
        if theta is None:
            # 6 parameters representing guideline likelihood rotation strengths
            self.theta = np.array([1.2, 1.5, 1.8, 1.2, 0.9, 1.3], dtype=float)
        else:
            self.theta = np.asarray(theta, dtype=float)

        self._dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self._dev, interface="numpy")
        def reasoning_circuit(prior_probs, features, weights):
            # 1. State preparation of diagnostic state on first 3 qubits (representing 6 classes)
            # Normalize and pad prior probabilities to 8-dimensional state vector
            state = np.zeros(8)
            state[:6] = prior_probs
            norm = np.linalg.norm(state)
            if norm > 1e-9:
                state = state / norm
            else:
                state = np.zeros(8)
                state[0] = 1.0

            qml.StatePrep(state, wires=[0, 1, 2])

            # 2. Encode clinical features on qubits 3, 4, 5
            # features = [cardiac_evidence, infectious_evidence, malignancy_obstructive_evidence]
            for i in range(3):
                qml.RY(np.pi * features[i], wires=3 + i)

            # 3. Apply conditional quantum Bayesian updates (entanglement)
            qml.ControlledPhaseShift(weights[0], wires=[3, 2])
            qml.CRZ(weights[1], wires=[3, 1])

            # If infectious evidence (qubit 4) is 1, rotate diagnostic qubits towards Pneumonia (|010>)
            qml.ControlledPhaseShift(weights[2], wires=[4, 1])
            qml.CRX(weights[3], wires=[4, 0])

            # If malignancy/obstructive evidence (qubit 5) is 1, rotate towards Malignancy (|100>) and COPD (|001>)
            qml.ControlledPhaseShift(weights[4], wires=[5, 0])
            qml.CRY(weights[5], wires=[5, 2])

            # Return joint probability distribution of the diagnostic qubits
            return qml.probs(wires=[0, 1, 2])

        self._circuit = reasoning_circuit

    def reason(
        self,
        prior_posterior: dict[Diagnosis, float],
        features: list[float],
    ) -> np.ndarray:
        """Update diagnostic posterior using the QBN.

        features: [cardiac_evidence, infectious_evidence, malignancy_obstructive_evidence]
        """
        p0 = np.array([max(1e-9, prior_posterior.get(d, 0.0)) for d in DIAGNOSES])
        p0 = p0 / p0.sum()

        res_probs = self._circuit(p0, features, self.theta)

        # Slice the first 6 elements as our 6 diagnoses and normalize
        adjusted = np.array(res_probs[:6], dtype=float)
        norm = adjusted.sum()
        if norm > 1e-9:
            adjusted = adjusted / norm
        else:
            adjusted = p0

        return adjusted

    def save(self, path: Path | None = None) -> None:
        path = path or (ARTIFACTS / "reasoning_qbn.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=self.theta)

    @classmethod
    def load(cls, path: Path | None = None) -> QuantumBayesianNetwork:
        path = path or (ARTIFACTS / "reasoning_qbn.npz")
        if not path.exists():
            return cls()
        try:
            d = np.load(path)
            return cls(d["theta"])
        except Exception:
            return cls()
