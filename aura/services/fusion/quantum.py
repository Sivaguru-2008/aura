"""Quantum fusion backend — serves the trained VQC and reports shot-noise uncertainty.

Serving path uses analytic expectations (fast, deterministic) for the posterior,
and propagates *finite-shot* measurement variance through the readout to produce
an honest fusion-level uncertainty — the natural "uncertainty for free" that
measurement statistics give you on real hardware.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS, get_settings
from common.mathx import softmax
from schemas.clinical import DIAGNOSES, Diagnosis
from services.fusion.device import make_qnode

MODEL_VERSION = "fusion-vqc-v1"


class QuantumFusion:
    def __init__(self, theta: np.ndarray, W: np.ndarray, b: np.ndarray,
                 n_qubits: int, n_layers: int):
        self.theta = np.asarray(theta, dtype=float)
        self.W = np.asarray(W, dtype=float)      # (n_dx, n_qubits)
        self.b = np.asarray(b, dtype=float)      # (n_dx,)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = "quantum"
        self.model_version = MODEL_VERSION
        self._circuit = make_qnode(n_qubits, n_layers, interface="numpy")

    @classmethod
    def load(cls, path: Path | None = None) -> "QuantumFusion | None":
        path = path or (ARTIFACTS / "fusion_quantum.npz")
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["theta"], d["W"], d["b"], int(d["n_qubits"]), int(d["n_layers"]))

    def _expectations(self, x: np.ndarray) -> np.ndarray:
        z = self._circuit(x, self.theta)
        return np.array([float(v) for v in z], dtype=float)

    def fuse(self, x: np.ndarray, n_shots: int | None = None):
        """Return (posterior, posterior_std) as dicts over Diagnosis."""
        n_shots = n_shots or get_settings().n_shots
        z = self._expectations(x)
        logits = self.W @ z + self.b
        posterior = softmax(logits)

        # Finite-shot variance of each <Z_i>: Var = (1 - <Z>^2) / n_shots.
        var_z = np.clip((1.0 - z**2) / max(n_shots, 1), 0.0, None)
        std_z = np.sqrt(var_z)
        # Monte-Carlo propagate through linear + softmax.
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(128):
            zz = z + rng.normal(0.0, std_z)
            samples.append(softmax(self.W @ zz + self.b))
        samples = np.array(samples)
        post_std = samples.std(axis=0)

        posterior_d = {d: float(posterior[i]) for i, d in enumerate(DIAGNOSES)}
        std_d = {d: float(post_std[i]) for i, d in enumerate(DIAGNOSES)}
        return posterior_d, std_d

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Raw logits — used by the safety engine for OOD energy scoring."""
        return self.W @ self._expectations(x) + self.b
