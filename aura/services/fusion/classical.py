"""Classical fusion backend — Bayesian product-of-experts (log-linear).

Ships in the same service as the quantum backend and is the honest baseline the
benchmark compares against. Each evidence channel is an independent "expert"
contributing a log-likelihood-ratio per diagnosis; posteriors combine as a
weighted product (sum in log space) — the standard naive-Bayes/PoE fusion that
by construction cannot represent higher-order evidence interactions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS
from common.mathx import softmax
from schemas.clinical import DIAGNOSES, Diagnosis

MODEL_VERSION = "fusion-poe-v1"


class ClassicalFusion:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = np.asarray(W, dtype=float)      # (n_dx, n_channels)
        self.b = np.asarray(b, dtype=float)      # (n_dx,) log-prior
        self.backend = "classical"
        self.model_version = MODEL_VERSION

    @classmethod
    def load(cls, path: Path | None = None) -> "ClassicalFusion | None":
        path = path or (ARTIFACTS / "fusion_classical.npz")
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["W"], d["b"])

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(x, dtype=float) + self.b

    def fuse(self, x: np.ndarray, n_shots: int | None = None):
        logits = self.logits(x)
        posterior = softmax(logits)
        posterior_d = {d: float(posterior[i]) for i, d in enumerate(DIAGNOSES)}
        std_d = {d: 0.0 for d in DIAGNOSES}      # deterministic backend
        return posterior_d, std_d
