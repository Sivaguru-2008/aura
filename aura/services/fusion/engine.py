"""FusionEngine — selects the backend and exposes the `fuse()` contract.

Picks quantum or classical from config; falls back to classical automatically if
quantum artifacts are absent, so the demo never hard-fails on a missing model.
"""
from __future__ import annotations

import numpy as np

from common.config import get_settings
from schemas.contracts import FusionResult, StructuredPriors, VisionResult
from services.fusion.classical import ClassicalFusion
from services.fusion.evidence import encode
from services.fusion.learnable import LearnableFusion
from services.fusion.quantum import QuantumFusion


class FusionEngine:
    def __init__(self, backend: str | None = None):
        s = get_settings()
        self.requested = backend or s.fusion_backend
        self.quantum = QuantumFusion.load()
        self.classical = ClassicalFusion.load()
        self.learnable = LearnableFusion.load()
        self.backend = self._resolve()

    def _resolve(self) -> str:
        if self.requested == "quantum" and self.quantum is not None:
            return "quantum"
        if self.requested == "learnable" and self.learnable is not None:
            return "learnable"
        return "classical"

    @property
    def model(self):
        return {
            "quantum": self.quantum,
            "learnable": self.learnable,
            "classical": self.classical,
        }.get(self.backend, self.classical)

    def is_trained(self) -> bool:
        return self.model is not None

    def fuse_vector(self, x: np.ndarray, study_id: str = "") -> FusionResult:
        model = self.model
        if model is None:
            raise RuntimeError("No fusion model trained. Run `aura_cli train` first.")
        posterior, std = model.fuse(x)
        return FusionResult(
            study_id=study_id,
            backend=self.backend,
            posterior=posterior,
            posterior_std=std,
            evidence_vector=[round(float(v), 5) for v in x],
            n_shots=get_settings().n_shots if self.backend == "quantum" else 0,
            model_version=model.model_version,
        )

    def fuse(self, vision: VisionResult, priors: StructuredPriors) -> FusionResult:
        x = encode(vision, priors)
        return self.fuse_vector(x, study_id=vision.study_id)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.model.logits(x)
