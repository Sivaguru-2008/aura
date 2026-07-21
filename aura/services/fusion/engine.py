"""FusionEngine — selects the backend and exposes the `fuse()` contract.

Picks quantum or classical from config; falls back to classical automatically if
quantum artifacts are absent, so the demo never hard-fails on a missing model.
"""
from __future__ import annotations

import numpy as np

from common.config import get_settings
from common.mathx import softmax
from schemas.clinical import DIAGNOSES
from schemas.contracts import FusionResult, StructuredPriors, VisionResult
from services.fusion.classical import ClassicalFusion
from services.fusion.conflict import WassersteinTieBreaker
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
        # Conflict guard: only meaningful when the quantum backend is live *and*
        # the classical PoE is available as a fallback to defer to.
        self._guard_enabled = bool(
            getattr(s, "fusion_conflict_guard", True)
            and self.backend == "quantum"
            and self.classical is not None
        )
        self._tie_breaker = WassersteinTieBreaker(
            tau_base=getattr(s, "fusion_conflict_tau", 0.12)
        )

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

        # Wasserstein tie-breaker: when the VQC and the Bayesian PoE disagree by
        # more than the dynamic threshold on the clinical-severity axis, defer to
        # the interpretable classical estimator and flag high epistemic risk.
        resolved_backend = self.backend
        conflict_distance = conflict_threshold = 0.0
        fallback_triggered = False
        if self._guard_enabled:
            p_vqc = np.array([posterior[d] for d in DIAGNOSES], dtype=float)
            p_poe = softmax(self.classical.logits(x))
            res = self._tie_breaker.resolve(p_vqc, p_poe)
            resolved_backend = res["resolved_backend"]
            conflict_distance = res["distance"]
            conflict_threshold = res["threshold"]
            fallback_triggered = res["high_epistemic"]
            if fallback_triggered:
                posterior = {d: float(p_poe[i]) for i, d in enumerate(DIAGNOSES)}
                std = {d: 0.0 for d in DIAGNOSES}      # deterministic fallback

        return FusionResult(
            study_id=study_id,
            backend=self.backend,
            posterior=posterior,
            posterior_std=std,
            evidence_vector=[round(float(v), 5) for v in x],
            n_shots=get_settings().n_shots if self.backend == "quantum" else 0,
            model_version=model.model_version,
            resolved_backend=resolved_backend,
            conflict_distance=conflict_distance,
            conflict_threshold=conflict_threshold,
            fallback_triggered=fallback_triggered,
        )

    def fuse(self, vision: VisionResult, priors: StructuredPriors) -> FusionResult:
        x = encode(vision, priors)
        return self.fuse_vector(x, study_id=vision.study_id)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.model.logits(x)

    def resolved_logits(self, x: np.ndarray, result: FusionResult) -> np.ndarray:
        """Logits of the backend actually trusted after the conflict guard.

        When the Wasserstein tie-breaker (Module 5) fell back to the classical
        product-of-experts, its logits are returned so the safety engine calibrates
        and reports the *validated* posterior rather than the discarded VQC one
        (audit F2). Otherwise the default backend's logits are returned unchanged.
        """
        if result.fallback_triggered and self.classical is not None:
            return self.classical.logits(x)
        return self.model.logits(x)
