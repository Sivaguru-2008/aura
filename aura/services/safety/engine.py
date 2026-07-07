"""SafetyEngine — assess a case's evidence vector for trustworthiness.

Consumes the fusion model's raw logits (so it can re-scale them) plus the
evidence vector (so it can Monte-Carlo perturb it for epistemic uncertainty).
"""
from __future__ import annotations

import numpy as np

from common.config import get_settings
from common.mathx import energy_score, entropy, softmax
from schemas.clinical import DIAGNOSES, Diagnosis
from schemas.contracts import AbstentionReason, Prediction, SafetyAssessment
from services.safety.calibration import Calibration

MODEL_VERSION = "safety-v1"


class SafetyEngine:
    def __init__(self, calibration: Calibration | None = None):
        self.cal = calibration or Calibration.load()
        self.settings = get_settings()
        self.model_version = MODEL_VERSION

    def calibrated_posterior(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.cal.temperature)

    def _epistemic(self, fusion_model, x: np.ndarray, k: int = 24) -> tuple[float, np.ndarray]:
        """MC estimate: perturb the evidence vector and measure posterior spread.

        Stands in for a deep ensemble — variance of the top-class probability
        under small evidence perturbations is our epistemic uncertainty proxy.
        """
        rng = np.random.default_rng(self.settings.seed)
        posts = []
        for _ in range(k):
            xp = np.clip(x + rng.normal(0.0, 0.05, size=x.shape), 0.0, 1.0)
            posts.append(self.calibrated_posterior(fusion_model.logits(xp)))
        posts = np.array(posts)
        return float(posts.std(axis=0).max()), posts.mean(axis=0)

    def assess(self, study_id: str, x: np.ndarray, fusion_model) -> SafetyAssessment:
        logits = fusion_model.logits(x)
        probs = self.calibrated_posterior(logits)

        epistemic, mc_mean = self._epistemic(fusion_model, x)
        # Aleatoric proxy: normalized entropy of the calibrated posterior.
        aleatoric = float(entropy(probs) / np.log2(len(DIAGNOSES)))

        order = np.argsort(-probs)
        predictions: list[Prediction] = []
        for i in order:
            p = float(probs[i])
            half = float(min(0.5, epistemic + 0.5 * probs[i] * (1 - probs[i])))
            predictions.append(
                Prediction(
                    diagnosis=DIAGNOSES[i],
                    probability=round(p, 4),
                    ci_low=round(max(0.0, p - half), 4),
                    ci_high=round(min(1.0, p + half), 4),
                )
            )

        # Conformal set: keep labels whose nonconformity <= qhat.
        conformal_set = [
            DIAGNOSES[i] for i in range(len(DIAGNOSES))
            if (1.0 - probs[i]) <= self.cal.conformal_qhat
        ]
        if not conformal_set:
            conformal_set = [DIAGNOSES[int(order[0])]]

        # OOD via energy z-score against in-distribution stats.
        e = energy_score(logits, self.cal.temperature)
        z = (e - self.cal.ood_mean) / self.cal.ood_std
        is_ood = bool(z > self.settings.ood_energy_threshold)

        top_i = int(order[0])
        top = DIAGNOSES[top_i]
        top_p = float(probs[top_i])

        reason = AbstentionReason.NONE
        abstained = False
        if is_ood:
            abstained, reason = True, AbstentionReason.OUT_OF_DISTRIBUTION
        elif top_p < self.settings.low_confidence_threshold:
            abstained, reason = True, AbstentionReason.LOW_CONFIDENCE
        elif len(conformal_set) > self.settings.abstention_conformal_size:
            abstained, reason = True, AbstentionReason.LARGE_CONFORMAL_SET
        elif epistemic > 0.20:
            abstained, reason = True, AbstentionReason.HIGH_EPISTEMIC

        return SafetyAssessment(
            study_id=study_id,
            predictions=predictions,
            top=top,
            top_probability=round(top_p, 4),
            conformal_set=conformal_set,
            conformal_coverage=self.cal.coverage,
            epistemic_uncertainty=round(epistemic, 4),
            aleatoric_uncertainty=round(aleatoric, 4),
            ood_energy=round(float(z), 4),
            is_ood=is_ood,
            abstained=abstained,
            abstention_reason=reason,
            model_version=self.model_version,
        )
