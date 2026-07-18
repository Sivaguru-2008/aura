"""SafetyEngine — assess a case's evidence vector for trustworthiness.

Consumes the fusion model's raw logits (so it can re-scale them) plus the evidence
vector. Epistemic uncertainty comes from a **deep ensemble** when one is trained
(member disagreement → aleatoric/epistemic split via mutual information); it falls
back to the input-perturbation proxy only when no ensemble is present. Conformal
sets are class-conditional (Mondrian) when per-class thresholds are available,
otherwise marginal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS, get_settings
from common.mathx import energy_score, entropy, softmax
from schemas.clinical import DIAGNOSES, Diagnosis
from schemas.contracts import AbstentionReason, Prediction, SafetyAssessment
from services.fusion.ensemble import DeepEnsemble
from services.safety.calibration import Calibration
from services.safety.uncertainty import ensemble_decomposition, mondrian_set

MODEL_VERSION = "safety-v2"


class SafetyEngine:
    def __init__(self, calibration: Calibration | None = None,
                 ensemble: DeepEnsemble | None = None,
                 mondrian_qhats: np.ndarray | None = None):
        self.cal = calibration or Calibration.load()
        self.settings = get_settings()
        self.ensemble = ensemble if ensemble is not None else DeepEnsemble.load()
        self.mondrian_qhats = (
            mondrian_qhats if mondrian_qhats is not None else self._load_mondrian()
        )
        self.model_version = MODEL_VERSION

    @staticmethod
    def _load_mondrian(path: Path | None = None) -> np.ndarray | None:
        path = path or (ARTIFACTS / "conformal_mondrian.npy")
        if Path(path).exists():
            return np.load(path)
        return None

    def calibrated_posterior(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.cal.temperature)

    # ---- epistemic uncertainty ------------------------------------------- #
    def _epistemic_ensemble(self, x: np.ndarray) -> dict:
        """Real deep-ensemble decomposition (temperature-scaled members)."""
        member_logits = self.ensemble.member_logits(x)
        member_probs = np.array([softmax(r / self.cal.temperature) for r in member_logits])
        dec = ensemble_decomposition(member_probs)
        dec["method"] = "deep_ensemble"
        dec["n"] = self.ensemble.n_members
        return dec

    def _epistemic_perturbation(self, fusion_model, x: np.ndarray, k: int = 24) -> dict:
        """Fallback proxy: perturb the evidence vector, measure posterior spread."""
        rng = np.random.default_rng(self.settings.seed)
        posts = []
        for _ in range(k):
            xp = np.clip(x + rng.normal(0.0, 0.05, size=x.shape), 0.0, 1.0)
            posts.append(self.calibrated_posterior(fusion_model.logits(xp)))
        posts = np.array(posts)
        mean = posts.mean(axis=0)
        return {
            "mean": mean,
            "epistemic_std": float(posts.std(axis=0).max()),
            "epistemic_mi": 0.0,
            "predictive_entropy": entropy(mean),
            "aleatoric_entropy": entropy(mean),
            "method": "input_perturbation",
            "n": 0,
        }

    def assess(self, study_id: str, x: np.ndarray, fusion_model) -> SafetyAssessment:
        logits = fusion_model.logits(x)
        probs = self.calibrated_posterior(logits)
        n_dx = len(DIAGNOSES)
        log2n = float(np.log2(n_dx))

        if self.ensemble is not None:
            dec = self._epistemic_ensemble(x)
        else:
            dec = self._epistemic_perturbation(fusion_model, x)
        epistemic = float(dec["epistemic_std"])
        epistemic_mi = float(dec["epistemic_mi"])
        predictive_entropy = float(dec["predictive_entropy"])
        aleatoric = float(dec["aleatoric_entropy"]) / log2n

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

        # Conformal set: class-conditional (Mondrian) when available, else marginal.
        if self.mondrian_qhats is not None and len(self.mondrian_qhats) == n_dx:
            keep = mondrian_set(probs, self.mondrian_qhats)
            conformal_set = [DIAGNOSES[i] for i in keep]
            conformal_method = "mondrian"
        else:
            conformal_set = [
                DIAGNOSES[i] for i in range(n_dx)
                if (1.0 - probs[i]) <= self.cal.conformal_qhat
            ]
            if not conformal_set:
                conformal_set = [DIAGNOSES[int(order[0])]]
            conformal_method = "marginal"

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
            conformal_method=conformal_method,
            epistemic_uncertainty=round(epistemic, 4),
            aleatoric_uncertainty=round(aleatoric, 4),
            epistemic_mi=round(epistemic_mi, 4),
            predictive_entropy=round(predictive_entropy, 4),
            uncertainty_method=dec["method"],
            n_ensemble=int(dec["n"]),
            ood_energy=round(float(z), 4),
            is_ood=is_ood,
            abstained=abstained,
            abstention_reason=reason,
            model_version=self.model_version,
        )
