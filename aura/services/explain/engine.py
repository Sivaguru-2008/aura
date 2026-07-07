"""ExplainEngine implementation."""
from __future__ import annotations

import numpy as np

from common.mathx import softmax
from schemas.clinical import DIAGNOSES, Diagnosis
from schemas.contracts import Explanation
from services.fusion.evidence import EVIDENCE_CHANNELS
from services.vision.features import _resize_to
from ml.data import IMG


class ExplainEngine:
    def __init__(self, window: int = 12, stride: int = 6):
        self.window = window
        self.stride = stride
        self.model_version = "explain-v1"

    def occlusion_saliency(self, vision_engine, img: np.ndarray) -> np.ndarray:
        """Slide an occluding patch; importance = drop in the top finding's prob.

        Genuinely model-agnostic: only calls the vision engine's public
        `score_findings`, so it stays valid if the vision model is swapped.
        """
        img = _resize_to(img)
        base = vision_engine.score_findings(img)
        top_finding = max(base, key=base.get)
        base_p = base[top_finding]
        baseline_val = 0.18                      # lung baseline intensity
        sal = np.zeros((IMG, IMG), dtype=float)
        counts = np.zeros((IMG, IMG), dtype=float)
        for r in range(0, IMG - 1, self.stride):
            for c in range(0, IMG - 1, self.stride):
                r1, c1 = min(IMG, r + self.window), min(IMG, c + self.window)
                patch = img[r:r1, c:c1].copy()
                img[r:r1, c:c1] = baseline_val
                p = vision_engine.score_findings(img)[top_finding]
                img[r:r1, c:c1] = patch          # restore
                drop = max(0.0, base_p - p)
                sal[r:r1, c:c1] += drop
                counts[r:r1, c:c1] += 1
        counts[counts == 0] = 1
        sal = sal / counts
        m = sal.max()
        return sal / m if m > 1e-9 else sal

    def evidence_attribution(self, fusion_model, x: np.ndarray, top: Diagnosis):
        """Leave-one-out attribution to the top diagnosis probability.

        attribution_i = P(top | x) - P(top | x with channel_i removed).
        Positive = the evidence pushed the diagnosis up. Also serves as the
        counterfactual ("remove this evidence -> prob changes by -attribution").
        """
        top_i = DIAGNOSES.index(top)
        base_p = float(softmax(fusion_model.logits(x))[top_i])
        attribution: dict[str, float] = {}
        counterfactual: dict[str, float] = {}
        for j, name in enumerate(EVIDENCE_CHANNELS):
            xp = x.copy()
            xp[j] = 0.0
            p = float(softmax(fusion_model.logits(xp))[top_i])
            attribution[name] = round(base_p - p, 4)
            counterfactual[name] = round(p - base_p, 4)   # signed change if removed
        return attribution, counterfactual

    def explain(self, study_id: str, vision_engine, img: np.ndarray,
                fusion_model, x: np.ndarray, top: Diagnosis) -> Explanation:
        sal = self.occlusion_saliency(vision_engine, img)
        attribution, counterfactual = self.evidence_attribution(fusion_model, x, top)
        return Explanation(
            study_id=study_id,
            saliency=[round(float(v), 4) for v in sal.flatten()],
            saliency_shape=(IMG, IMG),
            evidence_attribution=attribution,
            counterfactuals=counterfactual,
            method="occlusion+leave-one-out",
        )
