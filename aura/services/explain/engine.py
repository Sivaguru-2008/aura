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

    def _top_finding(self, vision_engine, img: np.ndarray):
        base = vision_engine.score_findings(img)
        return max(base, key=base.get)

    def visual_explanations(self, vision_engine, img: np.ndarray):
        """Return (primary_map, {method: map}, target_finding, method_label).

        With a CNN backbone: the full gradient suite (Grad-CAM, Grad-CAM++,
        Integrated Gradients, SmoothGrad) plus occlusion as a cross-check, with
        Grad-CAM++ as the primary overlay. Without one: occlusion only.
        """
        top_finding = self._top_finding(vision_engine, img)
        backbone = getattr(vision_engine, "backbone", None)
        if backbone is not None:
            from services.explain import methods as M

            # Gradient methods only in the live path (each is 1–30 passes); the
            # 100+-pass occlusion map stays available via methods.occlusion for
            # the model-agnostic cross-check / the feature model.
            maps = M.all_methods(backbone, img, top_finding, out_size=IMG)
            primary = maps.get("grad_cam++", maps.get("grad_cam"))
            if primary is None:
                primary = self.occlusion_saliency(vision_engine, img)
                maps["occlusion"] = primary
            label = "grad_cam++"
            return primary, maps, top_finding, label
        sal = self.occlusion_saliency(vision_engine, img)
        return sal, {"occlusion": sal}, top_finding, "occlusion"

    def explain(self, study_id: str, vision_engine, img: np.ndarray,
                fusion_model, x: np.ndarray, top: Diagnosis) -> Explanation:
        primary, maps, target, label = self.visual_explanations(vision_engine, img)
        attribution, counterfactual = self.evidence_attribution(fusion_model, x, top)
        return Explanation(
            study_id=study_id,
            saliency=[round(float(v), 4) for v in np.asarray(primary).flatten()],
            saliency_shape=(IMG, IMG),
            saliency_methods={
                k: [round(float(v), 4) for v in np.asarray(m).flatten()]
                for k, m in maps.items()
            },
            saliency_target=target.value,
            evidence_attribution=attribution,
            counterfactuals=counterfactual,
            method=f"{label}+leave-one-out",
        )
