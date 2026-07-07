"""RecommendEngine — expected-information-gain ranking of next diagnostic steps.

EIG(a) = H(D | E) - E_outcomes[ H(D | E U a) ]

We model each candidate acquisition as *sharpening* one or more evidence channels
from their current soft value to a crisp 0/1, with the positive outcome weighted
by the channel's current probability. Expected posterior entropy is computed by
running the fusion model over each possible resolved evidence vector.
"""
from __future__ import annotations

import itertools

import numpy as np

from common.mathx import entropy, softmax
from schemas.contracts import Recommendation
from services.fusion.evidence import EVIDENCE_CHANNELS

_COST_W = {"low": 1.0, "medium": 2.0, "high": 4.0}
_RISK_W = {"none": 1.0, "low": 1.5, "medium": 2.5}

# Catalog of acquisitions -> which evidence channels each would resolve.
CATALOG = [
    dict(action="acquire_lateral_view", display="Acquire lateral chest view",
         channels=["effusion", "opacity"], cost="low", risk="none",
         why="A lateral projection disambiguates airspace opacity from pleural fluid."),
    dict(action="order_ct_chest", display="Order CT chest (low-dose)",
         channels=["nodule", "consolidation", "opacity"], cost="high", risk="low",
         why="CT characterizes nodules and consolidation the frontal film leaves ambiguous."),
    dict(action="retrieve_prior_films", display="Retrieve & compare prior films",
         channels=["nodule", "opacity"], cost="low", risk="none",
         why="Temporal stability changes malignancy likelihood markedly."),
    dict(action="order_bnp_echo", display="Order BNP + bedside echo",
         channels=["cardiomegaly"], cost="medium", risk="none",
         why="Confirms or excludes cardiac cause of an enlarged silhouette."),
    dict(action="sputum_culture", display="Send sputum culture + inflammatory markers",
         channels=["consolidation"], cost="low", risk="none",
         why="Microbiology raises or lowers the pneumonia posterior."),
]


class RecommendEngine:
    def __init__(self, min_gain: float = 0.02):
        self.min_gain = min_gain
        self.model_version = "recommend-eig-v1"

    def _entropy(self, fusion_model, x: np.ndarray) -> float:
        return entropy(softmax(fusion_model.logits(x)))

    def _eig(self, fusion_model, x: np.ndarray, channels: list[str]) -> float:
        h0 = self._entropy(fusion_model, x)
        idx = [EVIDENCE_CHANNELS.index(c) for c in channels]
        # Only channels with residual uncertainty are worth resolving.
        idx = [j for j in idx if 0.08 < x[j] < 0.92]
        if not idx:
            return 0.0
        expected_h = 0.0
        for outcome in itertools.product([0.0, 1.0], repeat=len(idx)):
            w = 1.0
            xp = x.copy()
            for j, val in zip(idx, outcome):
                p_pos = float(x[j])
                w *= p_pos if val == 1.0 else (1.0 - p_pos)
                xp[j] = val
            if w <= 0:
                continue
            expected_h += w * self._entropy(fusion_model, xp)
        return max(0.0, h0 - expected_h)

    def recommend(self, fusion_model, x: np.ndarray) -> list[Recommendation]:
        recs: list[Recommendation] = []
        for item in CATALOG:
            eig = self._eig(fusion_model, x, item["channels"])
            if eig < self.min_gain:
                continue
            cost_w, risk_w = _COST_W[item["cost"]], _RISK_W[item["risk"]]
            utility = eig / (cost_w * risk_w)
            pct = int(round(100 * eig / max(1e-6, self._entropy(fusion_model, x))))
            recs.append(
                Recommendation(
                    action=item["action"],
                    display=item["display"],
                    expected_info_gain=round(float(eig), 4),
                    cost_tier=item["cost"],
                    risk_tier=item["risk"],
                    utility=round(float(utility), 4),
                    rationale=f"{item['why']} Expected to reduce diagnostic "
                              f"uncertainty by ~{pct}%.",
                )
            )
        recs.sort(key=lambda r: -r.utility)
        return recs
