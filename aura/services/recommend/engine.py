"""RecommendEngine — decision-theoretic Expected Value of Information (EVOI).

Why this changed
----------------
The previous ranker scored each test by expected *information gain* (bits) and
divided by cost·risk. Bits are the wrong currency: reducing entropy about a benign
vs. benign distinction is worthless, while a small probability shift on a
pneumothorax can change management. It was also greedy and single-action, blind to
redundancy between tests.

This engine ranks by **EVOI in clinical-loss units** — the expected reduction in
Bayes risk from acquiring the evidence — under a severity-weighted loss where
missing a dangerous diagnosis is expensive. It then builds a **panel** by greedy
forward selection over the joint evidence (so redundant tests aren't double-counted)
subject to a cost budget. EIG in bits is still reported for continuity.

Contract unchanged: ``recommend(fusion_model, x) -> list[Recommendation]``.
"""
from __future__ import annotations

import itertools

import numpy as np

from common.mathx import entropy, softmax
from schemas.clinical import DIAGNOSES, Diagnosis
from schemas.contracts import Recommendation
from services.fusion.evidence import EVIDENCE_CHANNELS

_COST_W = {"low": 1.0, "medium": 2.0, "high": 4.0}
_RISK_W = {"none": 1.0, "low": 1.5, "medium": 2.5}

# Severity weight per diagnosis: the cost of *not* acting on it correctly.
# Missing a pneumothorax or malignancy dominates; normal is cheap to miss.
_SEVERITY: dict[Diagnosis, float] = {
    Diagnosis.PNEUMOTHORAX: 1.00,
    Diagnosis.MALIGNANCY: 0.95,
    Diagnosis.HEART_FAILURE: 0.70,
    Diagnosis.PNEUMONIA: 0.60,
    Diagnosis.COPD: 0.40,
    Diagnosis.NORMAL: 0.20,
}
_SEV = np.array([_SEVERITY[d] for d in DIAGNOSES], dtype=float)

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
    def __init__(self, min_gain: float = 0.005, budget: float = 6.0, max_panel: int = 3):
        self.min_gain = min_gain          # min EVOI (loss units) to bother recommending
        self.budget = budget              # total cost·risk budget for a panel
        self.max_panel = max_panel
        self.model_version = "recommend-evoi-v1"

    # ---- posterior + risk helpers ---------------------------------------- #
    def _posterior(self, fusion_model, x: np.ndarray) -> np.ndarray:
        return softmax(fusion_model.logits(x))

    def _bayes_risk(self, p: np.ndarray) -> float:
        """Min expected severity-weighted loss = E[sev] - max_d p_d·sev_d."""
        return float((p * _SEV).sum() - np.max(p * _SEV))

    def _entropy(self, fusion_model, x: np.ndarray) -> float:
        return entropy(self._posterior(fusion_model, x))

    def _resolvable(self, x: np.ndarray, channels: list[str]) -> list[int]:
        idx = [EVIDENCE_CHANNELS.index(c) for c in channels]
        return [j for j in idx if 0.08 < x[j] < 0.92]

    def _expected_over_outcomes(self, fusion_model, x, idx, fn):
        """E over 2^|idx| resolved outcomes of fn(posterior(resolved x))."""
        total = 0.0
        for outcome in itertools.product([0.0, 1.0], repeat=len(idx)):
            w = 1.0
            xp = x.copy()
            for j, val in zip(idx, outcome):
                p_pos = float(x[j])
                w *= p_pos if val == 1.0 else (1.0 - p_pos)
                xp[j] = val
            if w <= 0:
                continue
            total += w * fn(self._posterior(fusion_model, xp))
        return total

    def _evoi_and_eig(self, fusion_model, x, channels):
        """(EVOI in loss units, EIG in bits) for resolving the given channels jointly."""
        idx = self._resolvable(x, channels)
        if not idx:
            return 0.0, 0.0
        p0 = self._posterior(fusion_model, x)
        r0, h0 = self._bayes_risk(p0), entropy(p0)
        exp_risk = self._expected_over_outcomes(fusion_model, x, idx, self._bayes_risk)
        exp_h = self._expected_over_outcomes(fusion_model, x, idx, entropy)
        return max(0.0, r0 - exp_risk), max(0.0, h0 - exp_h)

    # ---- panel selection -------------------------------------------------- #
    def _greedy_panel(self, fusion_model, x, singles):
        """Forward-select tests maximizing marginal EVOI per cost within budget.

        Joint EVOI over the union of channels handles redundancy: a second test
        that resolves already-resolved evidence adds ~0 and is skipped.
        """
        chosen, used_channels, spent = [], set(), 0.0
        remaining = list(singles)
        while remaining and len(chosen) < self.max_panel:
            best, best_ratio = None, 0.0
            for item in remaining:
                cost = _COST_W[item["cost"]] * _RISK_W[item["risk"]]
                if spent + cost > self.budget:
                    continue
                union = sorted(used_channels | set(item["channels"]))
                joint_evoi, _ = self._evoi_and_eig(fusion_model, x, union)
                marginal = joint_evoi - self._panel_evoi
                ratio = marginal / cost
                if ratio > best_ratio:
                    best, best_ratio, best_marginal = item, ratio, marginal
            if best is None or best_marginal < self.min_gain:
                break
            chosen.append(best)
            used_channels |= set(best["channels"])
            spent += _COST_W[best["cost"]] * _RISK_W[best["risk"]]
            self._panel_evoi += best_marginal
            remaining.remove(best)
        return chosen, sorted(used_channels), spent

    def recommend(self, fusion_model, x: np.ndarray) -> list[Recommendation]:
        x = np.asarray(x, dtype=float)
        h0 = max(1e-6, self._entropy(fusion_model, x))
        recs: list[Recommendation] = []
        singles = []
        for item in CATALOG:
            evoi, eig = self._evoi_and_eig(fusion_model, x, item["channels"])
            if evoi < self.min_gain and eig < 0.02:
                continue
            singles.append(item)
            cost_w, risk_w = _COST_W[item["cost"]], _RISK_W[item["risk"]]
            utility = evoi / (cost_w * risk_w)
            pct = int(round(100 * eig / h0))
            recs.append(
                Recommendation(
                    action=item["action"],
                    display=item["display"],
                    expected_info_gain=round(float(eig), 4),
                    cost_tier=item["cost"],
                    risk_tier=item["risk"],
                    utility=round(float(utility), 4),
                    rationale=(
                        f"{item['why']} Expected value of information "
                        f"{evoi:.3f} (loss units); reduces diagnostic uncertainty ~{pct}%."
                    ),
                )
            )

        # Build the best multi-test panel and surface it if it beats every single.
        self._panel_evoi = 0.0
        panel, channels, spent = self._greedy_panel(fusion_model, x, singles)
        if len(panel) > 1:
            names = ", ".join(p["display"].lower() for p in panel)
            best_single = max((r.utility for r in recs), default=0.0)
            panel_utility = self._panel_evoi / max(spent, 1e-6)
            if panel_utility > best_single:
                recs.insert(0, Recommendation(
                    action="acquire_panel:" + "+".join(p["action"] for p in panel),
                    display=f"Diagnostic panel: {names}",
                    expected_info_gain=round(float(self._evoi_and_eig(fusion_model, x, channels)[1]), 4),
                    cost_tier="high" if spent > 4 else "medium",
                    risk_tier="low",
                    utility=round(float(panel_utility), 4),
                    rationale=(
                        f"Jointly-optimal panel (greedy EVOI, budget {self.budget:.0f}): "
                        f"total EVOI {self._panel_evoi:.3f} loss units at cost {spent:.1f}, "
                        f"chosen to avoid redundant tests."
                    ),
                ))
                return recs

        recs.sort(key=lambda r: -r.utility)
        return recs
