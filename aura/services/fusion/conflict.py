"""Conflict-resolution engine: Wasserstein tie-breaker between VQC and PoE.

Why this exists
---------------
The VQC (``QuantumFusion``) and the classical product-of-experts
(``ClassicalFusion``) are two independent estimators of the *same* diagnosis
posterior. When they agree, either is fine. When they **disagree sharply**, that
disagreement is itself a signal: the VQC has likely wandered — an optimisation
artefact (a residual barren-plateau region), an out-of-training-manifold input,
or genuine higher-order structure the PoE cannot represent. In a clinical setting
we do not gamble on which; we fall back to the interpretable, monotone,
provably-well-behaved Bayesian estimator and flag the case.

The right way to measure "sharply disagree" over a *categorical* posterior is not
KL or total-variation — those ignore how clinically far apart the mass is. Moving
probability from ``pneumonia`` to ``pneumothorax_dx`` is dangerous; moving it from
``copd`` to ``normal`` is not. We therefore embed diagnoses on a **clinical
severity axis** and use the **1-Wasserstein distance** (Earth Mover's Distance):
the minimum probability-mass × severity-distance needed to turn one posterior into
the other. EMD with a severity ground metric = "how much dangerous disagreement".

  W₁(p, q) = min over couplings γ of  Σ_{i,j} γ_{ij} · |s_i − s_j|
           = ∫ |F_p(t) − F_q(t)| dt        (the 1-D closed form we use)

where ``s_i`` is the severity coordinate of diagnosis ``i`` and ``F`` is the CDF
along that axis. If ``W₁ > τ_t`` (a *dynamic* threshold, below) we return the
classical posterior and raise ``high_epistemic``.

Pure numpy + scipy (both already dependencies).
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import wasserstein_distance

from schemas.clinical import DIAGNOSES, Diagnosis

# Clinical severity coordinate per diagnosis — the ground metric for the EMD.
# Same ordering intent as recommend.engine._SEVERITY: the cost of being wrong
# about this label. Spacing encodes *how far apart* two labels are clinically.
_SEVERITY: dict[Diagnosis, float] = {
    Diagnosis.NORMAL: 0.00,
    Diagnosis.COPD: 0.40,
    Diagnosis.PNEUMONIA: 0.60,
    Diagnosis.HEART_FAILURE: 0.70,
    Diagnosis.MALIGNANCY: 0.95,
    Diagnosis.PNEUMOTHORAX: 1.00,
}
SEVERITY_SUPPORT = np.array([_SEVERITY[d] for d in DIAGNOSES], dtype=float)


class WassersteinTieBreaker:
    """Resolve VQC-vs-PoE disagreement by Earth Mover's Distance on the severity axis.

    ``tau_base``     : static floor for the divergence threshold.
    ``dynamic``      : when True, the effective threshold adapts to the recent
                       distribution of observed distances (EWMA mean + k·std), so a
                       clinic experiencing covariate shift auto-tightens/loosens
                       instead of trusting a hand-set constant.
    ``window``       : rolling history length for the dynamic threshold.
    ``k``            : how many std above the running mean counts as "a conflict".
    """

    def __init__(self, tau_base: float = 0.12, dynamic: bool = True,
                 window: int = 128, k: float = 3.0,
                 support: np.ndarray | None = None):
        self.tau_base = float(tau_base)
        self.dynamic = bool(dynamic)
        self.k = float(k)
        self.support = SEVERITY_SUPPORT if support is None else np.asarray(support, dtype=float)
        self._hist: deque[float] = deque(maxlen=window)

    # ---- distance -------------------------------------------------------- #
    def distance(self, p_vqc: np.ndarray, p_classical: np.ndarray) -> float:
        """1-Wasserstein distance between the two posteriors on the severity axis."""
        p = _normalize(p_vqc)
        q = _normalize(p_classical)
        return float(wasserstein_distance(self.support, self.support, p, q))

    def distance_cost_matrix(self, p_vqc: np.ndarray, p_classical: np.ndarray,
                             cost: np.ndarray) -> float:
        """General EMD for an arbitrary ground-metric matrix ``cost`` (K×K).

        Solves the exact transportation LP with scipy. Use when a full clinical
        dissimilarity matrix (not a 1-D severity embedding) is available. Falls
        back to the closed-form 1-D distance if the LP is unavailable.
        """
        from scipy.optimize import linprog

        p = _normalize(p_vqc)
        q = _normalize(p_classical)
        K = len(p)
        c = np.asarray(cost, dtype=float).reshape(-1)          # (K*K,)
        # Marginals: row sums = p, col sums = q.
        A_eq_rows = np.zeros((K, K * K))
        A_eq_cols = np.zeros((K, K * K))
        for i in range(K):
            A_eq_rows[i, i * K:(i + 1) * K] = 1.0
            A_eq_cols[i, i::K] = 1.0
        A_eq = np.vstack([A_eq_rows, A_eq_cols])
        b_eq = np.concatenate([p, q])
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if not res.success:
            return self.distance(p, q)
        return float(res.fun)

    # ---- dynamic threshold ----------------------------------------------- #
    def threshold(self) -> float:
        """Current effective threshold τ_t (static floor, raised by recent spread)."""
        if not self.dynamic or len(self._hist) < 8:
            return self.tau_base
        arr = np.fromiter(self._hist, dtype=float)
        return float(max(self.tau_base, arr.mean() + self.k * arr.std()))

    # ---- resolution ------------------------------------------------------ #
    def resolve(self, p_vqc: np.ndarray, p_classical: np.ndarray) -> dict:
        """Decide which posterior to trust.

        Returns a dict:
          posterior        : the chosen posterior (numpy, sums to 1)
          resolved_backend : "quantum" | "classical"
          distance         : the EMD between the two
          threshold        : τ_t used for the decision
          high_epistemic   : True when the fallback fired (flag the case)
          conflict         : alias of high_epistemic (readability at call sites)
        """
        d = self.distance(p_vqc, p_classical)
        tau = self.threshold()
        self._hist.append(d)                       # record *before* deciding, EWMA-style
        conflict = d > tau
        chosen = _normalize(p_classical if conflict else p_vqc)
        return {
            "posterior": chosen,
            "resolved_backend": "classical" if conflict else "quantum",
            "distance": round(d, 6),
            "threshold": round(tau, 6),
            "high_epistemic": bool(conflict),
            "conflict": bool(conflict),
        }


def _normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    return p / p.sum()
