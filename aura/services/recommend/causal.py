"""Joint Expected-Information-Gain with causal redundancy masking (Module 14).

Why this exists
---------------
``RecommendEngine`` scores a *panel* of tests by joint value of information, and
computes the joint term by enumerating every ``2**|channels|`` resolution outcome
(``_expected_over_outcomes``). That is correct but **exponential**: a panel that
touches 6 evidence channels costs 64 posterior evaluations *per candidate per
greedy step*. On an edge device that is the reported "computational explosion".

It is also statistically wasteful. Correlated markers double-count. Troponin and
BNP both rise in cardiac decompensation; scoring each one's information about the
diagnosis *independently* and adding them overstates the value of ordering both —
once you have troponin, BNP tells you much less than its standalone EIG suggests.

This module replaces the exponential joint with the **mutual-information chain
rule**, which is an *exact identity* (not an approximation):

    I(Y; X_i, X_j) = I(Y; X_i) + I(Y; X_j | X_i).

The only thing we approximate is the conditional term. For jointly-Gaussian
markers the information X_j carries about Y that is *not already in X_i* shrinks
with the squared correlation:

    I(Y; X_j | X_i) ≈ I(Y; X_j) · (1 − ρ_ij²),

because ρ_ij² is the fraction of X_j's variance explained by X_i (the "novel"
fraction is ``1 − ρ_ij²``; exact for Gaussians via the partial correlation — see
``docs/ARCHITECTURE_REFACTOR.md`` for the derivation and its caveats). We further
gate ρ_ij by a **hardcoded directed clinical dependency graph** so the
deweighting is *causal*, not merely correlational: an edge ``i → j`` means "once
``i`` is known, ``j`` is partly explained", and only then is ``j`` deweighted.

Greedy submodular selection over this surrogate is **O(K · N)** — precompute the N
marginal EIGs once, then K greedy steps each rescanning N candidates against a
cached novelty vector (updated in O(N) when a marker is picked). No per-step
posterior enumeration.
"""
from __future__ import annotations

import numpy as np

# Markers = the 8 imaging evidence channels + the correlated cardiac / inflammatory
# labs that motivate the redundancy handling (the Troponin/BNP case in the brief).
IMAGING_CHANNELS = [
    "opacity", "consolidation", "effusion", "cardiomegaly",
    "nodule", "hyperinflation", "pneumothorax", "prior_risk",
]
LAB_MARKERS = ["troponin", "bnp", "d_dimer", "wbc", "crp"]
MARKERS = IMAGING_CHANNELS + LAB_MARKERS
_IDX = {m: i for i, m in enumerate(MARKERS)}

# Directed clinical dependency graph: (parent, child, strength in [0,1]).
# "parent explains child": selecting the parent deweights the child's marginal EIG.
CLINICAL_DEPENDENCY_EDGES: list[tuple[str, str, float]] = [
    ("consolidation", "opacity", 0.90),   # consolidation is a dense opacity — subsumes it
    ("opacity", "consolidation", 0.50),   # weaker reverse: opacity hints at consolidation
    ("nodule", "opacity", 0.35),
    ("cardiomegaly", "bnp", 0.80),        # enlarged silhouette explains a raised BNP
    ("troponin", "bnp", 0.70),            # both cardiac; troponin subsumes some BNP info
    ("bnp", "cardiomegaly", 0.55),
    ("effusion", "cardiomegaly", 0.40),   # cardiogenic effusion co-occurs with big heart
    ("wbc", "crp", 0.65), ("crp", "wbc", 0.65),   # inflammatory pair, near-symmetric
    ("consolidation", "crp", 0.30),
]

# Historical correlation matrix ρ over markers (a plausible clinical prior; replace
# with an estimate from your local outcome log). Symmetric; magnitude = the strongest
# clinical link between the pair (correlation carries the edge strength directly).
def _build_correlation() -> np.ndarray:
    n = len(MARKERS)
    R = np.eye(n)
    for parent, child, w in CLINICAL_DEPENDENCY_EDGES:
        i, j = _IDX[parent], _IDX[child]
        R[i, j] = max(R[i, j], w)
        R[j, i] = max(R[j, i], w)         # correlation itself is symmetric
    return R


HISTORICAL_CORRELATION = _build_correlation()


class CausalDependencyGraph:
    """Directed clinical graph + historical correlation → a redundancy operator.

    ``redundancy(i, j)`` returns ``r_ij ∈ [0, 1]``: the fraction of marker ``j``'s
    information about the diagnosis that is already implied once marker ``i`` is
    known. It combines the *symmetric* correlation ρ_ij (which explains a ρ² share
    of variance) with the *directed* edge gate m_ij ∈ [0,1], so
    ``r_ij = m_ij · ρ_ij²``. With no ``i → j`` edge, m_ij = 0 and ``j`` is not
    deweighted by ``i`` (correlation alone does not imply causal subsumption).
    """

    def __init__(self, markers: list[str] | None = None,
                 correlation: np.ndarray | None = None,
                 edges: list[tuple[str, str, float]] | None = None):
        self.markers = markers or MARKERS
        self.idx = {m: i for i, m in enumerate(self.markers)}
        self.R = HISTORICAL_CORRELATION if correlation is None else np.asarray(correlation, float)
        n = len(self.markers)
        self.M = np.zeros((n, n))          # directed gate m_ij
        for parent, child, w in (edges or CLINICAL_DEPENDENCY_EDGES):
            if parent in self.idx and child in self.idx:
                self.M[self.idx[parent], self.idx[child]] = float(w)

    def redundancy(self, i: int, j: int) -> float:
        """r_ij = m_ij · ρ_ij² — directed-gated squared correlation (variance share)."""
        rho = float(self.R[i, j])
        gate = float(self.M[i, j])
        return float(min(1.0, gate * rho ** 2))

    def redundancy_vector(self, i: int) -> np.ndarray:
        """r_i· for every j at once (used to update the novelty cache in O(N))."""
        return np.clip(self.M[i] * self.R[i] ** 2, 0.0, 1.0)


class JointEIGSelector:
    """Greedy submodular panel selection under chained MI + causal masking.

    Contract:
        select(marginal_eig, k, min_gain) ->
            {"order": [marker,...], "joint_eig": float, "marginal_after_mask": {...}}

    ``marginal_eig`` maps marker name -> I(Y; X_marker) in bits (the standalone
    single-test EIG the caller already computes). The selector never re-enumerates
    outcomes; all redundancy comes from the graph. Total cost is O(k · N).
    """

    def __init__(self, graph: CausalDependencyGraph | None = None):
        self.graph = graph or CausalDependencyGraph()

    def select(self, marginal_eig: dict[str, float], k: int = 3,
               min_gain: float = 1e-3) -> dict:
        markers = self.graph.markers
        n = len(markers)
        I = np.array([float(marginal_eig.get(m, 0.0)) for m in markers])
        novelty = np.ones(n)               # ∏_{i∈S}(1 - r_ij) cached per candidate
        chosen: list[int] = []
        joint = 0.0
        masked_trace: dict[str, float] = {}

        for _ in range(min(k, n)):
            scores = I * novelty
            for c in chosen:
                scores[c] = -np.inf        # no re-selection
            j = int(np.argmax(scores))
            gain = float(scores[j])
            if gain < min_gain:
                break
            chosen.append(j)
            joint += gain                  # chain rule: Σ marginal-novel contributions
            masked_trace[markers[j]] = round(gain, 5)
            # Deweight every remaining candidate correlated-and-caused by j, O(N).
            novelty = novelty * (1.0 - self.graph.redundancy_vector(j))

        return {
            "order": [markers[c] for c in chosen],
            "joint_eig": round(joint, 5),
            "marginal_after_mask": masked_trace,
        }

    def joint_eig(self, markers_selected: list[str], marginal_eig: dict[str, float]) -> float:
        """Evaluate the chained-MI joint EIG of a *fixed* ordered marker list.

        Used to compare a proposed panel against the independent-sum baseline
        (which is ``Σ marginal_eig`` and provably ≥ this — the double-counting).
        """
        joint = 0.0
        novelty = {m: 1.0 for m in marker_names(self.graph)}
        for m in markers_selected:
            if m not in self.graph.idx:
                continue
            joint += float(marginal_eig.get(m, 0.0)) * novelty[m]
            rv = self.graph.redundancy_vector(self.graph.idx[m])
            for other, oi in self.graph.idx.items():
                novelty[other] *= (1.0 - float(rv[oi]))
        return round(joint, 5)


def marker_names(graph: CausalDependencyGraph) -> list[str]:
    return list(graph.markers)
