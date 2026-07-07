"""Evidence encoder: vision findings + structured priors -> 8-channel vector in [0,1].

Eight channels == n_qubits. Each channel becomes one qubit's rotation angle.
This compression is the whole reason quantum is tractable here: we reason over a
handful of clinically-meaningful evidence dimensions, not raw pixels.
"""
from __future__ import annotations

import numpy as np

from schemas.clinical import Finding
from schemas.contracts import EvidenceItem, EvidenceKind, StructuredPriors, VisionResult

EVIDENCE_CHANNELS = [
    "opacity",
    "consolidation",
    "effusion",
    "cardiomegaly",
    "nodule",
    "hyperinflation",
    "pneumothorax",
    "prior_risk",
]

_FINDING_CHANNEL = {
    Finding.OPACITY: "opacity",
    Finding.CONSOLIDATION: "consolidation",
    Finding.EFFUSION: "effusion",
    Finding.CARDIOMEGALY: "cardiomegaly",
    Finding.NODULE: "nodule",
    Finding.HYPERINFLATION: "hyperinflation",
    Finding.PNEUMOTHORAX: "pneumothorax",
}


def prior_risk_score(p: StructuredPriors) -> float:
    """Composite malignancy/severity risk from structured priors, in [0,1]."""
    s = 0.0
    s += 0.35 if p.smoker else 0.0
    s += 0.30 if p.prior_cancer else 0.0
    s += 0.20 if p.fever else 0.0
    s += 0.15 if p.age_band == "65+" else (0.07 if p.age_band == "40-65" else 0.0)
    s += 0.10 if p.immunocompromised else 0.0
    return float(min(1.0, s))


def encode(vision: VisionResult, priors: StructuredPriors) -> np.ndarray:
    ch = {c: 0.0 for c in EVIDENCE_CHANNELS}
    for fs in vision.findings:
        c = _FINDING_CHANNEL.get(fs.finding)
        if c is not None:
            ch[c] = max(ch[c], float(fs.probability))
    ch["prior_risk"] = prior_risk_score(priors)
    return np.array([ch[c] for c in EVIDENCE_CHANNELS], dtype=float)


def to_evidence_items(vec: np.ndarray, priors: StructuredPriors) -> list[EvidenceItem]:
    """Human-facing evidence-graph nodes, including notable *absent* evidence."""
    items: list[EvidenceItem] = []
    for name, v in zip(EVIDENCE_CHANNELS, vec):
        if name == "prior_risk":
            items.append(
                EvidenceItem(
                    kind=EvidenceKind.STRUCTURED_PRIOR,
                    name="prior_risk",
                    value=round(float(v), 4),
                    probability=round(float(v), 4),
                    source_service="fusion.evidence",
                )
            )
        else:
            items.append(
                EvidenceItem(
                    kind=(
                        EvidenceKind.IMAGING_FINDING
                        if v >= 0.5
                        else EvidenceKind.ABSENT_EVIDENCE
                    ),
                    name=name,
                    value=round(float(v), 4),
                    probability=round(float(v), 4),
                    uncertainty=round(float(0.5 - abs(v - 0.5)), 4),  # max near 0.5
                    source_service="vision",
                )
            )
    return items
