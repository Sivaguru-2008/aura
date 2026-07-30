from __future__ import annotations

import numpy as np

from aura.schemas.clinical import CHEST_DIAGNOSES, DIAGNOSES, Diagnosis, Finding, Modality
from aura.services.agent.base import AgentVerdict, ClinicalAgent

_FINDING_TO_DX: dict[Finding, list[Diagnosis]] = {
    Finding.OPACITY: [Diagnosis.PNEUMONIA, Diagnosis.HEART_FAILURE],
    Finding.CONSOLIDATION: [Diagnosis.PNEUMONIA],
    Finding.EFFUSION: [Diagnosis.HEART_FAILURE, Diagnosis.PNEUMONIA],
    Finding.CARDIOMEGALY: [Diagnosis.HEART_FAILURE],
    Finding.NODULE: [Diagnosis.MALIGNANCY],
    Finding.PNEUMOTHORAX: [Diagnosis.PNEUMOTHORAX],
    Finding.HYPERINFLATION: [Diagnosis.COPD],
}


class RadiologistAgent(ClinicalAgent):
    agent_name = "radiologist"

    def __init__(self, version: str = "1.0"):
        self.agent_version = version

    def get_base_weight(self) -> float:
        return 0.35

    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        findings: dict[Finding, float] = evidence.get("findings", {})
        embedding: list[float] = evidence.get("embedding", [])

        scores = {d: 0.0 for d in CHEST_DIAGNOSES}
        args: list[str] = []
        refs: list[str] = []
        activated = []

        for finding, prob in findings.items():
            if prob >= 0.5:
                for dx in _FINDING_TO_DX.get(finding, []):
                    scores[dx] = max(scores[dx], prob)
                activated.append(finding)

        if activated:
            desc = ", ".join(f"{f.value} ({findings[f]:.0%})" for f in activated)
            args.append(f"Visual analysis reveals {desc}.")
        else:
            args.append("No significant radiographic abnormalities detected.")
            scores[Diagnosis.NORMAL] = 0.6

        if "consolidation" in [f.value for f in activated]:
            args.append("Airspace consolidation suggests parenchymal infection.")
            refs.append("Fleischner Society: consolidation pattern")
        if "nodule" in [f.value for f in activated]:
            args.append("Pulmonary nodule identified — assess size, margins, and growth.")
            refs.append("Fleischner Society pulmonary nodule guidelines")

        total = sum(scores.values()) or 1.0
        normalized = {d: v / total for d, v in scores.items()}

        conf = min(0.95, max(0.3, max(normalized.values())))

        return AgentVerdict(
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            findings=normalized,
            confidence=round(conf, 4),
            arguments=args,
            guideline_references=refs,
        )
