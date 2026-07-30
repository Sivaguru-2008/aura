from __future__ import annotations

import numpy as np

from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Finding, Modality
from aura.services.agent.base import AgentVerdict, ClinicalAgent


class PulmonologistAgent(ClinicalAgent):
    agent_name = "pulmonologist"

    def __init__(self, version: str = "1.0"):
        self.agent_version = version

    def get_base_weight(self) -> float:
        return 0.25

    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        findings: dict[Finding, float] = evidence.get("findings", {})
        labs = evidence.get("labs", {})
        symptoms = evidence.get("symptoms", {})

        scores = {d: 0.0 for d in CHEST_DIAGNOSES}
        args: list[str] = []
        refs: list[str] = []

        bnp = labs.get("bnp")
        wbc = labs.get("wbc")
        procalcitonin = labs.get("procalcitonin")
        spo2 = labs.get("spo2")
        crp = labs.get("crp")

        orthopnea = symptoms.get("orthopnea", False)
        dyspnea = symptoms.get("dyspnea", False)
        fever = symptoms.get("fever", False)

        if bnp is not None and bnp >= 400:
            scores[Diagnosis.HEART_FAILURE] = max(scores[Diagnosis.HEART_FAILURE], 0.85)
            args.append(f"BNP {bnp:.0f} pg/mL is markedly elevated, consistent with acute decompensated heart failure.")
            refs.append("ACC/AHA HF guideline: BNP >400 pg/mL")
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.0)
            scores[Diagnosis.NORMAL] = max(scores[Diagnosis.NORMAL], 0.0)

        if bnp is not None and bnp < 100:
            scores[Diagnosis.HEART_FAILURE] = max(scores[Diagnosis.HEART_FAILURE], 0.0)
            args.append("BNP <100 pg/mL argues against heart failure as the primary driver.")
            refs.append("ACC/AHA HF guideline: BNP <100 pg/mL")

        if wbc is not None and wbc > 11:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.7)
            args.append(f"Leukocytosis (WBC {wbc:.1f}) suggests systemic infection.")
            refs.append("IDSA/ATS CAP guideline")
            if procalcitonin is not None and procalcitonin >= 0.5:
                scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.85)
                args.append(f"Elevated procalcitonin ({procalcitonin:.2f}) indicates bacterial aetiology.")
                refs.append("IDSA/ATS CAP: procalcitonin >0.5 ng/mL")

        if crp is not None and crp > 50:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.6)
            args.append(f"CRP {crp:.0f} mg/L confirms active inflammation.")
            refs.append("IDSA/ATS CAP guideline")

        if spo2 is not None and spo2 < 92:
            args.append(f"Hypoxaemia (SpO2 {spo2:.0f}%) indicates clinically significant cardiopulmonary impairment.")
            refs.append("Oxygenation red-flag threshold")
            scores[Diagnosis.NORMAL] = 0.0

        if orthopnea:
            scores[Diagnosis.HEART_FAILURE] = max(scores[Diagnosis.HEART_FAILURE], 0.75)
            args.append("Orthopnoea is a specific symptom of congestive heart failure.")
            refs.append("ACC/AHA HF: orthopnoea as clinical sign")

        if dyspnea and not orthopnea:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.4)
            args.append("Dyspnoea without orthopnoea supports a pulmonary aetiology.")

        consol = findings.get(Finding.CONSOLIDATION, 0.0)
        if consol >= 0.4:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.8)
            args.append("Radiographic consolidation with compatible labs supports community-acquired pneumonia.")
            refs.append("IDSA/ATS CAP: clinical + radiographic diagnosis")

        hyper = findings.get(Finding.HYPERINFLATION, 0.0)
        if hyper >= 0.4:
            scores[Diagnosis.COPD] = max(scores[Diagnosis.COPD], hyper)
            args.append("Hyperinflation on CXR is consistent with COPD.")
            refs.append("GOLD COPD report")

        if scores[Diagnosis.COPD] > 0.0 and bnp is not None and bnp < 100:
            args.append("BNP is low; dyspnoea is more likely COPD exacerbation than HF.")
            refs.append("GOLD COPD: differential from HF")

        total = sum(scores.values()) or 1.0
        normalized = {d: v / total for d, v in scores.items()}
        conf = min(0.9, max(0.3, max(normalized.values())))

        return AgentVerdict(
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            findings=normalized,
            confidence=round(conf, 4),
            arguments=args,
            guideline_references=refs,
        )
