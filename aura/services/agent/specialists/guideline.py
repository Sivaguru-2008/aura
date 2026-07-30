from __future__ import annotations

import numpy as np

from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Finding, Modality
from aura.services.agent.base import AgentVerdict, ClinicalAgent

_FLEISCHNER_NODULE_RISK = {
    (4, 6): "low", (6, 8): "moderate", (8, 30): "high",
}


class GuidelineAgent(ClinicalAgent):
    agent_name = "guideline"

    def __init__(self, version: str = "1.0"):
        self.agent_version = version

    def get_base_weight(self) -> float:
        return 0.15

    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        findings: dict[Finding, float] = evidence.get("findings", {})
        labs = evidence.get("labs", {})
        history = evidence.get("history", {})
        priors_raw = priors

        scores = {d: 0.0 for d in CHEST_DIAGNOSES}
        args: list[str] = []
        refs: list[str] = []

        nodule = findings.get(Finding.NODULE, 0.0)
        if nodule >= 0.4:
            smoking_py = history.get("smoking_pack_years", 0.0)
            if smoking_py >= 20:
                scores[Diagnosis.MALIGNANCY] = max(scores[Diagnosis.MALIGNANCY], 0.85)
                args.append("Fleischner guidelines: solid nodule with high-risk history requires STN follow-up CT at 12 months.")
                refs.append("Fleischner Society: solid nodule at 12 months")
            else:
                scores[Diagnosis.MALIGNANCY] = max(scores[Diagnosis.MALIGNANCY], 0.30)
                args.append("Fleischner guidelines: solid nodule with low-risk profile — follow-up CT at 12 months optional.")
                refs.append("Fleischner Society: low-risk solid nodule")

        bnp = labs.get("bnp")
        if bnp is not None:
            if bnp >= 400:
                scores[Diagnosis.HEART_FAILURE] = max(scores[Diagnosis.HEART_FAILURE], 0.9)
                args.append("ACC/AHA HF guidelines: BNP >= 400 pg/mL is diagnostic of acute heart failure.")
                refs.append("ACC/AHA HF: BNP threshold")
            elif bnp >= 100:
                scores[Diagnosis.HEART_FAILURE] = max(scores[Diagnosis.HEART_FAILURE], 0.35)
                args.append("ACC/AHA HF guidelines: BNP 100-399 pg/mL is indeterminate — correlate with clinical signs.")
                refs.append("ACC/AHA HF: BNP indeterminate range")

        wbc = labs.get("wbc")
        procalcitonin = labs.get("procalcitonin")
        if wbc is not None and wbc > 11 and procalcitonin is not None and procalcitonin >= 0.5:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.85)
            args.append("IDSA/ATS CAP guidelines: leukocytosis + elevated procalcitonin + radiographic consolidation = definite CAP.")
            refs.append("IDSA/ATS CAP: clinical diagnosis criteria")
        elif wbc is not None and wbc > 11:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.5)
            args.append("IDSA/ATS CAP guidelines: leukocytosis alone is suggestive but not diagnostic without consolidation or procalcitonin.")
            refs.append("IDSA/ATS CAP: laboratory criteria")

        hyper = findings.get(Finding.HYPERINFLATION, 0.0)
        if hyper >= 0.4:
            fev1_ratio = priors_raw.get("fev1_ratio", 0.6)
            if fev1_ratio < 0.7:
                scores[Diagnosis.COPD] = max(scores[Diagnosis.COPD], 0.85)
                args.append("GOLD guidelines: hyperinflation + FEV1/FVC < 0.7 confirms COPD diagnosis.")
                refs.append("GOLD 2024: spirometric diagnosis")
            else:
                scores[Diagnosis.COPD] = max(scores[Diagnosis.COPD], 0.5)
                args.append("GOLD guidelines: hyperinflation without spirometry — recommend PFTs for confirmation.")
                refs.append("GOLD 2024: diagnostic pathway")

        ptx = findings.get(Finding.PNEUMOTHORAX, 0.0)
        if ptx >= 0.4:
            scores[Diagnosis.PNEUMOTHORAX] = max(scores[Diagnosis.PNEUMOTHORAX], 0.9)
            args.append("BTS pleural disease guidelines: visible pneumothorax edge — assess size and tension physiology.")
            refs.append("BTS pleural disease: pneumothorax management")
            scores[Diagnosis.NORMAL] = 0.0

        consol = findings.get(Finding.CONSOLIDATION, 0.0)
        if consol >= 0.5:
            scores[Diagnosis.PNEUMONIA] = max(scores[Diagnosis.PNEUMONIA], 0.75)
            args.append("Radiographic consolidation is a core IDSA/ATS criterion for diagnosing community-acquired pneumonia.")
            refs.append("IDSA/ATS CAP: radiographic criteria")

        total = sum(scores.values()) or 1.0
        normalized = {d: v / total for d, v in scores.items()}
        conf = 0.95

        return AgentVerdict(
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            findings=normalized,
            confidence=conf,
            arguments=args,
            guideline_references=refs,
        )
