from __future__ import annotations

import numpy as np

from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Finding, Modality
from aura.services.agent.base import AgentVerdict, ClinicalAgent


class PathologyAgent(ClinicalAgent):
    agent_name = "pathology"

    def __init__(self, version: str = "1.0"):
        self.agent_version = version

    def get_base_weight(self) -> float:
        return 0.15

    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        findings: dict[Finding, float] = evidence.get("findings", {})
        history = evidence.get("history", {})
        priors_raw = priors

        scores = {d: 0.0 for d in CHEST_DIAGNOSES}
        args: list[str] = []
        refs: list[str] = []

        nodule = findings.get(Finding.NODULE, 0.0)
        opacity = findings.get(Finding.OPACITY, 0.0)
        smoking_py = history.get("smoking_pack_years", 0.0)
        prior_cancer = history.get("prior_cancer", False) or priors_raw.get("prior_cancer", False)
        smoker = history.get("smoker", False) or priors_raw.get("smoker", False)
        age_band = priors_raw.get("age_band", "unknown")

        malignancy_risk = 0.0
        if nodule >= 0.4:
            malignancy_risk += nodule * 0.6
            args.append(f"Nodule probability {nodule:.0%} on imaging suggests a pulmonary lesion requiring tissue characterisation.")
            refs.append("Fleischner Society: solid nodule risk stratification")

            if smoking_py >= 20 or smoker:
                malignancy_risk += 0.25
                args.append(f"Heavy smoking history ({smoking_py:.0f} pack-years) significantly elevates malignancy risk.")
                refs.append("NCCN lung cancer screening: high-risk smoking history")

            if prior_cancer:
                malignancy_risk += 0.3
                args.append("Prior cancer history raises suspicion for metastatic disease or second primary.")
                refs.append("NCCN guidelines: prior malignancy surveillance")

            if age_band == "65+":
                malignancy_risk += 0.1
                args.append("Age 65+ is an independent risk factor for pulmonary malignancy.")
                refs.append("SEER lung cancer incidence data")

            scores[Diagnosis.MALIGNANCY] = min(0.95, malignancy_risk)

            if nodule >= 0.4 and smoking_py < 10 and not prior_cancer and age_band != "65+":
                args.append("Nodule identified but low-risk profile; inflammation or granuloma remains possible.")
                scores[Diagnosis.MALIGNANCY] = min(0.4, nodule * 0.5)
                refs.append("Fleischner Society: low-risk nodule management")

        consol = findings.get(Finding.CONSOLIDATION, 0.0)
        if consol >= 0.5 and malignancy_risk < 0.5:
            args.append("Consolidation without high-risk nodule features is more consistent with inflammation than neoplasm.")
            scores[Diagnosis.PNEUMONIA] = max(scores.get(Diagnosis.PNEUMONIA, 0.0), 0.6)

        opacity_score = findings.get(Finding.OPACITY, 0.0)
        if opacity_score >= 0.4 and malignancy_risk < 0.3:
            scores[Diagnosis.PNEUMONIA] = max(scores.get(Diagnosis.PNEUMONIA, 0.0), 0.4)

        if scores[Diagnosis.MALIGNANCY] >= 0.5:
            scores[Diagnosis.NORMAL] = 0.0
            args.append(f"Malignancy risk score {scores[Diagnosis.MALIGNANCY]:.0%} — recommend tissue sampling for definitive diagnosis.")
            refs.append("NCCN non-small cell lung cancer: tissue diagnosis pathway")

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
