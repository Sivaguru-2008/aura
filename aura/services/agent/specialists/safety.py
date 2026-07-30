from __future__ import annotations

from aura.schemas.clinical import Diagnosis, Modality
from aura.services.agent.base import AgentVerdict, ClinicalAgent


class SafetyAgent(ClinicalAgent):
    agent_name = "safety"

    def __init__(self, version: str = "1.0"):
        self.agent_version = version

    def supported(self, modality: Modality) -> bool:
        return True

    def get_base_weight(self) -> float:
        return 0.10

    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        safety_data = evidence.get("safety", {})
        calibration = evidence.get("calibration", {})

        args: list[str] = []
        refs: list[str] = []
        metadata: dict = {}

        is_ood = safety_data.get("is_ood", False)
        ood_energy = safety_data.get("ood_energy", 0.0)
        ood_threshold = safety_data.get("ood_threshold", 1.5)
        conformal_coverage = safety_data.get("conformal_coverage", 0.90)
        conformal_set_size = safety_data.get("conformal_set_size", 1)
        epistemic_uncertainty = safety_data.get("epistemic_uncertainty", 0.0)
        epistemic_threshold = safety_data.get("epistemic_threshold", 0.15)
        entropy = safety_data.get("predictive_entropy", 0.0)

        confident = True
        abstain = False

        if is_ood:
            args.append(f"OOD energy {ood_energy:.3f} exceeds threshold {ood_threshold}; scan falls outside validated distribution.")
            refs.append("Safety policy: OOD energy gate")
            metadata["ood_flagged"] = True
            confident = False
            abstain = True
        else:
            args.append("Scan is within the validated distribution (OOD energy within threshold).")
            metadata["ood_flagged"] = False

        if epistemic_uncertainty > epistemic_threshold:
            args.append(f"Epistemic uncertainty {epistemic_uncertainty:.3f} exceeds safety threshold {epistemic_threshold}.")
            refs.append("Safety policy: epistemic uncertainty gate")
            confident = False
            if epistemic_uncertainty > epistemic_threshold * 1.5:
                abstain = True

        if conformal_set_size > 4:
            args.append(f"Conformal set size {conformal_set_size} exceeds abstention threshold; differential remains wide.")
            refs.append("Safety policy: conformal abstention gate")
            confident = False

        if entropy > 1.5:
            args.append(f"Predictive entropy {entropy:.2f} bits is elevated; model is uncertain about the diagnosis.")
            refs.append("Safety policy: entropy ceiling")

        if abstain:
            args.append("RECOMMENDATION: Defer to senior review — automatic diagnosis not safe.")
        elif confident:
            args.append("All safety checks passed — prediction meets readiness criteria for autonomous reporting.")
        else:
            args.append("Proceed with caution — some safety checks are borderline but non-critical.")

        metadata["abstain"] = abstain
        metadata["confident"] = confident
        metadata["ood_energy"] = round(ood_energy, 4)
        metadata["epistemic_uncertainty"] = round(epistemic_uncertainty, 4)
        metadata["conformal_set_size"] = conformal_set_size

        confidence = 0.0 if abstain else (0.95 if confident else 0.6)

        return AgentVerdict(
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            findings={},
            confidence=round(confidence, 4),
            arguments=args,
            guideline_references=refs,
            metadata=metadata,
        )
