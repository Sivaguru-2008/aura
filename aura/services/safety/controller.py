from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Optional

from common.config import get_settings
from common.mathx import energy_score
from backend.core.shared.errors import AuraBackendError

class SafetyControllerOutput(BaseModel):
    status: str  # "PASSED" or "FAILED"
    failed_checks: List[str]
    scores: Dict[str, float]
    thresholds: Dict[str, float]
    recommendation: Optional[str] = None
    detail: Optional[str] = None

class ClinicalSafetyException(AuraBackendError):
    code = "clinical_safety_violation"
    http_status = 422

    def __init__(self, reason: str, *, detail: dict | None = None) -> None:
        super().__init__(reason, detail=detail)

def compute_safety_confidence(measured: float, threshold: float, scale: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-(measured - threshold) / scale)))

class ClinicalSafetyController:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.active_policy = self.settings.active_policy
        self.ood_threshold = self.settings.ood_threshold
        self.epistemic_threshold = self.settings.epistemic_threshold
        self.min_coverage = self.settings.min_coverage

    def inspect_data_integrity(self, study, img: np.ndarray) -> tuple[bool, str, dict]:
        checks = {}
        modality = getattr(study, "modality", None)
        
        # Check if CXR modality
        is_cxr = True
        if modality is not None:
            is_cxr = (modality.value == "CXR" if hasattr(modality, "value") else str(modality) == "CXR")
            
        if is_cxr:
            h, w = img.shape[:2]
            aspect = h / max(1, w)
            checks["aspect_ratio"] = aspect
            
            # Aspect ratio check
            if not (0.4 <= aspect <= 2.5):
                return False, f"Image proportions (aspect ratio {aspect:.2f}) do not match a radiograph", checks
                
            # Unreadable pixels (gray std deviation check)
            std = float(img.std())
            checks["gray_std"] = std
            if std < 0.04:
                return False, f"Image is nearly uniform (std {std:.4f}) — no anatomical content", checks
                
            return True, "", checks
        else:
            # MR modality checks (incomplete sequence check)
            # study.payload holds MultiSequenceStudy
            payload = getattr(study, "payload", None) or study
            seqs = getattr(payload, "sequence_keys", None) or []
            missing = [s for s in ["flair", "t1", "t1ce", "t2"] if s not in seqs]
            if missing:
                checks["missing_sequences_count"] = float(len(missing))
                return False, f"Missing required MRI sequences: {', '.join(missing)}", checks
                
            return True, "", checks

    def assess(self, study, img: np.ndarray, x: np.ndarray, fusion_model, resolved_logits: np.ndarray, safety_engine) -> SafetyControllerOutput:
        # 1. Data Integrity veto check
        di_ok, di_reason, di_checks = self.inspect_data_integrity(study, img)
        
        # 2. OOD veto check
        cal = safety_engine.cal
        e = energy_score(resolved_logits, cal.temperature)
        z = (e - cal.ood_mean) / cal.ood_std
        ood_ok = z <= self.ood_threshold
        
        # 3. Epistemic veto check
        if safety_engine.ensemble is not None:
            dec = safety_engine._epistemic_ensemble(x)
        else:
            dec = safety_engine._epistemic_perturbation(fusion_model, x)
        epistemic = float(dec["epistemic_std"])
        epistemic_ok = epistemic <= self.epistemic_threshold

        failed_checks = []
        if not di_ok:
            failed_checks.append("DATA_INTEGRITY")
        if not ood_ok:
            failed_checks.append("OOD")
        if not epistemic_ok:
            failed_checks.append("EPISTEMIC")

        status = "FAILED" if failed_checks else "PASSED"
        
        scores = {
            "aspect_ratio": float(di_checks.get("aspect_ratio", 1.0)),
            "gray_std": float(di_checks.get("gray_std", 1.0)),
            "ood_energy_z": float(z),
            "epistemic_std": float(epistemic)
        }
        
        thresholds = {
            "ood_threshold": self.ood_threshold,
            "epistemic_threshold": self.epistemic_threshold,
            "min_coverage": self.min_coverage
        }
        
        recommendation = None
        if "DATA_INTEGRITY" in failed_checks:
            recommendation = "REPEAT_ACQUISITION"
        elif "OOD" in failed_checks or "EPISTEMIC" in failed_checks:
            recommendation = "HUMAN_ESCALATION"

        return SafetyControllerOutput(
            status=status,
            failed_checks=failed_checks,
            scores=scores,
            thresholds=thresholds,
            recommendation=recommendation,
            detail=di_reason if not di_ok else (f"Safety violation: {', '.join(failed_checks)}" if failed_checks else None)
        )
