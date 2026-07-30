"""ClinicalSafetyController — Layer 1 veto checks on incoming evidence.

Runs immediately after image analysis and before reasoning/recommendation.
If any veto check fails the pipeline aborts; the controller emits structured
output so the dashboard can show exactly which parameter breached the safety
envelope and what clinical action is recommended.
"""
from __future__ import annotations

import numpy as np

from aura.common.config import get_safety_policy, get_settings
from aura.common.mathx import energy_score, softmax
from aura.schemas.contracts import (
    MitigationAction,
    SafetyControllerCheck,
    SafetyControllerOutput,
)

CONTROLLER_VERSION = "controller-v1"

# Recommended clinical actions keyed by failed check name.
_RECOMMENDATIONS: dict[str, str] = {
    "data_integrity": (
        "Image data integrity check failed. REPEAT_ACQUISITION: "
        "re-acquire the study with proper technique."
    ),
    "ood_energy": (
        "Out-of-distribution energy score exceeds threshold. "
        "ESC_HUMAN_EXPERT_REVIEW: escalate to attending radiologist."
    ),
    "epistemic": (
        "Epistemic uncertainty exceeds threshold. "
        "ORDER_CONFIRMATORY_EVIDENCE: obtain additional clinical data."
    ),
    "data_quality": (
        "Image quality is below the minimum threshold. "
        "REPEAT_ACQUISITION: re-acquire the study."
    ),
}


def compute_safety_confidence(measured: float, threshold: float,
                              scale: float = 1.0) -> float:
    """Sigmoid scaling of violation severity.

    Returns a value in [0, 1] where 0 means far above threshold (severe
    violation) and 1 means well within bounds (safe).
    """
    return 1.0 / (1.0 + np.exp(-(threshold - measured) / max(scale, 1e-6)))


class ClinicalSafetyController:
    """Layer 1 safety gate — runs veto checks before reasoning.

    Loads the active policy threshold profile on initialization and applies
    data-integrity, OOD-energy, epistemic, and data-quality checks against
    the configured limits.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.policy = get_safety_policy()
        self.policy_name = self._resolve_policy_name()

    def _resolve_policy_name(self) -> str:
        import os
        return os.environ.get("AURA_SAFETY_POLICY", "community_conservative")

    def check(
        self,
        evidence_vector: np.ndarray,
        logits: np.ndarray,
        temperature: float,
        ood_mean: float,
        ood_std: float,
        epistemic_std: float,
        epistemic_mi: float,
        image_quality: float | None = None,
        image_shape: tuple[int, int] | None = None,
        aspect_ratio: float | None = None,
    ) -> SafetyControllerOutput:
        """Run all veto checks and return structured output.

        Parameters
        ----------
        evidence_vector : the encoded evidence vector x.
        logits : raw fusion logits (default backend).
        temperature : calibrated temperature.
        ood_mean, ood_std : in-distribution energy statistics.
        epistemic_std : ensemble top-class disagreement.
        epistemic_mi : mutual information (BALD).
        image_quality : optional quality score in [0, 1].
        image_shape : optional (rows, cols) for aspect-ratio check.
        aspect_ratio : optional pre-computed aspect ratio.
        """
        checks: list[SafetyControllerCheck] = []
        failed = False

        # --- 1. Data Integrity Check ---
        di_passed = True
        di_severity = 0.0
        di_detail = ""
        if image_shape is not None:
            rows, cols = image_shape
            ar = aspect_ratio if aspect_ratio is not None else (cols / max(rows, 1))
            if ar < 0.4 or ar > 2.5:
                di_passed = False
                di_severity = compute_safety_confidence(ar, 1.5, scale=0.3)
                di_detail = f"Aspect ratio {ar:.2f} outside valid range [0.40, 2.50]"
        if evidence_vector is not None and len(evidence_vector) > 0:
            flat = np.asarray(evidence_vector, dtype=float).ravel()
            unreadable = float(np.mean((flat <= 0.0) | (flat >= 1.0)))
            if unreadable > 0.95:
                di_passed = False
                di_severity = max(di_severity, compute_safety_confidence(0.95, unreadable, 0.05))
                di_detail = f"{unreadable*100:.1f}% unreadable pixels"
        if not di_passed and not di_detail:
            di_detail = "Data integrity check failed"
        checks.append(SafetyControllerCheck(
            name="data_integrity", passed=di_passed,
            measured=0.0, threshold=0.0,
            severity=di_severity, detail=di_detail,
        ))
        if not di_passed:
            failed = True

        # --- 2. OOD Energy Check ---
        ood_energy = energy_score(logits, temperature)
        z = (ood_energy - ood_mean) / max(ood_std, 1e-6)
        ood_passed = bool(z <= self.policy.ood_threshold)
        ood_severity = compute_safety_confidence(z, self.policy.ood_threshold, scale=1.0)
        checks.append(SafetyControllerCheck(
            name="ood_energy", passed=ood_passed,
            measured=round(z, 4), threshold=self.policy.ood_threshold,
            severity=round(1.0 - ood_severity, 4),
            detail="" if ood_passed else f"OOD z-score {z:.2f} > threshold {self.policy.ood_threshold}",
        ))
        if not ood_passed:
            failed = True

        # --- 3. Epistemic Uncertainty Check ---
        epi_passed = bool(epistemic_std <= self.policy.epistemic_threshold)
        epi_severity = compute_safety_confidence(
            epistemic_std, self.policy.epistemic_threshold, scale=0.05
        )
        checks.append(SafetyControllerCheck(
            name="epistemic", passed=epi_passed,
            measured=round(epistemic_std, 4), threshold=self.policy.epistemic_threshold,
            severity=round(1.0 - epi_severity, 4),
            detail="" if epi_passed else (
                f"Epistemic std {epistemic_std:.4f} > threshold {self.policy.epistemic_threshold}"
            ),
        ))
        if not epi_passed:
            failed = True

        # --- 4. Data Quality Check (optional) ---
        if image_quality is not None:
            qual_passed = bool(image_quality >= 0.3)
            qual_severity = compute_safety_confidence(image_quality, 0.3, scale=0.1)
            checks.append(SafetyControllerCheck(
                name="data_quality", passed=qual_passed,
                measured=round(image_quality, 4), threshold=0.3,
                severity=round(1.0 - qual_severity, 4),
                detail="" if qual_passed else f"Image quality {image_quality:.3f} < 0.30",
            ))
            if not qual_passed:
                failed = True

        # --- Aggregate safety confidence ---
        if checks:
            overall = float(np.mean([c.severity for c in checks]))
        else:
            overall = 1.0

        # --- Build recommendation and mitigations ---
        failed_names = [c.name for c in checks if not c.passed]
        if failed_names:
            recommendation = "; ".join(_RECOMMENDATIONS.get(n, n) for n in failed_names)
        else:
            recommendation = "All safety checks passed. Proceed with reasoning."

        mitigations: list[MitigationAction] = []
        if not ood_passed:
            mitigations.append(MitigationAction.ESC_HUMAN_EXPERT_REVIEW)
        if not epi_passed:
            mitigations.append(MitigationAction.ORDER_CONFIRMATORY_EVIDENCE)
        if epistemic_mi > 0.1 and not epi_passed:
            mitigations.append(MitigationAction.SHOW_COMPETING_HYPOTHESES)

        state = "FAILED" if failed else ("WARNING" if overall < 0.8 else "PASSED")

        return SafetyControllerOutput(
            state=state,
            policy_name=self.policy_name,
            checks=checks,
            safety_confidence=round(overall, 4),
            recommendation=recommendation,
            mitigations=mitigations,
            model_version=CONTROLLER_VERSION,
        )

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
            payload = getattr(study, "payload", None) or study
            seqs = getattr(payload, "sequence_keys", None) or []
            missing = [s for s in ["flair", "t1", "t1ce", "t2"] if s not in seqs]
            if missing:
                checks["missing_sequences_count"] = float(len(missing))
                return False, f"Missing required MRI sequences: {', '.join(missing)}", checks
                
            return True, "", checks

    def assess(self, study, img: np.ndarray, x: np.ndarray, fusion_model, resolved_logits: np.ndarray, safety_engine) -> SafetyControllerOutput:
        # Run Data Integrity check
        di_ok, di_reason, di_checks = self.inspect_data_integrity(study, img)
        
        # OOD veto check
        cal = safety_engine.cal
        e = energy_score(resolved_logits, cal.temperature)
        z = (e - cal.ood_mean) / cal.ood_std
        
        # Epistemic check
        if safety_engine.ensemble is not None:
            dec = safety_engine._epistemic_ensemble(x)
        else:
            dec = safety_engine._epistemic_perturbation(fusion_model, x)
        epistemic = float(dec["epistemic_std"])
        epistemic_mi = float(dec.get("epistemic_mi", 0.0))

        # Delegate to check()
        return self.check(
            evidence_vector=x,
            logits=resolved_logits,
            temperature=cal.temperature,
            ood_mean=cal.ood_mean,
            ood_std=cal.ood_std,
            epistemic_std=epistemic,
            epistemic_mi=epistemic_mi,
            aspect_ratio=di_checks.get("aspect_ratio"),
            image_quality=1.0 if di_ok else 0.0,
        )


try:
    from aura.backend.core.shared.errors import AuraBackendError
    class ClinicalSafetyException(AuraBackendError):
        code = "clinical_safety_violation"
except ImportError:
    class ClinicalSafetyException(Exception):
        def __init__(self, reason: str, detail: dict | None = None):
            self.reason = reason
            self.detail = detail or {}
            super().__init__(reason)
