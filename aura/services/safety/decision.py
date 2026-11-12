"""Decision Assurance Engine — 3-tier clinical decision assurance.

Implements rigorous decision rules combining:
1. Conformal prediction set coverage (90% guarantee)
2. Calibrated confidence thresholds
3. Out-of-Distribution (OOD) energy detector z-scores
4. Epistemic uncertainty / ensemble variance
5. Clinical safety controller vetoes and CDRE readiness profiles

Outputs:
  * ACCEPT: Clear, highly confident, in-distribution call meeting all safety gates.
  * REVIEW_REQUIRED: Competing hypotheses, borderline confidence, or elevated uncertainty.
  * ABSTAIN: Severe OOD, low confidence, huge conformal set, or safety veto.
"""
from __future__ import annotations

from typing import Optional
from datetime import datetime, timezone

from aura.common.config import get_settings
from aura.schemas.contracts import (
    DecisionAssurance,
    DecisionVerdict,
    MitigationAction,
    ReadinessState,
    SafetyAssessment,
    SafetyControllerOutput,
    DecisionReadinessProfile,
)


class DecisionAssuranceEngine:
    """Evaluates case safety metrics and produces a 3-tier assurance verdict."""

    def __init__(
        self,
        min_accept_confidence: float = 0.65,
        ood_warn_threshold: float = 1.0,
        epistemic_warn_threshold: float = 0.12,
        max_accept_conformal_size: int = 2,
    ):
        self.settings = get_settings()
        self.min_accept_confidence = min_accept_confidence
        self.ood_warn_threshold = ood_warn_threshold
        self.epistemic_warn_threshold = epistemic_warn_threshold
        self.max_accept_conformal_size = max_accept_conformal_size
        self.model_version = "decision-assurance-v1"

    def evaluate(
        self,
        safety: SafetyAssessment,
        readiness: Optional[DecisionReadinessProfile] = None,
        controller: Optional[SafetyControllerOutput] = None,
    ) -> DecisionAssurance:
        reasons: list[str] = []
        mitigations: list[str] = []
        limiting_factor = ""

        # Extract metrics safely
        top_prob = float(safety.top_probability) if safety.top_probability is not None else 0.0
        primary_dx = safety.top.value if hasattr(safety.top, "value") else str(safety.top)
        conformal_set = [d.value if hasattr(d, "value") else str(d) for d in safety.conformal_set]
        conformal_size = len(conformal_set)
        ood_energy = float(safety.ood_energy) if safety.ood_energy is not None else 0.0
        is_ood = bool(safety.is_ood)
        epistemic = float(safety.epistemic_uncertainty) if safety.epistemic_uncertainty is not None else 0.0
        coverage = float(safety.conformal_coverage) if safety.conformal_coverage is not None else 0.90
        method = safety.conformal_method or "marginal"

        # -------------------------------------------------------------
        # 1. ABSTAIN CRITERIA (Hard veto gates)
        # -------------------------------------------------------------
        abstain_triggered = False

        if controller and controller.state == "FAILED":
            abstain_triggered = True
            reasons.append("Clinical safety controller vetoed execution (state: FAILED)")
            limiting_factor = "safety_controller_veto"

        if is_ood or ood_energy >= self.settings.ood_energy_threshold:
            abstain_triggered = True
            reasons.append(
                f"Out-of-distribution input detected (OOD energy z-score: {ood_energy:.2f} >= {self.settings.ood_energy_threshold:.2f})"
            )
            if not limiting_factor:
                limiting_factor = "out_of_distribution"

        if top_prob < self.settings.low_confidence_threshold:
            abstain_triggered = True
            reasons.append(
                f"Low predictive confidence ({top_prob:.2f} < {self.settings.low_confidence_threshold:.2f})"
            )
            if not limiting_factor:
                limiting_factor = "low_confidence"

        if conformal_size > self.settings.abstention_conformal_size:
            abstain_triggered = True
            reasons.append(
                f"Conformal set size too large ({conformal_size} candidates > {self.settings.abstention_conformal_size})"
            )
            if not limiting_factor:
                limiting_factor = "large_conformal_set"

        if epistemic >= self.settings.epistemic_abstention_threshold:
            abstain_triggered = True
            reasons.append(
                f"High epistemic uncertainty / ensemble disagreement ({epistemic:.2f} >= {self.settings.epistemic_abstention_threshold:.2f})"
            )
            if not limiting_factor:
                limiting_factor = "high_epistemic_uncertainty"

        if safety.abstained:
            abstain_triggered = True
            reason_str = (
                safety.abstention_reason.value
                if hasattr(safety.abstention_reason, "value")
                else str(safety.abstention_reason)
            )
            if not any("abstained" in r.lower() for r in reasons):
                reasons.append(f"Safety engine flagged abstention: {reason_str}")
            if not limiting_factor:
                limiting_factor = "safety_engine_abstained"

        if abstain_triggered:
            mitigations.extend([
                "esc_human_expert_review",
                "re_acquire_shots_or_study",
                "clinical_investigation_required",
            ])
            return DecisionAssurance(
                verdict=DecisionVerdict.ABSTAIN,
                primary_diagnosis=primary_dx,
                confidence=round(top_prob, 4),
                conformal_set=conformal_set,
                conformal_coverage=coverage,
                conformal_method=method,
                ood_zscore=round(ood_energy, 4),
                is_ood=is_ood,
                epistemic_uncertainty=round(epistemic, 4),
                limiting_factor=limiting_factor or "safety_abstention",
                reasons=reasons,
                mitigations=list(dict.fromkeys(mitigations)),
                requires_human_review=True,
                model_version=self.model_version,
            )

        # -------------------------------------------------------------
        # 2. REVIEW_REQUIRED CRITERIA (Secondary clinical read needed)
        # -------------------------------------------------------------
        review_triggered = False

        if top_prob < self.min_accept_confidence:
            review_triggered = True
            reasons.append(
                f"Moderate confidence ({top_prob:.2f} in [{self.settings.low_confidence_threshold:.2f}, {self.min_accept_confidence:.2f}))"
            )
            if not limiting_factor:
                limiting_factor = "moderate_confidence"

        if conformal_size > self.max_accept_conformal_size:
            review_triggered = True
            reasons.append(
                f"Multiple competing hypotheses in conformal set ({conformal_size} candidates: {', '.join(conformal_set)})"
            )
            if not limiting_factor:
                limiting_factor = "competing_hypotheses"

        if ood_energy >= self.ood_warn_threshold:
            review_triggered = True
            reasons.append(
                f"Mild distribution shift warning (OOD energy z-score: {ood_energy:.2f} >= {self.ood_warn_threshold:.2f})"
            )
            if not limiting_factor:
                limiting_factor = "borderline_ood"

        if epistemic >= self.epistemic_warn_threshold:
            review_triggered = True
            reasons.append(
                f"Elevated ensemble disagreement (epistemic std: {epistemic:.2f} >= {self.epistemic_warn_threshold:.2f})"
            )
            if not limiting_factor:
                limiting_factor = "elevated_epistemic"

        if controller and controller.state == "WARNING":
            review_triggered = True
            reasons.append("Clinical safety controller issued safety warning")
            if not limiting_factor:
                limiting_factor = "controller_warning"

        if readiness and readiness.state != ReadinessState.READY:
            review_triggered = True
            reasons.append(
                f"Decision readiness profile state is '{readiness.state.value}' (limiting dimension: {readiness.limiting_factor})"
            )
            if not limiting_factor:
                limiting_factor = f"readiness_{readiness.limiting_factor}"

        if review_triggered:
            mitigations.extend([
                "secondary_radiologist_review",
                "correlate_clinical_labs",
                "evaluate_competing_hypotheses",
            ])
            return DecisionAssurance(
                verdict=DecisionVerdict.REVIEW_REQUIRED,
                primary_diagnosis=primary_dx,
                confidence=round(top_prob, 4),
                conformal_set=conformal_set,
                conformal_coverage=coverage,
                conformal_method=method,
                ood_zscore=round(ood_energy, 4),
                is_ood=is_ood,
                epistemic_uncertainty=round(epistemic, 4),
                limiting_factor=limiting_factor or "clinical_review",
                reasons=reasons,
                mitigations=list(dict.fromkeys(mitigations)),
                requires_human_review=True,
                model_version=self.model_version,
            )

        # -------------------------------------------------------------
        # 3. ACCEPT (High certainty, in-distribution, verified)
        # -------------------------------------------------------------
        reasons.append(
            "Diagnostic evidence satisfies all assurance criteria (high confidence, tight conformal set, in-distribution, low epistemic uncertainty)."
        )
        mitigations.extend([
            "routine_clinical_signoff",
            "standard_drift_monitoring",
        ])
        return DecisionAssurance(
            verdict=DecisionVerdict.ACCEPT,
            primary_diagnosis=primary_dx,
            confidence=round(top_prob, 4),
            conformal_set=conformal_set,
            conformal_coverage=coverage,
            conformal_method=method,
            ood_zscore=round(ood_energy, 4),
            is_ood=is_ood,
            epistemic_uncertainty=round(epistemic, 4),
            limiting_factor="none",
            reasons=reasons,
            mitigations=list(dict.fromkeys(mitigations)),
            requires_human_review=False,
            model_version=self.model_version,
        )
