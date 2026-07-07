"""ReportEngine — structured findings -> grounded clinician-style report."""
from __future__ import annotations

from schemas.clinical import (
    DIAGNOSIS_LABELS,
    FINDING_LABELS,
    Diagnosis,
    Finding,
)
from schemas.contracts import (
    AbstentionReason,
    Recommendation,
    ReportDraft,
    SafetyAssessment,
    VisionResult,
)

_ABSTENTION_TEXT = {
    AbstentionReason.OUT_OF_DISTRIBUTION:
        "Image appears outside the model's validated distribution; automated "
        "interpretation withheld pending human review.",
    AbstentionReason.LOW_CONFIDENCE:
        "No diagnosis reached the confidence threshold; findings are reported "
        "without a committed impression.",
    AbstentionReason.LARGE_CONFORMAL_SET:
        "Multiple diagnoses remain statistically plausible; a committed impression "
        "is deferred.",
    AbstentionReason.HIGH_EPISTEMIC:
        "Model uncertainty is high for this case; senior review recommended.",
}


class ReportEngine:
    def __init__(self):
        self.model_version = "report-v1"

    def compose(self, vision: VisionResult, safety: SafetyAssessment,
                recommendations: list[Recommendation]) -> ReportDraft:
        grounding: dict[str, list[str]] = {}

        # ---- Findings: only findings the vision engine actually asserted. ----
        positives = [f for f in vision.findings if f.probability >= 0.5]
        if positives:
            parts = []
            for fs in positives:
                label = FINDING_LABELS[fs.finding]
                parts.append(f"{label} (p={fs.probability:.2f})")
                grounding.setdefault("findings", []).append(fs.finding.value)
            findings_text = "Findings: " + "; ".join(parts) + "."
        else:
            findings_text = ("Findings: lungs are clear without focal consolidation, "
                             "effusion, or pneumothorax; cardiomediastinal silhouette "
                             "within normal limits.")
            grounding.setdefault("findings", []).append("no_positive_findings")

        # ---- Impression: driven by the safety-calibrated top diagnosis. ----
        if safety.abstained:
            impression_text = "Impression: " + _ABSTENTION_TEXT.get(
                safety.abstention_reason, "Automated impression withheld."
            )
            grounding["impression"] = ["abstained:" + safety.abstention_reason.value]
        else:
            label = DIAGNOSIS_LABELS[safety.top]
            conf = safety.top_probability
            dx_set = ", ".join(DIAGNOSIS_LABELS[d] for d in safety.conformal_set)
            impression_text = (
                f"Impression: findings most consistent with {label.lower()} "
                f"(calibrated probability {conf:.0%}). "
                f"{int(safety.conformal_coverage * 100)}% confidence set: {dx_set}."
            )
            grounding["impression"] = [safety.top.value] + [
                d.value for d in safety.conformal_set
            ]

        # ---- Recommendation: top value-of-information action. ----
        if recommendations:
            r = recommendations[0]
            recommendation_text = f"Recommendation: {r.display.lower()}. {r.rationale}"
            grounding["recommendation"] = [r.action]
        else:
            recommendation_text = ("Recommendation: no additional imaging indicated by "
                                   "information-gain analysis; correlate clinically.")
            grounding["recommendation"] = ["none"]

        return ReportDraft(
            study_id=vision.study_id,
            findings_text=findings_text,
            impression_text=impression_text,
            recommendation_text=recommendation_text,
            grounding=grounding,
            generator="structured+template",
        )
