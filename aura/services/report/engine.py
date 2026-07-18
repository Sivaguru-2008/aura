"""ReportEngine — structured findings -> grounded clinician-style report.

Every sentence is traceable: the ``grounding`` map ties each section to the
evidence nodes (imaging findings, calibrated diagnoses, reasoning steps, guideline
citations, recommended actions) that produced it. Beyond the impression it now
narrates a **differential diagnosis** (from the clinical reasoner, each alternative
with its supporting/opposing evidence) and an explicit **confidence explanation**
(calibrated probability, conformal set, and the aleatoric/epistemic split).
"""
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
    ReasoningTrace,
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

# Human phrasing for evidence node ids used in reasoning steps.
_EV_LABEL = {
    "labs.bnp": "BNP", "labs.wbc": "white-cell count", "labs.crp": "CRP",
    "labs.procalcitonin": "procalcitonin", "labs.spo2": "oxygen saturation",
    "history.smoking_pack_years": "smoking history", "history.prior_cancer": "prior malignancy",
    "history.copd": "COPD history", "history.immunosuppression": "immunosuppression",
    "symptoms.fever": "fever", "symptoms.orthopnea": "orthopnoea",
    "symptoms.hemoptysis": "haemoptysis", "symptoms.pleuritic_chest_pain": "pleuritic chest pain",
    "priors.age_band": "age",
}


def _pretty_ev(ids: list[str]) -> str:
    seen, out = set(), []
    for e in ids:
        lbl = _EV_LABEL.get(e, e.split(".")[-1].replace("_", " "))
        if lbl not in seen:
            seen.add(lbl); out.append(lbl)
    return ", ".join(out)


class ReportEngine:
    def __init__(self):
        self.model_version = "report-v2"

    def compose(self, vision: VisionResult, safety: SafetyAssessment,
                recommendations: list[Recommendation],
                reasoning: ReasoningTrace | None = None) -> ReportDraft:
        grounding: dict[str, list[str]] = {}

        # ---- Findings: only findings the vision engine actually asserted. ----
        positives = [f for f in vision.findings if f.probability >= 0.5]
        if positives:
            parts = []
            for fs in positives:
                parts.append(f"{FINDING_LABELS[fs.finding]} (p={fs.probability:.2f})")
                grounding.setdefault("findings", []).append(fs.finding.value)
            findings_text = "Findings: " + "; ".join(parts) + "."
        else:
            findings_text = ("Findings: lungs are clear without focal consolidation, "
                             "effusion, or pneumothorax; cardiomediastinal silhouette "
                             "within normal limits.")
            grounding.setdefault("findings", []).append("no_positive_findings")

        # ---- Impression: safety-calibrated top diagnosis. ----
        if safety.abstained:
            impression_text = "Impression: " + _ABSTENTION_TEXT.get(
                safety.abstention_reason, "Automated impression withheld.")
            grounding["impression"] = ["abstained:" + safety.abstention_reason.value]
        else:
            label = DIAGNOSIS_LABELS[safety.top]
            dx_set = ", ".join(DIAGNOSIS_LABELS[d] for d in safety.conformal_set)
            impression_text = (
                f"Impression: findings most consistent with {label.lower()} "
                f"(calibrated probability {safety.top_probability:.0%}). "
                f"{int(safety.conformal_coverage * 100)}% {safety.conformal_method} "
                f"confidence set: {dx_set}.")
            grounding["impression"] = [safety.top.value] + [d.value for d in safety.conformal_set]

        # ---- Differential: alternatives with their supporting/opposing evidence. ----
        differential_text = ""
        if reasoning and reasoning.differential:
            lines = []
            for item in reasoning.differential:
                lbl = DIAGNOSIS_LABELS[item.diagnosis]
                seg = f"{lbl} {item.probability:.0%}"
                if item.supporting:
                    seg += f" — supported by {_pretty_ev(item.supporting)}"
                if item.opposing:
                    seg += f"; against: {_pretty_ev(item.opposing)}"
                lines.append(seg)
                grounding.setdefault("differential", []).append(item.diagnosis.value)
            differential_text = "Differential: " + "; ".join(lines) + "."
            if reasoning.guideline_citations:
                differential_text += (" Reasoning applied "
                                      + ", ".join(reasoning.guideline_citations) + ".")
                grounding["differential_guidelines"] = list(reasoning.guideline_citations)

        # ---- Confidence explanation: calibrated prob + uncertainty decomposition. ----
        cu = safety
        drivers = []
        if cu.uncertainty_method == "deep_ensemble":
            drivers.append(f"{cu.n_ensemble}-member deep ensemble")
        drivers.append(f"epistemic (model) {cu.epistemic_uncertainty:.2f}")
        drivers.append(f"aleatoric (data) {cu.aleatoric_uncertainty:.2f}")
        if cu.epistemic_mi:
            drivers.append(f"mutual information {cu.epistemic_mi:.2f} bits")
        confidence_text = (
            f"Confidence: top calibrated probability {cu.top_probability:.0%} with a "
            f"{len(cu.conformal_set)}-label {cu.conformal_method} conformal set at "
            f"{int(cu.conformal_coverage*100)}% coverage; uncertainty via "
            + ", ".join(drivers) + ".")
        if cu.is_ood:
            confidence_text += " Input flagged out-of-distribution."
        grounding["confidence"] = [cu.uncertainty_method, cu.conformal_method]

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
            differential_text=differential_text,
            confidence_text=confidence_text,
            grounding=grounding,
            generator="structured+reasoning",
        )
