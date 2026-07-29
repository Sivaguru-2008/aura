from __future__ import annotations
from typing import Any
from aura.schemas.contracts import CaseBundle
from aura.schemas.clinical import Diagnosis, Finding, DIAGNOSIS_LABELS, FINDING_LABELS

class SpecialistOpinion:
    """One specialist's view, produced by deterministic rules over the case.

    The prose is rule-selected, but ``confidence`` is *measured*: each specialist
    reports the model's own strength of evidence for the claim they are making —
    the calibrated diagnosis posterior for the specialists who argue a diagnosis
    (Pulmonology, Cardiology, Oncology), and the per-finding probability for the
    ones who report observations (Radiology, Neurology). Negative assertions are
    scored as ``1 - max(p)`` over what is being ruled out, so "the film is clear"
    is only confident when the model is confident nothing is there.

    ``confidence_basis`` says which happened: ``model`` when the number came off
    this case's own inference, ``rule_weight`` when the source was unavailable and
    a fixed fallback was used. A client must not present the two identically.
    """

    def __init__(self, specialist: str, opinion: str, confidence: float,
                 supporting_evidence: list[str], confidence_basis: str = "rule_weight"):
        self.specialist = specialist
        self.opinion = opinion
        self.confidence = confidence
        self.supporting_evidence = supporting_evidence
        self.confidence_basis = confidence_basis

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "opinion": self.opinion,
            "confidence": round(self.confidence, 4),
            "confidence_basis": self.confidence_basis,
            "supporting_evidence": self.supporting_evidence
        }


# --------------------------------------------------------------------------- #
# Confidence sources. Each returns ``(value, basis)`` so a missing source degrades
# to the documented fallback instead of silently inventing a number.
# --------------------------------------------------------------------------- #
def _calibrated_posterior(b: CaseBundle) -> dict[Diagnosis, float]:
    """The calibrated per-diagnosis posterior the safety engine emitted."""
    out: dict[Diagnosis, float] = {}
    preds = getattr(b.safety, "predictions", None) if b.safety else None
    for p in (preds or []):
        dx = p.diagnosis
        if not isinstance(dx, Diagnosis):
            try:
                dx = Diagnosis(dx)
            except ValueError:
                continue
        out[dx] = float(p.probability)
    return out


def _dx_conf(post: dict, dx: Diagnosis, fallback: float) -> tuple[float, str]:
    """Confidence in asserting ``dx`` — its calibrated posterior."""
    if dx in post:
        return float(post[dx]), "model"
    return fallback, "rule_weight"


def _not_dx_conf(post: dict, dxs: list[Diagnosis], fallback: float) -> tuple[float, str]:
    """Confidence in ruling ``dxs`` out — 1 minus the strongest of them."""
    vals = [post[d] for d in dxs if d in post]
    if vals:
        return float(1.0 - max(vals)), "model"
    return fallback, "rule_weight"


def _finding_conf(fmap: dict, findings: list[Finding], fallback: float) -> tuple[float, str]:
    """Confidence in asserting observations — the mean of their probabilities."""
    vals = [float(fmap[f]) for f in findings if f in fmap]
    if vals:
        return sum(vals) / len(vals), "model"
    return fallback, "rule_weight"


def _no_finding_conf(fmap: dict, findings: list[Finding], fallback: float) -> tuple[float, str]:
    """Confidence in calling a study clear — 1 minus the strongest finding."""
    vals = [float(fmap[f]) for f in findings if f in fmap]
    if vals:
        return float(1.0 - max(vals)), "model"
    return fallback, "rule_weight"


def _coalesce(*attempts: tuple[float, str], fallback: float) -> tuple[float, str]:
    """Take the first genuinely measured source; otherwise the stated fallback.

    Lets a branch prefer its most specific evidence (a per-finding probability)
    and still stay model-derived when that is absent. A normal brain study, for
    instance, segments nothing and so reports no findings at all — but the
    presence head still emits a calibrated posterior, and "no tumour" should be
    scored from that rather than from a constant.
    """
    for value, basis in attempts:
        if basis == "model":
            return value, basis
    return fallback, "rule_weight"


CHEST_FINDINGS = [Finding.CONSOLIDATION, Finding.CARDIOMEGALY, Finding.NODULE,
                  Finding.EFFUSION, Finding.PNEUMOTHORAX, Finding.OPACITY,
                  Finding.HYPERINFLATION]


class SpecialistDiscussionEngine:
    @staticmethod
    def discuss(b: CaseBundle) -> dict[str, Any]:
        # Determine if it's a brain MRI study or chest X-ray
        is_brain = b.study_id.startswith("STU-MR") or (b.fusion and b.fusion.backend == "brain-vision-presence-head")

        opinions: list[SpecialistOpinion] = []

        # Extract findings and check significance
        findings_map = {f.finding: f.probability for f in b.vision.findings} if b.vision else {}
        priors = b.priors
        multimodal = b.multimodal
        post = _calibrated_posterior(b)

        # 1. Radiologist
        if is_brain:
            has_tumor = findings_map.get(Finding.MASS_LESION, 0.0) >= 0.5
            vol = b.volumes.get("whole_tumor", {}).get("volume_mm3", 0.0) if b.volumes else 0.0
            edema_vol = b.volumes.get("edema", {}).get("volume_mm3", 0.0) if b.volumes else 0.0

            if has_tumor:
                opinion = (
                    f"A volumetric mass lesion is identified in the brain parenchyma with a whole tumor volume of "
                    f"{vol:.1f} mm³. The lesion shows enhancing components and peritumoral edema measured at {edema_vol:.1f} mm³."
                )
                conf, basis = _coalesce(
                    _finding_conf(findings_map, [Finding.MASS_LESION], 0.0),
                    _dx_conf(post, Diagnosis.BRAIN_TUMOR, 0.0),
                    fallback=0.8)
                evidence = ["segmented_whole_tumor", "segmented_edema"]
            else:
                opinion = "MRI sequences demonstrate no localized mass effect, midline shift, or pathologic enhancement. Ventricles are normal."
                conf, basis = _coalesce(
                    _not_dx_conf(post, [Diagnosis.BRAIN_TUMOR], 0.0),
                    _no_finding_conf(findings_map, [Finding.MASS_LESION, Finding.MIDLINE_SHIFT], 0.0),
                    fallback=0.95)
                evidence = []
            opinions.append(SpecialistOpinion("Radiologist", opinion, conf, evidence, basis))

        else: # Chest X-ray
            consol = findings_map.get(Finding.CONSOLIDATION, 0.0)
            cardiomegaly = findings_map.get(Finding.CARDIOMEGALY, 0.0)
            nodule = findings_map.get(Finding.NODULE, 0.0)
            effusion = findings_map.get(Finding.EFFUSION, 0.0)
            pneumo = findings_map.get(Finding.PNEUMOTHORAX, 0.0)

            ev = []
            obs = []
            asserted: list[Finding] = []
            if consol >= 0.4:
                obs.append("focal consolidation")
                ev.append("imaging.consolidation")
                asserted.append(Finding.CONSOLIDATION)
            if cardiomegaly >= 0.4:
                obs.append("cardiomegaly")
                ev.append("imaging.cardiomegaly")
                asserted.append(Finding.CARDIOMEGALY)
            if nodule >= 0.4:
                obs.append("pulmonary nodule")
                ev.append("imaging.nodule")
                asserted.append(Finding.NODULE)
            if effusion >= 0.4:
                obs.append("pleural effusion")
                ev.append("imaging.pleural_effusion")
                asserted.append(Finding.EFFUSION)
            if pneumo >= 0.4:
                obs.append("pneumothorax pleural edge")
                ev.append("imaging.pneumothorax")
                asserted.append(Finding.PNEUMOTHORAX)

            if obs:
                opinion = f"Chest radiograph reveals notable findings: {', '.join(obs)}. Lung volumes are otherwise clear."
                conf, basis = _finding_conf(findings_map, asserted, 0.85)
            else:
                opinion = "Chest radiograph shows no acute consolidations, pneumothorax, or cardiomegaly. Lungs are clear."
                conf, basis = _no_finding_conf(findings_map, CHEST_FINDINGS, 0.95)
            opinions.append(SpecialistOpinion("Radiologist", opinion, conf, ev, basis))

        # 2. Pulmonologist (Chest cases only)
        if not is_brain:
            fever = priors.fever or (multimodal and multimodal.symptoms.fever)
            cough = multimodal and multimodal.symptoms.productive_cough
            copd = multimodal and multimodal.history.copd
            consol = findings_map.get(Finding.CONSOLIDATION, 0.0)

            ev = []
            if fever: ev.append("symptoms.fever")
            if cough: ev.append("symptoms.productive_cough")
            if copd: ev.append("history.copd")

            if consol >= 0.5 and (fever or cough):
                opinion = "The patient presents with acute fever and productive cough. Combined with the focal consolidation on chest film, this strongly supports bacterial pneumonia."
                conf, basis = _dx_conf(post, Diagnosis.PNEUMONIA, 0.88)
            elif copd and findings_map.get(Finding.HYPERINFLATION, 0.0) >= 0.4:
                opinion = "Clinically diagnosed COPD is supported by the hyperinflation observed on imaging, consistent with obstructive airway disease."
                conf, basis = _dx_conf(post, Diagnosis.COPD, 0.82)
                ev.append("imaging.hyperinflation")
            else:
                opinion = "No significant pulmonary symptoms or history that suggest active pneumonia or COPD flare."
                conf, basis = _not_dx_conf(post, [Diagnosis.PNEUMONIA, Diagnosis.COPD], 0.70)
            opinions.append(SpecialistOpinion("Pulmonologist", opinion, conf, ev, basis))

        # 3. Cardiologist (Chest cases only)
        if not is_brain:
            bnp = multimodal and multimodal.labs.bnp
            orthopnea = multimodal and multimodal.symptoms.orthopnea
            cardiomegaly = findings_map.get(Finding.CARDIOMEGALY, 0.0)
            effusion = findings_map.get(Finding.EFFUSION, 0.0)

            ev = []
            if bnp is not None: ev.append("labs.bnp")
            if orthopnea: ev.append("symptoms.orthopnea")
            if cardiomegaly >= 0.4: ev.append("imaging.cardiomegaly")
            if effusion >= 0.4: ev.append("imaging.pleural_effusion")

            if bnp is not None and bnp >= 400:
                opinion = f"Markedly elevated BNP of {bnp:.0f} pg/mL and orthopnea suggest cardiac decompensation. The pleural effusion and cardiomegaly support heart failure."
                conf, basis = _dx_conf(post, Diagnosis.HEART_FAILURE, 0.92)
            elif cardiomegaly >= 0.5:
                opinion = "Cardiomegaly is present on imaging. In the absence of highly elevated BNP, we should rule out cardiomyopathy but decompensated heart failure is less likely."
                conf, basis = _dx_conf(post, Diagnosis.HEART_FAILURE, 0.75)
            else:
                opinion = "Cardiac findings are negative. Low probability of active heart failure."
                conf, basis = _not_dx_conf(post, [Diagnosis.HEART_FAILURE], 0.80)
            opinions.append(SpecialistOpinion("Cardiologist", opinion, conf, ev, basis))

        # 4. Neurologist (Brain cases only)
        if is_brain:
            has_tumor = findings_map.get(Finding.MASS_LESION, 0.0) >= 0.5
            shift = findings_map.get(Finding.MIDLINE_SHIFT, 0.0) >= 0.5

            ev = []
            if has_tumor: ev.append("imaging.mass_lesion")
            if shift: ev.append("imaging.midline_shift")

            if has_tumor:
                if shift:
                    opinion = "Intracranial mass lesion is associated with a midline shift. There is high risk of mass effect and herniation. Recommend urgent neurosurgical decompression."
                    conf, basis = _finding_conf(findings_map, [Finding.MIDLINE_SHIFT], 0.95)
                else:
                    opinion = "A distinct mass lesion is present, but without significant midline shift. We must rule out a primary neuroepithelial tumor. Brain parenchyma otherwise intact."
                    conf, basis = _coalesce(
                        _finding_conf(findings_map, [Finding.MASS_LESION], 0.0),
                        _dx_conf(post, Diagnosis.BRAIN_TUMOR, 0.0),
                        fallback=0.88)
            else:
                opinion = "Neurological imaging is unremarkable. No signs of vascular, infectious, or demyelinating processes."
                conf, basis = _coalesce(
                    _not_dx_conf(post, [Diagnosis.BRAIN_TUMOR], 0.0),
                    _no_finding_conf(findings_map, [Finding.MASS_LESION], 0.0),
                    fallback=0.90)
            opinions.append(SpecialistOpinion("Neurologist", opinion, conf, ev, basis))

        # 5. Oncologist
        if is_brain:
            has_tumor = findings_map.get(Finding.MASS_LESION, 0.0) >= 0.5
            subtypes = b.volumes.get("tumor_subtypes") if b.volumes else None

            ev = ["imaging.mass_lesion"]
            if has_tumor:
                # The subtype claim is only made when the engine explicitly says the
                # subtype classifier ran. Defaulting this to True would let a payload
                # that merely *carries* a subtype dict assert a subtype the served
                # brain model cannot distinguish (it is glioma-only — see the
                # Diagnosis enum note), so the default is False.
                if subtypes and subtypes.get("qkl_enabled", False):
                    top_subtype = max(["glioma", "meningioma", "metastasis"], key=lambda k: subtypes.get(k, 0.0))
                    prob = subtypes.get(top_subtype, 0.0)
                    opinion = (
                        f"The mass lesion features are highly suspicious for malignancy. Quantum Kernel Classifier (QKL) "
                        f"indicates {top_subtype.upper()} as the most likely subtype ({prob*100:.1f}% confidence). Recommend biopsy."
                    )
                    conf, basis = float(prob), "model"
                else:
                    opinion = "Intracranial mass lesion requires histopathological verification to rule out high-grade glioma vs. solitary metastasis."
                    conf, basis = _coalesce(
                        _finding_conf(findings_map, [Finding.MASS_LESION], 0.0),
                        _dx_conf(post, Diagnosis.BRAIN_TUMOR, 0.0),
                        fallback=0.85)
            else:
                opinion = "No neoplastic mass detected. Low suspicion for primary or secondary intracranial malignancy."
                conf, basis = _coalesce(
                    _not_dx_conf(post, [Diagnosis.BRAIN_TUMOR], 0.0),
                    _no_finding_conf(findings_map, [Finding.MASS_LESION], 0.0),
                    fallback=0.95)
            opinions.append(SpecialistOpinion("Oncologist", opinion, conf, ev, basis))

        else: # Chest
            nodule = findings_map.get(Finding.NODULE, 0.0)
            smoker = priors.smoker or (multimodal and (multimodal.history.smoking_pack_years or 0.0) >= 20)
            prior_cancer = multimodal and multimodal.history.prior_cancer

            ev = []
            if nodule >= 0.4: ev.append("imaging.nodule")
            if smoker: ev.append("priors.smoker")
            if prior_cancer: ev.append("history.prior_cancer")

            if nodule >= 0.5:
                if smoker or prior_cancer:
                    opinion = f"A pulmonary nodule of {nodule*100:.1f}% probability in a high-risk patient (smoking/prior malignancy) is suspicious for primary bronchogenic carcinoma. Fleischner criteria follow-up recommended."
                    conf, basis = _dx_conf(post, Diagnosis.MALIGNANCY, 0.88)
                else:
                    opinion = "Solitary pulmonary nodule detected in a low-risk patient. Follow-up CT scan in 6 months recommended to check for stability."
                    conf, basis = _dx_conf(post, Diagnosis.MALIGNANCY, 0.75)
            else:
                opinion = "No pulmonary nodules or masses suspicious for neoplastic process."
                conf, basis = _not_dx_conf(post, [Diagnosis.MALIGNANCY], 0.90)
            opinions.append(SpecialistOpinion("Oncologist", opinion, conf, ev, basis))

        # Consensus Engine
        # Combine the expert opinions into a final recommendation. The consensus
        # confidence is the mean of the contributing specialists', so it inherits
        # their basis: it is only "model" when every contributor was measured.
        consensus_opinion = ""
        consensus_confidence = 0.0

        top_dx = b.safety.top if (b.safety and b.safety.top) else None
        top_dx_label = DIAGNOSIS_LABELS.get(top_dx, str(top_dx)) if top_dx else "Unknown"

        def _mean(contributors: list[SpecialistOpinion]) -> tuple[float, str]:
            vals = [o.confidence for o in contributors]
            basis = "model" if all(o.confidence_basis == "model" for o in contributors) else "rule_weight"
            return (sum(vals) / len(vals) if vals else 0.0), basis

        if is_brain:
            has_tumor = findings_map.get(Finding.MASS_LESION, 0.0) >= 0.5
            rad = next(o for o in opinions if o.specialist == "Radiologist")
            neuro = next(o for o in opinions if o.specialist == "Neurologist")
            onc = next(o for o in opinions if o.specialist == "Oncologist")

            if has_tumor:
                consensus_opinion = (
                    f"Consensus reached: High suspicion of active intracranial tumor. Radiology and Oncology agree on volumetric "
                    f"findings. Neurology recommends surgical and ICP management. Recommend neurosurgical consult for biopsy."
                )
            else:
                consensus_opinion = "Consensus reached: Normal brain study. Low probability of structural lesion."
            consensus_confidence, consensus_basis = _mean([rad, neuro, onc])
        else:
            # Chest consensus
            rad = next(o for o in opinions if o.specialist == "Radiologist")
            pul = next(o for o in opinions if o.specialist == "Pulmonologist")
            card = next(o for o in opinions if o.specialist == "Cardiologist")
            onc = next(o for o in opinions if o.specialist == "Oncologist")

            # Find diagnosis from bundle safety
            if top_dx == Diagnosis.PNEUMONIA:
                consensus_opinion = "Consensus reached: Findings are highly consistent with acute bacterial Pneumonia. Pulmonology and Radiology recommend antibiotic therapy and clinical monitoring."
                consensus_confidence, consensus_basis = _mean([rad, pul])
            elif top_dx == Diagnosis.HEART_FAILURE:
                consensus_opinion = "Consensus reached: Congestive Heart Failure is the primary working diagnosis, supported by clinical history, BNP, and pleural effusion. Cardiology recommends diuresis."
                consensus_confidence, consensus_basis = _mean([rad, card])
            elif top_dx == Diagnosis.MALIGNANCY:
                consensus_opinion = "Consensus reached: Suspicious pulmonary nodule in high-risk patient. Oncology recommends urgent thoracic CT scan and biopsy consultation."
                consensus_confidence, consensus_basis = _mean([rad, onc])
            elif top_dx == Diagnosis.COPD:
                consensus_opinion = "Consensus reached: Obstructive pulmonary disease flare. Pulmonology recommends bronchodilators and steroid management."
                consensus_confidence, consensus_basis = _mean([rad, pul])
            else:
                consensus_opinion = "Consensus reached: Negative study. No acute cardiopulmonary abnormality detected. Reassure patient."
                consensus_confidence, consensus_basis = _mean([rad, pul, card, onc])

        return {
            "case_id": b.case_id,
            "opinions": [o.to_dict() for o in opinions],
            "consensus": {
                "opinion": consensus_opinion,
                "confidence": round(consensus_confidence, 4),
                "confidence_basis": consensus_basis
            }
        }
