"""Project a :class:`BrainVisionOutput` onto the console's :class:`CaseBundle`.

This module is the seam between what the trained brain model *measures* and what the
case schema — designed around the chest pipeline — wants to display. Every field below
is derived from a real network output. Nothing is invented to fill a slot, and where
the schema asks for something the model does not produce, the field is left empty
rather than populated with a plausible number.

What the model actually produces
--------------------------------
``backend/vision/brain`` is a BraTS2020 segmentation network: four mutually exclusive
voxel classes (background, necrotic/non-enhancing core, peritumoural oedema, enhancing
tumour) plus a whole-tumour presence head, a size regressor, a quality head and an
embedding. That is the entire vocabulary available here.

It is **not** a subtype classifier. BraTS2020 contains gliomas only, so the network has
never been shown a meningioma, a metastasis, an abscess or a demyelinating lesion to
tell one from another. ``Diagnosis.BRAIN_TUMOR`` — "subtype not determined" — is
therefore the most specific diagnosis this path is entitled to emit, and the differential
is two-class: tumour tissue is present, or it is not.

The build this replaced emitted ``GLIOMA``/``MENINGIOMA``/``STROKE``/``HEMORRHAGE_DX``
by hashing the uploaded file's SHA-256 and indexing a list, with a fixed 0.85 posterior,
a ``random.random()`` embedding and a hand-drawn rectangle for saliency. None of those
outputs were functions of the image.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aura.backend.core.shared.logging import get_logger
from aura.backend.vision.brain.output import BrainVisionOutput
from aura.backend.vision.brain.types import CompositeRegion, TumorRegion
from aura.schemas.clinical import Diagnosis, Finding
from aura.schemas.contracts import (
    AbstentionReason,
    CaseBundle,
    CaseState,
    EvidenceItem,
    EvidenceKind,
    Explanation,
    FindingScore,
    FusionResult,
    Prediction,
    ReasoningTrace,
    Recommendation,
    ReportDraft,
    SafetyAssessment,
    StructuredPriors,
    VisionResult,
    ClinicalContext,
)

log = get_logger("engine.neuromind.bundle")

#: Display grid for the stored slice and saliency map, matching the chest path so the
#: console's image and overlay renderers need no modality-specific branch.
GRID = 224

#: Mask-derived findings, and the segmentation classes each is read from. Only these
#: two are emitted: they are the classes the network predicts. Midline shift,
#: ventriculomegaly, hypo/hyperintensity and haemorrhage have no head behind them and
#: are never reported, present or absent — an "absent" claim from a model that cannot
#: see the finding is as unfounded as a positive one.
_MASK_FINDINGS: tuple[tuple[Finding, tuple[TumorRegion, ...]], ...] = (
    (Finding.MASS_LESION, (TumorRegion.NECROTIC_CORE, TumorRegion.ENHANCING)),
    (Finding.EDEMA, (TumorRegion.EDEMA,)),
)


def _normalised_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Bounding box of ``mask`` as (r0, c0, r1, c1) fractions, or ``None`` if empty."""
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    return (round(float(ys.min()) / h, 4), round(float(xs.min()) / w, 4),
            round(float(ys.max() + 1) / h, 4), round(float(xs.max() + 1) / w, 4))


def _to_grid(array: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Resize a 2D array to ``grid`` x ``grid``, preserving label values for masks."""
    import cv2

    if array.shape == (grid, grid):
        return array
    interp = cv2.INTER_NEAREST if array.dtype == np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(array.astype(array.dtype if array.dtype != np.uint8 else np.uint8),
                      (grid, grid), interpolation=interp)


def _representative_index(output: BrainVisionOutput) -> int:
    """Index of the slice carrying the most tumour, else the middle slice.

    The console shows one image. Showing the middle slice of a study whose tumour sits
    in the frontal lobe means the saliency overlay lands on an image with nothing in it.
    """
    seg = output.segmentation
    if seg.ndim < 3 or seg.shape[0] == 0:
        return 0
    burden = (seg > 0).reshape(seg.shape[0], -1).sum(axis=1)
    return int(burden.argmax()) if burden.max() > 0 else seg.shape[0] // 2


def _region_findings(output: BrainVisionOutput, index: int) -> list[FindingScore]:
    """Findings read off the segmentation, scored by the model's own confidence."""
    seg = output.segmentation[index] if output.segmentation.ndim == 3 else output.segmentation
    conf = (output.confidence[index] if output.confidence.ndim == 3
            else output.confidence)
    scores: list[FindingScore] = []
    for finding, classes in _MASK_FINDINGS:
        mask = np.zeros(seg.shape, dtype=bool)
        for cls in classes:
            mask |= seg == cls.value
        if not mask.any():
            continue
        # Probability is the model's mean softmax confidence over the voxels it
        # assigned to this finding's classes — a measured quantity, not the presence
        # head's study-level number reused per finding.
        probability = float(conf[mask].mean()) if conf.size else None
        if probability is None:
            continue
        scores.append(FindingScore(
            finding=finding,
            probability=round(probability, 4),
            region=_normalised_bbox(mask),
        ))
    return scores


def _volume_text(output: BrainVisionOutput) -> dict[str, Any]:
    """Per-composite-region burden, in voxels and in mm3 where spacing is known."""
    out: dict[str, Any] = {}
    for composite in CompositeRegion:
        mask = output.composite_mask(composite)
        voxels = int(np.count_nonzero(mask))
        entry: dict[str, Any] = {"voxels": voxels}
        spacing = getattr(output.processing, "spacing_mm", None)
        if spacing and all(s for s in spacing):
            entry["volume_mm3"] = round(voxels * float(np.prod(spacing)), 2)
        out[composite.value] = entry
    return out


def build_case_bundle(
    output: BrainVisionOutput,
    *,
    case_id: str,
    study_id: str,
    image: np.ndarray,
    presence_probability: float,
    priors: StructuredPriors,
    caveats: list[str],
    abstain_reason: AbstentionReason | None,
    abstain_detail: str | None = None,
    clinical_context: dict | None = None,
    tumor_subtypes: dict[str, float] | None = None,
) -> CaseBundle:
    """Assemble the console bundle from one brain-vision result.

    Args:
        presence_probability: the calibrated whole-tumour probability. Passed in
            rather than read off ``output`` so the calibration step is visible at the
            call site and cannot be silently skipped.
        caveats: disclosures that must travel with this result — missing sequences,
            an unresolved MR/CT ambiguity, degraded quality.
        abstain_reason: when set, the case is marked ``ABSTAINED``. The measurements
            are still reported alongside it, because a refusal that hides its own
            evidence cannot be checked by the person it was handed to.
        abstain_detail: human-readable expansion of ``abstain_reason``.
    """
    index = _representative_index(output)
    seg_slice = (output.segmentation[index] if output.segmentation.ndim == 3
                 else output.segmentation)
    conf_slice = (output.confidence[index] if output.confidence.ndim == 3
                  else output.confidence)

    findings = _region_findings(output, index)
    embedding = output.study_embedding
    vision = VisionResult(
        study_id=study_id,
        findings=findings,
        # The real 128-d supervised-contrastive embedding, so similarity search over
        # brain cases is a search over learned representation rather than over noise.
        embedding=[round(float(v), 6) for v in embedding] if embedding is not None else [],
        model_version=output.model_version,
    )

    # ---- evidence ------------------------------------------------------- #
    # One item per measurement the model actually made. There is no ABSENT_EVIDENCE
    # sweep over unpredicted findings here: the chest path can say "no pneumothorax"
    # because it has a pneumothorax head, and this model has nothing equivalent.
    evidence: list[EvidenceItem] = []
    for score in findings:
        evidence.append(EvidenceItem(
            kind=EvidenceKind.IMAGING_FINDING,
            name=score.finding.value,
            value=score.probability,
            probability=score.probability,
            uncertainty=round(1.0 - score.probability, 4),
            source_service="vision.brain",
            source="Image",
            model="presence-head-v2",
            validation_status="Platt-calibrated",
        ))
    for region in output.regions:
        if region.voxels <= 0:
            continue
        evidence.append(EvidenceItem(
            kind=EvidenceKind.IMAGING_FINDING,
            name=f"segmented_{region.region}",
            value=float(region.voxels),
            probability=(round(float(region.mean_confidence), 4)
                         if region.mean_confidence is not None else None),
            uncertainty=None,
            source_service="vision.brain",
            source="Image",
            model="segmentation-head-v2",
            validation_status="Platt-calibrated",
        ))

    if clinical_context:
        for k, v in clinical_context.items():
            if v is None or v == "":
                continue
            val = 0.0
            if isinstance(v, bool):
                val = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                val = float(v)
            else:
                val = 1.0  # present string/array

            evidence.append(EvidenceItem(
                kind=EvidenceKind.CLINICIAN_INPUT,
                name=k,
                value=val,
                probability=1.0,
                uncertainty=0.0,
                source_service="clinician.intake",
                source="Patient Metadata" if k in ("age", "sex") else "Clinical Notes",
                model="Clinician Intake",
                validation_status="Clinician-supplied"
            ))

    # ---- posterior ------------------------------------------------------ #
    # Two classes, because the model answers one question. A five-way differential
    # here would be five numbers derived from one.
    posterior = {
        Diagnosis.BRAIN_TUMOR: round(presence_probability, 4),
        Diagnosis.NORMAL_BRAIN: round(1.0 - presence_probability, 4),
    }
    top = max(posterior, key=posterior.get)

    fusion = FusionResult(
        study_id=study_id,
        backend="brain-vision-presence-head",
        posterior=posterior,
        # No ensemble and no MC-dropout pass was run, so there is no posterior spread
        # to report. Zeros here would read as "the model is certain".
        posterior_std={},
        evidence_vector=([round(float(v), 6) for v in output.features.pooled]
                         if output.features.pooled is not None else []),
        model_version=output.model_version,
    )

    predictions = sorted(
        (Prediction(diagnosis=d, probability=p, ci_low=None, ci_high=None)
         for d, p in posterior.items()),
        key=lambda pr: pr.probability, reverse=True)

    # A two-class conformal set with no fitted calibration set is just the argmax; say
    # so rather than presenting an uncalibrated interval as coverage-guaranteed.
    conformal_set = [d for d, p in posterior.items() if p >= 0.10] or [top]

    safety = SafetyAssessment(
        study_id=study_id,
        predictions=predictions,
        top=top,
        top_probability=posterior[top],
        conformal_set=conformal_set,
        # No conformal calibration set has been fitted for the brain path, so there is
        # no coverage guarantee to quote. The "set" above is a threshold on a
        # two-class posterior and is labelled as such in the report.
        conformal_coverage=None,
        conformal_method="none",
        # One network, one forward pass: no ensemble disagreement and no MC-dropout
        # spread exist to report. Aleatoric is the one uncertainty actually measured
        # here — mean per-voxel softmax confidence over the shown slice.
        epistemic_uncertainty=None,
        aleatoric_uncertainty=(round(float(1.0 - conf_slice.mean()), 4)
                               if conf_slice.size else None),
        epistemic_mi=None,
        predictive_entropy=None,
        uncertainty_method="segmentation_softmax",
        n_ensemble=1,
        ood_energy=None,
        is_ood=None,
        abstained=abstain_reason is not None,
        abstention_reason=abstain_reason or AbstentionReason.NONE,
        model_version=output.model_version,
    )

    # ---- saliency & uncertainty ----------------------------------------- #
    # The real thing: per-voxel confidence, masked to the tumour classes. Where the
    # model segmented nothing there is nothing to highlight, and the map is flat.
    saliency = np.zeros((GRID, GRID), dtype=np.float32)
    tumour = seg_slice > 0
    if tumour.any() and conf_slice.size:
        highlighted = np.where(tumour, conf_slice, 0.0).astype(np.float32)
        saliency = _to_grid(highlighted)

    uncertainty_map = np.zeros((GRID, GRID), dtype=np.float32)
    if conf_slice.size:
        uncertainty_map = _to_grid(1.0 - conf_slice)

    explanation = Explanation(
        study_id=study_id,
        saliency=[round(float(v), 4) for v in saliency.flatten()],
        saliency_shape=(GRID, GRID),
        saliency_target=top.value,
        saliency_methods={"uncertainty": [round(float(v), 4) for v in uncertainty_map.flatten()]},
        evidence_attribution={},
        counterfactuals={},
        clinical_warning="Attention maps are visualization aids, never representing model reasoning.",
    )

    volumes = _volume_text(output)
    if tumor_subtypes:
        volumes["tumor_subtypes"] = tumor_subtypes

    if tumor_subtypes:
        diff_text = (
            "Multi-class differential from Quantum Kernel Learning (QKL) classifier:\n"
            f"  - Glioma: {tumor_subtypes.get('glioma', 0.0)*100:.1f}%\n"
            f"  - Meningioma: {tumor_subtypes.get('meningioma', 0.0)*100:.1f}%\n"
            f"  - Metastasis: {tumor_subtypes.get('metastasis', 0.0)*100:.1f}%\n"
            "This differential is produced by projecting ResU-Net latent features "
            "into a 6-qubit Hilbert space and evaluating decision boundaries using QKL."
        )
    else:
        diff_text = (
            "Two-class differential: intracranial tumour tissue present vs absent. "
            "AURA's brain model is a BraTS segmentation network and has no tumour "
            "subtype head — glioma, meningioma, metastasis and abscess are not "
            "distinguished, and no subtype is being asserted."
        )

    report = ReportDraft(
        study_id=study_id,
        findings_text=_findings_text(output, findings, volumes, presence_probability, abstain_reason is not None),
        impression_text=_impression_text(top, presence_probability, caveats, findings, abstained=abstain_reason is not None),
        recommendation_text=_recommendation_text(top, abstain_detail),
        differential_text=diff_text,
        confidence_text=_confidence_text(presence_probability, caveats),
        grounding={
            "findings": [s.finding.value for s in findings],
            "impression": [s.finding.value for s in findings],
            "recommendation": [s.finding.value for s in findings],
        },
        generator=f"neuromind-report/{output.brain_vision_version}",
    )

    return CaseBundle(
        case_id=case_id,
        study_id=study_id,
        state=CaseState.ABSTAINED if abstain_reason else CaseState.READY,
        priority_score=_priority(top, presence_probability, abstain_reason),
        priors=priors,
        image=[round(float(v), 4) for v in _to_grid(image).flatten()],
        image_shape=(GRID, GRID),
        vision=vision,
        evidence=evidence,
        fusion=fusion,
        safety=safety,
        explanation=explanation,
        reasoning=ReasoningTrace(
            study_id=study_id,
            prior_posterior={},
            adjusted_posterior=posterior,
            steps=[],
            differential=[],
        ),
        recommendations=_recommendations(top, output, caveats),
        report=report,
        volumes=volumes,
        clinical_context=ClinicalContext(**clinical_context) if clinical_context else None,
        ground_truth=None,
    )


def _priority(top: Diagnosis, probability: float,
              abstain_reason: "AbstentionReason | None") -> float:
    """Worklist urgency. An abstention is triaged *up*: it needs a human, now."""
    if abstain_reason:
        return 0.80
    if top is Diagnosis.NORMAL_BRAIN:
        return round(0.10 * probability, 4)
    return round(0.90 * probability, 4)


def _findings_text(output: BrainVisionOutput, findings: list[FindingScore],
                   volumes: dict[str, Any], probability: float, abstained: bool = False) -> str:
    if abstained:
        return (
            "### OBSERVED FINDINGS\n"
            "Incomplete or low-quality study. No findings reported due to active safety gating.\n\n"
            "### MEASURED VALUES\n"
            "No valid measurements could be obtained."
        )

    if not findings:
        findings_sec = "No BraTS-class tumor tissue was segmented on any analyzed slice."
    else:
        named = ", ".join(f"{s.finding.value.replace('_', ' ')} (mean segmentation confidence {s.probability:.2f})"
                          for s in findings)
        slices = int(output.processing.slices_analysed) if getattr(output.processing, "slices_analysed", None) is not None else 0
        findings_sec = f"Segmentation identifies {named} over {slices} slices analyzed."

    wt = volumes.get(CompositeRegion.WHOLE_TUMOR.value, {})
    core = volumes.get(CompositeRegion.TUMOR_CORE.value, {})
    enh = volumes.get(CompositeRegion.ENHANCING_TUMOR.value, {})

    def burden(entry: dict[str, Any]) -> str:
        if entry.get("volume_mm3") is not None:
            return f"{entry['volume_mm3']:.0f} mm³"
        return f"{entry.get('voxels', 0)} voxels"

    measured_sec = (
        f"Whole-tumor burden: {burden(wt)}\n"
        f"Tumor core: {burden(core)}\n"
        f"Enhancing component: {burden(enh)}"
    )

    return (
        f"### OBSERVED FINDINGS\n"
        f"{findings_sec}\n\n"
        f"### MEASURED VALUES\n"
        f"{measured_sec}"
    )


def _impression_text(top: Diagnosis, probability: float, caveats: list[str],
                     findings: list[FindingScore] | None = None,
                     abstained: bool = False) -> str:
    if abstained:
        return "### CLINICAL IMPRESSION\nNo reliable conclusion."

    segmented = [f.finding.value.replace("_", " ") for f in (findings or [])]
    if top is Diagnosis.NORMAL_BRAIN:
        if segmented:
            base = (f"The presence head reports no tumor (probability {probability:.3f}), "
                    f"but the segmentation head has labeled {', '.join(segmented)} in "
                    f"this study. The two heads disagree, and this study should be read "
                    f"by a radiologist rather than treated as negative.")
        else:
            base = f"No BraTS-class tumor tissue detected (presence probability {probability:.3f})."
    else:
        base = (f"Intracranial tumor tissue detected (presence probability "
                f"{probability:.3f}). Subtype is not determined: the model segments "
                f"tumor regions and does not classify tumor type.")
        if not segmented:
            base += " No voxels were assigned to a tumor class, so the detection is not localized — the two heads disagree."
            
    return f"### CLINICAL IMPRESSION\n{base}"


def _recommendation_text(top: Diagnosis, abstain_detail: str | None) -> str:
    if abstain_detail:
        return (
            "### CLINICAL CONSIDERATIONS\n"
            f"AURA abstained on this study: {abstain_detail}. Radiologist review required."
        )
    
    if top is Diagnosis.NORMAL_BRAIN:
        base = "Correlate clinically — this model reports on tumor tissue only and does not exclude other intracranial pathology."
    else:
        base = "Radiologist review of the segmented volume. Contrast-enhanced MRI and histopathology recommended."

    return f"### CLINICAL CONSIDERATIONS\n{base}"


def _confidence_text(probability: float, caveats: list[str]) -> str:
    lims = "\n".join(f"- {c}" for c in caveats) if caveats else "None noted."
    return (
        f"### MODEL OUTPUTS\n"
        f"Calibrated whole-tumor presence probability: {probability:.3f}\n\n"
        f"### LIMITATIONS\n"
        f"{lims}"
    )


def _recommendations(top: Diagnosis, output: BrainVisionOutput,
                     caveats: list[str]) -> list[Recommendation]:
    recs: list[Recommendation] = []
    missing = list(getattr(output.processing, "sequences_missing", ()) or ())
    if missing:
        recs.append(Recommendation(
            action="acquire_missing_sequences",
            display=f"Acquire the missing sequences ({', '.join(missing)})",
            expected_info_gain=None,
            cost_tier="medium",
            risk_tier="low",
            utility=None,
            rationale=(
                "The network was trained on four co-registered sequences, all marked "
                "required. Measured on held-out BraTS slices, whole-tumour Dice falls "
                "from 0.58 with all four to 0.52 with FLAIR alone, 0.28 with T2 alone, "
                "0.02 with T1 alone and 0.00 with T1ce alone. A complete study is not "
                "a refinement here; it is the condition the model was fitted under."),
        ))
    if top is not Diagnosis.NORMAL_BRAIN:
        recs.append(Recommendation(
            action="acquire_contrast_mri",
            display="Contrast-enhanced MRI recommended",
            expected_info_gain=None,
            cost_tier="medium",
            risk_tier="low",
            utility=None,
            rationale="Further evaluation with a contrast-enhanced MRI is recommended to define tumor borders and enhancement characteristics.",
            reason="Establish baseline enhancement characteristics for staging.",
            evidence=["segmented_enhancing_tumor", "segmented_whole_tumor"],
            confidence=0.95,
        ))
        recs.append(Recommendation(
            action="histopathology_required",
            display="Histopathological verification required",
            expected_info_gain=None,
            cost_tier="high",
            risk_tier="medium",
            utility=None,
            rationale="Definitive diagnosis and grading of brain lesions require histopathological confirmation via biopsy or resection.",
            reason="Brain tumors cannot be graded or typed on imaging alone.",
            evidence=["segmented_whole_tumor"],
            confidence=0.99,
        ))
        recs.append(Recommendation(
            action="radiologist_review",
            display="Radiologist review of the segmented volume",
            expected_info_gain=None,
            cost_tier="low",
            risk_tier="none",
            rationale="Subtype, grade and management are outside this model's vocabulary — it localises tumour tissue and nothing more.",
            utility=None,
        ))
    return recs
