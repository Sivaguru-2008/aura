"""Clinical vocabulary for the P0 chest-X-ray cut.

Kept deliberately small and explicit. In production this maps to RadLex /
SNOMED-CT codes; here it is a closed set so the whole pipeline has ground truth.
"""
from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    CXR = "CXR"          # chest radiograph — the P0 modality
    CT = "CT"
    MR = "MR"


class Finding(str, Enum):
    """Image-level observations the vision engine can report.

    These are *observations*, not diagnoses. Fusion turns findings (+ priors)
    into a diagnosis posterior.
    """
    OPACITY = "opacity"
    CONSOLIDATION = "consolidation"
    EFFUSION = "pleural_effusion"
    CARDIOMEGALY = "cardiomegaly"
    NODULE = "nodule"
    PNEUMOTHORAX = "pneumothorax"
    HYPERINFLATION = "hyperinflation"

    # Brain MRI findings
    MASS_LESION = "mass_lesion"
    MIDLINE_SHIFT = "midline_shift"
    EDEMA = "edema"
    HEMORRHAGE = "hemorrhage"
    VENTRICULOMEGALY = "ventriculomegaly"
    HYPINTENSITY = "hypintensity"
    HYPERINTENSITY = "hyperintensity"


class Diagnosis(str, Enum):
    """The differential AURA reasons over. `NORMAL` is a first-class label."""
    NORMAL = "normal"
    PNEUMONIA = "pneumonia"
    HEART_FAILURE = "heart_failure"
    COPD = "copd"
    MALIGNANCY = "malignancy"
    PNEUMOTHORAX = "pneumothorax_dx"

    # Brain MRI diagnoses.
    #
    # BRAIN_TUMOR and NORMAL_BRAIN are the only two AURA can currently *emit*: the
    # trained brain model (backend/vision/brain) is a BraTS segmentation network with a
    # whole-tumour presence head, so what it measures is "is there tumour tissue here,
    # and where" — not which tumour. Its corpus is glioma-only, which means it has
    # never been shown a meningioma or a metastasis to distinguish one from, and
    # reporting a subtype from it would be a claim about data the model has not seen.
    #
    # The four subtype labels below are vocabulary for a subtype classifier that does
    # not exist yet. They stay in the enum because the report, safety, and UI layers
    # are already written against them, and because removing a label is not the same
    # as earning the right to use it. Nothing in the serving path emits them today; a
    # previous build did, by hashing the uploaded file's SHA-256 and indexing this
    # list, which is why the constraint is spelled out here rather than assumed.
    BRAIN_TUMOR = "brain_tumor"
    GLIOMA = "glioma"
    MENINGIOMA = "meningioma"
    STROKE = "stroke"
    HEMORRHAGE_DX = "hemorrhage_dx"
    NORMAL_BRAIN = "normal_brain"


# --------------------------------------------------------------------------- #
# Per-modality vocabularies
# --------------------------------------------------------------------------- #
# These lists are **positional class vectors**, not just membership sets. The chest
# pipeline maps its model's output tensor onto ``CHEST_DIAGNOSES`` by index
# (``{d: posterior[i] for i, d in enumerate(...)}`` in services/fusion), so their order
# is a wire format and adding, removing or reordering an entry silently relabels every
# prediction the model makes.
#
# That is not hypothetical. When the brain labels were added to the ``Diagnosis`` enum,
# ``DIAGNOSES = list(Diagnosis)`` grew from 6 to 12 while the trained fusion model still
# emitted 6 classes — so the chest severity maps raised ``KeyError`` on the first brain
# label, and every positional mapping in the fusion path was reading a 6-vector against
# a 12-name list. Splitting the vocabulary per modality is what makes that class of
# error impossible rather than merely unlikely.
#
# ``DIAGNOSES``/``FINDINGS`` remain the *chest* vectors, because that is what every
# existing consumer means by them. Use ``ALL_DIAGNOSES``/``ALL_FINDINGS`` for label
# lookups and UI vocabulary, and the per-modality lists for anything positional.

#: Chest differential, in the exact order the trained fusion model emits.
CHEST_DIAGNOSES: list[Diagnosis] = [
    Diagnosis.NORMAL,
    Diagnosis.PNEUMONIA,
    Diagnosis.HEART_FAILURE,
    Diagnosis.COPD,
    Diagnosis.MALIGNANCY,
    Diagnosis.PNEUMOTHORAX,
]

#: Brain differential. Only the first two are emitted by any current engine — see the
#: note on the enum members themselves.
BRAIN_DIAGNOSES: list[Diagnosis] = [
    Diagnosis.BRAIN_TUMOR,
    Diagnosis.NORMAL_BRAIN,
    Diagnosis.GLIOMA,
    Diagnosis.MENINGIOMA,
    Diagnosis.STROKE,
    Diagnosis.HEMORRHAGE_DX,
]

#: Chest findings, in the order the vision model's heads are indexed.
CHEST_FINDINGS: list[Finding] = [
    Finding.OPACITY,
    Finding.CONSOLIDATION,
    Finding.EFFUSION,
    Finding.CARDIOMEGALY,
    Finding.NODULE,
    Finding.PNEUMOTHORAX,
    Finding.HYPERINFLATION,
]

BRAIN_FINDINGS: list[Finding] = [
    Finding.MASS_LESION,
    Finding.MIDLINE_SHIFT,
    Finding.EDEMA,
    Finding.HEMORRHAGE,
    Finding.VENTRICULOMEGALY,
    Finding.HYPINTENSITY,
    Finding.HYPERINTENSITY,
]

#: Every label in the vocabulary. For UI maps and label lookups — never for indexing
#: a model output.
ALL_DIAGNOSES: list[Diagnosis] = list(Diagnosis)
ALL_FINDINGS: list[Finding] = list(Finding)

#: The chest vectors, under the names the existing pipeline imports.
FINDINGS: list[Finding] = CHEST_FINDINGS
DIAGNOSES: list[Diagnosis] = CHEST_DIAGNOSES

# Human-readable labels for reports / UI.
DIAGNOSIS_LABELS: dict[Diagnosis, str] = {
    Diagnosis.NORMAL: "No acute cardiopulmonary abnormality",
    Diagnosis.PNEUMONIA: "Pneumonia",
    Diagnosis.HEART_FAILURE: "Congestive heart failure",
    Diagnosis.COPD: "Chronic obstructive pulmonary disease",
    Diagnosis.MALIGNANCY: "Suspicious pulmonary malignancy",
    Diagnosis.PNEUMOTHORAX: "Pneumothorax",

    # Brain MRI
    Diagnosis.BRAIN_TUMOR: "Intracranial tumour — subtype not determined",
    Diagnosis.GLIOMA: "High-grade glioma",
    Diagnosis.MENINGIOMA: "Meningioma",
    Diagnosis.STROKE: "Acute ischemic stroke",
    Diagnosis.HEMORRHAGE_DX: "Intracranial hemorrhage",
    Diagnosis.NORMAL_BRAIN: "No acute intracranial abnormality",
}

FINDING_LABELS: dict[Finding, str] = {
    Finding.OPACITY: "Airspace opacity",
    Finding.CONSOLIDATION: "Consolidation",
    Finding.EFFUSION: "Pleural effusion",
    Finding.CARDIOMEGALY: "Cardiomegaly",
    Finding.NODULE: "Pulmonary nodule",
    Finding.PNEUMOTHORAX: "Pneumothorax",
    Finding.HYPERINFLATION: "Hyperinflation",

    # Brain MRI
    Finding.MASS_LESION: "Intracranial mass lesion",
    Finding.MIDLINE_SHIFT: "Midline shift",
    Finding.EDEMA: "Perilesional edema",
    Finding.HEMORRHAGE: "Intracranial hemorrhage",
    Finding.VENTRICULOMEGALY: "Ventriculomegaly",
    Finding.HYPINTENSITY: "T1 hypointensity",
    Finding.HYPERINTENSITY: "T2/FLAIR hyperintensity",
}
