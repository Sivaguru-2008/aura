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


class Diagnosis(str, Enum):
    """The differential AURA reasons over. `NORMAL` is a first-class label."""
    NORMAL = "normal"
    PNEUMONIA = "pneumonia"
    HEART_FAILURE = "heart_failure"
    COPD = "copd"
    MALIGNANCY = "malignancy"
    PNEUMOTHORAX = "pneumothorax_dx"


FINDINGS: list[Finding] = list(Finding)
DIAGNOSES: list[Diagnosis] = list(Diagnosis)

# Human-readable labels for reports / UI.
DIAGNOSIS_LABELS: dict[Diagnosis, str] = {
    Diagnosis.NORMAL: "No acute cardiopulmonary abnormality",
    Diagnosis.PNEUMONIA: "Pneumonia",
    Diagnosis.HEART_FAILURE: "Congestive heart failure",
    Diagnosis.COPD: "Chronic obstructive pulmonary disease",
    Diagnosis.MALIGNANCY: "Suspicious pulmonary malignancy",
    Diagnosis.PNEUMOTHORAX: "Pneumothorax",
}

FINDING_LABELS: dict[Finding, str] = {
    Finding.OPACITY: "Airspace opacity",
    Finding.CONSOLIDATION: "Consolidation",
    Finding.EFFUSION: "Pleural effusion",
    Finding.CARDIOMEGALY: "Cardiomegaly",
    Finding.NODULE: "Pulmonary nodule",
    Finding.PNEUMOTHORAX: "Pneumothorax",
    Finding.HYPERINFLATION: "Hyperinflation",
}
