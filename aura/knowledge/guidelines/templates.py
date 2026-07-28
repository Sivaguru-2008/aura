from dataclasses import dataclass
from typing import List
from schemas.clinical import Finding, Diagnosis

@dataclass
class GuidelineTemplate:
    imaging: List[Finding]
    symptoms: List[str]
    labs: List[str]

GUIDELINE_TEMPLATES: dict[Diagnosis, GuidelineTemplate] = {
    Diagnosis.PNEUMONIA: GuidelineTemplate(
        imaging=[Finding.CONSOLIDATION, Finding.OPACITY],
        symptoms=["fever"],
        labs=["wbc", "procalcitonin", "crp"]
    ),
    Diagnosis.HEART_FAILURE: GuidelineTemplate(
        imaging=[Finding.CARDIOMEGALY, Finding.EFFUSION],
        symptoms=["orthopnea"],
        labs=["bnp"]
    ),
    Diagnosis.COPD: GuidelineTemplate(
        imaging=[Finding.HYPERINFLATION],
        symptoms=[],
        labs=[]
    ),
    Diagnosis.MALIGNANCY: GuidelineTemplate(
        imaging=[Finding.NODULE],
        symptoms=["hemoptysis"],
        labs=[]
    ),
    Diagnosis.PNEUMOTHORAX: GuidelineTemplate(
        imaging=[Finding.PNEUMOTHORAX],
        symptoms=["pleuritic_chest_pain", "acute_onset"],
        labs=[]
    ),
    Diagnosis.NORMAL: GuidelineTemplate(
        imaging=[],
        symptoms=[],
        labs=["spo2"]
    ),
}
