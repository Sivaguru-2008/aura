from dataclasses import dataclass
from typing import List
from aura.schemas.clinical import Finding, Diagnosis

@dataclass
class GuidelineTemplate:
    imaging: List[Finding]
    symptoms: List[str]
    labs: List[str]
    diagnosis: str = ""
    guideline_source: str = "AURA Clinical Guidelines"

GUIDELINE_TEMPLATES: dict[str, GuidelineTemplate] = {
    "pneumonia": GuidelineTemplate(
        imaging=[Finding.CONSOLIDATION, Finding.OPACITY],
        symptoms=["fever"],
        labs=["wbc", "procalcitonin", "crp"]
    ),
    "heart_failure": GuidelineTemplate(
        imaging=[Finding.CARDIOMEGALY, Finding.EFFUSION],
        symptoms=["orthopnea"],
        labs=["bnp"]
    ),
    "copd": GuidelineTemplate(
        imaging=[Finding.HYPERINFLATION],
        symptoms=[],
        labs=[]
    ),
    "malignancy": GuidelineTemplate(
        imaging=[Finding.NODULE],
        symptoms=["hemoptysis"],
        labs=[]
    ),
    "pneumothorax": GuidelineTemplate(
        imaging=[Finding.PNEUMOTHORAX],
        symptoms=["pleuritic_chest_pain", "acute_onset"],
        labs=[]
    ),
    "normal": GuidelineTemplate(
        imaging=[],
        symptoms=[],
        labs=["spo2"]
    ),
}

# Dynamically set the diagnosis field on templates
for key, tmpl in GUIDELINE_TEMPLATES.items():
    tmpl.diagnosis = key


def get_template(diagnosis: str) -> GuidelineTemplate | None:
    diag_str = diagnosis.value if hasattr(diagnosis, "value") else str(diagnosis)
    return GUIDELINE_TEMPLATES.get(diag_str.lower())


def coverage_ratio(
    diagnosis: str,
    available_imaging: list[str] | None = None,
    available_labs: list[str] | None = None,
    available_symptoms: list[str] | None = None,
) -> float:
    diag_str = diagnosis.value if hasattr(diagnosis, "value") else str(diagnosis)
    tmpl = get_template(diag_str)
    if tmpl is None:
        return 0.0

    req_imaging = [str(x.value if hasattr(x, "value") else x).lower() for x in tmpl.imaging]
    req_labs = [str(x.value if hasattr(x, "value") else x).lower() for x in tmpl.labs]
    req_symptoms = [str(x.value if hasattr(x, "value") else x).lower() for x in tmpl.symptoms]
    
    total_required = len(req_imaging) + len(req_labs) + len(req_symptoms)
    if total_required == 0:
        return 1.0
        
    avail_img = [str(x.value if hasattr(x, "value") else x).lower() for x in (available_imaging or [])]
    avail_lab = [str(x.value if hasattr(x, "value") else x).lower() for x in (available_labs or [])]
    avail_sym = [str(x.value if hasattr(x, "value") else x).lower() for x in (available_symptoms or [])]
    
    present = 0
    for item in req_imaging:
        if item in avail_img:
            present += 1
    for item in req_labs:
        if item in avail_lab:
            present += 1
    for item in req_symptoms:
        if item in avail_sym:
            present += 1
            
    return float(present) / total_required
