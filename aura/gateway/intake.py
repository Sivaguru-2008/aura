"""Clinical-first intake: the pre-imaging baseline a clinician files before the scan.

The dashboard's intake form is filled in *before* the study is uploaded, so the
facts it carries (priors, labs, symptoms, history, free-text notes) exist while
there is still no image, no vision result and no posterior. This module models
that stage on its own terms:

  * :class:`IntakeRequest` — the form payload, mapped onto the same contracts the
    pipeline already fuses (``StructuredPriors``, ``MultimodalContext``,
    ``ClinicalContext``), so nothing new has to be taught downstream.
  * :func:`triage_priority` — a pre-imaging acuity score for the pending panel.

``triage_priority`` is an explicit **rule-based triage heuristic, not a learned
model**, and it is not calibrated: it exists to order intakes in the queue before
any imaging evidence is available. The moment the study is analysed, the case
carries ``Pipeline._priority`` — the model-derived worklist priority — and this
number is not used again. Every threshold below is the one the reasoning engine
already cites (``services/reasoning/engine.py``: WBC > 11, procalcitonin >= 0.5,
CRP > 50, BNP >= 400, SpO2 < 92) so a red flag means the same thing in triage as
it does in the reasoning trace.
"""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field

from aura.schemas.contracts import (
    ClinicalContext,
    ClinicalHistory,
    LabPanel,
    MultimodalContext,
    StructuredPriors,
    Symptoms,
)


class IntakeRequest(BaseModel):
    """The intake form as the dashboard posts it (``POST /v1/cases``)."""

    age_band: str = "unknown"
    sex: str = "unknown"
    age: Optional[int] = None
    smoker: bool = False
    fever: bool = False
    prior_cancer: bool = False
    immunocompromised: bool = False
    symptoms: Symptoms = Field(default_factory=Symptoms)
    labs: LabPanel = Field(default_factory=LabPanel)
    history: ClinicalHistory = Field(default_factory=ClinicalHistory)
    clinical_notes: Optional[str] = None

    def priors(self) -> StructuredPriors:
        return StructuredPriors(
            age_band=self.age_band or "unknown",
            sex=self.sex or "unknown",
            smoker=bool(self.smoker),
            fever=bool(self.fever),
            prior_cancer=bool(self.prior_cancer),
            immunocompromised=bool(self.immunocompromised),
        )

    def multimodal(self) -> MultimodalContext:
        return MultimodalContext(labs=self.labs, symptoms=self.symptoms,
                                 history=self.history)

    def clinical_context(self) -> ClinicalContext:
        return ClinicalContext(
            age=self.age,
            sex=None if self.sex in ("", "unknown", None) else self.sex,
            symptoms=symptoms_text(self.symptoms) or None,
            clinical_notes=self.clinical_notes or None,
        )


#: Human-readable label per symptom flag, in the order they are reported.
_SYMPTOM_LABELS: tuple[tuple[str, str], ...] = (
    ("dyspnea", "dyspnoea"),
    ("productive_cough", "productive cough"),
    ("fever", "fever"),
    ("pleuritic_chest_pain", "pleuritic chest pain"),
    ("hemoptysis", "haemoptysis"),
    ("orthopnea", "orthopnoea"),
    ("acute_onset", "acute onset"),
)


def symptoms_text(symptoms: Symptoms) -> str:
    """Comma-joined symptom summary — carried into ``/v1/studies/analyze``."""
    return ", ".join(label for attr, label in _SYMPTOM_LABELS
                     if getattr(symptoms, attr, False))


def triage_priority(req: IntakeRequest) -> tuple[float, list[str]]:
    """Pre-imaging acuity in [0, 1) plus the red flags that produced it.

    Weights are additive and then saturated through ``1 - exp(-total)``, so no
    single flag can dominate and a long tail of minor findings cannot exceed a
    genuine red flag. The drivers are returned alongside the number because an
    unexplained priority is not actionable at triage.
    """
    total = 0.0
    drivers: list[str] = []

    def flag(weight: float, why: str) -> None:
        nonlocal total
        total += weight
        drivers.append(why)

    labs, symp, hx = req.labs, req.symptoms, req.history

    # --- physiology first: these are the ones that change disposition ---
    if labs.spo2 is not None and labs.spo2 < 92:
        flag(0.35, f"hypoxaemia (SpO2 {labs.spo2:.0f}%)")
        if labs.spo2 < 88:
            flag(0.15, "severe hypoxaemia (SpO2 < 88%)")
    if labs.troponin is not None and labs.troponin >= 14:      # hs-cTn 99th pct
        flag(0.25, f"troponin {labs.troponin:.0f} ng/L above the 99th percentile")
    if labs.bnp is not None and labs.bnp >= 400:
        flag(0.20, f"BNP {labs.bnp:.0f} pg/mL")
    if labs.procalcitonin is not None and labs.procalcitonin >= 0.5:
        flag(0.20, f"procalcitonin {labs.procalcitonin:.2f} ng/mL")
    if labs.d_dimer is not None and labs.d_dimer >= 500:
        flag(0.15, f"D-dimer {labs.d_dimer:.0f} ng/mL")
    if labs.wbc is not None and labs.wbc > 11:
        flag(0.10, f"leukocytosis (WBC {labs.wbc:.1f})")
    if labs.crp is not None and labs.crp > 50:
        flag(0.10, f"CRP {labs.crp:.0f} mg/L")

    # --- presenting complaint ---
    if symp.hemoptysis:
        flag(0.20, "haemoptysis")
    if symp.acute_onset:
        flag(0.15, "acute onset")
    if symp.dyspnea:
        flag(0.15, "dyspnoea")
    if symp.pleuritic_chest_pain:
        flag(0.10, "pleuritic chest pain")
    if symp.orthopnea:
        flag(0.10, "orthopnoea")
    if symp.fever or req.fever:
        flag(0.05, "fever")

    # --- standing risk ---
    if req.prior_cancer or hx.prior_cancer:
        flag(0.15, "prior malignancy")
    if req.immunocompromised or hx.immunosuppression:
        flag(0.15, "immunocompromised")
    if hx.recent_surgery:
        flag(0.10, "recent surgery")
    if req.age_band == "65+" or (req.age is not None and req.age >= 65):
        flag(0.10, "age 65+")
    if hx.smoking_pack_years and hx.smoking_pack_years >= 20:
        flag(0.05, f"{hx.smoking_pack_years:.0f} pack-year history")
    elif req.smoker:
        flag(0.05, "current smoker")
    if hx.heart_failure:
        flag(0.05, "heart failure")
    if hx.copd:
        flag(0.05, "COPD")

    return round(1.0 - math.exp(-total), 4), drivers
