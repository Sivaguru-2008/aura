"""Confidence & Safety Engine — turns a raw posterior into a *trustworthy* one.

Calibration (temperature scaling), distribution-free conformal prediction sets,
epistemic + aleatoric uncertainty, energy-score OOD detection, and an explicit
abstention policy. No silent failures: anything uncertain is flagged, not hidden.
"""
from services.safety.engine import SafetyEngine
from services.safety.controller import ClinicalSafetyController, ClinicalSafetyException, SafetyControllerOutput
from services.safety.readiness import ClinicalDecisionReadinessEngine, DecisionReadinessProfile

__all__ = [
    "SafetyEngine",
    "ClinicalSafetyController",
    "ClinicalSafetyException",
    "SafetyControllerOutput",
    "ClinicalDecisionReadinessEngine",
    "DecisionReadinessProfile"
]
