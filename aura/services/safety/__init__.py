"""Confidence & Safety Engine — turns a raw posterior into a *trustworthy* one.

Calibration (temperature scaling), distribution-free conformal prediction sets,
epistemic + aleatoric uncertainty, energy-score OOD detection, and an explicit
abstention policy. No silent failures: anything uncertain is flagged, not hidden.

Layer 1: ClinicalSafetyController — veto checks before reasoning.
Layer 2: ClinicalDecisionReadinessEngine — multi-dimensional readiness profile.
"""
from aura.services.safety.engine import SafetyEngine
from aura.services.safety.controller import ClinicalSafetyController, ClinicalSafetyException
from aura.services.safety.readiness import ClinicalDecisionReadinessEngine

__all__ = ["SafetyEngine", "ClinicalSafetyController", "ClinicalSafetyException",
           "ClinicalDecisionReadinessEngine"]
