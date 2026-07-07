"""Confidence & Safety Engine — turns a raw posterior into a *trustworthy* one.

Calibration (temperature scaling), distribution-free conformal prediction sets,
epistemic + aleatoric uncertainty, energy-score OOD detection, and an explicit
abstention policy. No silent failures: anything uncertain is flagged, not hidden.
"""
from services.safety.engine import SafetyEngine

__all__ = ["SafetyEngine"]
