"""Model Management Layer (P0 seed).

A lightweight registry reading artifact metadata: versions, calibration snapshot,
benchmark metrics, and status. Stands in for MLflow; same conceptual surface
(list versions, read metrics, mark active) the dashboard's admin view consumes.
"""
from services.models.registry import ModelRegistry

__all__ = ["ModelRegistry"]
