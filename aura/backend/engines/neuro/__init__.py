"""AURA NeuroMind — brain MRI analysis.

Serves the trained BraTS2020 network (``backend/vision/brain``) behind a Platt-calibrated
presence head. Requires a volumetric MR study with all four sequences — a 2D PNG/JPEG
export is refused with the measured single-sequence Dice attached — and it does not
classify tumour subtype. See ``engine.py`` for the two load-bearing refusals.
"""

from .engine import NeuroMindEngine

__all__ = ["NeuroMindEngine"]
