"""Real MIMIC-CXR data integration for AURA.

This package replaces the synthetic chest-X-ray world (``ml/data.py`` +
``gateway/seed.py``) with real MIMIC-CXR patients: chest radiographs plus their
radiology reports, one longitudinal record per ``subject_id``.

It is strictly additive — nothing here changes existing service APIs, the
FastAPI gateway, the schemas, or the vision/CNN pipeline. Downstream code keeps
consuming the same ``StudyInput`` / ``MultimodalContext`` contracts; this package
just produces them from real data instead of ``numpy`` fabrications.

Layout (built step by step):

    config.py    dataset paths + schema constants (Step 1)
    verify.py    dataset verification report                (Step 1)
    loaders.py   lazy, chunked, validated CSV loaders       (Step 2)
    ...
"""
from __future__ import annotations

from .config import MimicPaths, get_mimic_paths

__all__ = ["MimicPaths", "get_mimic_paths"]
