"""Shared anatomical-grid primitives — the one place the feature grid is defined.

Why this module exists
----------------------
The region-feature extractor (``services.vision.features``) and the saliency
grids (``services.explain``) need the working image side length, the normalized
anatomical region boxes, and a couple of pure-numpy geometry helpers. Those same
constants are also used by the synthetic generator (``ml.data``).

They previously lived in ``ml.data``, which meant the request-serving code in
``services/`` imported the *synthetic-data* module — a layering violation of the
rule stated in ``ml/__init__.py`` ("Never imported by request-serving code paths")
and ``services/__init__.py`` ("No engine imports another"). Hoisting them into
``common/`` (the dependency-free base layer both ``ml`` and ``services`` may
import) removes the violation without changing any value or behaviour. ``ml.data``
re-exports these names so its public API is unchanged.

Pure numpy; no service/ml dependencies.
"""
from __future__ import annotations

import numpy as np

#: Working grid side length for region features and saliency maps. Independent of
#: the intake image resolution (which is 224 for the CNN after audit F5); any input
#: is resized to this grid for the interpretable numpy feature/occlusion path.
IMG = 64

#: Anatomical regions in normalized (row0, col0, row1, col1) coordinates.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "right_lung": (0.15, 0.08, 0.75, 0.44),
    "left_lung": (0.15, 0.56, 0.75, 0.92),
    "right_apex": (0.12, 0.12, 0.32, 0.42),
    "left_apex": (0.12, 0.58, 0.32, 0.88),
    "heart": (0.42, 0.34, 0.82, 0.66),
    "right_cp_angle": (0.72, 0.10, 0.92, 0.40),
    "left_cp_angle": (0.72, 0.60, 0.92, 0.90),
}


def _px(region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Normalized region box -> integer pixel bounds on the ``IMG``×``IMG`` grid."""
    r0, c0, r1, c1 = region
    return (int(r0 * IMG), int(c0 * IMG), int(r1 * IMG), int(c1 * IMG))


def resize_to(img: np.ndarray, side: int = IMG) -> np.ndarray:
    """Nearest-neighbour resize so any input maps to the feature grid (no PIL dep)."""
    img = np.asarray(img, dtype=np.float32)
    if img.shape == (side, side):
        return img
    rows = (np.linspace(0, img.shape[0] - 1, side)).astype(int)
    cols = (np.linspace(0, img.shape[1] - 1, side)).astype(int)
    return img[np.ix_(rows, cols)]
