"""Region-based feature extraction from a normalized CXR array.

Features are anatomically motivated so downstream finding scores and occlusion
saliency are interpretable. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# Grid/anatomy primitives come from the dependency-free common layer, not the
# synthetic-data module — keeps the serving path free of an ml/ import (audit §11.3).
from common.anatomy import IMG, REGIONS, _px, resize_to

FEATURE_NAMES = [
    "right_lung_bright",
    "left_lung_bright",
    "lung_mean",
    "heart_ratio",
    "cp_bright",
    "nodule_contrast",
    "lung_asymmetry",
    "vertical_line",
    "heart_width",
    "nodule_tophat",
]


def _region_mean(img: np.ndarray, region) -> float:
    r0, c0, r1, c1 = _px(region)
    return float(img[r0:r1, c0:c1].mean())


# Backward-compatible alias — the implementation now lives in common.anatomy so it
# is shared without a cross-service import. Kept so existing callers keep working.
_resize_to = resize_to


def extract_features(img: np.ndarray) -> dict[str, float]:
    img = _resize_to(img)
    lung_baseline = 0.18

    rl = _region_mean(img, REGIONS["right_lung"])
    ll = _region_mean(img, REGIONS["left_lung"])
    lung_mean = 0.5 * (rl + ll)

    heart = _region_mean(img, REGIONS["heart"])
    # Compare heart brightness/extent to a normal reference box just lateral to it.
    ref = 0.5 * (
        _region_mean(img, (0.45, 0.18, 0.74, 0.32))
        + _region_mean(img, (0.45, 0.68, 0.74, 0.82))
    )
    heart_ratio = float(heart / (ref + 1e-6))

    cp = 0.5 * (
        _region_mean(img, REGIONS["right_cp_angle"])
        + _region_mean(img, REGIONS["left_cp_angle"])
    )

    # Nodule: strongest small-scale bright deviation inside a lung field.
    nodule_contrast = 0.0
    for side in ("right_lung", "left_lung"):
        r0, c0, r1, c1 = _px(REGIONS[side])
        patch = img[r0:r1, c0:c1]
        if patch.size:
            local = patch.max() - np.median(patch)
            nodule_contrast = max(nodule_contrast, float(local))

    lung_asymmetry = float(abs(rl - ll))

    # Vertical line detector (pneumothorax pleural edge): max column-gradient energy.
    gx = np.abs(np.diff(img, axis=1))
    col_energy = gx.mean(axis=0)
    vertical_line = float(col_energy.max())

    # Cardiothoracic-width proxy: fraction of the central cardiac band that is
    # bright (an enlarged heart widens the central silhouette).
    hr0, hc0, hr1, hc1 = _px((0.50, 0.30, 0.74, 0.70))
    band = img[hr0:hr1, hc0:hc1]
    heart_width = float((band.mean(axis=0) > 0.32).mean()) if band.size else 0.0

    # Nodule top-hat: sharpest small-scale bright spot in a lung field (a nodule
    # is focal and bright, unlike diffuse opacity).
    nodule_tophat = 0.0
    for side in ("right_lung", "left_lung", "right_apex", "left_apex"):
        r0, c0, r1, c1 = _px(REGIONS[side])
        patch = img[r0:r1, c0:c1]
        if patch.size >= 25:
            local = uniform_filter(patch, size=7)
            nodule_tophat = max(nodule_tophat, float((patch - local).max()))

    return {
        "right_lung_bright": max(0.0, rl - lung_baseline),
        "left_lung_bright": max(0.0, ll - lung_baseline),
        "lung_mean": lung_mean,
        "heart_ratio": heart_ratio,
        "cp_bright": max(0.0, cp - lung_baseline),
        "nodule_contrast": nodule_contrast,
        "lung_asymmetry": lung_asymmetry,
        "vertical_line": vertical_line,
        "heart_width": heart_width,
        "nodule_tophat": nodule_tophat,
    }


def feature_vector(img: np.ndarray) -> np.ndarray:
    f = extract_features(img)
    return np.array([f[name] for name in FEATURE_NAMES], dtype=float)
