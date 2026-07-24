"""Pixel descriptor used to sanity-check an MR study's channel order.

Deliberately intensity-*shape* features rather than absolute intensities: MR has no
standardised scale, so anything that keys off raw brightness would encode the exporter
rather than the sequence. Everything here is computed on brain voxels only, after
z-scoring within the brain, so a study rescaled by its exporter produces the same
vector.

Shared by the training script and the serving path — the classifier is only valid
against the exact feature layout it was fitted on, and two implementations of "the same"
descriptor is the standard way that invariant gets broken.
"""
from __future__ import annotations

import numpy as np

#: Brain-voxel threshold on a [0,1]-normalised slice. Matches the ingest convention.
_BRAIN_LEVEL = 0.02
#: Minimum brain voxels for the descriptor to be meaningful.
_MIN_BRAIN_VOXELS = 200
#: Histogram bins over the z-scored brain intensity, and the range they cover.
_HIST_BINS = 24
_HIST_RANGE = (-3.0, 4.0)
_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

#: Length of the returned vector. Asserted by the loader so a changed descriptor
#: cannot be silently fed to a classifier fitted on the old one.
FEATURE_DIM = len(_PERCENTILES) + _HIST_BINS + 8


def sequence_features(slice_2d: np.ndarray) -> np.ndarray | None:
    """Descriptor for one [0,1]-normalised 2D slice, or ``None`` if too little brain."""
    a = np.asarray(slice_2d, dtype=np.float32)
    brain = a > _BRAIN_LEVEL
    if int(brain.sum()) < _MIN_BRAIN_VOXELS:
        return None
    v = a[brain]
    v = (v - v.mean()) / (v.std() + 1e-6)
    hist, _ = np.histogram(v, bins=_HIST_BINS, range=_HIST_RANGE, density=True)
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)[brain]
    centred = v - v.mean()
    return np.concatenate([
        np.percentile(v, _PERCENTILES),
        hist,
        [v.mean(), v.std(),
         float((centred ** 3).mean()), float((centred ** 4).mean()),
         g.mean(), g.std(), float(np.percentile(g, 95)),
         float(brain.mean())],
    ]).astype(np.float32)
