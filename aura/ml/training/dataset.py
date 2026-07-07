"""Turn the synthetic image world into an evidence-vector dataset for fusion.

Runs the real vision engine on each generated image (so training sees exactly
what serving sees) and pairs the 8-channel evidence vector with the ground-truth
diagnosis label.
"""
from __future__ import annotations

import numpy as np

from schemas.clinical import DIAGNOSES
from services.fusion.evidence import encode
from services.vision import VisionEngine
from ml.data import Sample, make_dataset


def build_evidence_dataset(samples: list[Sample], vision: VisionEngine | None = None):
    vision = vision or VisionEngine()
    X, y = [], []
    for i, s in enumerate(samples):
        vr = vision.analyze(f"train-{i}", s.image)
        X.append(encode(vr, s.priors))
        y.append(DIAGNOSES.index(s.diagnosis))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def make_splits(n: int, seed: int = 7):
    """Returns (train, calibration, test) sample lists."""
    samples = make_dataset(n, seed=seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    n_tr = int(0.6 * n)
    n_cal = int(0.2 * n)
    return samples[:n_tr], samples[n_tr:n_tr + n_cal], samples[n_tr + n_cal:]
