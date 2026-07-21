"""Build the evidence-vector dataset the fusion models train on.

Runs the **same** vision engine serving uses over each image and pairs the
8-channel evidence vector with a ground-truth diagnosis, so the fusion model
trains on exactly the distribution it will see in production.

Two sources are supported, both routed through the real vision backbone:

  * ``build_real_evidence_dataset`` — real MIMIC-CXR films with report-derived
    diagnoses. This is the correct production source: at serving, evidence is the
    real DenseNet's finding probabilities on real films, so fusion must train on
    that same distribution (audit F1).
  * ``build_evidence_dataset`` — the synthetic world, kept for the offline demo and
    for CI where the corpus is absent. It now defaults to ``VisionEngine.load()``
    (the trained backbone) rather than a bare ``VisionEngine()`` (the untrained
    heuristic), so even the synthetic path uses the serving vision model (audit F1).
"""
from __future__ import annotations

import logging

import numpy as np

from schemas.clinical import DIAGNOSES
from services.fusion.evidence import encode
from services.vision import VisionEngine
from ml.data import Sample, make_dataset

log = logging.getLogger("ml.training.dataset")


def build_evidence_dataset(samples: list[Sample], vision: VisionEngine | None = None):
    # Default to the *loaded* (trained) vision engine so training evidence is drawn
    # from the same model serving uses — never the untrained heuristic (audit F1).
    vision = vision or VisionEngine.load()
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


def build_real_evidence_dataset(
    n: int = 900,
    split: str = "train",
    vision: VisionEngine | None = None,
    scan_limit: int = 8000,
    per_class_cap: int | None = None,
    seed: int = 7,
):
    """Build (X, y) fusion evidence from **real MIMIC-CXR** studies (audit F1).

    For each real patient's most recent study: load the film through the standard
    intake seam, run the trained vision backbone, encode the 8-channel evidence
    vector, and label it with the report-derived diagnosis. One study per patient
    keeps the set patient-disjoint. ``per_class_cap`` optionally balances the
    naturally-skewed diagnosis distribution so rare-but-dangerous classes
    (pneumothorax, malignancy) are represented.

    Returns ``(X, y)`` numpy arrays, or ``(None, None)`` if the corpus is absent so
    callers can fall back to the synthetic path.
    """
    from mimic.config import get_mimic_paths
    from mimic.patient import iter_patients

    paths = get_mimic_paths()
    if not paths.validate_csv.is_file():
        log.warning("MIMIC-CXR not found at %s; cannot build real evidence set", paths.root)
        return None, None

    vision = vision or VisionEngine.load()
    per_class: dict[int, int] = {i: 0 for i in range(len(DIAGNOSES))}
    cap = per_class_cap if per_class_cap is not None else n
    X: list[np.ndarray] = []
    y: list[int] = []
    scanned = 0

    for patient in iter_patients(split, limit=scan_limit):
        if len(X) >= n:
            break
        if patient.n_studies == 0:
            continue
        scanned += 1
        try:
            study = patient.to_study_input(study_index=-1)
        except (ValueError, OSError):
            continue
        if study.ground_truth is None:
            continue
        cls = DIAGNOSES.index(study.ground_truth)
        if per_class[cls] >= cap:
            continue                              # balance: skip over-represented class
        img = np.array(study.image, dtype=float).reshape(study.image_shape)
        vr = vision.analyze(study.study_id, img)
        X.append(encode(vr, study.priors))
        y.append(cls)
        per_class[cls] += 1

    if len(X) < len(DIAGNOSES) * 3:               # too few to train a 6-class head
        log.warning("only %d real studies gathered (scanned %d) — insufficient", len(X), scanned)
        return None, None

    counts = {DIAGNOSES[i].value: per_class[i] for i in range(len(DIAGNOSES))}
    log.info("real evidence set: %d studies over %d scanned | per-class %s",
             len(X), scanned, counts)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def real_evidence_splits(
    n: int = 900, split: str = "train", vision: VisionEngine | None = None,
    seed: int = 7, per_class_cap: int | None = None,
):
    """Patient-disjoint 60/20/20 (train/calibration/test) split of real evidence.

    Returns ``(Xtr, ytr, Xcal, ycal, Xte, yte)`` or ``None`` if the corpus is
    unavailable / too small, so the caller falls back to the synthetic path.
    """
    X, y = build_real_evidence_dataset(
        n=n, split=split, vision=vision, per_class_cap=per_class_cap, seed=seed
    )
    if X is None:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr, n_cal = int(0.6 * len(X)), int(0.2 * len(X))
    return (X[:n_tr], y[:n_tr],
            X[n_tr:n_tr + n_cal], y[n_tr:n_tr + n_cal],
            X[n_tr + n_cal:], y[n_tr + n_cal:])
