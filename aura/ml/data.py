"""Synthetic chest-X-ray world with ground truth.

This is a *deliberate* design choice, not a shortcut: a closed synthetic world
gives every stage of the pipeline a ground-truth diagnosis and ground-truth
findings, so we can measure calibration (ECE), conformal coverage, and
quantum-vs-classical fusion lift honestly. The vision engine is contractually
swappable for a real CXR model (torchxrayvision / ONNX) — see services/vision.

Each 64x64 grayscale image plants pathology into anatomically-plausible regions:

    heart box (center)         -> cardiomegaly
    lower lung / CP angles     -> effusion, consolidation
    mid/upper lung fields      -> opacity, nodule
    whole lung darkness        -> hyperinflation (COPD)
    unilateral dark + line     -> pneumothorax

so that region-based feature extraction and occlusion saliency both work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schemas.clinical import Diagnosis, Finding
from schemas.contracts import StructuredPriors

# The grid + anatomy primitives now live in the dependency-free ``common`` layer so
# request-serving ``services/`` code no longer has to import this synthetic-data
# module (fixes the ml→services layering violation). Re-exported here to keep
# ``ml.data.IMG`` / ``REGIONS`` / ``_px`` a stable public API.
from common.anatomy import IMG, REGIONS, _px  # noqa: F401  (re-export)


@dataclass
class Sample:
    image: np.ndarray                 # (IMG, IMG) float32 in [0,1]
    diagnosis: Diagnosis
    findings: dict[Finding, float]    # ground-truth finding strengths in [0,1]
    priors: StructuredPriors


def _base_thorax(rng: np.random.Generator) -> np.ndarray:
    """A plausible normal chest: dark lung fields, brighter mediastinum/heart, ribs."""
    img = np.full((IMG, IMG), 0.18, dtype=np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    # Bright central mediastinum column.
    med = np.exp(-((xx - IMG / 2) ** 2) / (2 * (IMG * 0.09) ** 2))
    img += 0.35 * med
    # Normal-size heart shadow (center-lower).
    hr0, hc0, hr1, hc1 = _px((0.45, 0.38, 0.74, 0.62))
    img[hr0:hr1, hc0:hc1] += 0.22
    # Faint rib texture (diagonal sinusoid) so occlusion/texture features have signal.
    ribs = 0.05 * np.sin((yy * 0.9 + xx * 0.3) * 0.8)
    img += ribs
    # Diaphragm brightening at the very bottom.
    img[int(0.86 * IMG):, :] += 0.15
    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def _blob(img: np.ndarray, region, intensity: float, rng, spread: float = 0.6) -> None:
    r0, c0, r1, c1 = _px(region)
    cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
    sr, sc = max(1.0, (r1 - r0) * spread / 2), max(1.0, (c1 - c0) * spread / 2)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    g = np.exp(-(((yy - cr) ** 2) / (2 * sr**2) + ((xx - cc) ** 2) / (2 * sc**2)))
    jitter = rng.normal(1.0, 0.05)
    img += (intensity * jitter * g).astype(np.float32)


def _priors_for(dx: Diagnosis, rng: np.random.Generator) -> StructuredPriors:
    """Conditional priors — correlated with, but not deterministic of, the diagnosis."""
    def band():
        return rng.choice(["18-40", "40-65", "65+"], p=[0.3, 0.4, 0.3])

    sex = rng.choice(["F", "M"])
    p = StructuredPriors(age_band=band(), sex=str(sex))
    if dx == Diagnosis.PNEUMONIA:
        p.fever = rng.random() < 0.8
    elif dx == Diagnosis.HEART_FAILURE:
        p.age_band = rng.choice(["40-65", "65+"], p=[0.35, 0.65])
    elif dx == Diagnosis.COPD:
        p.smoker = rng.random() < 0.85
        p.age_band = rng.choice(["40-65", "65+"], p=[0.4, 0.6])
    elif dx == Diagnosis.MALIGNANCY:
        p.smoker = rng.random() < 0.7
        p.prior_cancer = rng.random() < 0.4
        p.age_band = rng.choice(["40-65", "65+"], p=[0.4, 0.6])
    # small chance of incidental fever/smoker noise in any case
    p.smoker = p.smoker or (rng.random() < 0.1)
    p.fever = p.fever or (rng.random() < 0.08)
    p.immunocompromised = rng.random() < 0.08
    return p


def make_sample(dx: Diagnosis, rng: np.random.Generator) -> Sample:
    img = _base_thorax(rng)
    f: dict[Finding, float] = {k: 0.0 for k in Finding}

    if dx == Diagnosis.PNEUMONIA:
        side = rng.choice(["right_lung", "left_lung"])
        _blob(img, REGIONS[side], intensity=rng.uniform(0.35, 0.55), rng=rng, spread=0.7)
        f[Finding.OPACITY] = rng.uniform(0.7, 0.95)
        f[Finding.CONSOLIDATION] = rng.uniform(0.6, 0.9)

    elif dx == Diagnosis.HEART_FAILURE:
        _blob(img, REGIONS["heart"], intensity=rng.uniform(0.28, 0.42), rng=rng, spread=0.95)
        _blob(img, REGIONS["right_cp_angle"], intensity=rng.uniform(0.25, 0.4), rng=rng)
        _blob(img, REGIONS["left_cp_angle"], intensity=rng.uniform(0.25, 0.4), rng=rng)
        f[Finding.CARDIOMEGALY] = rng.uniform(0.7, 0.95)
        f[Finding.EFFUSION] = rng.uniform(0.6, 0.9)
        f[Finding.OPACITY] = rng.uniform(0.3, 0.5)

    elif dx == Diagnosis.COPD:
        img *= rng.uniform(0.6, 0.75)                      # hyperlucent (darker) lungs
        img[int(0.8 * IMG):, :] *= 0.8                     # flattened diaphragm
        f[Finding.HYPERINFLATION] = rng.uniform(0.7, 0.95)

    elif dx == Diagnosis.MALIGNANCY:
        side = rng.choice(["right_lung", "left_lung", "right_apex", "left_apex"])
        _blob(img, REGIONS[side], intensity=rng.uniform(0.45, 0.7), rng=rng, spread=0.25)
        f[Finding.NODULE] = rng.uniform(0.7, 0.95)
        f[Finding.OPACITY] = rng.uniform(0.3, 0.5)

    elif dx == Diagnosis.PNEUMOTHORAX:
        side = rng.choice(["right_lung", "left_lung"])
        r0, c0, r1, c1 = _px(REGIONS[side])
        img[r0:r1, c0:c1] *= rng.uniform(0.35, 0.5)        # collapsed, hyperlucent side
        line = int((c0 + c1) / 2)                           # visible pleural line
        img[r0:r1, max(0, line - 1):line + 1] += 0.35
        f[Finding.PNEUMOTHORAX] = rng.uniform(0.7, 0.95)

    # NORMAL: leave findings at 0.
    img = np.clip(img + rng.normal(0, 0.015, img.shape).astype(np.float32), 0, 1)
    return Sample(image=img, diagnosis=dx, findings=f, priors=_priors_for(dx, rng))


def make_dataset(n: int, seed: int = 7) -> list[Sample]:
    rng = np.random.default_rng(seed)
    # Realistic-ish prevalence: normal is common.
    dxs = list(Diagnosis)
    probs = np.array([0.34, 0.18, 0.15, 0.13, 0.12, 0.08])
    probs = probs / probs.sum()
    out = []
    for _ in range(n):
        dx = dxs[int(rng.choice(len(dxs), p=probs))]
        out.append(make_sample(dx, rng))
    return out


def make_multimodal(dx: Diagnosis, rng: np.random.Generator):
    """Plausible labs/symptoms/history correlated with the diagnosis.

    Lets the clinical reasoning engine actually fire in the demo (guideline rules
    key off BNP, WBC, procalcitonin, symptoms, and history). Deliberately noisy so
    reasoning must weigh, not memorize.
    """
    from schemas.contracts import (
        ClinicalHistory, LabPanel, MultimodalContext, Symptoms,
    )
    labs, sym, hist = LabPanel(), Symptoms(), ClinicalHistory()
    labs.wbc = round(float(rng.normal(7.5, 1.5)), 1)
    labs.spo2 = round(float(rng.normal(97, 1.5)), 0)

    if dx == Diagnosis.PNEUMONIA:
        sym.fever = rng.random() < 0.85
        sym.productive_cough = rng.random() < 0.8
        labs.wbc = round(float(rng.normal(15, 2.5)), 1)
        labs.crp = round(float(rng.uniform(60, 180)), 0)
        labs.procalcitonin = round(float(rng.uniform(0.5, 4.0)), 2)
        labs.spo2 = round(float(rng.normal(93, 2)), 0)
    elif dx == Diagnosis.HEART_FAILURE:
        sym.dyspnea = True
        sym.orthopnea = rng.random() < 0.75
        labs.bnp = round(float(rng.uniform(500, 2500)), 0)
        labs.spo2 = round(float(rng.normal(92, 2)), 0)
        hist.heart_failure = rng.random() < 0.5
    elif dx == Diagnosis.COPD:
        hist.copd = True
        hist.smoking_pack_years = round(float(rng.uniform(20, 60)), 0)
        sym.dyspnea = rng.random() < 0.7
        labs.bnp = round(float(rng.uniform(20, 120)), 0)
    elif dx == Diagnosis.MALIGNANCY:
        hist.smoking_pack_years = round(float(rng.uniform(15, 70)), 0)
        hist.prior_cancer = rng.random() < 0.4
        sym.hemoptysis = rng.random() < 0.35
    elif dx == Diagnosis.PNEUMOTHORAX:
        sym.acute_onset = True
        sym.pleuritic_chest_pain = rng.random() < 0.85
        sym.dyspnea = rng.random() < 0.7
        labs.spo2 = round(float(rng.normal(93, 3)), 0)

    if rng.random() < 0.1:                          # incidental immunosuppression
        hist.immunosuppression = True
    return MultimodalContext(labs=labs, symptoms=sym, history=hist)


def make_ood_sample(seed: int = 99) -> np.ndarray:
    """An out-of-distribution image (structured noise, wrong modality look)."""
    rng = np.random.default_rng(seed)
    img = rng.random((IMG, IMG)).astype(np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img += 0.5 * np.sin(xx * 1.7) * np.cos(yy * 1.3)       # high-freq grid, unlike a CXR
    return np.clip((img - img.min()) / (np.ptp(img) + 1e-9), 0, 1)
