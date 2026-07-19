"""X-ray intake gate — decide whether an uploaded file is a chest radiograph.

Uploads are the one door where arbitrary images can enter the pipeline, and the
pipeline's OOD abstention only fires *after* a case exists. This gate runs first:
anything that is clearly not a radiograph (color photos, screenshots, documents,
logos) is rejected with a named reason before a case is ever created. Genuine
radiographs — including odd or out-of-distribution ones — pass through, where the
safety engine's conformal/OOD machinery remains the clinical backstop.

Checks are layered cheapest-first, numpy/PIL only:

  hard gates (any failure rejects)
    * decodable image (or DICOM)
    * DICOM modality must be radiographic (CR/DX) when tags are present
    * aspect ratio in a plausible radiograph range
    * grayscale content — radiographs carry no color
    * tonal depth — enough distinct gray levels / histogram entropy
    * dynamic range — not a near-solid image

  chest-structure gates (any failure rejects)
    * central column brightness — mediastinum/spine make the central vertical
      third brighter than the lung fields on every real chest film measured
    * column-profile variation — the dark-bright-dark column signature of a
      chest, absent in flat photos and gradients

  structural score (2 of 3 soft signals must hold)
    * smoothness — radiographs lack the high-frequency edge density of
      photos/screenshots/text
    * mid-gray tonal mass — exposure concentrates away from the extremes
    * tonal spread — a broad, smooth histogram rather than a few flat bands

A statistically radiograph-like impostor can still slip through heuristics; the
safety engine's OOD abstention downstream is the designed catch for that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Radiographic DICOM modalities (plain films / digital radiography).
_XRAY_MODALITIES = {"CR", "DX", "RG", "XA"}

# Hard-gate thresholds — calibrated against MIMIC-CXR JPEG exports (pass with
# wide margin) vs. photos/screenshots/solid fills (fail decisively).
_ASPECT_RANGE = (0.4, 2.5)          # h/w — films are roughly square-ish
_MAX_MEAN_SATURATION = 0.08         # radiographs are grayscale
_MAX_COLORED_FRACTION = 0.10        # share of pixels with visible chroma
_MIN_TONAL_ENTROPY_BITS = 4.0       # 256-bin histogram entropy; CXRs sit ~6-7.5
_MIN_GRAY_STD = 0.04                # near-solid images have almost none

# Chest-structure thresholds — MIMIC-CXR frontal+lateral films measure
# center_ratio in [1.08, 2.05] and col_var in [0.11, 0.59]; flat grayscale
# photos/noise sit near 1.0 and 0.02. Margins below the observed CXR minimum.
_MIN_CENTER_RATIO = 1.05            # central third vs lateral thirds brightness
_MIN_COLUMN_VARIATION = 0.09        # std/mean of the column brightness profile

# Soft structural thresholds (2 of 3 must hold).
_MAX_EDGE_DENSITY = 0.10            # mean |gradient| on a 256px grayscale
_MIN_MIDTONE_MASS = 0.60            # fraction of pixels in (0.06, 0.94)
_MIN_OCCUPIED_BINS = 96             # distinct populated gray levels of 256


@dataclass
class GateResult:
    ok: bool
    reason: str = ""
    checks: dict = field(default_factory=dict)


def _load_rgb(path: str | Path) -> np.ndarray:
    """Decode PNG/JPG/TIFF to float RGB in [0,1], downscaled to ~256px."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


def _entropy_bits(gray: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _structural_score(gray: np.ndarray) -> tuple[int, dict]:
    """Score soft radiograph-shaped signals on a [0,1] grayscale array."""
    gy, gx = np.gradient(gray)
    edge_density = float(np.mean(np.hypot(gx, gy)))
    midtone_mass = float(np.mean((gray > 0.06) & (gray < 0.94)))
    occupied = int((np.histogram(gray, bins=256, range=(0.0, 1.0))[0] > 0).sum())
    signals = {
        "smoothness": edge_density <= _MAX_EDGE_DENSITY,
        "midtone_mass": midtone_mass >= _MIN_MIDTONE_MASS,
        "tonal_spread": occupied >= _MIN_OCCUPIED_BINS,
    }
    detail = {
        "edge_density": round(edge_density, 4),
        "midtone_mass": round(midtone_mass, 4),
        "occupied_gray_levels": occupied,
        "signals": signals,
    }
    return sum(signals.values()), detail


def _gate_gray(gray: np.ndarray, checks: dict) -> GateResult:
    """Shared grayscale gates: aspect, entropy, range, structure."""
    h, w = gray.shape
    aspect = h / max(1, w)
    checks["aspect_ratio"] = round(aspect, 3)
    if not (_ASPECT_RANGE[0] <= aspect <= _ASPECT_RANGE[1]):
        return GateResult(False, "image proportions do not match a radiograph", checks)

    std = float(gray.std())
    checks["gray_std"] = round(std, 4)
    if std < _MIN_GRAY_STD:
        return GateResult(False, "image is nearly uniform — no anatomical content", checks)

    ent = _entropy_bits(gray)
    checks["tonal_entropy_bits"] = round(ent, 3)
    if ent < _MIN_TONAL_ENTROPY_BITS:
        return GateResult(
            False, "tonal histogram is too flat for a radiograph "
                   "(looks like a graphic, document, or screenshot)", checks)

    # Chest structure: bright mediastinum column against darker lung fields.
    thirds = w // 3
    lateral = (gray[:, :thirds].mean() + gray[:, 2 * thirds:].mean()) / 2
    center_ratio = float(gray[:, thirds:2 * thirds].mean() / max(1e-6, lateral))
    colprof = gray.mean(axis=0)
    col_var = float(colprof.std() / max(1e-6, colprof.mean()))
    checks["center_ratio"] = round(center_ratio, 3)
    checks["column_variation"] = round(col_var, 3)
    if center_ratio < _MIN_CENTER_RATIO or col_var < _MIN_COLUMN_VARIATION:
        return GateResult(
            False, "no chest anatomy detected — the bright central mediastinum "
                   "column of a chest film is missing", checks)

    score, detail = _structural_score(gray)
    checks["structure"] = detail
    if score < 2:
        return GateResult(
            False, "image structure does not match a chest radiograph", checks)

    return GateResult(True, "", checks)


def _validate_dicom(path: str | Path) -> GateResult:
    import pydicom
    from services.vision.io import load_dicom

    checks: dict = {"format": "dicom"}
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    modality = str(getattr(ds, "Modality", "")).upper()
    checks["modality"] = modality or "(absent)"
    if modality and modality not in _XRAY_MODALITIES:
        return GateResult(
            False, f"DICOM modality '{modality}' is not a radiograph "
                   f"(expected one of {sorted(_XRAY_MODALITIES)})", checks)
    body_part = str(getattr(ds, "BodyPartExamined", "")).upper()
    if body_part:
        checks["body_part"] = body_part
        if "CHEST" not in body_part and "THORAX" not in body_part:
            return GateResult(
                False, f"DICOM body part '{body_part}' is not a chest study", checks)
    return _gate_gray(load_dicom(path), checks)


def validate_cxr(path: str | Path) -> GateResult:
    """Return whether ``path`` plausibly contains a chest radiograph.

    Never raises for bad input — undecodable files come back as a rejection.
    """
    ext = Path(path).suffix.lower()
    if ext in (".dcm", ".dicom", ""):
        try:
            return _validate_dicom(path)
        except Exception:
            pass  # not a real DICOM — fall through to plain-image handling

    try:
        rgb = _load_rgb(path)
    except Exception:
        return GateResult(False, "file could not be decoded as an image")

    checks: dict = {"format": "image"}
    # Color gate: per-pixel chroma = max(R,G,B) - min(R,G,B).
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mean_sat = float(chroma.mean())
    colored = float(np.mean(chroma > 0.15))
    checks["mean_saturation"] = round(mean_sat, 4)
    checks["colored_fraction"] = round(colored, 4)
    if mean_sat > _MAX_MEAN_SATURATION or colored > _MAX_COLORED_FRACTION:
        return GateResult(
            False, "color content detected — radiographs are grayscale", checks)

    return _gate_gray(rgb.mean(axis=2), checks)
