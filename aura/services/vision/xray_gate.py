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

# --------------------------------------------------------------------------- #
# Cross-sectional-head veto
# --------------------------------------------------------------------------- #
# The checks above test "is this a grayscale medical image with a bright central
# column". An axial slice through a head satisfies every one of them: the brain's
# midline is brighter than the temporal lobes, so ``center_ratio`` clears the
# mediastinum test. Measured on the brain MRI of CASE-UPLOAD-27 the gate returned
# ok=True with center_ratio 1.66 and no failing check, and the study was analysed by
# the chest model, which reported pneumonia. Rejecting an image that is positively
# something *else* is the gate's job, not the downstream OOD engine's.
#
# Discriminant, measured on 1500 random MIMIC-CXR films vs 600 BraTS2020 axial slices
# (half the brain slices scored after the console's own non-aspect-preserving 224-grid
# resize, to confirm the rule survives re-encoding):
#
#             background_fraction   dark_edge_count   foreground_bbox_fraction
#   chest     med 0.117  max 0.535   med 0   max 3     med 0.980  min 0.527
#   brain     med 0.798  min 0.655   always 4          med 0.273  max 0.448
#
#   head score:  chest med 0.000, p99 0.518, max 0.801
#                brain min 0.922, p1 0.993, med 1.000
#
# The classes are linearly separable on this score, with an empty band between 0.801
# and 0.922. The threshold is placed inside that band, so at the measured operating
# point the veto rejects 0.00% of real chest films and catches 100.0% of real brain
# slices. Both error rates are sample estimates, not guarantees: treat the gap as the
# margin it is, and re-measure if either corpus changes.
#
# Only these three features are used. ``centroid_offset`` and ``foreground_occupancy``
# were measured and discarded: they do not separate the classes (centroid median 0.061
# chest vs 0.052 brain; occupancy 0.777 vs 0.735), so including them in a conjunction
# only lets a non-discriminating term veto three discriminating ones.
#
# ---------------------------------------------------------------------------
# Second route: the head that has been cropped to its own outline
# ---------------------------------------------------------------------------
# Everything above keys on *framing* — a small subject floating in a large field of
# air. That is what a clean export looks like, and it is exactly what a crop destroys.
# CASE-UPLOAD-27's actual upload is a head cropped to its bounding box with a lifted
# black floor (a viewer screenshot, most likely): it scores 0.384 on the framing route
# and would still have been analysed as a chest film.
#
# So a second route measures the *shape of the subject* instead, which survives
# cropping. Re-measured on 500 MIMIC films against 500 BraTS slices put through that
# same crop-and-lift transformation:
#
#                     chest med / p95      cropped-brain med / p5     CASE-UPLOAD-27
#   corner_mean         0.261 / 0.472         0.039 / 0.022               0.018
#   ellipse_fit         0.730 / 0.900         0.991 / 0.887               0.852
#   solidity            0.843 / 0.949         0.956 / 0.824               0.857
#
#   corner_mean <= 0.08 & ellipse_fit >= 0.80          98.4% cropped brain, 3.2% chest
#   corner_mean <= 0.06 & ellipse >= 0.82 & sol >= 0.82  91.8% cropped brain, 2.0% chest
#
# The tighter rule ships. End-to-end on the full routing path the looser one cost 4.5%
# of real chest films (98.0% -> 93.5% reaching Thorax), which is too much to pay on the
# production path; the tighter one keeps the cropped-head catch while halving that. All
# three of CASE-UPLOAD-27's values clear it with margin.
#
# A residual chest cost is accepted deliberately, because the two errors are not
# comparable. A chest film diverted by this rule is *refused*: NeuroMind requires a
# volumetric multi-sequence MR study and rejects a 2D radiograph outright, so the user
# sees a named refusal and no case is created. A brain study that stays with the chest
# model is *analysed*, and comes back as a chest diagnosis — CASE-UPLOAD-27 was reported
# as pneumonia at p=0.25. Tune these thresholds only with that asymmetry in view.
_HEAD_CORNER_MEAN_MAX = 0.06
_HEAD_ELLIPSE_FIT_MIN = 0.82
_HEAD_SOLIDITY_MIN = 0.82
#: Score awarded by the shape route. Above the commit threshold, and below the framing
#: route's ceiling so a well-framed head still reports the stronger evidence.
_CROPPED_HEAD_SCORE = 0.90
_BACKGROUND_LEVEL = 0.10            # pixel value at or below which a pixel is "air"
_EDGE_BAND = 0.06                   # border strip width, as a fraction of the side
_DARK_EDGE_LEVEL = 0.06             # mean strip value below which an edge is "dark"

#: Head score at or above which an image is treated as a cross-sectional head study.
#: Placed in the measured empty band between the chest maximum (0.801) and the brain
#: minimum (0.922) rather than at either edge, so neither class sits on the boundary.
HEAD_COMMIT_SCORE = 0.85


def head_geometry_score(background_fraction: float, dark_edge_count: int,
                        foreground_bbox_fraction: float) -> float:
    """Evidence that an image is an axial slice through a head, in [0,1].

    Additive rather than a conjunction so that one weak term cannot veto the others —
    the failure mode measured on the previous six-way AND, which detected only 81.7%
    of real brain slices. Each term is clipped to its own measured separation band, so
    a feature deep in chest territory contributes nothing rather than going negative.

    Shared by the intake gate and by
    :class:`~backend.core.router.signatures.BrainMRISignature` so the two cannot drift
    apart: the router and the gate must agree about what a head slice looks like.
    """
    def ramp(value: float, lo: float, hi: float) -> float:
        return min(1.0, max(0.0, (value - lo) / (hi - lo)))

    return (
        0.40 * ramp(background_fraction, 0.32, 0.60)     # air around the subject
        + 0.30 * (max(0, min(4, dark_edge_count)) / 4.0)  # touches no frame edge
        + 0.30 * ramp(-foreground_bbox_fraction, -0.95, -0.50)  # compact subject
    )


def cropped_head_shape(gray: np.ndarray) -> tuple[bool, dict]:
    """Test the subject's *shape* for a cropped head. Survives tight cropping.

    Fits the largest foreground component and asks two questions the framing route
    cannot: are the frame corners air (an ellipse cannot fill its own bounding box), and
    is the subject actually elliptical?
    """
    detail: dict = {}
    try:
        import cv2
    except ImportError:
        return False, {"available": False}

    h, w = gray.shape
    threshold = max(0.12, float(gray.mean()) * 0.45)
    foreground = (gray > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return False, {"available": True, "reason": "no foreground component"}

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8)
    area = float(blob.sum())
    if area < 0.02 * h * w:
        return False, {"available": True, "reason": "subject too small to characterise"}

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, {"available": True, "reason": "no contour"}
    contour = max(contours, key=cv2.contourArea)

    ellipse_fit = 0.0
    if len(contour) >= 5:
        (_, _), (major, minor), _ = cv2.fitEllipse(contour)
        ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
        ellipse_fit = float(min(area, ellipse_area) / max(area, ellipse_area, 1.0))

    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or 1.0
    solidity = area / hull_area

    ch, cw = max(2, h // 8), max(2, w // 8)
    corner_mean = float(np.mean([
        gray[:ch, :cw].mean(), gray[:ch, -cw:].mean(),
        gray[-ch:, :cw].mean(), gray[-ch:, -cw:].mean()]))

    matches = (corner_mean <= _HEAD_CORNER_MEAN_MAX
               and ellipse_fit >= _HEAD_ELLIPSE_FIT_MIN
               and solidity >= _HEAD_SOLIDITY_MIN)
    detail = {
        "available": True,
        "corner_mean": round(corner_mean, 4),
        "corner_mean_max": _HEAD_CORNER_MEAN_MAX,
        "ellipse_fit": round(ellipse_fit, 4),
        "ellipse_fit_min": _HEAD_ELLIPSE_FIT_MIN,
        "solidity": round(solidity, 4),
        "solidity_min": _HEAD_SOLIDITY_MIN,
        "matches": matches,
    }
    return matches, detail


def _head_geometry(gray: np.ndarray) -> tuple[float, dict]:
    """Head-geometry evidence for a [0,1] grayscale array, over both routes.

    The framing formulas match :func:`backend.core.router.features._measure` exactly;
    the router's fingerprint and this gate must produce the same numbers for the same
    image. The score is the stronger of the two routes, so a well-framed head still
    reports the framing evidence and a cropped one is not missed for lacking it.
    """
    h, w = gray.shape
    band_h = max(1, int(h * _EDGE_BAND))
    band_w = max(1, int(w * _EDGE_BAND))
    edges = np.array([
        gray[:band_h, :].mean(), gray[-band_h:, :].mean(),
        gray[:, :band_w].mean(), gray[:, -band_w:].mean(),
    ], dtype=np.float32)
    dark_edges = int((edges < _DARK_EDGE_LEVEL).sum())

    fg_threshold = max(0.12, float(gray.mean()) * 0.45)
    ys, xs = np.nonzero(gray > fg_threshold)
    if ys.size:
        bbox_area = (int(ys.max()) - int(ys.min()) + 1) * (int(xs.max()) - int(xs.min()) + 1)
        bbox_fraction = bbox_area / float(h * w)
    else:
        bbox_fraction = 0.0

    background = float(np.mean(gray <= _BACKGROUND_LEVEL))
    framing_score = head_geometry_score(background, dark_edges, bbox_fraction)
    cropped, shape_detail = cropped_head_shape(gray)
    score = max(framing_score, _CROPPED_HEAD_SCORE if cropped else 0.0)

    detail = {
        "background_fraction": round(background, 3),
        "dark_edge_count": dark_edges,
        "foreground_bbox_fraction": round(bbox_fraction, 3),
        "framing_score": round(framing_score, 3),
        "cropped_head_shape": shape_detail,
        "head_score": round(score, 3),
        "commit_threshold": HEAD_COMMIT_SCORE,
        "route": ("cropped-head shape" if cropped and score == _CROPPED_HEAD_SCORE
                  else "framing"),
    }
    return score, detail


def head_geometry_from_path(path: str | Path) -> tuple[float, dict]:
    """Head-geometry evidence for an image file. Never raises.

    The router's brain signature calls this so it sees the same measurement the gate
    used to reject the image, rather than a second implementation of it.
    """
    try:
        ext = Path(path).suffix.lower()
        if ext in (".dcm", ".dicom", ""):
            try:
                from services.vision.io import load_dicom

                return _head_geometry(load_dicom(path))
            except Exception:
                pass
        return _head_geometry(_load_rgb(path).mean(axis=2))
    except Exception:
        return 0.0, {"available": False, "reason": "image could not be decoded"}


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

    # Positive rejection: this is a cross-sectional head study, not a chest film.
    # Runs *before* the mediastinum test because a brain's midline passes that test —
    # the check this defeats is exactly the one that let CASE-UPLOAD-27 through.
    head_score, head_detail = _head_geometry(gray)
    checks["head_geometry"] = head_detail
    if head_score >= HEAD_COMMIT_SCORE:
        return GateResult(
            False, "this is a cross-sectional slice through a head (compact subject "
                   "surrounded by signal-free air), not a chest radiograph — upload it "
                   "as a brain MRI or head CT", checks)

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
