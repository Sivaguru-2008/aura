"""Feature extraction for modality detection.

Every signature scores the *same* feature bundle, computed once per upload. That
matters for two reasons: decoding a large DICOM twice is wasteful, and — more
importantly — it makes the signatures directly comparable, because they are all
looking at identical measurements.

Two independent evidence channels:

**DICOM header** (:class:`DicomMetadata`) — ``Modality`` and ``BodyPartExamined`` are
written by the acquisition device. When present they are near-definitive and no pixel
heuristic should be allowed to overrule them.

**Pixel geometry** (:class:`PixelFeatures`) — for PNG/JPEG exports, which carry no
header at all. The features below were chosen by measuring what actually separates
the classes, not by intuition; the numbers in each docstring are from that
measurement (600 random real MIMIC-CXR films, plus the real MR/CR/CT/US DICOMs
bundled with pydicom).

Two findings from that measurement shaped this module:

* Dark corners are **not** a head-imaging signal. 50% of real chest films already
  have corners below 0.10 from collimation, so a corner-darkness rule would misfire
  constantly.
* An axial head CT and an axial head MRI produce **the same** pixel geometry: a
  compact bright blob floating in air-black. Pixels can identify a cross-sectional
  head study; they cannot tell you which scanner made it. That separation is what
  DICOM metadata is for, and it is why the pixel-only route for brain MRI carries
  capped confidence.

Nothing here raises. An undecodable file comes back with ``decodable=False`` and the
router turns that into a clean rejection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..shared.logging import get_logger

log = get_logger("router.features")

#: Working resolution for feature extraction. Every threshold in ``signatures`` was
#: measured at this size; changing it invalidates them.
_WORK_SIZE = 256

#: A pixel at or below this intensity counts as background/air.
_BACKGROUND_LEVEL = 0.10

#: Border strip thickness as a fraction of each side, used for the edge means.
_EDGE_BAND = 0.06


@dataclass(frozen=True)
class DicomMetadata:
    """The acquisition tags relevant to routing. Empty when the file is not DICOM."""

    present: bool = False
    modality: str = ""              # (0008,0060) Modality — "MR", "CR", "DX", "CT", ...
    body_part: str = ""             # (0018,0015) BodyPartExamined — "CHEST", "HEAD", ...
    study_description: str = ""
    series_description: str = ""
    protocol_name: str = ""

    @property
    def text_blob(self) -> str:
        """Free-text fields joined, uppercased — for body-region keyword matching.

        ``BodyPartExamined`` is frequently absent in the wild even when the study
        description says "MRI BRAIN W/O CONTRAST", so the descriptions are a real
        secondary source rather than a nicety.
        """
        return " ".join(
            p for p in (self.body_part, self.study_description,
                        self.series_description, self.protocol_name) if p
        ).upper()

    def to_dict(self) -> dict[str, Any]:
        if not self.present:
            return {"present": False}
        return {"present": True, "modality": self.modality or "(absent)",
                "body_part": self.body_part or "(absent)",
                "study_description": self.study_description,
                "series_description": self.series_description}


@dataclass(frozen=True)
class PixelFeatures:
    """Geometry and tone statistics of the decoded image."""

    decodable: bool = False
    height: int = 0
    width: int = 0
    aspect_ratio: float = 0.0

    #: Mean per-pixel chroma, max(R,G,B)-min(R,G,B). Radiographs and MR are
    #: grayscale (~0.00); fundus photographs and screenshots are not.
    mean_saturation: float = 0.0
    is_grayscale: bool = True

    #: Fraction of pixels at or below the background level. Real CXR: median 0.12,
    #: p95 0.33. Axial head CT/MR: 0.57-0.65.
    background_fraction: float = 0.0

    #: Brightest of the four border strips. THE discriminator between a frame-filling
    #: radiograph and a study floating in air. Real CXR: p1 0.12, median 0.72. Axial
    #: head CT/MR: 0.00-0.14.
    edge_max: float = 0.0
    #: How many of the four border strips are essentially black (< 0.06). All four
    #: dark means the subject does not touch any frame edge.
    dark_edge_count: int = 0

    #: Area of the foreground bounding box / frame area. Real CXR: p1 0.67, median
    #: 0.98 (anatomy spans the frame). Axial head studies: 0.47-0.79.
    foreground_bbox_fraction: float = 0.0
    #: Fraction of the whole frame that is foreground. CXR median 0.76; head 0.33-0.38.
    foreground_fill: float = 0.0
    #: Foreground pixels / bounding-box area — how solidly the subject fills its own
    #: box. A head cross-section is a convex oval and scores high inside a small box.
    foreground_occupancy: float = 0.0
    #: Distance of the foreground centroid from frame centre, in frame widths.
    centroid_offset: float = 0.0

    #: Mean of the central vertical third / mean of the outer thirds. High for chest
    #: films (mediastinum + spine) but ALSO high for head cross-sections (subject
    #: centred in air), so it is diagnostic of "not a flat photo", not of chest.
    center_ratio: float = 0.0
    #: 256-bin histogram entropy. Radiographs sit at 6-7.5 bits; graphics and
    #: screenshots fall below 4.
    tonal_entropy_bits: float = 0.0
    gray_std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        if not self.decodable:
            return {"decodable": False}
        return {
            "decodable": True,
            "size": [self.height, self.width],
            "aspect_ratio": round(self.aspect_ratio, 3),
            "mean_saturation": round(self.mean_saturation, 4),
            "background_fraction": round(self.background_fraction, 3),
            "edge_max": round(self.edge_max, 3),
            "dark_edge_count": self.dark_edge_count,
            "foreground_bbox_fraction": round(self.foreground_bbox_fraction, 3),
            "foreground_fill": round(self.foreground_fill, 3),
            "foreground_occupancy": round(self.foreground_occupancy, 3),
            "centroid_offset": round(self.centroid_offset, 3),
            "center_ratio": round(self.center_ratio, 3),
            "tonal_entropy_bits": round(self.tonal_entropy_bits, 2),
        }


@dataclass(frozen=True)
class ImageFingerprint:
    """Everything the detector knows about one upload before scoring it."""

    path: Path
    dicom: DicomMetadata = field(default_factory=DicomMetadata)
    pixels: PixelFeatures = field(default_factory=PixelFeatures)
    #: Set when nothing could be read at all.
    error: str = ""

    @property
    def readable(self) -> bool:
        return self.pixels.decodable

    def to_dict(self) -> dict[str, Any]:
        out = {"dicom": self.dicom.to_dict(), "pixels": self.pixels.to_dict()}
        if self.error:
            out["error"] = self.error
        return out


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _normalize01(a: np.ndarray) -> np.ndarray:
    """Percentile-clipped rescale to [0,1] — matches ``services.vision.io``.

    Using the same normalisation as the production CXR loader keeps the detector's
    view of an image identical to the engine's, so a threshold measured here means
    the same thing downstream.
    """
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _read_dicom(path: Path) -> tuple[DicomMetadata, np.ndarray | None]:
    """Read DICOM tags and pixels. Returns ``(empty metadata, None)`` if not DICOM."""
    try:
        import pydicom
    except ImportError:
        log.debug("pydicom unavailable; DICOM metadata channel disabled")
        return DicomMetadata(), None

    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception:
        return DicomMetadata(), None                    # not a DICOM — caller falls back

    meta = DicomMetadata(
        present=True,
        modality=str(getattr(ds, "Modality", "") or "").upper().strip(),
        body_part=str(getattr(ds, "BodyPartExamined", "") or "").upper().strip(),
        study_description=str(getattr(ds, "StudyDescription", "") or "").strip(),
        series_description=str(getattr(ds, "SeriesDescription", "") or "").strip(),
        protocol_name=str(getattr(ds, "ProtocolName", "") or "").strip(),
    )

    # Pixels are best-effort: a header-only or unsupported-transfer-syntax file still
    # yields usable metadata, which for routing is the stronger signal anyway.
    try:
        arr = ds.pixel_array
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):        # colour DICOM
            arr = arr[..., :3].mean(axis=-1)
        while arr.ndim > 2:                                   # multi-frame: middle slice
            arr = arr[arr.shape[0] // 2]
        try:
            from pydicom.pixels import apply_voi_lut
        except ImportError:                                   # pydicom < 3
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass                                              # no LUT — raw values are fine
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr                             # invert so bone is bright
        return meta, np.asarray(arr, dtype=np.float32)
    except Exception as exc:
        log.debug(f"DICOM pixels unreadable ({type(exc).__name__}); metadata retained")
        return meta, None


def _read_plain_image(path: Path) -> tuple[np.ndarray | None, float]:
    """Decode PNG/JPEG/TIFF. Returns ``(grayscale, mean_saturation)``."""
    try:
        from PIL import Image
    except ImportError:                                       # pragma: no cover
        return None, 0.0
    try:
        img = Image.open(path)
        img.draft("RGB", (_WORK_SIZE * 2, _WORK_SIZE * 2))    # fast JPEG downscale
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None, 0.0
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return rgb.mean(axis=2), float(chroma.mean())


def _downscale(a: np.ndarray, size: int = _WORK_SIZE) -> np.ndarray:
    """Area-average down to ``size`` on the long edge, preserving aspect ratio."""
    h, w = a.shape[:2]
    if max(h, w) <= size:
        return a
    scale = size / float(max(h, w))
    th, tw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        import cv2

        return cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Block-mean fallback: still averages (unlike point sampling), so the tone
        # statistics stay comparable to the OpenCV path.
        ys = np.linspace(0, h, th + 1).astype(int)
        xs = np.linspace(0, w, tw + 1).astype(int)
        return np.array([[a[ys[i]:max(ys[i] + 1, ys[i + 1]),
                            xs[j]:max(xs[j] + 1, xs[j + 1])].mean()
                          for j in range(tw)] for i in range(th)], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _entropy_bits(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _measure(gray: np.ndarray, mean_saturation: float) -> PixelFeatures:
    """Compute the full feature bundle from a [0,1] grayscale image."""
    h, w = gray.shape
    band_h = max(1, int(h * _EDGE_BAND))
    band_w = max(1, int(w * _EDGE_BAND))
    edges = np.array([
        gray[:band_h, :].mean(), gray[-band_h:, :].mean(),
        gray[:, :band_w].mean(), gray[:, -band_w:].mean(),
    ], dtype=np.float32)

    # Foreground threshold is relative to the image's own exposure, so it survives
    # the wide brightness variation between scanners and export pipelines. The 0.12
    # floor stops a near-black frame from calling its own noise "foreground".
    fg_threshold = max(0.12, float(gray.mean()) * 0.45)
    fg = gray > fg_threshold
    ys, xs = np.nonzero(fg)
    if ys.size:
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        bbox_fraction = bbox_area / float(h * w)
        occupancy = float(fg.sum()) / float(max(1, bbox_area))
        centroid_offset = float(np.hypot(ys.mean() / h - 0.5, xs.mean() / w - 0.5))
    else:
        bbox_fraction = occupancy = 0.0
        centroid_offset = 0.0

    third = max(1, w // 3)
    lateral = (gray[:, :third].mean() + gray[:, 2 * third:].mean()) / 2.0
    center_ratio = float(gray[:, third:2 * third].mean() / max(1e-6, lateral))

    return PixelFeatures(
        decodable=True,
        height=h,
        width=w,
        aspect_ratio=h / max(1, w),
        mean_saturation=mean_saturation,
        is_grayscale=mean_saturation <= 0.08,
        background_fraction=float(np.mean(gray <= _BACKGROUND_LEVEL)),
        edge_max=float(edges.max()),
        dark_edge_count=int((edges < 0.06).sum()),
        foreground_bbox_fraction=float(bbox_fraction),
        foreground_fill=float(fg.mean()),
        foreground_occupancy=float(occupancy),
        centroid_offset=centroid_offset,
        center_ratio=center_ratio,
        tonal_entropy_bits=_entropy_bits(gray),
        gray_std=float(gray.std()),
    )


def fingerprint(path: str | Path) -> ImageFingerprint:
    """Extract the full evidence bundle for one file. Never raises.

    DICOM is tried first (by content, not extension — DICOM files frequently arrive
    with no suffix at all), falling back to plain-image decoding.
    """
    p = Path(path)
    meta, arr = _read_dicom(p)
    saturation = 0.0

    if arr is None:
        arr, saturation = _read_plain_image(p)

    if arr is None:
        reason = ("DICOM header parsed but pixel data is unreadable"
                  if meta.present else "file could not be decoded as an image")
        log.info("fingerprint failed", extra={"context": {"reason": reason}})
        return ImageFingerprint(path=p, dicom=meta, error=reason)

    gray = _normalize01(_downscale(np.asarray(arr, dtype=np.float32)))
    return ImageFingerprint(path=p, dicom=meta, pixels=_measure(gray, saturation))
