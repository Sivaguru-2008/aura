"""Real chest-radiograph loading (DICOM / PNG / JPG) -> normalized array.

The synthetic world hands the pipeline a 64x64 array; production intake is a DICOM
from PACS or an exported PNG. This module turns either into a grayscale float image
in [0,1] the CNN backbone can consume, handling the parts that actually bite in the
clinic: DICOM VOI-LUT windowing and MONOCHROME1 inversion. Optional deps (pydicom,
opencv/PIL) are imported lazily so the offline demo never requires them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from schemas.clinical import Modality
from schemas.contracts import StructuredPriors, StudyInput


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() + 1e-6)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def load_dicom(path: str | Path) -> np.ndarray:
    """DICOM -> [0,1] grayscale, applying VOI LUT and MONOCHROME1 inversion."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr                     # invert so bone is bright
    return _normalize01(arr)


def load_image(path: str | Path) -> np.ndarray:
    """PNG/JPG/TIFF -> [0,1] grayscale."""
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cv2 could not read {path}")
        return _normalize01(img)
    except ImportError:
        from PIL import Image

        return _normalize01(np.asarray(Image.open(path).convert("L")))


def load_cxr(path: str | Path) -> np.ndarray:
    """Dispatch on extension. Returns full-resolution [0,1] grayscale."""
    ext = Path(path).suffix.lower()
    if ext in (".dcm", ".dicom", ""):
        try:
            return load_dicom(path)
        except Exception:
            return load_image(path)
    return load_image(path)


#: Default grid the CXR intake produces. 224 == the DenseNet-121 input resolution,
#: so the film reaches the CNN at full fidelity (audit F5). Overridable via
#: ``AURA_IMAGE_GRID`` for callers that deliberately want a smaller stored image.
DEFAULT_GRID = 224


def _resize_grid(full: np.ndarray, grid: int) -> np.ndarray:
    """Downscale ``full`` to ``grid``×``grid`` with anti-aliasing.

    Uses OpenCV ``INTER_AREA`` (area averaging) — the correct kernel for
    *downscaling*, which integrates every source pixel instead of point-sampling a
    sparse lattice. The previous ``linspace`` index-selection kept only ``grid**2``
    of ~millions of pixels and dropped the rest, erasing thin structures like a
    pneumothorax pleural line or a small nodule before the CNN ever saw them
    (audit F5). Falls back to stride sampling only if OpenCV is unavailable.
    """
    if full.shape[:2] == (grid, grid):
        return full
    try:
        import cv2

        interp = cv2.INTER_AREA if min(full.shape[:2]) >= grid else cv2.INTER_LINEAR
        return cv2.resize(full.astype(np.float32), (grid, grid), interpolation=interp)
    except ImportError:                          # numpy-only fallback (offline demo)
        rows = np.linspace(0, full.shape[0] - 1, grid).astype(int)
        cols = np.linspace(0, full.shape[1] - 1, grid).astype(int)
        return full[np.ix_(rows, cols)]


def study_from_cxr(
    path: str | Path,
    study_id: str | None = None,
    priors: StructuredPriors | None = None,
    grid: int | None = None,
) -> StudyInput:
    """Build a StudyInput from a real radiograph.

    The film is area-averaged down to ``grid``×``grid`` (default 224 = the CNN's
    native input) and that image both drives the CNN and is stored in
    ``image``/``image_shape``, so saliency is computed on exactly what the model
    saw. This is the PACS -> pipeline seam. A smaller ``grid`` may be requested to
    shrink the stored bundle, at the cost of fidelity for small findings.
    """
    if grid is None:
        import os
        grid = int(os.environ.get("AURA_IMAGE_GRID", DEFAULT_GRID))

    full = load_cxr(path)
    small = _resize_grid(full, grid)
    return StudyInput(
        study_id=study_id or Path(path).stem,
        modality=Modality.CXR,
        image=[float(v) for v in small.flatten()],
        image_shape=(grid, grid),
        priors=priors or StructuredPriors(),
    )
