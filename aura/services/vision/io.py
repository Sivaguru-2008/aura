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


def study_from_cxr(
    path: str | Path,
    study_id: str | None = None,
    priors: StructuredPriors | None = None,
    grid: int = 64,
) -> StudyInput:
    """Build a StudyInput from a real radiograph.

    The full image drives the CNN (it resizes internally to 224); a ``grid``x``grid``
    downscale is stored in ``image``/``image_shape`` so the existing overlay/feature
    fallbacks keep working. This is the PACS -> pipeline seam.
    """
    import numpy as _np

    full = load_cxr(path)
    rows = _np.linspace(0, full.shape[0] - 1, grid).astype(int)
    cols = _np.linspace(0, full.shape[1] - 1, grid).astype(int)
    small = full[_np.ix_(rows, cols)]
    return StudyInput(
        study_id=study_id or Path(path).stem,
        modality=Modality.CXR,
        image=[float(v) for v in small.flatten()],
        image_shape=(grid, grid),
        priors=priors or StructuredPriors(),
    )
