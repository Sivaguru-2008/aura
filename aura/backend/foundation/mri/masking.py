"""Masks: the head/foreground estimate, and the slot a real brain mask goes into.

The distinction this module enforces is the reason it exists as its own file.

A threshold-and-largest-component mask captures the **head**: brain, CSF, skull,
scalp, orbital fat, and usually some neck. It is cheap, deterministic, robust, and
genuinely useful — it drives cropping and supplies the air region the noise estimate
needs. It is **not** a brain mask, and treating it as one would inflate every
downstream volumetric measurement by tens of percent while producing numbers that look
entirely reasonable. Intracranial volume from a head mask is wrong by roughly the
volume of the skull and scalp, and nothing downstream can detect that.

So :class:`BrainMaskSlot` carries provenance, and ``is_brain_mask`` is ``True`` only
when a real skull stripper or an external mask filled it. Anything consuming a mask
can and should check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aura.backend.core.shared.logging import get_logger
from .types import MaskProvenance

log = get_logger("foundation.mri.masking")

#: Histogram resolution for the Otsu threshold. 256 is the conventional choice and is
#: ample: the threshold separates air from tissue, a split with enormous margin.
_HISTOGRAM_BINS = 256


def otsu_threshold(values: np.ndarray, bins: int = _HISTOGRAM_BINS) -> float:
    """Otsu's threshold: the intensity that maximises between-class variance.

    Chosen over a fixed percentile because the air/tissue split in MR is genuinely
    bimodal and its location moves with receiver gain, coil, and sequence — a fixed
    cut that works on a T1 fails on a FLAIR from the same scanner.

    Returns the volume's minimum when the data is constant, which yields an
    all-foreground mask; the intensity check catches a constant volume separately and
    with a much better error message than a mask function could.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    low, high = float(finite.min()), float(finite.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return low

    counts, edges = np.histogram(finite, bins=bins, range=(low, high))
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return low

    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(counts)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return low

    cumulative_mean = np.cumsum(counts * centres)
    grand_mean = cumulative_mean[-1]
    mean_bg = np.divide(cumulative_mean, weight_bg, out=np.zeros_like(cumulative_mean),
                        where=weight_bg > 0)
    mean_fg = np.divide(grand_mean - cumulative_mean, weight_fg,
                        out=np.zeros_like(cumulative_mean), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def estimate_foreground_mask(array: np.ndarray, *, largest_component: bool = True,
                             fill_holes: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the head/foreground mask. **Not a brain mask** — see the module docstring.

    Three steps, each of which earns its place:

    1. Otsu threshold — separates tissue from air.
    2. Largest connected component — discards coil elements, the head holder, and
       noise speckle, which threshold alone keeps.
    3. Hole filling — the marrow-free skull interior and dark CSF spaces threshold out
       and would otherwise punch holes through the middle of the head.

    Steps 2 and 3 need ``scipy.ndimage``. Without it the threshold mask is returned
    and the degradation is reported in the returned details rather than hidden.
    """
    finite = np.isfinite(array)
    threshold = otsu_threshold(array[finite]) if finite.any() else float("nan")
    if not np.isfinite(threshold):
        return np.zeros(array.shape, dtype=bool), {
            "method": "otsu", "threshold": None,
            "note": "the volume has no finite voxels; the mask is empty"}

    mask = np.zeros(array.shape, dtype=bool)
    np.greater(array, threshold, out=mask, where=finite)
    details: dict[str, Any] = {"method": "otsu", "threshold": round(float(threshold), 6),
                              "largest_component": False, "holes_filled": False}

    try:
        from scipy import ndimage
    except ImportError:                              # pragma: no cover - env specific
        details["note"] = ("scipy.ndimage is unavailable; the mask is a bare threshold "
                           "and may include disconnected coil or noise regions")
        details["voxels"] = int(mask.sum())
        return mask, details

    if largest_component and mask.any():
        labels, count = ndimage.label(mask)
        if count > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0                              # background label
            mask = labels == int(np.argmax(sizes))
            details["largest_component"] = True
            details["components_found"] = int(count)
    if fill_holes and mask.any():
        mask = ndimage.binary_fill_holes(mask)
        details["holes_filled"] = True

    details["voxels"] = int(mask.sum())
    details["coverage_fraction"] = round(float(mask.sum() / mask.size), 6)
    return mask, details


@dataclass
class BrainMaskSlot:
    """Where a brain mask lives on a foundation study, and what it actually is.

    Empty by default. The pipeline fills it with a foreground estimate (clearly
    labelled) when configured to, and a real skull stripper would replace it with
    ``provenance=SKULL_STRIPPED``. Consumers that need a true brain mask check
    :attr:`is_brain_mask` and get a defensible answer either way.
    """

    mask: np.ndarray | None = None
    provenance: MaskProvenance = MaskProvenance.ABSENT
    method: str = ""
    details: dict[str, Any] | None = None

    @property
    def present(self) -> bool:
        return self.mask is not None

    @property
    def is_brain_mask(self) -> bool:
        """True only for a mask that actually excludes skull and scalp."""
        return self.provenance in (MaskProvenance.SKULL_STRIPPED,
                                   MaskProvenance.EXTERNAL)

    @property
    def coverage_fraction(self) -> float | None:
        if self.mask is None:
            return None
        return float(self.mask.sum() / self.mask.size)

    def bounding_box(self) -> tuple[slice, slice, slice] | None:
        """Tight bounding box of the mask, as slices. ``None`` when empty."""
        if self.mask is None or not self.mask.any():
            return None
        bounds = []
        for axis in range(3):
            other = tuple(a for a in range(3) if a != axis)
            projection = self.mask.any(axis=other)
            indices = np.flatnonzero(projection)
            bounds.append(slice(int(indices[0]), int(indices[-1]) + 1))
        return tuple(bounds)                          # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "provenance": self.provenance.value,
            "is_brain_mask": self.is_brain_mask,
            "method": self.method,
            "coverage_fraction": (round(self.coverage_fraction, 6)
                                  if self.coverage_fraction is not None else None),
            "details": dict(self.details or {}),
        }


__all__ = ["BrainMaskSlot", "estimate_foreground_mask", "otsu_threshold"]
