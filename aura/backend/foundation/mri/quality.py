"""MRI Quality Inspector — module 4.

Seven checks over one volume, its geometry, and its integrity report. Every number in
the output is measured from the data; nothing is sampled, assumed, or defaulted.

The calibration split
---------------------
Checks fall into two classes and the report never blurs them:

* **Calibrated** — grounded in physics, anatomy, or arithmetic. A 12 mm voxel is not
  a brain MRI; a volume with one distinct intensity carries no information; a slice
  gap that is twice the median spacing means a slice is missing. These can fail a
  study.
* **Provisional** — motion and SNR heuristics whose thresholds were *not* fitted on a
  labelled corpus, because no labelled motion corpus was available in this
  deployment. They are computed, reported, and allowed to warn. They can never
  reject, and :func:`_check` enforces that structurally rather than by convention: a
  check constructed with ``calibrated=False`` cannot hold ``FAIL``.

That is the same posture the modality router takes with its uncalibrated pixel-only
path, and it exists for the same reason. A number nobody validated must not be able to
look like a number somebody did.

Method notes
------------
*Noise / SNR*: magnitude MR background follows a Rayleigh distribution, whose standard
deviation is ``0.655 sigma`` of the underlying Gaussian noise. The estimator divides
the measured background standard deviation by that factor and forms
``mean(foreground) / sigma`` — the NEMA-style two-region SNR. It needs air in the
field of view, so it reports ``NOT_EVALUATED`` on a cropped or skull-stripped volume
instead of returning a meaningless number.

*Motion*: two physical proxies. Adjacent slices through a head correlate strongly
because anatomy changes slowly through-plane, so a correlation drop marks inter-slice
displacement. And motion ghosts replicate the object along the **phase-encode**
direction only, so structured signal in the air bands aligned with phase-encode,
relative to the frequency-encode bands, is the ghosting signature — the same
measurement ACR phantom QA uses. Both are honest measurements; only their thresholds
are provisional.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.core.shared.logging import get_logger
from backend.foundation.mri.config import QualityThresholds
from backend.foundation.mri.io.base import SeriesIntegrity
from backend.foundation.mri.masking import estimate_foreground_mask
from backend.foundation.mri.metadata import MRIMetadata
from backend.foundation.mri.types import CheckStatus, QualityVerdict
from backend.foundation.mri.volume import MRIVolume

log = get_logger("foundation.mri.quality")

#: Rayleigh sigma factor: std(Rayleigh) = sqrt(2 - pi/2) * sigma = 0.6551 * sigma.
_RAYLEIGH_STD_FACTOR = float(np.sqrt(2.0 - np.pi / 2.0))

#: Relative weight of each check in the aggregate score. Structural integrity and
#: geometry weigh most because a downstream model can tolerate a noisy volume far
#: better than a volume with the wrong orientation or a missing slice.
CHECK_WEIGHTS: dict[str, float] = {
    "slice_completeness": 2.0,
    "orientation": 2.0,
    "resolution": 1.5,
    "field_of_view": 1.0,
    "intensity": 1.5,
    "noise": 1.0,
    "motion": 1.0,
}


class QualityCheck(BaseModel):
    """One check's verdict, score, and the values it measured."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: CheckStatus
    score: float = Field(..., ge=0.0, le=1.0)
    calibrated: bool = Field(
        ..., description="True when this check's thresholds are grounded in physics, "
                         "anatomy, or arithmetic. False marks a provisional heuristic "
                         "that may warn but can never reject a study.")
    message: str = ""
    measured: dict[str, Any] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAIL


class QualityReport(BaseModel):
    """Study-level quality conclusion for one series."""

    model_config = ConfigDict(frozen=True)

    quality_score: float = Field(..., ge=0.0, le=1.0)
    verdict: QualityVerdict
    checks: tuple[QualityCheck, ...] = ()
    warnings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    reject_reason: str | None = None

    @property
    def acceptable(self) -> bool:
        return self.verdict is not QualityVerdict.REJECTED

    def check(self, name: str) -> QualityCheck | None:
        return next((c for c in self.checks if c.name == name), None)

    def summary(self) -> str:
        failed = [c.name for c in self.checks if c.status is CheckStatus.FAIL]
        warned = [c.name for c in self.checks if c.status is CheckStatus.WARN]
        parts = [f"quality {self.quality_score:.2f} ({self.verdict.value})"]
        if failed:
            parts.append(f"failed: {', '.join(failed)}")
        if warned:
            parts.append(f"warnings: {', '.join(warned)}")
        return "; ".join(parts)


def _check(name: str, status: CheckStatus, score: float, *, calibrated: bool,
           message: str, **measured: Any) -> QualityCheck:
    """Build a check, enforcing that a provisional check cannot fail a study.

    The clamp is here rather than in the caller so that adding a new uncalibrated
    check cannot accidentally gain reject authority.
    """
    if not calibrated and status is CheckStatus.FAIL:
        status = CheckStatus.WARN
        message = (f"{message} (reported as a warning: this check's thresholds are "
                   "provisional and are not permitted to reject a study)")
    return QualityCheck(name=name, status=status, score=float(np.clip(score, 0.0, 1.0)),
                        calibrated=calibrated, message=message, measured=measured)


class MRIQualityInspector:
    """Runs the quality checks. Thresholds are injected; the inspector holds no state."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._t = thresholds or QualityThresholds()

    @property
    def thresholds(self) -> QualityThresholds:
        return self._t

    # ------------------------------------------------------------------ #
    def inspect(self, volume: MRIVolume, *, metadata: MRIMetadata | None = None,
                integrity: SeriesIntegrity | None = None,
                mask: np.ndarray | None = None) -> QualityReport:
        """Assess one volume.

        Args:
            volume: the built volume, before standardisation. Inspecting the *source*
                volume is deliberate: resampling and normalisation change every
                intensity statistic, so a report computed afterwards describes the
                pipeline's output rather than the acquisition.
            metadata: acquisition metadata, when available. Supplies the phase-encode
                direction the ghosting check needs and the recorded slice thickness.
            integrity: the reader's structural findings. Without it, slice
                completeness reports ``NOT_EVALUATED`` rather than assuming success.
            mask: a precomputed foreground mask. Estimated when omitted.
        """
        array = volume.array
        if mask is None:
            mask, _ = estimate_foreground_mask(array)

        checks = [
            self._slice_completeness(volume, integrity),
            self._orientation(volume, metadata),
            self._resolution(volume, metadata),
            self._field_of_view(volume),
            self._intensity(array),
            self._noise(array, mask),
            self._motion(array, mask, metadata),
        ]

        score = self._aggregate(checks)
        verdict, reject_reason = self._verdict(checks, score)
        warnings = tuple(c.message for c in checks
                         if c.status in (CheckStatus.WARN, CheckStatus.FAIL) and c.message)
        report = QualityReport(
            quality_score=round(score, 4),
            verdict=verdict,
            checks=tuple(checks),
            warnings=warnings,
            recommendations=self._recommendations(checks),
            reject_reason=reject_reason,
        )
        log.info("quality inspection complete",
                 extra={"context": {"score": report.quality_score,
                                    "verdict": verdict.value,
                                    "failed": [c.name for c in checks if c.failed]}})
        return report

    # ------------------------------------------------------------------ #
    # 1. Slice completeness
    # ------------------------------------------------------------------ #
    def _slice_completeness(self, volume: MRIVolume,
                            integrity: SeriesIntegrity | None) -> QualityCheck:
        slices = int(volume.shape[2])
        if integrity is None:
            return _check("slice_completeness", CheckStatus.NOT_EVALUATED, 0.0,
                          calibrated=True,
                          message="no reader integrity report was supplied, so slice "
                                  "completeness could not be assessed",
                          slices=slices)

        missing = integrity.missing_slices_estimated
        corrupt = len(integrity.corrupt_files)
        loss = integrity.loss_fraction

        if slices < self._t.min_slices:
            return _check("slice_completeness", CheckStatus.FAIL, 0.0, calibrated=True,
                          message=f"the volume has {slices} slices, below the {self._t.min_slices} "
                                  "needed to span a brain",
                          slices=slices, missing=missing, corrupt=corrupt)
        if missing or corrupt:
            status = CheckStatus.FAIL if loss > 0.10 else CheckStatus.WARN
            return _check(
                "slice_completeness", status, max(0.0, 1.0 - 2.0 * loss),
                calibrated=True,
                message=(f"{missing} slice(s) appear absent and {corrupt} could not be "
                         f"decoded ({loss:.1%} of the acquisition); measurements that "
                         "span the missing region will be wrong"),
                slices=slices, missing=missing, corrupt=corrupt,
                loss_fraction=round(loss, 4))
        if integrity.duplicate_positions or not integrity.spacing_consistent:
            return _check(
                "slice_completeness", CheckStatus.WARN, 0.70, calibrated=True,
                message=(f"{integrity.duplicate_positions} duplicate slice position(s) "
                         f"and {len(integrity.irregular_gaps_mm)} irregular gap(s) were "
                         "found; the volume is not uniformly sampled"),
                slices=slices, duplicates=integrity.duplicate_positions,
                irregular_gaps=len(integrity.irregular_gaps_mm))
        return _check("slice_completeness", CheckStatus.PASS, 1.0, calibrated=True,
                      message="", slices=slices,
                      median_spacing_mm=integrity.median_slice_spacing_mm)

    # ------------------------------------------------------------------ #
    # 2. Orientation
    # ------------------------------------------------------------------ #
    def _orientation(self, volume: MRIVolume,
                     metadata: MRIMetadata | None) -> QualityCheck:
        geometry = volume.geometry
        codes = geometry.axis_codes
        measured = {"orientation": geometry.orientation,
                    "plane": geometry.plane.value,
                    "obliquity_deg": round(geometry.obliquity_deg, 3),
                    "determinant": round(float(np.linalg.det(geometry.affine[:3, :3])), 6)}

        if geometry.degenerate:
            return _check("orientation", CheckStatus.FAIL, 0.0, calibrated=True,
                          message="the affine is singular; this volume has no valid "
                                  "position or orientation in the world",
                          **measured)
        if "?" in codes:
            return _check("orientation", CheckStatus.FAIL, 0.0, calibrated=True,
                          message="one or more voxel axes could not be assigned to a "
                                  "world axis; the orientation is indeterminate",
                          **measured)

        source = (metadata.identifiers.source_name or "") if metadata else ""
        if metadata is not None and not metadata.geometry.world_orientation_known:
            return _check(
                "orientation", CheckStatus.FAIL, 0.15, calibrated=True,
                message="the source file does not record a world orientation, so "
                        "left and right cannot be established. Any lateralised "
                        "finding from this volume would be unsafe",
                source=source, **measured)

        if geometry.obliquity_deg > 20.0:
            return _check(
                "orientation", CheckStatus.WARN, 0.75, calibrated=True,
                message=(f"the acquisition is {geometry.obliquity_deg:.1f} deg oblique to "
                         "the cardinal axes; reorientation to canonical is a "
                         "permutation only and leaves the obliquity in the affine"),
                **measured)
        return _check("orientation", CheckStatus.PASS, 1.0, calibrated=True,
                      message="", **measured)

    # ------------------------------------------------------------------ #
    # 3. Resolution
    # ------------------------------------------------------------------ #
    def _resolution(self, volume: MRIVolume,
                    metadata: MRIMetadata | None) -> QualityCheck:
        spacing = volume.spacing
        in_plane = sorted(spacing)[:2]
        through_plane = max(spacing)
        anisotropy = volume.geometry.anisotropy
        measured = {"spacing_mm": [round(v, 4) for v in spacing],
                    "anisotropy": round(anisotropy, 3),
                    "voxel_volume_mm3": round(volume.geometry.voxel_volume_mm3, 5)}

        if any(not np.isfinite(s) or s <= 0 for s in spacing):
            return _check("resolution", CheckStatus.FAIL, 0.0, calibrated=True,
                          message="voxel spacing is zero or non-finite along at least "
                                  "one axis", **measured)
        if any(s < self._t.min_voxel_mm for s in in_plane):
            return _check(
                "resolution", CheckStatus.FAIL, 0.0, calibrated=True,
                message=(f"in-plane voxel size {min(in_plane):.3f} mm is below "
                         f"{self._t.min_voxel_mm} mm, which no clinical MR scanner "
                         "achieves; the spacing metadata is likely in the wrong units"),
                **measured)
        if any(s > self._t.max_voxel_mm for s in in_plane):
            return _check(
                "resolution", CheckStatus.FAIL, 0.10, calibrated=True,
                message=(f"in-plane voxel size {max(in_plane):.2f} mm exceeds the "
                         f"{self._t.max_voxel_mm} mm limit for brain MRI"),
                **measured)
        if through_plane > self._t.max_slice_thickness_mm:
            return _check(
                "resolution", CheckStatus.FAIL, 0.15, calibrated=True,
                message=(f"slice spacing {through_plane:.2f} mm exceeds "
                         f"{self._t.max_slice_thickness_mm} mm; the volume cannot "
                         "support 3D analysis"), **measured)

        notes: list[str] = []
        score = 1.0
        if through_plane > self._t.warn_slice_thickness_mm:
            notes.append(f"slice spacing is {through_plane:.2f} mm")
            score -= 0.25
        if anisotropy > self._t.warn_anisotropy:
            notes.append(f"voxels are {anisotropy:.1f}:1 anisotropic, so this is "
                         "effectively a 2D multi-slice acquisition")
            score -= 0.25
        if notes:
            return _check("resolution", CheckStatus.WARN, score, calibrated=True,
                          message="; ".join(notes) +
                                  " — resampling to isotropic will be interpolation-"
                                  "dominated through-plane", **measured)
        return _check("resolution", CheckStatus.PASS, 1.0, calibrated=True, message="",
                      **measured)

    # ------------------------------------------------------------------ #
    # 4. Field of view
    # ------------------------------------------------------------------ #
    def _field_of_view(self, volume: MRIVolume) -> QualityCheck:
        extent = volume.geometry.field_of_view_mm
        measured = {"field_of_view_mm": [round(v, 2) for v in extent],
                    "shape": list(volume.shape)}

        if min(extent) < self._t.min_head_fov_mm:
            axis = int(np.argmin(extent))
            # The through-plane axis is legitimately short on a targeted slab (a
            # pituitary or IAC protocol), so that case warns; a short in-plane axis
            # means the head does not fit in the field of view.
            status = CheckStatus.WARN if axis == 2 else CheckStatus.FAIL
            return _check(
                "field_of_view", status, 0.35, calibrated=True,
                message=(f"the field of view spans only {min(extent):.0f} mm along axis "
                         f"{axis}, below the {self._t.min_head_fov_mm:.0f} mm an adult "
                         "head requires" + (
                             "; this is consistent with a targeted slab rather than a "
                             "whole-brain acquisition" if axis == 2 else
                             "; the head does not fit in the acquired field of view")),
                **measured)
        if max(extent) > self._t.max_head_fov_mm:
            return _check(
                "field_of_view", CheckStatus.WARN, 0.60, calibrated=True,
                message=(f"the field of view spans {max(extent):.0f} mm, beyond the "
                         f"{self._t.max_head_fov_mm:.0f} mm expected for a head; the "
                         "volume includes anatomy outside the brain or the spacing "
                         "units are wrong"), **measured)
        return _check("field_of_view", CheckStatus.PASS, 1.0, calibrated=True,
                      message="", **measured)

    # ------------------------------------------------------------------ #
    # 5. Intensity
    # ------------------------------------------------------------------ #
    def _intensity(self, array: np.ndarray) -> QualityCheck:
        total = int(array.size)
        finite_mask = np.isfinite(array)
        finite = array[finite_mask]
        non_finite_fraction = 1.0 - (finite.size / total)

        if finite.size == 0:
            return _check("intensity", CheckStatus.FAIL, 0.0, calibrated=True,
                          message="every voxel is NaN or infinite", voxels=total)

        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            return _check("intensity", CheckStatus.FAIL, 0.0, calibrated=True,
                          message=f"every voxel holds the same value ({low:g}); the "
                                  "volume carries no image information",
                          voxels=total, value=low)

        distinct = int(np.unique(finite).size)
        distinct_ratio = distinct / finite.size
        saturated = float(np.count_nonzero(finite >= high - 1e-6) / finite.size)
        negative = float(np.count_nonzero(finite < 0) / finite.size)
        measured = {
            "min": round(low, 6), "max": round(high, 6),
            "mean": round(float(finite.mean()), 6),
            "std": round(float(finite.std()), 6),
            "distinct_values": distinct,
            "distinct_ratio": round(distinct_ratio, 8),
            "saturated_fraction": round(saturated, 6),
            "negative_fraction": round(negative, 6),
            "non_finite_fraction": round(non_finite_fraction, 6),
        }

        if non_finite_fraction > 0.01:
            return _check("intensity", CheckStatus.FAIL, 0.10, calibrated=True,
                          message=f"{non_finite_fraction:.1%} of voxels are NaN or "
                                  "infinite", **measured)
        if distinct_ratio < self._t.min_distinct_value_ratio and distinct < 64:
            return _check(
                "intensity", CheckStatus.FAIL, 0.10, calibrated=True,
                message=(f"the volume holds only {distinct} distinct values; this is a "
                         "label map or a failed export, not an intensity image"),
                **measured)

        notes: list[str] = []
        score = 1.0
        if saturated > self._t.max_saturated_fraction:
            notes.append(f"{saturated:.1%} of voxels sit at the maximum value, so the "
                         "receiver was saturated and the bright end is clipped")
            score -= 0.35
        if negative > 0 and self._t.warn_on_negative_intensities:
            notes.append(f"{negative:.1%} of voxels are negative, which magnitude MR "
                         "cannot be; this is a phase image, a derived map, or a "
                         "volume that has already been normalised")
            score -= 0.15
        if non_finite_fraction > 0:
            notes.append(f"{non_finite_fraction:.2%} of voxels are non-finite")
            score -= 0.10
        if notes:
            return _check("intensity", CheckStatus.WARN, score, calibrated=True,
                          message="; ".join(notes), **measured)
        return _check("intensity", CheckStatus.PASS, 1.0, calibrated=True, message="",
                      **measured)

    # ------------------------------------------------------------------ #
    # 6. Noise / SNR  (provisional thresholds)
    # ------------------------------------------------------------------ #
    def _noise(self, array: np.ndarray, mask: np.ndarray) -> QualityCheck:
        finite = np.isfinite(array)
        background = finite & ~mask
        foreground = finite & mask
        background_fraction = float(background.sum() / array.size)

        if background.sum() < 1000 or background_fraction < 0.01:
            return _check(
                "noise", CheckStatus.NOT_EVALUATED, 0.0, calibrated=False,
                message="too little air is present to estimate noise; the volume is "
                        "cropped, skull-stripped, or masked, and a background SNR "
                        "estimate would be meaningless",
                background_fraction=round(background_fraction, 5))
        if foreground.sum() < 1000:
            return _check("noise", CheckStatus.NOT_EVALUATED, 0.0, calibrated=False,
                          message="too few foreground voxels to estimate signal",
                          foreground_voxels=int(foreground.sum()))

        background_std = float(array[background].std())
        signal = float(np.median(array[foreground]))
        sigma = background_std / _RAYLEIGH_STD_FACTOR
        snr = float(signal / sigma) if sigma > 0 else float("inf")
        measured = {
            "snr": round(snr, 3) if np.isfinite(snr) else None,
            "sigma": round(sigma, 6),
            "background_std": round(background_std, 6),
            "signal_median": round(signal, 6),
            "background_fraction": round(background_fraction, 5),
            "estimator": "NEMA two-region, Rayleigh-corrected",
        }
        if sigma <= 0:
            return _check(
                "noise", CheckStatus.WARN, 0.50, calibrated=False,
                message="the background has zero variance; the volume has been "
                        "denoised, masked, or thresholded, so no acquisition noise "
                        "remains to measure", **measured)
        if snr < self._t.warn_snr:
            return _check(
                "noise", CheckStatus.WARN, float(np.clip(snr / self._t.warn_snr, 0, 1)),
                calibrated=False,
                message=(f"estimated SNR is {snr:.1f}, below the conventional "
                         f"diagnostic floor of {self._t.warn_snr:.0f}"), **measured)
        return _check("noise", CheckStatus.PASS, 1.0, calibrated=False, message="",
                      **measured)

    # ------------------------------------------------------------------ #
    # 7. Motion  (provisional thresholds)
    # ------------------------------------------------------------------ #
    def _motion(self, array: np.ndarray, mask: np.ndarray,
                metadata: MRIMetadata | None) -> QualityCheck:
        correlation = self._slice_correlation(array, mask)
        ghost = self._ghost_ratio(array, mask, metadata)
        gradient = self._edge_energy(array, mask)

        measured: dict[str, Any] = {
            "min_adjacent_slice_correlation": (round(correlation, 4)
                                               if correlation is not None else None),
            "ghost_ratio": round(ghost["ratio"], 4) if ghost["ratio"] is not None else None,
            "phase_encode_direction": ghost["direction"],
            "normalised_edge_energy": (round(gradient, 5)
                                       if gradient is not None else None),
            "note": "provisional heuristics; thresholds are not fitted on a labelled "
                    "motion corpus and this check can warn but never reject",
        }

        if correlation is None and ghost["ratio"] is None:
            return _check("motion", CheckStatus.NOT_EVALUATED, 0.0, calibrated=False,
                          message="the volume is too small or too uniform for the "
                                  "motion proxies to be computed", **measured)

        notes: list[str] = []
        score = 1.0
        if correlation is not None and correlation < self._t.warn_slice_correlation:
            notes.append(
                f"the weakest adjacent-slice correlation is {correlation:.2f}, below "
                f"{self._t.warn_slice_correlation:.2f}; anatomy changes slowly "
                "through-plane, so this is consistent with inter-slice displacement")
            score -= 0.35
        if ghost["ratio"] is not None and ghost["ratio"] > self._t.warn_ghost_ratio:
            notes.append(
                f"background signal along the phase-encode direction is "
                f"{ghost['ratio']:.2f}x that along frequency-encode, the signature of "
                "motion or flow ghosting")
            score -= 0.35
        if notes:
            return _check("motion", CheckStatus.WARN, score, calibrated=False,
                          message="; ".join(notes), **measured)
        return _check("motion", CheckStatus.PASS, 1.0, calibrated=False, message="",
                      **measured)

    @staticmethod
    def _slice_correlation(array: np.ndarray, mask: np.ndarray) -> float | None:
        """Lowest Pearson correlation between adjacent slices that both hold anatomy."""
        if array.shape[2] < 3:
            return None
        correlations: list[float] = []
        for k in range(array.shape[2] - 1):
            a, b = array[:, :, k], array[:, :, k + 1]
            valid = np.isfinite(a) & np.isfinite(b) & (mask[:, :, k] | mask[:, :, k + 1])
            if valid.sum() < 100:
                continue                       # an end slice holding only air
            x, y = a[valid], b[valid]
            sx, sy = float(x.std()), float(y.std())
            if sx <= 0 or sy <= 0:
                continue
            correlations.append(float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)))
        return min(correlations) if correlations else None

    @staticmethod
    def _ghost_ratio(array: np.ndarray, mask: np.ndarray,
                     metadata: MRIMetadata | None) -> dict[str, Any]:
        """Compare air-band signal along phase-encode vs frequency-encode.

        Measured on the slice holding the most anatomy: ghosting is strongest where
        the moving object is largest, and averaging over near-empty end slices would
        dilute it.
        """
        direction = (metadata.acquisition.phase_encoding_direction
                     if metadata else None) or "unknown"
        if array.shape[2] < 1 or not mask.any():
            return {"ratio": None, "direction": direction}

        counts = mask.reshape(-1, mask.shape[2]).sum(axis=0)
        plane_index = int(np.argmax(counts))
        plane, plane_mask = array[:, :, plane_index], mask[:, :, plane_index]
        if not plane_mask.any():
            return {"ratio": None, "direction": direction}

        rows = np.flatnonzero(plane_mask.any(axis=0))
        columns = np.flatnonzero(plane_mask.any(axis=1))
        if rows.size == 0 or columns.size == 0:
            return {"ratio": None, "direction": direction}

        air = np.isfinite(plane) & ~plane_mask
        # Bands that extend from the object along each axis. A ghost is a displaced
        # copy of the object, so it lands inside the band aligned with phase-encode.
        along_i = np.zeros_like(air)
        along_i[:, rows[0]:rows[-1] + 1] = True
        along_j = np.zeros_like(air)
        along_j[columns[0]:columns[-1] + 1, :] = True

        band_i = air & along_i
        band_j = air & along_j
        if band_i.sum() < 100 or band_j.sum() < 100:
            return {"ratio": None, "direction": direction}

        mean_i = float(np.mean(plane[band_i]))
        mean_j = float(np.mean(plane[band_j]))
        if mean_i <= 0 or mean_j <= 0:
            return {"ratio": None, "direction": direction}

        # DICOM 'ROW' means the phase-encode axis runs along a row, i.e. the column
        # index (our i). 'COL' means it runs down a column, i.e. the row index (j).
        if direction.upper() == "ROW":
            ratio = mean_i / mean_j
        elif direction.upper() == "COL":
            ratio = mean_j / mean_i
        else:
            # Without the tag, the best available statement is how asymmetric the two
            # directions are — reported with the direction marked unknown so nobody
            # reads it as a phase-encode measurement.
            ratio = max(mean_i, mean_j) / min(mean_i, mean_j)
        return {"ratio": float(ratio), "direction": direction}

    @staticmethod
    def _edge_energy(array: np.ndarray, mask: np.ndarray) -> float | None:
        """Mean gradient magnitude over anatomy, normalised by mean intensity.

        Reported as a measurement, not scored: motion lowers it and so does a
        low-resolution acquisition, and separating those needs the labelled corpus
        this deployment does not have. It is emitted so that a future calibration has
        the measurement it would need.
        """
        if not mask.any():
            return None
        # ``np.gradient`` needs at least two samples along every axis. A single-slice
        # volume is degenerate rather than exceptional, so this returns "not measured"
        # instead of raising into the middle of a quality report.
        if any(n < 2 for n in array.shape):
            return None
        working = np.where(np.isfinite(array), array, 0.0)
        gradients = np.gradient(working)
        magnitude = np.sqrt(sum(g ** 2 for g in gradients))
        signal = float(np.mean(working[mask]))
        if signal <= 0:
            return None
        return float(np.mean(magnitude[mask]) / signal)

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate(checks: Sequence[QualityCheck]) -> float:
        """Weighted mean over evaluated checks.

        A ``NOT_EVALUATED`` check is excluded from the mean rather than scored zero or
        one. Scoring it zero would punish a study for a check that could not run;
        scoring it one would let an unevaluated check raise the score. Excluding it is
        the only option that says what actually happened.
        """
        weighted = 0.0
        total_weight = 0.0
        for check in checks:
            if check.status is CheckStatus.NOT_EVALUATED:
                continue
            weight = CHECK_WEIGHTS.get(check.name, 1.0)
            weighted += weight * check.score
            total_weight += weight
        return weighted / total_weight if total_weight > 0 else 0.0

    def _verdict(self, checks: Sequence[QualityCheck],
                 score: float) -> tuple[QualityVerdict, str | None]:
        """Decide the verdict. Only calibrated checks may reject."""
        hard_failures = [c for c in checks if c.status is CheckStatus.FAIL and c.calibrated]
        if hard_failures:
            reasons = "; ".join(f"{c.name}: {c.message}" for c in hard_failures)
            return QualityVerdict.REJECTED, reasons
        if score < self._t.reject_below_score:
            return QualityVerdict.REJECTED, (
                f"the aggregate quality score {score:.2f} is below the "
                f"{self._t.reject_below_score:.2f} threshold")
        if score < self._t.warn_below_score or any(
                c.status is CheckStatus.WARN for c in checks):
            return QualityVerdict.ACCEPTABLE_WITH_WARNINGS, None
        return QualityVerdict.ACCEPTABLE, None

    @staticmethod
    def _recommendations(checks: Sequence[QualityCheck]) -> tuple[str, ...]:
        """Actionable next steps, derived from which checks fired.

        Each maps to something a human or a pipeline can actually do. A recommendation
        nobody can act on is noise in a report that must stay readable.
        """
        by_name = {c.name: c for c in checks}
        out: list[str] = []

        completeness = by_name.get("slice_completeness")
        if completeness and completeness.status in (CheckStatus.WARN, CheckStatus.FAIL):
            out.append("re-export the series from the source archive; missing or "
                       "duplicated instances are usually a transfer artefact rather "
                       "than an acquisition failure")

        orientation = by_name.get("orientation")
        if orientation and orientation.status is CheckStatus.FAIL:
            out.append("supply the study in a format that records world orientation "
                       "(DICOM, or NIfTI with sform/qform set) before any lateralised "
                       "interpretation")
        elif orientation and orientation.status is CheckStatus.WARN:
            out.append("resample to the canonical grid if the downstream model assumes "
                       "axis-aligned input; reorientation alone does not remove obliquity")

        resolution = by_name.get("resolution")
        if resolution and resolution.status in (CheckStatus.WARN, CheckStatus.FAIL):
            out.append("resample to isotropic spacing before 3D analysis, and treat "
                       "through-plane measurements as interpolated")

        intensity = by_name.get("intensity")
        if intensity and intensity.status in (CheckStatus.WARN, CheckStatus.FAIL):
            out.append("check the export path for receiver saturation or an applied "
                       "window/level transform; clipped intensities cannot be recovered "
                       "by normalisation")

        noise = by_name.get("noise")
        if noise and noise.status is CheckStatus.WARN:
            out.append("consider a higher-SNR acquisition or averaging; low SNR raises "
                       "the false-positive rate of any subsequent segmentation")

        motion = by_name.get("motion")
        if motion and motion.status is CheckStatus.WARN:
            out.append("have a human confirm motion artefact before analysis, and "
                       "prefer re-acquisition to retrospective correction where the "
                       "study is diagnostic")
        return tuple(out)


__all__ = ["MRIQualityInspector", "QualityCheck", "QualityReport", "CHECK_WEIGHTS"]
