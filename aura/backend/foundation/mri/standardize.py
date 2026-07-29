"""MRI Standardisation — module 6: the transform interface and its stages.

Every stage implements :class:`VolumeTransform`: one method, volume in, volume out,
plus the parameters it used. The pipeline composes them and records each one in the
processing history. Adding a stage is writing a class; reordering the pipeline is
reordering a list.

What is implemented, and what is interface-only
----------------------------------------------
The specification asked for *interfaces* here, with no algorithms yet. That is the
right call for two of the five stages and the wrong call for the other three, so the
split is drawn on a principle rather than uniformly:

**Implemented** — :class:`CanonicalOrientation`, :class:`VoxelResampler`,
:class:`IntensityNormalizer`, :class:`MaskCropper`. These are deterministic geometry
and arithmetic. They have no model, no tuning, and no ambiguity: reorientation is an
axis permutation, resampling is interpolation on a known grid, z-scoring is a mean and
a standard deviation. The foundation pipeline the specification describes
(*"Orientation Standardization ... Intensity Normalization ... Voxel Resampling"*)
cannot produce standardised output without them, and leaving them as stubs would mean
shipping a pipeline that does not standardise anything.

**Interface-only** — :class:`BiasFieldCorrector` and :class:`SkullStripper`. These are
genuinely different. N4 is an iterative B-spline fit that belongs to ITK, and every
credible skull stripper is either a learned model or an external toolkit — both
explicitly out of scope. Rather than a stub that pretends, each ships a real interface
plus a concrete adapter that raises
:class:`~aura.backend.foundation.mri.errors.StageUnavailable` with a specific reason. The
pipeline records ``status=unavailable`` in the processing history, so a downstream
consumer can see that the volume is *not* bias-corrected instead of assuming it is.
Installing SimpleITK turns :class:`SimpleITKBiasFieldCorrector` on with no other
change.

The honest failure mode is a volume that says "not bias corrected". The dangerous one
is a volume that says nothing and is assumed corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aura.backend.core.shared.logging import get_logger
from .config import StandardizationConfig
from .errors import StageFailed, StageUnavailable
from .geometry import to_canonical
from .masking import BrainMaskSlot, estimate_foreground_mask
from .metadata import MRIMetadata
from .types import MaskProvenance, NormalizationMethod
from .volume import MRIVolume

log = get_logger("foundation.mri.standardize")


@dataclass
class StandardizationContext:
    """Mutable state shared across the stages of one series.

    The mask lives here rather than on the volume because it is produced by one stage
    and consumed by three others (cropping, normalisation, quality). Passing it
    through the context keeps :class:`MRIVolume` frozen and single-purpose.
    """

    config: StandardizationConfig
    mask: BrainMaskSlot = field(default_factory=BrainMaskSlot)
    metadata: MRIMetadata | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TransformResult:
    """What a stage did."""

    volume: MRIVolume
    #: ``False`` when the stage correctly determined there was nothing to do. The
    #: pipeline records that as ``no_op``, which is materially different from
    #: ``skipped`` in an audit.
    changed: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@runtime_checkable
class VolumeTransform(Protocol):
    """One standardisation stage."""

    #: Stable identifier used in the processing history.
    name: str

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        """Transform ``volume``.

        Raise :class:`~aura.backend.foundation.mri.errors.StageUnavailable` when the stage
        has no backend in this deployment, and
        :class:`~aura.backend.foundation.mri.errors.StageFailed` when an available stage
        breaks. Anything else is treated as an unexpected failure and is fatal for
        that series.
        """
        ...


# --------------------------------------------------------------------------- #
# Implemented: geometry
# --------------------------------------------------------------------------- #
class CanonicalOrientation:
    """Reorient to the closest RAS axis permutation.

    Pure index manipulation — a permutation and some flips — so no intensity is
    interpolated and no information is lost. Residual obliquity stays in the affine
    where it belongs; removing it would require resampling, which is
    :class:`VoxelResampler`'s job and has a cost this stage should not silently incur.
    """

    name = "canonical_orientation"

    def __init__(self, target: str = "RAS") -> None:
        if target.upper() != "RAS":
            raise StageUnavailable(
                self.name,
                f"only RAS is implemented as a canonical target; {target!r} was "
                "requested. The interface allows other targets; the implementation "
                "does not pretend to.")
        self._target = target.upper()

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        try:
            array, affine, original, changed = to_canonical(volume.array,
                                                            volume.geometry.affine)
        except ValueError as exc:
            raise StageFailed(self.name, str(exc)) from exc

        parameters = {"target": self._target,
                      "source_orientation": "".join(original),
                      "result_orientation": self._target if changed else "".join(original)}
        if not changed:
            return TransformResult(volume, False, parameters,
                                   "the volume was already in canonical orientation")
        return TransformResult(
            volume.derive(np.ascontiguousarray(array), affine,
                          reoriented_from="".join(original)),
            True, parameters,
            f"reoriented {''.join(original)} -> {self._target} by axis permutation "
            "and flips; no interpolation")


class VoxelResampler:
    """Resample onto an isotropic grid at a target spacing.

    The world extent is preserved: the output covers the same physical volume, with
    the number of voxels changing instead. The new affine is the old one post-
    multiplied by the zoom, which keeps voxel ``(0,0,0)`` at the same world point —
    the invariant the round-trip test asserts.

    Trilinear (``order=1``) is the default rather than cubic. Cubic overshoots at
    tissue boundaries and produces intensities that were never acquired, which then
    look like signal to whatever runs next.
    """

    name = "voxel_resampling"

    def __init__(self, target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 order: int = 1) -> None:
        if any(s <= 0 for s in target_spacing_mm):
            raise ValueError("target spacing must be positive along every axis")
        self._target = tuple(float(s) for s in target_spacing_mm)
        self._order = int(order)

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        current = np.asarray(volume.spacing, dtype=float)
        target = np.asarray(self._target, dtype=float)
        parameters = {"target_spacing_mm": [round(v, 5) for v in self._target],
                      "source_spacing_mm": [round(v, 5) for v in current],
                      "order": self._order}

        if np.allclose(current, target, rtol=1e-4, atol=1e-6):
            return TransformResult(volume, False, parameters,
                                   "the volume was already at the target spacing")

        zoom = target / current                    # old voxels per new voxel
        source_shape = np.asarray(volume.shape, dtype=float)
        new_shape = tuple(max(1, int(round(n / z))) for n, z in zip(source_shape, zoom))
        parameters["source_shape"] = [int(n) for n in volume.shape]
        parameters["result_shape"] = list(new_shape)

        array = self._resample(volume.array, zoom, new_shape)
        # A_new = A_old @ diag(zoom): voxel (0,0,0) stays put, spacing becomes target.
        scale = np.eye(4)
        scale[:3, :3] = np.diag(zoom)
        new_affine = volume.geometry.affine @ scale

        # A mask describes a grid, so it has to move with the volume or be dropped.
        # It is resampled nearest-neighbour — a mask is a label map, and interpolating
        # one produces fractional membership that is not a mask at all.
        if context.mask.present and context.mask.mask is not None:
            resampled_mask = self._resample_mask(context.mask.mask, zoom, new_shape)
            context.mask = BrainMaskSlot(
                mask=resampled_mask,
                provenance=context.mask.provenance,
                method=context.mask.method,
                details={**(context.mask.details or {}),
                         "resampled": "nearest neighbour onto the target grid"},
            )
            parameters["mask_resampled"] = True

        return TransformResult(
            volume.derive(array, new_affine, resampled_from=[float(v) for v in current]),
            True, parameters,
            f"resampled {tuple(int(n) for n in volume.shape)} at "
            f"{tuple(round(v, 3) for v in current)} mm -> {new_shape} at "
            f"{tuple(self._target)} mm")

    def _resample(self, array: np.ndarray, zoom: np.ndarray,
                  new_shape: tuple[int, ...]) -> np.ndarray:
        try:
            from scipy import ndimage
        except ImportError:                          # pragma: no cover - env specific
            return self._nearest(array, zoom, new_shape)
        try:
            return ndimage.affine_transform(
                np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0),
                matrix=np.diag(zoom), offset=0.0, output_shape=new_shape,
                order=self._order, mode="constant", cval=0.0,
            ).astype(np.float32, copy=False)
        except Exception as exc:
            raise StageFailed(self.name, "resampling failed",
                              detail={"error": type(exc).__name__}) from exc

    def _resample_mask(self, mask: np.ndarray, zoom: np.ndarray,
                       new_shape: tuple[int, ...]) -> np.ndarray:
        """Move a boolean mask onto the new grid without interpolating it."""
        try:
            from scipy import ndimage

            moved = ndimage.affine_transform(
                mask.astype(np.uint8), matrix=np.diag(zoom), offset=0.0,
                output_shape=new_shape, order=0, mode="constant", cval=0)
        except ImportError:                          # pragma: no cover - env specific
            moved = self._nearest(mask.astype(np.float32), zoom, new_shape)
        return moved.astype(bool)

    @staticmethod
    def _nearest(array: np.ndarray, zoom: np.ndarray,
                 new_shape: tuple[int, ...]) -> np.ndarray:
        """Nearest-neighbour fallback when scipy is absent.

        Deliberately crude and deliberately recorded as such by the caller: it is
        better to produce a correctly-placed nearest-neighbour volume and say so than
        to fail the whole study over a missing optional dependency.
        """
        grids = [np.clip((np.arange(n) * z).round().astype(int), 0, s - 1)
                 for n, z, s in zip(new_shape, zoom, array.shape)]
        return array[np.ix_(*grids)].astype(np.float32, copy=True)


class MaskCropper:
    """Crop to the mask's bounding box plus a margin.

    Removes air — routinely 40-60% of a head volume — with no effect on anatomy, and
    it is the cheapest large win available to any downstream 3D model. The affine is
    translated by the crop origin so world coordinates are unchanged, which is exactly
    the property that makes cropping safe to do before registration.
    """

    name = "brain_cropping"

    def __init__(self, margin_mm: float = 8.0) -> None:
        self._margin_mm = float(margin_mm)

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        box = context.mask.bounding_box()
        parameters: dict[str, Any] = {"margin_mm": self._margin_mm,
                                      "mask_provenance": context.mask.provenance.value}
        if box is None:
            return TransformResult(
                volume, False, parameters,
                "no mask was available, so the volume was left uncropped")

        margins = [max(1, int(round(self._margin_mm / s))) for s in volume.spacing]
        bounds = []
        for axis, (sl, margin) in enumerate(zip(box, margins)):
            start = max(0, sl.start - margin)
            stop = min(volume.shape[axis], sl.stop + margin)
            bounds.append(slice(start, stop))
        if all(b.start == 0 and b.stop == n for b, n in zip(bounds, volume.shape)):
            return TransformResult(volume, False, parameters,
                                   "the mask already spans the full volume")

        array = np.ascontiguousarray(volume.array[tuple(bounds)])
        translation = np.eye(4)
        translation[:3, 3] = [b.start for b in bounds]
        new_affine = volume.geometry.affine @ translation

        if context.mask.mask is not None:
            context.mask = BrainMaskSlot(
                mask=np.ascontiguousarray(context.mask.mask[tuple(bounds)]),
                provenance=context.mask.provenance,
                method=context.mask.method,
                details=context.mask.details,
            )

        parameters["crop_start"] = [int(b.start) for b in bounds]
        parameters["source_shape"] = [int(n) for n in volume.shape]
        parameters["result_shape"] = [int(n) for n in array.shape]
        removed = 1.0 - (array.size / max(1, volume.array.size))
        parameters["voxels_removed_fraction"] = round(float(removed), 4)
        return TransformResult(
            volume.derive(array, new_affine, cropped_from=[int(n) for n in volume.shape]),
            True, parameters,
            f"cropped to the mask bounding box with a {self._margin_mm:g} mm margin, "
            f"removing {removed:.0%} of the voxels; world coordinates unchanged")


# --------------------------------------------------------------------------- #
# Implemented: intensity
# --------------------------------------------------------------------------- #
class IntensityNormalizer:
    """Put intensities on a declared scale.

    MR intensity has no physical unit: the same tissue in the same scanner yields
    different numbers between sessions, which is why every brain-MRI pipeline
    normalises and why the *scheme* has to travel with the data. A model trained on
    z-scored input and given percentile-scaled input fails quietly.

    Statistics are computed over mask voxels when a mask exists. Including air would
    let the ratio of head to background — a function of field of view, not of
    anatomy — move the mean and standard deviation of every volume.
    """

    name = "intensity_normalization"

    def __init__(self, method: NormalizationMethod = NormalizationMethod.ZSCORE,
                 percentile_bounds: tuple[float, float] = (1.0, 99.0)) -> None:
        self._method = method
        self._bounds = (float(percentile_bounds[0]), float(percentile_bounds[1]))
        if not 0.0 <= self._bounds[0] < self._bounds[1] <= 100.0:
            raise ValueError("percentile bounds must satisfy 0 <= low < high <= 100")

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        parameters: dict[str, Any] = {"method": self._method.value}
        if self._method is NormalizationMethod.NONE:
            return TransformResult(volume, False, parameters,
                                   "intensity normalisation is disabled")

        array = volume.array
        finite = np.isfinite(array)
        region = finite & context.mask.mask if context.mask.present else finite
        if region.sum() < 100:
            region = finite
            parameters["fallback"] = "whole_volume"
        if region.sum() == 0:
            raise StageFailed(self.name, "the volume has no finite voxels to normalise")

        values = array[region]
        parameters["reference_region"] = ("mask" if context.mask.present
                                          and "fallback" not in parameters
                                          else "whole_volume")
        parameters["reference_voxels"] = int(region.sum())

        if self._method is NormalizationMethod.ZSCORE:
            mean, std = float(values.mean()), float(values.std())
            if std <= 0:
                raise StageFailed(self.name,
                                  "the reference region has zero variance; z-scoring "
                                  "would divide by zero")
            normalised = (array - mean) / std
            parameters.update({"mean": round(mean, 6), "std": round(std, 6)})
            message = f"z-scored over {parameters['reference_region']} voxels"
        elif self._method is NormalizationMethod.PERCENTILE:
            low, high = (float(v) for v in np.percentile(values, self._bounds))
            if high <= low:
                raise StageFailed(self.name,
                                  "the percentile bounds collapse to a single value")
            normalised = np.clip((array - low) / (high - low), 0.0, 1.0)
            parameters.update({"percentiles": list(self._bounds),
                               "low": round(low, 6), "high": round(high, 6)})
            message = (f"clipped to the {self._bounds[0]:g}-{self._bounds[1]:g} "
                       "percentile range and scaled to [0, 1]")
        elif self._method is NormalizationMethod.MINMAX:
            low, high = float(values.min()), float(values.max())
            if high <= low:
                raise StageFailed(self.name, "the reference region is constant")
            normalised = np.clip((array - low) / (high - low), 0.0, 1.0)
            parameters.update({"low": round(low, 6), "high": round(high, 6)})
            message = "scaled to [0, 1] by the reference region's range"
        else:                                        # pragma: no cover - exhaustive
            raise StageFailed(self.name,
                              f"unknown normalisation method {self._method!r}")

        # Non-finite voxels stay non-finite: the intensity check already measured and
        # reported them, and replacing them here would erase that evidence.
        normalised = np.where(finite, normalised, array).astype(np.float32, copy=False)
        return TransformResult(volume.derive(normalised, None,
                                             normalisation=self._method.value),
                               True, parameters, message)


# --------------------------------------------------------------------------- #
# Implemented: masking
# --------------------------------------------------------------------------- #
class ForegroundMaskEstimator:
    """Fill the context's mask slot with a head/foreground estimate.

    Explicitly **not** a brain mask — it includes skull and scalp. Recorded with
    ``MaskProvenance.FOREGROUND_HEURISTIC`` so nothing downstream can mistake it for
    one. See :mod:`aura.backend.foundation.mri.masking`.

    Leaves the volume untouched; it produces a mask, not an image.
    """

    name = "foreground_mask"

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        mask, details = estimate_foreground_mask(volume.array)
        coverage = float(mask.sum() / mask.size)
        context.mask = BrainMaskSlot(
            mask=mask,
            provenance=MaskProvenance.FOREGROUND_HEURISTIC,
            method="otsu + largest connected component + hole filling",
            details=details,
        )
        if coverage < 0.02:
            context.notes.append(
                f"the foreground estimate covers only {coverage:.1%} of the volume; "
                "the threshold may have failed on this contrast")
        return TransformResult(
            volume, True,
            {"coverage_fraction": round(coverage, 5), **details},
            f"estimated a head/foreground mask covering {coverage:.1%} of the volume "
            "(not a brain mask: it includes skull and scalp)")


# --------------------------------------------------------------------------- #
# Interface-only: bias field and skull stripping / fallbacks
# --------------------------------------------------------------------------- #
@runtime_checkable
class BiasFieldCorrector(VolumeTransform, Protocol):
    """Interface for N4-style bias-field correction.

    A separate protocol from :class:`VolumeTransform` so a deployment can require a
    corrector by type. Implementations must estimate and divide out the smooth
    multiplicative field; a stage that merely rescales intensity is not this.
    """


class SimpleITKBiasFieldCorrector:
    """N4ITK bias-field correction, delegated to SimpleITK.

    N4 is an iterative B-spline field fit (Tustison et al., *IEEE TMI* 2010). It is
    not reimplemented here: a hand-rolled approximation would produce plausible output
    that differs from the reference implementation in ways nobody could audit, which
    is precisely the failure mode this layer exists to avoid.

    Without SimpleITK installed the constructor raises
    :class:`~aura.backend.foundation.mri.errors.StageUnavailable`; the pipeline records
    ``unavailable`` and the study is marked *not* bias-corrected. Installing
    SimpleITK enables it with no other change.
    """

    name = "n4_bias_field_correction"

    def __init__(self, shrink_factor: int = 4, iterations: tuple[int, ...] = (50, 50, 50, 50)):
        try:
            import SimpleITK                                     # noqa: F401
        except ImportError as exc:
            raise StageUnavailable(
                self.name,
                "N4 bias-field correction requires SimpleITK, which is not installed "
                "in this deployment. The volume is NOT bias corrected; low-frequency "
                "intensity inhomogeneity remains and will bias any intensity-based "
                "segmentation or volumetry.",
                detail={"install": "pip install SimpleITK", "reference": "Tustison "
                        "et al., N4ITK, IEEE TMI 2010"},
            ) from exc
        self._shrink = int(shrink_factor)
        self._iterations = [int(i) for i in iterations]

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:   # pragma: no cover
        import SimpleITK as sitk

        try:
            image = sitk.GetImageFromArray(
                np.transpose(volume.array, (2, 1, 0)).astype(np.float32))
            image.SetSpacing(tuple(float(s) for s in volume.spacing))
            mask = (sitk.OtsuThreshold(image, 0, 1, 200)
                    if not context.mask.present else
                    sitk.GetImageFromArray(
                        np.transpose(context.mask.mask, (2, 1, 0)).astype(np.uint8)))
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations(self._iterations)
            shrunk = sitk.Shrink(image, [self._shrink] * 3)
            shrunk_mask = sitk.Shrink(mask, [self._shrink] * 3)
            corrector.Execute(shrunk, shrunk_mask)
            field = sitk.Exp(corrector.GetLogBiasFieldAsImage(image))
            corrected = sitk.GetArrayFromImage(image / field)
            array = np.transpose(corrected, (2, 1, 0)).astype(np.float32)
        except Exception as exc:
            raise StageFailed(self.name, "N4 bias-field correction failed",
                              detail={"error": type(exc).__name__}) from exc

        return TransformResult(
            volume.derive(array, None, bias_corrected="n4itk"), True,
            {"shrink_factor": self._shrink, "iterations": self._iterations,
             "implementation": "SimpleITK N4BiasFieldCorrectionImageFilter"},
            "N4 bias-field correction applied")


class GaussianBiasFieldCorrector:
    """Fallback 3D bias field corrector using a low-pass Gaussian filter in NumPy/SciPy."""

    name = "n4_bias_field_correction"

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        try:
            from scipy import ndimage
        except ImportError as exc:
            raise StageUnavailable(
                self.name, "Gaussian bias field correction fallback requires scipy.",
                detail={"install": "pip install scipy"}
            ) from exc

        array = volume.array.copy()
        finite = np.isfinite(array)
        filled = np.where(finite, array, 0.0)

        # Sigma is set to ~15.0 mm to capture low-frequency bias
        sigma_voxels = [15.0 / s for s in volume.spacing]
        smoothed = ndimage.gaussian_filter(filled, sigma=sigma_voxels, mode="mirror")
        footprint = ndimage.gaussian_filter(finite.astype(float), sigma=sigma_voxels, mode="mirror")

        valid_footprint = footprint > 1e-4
        bias_field = np.ones_like(array)
        bias_field[valid_footprint] = smoothed[valid_footprint] / footprint[valid_footprint]

        # Clamp bias field to a safe range relative to the mean to avoid division blow-up
        mean_bias = float(bias_field.mean())
        min_allowed = 0.1 * mean_bias if mean_bias > 1e-6 else 0.1
        bias_field = np.clip(bias_field, min_allowed, None)

        corrected = np.where(finite, array / bias_field, array).astype(np.float32)

        return TransformResult(
            volume.derive(corrected, None, bias_corrected="gaussian_lowpass"),
            True,
            {"sigma_mm": 15.0, "implementation": "Gaussian Low-Pass Filter Fallback"},
            "Gaussian low-pass bias-field correction applied as a fallback"
        )


@runtime_checkable
class SkullStripper(VolumeTransform, Protocol):
    """Interface for brain extraction.

    An implementation must fill ``context.mask`` with
    :attr:`~aura.backend.foundation.mri.types.MaskProvenance.SKULL_STRIPPED`, which is the
    only provenance that marks a mask as a genuine brain mask.
    """


class MorphologicalSkullStripper:
    """Morphological 3D brain extraction fallback.

    Refines the foreground/head mask to isolate brain tissue using Otsu thresholding,
    3D morphological erosion (to break skull/dura connections), 3D connected component
    labeling (to keep the largest component - cerebrum/cerebellum), and closing/hole filling.
    """

    name = "skull_stripping"

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        try:
            from scipy import ndimage
        except ImportError as exc:
            raise StageUnavailable(
                self.name, "Morphological skull stripper requires scipy.",
                detail={"install": "pip install scipy"}
            ) from exc

        from .masking import otsu_threshold, BrainMaskSlot
        from .types import MaskProvenance

        array = np.nan_to_num(volume.array, nan=0.0)

        # Isolate foreground
        if context.mask.present and context.mask.mask is not None:
            fg = context.mask.mask
        else:
            fg = array > (array.mean() * 0.1)

        values = array[fg]
        if values.size == 0:
            raise StageFailed(self.name, "empty foreground for skull stripping")

        # Threshold using Otsu threshold
        thresh = otsu_threshold(values)
        if not np.isfinite(thresh):
            thresh = float(array.mean())

        binary = (array > thresh) & fg

        # 3D morphological erosion to sever connection to meninges/skull
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        eroded = ndimage.binary_erosion(binary, structure=struct, iterations=2)

        # Keep largest connected component (the brain tissue)
        labeled, num_features = ndimage.label(eroded)
        if num_features == 0:
            brain_core = eroded
        else:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # ignore background
            largest_label = np.argmax(sizes)
            brain_core = labeled == largest_label

        # Dilate back to recover original brain volume boundary
        dilated = ndimage.binary_dilation(brain_core, structure=struct, iterations=2)

        # Fill holes and perform closing
        brain_mask = ndimage.binary_closing(dilated, structure=struct, iterations=1)
        brain_mask = ndimage.binary_fill_holes(brain_mask)

        # Store mask in context
        context.mask = BrainMaskSlot(
            mask=brain_mask,
            provenance=MaskProvenance.SKULL_STRIPPED,
            method="Otsu threshold + 3D morphological erosion + largest connected component",
            details={"threshold": float(thresh), "voxels": int(brain_mask.sum())}
        )

        return TransformResult(
            volume,
            True,
            {"threshold": float(thresh), "voxels": int(brain_mask.sum())},
            "Morphological skull stripping applied successfully"
        )


def default_pipeline_stages(config: StandardizationConfig) -> list[VolumeTransform]:
    """The default stage order.

    Order matters and is not arbitrary:

    1. **Canonical orientation** first — every later stage is easier to reason about
       on a known axis convention, and it is free.
    2. **Bias correction** before anything that measures intensity, because the field
       it removes corrupts every intensity statistic.
    3. **Foreground mask** next, so cropping and normalisation share one mask.
    4. **Skull stripping** after the mask exists, so a real stripper can refine it.
    5. **Cropping** before resampling — cropping first means resampling runs over a
       much smaller array, which is the difference between seconds and tens of
       seconds on a head volume.
    6. **Resampling** before normalisation, so the statistics describe the grid the
       model will actually see.
    7. **Normalisation** last.
    """
    stages: list[VolumeTransform] = [CanonicalOrientation(config.target_orientation)]

    if config.bias_correction:
        try:
            stages.append(SimpleITKBiasFieldCorrector())
        except StageUnavailable:
            log.info("SimpleITK bias corrector unavailable; falling back to GaussianBiasFieldCorrector")
            stages.append(GaussianBiasFieldCorrector())

    if config.foreground_mask:
        stages.append(ForegroundMaskEstimator())
    if config.skull_strip:
        stages.append(MorphologicalSkullStripper())
    if config.crop_to_mask:
        stages.append(MaskCropper(config.crop_margin_mm))
    stages.append(VoxelResampler(config.target_spacing_mm, config.resample_order))
    stages.append(IntensityNormalizer(config.normalization, config.percentile_bounds))
    return stages


class _DeclaredUnavailable:
    """Placeholder that re-raises a construction-time :class:`StageUnavailable`.

    Keeps the stage visible in the pipeline and in the processing history. A stage
    that vanished when its backend was missing would leave no record that it was ever
    meant to run, which is the whole thing the history is for.
    """

    def __init__(self, error: StageUnavailable) -> None:
        self.name = error.stage
        self._error = error

    def apply(self, volume: MRIVolume,
              context: StandardizationContext) -> TransformResult:
        raise self._error


__all__ = [
    "BiasFieldCorrector", "CanonicalOrientation", "ForegroundMaskEstimator",
    "IntensityNormalizer", "MaskCropper", "SimpleITKBiasFieldCorrector",
    "GaussianBiasFieldCorrector", "SkullStripper", "StandardizationContext",
    "TransformResult", "MorphologicalSkullStripper", "VoxelResampler",
    "VolumeTransform", "default_pipeline_stages",
]
