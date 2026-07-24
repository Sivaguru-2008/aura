"""Configuration for the MRI Foundation Layer.

Frozen dataclasses, constructed by the caller and injected. There is no module-level
mutable state, no ``os.environ`` read at import time, and no path baked into any
default — a second pipeline with different thresholds can run in the same process
without disturbing the first, which is what makes threshold sweeps and A/B evaluation
possible at all.

Every threshold that encodes a clinical or physical assumption is named, given a
default, and its basis stated in a comment. Two categories, and the difference is
important:

* **Physical bounds** (voxel size, head field of view, plausible TR/TE) come from the
  physics and anatomy of brain MRI. They are cited, stable, and safe defaults.
* **Artefact thresholds** (motion score, SNR) are *provisional*. No labelled
  motion-artefact corpus was available in this deployment to fit them, so the checks
  that use them are emitted with ``calibrated=False`` and cannot on their own reject
  a study. That is the same posture the modality router takes for its uncalibrated
  pixel-only path, and for the same reason: a number nobody validated must not be
  allowed to look like a number somebody did.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from backend.foundation.mri.types import NormalizationMethod


@dataclass(frozen=True)
class LoaderConfig:
    """How studies are discovered, read, and checked for structural integrity."""

    #: File suffixes each reader claims. Extensionless files are probed as DICOM,
    #: which is how DICOM is usually exported.
    dicom_suffixes: tuple[str, ...] = (".dcm", ".dicom", ".ima", "")
    nifti_suffixes: tuple[str, ...] = (".nii", ".nii.gz", ".hdr", ".img")
    nrrd_suffixes: tuple[str, ...] = (".nrrd", ".nhdr")

    #: Directory recursion depth when scanning a study folder. DICOM exports nest
    #: patient/study/series; 6 covers every layout seen in practice without letting
    #: a mis-pointed root walk an entire drive.
    max_scan_depth: int = 6
    #: Hard cap on files examined during discovery. Protects against a pathological
    #: directory; exceeded means the caller pointed at the wrong root.
    max_files_scanned: int = 200_000

    #: A series with fewer slices than this cannot form a volume. Two is the
    #: arithmetic minimum for deriving slice spacing from positions.
    min_slices_for_volume: int = 2
    #: Fraction by which a single inter-slice gap may deviate from the series median
    #: before it counts as irregular. 0.10 accommodates float rounding in stored
    #: positions and genuine sub-percent scanner jitter; a real missing slice is a
    #: 100% deviation.
    slice_gap_tolerance: float = 0.10
    #: Positions closer together than this are the same location — a duplicated
    #: instance, not a thin slice. No MR protocol acquires 10 micron slices.
    duplicate_position_tolerance_mm: float = 0.01

    #: Continue when individual files in a series fail to decode. The surviving
    #: slices still form a volume, the failures are recorded in the integrity report,
    #: and quality control decides whether what remains is usable. Set False for
    #: archival ingestion where any loss is unacceptable.
    tolerate_corrupt_slices: bool = True
    #: Fraction of a series that may be unreadable before the series is abandoned.
    max_corrupt_fraction: float = 0.20


@dataclass(frozen=True)
class QualityThresholds:
    """Bounds for the quality checks.

    ``calibrated`` in the emitted report is driven by which group a threshold is in:
    physical bounds produce calibrated checks, artefact heuristics do not.
    """

    # -- resolution (physical) ---------------------------------------------- #
    #: Plausible in-plane voxel edge for brain MRI. 0.2 mm is beyond any clinical
    #: scanner; 5 mm is a thick-slice screening protocol's in-plane limit.
    min_voxel_mm: float = 0.2
    max_voxel_mm: float = 5.0
    #: Slice thickness above this is a legitimate but coarse clinical acquisition —
    #: warn, because 3D analysis on it will be interpolation-dominated.
    warn_slice_thickness_mm: float = 5.0
    max_slice_thickness_mm: float = 10.0
    #: Anisotropy above this warns: the volume is effectively 2D multi-slice.
    warn_anisotropy: float = 3.0

    # -- field of view (physical) ------------------------------------------- #
    #: Adult head extents. A 120 mm span cannot contain a brain; 400 mm means the
    #: field of view includes far more than the head (or the units are wrong).
    min_head_fov_mm: float = 120.0
    max_head_fov_mm: float = 400.0
    #: Minimum slices for the volume to span a brain in the through-plane direction.
    min_slices: int = 10

    # -- intensity (physical / arithmetic) ---------------------------------- #
    #: Fraction of voxels allowed to sit at the maximum stored value before the
    #: acquisition is considered clipped. Magnitude MR should have a long right tail,
    #: not a spike at the ceiling.
    max_saturated_fraction: float = 0.02
    #: Below this dynamic range (unique-value count relative to voxel count) the
    #: volume carries almost no information — a mask or a failed export.
    min_distinct_value_ratio: float = 1e-4
    #: Magnitude MR is non-negative. Phase and some derived maps are not, so this is
    #: a warning, not a failure.
    warn_on_negative_intensities: bool = True

    # -- noise / SNR (PROVISIONAL — not fitted on a labelled corpus) --------- #
    #: SNR below this warns. The value is the widely quoted floor for diagnostic
    #: brain MRI, but it was not re-measured here, so the check reports
    #: ``calibrated=False``.
    warn_snr: float = 10.0
    #: Fraction of the volume's smallest-intensity voxels used as the air/background
    #: sample for the noise estimate.
    background_quantile: float = 0.10

    # -- motion (PROVISIONAL — not fitted on a labelled corpus) -------------- #
    #: Correlation between adjacent slices below this suggests inter-slice
    #: displacement. Anatomy changes slowly through-plane, so healthy series sit high.
    warn_slice_correlation: float = 0.70
    #: Ratio of structured background energy along the phase-encode direction to the
    #: frequency-encode direction. Ghosts replicate along phase-encode, so a ratio
    #: well above 1 is the signature.
    warn_ghost_ratio: float = 1.60

    # -- aggregation --------------------------------------------------------- #
    #: Overall score below this yields ``REJECTED`` when any *calibrated* check also
    #: failed. An uncalibrated check alone can never reject — it can only warn.
    reject_below_score: float = 0.40
    #: Score below this yields ``ACCEPTABLE_WITH_WARNINGS``.
    warn_below_score: float = 0.75


@dataclass(frozen=True)
class StandardizationConfig:
    """Which standardisation stages run, and with what parameters."""

    #: Target world orientation. RAS is the only value the implemented stage supports;
    #: the field exists because the interface does not assume it.
    target_orientation: str = "RAS"
    #: Isotropic target voxel size in mm. 1 mm is the working resolution of nnU-Net's
    #: brain configurations and of most published brain-MRI models.
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Spline order for resampling. 1 (trilinear) is the right default for intensity
    #: volumes: order 3 overshoots at tissue boundaries and invents intensities that
    #: were never acquired. 0 is for label maps.
    resample_order: int = 1

    #: Intensity normalisation. Z-score over brain/foreground voxels is the standard
    #: for MR, whose intensities have no physical units.
    normalization: NormalizationMethod = NormalizationMethod.ZSCORE
    #: Percentile clip bounds, used by ``PERCENTILE`` normalisation.
    percentile_bounds: tuple[float, float] = (1.0, 99.0)

    #: Attempt N4 bias-field correction. Declared-but-unavailable by default in this
    #: deployment (no SimpleITK) — recorded as ``unavailable`` in the history, not
    #: silently skipped.
    bias_correction: bool = True
    #: Attempt skull stripping. Also interface-only: every credible stripper is
    #: either a learned model or an external toolkit, and this layer ships neither.
    skull_strip: bool = True
    #: Estimate a head/foreground mask. Cheap, deterministic, and explicitly *not* a
    #: brain mask — it drives cropping and the background noise sample.
    foreground_mask: bool = True

    #: Crop to the mask bounding box, with this margin in mm. Removes empty air —
    #: typically 40-60% of a head volume — without touching anatomy.
    crop_to_mask: bool = True
    crop_margin_mm: float = 8.0

    #: Fail the whole study when a stage is unavailable, instead of recording it and
    #: continuing. For pipelines whose downstream model requires bias correction.
    strict: bool = False


@dataclass(frozen=True)
class FoundationConfig:
    """Complete configuration for one pipeline instance."""

    loader: LoaderConfig = field(default_factory=LoaderConfig)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)

    #: Raise :class:`~backend.foundation.mri.errors.StudyRejected` when quality
    #: control rejects a study. Off by default: the report travels with the study and
    #: the caller decides, because "usable for triage, not for volumetry" is a real
    #: verdict a hard reject cannot express.
    reject_on_quality: bool = False
    #: Keep the pre-standardisation volume on the output. Doubles memory; invaluable
    #: when debugging a preprocessing regression.
    retain_source_volume: bool = False

    def with_overrides(self, **kwargs: Any) -> "FoundationConfig":
        """Return a copy with top-level fields replaced. Frozen in, frozen out."""
        return replace(self, **kwargs)
