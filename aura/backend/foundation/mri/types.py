"""Vocabulary of the MRI Foundation Layer.

Everything here is an enum or a small frozen value type. They are separated from the
modules that use them because they are the layer's public contract: a downstream
NeuroMind model imports :class:`SequenceType` to declare which inputs it accepts, and
must not have to import a loader to do it.

One convention decision is encoded here and worth stating once, loudly:

**World space is RAS+ throughout.** Voxel coordinates map to millimetres in a
right-anterior-superior frame, the convention NIfTI, nibabel, MONAI, and nnU-Net all
use. DICOM is LPS, so the DICOM reader converts on the way in (``diag(-1,-1,1,1)``)
and never afterwards. Picking one and converting at the boundary is the only way to
stop sign errors from accumulating; picking RAS+ specifically means a foundation
volume can be written to NIfTI and opened in any neuroimaging viewer without a
transform.
"""
from __future__ import annotations

from enum import Enum


class FileFormat(str, Enum):
    """Storage format a series was read from.

    ``HDF5`` is not a clinical interchange format and no clinical upload will ever
    arrive as one. It is here because research corpora are redistributed that way —
    BraTS2020's slice-wise HDF5 release is what
    :mod:`aura.backend.vision.brain.io.brats_h5` reads — and a training corpus that flowed
    through this layer should be able to say so rather than be recorded as
    ``UNKNOWN``. It sits last in
    :data:`~aura.backend.foundation.mri.loader.FORMAT_PREFERENCE` so no existing
    discovery decision changes.
    """

    DICOM = "dicom"
    NIFTI = "nifti"
    NRRD = "nrrd"
    HDF5 = "hdf5"
    UNKNOWN = "unknown"


class SequenceType(str, Enum):
    """MR pulse sequence class, as far as it can be determined from a study.

    These are the classes downstream models care about — a model trained on FLAIR
    must never be handed a DWI. ``PD`` is included because proton-density images sit
    in the same acquisition families and are otherwise silently mistaken for T2.

    ``UNKNOWN`` is a first-class answer, not a failure. An unlabelled NIfTI with no
    acquisition parameters genuinely cannot be classified, and reporting ``UNKNOWN``
    is the honest result; guessing ``T1`` because the filename said so is how a model
    ends up inferring on the wrong contrast.
    """

    T1 = "t1"
    T1CE = "t1ce"          # post-gadolinium T1
    T2 = "t2"
    FLAIR = "flair"
    DWI = "dwi"
    ADC = "adc"
    SWI = "swi"
    PD = "pd"
    UNKNOWN = "unknown"


#: Human-readable names, for reports and API responses.
SEQUENCE_LABELS: dict[SequenceType, str] = {
    SequenceType.T1: "T1-weighted",
    SequenceType.T1CE: "T1-weighted post-contrast",
    SequenceType.T2: "T2-weighted",
    SequenceType.FLAIR: "FLAIR",
    SequenceType.DWI: "Diffusion-weighted",
    SequenceType.ADC: "Apparent diffusion coefficient map",
    SequenceType.SWI: "Susceptibility-weighted",
    SequenceType.PD: "Proton density",
    SequenceType.UNKNOWN: "Unidentified sequence",
}


class ScannerVendor(str, Enum):
    """Normalised manufacturer.

    Vendors write ``Manufacturer`` inconsistently (``SIEMENS``, ``Siemens
    Healthineers``, ``GE MEDICAL SYSTEMS``, ``Philips Medical Systems``). Sequence
    heuristics and, later, harmonisation both branch on vendor, so the raw string is
    normalised once here and the original is preserved alongside it.
    """

    SIEMENS = "siemens"
    GE = "ge"
    PHILIPS = "philips"
    CANON = "canon"          # includes the former Toshiba medical business
    HITACHI = "hitachi"
    UNITED_IMAGING = "united_imaging"
    OTHER = "other"
    UNKNOWN = "unknown"


class MRAcquisitionType(str, Enum):
    """2D multi-slice vs 3D volumetric acquisition (DICOM 0018,0023)."""

    TWO_D = "2D"
    THREE_D = "3D"
    UNKNOWN = "unknown"


class AnatomicalPlane(str, Enum):
    """Acquisition plane, derived from the slice normal, not from a tag.

    ``OBLIQUE`` is reported when the normal is more than a configured angle away from
    every cardinal axis — common and clinically normal (an oblique-coronal hippocampal
    T2, say), but it changes how a downstream model should resample.
    """

    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    OBLIQUE = "oblique"
    UNKNOWN = "unknown"


class CheckStatus(str, Enum):
    """Outcome of one quality check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    #: The check could not run — a required input was absent (e.g. no background
    #: voxels for a noise estimate). Deliberately not ``PASS``: an unevaluated check
    #: must never be able to raise a quality score.
    NOT_EVALUATED = "not_evaluated"


class QualityVerdict(str, Enum):
    """Study-level quality conclusion."""

    ACCEPTABLE = "acceptable"
    #: Usable, with named caveats. The common real-world verdict.
    ACCEPTABLE_WITH_WARNINGS = "acceptable_with_warnings"
    #: A hard check failed; downstream inference on this study is not supported.
    REJECTED = "rejected"


class MaskProvenance(str, Enum):
    """How the mask in a :class:`~aura.backend.foundation.mri.masking.BrainMaskSlot` arose.

    The distinction between ``FOREGROUND_HEURISTIC`` and ``SKULL_STRIPPED`` is the
    whole point of this enum. A threshold-and-largest-component mask captures the
    *head* — skull, scalp, and often the neck. Calling that a brain mask would make
    every downstream volume measurement wrong by tens of percent while looking
    entirely plausible. Only a real stripper sets ``SKULL_STRIPPED``.
    """

    ABSENT = "absent"
    FOREGROUND_HEURISTIC = "foreground_heuristic"
    SKULL_STRIPPED = "skull_stripped"
    EXTERNAL = "external"


class StepStatus(str, Enum):
    """Outcome of one processing-history step."""

    APPLIED = "applied"
    #: Ran, correctly determined there was nothing to do (already canonical, already
    #: at target spacing). Distinct from ``SKIPPED`` so "no change needed" and "we
    #: chose not to run this" never look the same in an audit.
    NO_OP = "no_op"
    SKIPPED = "skipped"                 # disabled by configuration
    UNAVAILABLE = "unavailable"         # declared, no backend in this deployment
    FAILED = "failed"


class NormalizationMethod(str, Enum):
    """Intensity normalisation scheme.

    MR intensities have no physical units — the same tissue in the same scanner
    yields different numbers between sessions — so every downstream model needs a
    declared scheme, and needs to know *which* one ran.
    """

    NONE = "none"
    ZSCORE = "zscore"                   # mean/std over mask voxels
    PERCENTILE = "percentile"           # robust clip to [p_low, p_high] -> [0,1]
    MINMAX = "minmax"


#: Foundation output format version. Bumped when the shape of a
#: :class:`~aura.backend.foundation.mri.study.FoundationStudy` changes in a way a
#: consumer could notice. Stamped into every study so a cached artefact can be
#: matched against the code that produced it.
FOUNDATION_VERSION = "1.0.0"
