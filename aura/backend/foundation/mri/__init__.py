"""AURA NeuroMind — MRI Foundation Layer.

Raw brain MRI in, standardised model-ready study out. No models, no segmentation, no
interpretation: this layer's entire job is to make sure that whatever runs next is
looking at a volume it can trust, and knows exactly what was done to it.

    from backend.foundation.mri import MRIFoundationPipeline

    study = MRIFoundationPipeline().run("/data/studies/patient-004")
    t1 = study.first(SequenceType.T1)
    if t1 and t1.usable:
        array = t1.volume.array           # canonical RAS, isotropic, normalised
        affine = t1.volume.affine         # world coordinates preserved throughout
        assert t1.history.was_applied("voxel_resampling")

The pipeline (see :mod:`backend.foundation.mri.pipeline`)::

    study path
        -> study validation      loader: mixed formats, mixed studies, empty paths
        -> loading               DICOM series / NIfTI-1,2 / NRRD
        -> volume reconstruction slices to a 3D array with its affine
        -> metadata extraction   typed, patient-independent, allowlisted
        -> sequence detection    T1 / T1ce / T2 / FLAIR / DWI / ADC / SWI / PD
        -> quality assessment    7 checks, calibrated and provisional kept apart
        -> standardisation       orientation, bias field, mask, crop, resample, norm
        -> registration prep     inputs computed; no transform applied
        -> FoundationStudy

Three properties this layer holds to, because each one is a failure mode that is
invisible downstream:

* **The array and its affine never separate.** Every transform returns both. A volume
  reoriented without its affine looks perfectly normal and is left-right mirrored.
* **Absent is never zero.** A missing echo time stays ``None``. A stage with no
  backend is recorded ``unavailable``, not skipped. A check that could not run reports
  ``not_evaluated`` and is excluded from the score rather than scored as a pass.
* **Provisional numbers are labelled.** Motion and SNR thresholds were not fitted on a
  labelled corpus here, so those checks carry ``calibrated=False`` and cannot reject a
  study. A head mask is never called a brain mask.
"""
from backend.foundation.mri.config import (
    FoundationConfig,
    LoaderConfig,
    QualityThresholds,
    StandardizationConfig,
)
from backend.foundation.mri.errors import (
    CorruptStudy,
    MRIFoundationError,
    StageFailed,
    StageUnavailable,
    StudyNotFound,
    StudyRejected,
    StudyValidationError,
    UnsupportedStudyFormat,
)
from backend.foundation.mri.geometry import VoxelGeometry
from backend.foundation.mri.loader import LoadedStudy, MRIStudyLoader, load_study
from backend.foundation.mri.masking import BrainMaskSlot
from backend.foundation.mri.metadata import MetadataExtractor, MRIMetadata
from backend.foundation.mri.pipeline import (
    MRIFoundationPipeline,
    build_foundation_pipeline,
)
from backend.foundation.mri.quality import (
    MRIQualityInspector,
    QualityCheck,
    QualityReport,
)
from backend.foundation.mri.registration import RegistrationPlan, RegistrationPreparer
from backend.foundation.mri.sequence import (
    RuleBasedSequenceDetector,
    SequenceAssignment,
    SequenceDetector,
)
from backend.foundation.mri.standardize import (
    StandardizationContext,
    TransformResult,
    VolumeTransform,
)
from backend.foundation.mri.study import (
    FoundationStudy,
    ProcessingHistory,
    ProcessingStep,
    StandardizedSeries,
)
from backend.foundation.mri.types import (
    FOUNDATION_VERSION,
    AnatomicalPlane,
    CheckStatus,
    FileFormat,
    MaskProvenance,
    NormalizationMethod,
    QualityVerdict,
    ScannerVendor,
    SequenceType,
    StepStatus,
)
from backend.foundation.mri.volume import MRIVolume, VolumeBuilder

__all__ = [
    # pipeline
    "MRIFoundationPipeline", "build_foundation_pipeline",
    # data model
    "FoundationStudy", "StandardizedSeries", "ProcessingHistory", "ProcessingStep",
    "MRIVolume", "VoxelGeometry", "BrainMaskSlot", "RegistrationPlan",
    # components
    "MRIStudyLoader", "LoadedStudy", "load_study", "MetadataExtractor", "MRIMetadata",
    "RuleBasedSequenceDetector", "SequenceDetector", "SequenceAssignment",
    "MRIQualityInspector", "QualityReport", "QualityCheck", "VolumeBuilder",
    "RegistrationPreparer", "VolumeTransform", "TransformResult",
    "StandardizationContext",
    # configuration
    "FoundationConfig", "LoaderConfig", "QualityThresholds", "StandardizationConfig",
    # vocabulary
    "SequenceType", "FileFormat", "ScannerVendor", "AnatomicalPlane", "CheckStatus",
    "QualityVerdict", "MaskProvenance", "StepStatus", "NormalizationMethod",
    "FOUNDATION_VERSION",
    # errors
    "MRIFoundationError", "StudyNotFound", "UnsupportedStudyFormat", "CorruptStudy",
    "StudyValidationError", "StudyRejected", "StageUnavailable", "StageFailed",
]
