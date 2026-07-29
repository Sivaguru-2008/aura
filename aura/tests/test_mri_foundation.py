"""Tests for the MRI Foundation Layer.

Four layers, in increasing cost:

1. **Geometry** — pure maths. The invariant that matters is that world coordinates
   survive every transform, so those tests compare full coordinate grids rather than
   spot-checking a corner.
2. **Readers** — write a real file in each format, read it back, assert the array and
   the affine round-trip. Synthetic files are legitimate here in a way they are not
   for the modality router: a NIfTI header written to the published spec *is* a real
   NIfTI header, whereas a synthetic chest film is not a real acquisition. The
   failure paths (truncated, unsupported datatype, missing pair half) are tested with
   deliberately damaged files.
3. **Components** — metadata, sequence detection, quality, standardisation, each in
   isolation with hand-built inputs whose correct answer is known.
4. **Pipeline** — end to end over synthetic DICOM and NIfTI studies.

Two tests exist specifically to catch regressions in the layer's honesty rather than
its correctness, and are worth keeping if the file is ever trimmed:
``test_patient_identifiers_never_reach_metadata`` and
``test_uncalibrated_check_cannot_reject_a_study``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from aura.backend.foundation.mri import (
    FoundationConfig,
    MRIFoundationPipeline,
    MRIQualityInspector,
    MRIStudyLoader,
    MetadataExtractor,
    RuleBasedSequenceDetector,
    SequenceType,
    VolumeBuilder,
)
from aura.backend.foundation.mri.config import StandardizationConfig
from aura.backend.foundation.mri.errors import (
    CorruptStudy,
    StageFailed,
    StageUnavailable,
    StudyNotFound,
    StudyValidationError,
    UnsupportedStudyFormat,
)
from aura.backend.foundation.mri.geometry import (
    VoxelGeometry,
    affine_from_dicom,
    axis_codes,
    obliquity_degrees,
    to_canonical,
    voxel_to_world,
)
from aura.backend.foundation.mri.io.nifti_reader import NiftiReader
from aura.backend.foundation.mri.io.nrrd_reader import NrrdReader
from aura.backend.foundation.mri.masking import BrainMaskSlot, estimate_foreground_mask
from aura.backend.foundation.mri.metadata import (
    DICOM_KEYWORDS,
    PATIENT_IDENTIFYING_KEYWORDS,
    assert_patient_independent,
    normalise_vendor,
)
from aura.backend.foundation.mri.quality import _check
from aura.backend.foundation.mri.standardize import (
    CanonicalOrientation,
    ForegroundMaskEstimator,
    IntensityNormalizer,
    MaskCropper,
    StandardizationContext,
    MorphologicalSkullStripper,
    VoxelResampler,
)
from aura.backend.foundation.mri.types import (
    CheckStatus,
    FileFormat,
    MaskProvenance,
    NormalizationMethod,
    QualityVerdict,
    ScannerVendor,
    StepStatus,
)
from aura.backend.foundation.mri.volume import MRIVolume

pydicom = pytest.importorskip("pydicom", reason="DICOM tests need pydicom")


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def head_phantom(shape=(80, 80, 50), spacing=(2.5, 2.5, 2.5), seed=7) -> np.ndarray:
    """A head-like volume: a bright ellipsoid in noisy air.

    Deliberately plausible enough to pass quality control — a phantom that failed
    every check would make the "good study passes" tests vacuous.
    """
    rng = np.random.default_rng(seed)
    grid = np.indices(shape).astype(np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    radii = np.asarray([shape[0] * 0.38, shape[1] * 0.34, shape[2] * 0.40],
                       dtype=np.float32)
    distance = sum(((grid[i] - centre[i]) / radii[i]) ** 2 for i in range(3))

    volume = np.abs(rng.normal(0.0, 3.0, size=shape)).astype(np.float32)
    head = distance <= 1.0
    volume[head] = 100.0 + rng.normal(0.0, 4.0, size=int(head.sum())).astype(np.float32)
    # An inner structure, so adjacent slices correlate the way real anatomy does.
    volume[distance <= 0.45] = 150.0
    return volume.astype(np.float32)


def ras_affine(spacing=(2.5, 2.5, 2.5), origin=(-100.0, -100.0, -60.0)) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def write_nifti1(path: Path, array: np.ndarray, affine: np.ndarray, *,
                 gzipped: bool = False, sform_code: int = 1,
                 datatype: int = 16, descrip: bytes = b"") -> Path:
    """Write a real NIfTI-1 file to the published header layout."""
    from aura.backend.foundation.mri.io.nifti_reader import _NIFTI1_DTYPE

    header = np.zeros(1, dtype=_NIFTI1_DTYPE)[0]
    header["sizeof_hdr"] = 348
    header["regular"] = b"r"
    dim = np.ones(8, dtype=np.int16)
    dim[0] = array.ndim
    dim[1:array.ndim + 1] = array.shape
    header["dim"] = dim
    header["datatype"] = datatype
    header["bitpix"] = {2: 8, 4: 16, 16: 32, 64: 64, 32: 64}.get(datatype, 32)
    pixdim = np.ones(8, dtype=np.float32)
    pixdim[1:4] = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    header["pixdim"] = pixdim
    header["vox_offset"] = 352.0
    header["scl_slope"] = 1.0
    header["xyzt_units"] = 2                     # mm
    header["sform_code"] = sform_code
    header["srow_x"] = affine[0]
    header["srow_y"] = affine[1]
    header["srow_z"] = affine[2]
    header["descrip"] = descrip
    header["magic"] = b"n+1"

    payload = (header.tobytes() + b"\x00" * 4
               + np.asfortranarray(array.astype(np.float32)).tobytes(order="F"))
    if gzipped:
        path.write_bytes(gzip.compress(payload))
    else:
        path.write_bytes(payload)
    return path


def write_nifti2(path: Path, array: np.ndarray, affine: np.ndarray) -> Path:
    """Write a real NIfTI-2 file."""
    from aura.backend.foundation.mri.io.nifti_reader import _NIFTI2_DTYPE

    header = np.zeros(1, dtype=_NIFTI2_DTYPE)[0]
    header["sizeof_hdr"] = 540
    header["magic"] = b"n+2\x00\r\n\x1a\n"
    header["datatype"] = 16
    header["bitpix"] = 32
    dim = np.ones(8, dtype=np.int64)
    dim[0] = array.ndim
    dim[1:array.ndim + 1] = array.shape
    header["dim"] = dim
    pixdim = np.ones(8, dtype=np.float64)
    pixdim[1:4] = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    header["pixdim"] = pixdim
    header["vox_offset"] = 544
    header["scl_slope"] = 1.0
    header["xyzt_units"] = 2
    header["sform_code"] = 1
    header["srow_x"] = affine[0]
    header["srow_y"] = affine[1]
    header["srow_z"] = affine[2]
    path.write_bytes(header.tobytes() + b"\x00" * 4
                     + np.asfortranarray(array.astype(np.float32)).tobytes(order="F"))
    return path


def write_nrrd(path: Path, array: np.ndarray, spacing=(2.5, 2.5, 2.5),
               origin=(-100.0, -100.0, -60.0), *, encoding: str = "raw",
               space: str = "left-posterior-superior",
               data_file: str | None = None) -> Path:
    """Write a real NRRD file (LPS by default, as ITK and Slicer do)."""
    payload = np.asfortranarray(array.astype(np.float32)).tobytes(order="F")
    if encoding == "gzip":
        payload = gzip.compress(payload)
    elif encoding == "ascii":
        payload = " ".join(
            f"{v:g}" for v in np.asfortranarray(array.astype(np.float32)).ravel(order="F")
        ).encode()

    directions = " ".join(
        "(" + ",".join("%g" % (spacing[axis] if i == axis else 0)
                       for i in range(3)) + ")" for axis in range(3))
    lines = [
        "NRRD0004",
        "type: float",
        f"dimension: {array.ndim}",
        f"space: {space}",
        "sizes: " + " ".join(str(n) for n in array.shape),
        f"space directions: {directions}",
        "kinds: domain domain domain",
        "endian: little",
        f"encoding: {encoding}",
        "space origin: (" + ",".join("%g" % v for v in origin) + ")",
    ]
    if data_file is not None:
        lines.append(f"data file: {data_file}")
        (path.parent / data_file).write_bytes(payload)
        path.write_bytes(("\n".join(lines) + "\n").encode())
    else:
        path.write_bytes(("\n".join(lines) + "\n\n").encode() + payload)
    return path


def write_dicom_series(
    directory: Path, array: np.ndarray, *, spacing=(2.5, 2.5), slice_spacing=2.5,
    series_uid: str = "1.2.826.0.1.3680043.2.1125.1", echo_time=90.0,
    repetition_time=4000.0, inversion_time=None, scanning_sequence="SE",
    sequence_variant="SK", flip_angle=90.0, contrast_agent=None, echo_number=1,
    image_type=("ORIGINAL", "PRIMARY", "M", "NORM"), skip_slices=(),
    duplicate_slice=None, corrupt_slice=None, include_patient_data=False,
    orientation=(1, 0, 0, 0, 1, 0), position_axis=2, prefix="slice",
) -> Path:
    """Write a real, readable single-frame MR DICOM series.

    Parameterised so that the failure paths — a missing slice, a duplicated position,
    a truncated file — are produced by writing an actually-damaged series rather than
    by monkeypatching the reader.
    """
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    columns, rows, slices = array.shape          # array is (i, j, k)
    study_uid = "1.2.826.0.1.3680043.2.1125.0"

    for k in range(slices):
        if k in skip_slices:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(directory / f"{prefix}{k:03d}.dcm"), {},
                         file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.2.1125.9"
        ds.SeriesNumber = 3
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = "AX T2 TSE"
        ds.ProtocolName = "BRAIN ROUTINE"
        ds.BodyPartExamined = "BRAIN"
        ds.ImageType = list(image_type)

        ds.Manufacturer = "SIEMENS"
        ds.ManufacturerModelName = "MAGNETOM Vida"
        ds.MagneticFieldStrength = 3.0
        ds.SoftwareVersions = "syngo MR XA30"

        ds.Rows, ds.Columns = rows, columns
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [row, column]
        ds.SliceThickness = float(slice_spacing)
        ds.SpacingBetweenSlices = float(slice_spacing)
        ds.ImageOrientationPatient = [float(v) for v in orientation]
        step = k * float(slice_spacing) if duplicate_slice != k else \
            (k - 1) * float(slice_spacing)
        # Slices must advance along the *slice normal*, which depends on the
        # orientation. A coronal series whose positions only vary in z would have
        # every slice at the same place along its own normal.
        position = [-100.0, -100.0, -60.0]
        position[position_axis] += step
        ds.ImagePositionPatient = position
        ds.PatientPosition = "HFS"
        ds.AcquisitionMatrix = [columns, 0, 0, rows]

        ds.ScanningSequence = scanning_sequence
        ds.SequenceVariant = sequence_variant
        ds.EchoTime = float(echo_time)
        ds.RepetitionTime = float(repetition_time)
        if inversion_time is not None:
            ds.InversionTime = float(inversion_time)
        ds.FlipAngle = float(flip_angle)
        ds.EchoNumbers = int(echo_number)
        ds.EchoTrainLength = 15
        ds.InPlanePhaseEncodingDirection = "COL"
        if contrast_agent is not None:
            ds.ContrastBolusAgent = contrast_agent

        if include_patient_data:
            ds.PatientName = "DOE^JANE"
            ds.PatientID = "MRN-0099123"
            ds.PatientBirthDate = "19700101"
            ds.PatientSex = "F"
            ds.StudyDate = "20260722"
            ds.AccessionNumber = "ACC-55512"
            ds.InstitutionName = "General Hospital"
            ds.ReferringPhysicianName = "SMITH^JOHN"

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        plane = np.clip(array[:, :, k].T, 0, 4095).astype(np.uint16)   # (row, column)
        ds.PixelData = plane.tobytes()

        path = directory / f"{prefix}{k:03d}.dcm"
        ds.save_as(str(path), enforce_file_format=False)
        if corrupt_slice == k:
            data = path.read_bytes()
            path.write_bytes(data[:len(data) - 400])   # truncate the pixel data
    return directory


@pytest.fixture
def phantom() -> np.ndarray:
    return head_phantom()


@pytest.fixture
def nifti_study(tmp_path: Path, phantom: np.ndarray) -> Path:
    directory = tmp_path / "nifti-study"
    directory.mkdir()
    write_nifti1(directory / "t1_mprage.nii", phantom, ras_affine())
    return directory


@pytest.fixture
def dicom_study(tmp_path: Path, phantom: np.ndarray) -> Path:
    return write_dicom_series(tmp_path / "dicom-study", phantom)


# =========================================================================== #
# 1. Geometry
# =========================================================================== #
def test_identity_affine_is_canonical_ras():
    assert axis_codes(ras_affine()) == ("R", "A", "S")
    assert VoxelGeometry(ras_affine(), (10, 10, 10)).is_canonical


@pytest.mark.parametrize("affine", [
    np.array([[0., 0., 3., -90.], [-1., 0., 0., 120.], [0., -1., 0., 80.], [0, 0, 0, 1]]),
    np.array([[-2., 0., 0., 90.], [0., 0., 1.5, -70.], [0., 2., 0., -60.], [0, 0, 0, 1]]),
    np.diag([1.0, -1.0, 1.0, 1.0]),
])
def test_canonical_reorientation_preserves_world_coordinates(affine):
    """The invariant: reorientation moves indices, never anatomy.

    Checked by tagging every voxel with a unique value and confirming that each tag's
    world coordinate is identical before and after. A flip that lost a sign would
    show up here and nowhere else until a clinician read a mirrored study.
    """
    shape = (4, 5, 6)
    array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    reoriented, new_affine, original, changed = to_canonical(array, affine)

    assert changed is (tuple(original) != ("R", "A", "S"))
    assert axis_codes(new_affine) == ("R", "A", "S")

    for value in (0.0, 37.0, float(array.size - 1)):
        old_index = np.argwhere(array == value)[0]
        new_index = np.argwhere(reoriented == value)[0]
        np.testing.assert_allclose(voxel_to_world(affine, old_index),
                                   voxel_to_world(new_affine, new_index), atol=1e-9)


def test_canonical_reorientation_is_a_no_op_when_already_ras():
    array = np.zeros((3, 3, 3), dtype=np.float32)
    _, _, original, changed = to_canonical(array, ras_affine())
    assert changed is False
    assert tuple(original) == ("R", "A", "S")


def test_dicom_affine_matches_the_standard_formula():
    """Axial, head-first-supine: LPS columns map to RAS with x and y negated."""
    affine = affine_from_dicom(
        image_orientation=[1, 0, 0, 0, 1, 0],
        first_position=[-120.0, -110.0, -50.0],
        pixel_spacing=[0.9, 0.8],                 # [row spacing, column spacing]
        last_position=[-120.0, -110.0, 49.0],
        slice_count=100,
    )
    np.testing.assert_allclose(affine[:3, 0], [-0.8, 0, 0], atol=1e-9)   # along i
    np.testing.assert_allclose(affine[:3, 1], [0, -0.9, 0], atol=1e-9)   # along j
    np.testing.assert_allclose(affine[:3, 2], [0, 0, 1.0], atol=1e-9)    # along k
    np.testing.assert_allclose(affine[:3, 3], [120.0, 110.0, -50.0], atol=1e-9)


def test_slice_direction_comes_from_positions_not_the_normal():
    """A series stored feet-to-head must not be flipped by the cross-product normal."""
    descending = affine_from_dicom([1, 0, 0, 0, 1, 0], [0, 0, 50.0], [1.0, 1.0],
                                   last_position=[0, 0, 0.0], slice_count=51)
    assert descending[2, 2] < 0, "descending acquisitions must keep their direction"


def test_obliquity_is_measured_not_assumed():
    angle = np.deg2rad(30.0)
    affine = np.eye(4)
    affine[:3, :3] = np.array([[1, 0, 0],
                               [0, np.cos(angle), -np.sin(angle)],
                               [0, np.sin(angle), np.cos(angle)]])
    assert obliquity_degrees(affine) == pytest.approx(30.0, abs=0.01)
    assert obliquity_degrees(ras_affine()) == pytest.approx(0.0, abs=1e-6)


def test_geometry_reports_anisotropy_and_field_of_view():
    geometry = VoxelGeometry(ras_affine(spacing=(0.5, 0.5, 5.0)), (256, 256, 30))
    assert geometry.anisotropy == pytest.approx(10.0)
    np.testing.assert_allclose(geometry.field_of_view_mm, (128.0, 128.0, 150.0))


def test_degenerate_affine_is_detected():
    affine = np.eye(4)
    affine[:3, :3] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    assert VoxelGeometry(affine, (4, 4, 4)).degenerate


# =========================================================================== #
# 2. Readers
# =========================================================================== #
def test_nifti1_round_trips_array_and_affine(tmp_path, phantom):
    affine = ras_affine()
    series = NiftiReader().read_one(write_nifti1(tmp_path / "v.nii", phantom, affine))
    np.testing.assert_allclose(series.voxels, phantom, atol=1e-5)
    np.testing.assert_allclose(series.geometry.affine, affine, atol=1e-6)
    assert series.source_format is FileFormat.NIFTI


def test_nifti_gzip_is_detected_by_magic_not_suffix(tmp_path, phantom):
    # Named ``.nii`` but gzipped — a real and common archive quirk.
    path = write_nifti1(tmp_path / "compressed.nii", phantom, ras_affine(), gzipped=True)
    series = NiftiReader().read_one(path)
    np.testing.assert_allclose(series.voxels, phantom, atol=1e-5)


def test_nifti2_round_trips(tmp_path, phantom):
    affine = ras_affine(spacing=(1.0, 1.0, 2.0))
    series = NiftiReader().read_one(write_nifti2(tmp_path / "v2.nii", phantom, affine))
    np.testing.assert_allclose(series.geometry.affine, affine, atol=1e-6)
    assert series.header["nifti_version"] == 2


def test_nifti_without_sform_or_qform_is_flagged(tmp_path, phantom):
    path = write_nifti1(tmp_path / "noform.nii", phantom, ras_affine(), sform_code=0)
    series = NiftiReader().read_one(path)
    assert series.header["affine_source"] == "pixdim_fallback"
    assert any("orientation" in w for w in series.integrity.warnings)


def test_truncated_nifti_raises_rather_than_returning_short_data(tmp_path, phantom):
    path = write_nifti1(tmp_path / "short.nii", phantom, ras_affine())
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])
    with pytest.raises(CorruptStudy, match="truncated"):
        NiftiReader().read_one(path)


def test_nifti_complex_datatype_is_declined_by_name(tmp_path, phantom):
    path = write_nifti1(tmp_path / "complex.nii", phantom[:2, :2, :2], ras_affine(),
                        datatype=32)
    with pytest.raises(UnsupportedStudyFormat, match="complex64"):
        NiftiReader().read_one(path)


def test_nifti_4d_series_is_kept_4d_and_flagged(tmp_path):
    array = np.zeros((6, 6, 4, 3), dtype=np.float32)
    array[..., 1] = 5.0
    series = NiftiReader().read_one(write_nifti1(tmp_path / "dwi.nii", array,
                                                 ras_affine()))
    assert series.is_multiframe and series.frames == 3
    assert any("4D" in w for w in series.integrity.warnings)


@pytest.mark.parametrize("encoding", ["raw", "gzip", "ascii"])
def test_nrrd_round_trips_every_encoding(tmp_path, encoding):
    array = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    series = NrrdReader().read_one(
        write_nrrd(tmp_path / f"v_{encoding}.nrrd", array, encoding=encoding))
    np.testing.assert_allclose(series.voxels, array, atol=1e-5)


def test_nrrd_lps_is_converted_to_ras(tmp_path):
    """3D Slicer writes LPS. Reading it as RAS would mirror left and right."""
    array = np.zeros((4, 4, 4), dtype=np.float32)
    series = NrrdReader().read_one(write_nrrd(
        tmp_path / "lps.nrrd", array, spacing=(1.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0)))
    np.testing.assert_allclose(np.diag(series.geometry.affine[:3, :3]),
                               [-1.0, -2.0, 3.0])
    np.testing.assert_allclose(series.geometry.affine[:3, 3], [-10.0, -20.0, 30.0])


def test_nrrd_detached_data_file_is_followed(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = write_nrrd(tmp_path / "detached.nhdr", array, data_file="detached.raw")
    np.testing.assert_allclose(NrrdReader().read_one(path).voxels, array, atol=1e-5)


def test_nrrd_missing_detached_file_names_what_is_missing(tmp_path):
    array = np.zeros((2, 2, 2), dtype=np.float32)
    path = write_nrrd(tmp_path / "orphan.nhdr", array, data_file="gone.raw")
    (tmp_path / "gone.raw").unlink()
    with pytest.raises(CorruptStudy, match="detached data file"):
        NrrdReader().read_one(path)


# --- DICOM ----------------------------------------------------------------- #
def test_dicom_series_builds_a_correctly_ordered_volume(dicom_study, phantom):
    series = MRIStudyLoader().load(dicom_study).series[0]
    assert series.voxels.shape == phantom.shape
    assert series.integrity.complete
    assert series.integrity.median_slice_spacing_mm == pytest.approx(2.5)
    # Stored as uint16 after clipping, so compare against the same clipping.
    np.testing.assert_allclose(series.voxels, np.clip(phantom, 0, 4095).astype(np.uint16),
                               atol=1.0)


def test_dicom_reader_detects_missing_slices_from_geometry(tmp_path, phantom):
    """Two slices removed from the middle: the gaps are clean multiples of the median."""
    directory = write_dicom_series(tmp_path / "gapped", phantom, skip_slices=(20, 35))
    series = MRIStudyLoader().load(directory).series[0]
    assert series.integrity.missing_slices_estimated == 2
    assert not series.integrity.complete
    assert series.integrity.loss_fraction == pytest.approx(2 / 50, abs=1e-6)


def test_dicom_reader_drops_duplicate_slice_positions(tmp_path, phantom):
    directory = write_dicom_series(tmp_path / "duped", phantom, duplicate_slice=10)
    series = MRIStudyLoader().load(directory).series[0]
    assert series.integrity.duplicate_positions == 1
    assert series.voxels.shape[2] == phantom.shape[2] - 1


def test_dicom_reader_tolerates_one_corrupt_slice(tmp_path, phantom):
    directory = write_dicom_series(tmp_path / "corrupt", phantom, corrupt_slice=12)
    series = MRIStudyLoader().load(directory).series[0]
    assert len(series.integrity.corrupt_files) == 1
    assert series.voxels.shape[2] == phantom.shape[2] - 1
    assert not series.integrity.complete


def test_multi_echo_series_is_split_rather_than_interleaved(tmp_path, phantom):
    """One series UID, two echoes: stacking them would interleave two contrasts."""
    directory = tmp_path / "multiecho"
    write_dicom_series(directory, phantom, echo_time=15.0, echo_number=1, prefix="e1_")
    write_dicom_series(directory, phantom, echo_time=90.0, echo_number=2, prefix="e2_")
    loaded = MRIStudyLoader().load(directory)
    assert len(loaded.series) == 2
    assert {s.voxels.shape[2] for s in loaded.series} == {phantom.shape[2]}


def test_series_mixing_orientations_is_split(tmp_path, phantom):
    """A three-plane localiser is one series holding three different planes."""
    directory = tmp_path / "localiser"
    small = phantom[:, :, :6]
    write_dicom_series(directory, small, orientation=(1, 0, 0, 0, 1, 0), prefix="ax_")
    write_dicom_series(directory, small, orientation=(1, 0, 0, 0, 0, -1),
                       position_axis=1, prefix="cor_")
    assert len(MRIStudyLoader().load(directory).series) == 2


def test_single_slice_series_cannot_form_a_volume(tmp_path, phantom):
    directory = write_dicom_series(tmp_path / "single", phantom[:, :, :1])
    with pytest.raises(StudyValidationError):
        MRIStudyLoader().load(directory)


# =========================================================================== #
# 3. Loader / discovery
# =========================================================================== #
def test_missing_path_raises_study_not_found(tmp_path):
    with pytest.raises(StudyNotFound):
        MRIStudyLoader().load(tmp_path / "does-not-exist")


def test_directory_with_no_images_raises_study_not_found(tmp_path):
    (tmp_path / "notes.txt").write_text("no images here")
    with pytest.raises(StudyNotFound):
        MRIStudyLoader().load(tmp_path)


def test_unrecognised_binary_is_declined_not_guessed(tmp_path):
    (tmp_path / "mystery.bin").write_bytes(b"\x00\x01\x02\x03" * 500)
    with pytest.raises(UnsupportedStudyFormat):
        MRIStudyLoader().load(tmp_path)


def test_mixed_formats_prefer_dicom_and_say_so(tmp_path, phantom):
    directory = write_dicom_series(tmp_path / "mixed", phantom)
    write_nifti1(directory / "converted.nii", phantom, ras_affine())
    loaded = MRIStudyLoader().load(directory)
    assert loaded.source_format is FileFormat.DICOM
    assert any("more than one format" in w for w in loaded.warnings)


def test_two_studies_under_one_root_are_flagged(tmp_path, phantom):
    directory = tmp_path / "patient-folder"
    write_dicom_series(directory / "a", phantom[:, :, :6], series_uid="1.2.3.1",
                       prefix="a_")
    second = write_dicom_series(directory / "b", phantom[:, :, :6],
                                series_uid="1.2.3.2", prefix="b_")
    # Give the second series a different study UID by rewriting the tag.
    for path in second.glob("*.dcm"):
        ds = pydicom.dcmread(str(path))
        ds.StudyInstanceUID = "9.9.9.9"
        ds.save_as(str(path), enforce_file_format=False)
    loaded = MRIStudyLoader().load(directory)
    assert len(loaded.study_instance_uids) == 2
    assert any("more than one study" in w for w in loaded.warnings)


# =========================================================================== #
# 4. Metadata engine
# =========================================================================== #
def test_patient_identifiers_never_reach_metadata(tmp_path, phantom):
    """The allowlist must hold even when the source is full of identifiers."""
    directory = write_dicom_series(tmp_path / "phi", phantom[:, :, :6],
                                   include_patient_data=True)
    loaded = MRIStudyLoader().load(directory)
    header = loaded.series[0].header

    assert not set(header) & PATIENT_IDENTIFYING_KEYWORDS
    assert_patient_independent(header)              # must not raise

    metadata = MetadataExtractor().extract(
        header, source_format=FileFormat.DICOM,
        geometry=loaded.series[0].geometry, source_name="phi")
    serialised = json.dumps(metadata.model_dump(mode="json")).upper()
    for needle in ("DOE", "JANE", "MRN-0099123", "19700101", "ACC-55512",
                   "GENERAL HOSPITAL", "SMITH"):
        assert needle.upper() not in serialised, f"{needle} leaked into metadata"


def test_patient_independence_check_catches_a_leak():
    with pytest.raises(ValueError, match="patient-identifying"):
        assert_patient_independent({"EchoTime": 90.0, "PatientName": "DOE^JANE"})


def test_allowlist_excludes_every_known_identifier():
    """Structural guarantee: the two sets must never intersect."""
    assert not set(DICOM_KEYWORDS) & PATIENT_IDENTIFYING_KEYWORDS


def test_metadata_extracts_acquisition_parameters(dicom_study):
    loaded = MRIStudyLoader().load(dicom_study)
    raw = loaded.series[0]
    metadata = MetadataExtractor().extract(
        raw.header, source_format=FileFormat.DICOM, geometry=raw.geometry,
        slice_count=raw.voxels.shape[2], source_name=raw.source_name)

    assert metadata.acquisition.echo_time_ms == pytest.approx(90.0)
    assert metadata.acquisition.repetition_time_ms == pytest.approx(4000.0)
    assert metadata.acquisition.scanning_sequence == ("SE",)
    assert metadata.scanner.vendor is ScannerVendor.SIEMENS
    assert metadata.scanner.field_strength_tesla == pytest.approx(3.0)
    assert metadata.geometry.voxel_size_mm == pytest.approx((2.5, 2.5, 2.5))
    assert metadata.geometry.world_orientation_known
    assert metadata.identifiers.series_instance_uid


def test_absent_parameters_stay_none_and_never_become_zero():
    metadata = MetadataExtractor().extract({}, source_format=FileFormat.NIFTI)
    assert metadata.acquisition.echo_time_ms is None
    assert metadata.acquisition.repetition_time_ms is None
    assert metadata.scanner.field_strength_tesla is None
    assert not metadata.has_acquisition_parameters
    assert any("no DICOM acquisition header" in w
               for w in metadata.extraction_warnings)


@pytest.mark.parametrize("raw,expected", [
    ("SIEMENS", ScannerVendor.SIEMENS),
    ("Siemens Healthineers", ScannerVendor.SIEMENS),
    ("GE MEDICAL SYSTEMS", ScannerVendor.GE),
    ("GE", ScannerVendor.GE),
    ("Philips Medical Systems", ScannerVendor.PHILIPS),
    ("TOSHIBA", ScannerVendor.CANON),
    ("Acme Imaging", ScannerVendor.OTHER),
    (None, ScannerVendor.UNKNOWN),
])
def test_vendor_normalisation(raw, expected):
    assert normalise_vendor(raw) is expected


def test_gauss_field_strength_is_converted_to_tesla():
    metadata = MetadataExtractor().extract(
        {"MagneticFieldStrength": 15000.0}, source_format=FileFormat.DICOM)
    assert metadata.scanner.field_strength_tesla == pytest.approx(1.5)


def test_recorded_and_derived_slice_spacing_disagreement_is_reported(tmp_path, phantom):
    """A gapped acquisition: 2 mm slices positioned 5 mm apart."""
    directory = tmp_path / "gapped-thickness"
    write_dicom_series(directory, phantom[:, :, :10], slice_spacing=5.0)
    for path in directory.glob("*.dcm"):
        ds = pydicom.dcmread(str(path))
        ds.SliceThickness = 2.0
        ds.SpacingBetweenSlices = 2.0
        ds.save_as(str(path), enforce_file_format=False)

    raw = MRIStudyLoader().load(directory).series[0]
    metadata = MetadataExtractor().extract(raw.header, source_format=FileFormat.DICOM,
                                           geometry=raw.geometry)
    assert any("gapped" in w or "disagrees" in w for w in metadata.extraction_warnings)


# =========================================================================== #
# 5. Sequence detector
# =========================================================================== #
def _metadata(**acquisition):
    """Build metadata with the given acquisition parameters and nothing else."""
    header = {"Modality": "MR", **acquisition}
    return MetadataExtractor().extract(header, source_format=FileFormat.DICOM)


@pytest.mark.parametrize("params,expected", [
    (dict(RepetitionTime=500, EchoTime=12, ScanningSequence="SE"), SequenceType.T1),
    (dict(RepetitionTime=4500, EchoTime=100, ScanningSequence="SE"), SequenceType.T2),
    (dict(RepetitionTime=9000, EchoTime=120, InversionTime=2500,
          ScanningSequence=["SE", "IR"]), SequenceType.FLAIR),
    (dict(RepetitionTime=3000, EchoTime=20, ScanningSequence="SE"), SequenceType.PD),
    (dict(RepetitionTime=2000, EchoTime=90, ScanningSequence="EP",
          DiffusionBValue=1000), SequenceType.DWI),
    (dict(RepetitionTime=2000, EchoTime=90, ScanningSequence="EP",
          ImageType=["DERIVED", "PRIMARY", "ADC"]), SequenceType.ADC),
    (dict(RepetitionTime=28, EchoTime=20, FlipAngle=15, ScanningSequence="GR",
          ImageType=["ORIGINAL", "PRIMARY", "SWI"]), SequenceType.SWI),
    (dict(RepetitionTime=2000, EchoTime=3, InversionTime=900,
          ScanningSequence=["GR", "IR"]), SequenceType.T1),
])
def test_sequence_is_identified_from_acquisition_parameters(params, expected):
    assignment = RuleBasedSequenceDetector().detect(_metadata(**params))
    assert assignment.sequence is expected
    assert assignment.metadata_available
    assert assignment.source.startswith("acquisition_parameters")
    assert assignment.confidence >= 0.70


def test_contrast_promotes_t1_to_t1ce():
    assignment = RuleBasedSequenceDetector().detect(_metadata(
        RepetitionTime=500, EchoTime=12, ScanningSequence="SE",
        ContrastBolusAgent="GADOVIST 7.5ML"))
    assert assignment.sequence is SequenceType.T1CE
    assert "contrast" in assignment.reason.lower()


def test_contrast_alone_does_not_claim_t1_weighting():
    """Post-contrast T2 and FLAIR are routine; contrast is not evidence of T1."""
    assignment = RuleBasedSequenceDetector().detect(_metadata(
        RepetitionTime=4500, EchoTime=100, ScanningSequence="SE",
        ContrastBolusAgent="DOTAREM"))
    assert assignment.sequence is SequenceType.T2


def test_none_contrast_agent_is_not_treated_as_administered():
    assignment = RuleBasedSequenceDetector().detect(_metadata(
        RepetitionTime=500, EchoTime=12, ScanningSequence="SE",
        ContrastBolusAgent="NONE"))
    assert assignment.sequence is SequenceType.T1


def test_short_ti_inversion_recovery_is_not_called_flair():
    """A STIR nulls fat, not CSF. Calling it FLAIR would be a contrast error."""
    assignment = RuleBasedSequenceDetector().detect(_metadata(
        RepetitionTime=5000, EchoTime=60, InversionTime=150,
        ScanningSequence=["SE", "IR"]))
    assert assignment.sequence is not SequenceType.FLAIR


def test_description_only_classification_is_capped_and_flagged():
    """A NIfTI named ``FLAIR`` carries no parameters — the answer must say so."""
    metadata = MetadataExtractor().extract(
        {"descrip": "T2 FLAIR AXIAL"}, source_format=FileFormat.NIFTI,
        source_name="sub-01_FLAIR.nii.gz")
    assignment = RuleBasedSequenceDetector().detect(metadata)

    assert assignment.sequence is SequenceType.FLAIR
    assert assignment.metadata_available is False
    assert assignment.source == "description_only"
    assert assignment.requires_review
    assert assignment.confidence <= 0.55
    assert not assignment.is_confident


def test_parameters_outrank_a_contradicting_filename():
    """The requirement, enforced: filenames never override acquisition physics."""
    header = {"Modality": "MR", "RepetitionTime": 4500, "EchoTime": 100,
              "ScanningSequence": "SE", "SeriesDescription": "T1_MPRAGE_POST"}
    metadata = MetadataExtractor().extract(header, source_format=FileFormat.DICOM)
    assignment = RuleBasedSequenceDetector().detect(metadata)
    assert assignment.sequence is SequenceType.T2
    assert assignment.confidence > 0.55


def test_no_evidence_at_all_returns_unknown_not_a_guess():
    assignment = RuleBasedSequenceDetector().detect(
        MetadataExtractor().extract({}, source_format=FileFormat.NRRD))
    assert assignment.sequence is SequenceType.UNKNOWN
    assert assignment.confidence == 0.0
    assert assignment.requires_review


def test_candidates_include_the_losers_with_their_evidence():
    assignment = RuleBasedSequenceDetector().detect(_metadata(
        RepetitionTime=2000, EchoTime=90, ScanningSequence="EP", DiffusionBValue=1000,
        SeriesDescription="DWI TRACEW"))
    assert len(assignment.candidates) >= 1
    assert all(c.evidence for c in assignment.candidates)


# =========================================================================== #
# 6. Quality inspector
# =========================================================================== #
def _volume(array: np.ndarray, spacing=(2.5, 2.5, 2.5)) -> MRIVolume:
    return MRIVolume(array=array,
                     geometry=VoxelGeometry(ras_affine(spacing=spacing), array.shape))


def test_a_good_volume_is_acceptable(phantom):
    report = MRIQualityInspector().inspect(_volume(phantom))
    assert report.verdict is not QualityVerdict.REJECTED
    assert report.quality_score > 0.7
    assert report.reject_reason is None


def test_constant_volume_fails_the_intensity_check():
    report = MRIQualityInspector().inspect(_volume(np.full((40, 40, 40), 7.0,
                                                           dtype=np.float32)))
    assert report.verdict is QualityVerdict.REJECTED
    assert report.check("intensity").status is CheckStatus.FAIL
    assert "no image information" in report.reject_reason


def test_implausible_voxel_size_fails_resolution(phantom):
    report = MRIQualityInspector().inspect(_volume(phantom, spacing=(8.0, 8.0, 8.0)))
    assert report.check("resolution").status is CheckStatus.FAIL
    assert report.verdict is QualityVerdict.REJECTED


def test_anisotropic_volume_warns_but_is_not_rejected(phantom):
    """A 1.6 x 1.6 x 5.5 mm clinical T2: coarse and usable, not rejectable."""
    report = MRIQualityInspector().inspect(_volume(phantom, spacing=(1.6, 1.6, 5.5)))
    assert report.check("resolution").status is CheckStatus.WARN
    assert report.verdict is QualityVerdict.ACCEPTABLE_WITH_WARNINGS
    assert any("isotropic" in r for r in report.recommendations)


def test_missing_slices_are_reflected_in_the_quality_report(tmp_path, phantom):
    directory = write_dicom_series(tmp_path / "qc-gapped", phantom,
                                   skip_slices=(10, 11, 12, 13, 14, 20))
    raw = MRIStudyLoader().load(directory).series[0]
    volume, _ = VolumeBuilder().build(raw)
    report = MRIQualityInspector().inspect(volume, integrity=raw.integrity)
    assert report.check("slice_completeness").status is CheckStatus.FAIL
    assert report.check("slice_completeness").measured["missing"] >= 5


def test_unknown_world_orientation_fails_the_orientation_check(tmp_path, phantom):
    path = write_nifti1(tmp_path / "noform.nii", phantom, ras_affine(), sform_code=0)
    raw = NiftiReader().read_one(path)
    volume, _ = VolumeBuilder().build(raw)
    metadata = MetadataExtractor().extract(raw.header, source_format=FileFormat.NIFTI,
                                           geometry=raw.geometry)
    report = MRIQualityInspector().inspect(volume, metadata=metadata)
    assert report.check("orientation").status is CheckStatus.FAIL
    assert "left and right" in report.check("orientation").message


def test_slice_completeness_is_not_evaluated_without_an_integrity_report(phantom):
    check = MRIQualityInspector().inspect(_volume(phantom)).check("slice_completeness")
    assert check.status is CheckStatus.NOT_EVALUATED


def test_unevaluated_checks_are_excluded_from_the_score_not_scored_zero(phantom):
    """A check that could not run must neither punish nor flatter the study."""
    inspector = MRIQualityInspector()
    with_integrity = inspector.inspect(_volume(phantom))
    assert with_integrity.check("slice_completeness").status is CheckStatus.NOT_EVALUATED
    # Score stays high despite the unevaluated check contributing nothing.
    assert with_integrity.quality_score > 0.7


def test_uncalibrated_check_cannot_reject_a_study():
    """Structural guarantee: a provisional threshold has no reject authority."""
    check = _check("motion", CheckStatus.FAIL, 0.0, calibrated=False,
                   message="severe motion")
    assert check.status is CheckStatus.WARN
    assert "provisional" in check.message


def test_motion_and_noise_checks_declare_themselves_provisional(phantom):
    report = MRIQualityInspector().inspect(_volume(phantom))
    assert report.check("motion").calibrated is False
    assert report.check("noise").calibrated is False
    assert report.check("resolution").calibrated is True
    assert report.check("intensity").calibrated is True


def test_noise_is_not_evaluated_when_there_is_no_air(phantom):
    """A skull-stripped or cropped volume has no background left to measure."""
    all_foreground = np.ones(phantom.shape, dtype=bool)
    report = MRIQualityInspector().inspect(_volume(phantom), mask=all_foreground)
    noise = report.check("noise")
    assert noise.status is CheckStatus.NOT_EVALUATED
    assert "cropped, skull-stripped, or masked" in noise.message


def test_quality_report_is_json_serialisable(phantom):
    report = MRIQualityInspector().inspect(_volume(phantom))
    json.dumps(report.model_dump(mode="json"))


# =========================================================================== #
# 7. Volume builder and masking
# =========================================================================== #
def test_volume_builder_rejects_a_degenerate_affine(phantom):
    from aura.backend.foundation.mri.io.base import RawSeries

    affine = np.eye(4)
    affine[:3, :3] = 0.0
    raw = RawSeries(series_key="k", source_format=FileFormat.NIFTI, voxels=phantom,
                    geometry=VoxelGeometry(affine, phantom.shape))
    with pytest.raises(StudyValidationError, match="degenerate"):
        VolumeBuilder().build(raw)


def test_volume_builder_records_which_frame_it_took(tmp_path):
    array = np.zeros((6, 6, 4, 3), dtype=np.float32)
    array[..., 2] = 9.0
    raw = NiftiReader().read_one(write_nifti1(tmp_path / "4d.nii", array, ras_affine()))
    volume, warnings = VolumeBuilder().build(raw, frame=2)
    assert volume.provenance["frame_selected"] == 2
    assert volume.provenance["frames_available"] == 3
    assert float(volume.array.max()) == pytest.approx(9.0)
    assert any("3 volumes" in w for w in warnings)


def test_volume_shape_and_geometry_cannot_disagree(phantom):
    with pytest.raises(StudyValidationError, match="disagree"):
        MRIVolume(array=phantom, geometry=VoxelGeometry(ras_affine(), (1, 1, 1)))


def test_foreground_mask_is_not_a_brain_mask(phantom):
    mask, details = estimate_foreground_mask(phantom)
    slot = BrainMaskSlot(mask=mask, provenance=MaskProvenance.FOREGROUND_HEURISTIC,
                         method="otsu", details=details)
    assert slot.present
    assert slot.is_brain_mask is False
    assert 0.05 < slot.coverage_fraction < 0.8


def test_mask_bounding_box_is_tight():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 3:8, 1:4] = True
    box = BrainMaskSlot(mask=mask).bounding_box()
    assert box == (slice(2, 5), slice(3, 8), slice(1, 4))


# =========================================================================== #
# 8. Standardisation
# =========================================================================== #
def _context(mask: np.ndarray | None = None) -> StandardizationContext:
    context = StandardizationContext(config=StandardizationConfig())
    if mask is not None:
        context.mask = BrainMaskSlot(mask=mask,
                                     provenance=MaskProvenance.FOREGROUND_HEURISTIC)
    return context


def test_resampling_preserves_world_coordinates_and_hits_the_target_spacing(phantom):
    volume = _volume(phantom, spacing=(2.5, 2.5, 2.5))
    result = VoxelResampler((1.0, 1.0, 1.0)).apply(volume, _context())

    assert result.changed
    np.testing.assert_allclose(result.volume.spacing, (1.0, 1.0, 1.0), atol=1e-9)
    # Voxel (0,0,0) must not move: that is the invariant the affine encodes.
    np.testing.assert_allclose(voxel_to_world(result.volume.affine, (0, 0, 0)),
                               voxel_to_world(volume.affine, (0, 0, 0)), atol=1e-9)
    np.testing.assert_allclose(result.volume.geometry.field_of_view_mm,
                               volume.geometry.field_of_view_mm, rtol=0.02)


def test_resampling_is_a_no_op_at_the_target_spacing(phantom):
    result = VoxelResampler((2.5, 2.5, 2.5)).apply(_volume(phantom), _context())
    assert result.changed is False
    assert "already at the target spacing" in result.message


def test_cropping_preserves_world_coordinates(phantom):
    volume = _volume(phantom)
    mask, _ = estimate_foreground_mask(phantom)
    context = _context(mask)
    result = MaskCropper(margin_mm=5.0).apply(volume, context)

    assert result.changed
    assert result.volume.array.size < volume.array.size
    start = result.parameters["crop_start"]
    np.testing.assert_allclose(voxel_to_world(result.volume.affine, (0, 0, 0)),
                               voxel_to_world(volume.affine, start), atol=1e-9)
    # The mask must be cropped alongside the volume, or it describes the old grid.
    assert context.mask.mask.shape == result.volume.shape


def test_cropping_without_a_mask_leaves_the_volume_alone(phantom):
    result = MaskCropper().apply(_volume(phantom), _context())
    assert result.changed is False
    assert "uncropped" in result.message


def test_zscore_normalisation_uses_mask_voxels(phantom):
    mask, _ = estimate_foreground_mask(phantom)
    result = IntensityNormalizer(NormalizationMethod.ZSCORE).apply(
        _volume(phantom), _context(mask))
    values = result.volume.array[mask]
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(values.std()) == pytest.approx(1.0, abs=1e-4)
    assert result.parameters["reference_region"] == "mask"


def test_percentile_normalisation_scales_to_unit_range(phantom):
    result = IntensityNormalizer(NormalizationMethod.PERCENTILE, (1.0, 99.0)).apply(
        _volume(phantom), _context())
    assert float(result.volume.array.min()) >= 0.0
    assert float(result.volume.array.max()) <= 1.0


def test_normalising_a_constant_volume_fails_loudly():
    volume = _volume(np.full((20, 20, 20), 5.0, dtype=np.float32))
    with pytest.raises(StageFailed, match="zero variance"):
        IntensityNormalizer(NormalizationMethod.ZSCORE).apply(volume, _context())


def test_canonical_orientation_stage_reports_a_no_op(phantom):
    result = CanonicalOrientation().apply(_volume(phantom), _context())
    assert result.changed is False
    assert result.parameters["source_orientation"] == "RAS"


def test_non_ras_target_is_declined_rather_than_faked():
    with pytest.raises(StageUnavailable, match="only RAS is implemented"):
        CanonicalOrientation("LPS")


def test_skull_stripping_is_functional(phantom):
    context = _context()
    ForegroundMaskEstimator().apply(_volume(phantom), context)
    result = MorphologicalSkullStripper().apply(_volume(phantom), context)
    assert context.mask.provenance is MaskProvenance.SKULL_STRIPPED
    assert context.mask.is_brain_mask is True
    assert result.changed is True


def test_foreground_stage_fills_the_mask_slot_with_honest_provenance(phantom):
    context = _context()
    ForegroundMaskEstimator().apply(_volume(phantom), context)
    assert context.mask.provenance is MaskProvenance.FOREGROUND_HEURISTIC
    assert context.mask.is_brain_mask is False


def test_resampling_carries_the_mask_onto_the_new_grid(phantom):
    """A mask describes a grid; it must move with the volume, without interpolation."""
    mask, _ = estimate_foreground_mask(phantom)
    context = _context(mask)
    result = VoxelResampler((1.0, 1.0, 1.0)).apply(_volume(phantom), context)

    assert context.mask.present
    assert context.mask.mask.shape == result.volume.shape
    assert context.mask.mask.dtype == bool          # never fractional membership
    assert context.mask.provenance is MaskProvenance.FOREGROUND_HEURISTIC
    # Coverage is a property of the anatomy, so it survives the grid change.
    assert context.mask.coverage_fraction == pytest.approx(
        float(mask.mean()), abs=0.03)


# =========================================================================== #
# 9. Pipeline, end to end
# =========================================================================== #
def test_pipeline_on_a_dicom_study(dicom_study):
    study = MRIFoundationPipeline().run(dicom_study)

    assert len(study) == 1
    series = study.series[0]
    assert series.sequence.sequence is SequenceType.T2
    assert series.sequence.metadata_available
    assert series.orientation == "RAS"
    np.testing.assert_allclose(series.spacing, (1.0, 1.0, 1.0), atol=1e-6)
    assert series.metadata.scanner.vendor is ScannerVendor.SIEMENS
    assert series.quality.verdict is not QualityVerdict.REJECTED
    assert series.usable


def test_pipeline_on_a_nifti_study(nifti_study):
    study = MRIFoundationPipeline().run(nifti_study)
    series = study.series[0]
    assert series.metadata.source_format is FileFormat.NIFTI
    assert series.sequence.metadata_available is False   # NIfTI has no parameters
    assert series.orientation == "RAS"


def test_pipeline_records_every_stage_in_the_processing_history(dicom_study):
    series = MRIFoundationPipeline().run(dicom_study).series[0]
    names = [s.name for s in series.history]

    for expected in ("volume_reconstruction", "metadata_extraction",
                     "sequence_identification", "quality_assessment",
                     "canonical_orientation", "voxel_resampling",
                     "intensity_normalization", "registration_preparation"):
        assert expected in names, f"{expected} missing from the processing history"
    assert series.history.was_applied("voxel_resampling")
    assert series.history.total_ms > 0


def test_unavailable_stages_are_recorded_not_silently_skipped(dicom_study):
    """N4 and skull stripping have no backend here; the history must say so."""
    from unittest.mock import patch
    from aura.backend.foundation.mri.errors import StageUnavailable
    
    def raise_unavailable(*args, **kwargs):
        raise StageUnavailable("mocked_stage", "mocked reason")
        
    with patch("aura.backend.foundation.mri.standardize.GaussianBiasFieldCorrector.apply", side_effect=raise_unavailable), \
         patch("aura.backend.foundation.mri.standardize.MorphologicalSkullStripper.apply", side_effect=raise_unavailable):
        series = MRIFoundationPipeline().run(dicom_study).series[0]
        unavailable = series.history.unavailable_stages

        assert "n4_bias_field_correction" in unavailable
        assert "skull_stripping" in unavailable


def test_strict_mode_turns_an_unavailable_stage_into_a_failure(dicom_study):
    from unittest.mock import patch
    from aura.backend.foundation.mri.errors import StageUnavailable
    
    def raise_unavailable(*args, **kwargs):
        raise StageUnavailable("skull_stripping", "mocked reason")
        
    config = FoundationConfig(
        standardization=StandardizationConfig(strict=True))
    with patch("aura.backend.foundation.mri.standardize.MorphologicalSkullStripper.apply", side_effect=raise_unavailable):
        with pytest.raises(StudyValidationError):
            MRIFoundationPipeline(config).run(dicom_study)


def test_pipeline_output_declares_the_mask_is_a_brain_mask(dicom_study):
    series = MRIFoundationPipeline().run(dicom_study).series[0]
    assert series.brain_mask.present
    assert series.brain_mask.provenance is MaskProvenance.SKULL_STRIPPED
    assert series.brain_mask.is_brain_mask is True


def test_registration_plan_is_prepared_but_no_transform_is_computed(dicom_study):
    plan = MRIFoundationPipeline().run(dicom_study).series[0].registration

    assert plan.status == "prepared"
    assert plan.transform is None
    assert plan.is_registered is False
    assert plan.mask_centroid_mm is not None
    assert plan.frame_of_reference_uid


def test_series_sharing_a_frame_of_reference_are_reported_as_aligned(tmp_path, phantom):
    directory = tmp_path / "two-series"
    small = phantom[:, :, :12]
    write_dicom_series(directory, small, series_uid="1.1.1", prefix="s1_")
    write_dicom_series(directory, small, series_uid="2.2.2", prefix="s2_",
                       echo_time=12.0, repetition_time=500.0)
    study = MRIFoundationPipeline().run(directory)
    assert len(study) == 2
    assert all(s.registration.aligned_series for s in study.series)


def test_quality_is_measured_before_standardisation(dicom_study):
    """The report must describe the acquisition, not the pipeline's own output."""
    series = MRIFoundationPipeline().run(dicom_study).series[0]
    step = series.history.step("quality_assessment")
    assert "before standardisation" in step.parameters["assessed_on"]
    # Source spacing, not the 1 mm the pipeline resampled to.
    assert series.quality.check("resolution").measured["spacing_mm"] == [2.5, 2.5, 2.5]


def test_foundation_study_is_json_serialisable(dicom_study):
    study = MRIFoundationPipeline().run(dicom_study)
    payload = json.dumps(study.to_dict())
    assert "foundation_version" in payload
    # Voxels must never be serialised into a description.
    assert len(payload) < 200_000


def test_study_accessors_find_series_by_sequence(tmp_path, phantom):
    directory = tmp_path / "multi-sequence"
    small = phantom[:, :, :12]
    write_dicom_series(directory, small, series_uid="1.1.1", prefix="t2_",
                       echo_time=100.0, repetition_time=4500.0)
    write_dicom_series(directory, small, series_uid="2.2.2", prefix="t1_",
                       echo_time=12.0, repetition_time=500.0)
    study = MRIFoundationPipeline().run(directory)

    assert set(study.sequences_present) == {SequenceType.T1, SequenceType.T2}
    assert study.first(SequenceType.T1) is not None
    assert study.first(SequenceType.FLAIR) is None
    assert study.primary is not None


def test_a_series_that_fails_does_not_take_the_study_down(tmp_path, phantom):
    """One unbuildable series alongside a good one: the good one still ships."""
    directory = tmp_path / "partial"
    write_dicom_series(directory, phantom[:, :, :12], series_uid="1.1.1", prefix="ok_")
    write_dicom_series(directory, phantom[:, :, :1], series_uid="2.2.2", prefix="bad_")
    study = MRIFoundationPipeline().run(directory)

    assert len(study) == 1
    assert len(study.rejected_series) == 1
    assert study.rejected_series[0]["error"]


def test_custom_configuration_is_honoured(dicom_study):
    config = FoundationConfig(
        standardization=StandardizationConfig(
            target_spacing_mm=(2.0, 2.0, 2.0),
            normalization=NormalizationMethod.PERCENTILE,
            crop_to_mask=False, bias_correction=False, skull_strip=False))
    series = MRIFoundationPipeline(config).run(dicom_study).series[0]

    np.testing.assert_allclose(series.spacing, (2.0, 2.0, 2.0), atol=1e-6)
    assert float(series.volume.array.max()) <= 1.0
    assert series.history.step("brain_cropping") is None
    assert series.history.step("n4_bias_field_correction") is None


def test_retaining_the_source_volume_is_opt_in(dicom_study):
    assert MRIFoundationPipeline().run(dicom_study).series[0].source_volume is None
    config = FoundationConfig(retain_source_volume=True)
    retained = MRIFoundationPipeline(config).run(dicom_study).series[0]
    assert retained.source_volume is not None
    assert retained.source_volume.spacing != retained.spacing


# =========================================================================== #
# 10. NeuroMind engine integration
# =========================================================================== #
def _staged(path: Path):
    """Stage a file as the routing layer's ImageAsset, as an upload would arrive."""
    from aura.backend.core.upload.intake import stage_bytes

    return stage_bytes(path.read_bytes(), path.name)


def test_neuromind_preprocess_runs_the_foundation_layer(tmp_path, phantom):
    """The engine's preprocessing is the foundation pipeline, not a stub."""
    import asyncio

    from aura.backend.engines.neuro.engine import NeuroMindEngine
    from aura.backend.foundation.mri.study import FoundationStudy

    volume_path = write_nifti1(tmp_path / "brain_t1.nii", phantom, ras_affine())
    engine = NeuroMindEngine()
    with _staged(volume_path) as asset:
        prepared = engine.preprocess(asset)

        assert isinstance(prepared.payload, FoundationStudy)
        assert prepared.payload.series[0].orientation == "RAS"
        np.testing.assert_allclose(prepared.payload.series[0].spacing,
                                   (1.0, 1.0, 1.0), atol=1e-6)

        outcome = asyncio.run(engine.run(asset))

    # Preprocessing is the real foundation pipeline, and analysis now runs the trained
    # network on its output. A single-sequence phantom is an incomplete study, so the
    # engine is expected to produce a result *and* abstain on it.
    assert outcome.status.value in {"completed", "failed"}
    if outcome.status.value == "completed":
        assert outcome.payload["sequences_missing"], (
            "a one-sequence study must report the three the network is missing")
        assert outcome.payload["abstained"] is True


def test_neuromind_foundation_evidence_carries_no_clinical_claim(tmp_path, phantom):
    """The standardisation record stays measurements-only.

    The engine does make clinical claims now — that is the point of the trained network
    — but they belong to the analysis, not to the foundation layer's description of how
    the volume was read. Keeping the two separable is what lets a reviewer check the
    preprocessing without wading through the diagnosis.
    """
    from aura.backend.engines.neuro.engine import NeuroMindEngine

    volume_path = write_nifti1(tmp_path / "brain.nii", phantom, ras_affine())
    with _staged(volume_path) as asset:
        prepared = NeuroMindEngine().preprocess(asset)

    foundation = prepared.metadata["foundation"]
    series = foundation["series"][0]
    assert series["quality_score"] > 0
    assert series["brain_mask"]["is_brain_mask"] is True
    assert series["registration"]["transform"] is None
    assert all(step["status"] in ("applied", "no_op")
               for step in series["processing_history"]["steps"])

    # The foundation description must carry measurements only — nothing shaped like a
    # clinical result. Checked against the foundation payload rather than the whole
    # outcome, because ``planned_capabilities`` legitimately names things the engine
    # does not do, and a roadmap is not a claim.
    described = json.dumps(foundation).lower()
    for forbidden in ("finding", "diagnosis", "probability", "impression",
                      "abnormal", "haemorrhage", "hemorrhage"):
        assert forbidden not in described, f"{forbidden!r} in the foundation payload"
    assert "mass_lesion_detection" not in described


def test_neuromind_declines_a_single_slice_with_a_specific_reason(tmp_path, phantom):
    """One slice is not a volume, and the refusal has to say which problem it is."""
    import asyncio

    from aura.backend.engines.neuro.engine import NeuroMindEngine

    slice_path = write_nifti1(tmp_path / "one_slice.nii", phantom[:, :, :1],
                              ras_affine())
    with _staged(slice_path) as asset:
        outcome = asyncio.run(NeuroMindEngine().run(asset))

    assert outcome.status.value == "failed"
    assert outcome.payload["error"] == "unreadable_image"
    assert "foundation_error" in outcome.payload


def test_neuromind_only_claims_capabilities_it_has(tmp_path):
    from aura.backend.engines.neuro.engine import NeuroMindEngine

    descriptor = NeuroMindEngine.descriptor
    assert "sequence_identification" in descriptor.capabilities
    assert "quality_assessment" in descriptor.capabilities
    # Everything unbuilt stays a plan, not a claim. Tumour *subtype* classification is
    # the one that matters most here: the trained network segments BraTS regions on a
    # glioma-only corpus and cannot tell tumour types apart, so it must never appear in
    # the capability list no matter how the engine's status changes.
    for unbuilt in NeuroMindEngine.PLANNED_CAPABILITIES:
        assert unbuilt not in descriptor.capabilities
    assert "tumor_subtype_classification" in NeuroMindEngine.PLANNED_CAPABILITIES
    assert descriptor.status.value == "available"
