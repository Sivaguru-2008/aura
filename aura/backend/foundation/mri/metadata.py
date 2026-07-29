"""MRI Metadata Engine — typed, patient-independent acquisition metadata.

Readers hand this module a flat mapping of header values. It produces
:class:`MRIMetadata`: a pydantic model with the acquisition parameters every
downstream stage needs, and nothing else.

Two design rules, both load-bearing.

**Patient-independent by construction, not by convention.** Extraction works from an
*allowlist* (:data:`DICOM_KEYWORDS`) rather than by copying a dataset and deleting the
sensitive parts. A denylist is one new DICOM keyword away from leaking; an allowlist
fails closed. Dates are excluded along with names and identifiers — study and birth
dates are identifiers under HIPAA Safe Harbor, and nothing in this layer needs them.
:func:`assert_patient_independent` re-checks the finished model, and a unit test runs
it against a synthetic study stuffed with every identifier.

**Absent is not zero.** Every field is ``Optional`` and defaults to ``None``. A
missing ``EchoTime`` must never arrive downstream as ``0.0``: the sequence detector
treats "TE unknown" and "TE = 0 ms" completely differently, and only one of them is
true. The layer never invents a value it did not read.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .geometry import VoxelGeometry
from .types import (
    AnatomicalPlane,
    FileFormat,
    MRAcquisitionType,
    ScannerVendor,
)

# --------------------------------------------------------------------------- #
# What may be read out of a DICOM header
# --------------------------------------------------------------------------- #
#: The complete set of DICOM keywords this layer extracts. Readers copy these and
#: only these out of a dataset, so a value that is not listed here cannot reach the
#: rest of AURA even by accident.
DICOM_KEYWORDS: tuple[str, ...] = (
    # identity of the acquisition (UIDs are study/series scoped, not patient scoped)
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID",
    "FrameOfReferenceUID", "SeriesNumber", "InstanceNumber", "AcquisitionNumber",
    "Modality", "StudyDescription", "SeriesDescription", "ProtocolName",
    "BodyPartExamined", "ImageType", "ImageComments",
    # scanner
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "SoftwareVersions", "ReceiveCoilName", "TransmitCoilName", "ImagingFrequency",
    # geometry
    "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "ImageOrientationPatient", "ImagePositionPatient", "SliceLocation",
    "AcquisitionMatrix", "ReconstructionDiameter", "PatientPosition",
    "PercentPhaseFieldOfView", "PercentSampling", "NumberOfFrames",
    # pulse sequence
    "ScanningSequence", "SequenceVariant", "ScanOptions", "SequenceName",
    "MRAcquisitionType", "EchoTime", "RepetitionTime", "InversionTime", "FlipAngle",
    "EchoTrainLength", "EchoNumbers", "NumberOfAverages", "PixelBandwidth",
    "InPlanePhaseEncodingDirection", "NumberOfTemporalPositions",
    "TemporalPositionIdentifier",
    # contrast and diffusion
    "ContrastBolusAgent", "ContrastBolusVolume", "ContrastBolusRoute",
    "DiffusionBValue", "DiffusionDirectionality", "DiffusionGradientOrientation",
    # pixel value transform
    "RescaleSlope", "RescaleIntercept", "RescaleType", "PixelRepresentation",
    "BitsAllocated", "BitsStored", "PhotometricInterpretation",
    "SamplesPerPixel", "TransferSyntaxUID",
)

#: Keywords whose presence in an extracted mapping is a privacy defect. Checked by
#: :func:`assert_patient_independent` as defence in depth behind the allowlist.
PATIENT_IDENTIFYING_KEYWORDS: frozenset[str] = frozenset({
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientSex",
    "PatientAge", "PatientWeight", "PatientSize", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames",
    "IssuerOfPatientID", "EthnicGroup", "PatientComments", "MilitaryRank",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "NameOfPhysiciansReadingStudy", "StationName",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime",
    "SeriesTime", "AcquisitionTime", "ContentTime", "AccessionNumber",
    "StudyID", "AdmissionID", "CurrentPatientLocation", "DeviceSerialNumber",
})


def assert_patient_independent(mapping: Mapping[str, Any]) -> None:
    """Raise if ``mapping`` carries a patient identifier.

    Cheap enough to run on every extraction. The cost of the check is nothing next to
    the cost of a patient name reaching a log line or a cached foundation artefact.
    """
    leaked = sorted(set(mapping) & PATIENT_IDENTIFYING_KEYWORDS)
    if leaked:
        raise ValueError(
            f"patient-identifying keys must not appear in MRI metadata: {leaked}"
        )


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
def _as_float(value: Any) -> float | None:
    """Coerce a header value to float, or ``None``.

    DICOM numeric strings arrive as ``DSfloat``, ``IS``, ``bytes``, plain ``str``, or
    a one-element multi-value. Anything that does not convert is ``None`` — never a
    default number.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        return _as_float(value[0]) if len(value) == 1 else None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None    # reject NaN/inf


def _as_int(value: Any) -> int | None:
    out = _as_float(value)
    return int(round(out)) if out is not None else None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    text = str(value).strip()
    return text or None


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Normalise a DICOM multi-valued string (``ImageType``, ``ScanningSequence``).

    Upper-cased and backslash-split, because a single-valued DICOM string and a
    multi-valued one arrive as different Python types for the same clinical concept.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items: Iterable[Any] = value
    else:
        items = str(value).split("\\")
    return tuple(s.strip().upper() for s in (_as_str(v) for v in items) if s)


def _as_float_tuple(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split("\\")
    out = [_as_float(v) for v in items]
    if len(out) != size or any(v is None for v in out):
        return None
    return tuple(float(v) for v in out)                      # type: ignore[arg-type]


_VENDOR_PATTERNS: tuple[tuple[str, ScannerVendor], ...] = (
    ("SIEMENS", ScannerVendor.SIEMENS),
    ("GE MEDICAL", ScannerVendor.GE),
    ("GE HEALTHCARE", ScannerVendor.GE),
    ("GENERAL ELECTRIC", ScannerVendor.GE),
    ("PHILIPS", ScannerVendor.PHILIPS),
    ("CANON", ScannerVendor.CANON),
    ("TOSHIBA", ScannerVendor.CANON),          # Canon acquired Toshiba Medical
    ("HITACHI", ScannerVendor.HITACHI),
    ("FUJIFILM", ScannerVendor.HITACHI),       # Fujifilm acquired Hitachi's MR line
    ("UIH", ScannerVendor.UNITED_IMAGING),
    ("UNITED IMAGING", ScannerVendor.UNITED_IMAGING),
)


def normalise_vendor(manufacturer: str | None) -> ScannerVendor:
    """Map a raw ``Manufacturer`` string onto :class:`ScannerVendor`.

    Bare ``GE`` is matched exactly rather than by substring: "GE" appears inside
    plenty of unrelated manufacturer strings ("IMAGE**GE**N"), and a false vendor
    match would send a sequence heuristic down the wrong vendor branch.
    """
    if not manufacturer:
        return ScannerVendor.UNKNOWN
    text = manufacturer.strip().upper()
    for needle, vendor in _VENDOR_PATTERNS:
        if needle in text:
            return vendor
    if re.fullmatch(r"GE\.?", text):
        return ScannerVendor.GE
    return ScannerVendor.OTHER


# --------------------------------------------------------------------------- #
# Typed models
# --------------------------------------------------------------------------- #
class _Model(BaseModel):
    """Shared configuration: reject unknown fields so a typo in a reader fails loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SeriesIdentifiers(_Model):
    """Study/series identity. UIDs only — no accession number, no dates, no station."""

    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    frame_of_reference_uid: str | None = Field(
        None, description="Shared by series acquired in the same physical frame; the "
                          "hint that two series are already co-registered.")
    sop_class_uid: str | None = None
    series_number: int | None = None
    modality: str | None = Field(None, description="DICOM Modality, expected 'MR'.")
    study_description: str | None = Field(
        None, description="Protocol free-text. Corroborating evidence for sequence "
                          "detection, never primary evidence.")
    series_description: str | None = None
    protocol_name: str | None = None
    body_part_examined: str | None = None
    source_name: str | None = Field(
        None, description="Filename or directory the series was read from. Diagnostic "
                          "only — never used alone to classify a sequence.")


class ScannerInfo(_Model):
    """Acquisition hardware. Vendor-normalised, with the raw string preserved."""

    vendor: ScannerVendor = ScannerVendor.UNKNOWN
    manufacturer: str | None = None
    model_name: str | None = None
    field_strength_tesla: float | None = None
    software_versions: str | None = None
    receive_coil: str | None = None
    imaging_frequency_mhz: float | None = None


class GeometryMetadata(_Model):
    """Sampling geometry, as recorded *and* as derived from the built affine.

    Both are kept. The recorded tags are what the scanner wrote; the derived values
    are what the volume actually has. They normally agree, and when they do not, that
    disagreement is itself the finding — a ``SliceThickness`` of 1 mm alongside a
    derived 3 mm slice spacing means a gapped acquisition, which changes what a 3D
    model may conclude.
    """

    rows: int | None = None
    columns: int | None = None
    number_of_slices: int | None = None

    pixel_spacing_mm: tuple[float, float] | None = Field(
        None, description="DICOM PixelSpacing: (between-row, between-column).")
    slice_thickness_mm: float | None = None
    spacing_between_slices_mm: float | None = None
    acquisition_matrix: tuple[int, ...] | None = Field(
        None, description="DICOM AcquisitionMatrix: frequency rows/cols, phase "
                          "rows/cols — the *acquired* matrix before interpolation.")
    reconstruction_diameter_mm: float | None = None
    patient_position: str | None = Field(
        None, description="Table positioning code (HFS, FFS...). Setup, not identity.")
    image_orientation_patient: tuple[float, ...] | None = None
    image_position_first: tuple[float, ...] | None = None
    image_position_last: tuple[float, ...] | None = None

    #: ``False`` when the source recorded no world orientation — a NIfTI with neither
    #: sform nor qform, an NRRD with no space directions, a DICOM with no
    #: ``ImageOrientationPatient``. The affine then carries spacing but its
    #: left-right direction is an assumption, which makes any lateralised finding
    #: unsafe. A typed flag rather than a warning string because the quality
    #: inspector has to branch on it.
    world_orientation_known: bool = True

    # -- derived from the affine ------------------------------------------- #
    voxel_size_mm: tuple[float, float, float] | None = None
    orientation: str | None = Field(
        None, description="Axis codes of the volume as stored, e.g. 'LAS'.")
    plane: AnatomicalPlane = AnatomicalPlane.UNKNOWN
    obliquity_deg: float | None = None
    field_of_view_mm: tuple[float, float, float] | None = None
    anisotropy: float | None = None


class AcquisitionParameters(_Model):
    """Pulse-sequence parameters — the primary evidence for sequence identification."""

    scanning_sequence: tuple[str, ...] = Field(
        (), description="DICOM ScanningSequence: SE, IR, GR, EP, RM.")
    sequence_variant: tuple[str, ...] = Field(
        (), description="DICOM SequenceVariant: SK, MTC, SS, TRSS, SP, MP, OSP.")
    scan_options: tuple[str, ...] = ()
    sequence_name: str | None = None
    mr_acquisition_type: MRAcquisitionType = MRAcquisitionType.UNKNOWN
    image_type: tuple[str, ...] = Field(
        (), description="DICOM ImageType. Carries ORIGINAL/DERIVED, PRIMARY/SECONDARY, "
                        "and vendor markers such as ADC, TRACEW, MOSAIC, SWI.")

    echo_time_ms: float | None = None
    repetition_time_ms: float | None = None
    inversion_time_ms: float | None = None
    flip_angle_deg: float | None = None
    echo_train_length: int | None = None
    echo_number: int | None = None
    number_of_averages: float | None = None
    pixel_bandwidth_hz: float | None = None
    phase_encoding_direction: str | None = Field(
        None, description="'ROW' or 'COL'. Needed by the ghosting check: motion "
                          "ghosts replicate along the phase-encode axis.")
    number_of_temporal_positions: int | None = None

    contrast_agent: str | None = None
    contrast_administered: bool = Field(
        False, description="True when a contrast agent tag is present and non-empty.")

    diffusion_b_value: float | None = None
    diffusion_directionality: str | None = None

    @property
    def available(self) -> bool:
        """True when enough parameters exist to attempt metadata-first classification.

        The bar is deliberately low — TR and TE alone already separate most families —
        because the alternative to "attempt it" is description-only guessing.
        """
        return any(v is not None for v in (self.echo_time_ms, self.repetition_time_ms,
                                           self.inversion_time_ms)) or bool(
            self.scanning_sequence)


class MRIMetadata(_Model):
    """Complete patient-independent metadata for one MR series."""

    source_format: FileFormat = FileFormat.UNKNOWN
    identifiers: SeriesIdentifiers = Field(default_factory=SeriesIdentifiers)
    scanner: ScannerInfo = Field(default_factory=ScannerInfo)
    geometry: GeometryMetadata = Field(default_factory=GeometryMetadata)
    acquisition: AcquisitionParameters = Field(default_factory=AcquisitionParameters)
    #: Everything the extractor wanted and did not find, in plain language. Surfaced
    #: in the foundation output so a downstream failure can be traced to missing
    #: input rather than to a bug.
    extraction_warnings: tuple[str, ...] = ()

    @property
    def has_acquisition_parameters(self) -> bool:
        return self.acquisition.available

    def describe(self) -> str:
        """One-line human summary, for logs and reports."""
        parts = [self.identifiers.series_description or
                 self.identifiers.source_name or "series"]
        if self.geometry.voxel_size_mm:
            parts.append("x".join(f"{v:.2f}" for v in self.geometry.voxel_size_mm) + "mm")
        if self.geometry.orientation:
            parts.append(self.geometry.orientation)
        if self.scanner.field_strength_tesla:
            parts.append(f"{self.scanner.field_strength_tesla:g}T")
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
class MetadataExtractor:
    """Turn a reader's raw header mapping into :class:`MRIMetadata`.

    One class handles all three formats because the *output* is format-independent by
    design — that is the point of a foundation layer. What differs is how much of the
    model each format can fill, and the gaps are recorded as warnings rather than
    papered over. A NIfTI has no acquisition parameters at all; saying so explicitly
    is what lets the sequence detector cap its own confidence honestly.
    """

    def extract(
        self,
        header: Mapping[str, Any],
        *,
        source_format: FileFormat,
        geometry: VoxelGeometry | None = None,
        slice_count: int | None = None,
        source_name: str | None = None,
    ) -> MRIMetadata:
        assert_patient_independent(header)
        warnings: list[str] = []

        if source_format is FileFormat.DICOM:
            identifiers = self._dicom_identifiers(header, source_name)
            scanner = self._dicom_scanner(header)
            acquisition = self._dicom_acquisition(header)
            recorded = self._dicom_geometry_tags(header)
        else:
            identifiers = self._plain_identifiers(header, source_name)
            scanner = ScannerInfo()
            acquisition = self._plain_acquisition(header)
            recorded = {}
            warnings.append(
                f"{source_format.value.upper()} carries no DICOM acquisition header; "
                "scanner and pulse-sequence parameters are unavailable and sequence "
                "identification must fall back to description evidence"
            )

        geometry_model = self._geometry(recorded, geometry, slice_count,
                                        self._orientation_known(header))
        warnings.extend(self._geometry_warnings(geometry_model, geometry))
        if not geometry_model.world_orientation_known:
            warnings.append(
                "the source records no world orientation (no sform/qform, no space "
                "directions, or no ImageOrientationPatient); voxel spacing is known "
                "but left-right cannot be established from this study")
        if source_format is FileFormat.DICOM and not acquisition.available:
            warnings.append(
                "DICOM header contains no usable TR/TE/TI or ScanningSequence; "
                "sequence identification is degraded"
            )

        return MRIMetadata(
            source_format=source_format,
            identifiers=identifiers,
            scanner=scanner,
            geometry=geometry_model,
            acquisition=acquisition,
            extraction_warnings=tuple(warnings),
        )

    # -- DICOM -------------------------------------------------------------- #
    @staticmethod
    def _dicom_identifiers(h: Mapping[str, Any],
                           source_name: str | None) -> SeriesIdentifiers:
        return SeriesIdentifiers(
            study_instance_uid=_as_str(h.get("StudyInstanceUID")),
            series_instance_uid=_as_str(h.get("SeriesInstanceUID")),
            frame_of_reference_uid=_as_str(h.get("FrameOfReferenceUID")),
            sop_class_uid=_as_str(h.get("SOPClassUID")),
            series_number=_as_int(h.get("SeriesNumber")),
            modality=_as_str(h.get("Modality")),
            study_description=_as_str(h.get("StudyDescription")),
            series_description=_as_str(h.get("SeriesDescription")),
            protocol_name=_as_str(h.get("ProtocolName")),
            body_part_examined=_as_str(h.get("BodyPartExamined")),
            source_name=source_name,
        )

    @staticmethod
    def _dicom_scanner(h: Mapping[str, Any]) -> ScannerInfo:
        manufacturer = _as_str(h.get("Manufacturer"))
        field_strength = _as_float(h.get("MagneticFieldStrength"))
        # Some vendors write the field strength in gauss (30000 G = 3 T). Values that
        # large are unambiguously not tesla, so converting is safe and the alternative
        # is a nonsensical "30000 T" in a report.
        if field_strength is not None and field_strength > 100:
            field_strength = field_strength / 10_000.0
        return ScannerInfo(
            vendor=normalise_vendor(manufacturer),
            manufacturer=manufacturer,
            model_name=_as_str(h.get("ManufacturerModelName")),
            field_strength_tesla=field_strength,
            software_versions=_as_str(h.get("SoftwareVersions")),
            receive_coil=_as_str(h.get("ReceiveCoilName")),
            imaging_frequency_mhz=_as_float(h.get("ImagingFrequency")),
        )

    @staticmethod
    def _dicom_acquisition(h: Mapping[str, Any]) -> AcquisitionParameters:
        acquisition_type = _as_str(h.get("MRAcquisitionType"))
        try:
            mr_type = MRAcquisitionType(acquisition_type) if acquisition_type \
                else MRAcquisitionType.UNKNOWN
        except ValueError:
            mr_type = MRAcquisitionType.UNKNOWN

        agent = _as_str(h.get("ContrastBolusAgent"))
        # Scanners write "NONE", "-", or a single space when nothing was given.
        administered = bool(agent) and agent.upper() not in {"NONE", "NO", "N", "-"}

        return AcquisitionParameters(
            scanning_sequence=_as_str_tuple(h.get("ScanningSequence")),
            sequence_variant=_as_str_tuple(h.get("SequenceVariant")),
            scan_options=_as_str_tuple(h.get("ScanOptions")),
            sequence_name=_as_str(h.get("SequenceName")),
            mr_acquisition_type=mr_type,
            image_type=_as_str_tuple(h.get("ImageType")),
            echo_time_ms=_as_float(h.get("EchoTime")),
            repetition_time_ms=_as_float(h.get("RepetitionTime")),
            inversion_time_ms=_as_float(h.get("InversionTime")),
            flip_angle_deg=_as_float(h.get("FlipAngle")),
            echo_train_length=_as_int(h.get("EchoTrainLength")),
            echo_number=_as_int(h.get("EchoNumbers")),
            number_of_averages=_as_float(h.get("NumberOfAverages")),
            pixel_bandwidth_hz=_as_float(h.get("PixelBandwidth")),
            phase_encoding_direction=_as_str(h.get("InPlanePhaseEncodingDirection")),
            number_of_temporal_positions=_as_int(h.get("NumberOfTemporalPositions")),
            contrast_agent=agent if administered else None,
            contrast_administered=administered,
            diffusion_b_value=_as_float(h.get("DiffusionBValue")),
            diffusion_directionality=_as_str(h.get("DiffusionDirectionality")),
        )

    @staticmethod
    def _dicom_geometry_tags(h: Mapping[str, Any]) -> dict[str, Any]:
        matrix = h.get("AcquisitionMatrix")
        matrix_values: tuple[int, ...] | None = None
        if matrix is not None:
            items = matrix if isinstance(matrix, (list, tuple)) else str(matrix).split("\\")
            parsed = [_as_int(v) for v in items]
            if parsed and all(v is not None for v in parsed):
                matrix_values = tuple(int(v) for v in parsed)   # type: ignore[arg-type]
        return {
            "rows": _as_int(h.get("Rows")),
            "columns": _as_int(h.get("Columns")),
            "pixel_spacing_mm": _as_float_tuple(h.get("PixelSpacing"), 2),
            "slice_thickness_mm": _as_float(h.get("SliceThickness")),
            "spacing_between_slices_mm": _as_float(h.get("SpacingBetweenSlices")),
            "acquisition_matrix": matrix_values,
            "reconstruction_diameter_mm": _as_float(h.get("ReconstructionDiameter")),
            "patient_position": _as_str(h.get("PatientPosition")),
            "image_orientation_patient": _as_float_tuple(
                h.get("ImageOrientationPatient"), 6),
            "image_position_first": _as_float_tuple(h.get("ImagePositionPatient"), 3),
            "image_position_last": _as_float_tuple(h.get("_ImagePositionPatientLast"), 3),
        }

    # -- NIfTI / NRRD ------------------------------------------------------- #
    @staticmethod
    def _plain_identifiers(h: Mapping[str, Any],
                           source_name: str | None) -> SeriesIdentifiers:
        # NIfTI's 80-byte ``descrip`` and NRRD's free-form fields are the only text a
        # converter can leave behind; treated as a series description, with the same
        # low evidential weight.
        return SeriesIdentifiers(
            series_description=_as_str(h.get("descrip") or h.get("content")),
            modality=_as_str(h.get("modality")),
            source_name=source_name,
        )

    @staticmethod
    def _plain_acquisition(h: Mapping[str, Any]) -> AcquisitionParameters:
        # A few converters preserve the b-value in an NRRD key/value field; if it is
        # there it is real acquisition evidence and worth keeping.
        return AcquisitionParameters(
            diffusion_b_value=_as_float(
                h.get("DWMRI_b-value") or h.get("dwmri_b-value")),
        )

    # -- geometry ----------------------------------------------------------- #
    @staticmethod
    def _orientation_known(header: Mapping[str, Any]) -> bool:
        """Whether the source actually recorded a world orientation.

        Readers set these markers when they had to assume one. Reading a flag beats
        pattern-matching on warning text: a reworded warning would silently disable
        the quality check that depends on this.
        """
        if header.get("affine_source") in ("pixdim_fallback", "assumed"):
            return False
        if header.get("_OrientationAssumed"):
            return False
        return True

    @staticmethod
    def _geometry(recorded: Mapping[str, Any], geometry: VoxelGeometry | None,
                  slice_count: int | None,
                  orientation_known: bool = True) -> GeometryMetadata:
        derived: dict[str, Any] = {}
        if geometry is not None:
            derived = {
                "voxel_size_mm": tuple(round(v, 6) for v in geometry.spacing),
                "orientation": geometry.orientation,
                "plane": geometry.plane,
                "obliquity_deg": round(geometry.obliquity_deg, 4),
                "field_of_view_mm": tuple(round(v, 4) for v in geometry.field_of_view_mm),
                "anisotropy": round(geometry.anisotropy, 4),
                "number_of_slices": int(geometry.shape[2]),
            }
            derived.setdefault("rows", int(geometry.shape[1]))
            derived.setdefault("columns", int(geometry.shape[0]))
        if slice_count is not None:
            derived["number_of_slices"] = int(slice_count)

        merged = {k: v for k, v in recorded.items() if v is not None}
        for key, value in derived.items():
            # Recorded matrix dimensions win over derived ones: the header states what
            # the scanner wrote, and the derived pair only restates the array shape.
            # Everything else is better measured from the affine we actually built.
            if key in ("rows", "columns") and merged.get(key) is not None:
                continue
            merged[key] = value
        merged["world_orientation_known"] = orientation_known
        return GeometryMetadata(**merged)

    @staticmethod
    def _geometry_warnings(model: GeometryMetadata,
                           geometry: VoxelGeometry | None) -> list[str]:
        """Flag disagreements between recorded and derived geometry.

        The recorded/derived slice-thickness comparison is the one that matters: a
        gap between them means the acquisition has inter-slice gaps, and a volumetric
        measurement made without knowing that is wrong by exactly that ratio.
        """
        out: list[str] = []
        if geometry is None:
            out.append("no affine was available; derived geometry is absent")
            return out
        recorded_slice = model.spacing_between_slices_mm or model.slice_thickness_mm
        derived_slice = model.voxel_size_mm[2] if model.voxel_size_mm else None
        if recorded_slice and derived_slice and recorded_slice > 0:
            ratio = derived_slice / recorded_slice
            if not 0.9 <= ratio <= 1.1:
                out.append(
                    f"recorded slice spacing {recorded_slice:.3f} mm disagrees with the "
                    f"spacing derived from slice positions {derived_slice:.3f} mm "
                    f"(ratio {ratio:.2f}); the acquisition is likely gapped or "
                    "irregularly sampled"
                )
        if model.pixel_spacing_mm and model.voxel_size_mm:
            recorded_ip = sorted(model.pixel_spacing_mm)
            derived_ip = sorted(model.voxel_size_mm[:2])
            if any(abs(a - b) > 0.01 for a, b in zip(recorded_ip, derived_ip)):
                out.append(
                    f"recorded PixelSpacing {tuple(model.pixel_spacing_mm)} disagrees "
                    f"with in-plane spacing derived from the affine "
                    f"{tuple(round(v, 3) for v in model.voxel_size_mm[:2])}"
                )
        return out


def dicom_header_subset(dataset: Any, *, extra: Mapping[str, Any] | None = None
                        ) -> dict[str, Any]:
    """Copy the allowlisted keywords out of a pydicom dataset into a plain dict.

    Plain Python types only: the metadata engine, the sequence detector, and the
    quality inspector must not depend on pydicom being importable, and a pydicom
    value object holds a reference to the whole dataset — including the pixel buffer
    and every tag the allowlist exists to exclude.
    """
    header: dict[str, Any] = {}
    for keyword in DICOM_KEYWORDS:
        if not hasattr(dataset, keyword):
            continue
        value = getattr(dataset, keyword)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
            header[keyword] = [_plain(v) for v in value]
        else:
            header[keyword] = _plain(value)
    if extra:
        header.update(dict(extra))
    assert_patient_independent(header)
    return header


def _plain(value: Any) -> Any:
    """Reduce a pydicom value type to str/int/float."""
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    try:
        return float(value) if "." in str(value) else int(value)
    except (TypeError, ValueError):
        return str(value)


def sequence_evidence_text(metadata: MRIMetadata) -> str:
    """Concatenated free-text fields, upper-cased, for description-based matching.

    Assembled in one place so the sequence detector cannot accidentally weigh a
    filename as if it were a protocol name — both land in this string and both are
    treated as the same low-weight channel.
    """
    ident = metadata.identifiers
    parts: Sequence[str | None] = (
        ident.series_description, ident.protocol_name, ident.study_description,
        metadata.acquisition.sequence_name, ident.source_name,
        " ".join(metadata.acquisition.image_type),
    )
    return " ".join(p for p in parts if p).upper()
