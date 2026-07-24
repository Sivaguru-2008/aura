"""DICOM series reader: a directory of files to sorted, validated volumes.

This is where most real MRI data arrives and where most of the ways a study can be
subtly wrong live. The reader's job is to turn files into volumes *and* to notice
everything that would otherwise be lost in the stacking.

Grouping
--------
``SeriesInstanceUID`` alone is not a volume. Real exports routinely put several
distinct volumes under one series UID, and stacking them produces an array with
interleaved contrasts that looks entirely plausible. The composite key adds:

* **echo number** — a multi-echo acquisition stores every echo under one series;
  stacking them interleaves two different contrasts.
* **image orientation** — a three-plane localiser is one series containing axial,
  coronal, and sagittal slices.
* **matrix size** — a series that also holds a derived, differently-sized image.
* **magnitude / phase / real / imaginary** — the same acquisition reconstructed four
  ways; phase images have completely different intensity semantics.

This mirrors what ``dcm2niix`` splits on, for the same reasons.

Sorting
-------
By the projection of ``ImagePositionPatient`` onto the slice normal, not by
``InstanceNumber``. Instance numbers are reliable until they are not — reordered
exports, interleaved acquisitions, PACS renumbering — and a position projection is
geometry, which cannot be renumbered. ``InstanceNumber`` is the documented fallback
when positions are absent, and ties are broken with it.

Missing slices
--------------
Detected from the sorted positions: a gap that is a clean multiple of the series'
own median spacing is a dropped slice, and the multiple says how many. This is the
only place in the pipeline where that evidence still exists — once the array is
stacked, a missing slice and a thicker slice are the same thing.

Intensity
---------
``RescaleSlope``/``RescaleIntercept`` are applied because they are part of the stored
value's definition. ``MONOCHROME1`` is **not** inverted: inversion is a display
transform, and inverting here would corrupt every quantitative measurement made
downstream. The photometric interpretation is recorded instead, and the quality
inspector warns on it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.foundation.mri.config import LoaderConfig
from backend.foundation.mri.errors import CorruptStudy, StudyValidationError
from backend.foundation.mri.geometry import VoxelGeometry, affine_from_dicom
from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity, SliceIssue
from backend.foundation.mri.metadata import dicom_header_subset
from backend.foundation.mri.types import FileFormat

log = get_logger("foundation.mri.io.dicom")

#: ImageType markers that identify a distinct reconstruction of the same acquisition.
_RECONSTRUCTION_MARKERS = ("PHASE", "REAL", "IMAGINARY", "MAGNITUDE")


@dataclass
class _SliceRef:
    """One DICOM instance's header plus everything needed to place it in a volume."""

    path: Path
    header: dict[str, Any]
    position: np.ndarray | None
    orientation: np.ndarray | None
    instance_number: int
    projection: float


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else [value]
    try:
        return [float(v) for v in items]
    except (TypeError, ValueError):
        return None


class DicomSeriesReader:
    """Reads every MR series present among a set of DICOM files."""

    file_format = FileFormat.DICOM

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self._config = config or LoaderConfig()

    # ------------------------------------------------------------------ #
    # Discovery support
    # ------------------------------------------------------------------ #
    def can_read(self, path: Path) -> bool:
        """Cheap DICOM test: the ``DICM`` preamble marker at offset 128.

        Content-sniffed rather than suffix-matched, because DICOM is very often
        exported with no extension at all. Files without the preamble are probed with
        pydicom's ``force`` reader only when their suffix claims DICOM, which keeps a
        directory scan from paying a parse attempt per unrelated file.
        """
        try:
            with open(path, "rb") as fh:
                fh.seek(128)
                if fh.read(4) == b"DICM":
                    return True
        except OSError:
            return False
        if path.suffix.lower() not in (".dcm", ".dicom", ".ima"):
            return False
        return self._probe_headerless(path)

    @staticmethod
    def _probe_headerless(path: Path) -> bool:
        """Last-resort parse for DICOM written without the 128-byte preamble."""
        try:
            import pydicom

            pydicom.dcmread(str(path), stop_before_pixels=True, force=True,
                            specific_tags=["SOPClassUID"])
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def read(self, paths: Sequence[Path], *,
             issues: list[dict[str, Any]] | None = None) -> list[RawSeries]:
        """Group ``paths`` into series and decode each one."""
        try:
            import pydicom                                        # noqa: F401
        except ImportError as exc:                    # pragma: no cover - env specific
            raise CorruptStudy(
                "DICOM support requires pydicom, which is not installed in this "
                "deployment") from exc

        groups, unreadable = self._group(paths)
        if not groups:
            raise CorruptStudy(
                "no DICOM instance in the study could be read",
                detail={"files_examined": len(paths),
                        "failures": [i.to_dict() for i in unreadable[:10]]})

        series: list[RawSeries] = []
        for key, refs in groups.items():
            try:
                series.append(self._build_series(key, refs, unreadable))
            except StudyValidationError as exc:
                log.warning("skipping series that cannot form a volume",
                            extra={"context": {"series": key, "reason": exc.reason}})
                if issues is not None:
                    issues.append({"series_key": key, "error": exc.code,
                                   "reason": exc.reason, **({"detail": exc.detail}
                                                            if exc.detail else {})})
        if not series:
            raise StudyValidationError(
                "DICOM instances were read but none formed a usable volume",
                detail={"series_examined": len(groups)})
        return series

    # ------------------------------------------------------------------ #
    # Grouping
    # ------------------------------------------------------------------ #
    def _group(self, paths: Sequence[Path]
               ) -> tuple[dict[str, list[_SliceRef]], list[SliceIssue]]:
        import pydicom
        from backend.foundation.mri.errors import StudyValidationError

        groups: dict[str, list[_SliceRef]] = defaultdict(list)
        unreadable: list[SliceIssue] = []
        patient_ids = set()
        patient_names = set()

        for path in paths:
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=True,
                                          force=True)
            except Exception as exc:
                unreadable.append(SliceIssue(path.name, f"header unreadable: "
                                                        f"{type(exc).__name__}"))
                continue

            modality = str(getattr(dataset, "Modality", "") or "").upper()
            if modality and modality != "MR":
                # Not an error — a study folder legitimately contains a dose report or
                # a scanned requisition. It is simply not ours.
                log.debug("ignoring non-MR instance",
                          extra={"context": {"file": path.name, "modality": modality}})
                continue

            pid = str(getattr(dataset, "PatientID", "") or "").strip()
            pname = str(getattr(dataset, "PatientName", "") or "").strip()
            if pid:
                patient_ids.add(pid)
            if pname:
                patient_names.add(pname)

            header = dicom_header_subset(dataset)
            orientation = _float_list(header.get("ImageOrientationPatient"))
            position = _float_list(header.get("ImagePositionPatient"))
            ref = _SliceRef(
                path=path,
                header=header,
                position=np.asarray(position, dtype=float)
                if position and len(position) == 3 else None,
                orientation=np.asarray(orientation, dtype=float)
                if orientation and len(orientation) == 6 else None,
                instance_number=int(header.get("InstanceNumber") or 0),
                projection=0.0,
            )
            groups[self._series_key(header, ref)].append(ref)

        if len(patient_ids) > 1 or len(patient_names) > 1:
            raise StudyValidationError(
                "mixed patients detected in the uploaded study",
                detail={
                    "patient_ids": list(patient_ids),
                    "patient_names": [str(name) for name in patient_names],
                }
            )

        return dict(groups), unreadable

    @staticmethod
    def _series_key(header: dict[str, Any], ref: _SliceRef) -> str:
        """Composite key identifying one *volume*, not merely one series.

        See the module docstring for why the series UID alone is not enough.
        """
        parts: list[str] = [str(header.get("SeriesInstanceUID") or "no-series-uid")]
        echo = header.get("EchoNumbers")
        if echo is not None:
            parts.append(f"echo{echo}")
        if ref.orientation is not None:
            parts.append("iop" + ",".join(f"{v:.4f}" for v in ref.orientation))
        rows, columns = header.get("Rows"), header.get("Columns")
        if rows and columns:
            parts.append(f"{rows}x{columns}")
        image_type = [str(v).upper() for v in (header.get("ImageType") or [])]
        marker = next((m for m in _RECONSTRUCTION_MARKERS if m in image_type), None)
        if marker:
            parts.append(marker)
        return "|".join(parts)

    # ------------------------------------------------------------------ #
    # Series assembly
    # ------------------------------------------------------------------ #
    def _build_series(self, key: str, refs: list[_SliceRef],
                      study_failures: list[SliceIssue]) -> RawSeries:
        first_header = refs[0].header
        frames = int(first_header.get("NumberOfFrames") or 1)
        if len(refs) == 1 and frames > 1:
            return self._build_multiframe(key, refs[0])

        warnings: list[str] = []
        geometry_consistent = self._check_geometry_consistency(refs, warnings)
        ordered = self._sort(refs, warnings)
        ordered, duplicates = self._drop_duplicates(ordered, warnings)

        if len(ordered) < self._config.min_slices_for_volume:
            raise StudyValidationError(
                f"a volume needs at least {self._config.min_slices_for_volume} slices; "
                f"this series has {len(ordered)}",
                detail={"slices": len(ordered)})

        spacing_stats = self._analyse_spacing(ordered, warnings)
        voxels, corrupt = self._stack(ordered, warnings)

        if len(voxels) < self._config.min_slices_for_volume:
            raise StudyValidationError(
                "too few slices in this series could be decoded to form a volume",
                detail={"decoded": len(voxels), "expected": len(ordered)})
        corrupt_fraction = len(corrupt) / max(1, len(ordered))
        if corrupt and not self._config.tolerate_corrupt_slices:
            raise StudyValidationError(
                f"{len(corrupt)} slice(s) in this series could not be decoded",
                detail={"corrupt": [c.to_dict() for c in corrupt]})
        if corrupt_fraction > self._config.max_corrupt_fraction:
            raise StudyValidationError(
                f"{corrupt_fraction:.0%} of this series could not be decoded, above "
                f"the {self._config.max_corrupt_fraction:.0%} tolerance",
                detail={"corrupt": [c.to_dict() for c in corrupt[:10]]})

        kept = [ref for ref in ordered if ref.path.name not in
                {c.name for c in corrupt}]
        geometry = self._geometry(kept, voxels, spacing_stats, warnings)
        # (rows, columns) per slice, stacked along k -> transpose to (i, j, k).
        array = np.transpose(np.stack(voxels, axis=0), (2, 1, 0))

        # Taken from the first slice in *geometric* order, not the first file read:
        # its ImagePositionPatient is the volume's origin, and pairing it with the
        # last slice's position is what makes the recorded geometry describe the
        # array that was actually built.
        header = dict(kept[0].header) if kept else dict(first_header)
        if kept and kept[-1].position is not None:
            header["_ImagePositionPatientLast"] = [float(v) for v in kept[-1].position]
        header["_SliceCount"] = len(kept)

        integrity = SeriesIntegrity(
            files_found=len(refs),
            slices_loaded=len(kept),
            corrupt_files=tuple(corrupt),
            duplicate_positions=duplicates,
            missing_slices_estimated=spacing_stats["missing"],
            irregular_gaps_mm=tuple(spacing_stats["irregular"]),
            median_slice_spacing_mm=spacing_stats["median"],
            spacing_consistent=not spacing_stats["irregular"],
            geometry_consistent=geometry_consistent,
            warnings=tuple(warnings),
        )
        return RawSeries(
            series_key=key,
            source_format=FileFormat.DICOM,
            voxels=array,
            geometry=geometry,
            header=header,
            integrity=integrity,
            source_name=str(first_header.get("SeriesDescription")
                            or first_header.get("ProtocolName") or key.split("|")[0]),
            contributing_files=tuple(ref.path.name for ref in kept),
        )

    # -- consistency --------------------------------------------------------- #
    @staticmethod
    def _check_geometry_consistency(refs: list[_SliceRef],
                                    warnings: list[str]) -> bool:
        """Verify every slice shares one matrix size and one orientation."""
        consistent = True
        sizes = {(r.header.get("Rows"), r.header.get("Columns")) for r in refs}
        if len(sizes) > 1:
            consistent = False
            warnings.append(
                f"slices in this series have different matrix sizes {sorted(sizes)}; "
                "they do not describe one volume")
        spacings = {tuple(_float_list(r.header.get("PixelSpacing")) or ())
                    for r in refs}
        spacings.discard(())
        if len(spacings) > 1:
            consistent = False
            warnings.append(
                f"slices in this series have different pixel spacings {sorted(spacings)}")
        orientations = {tuple(round(v, 4) for v in r.orientation)
                        for r in refs if r.orientation is not None}
        if len(orientations) > 1:
            consistent = False
            warnings.append(
                f"slices in this series have {len(orientations)} distinct orientations; "
                "the series mixes acquisition planes")
        return consistent

    # -- ordering ------------------------------------------------------------ #
    @staticmethod
    def _sort(refs: list[_SliceRef], warnings: list[str]) -> list[_SliceRef]:
        """Order slices along the slice normal; fall back to instance number."""
        oriented = next((r for r in refs if r.orientation is not None), None)
        positioned = [r for r in refs if r.position is not None]

        if oriented is None or len(positioned) < len(refs):
            warnings.append(
                "not every slice carries ImagePositionPatient/ImageOrientationPatient; "
                "slices were ordered by InstanceNumber, which is less reliable than "
                "geometry")
            return sorted(refs, key=lambda r: r.instance_number)

        normal = np.cross(oriented.orientation[:3], oriented.orientation[3:])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            warnings.append("ImageOrientationPatient is degenerate; slices were "
                            "ordered by InstanceNumber")
            return sorted(refs, key=lambda r: r.instance_number)
        normal = normal / norm
        for ref in refs:
            ref.projection = float(np.dot(ref.position, normal))

        ordered = sorted(refs, key=lambda r: (r.projection, r.instance_number))
        instance_order = [r.instance_number for r in ordered]
        if instance_order == sorted(instance_order, reverse=True) \
                and len(set(instance_order)) > 1:
            # Geometric order is the reverse of instance order. Normal for many
            # protocols (feet-first, or descending acquisition); worth recording
            # because it is also what a mis-sorted export looks like.
            warnings.append(
                "slice positions run opposite to InstanceNumber; geometric order was "
                "used")
        return ordered

    def _drop_duplicates(self, ordered: list[_SliceRef],
                         warnings: list[str]) -> tuple[list[_SliceRef], int]:
        """Remove slices that occupy a position already taken."""
        tolerance = self._config.duplicate_position_tolerance_mm
        kept: list[_SliceRef] = []
        duplicates = 0
        for ref in ordered:
            if kept and ref.position is not None and kept[-1].position is not None \
                    and abs(ref.projection - kept[-1].projection) < tolerance:
                duplicates += 1
                continue
            kept.append(ref)
        if duplicates:
            warnings.append(
                f"{duplicates} slice(s) duplicated an existing slice position and were "
                "dropped; if this series is multi-echo or multi-reconstruction, the "
                "split key did not separate it")
        return kept, duplicates

    # -- spacing ------------------------------------------------------------- #
    def _analyse_spacing(self, ordered: list[_SliceRef],
                         warnings: list[str]) -> dict[str, Any]:
        """Measure inter-slice distances and infer how many slices are absent."""
        projections = [r.projection for r in ordered if r.position is not None]
        if len(projections) < 2:
            recorded = _float_list(ordered[0].header.get("SpacingBetweenSlices")) \
                or _float_list(ordered[0].header.get("SliceThickness"))
            return {"median": float(recorded[0]) if recorded else None,
                    "missing": 0, "irregular": []}

        gaps = np.diff(np.asarray(projections, dtype=float))
        median = float(np.median(np.abs(gaps)))
        if median <= 0:
            warnings.append("consecutive slices share the same position; the slice "
                            "spacing could not be measured")
            return {"median": None, "missing": 0, "irregular": []}

        tolerance = self._config.slice_gap_tolerance
        irregular: list[float] = []
        missing = 0
        for gap in np.abs(gaps):
            ratio = gap / median
            if abs(ratio - 1.0) <= tolerance:
                continue
            irregular.append(float(gap))
            if ratio > 1.0:
                missing += max(0, int(round(ratio)) - 1)

        if missing:
            warnings.append(
                f"{missing} slice(s) appear to be absent: {len(irregular)} inter-slice "
                f"gap(s) are whole multiples of the series median spacing "
                f"({median:.3f} mm)")
        elif irregular:
            warnings.append(
                f"{len(irregular)} inter-slice gap(s) deviate from the median spacing "
                f"({median:.3f} mm) by more than {tolerance:.0%}; the acquisition is "
                "irregularly sampled")
        return {"median": median, "missing": missing, "irregular": irregular}

    # -- pixels -------------------------------------------------------------- #
    @staticmethod
    def _stack(ordered: list[_SliceRef],
               warnings: list[str]) -> tuple[list[np.ndarray], list[SliceIssue]]:
        """Decode each slice's pixel data, applying the declared rescale."""
        import pydicom

        planes: list[np.ndarray] = []
        corrupt: list[SliceIssue] = []
        photometric: set[str] = set()
        reference_shape: tuple[int, ...] | None = None

        for ref in ordered:
            try:
                dataset = pydicom.dcmread(str(ref.path), force=True)
                pixels = dataset.pixel_array
            except Exception as exc:
                corrupt.append(SliceIssue(
                    ref.path.name, f"pixel data unreadable: {type(exc).__name__}"))
                continue

            if pixels.ndim != 2:
                corrupt.append(SliceIssue(
                    ref.path.name,
                    f"expected a single 2D image, found shape {tuple(pixels.shape)}"))
                continue
            if reference_shape is None:
                reference_shape = pixels.shape
            elif pixels.shape != reference_shape:
                corrupt.append(SliceIssue(
                    ref.path.name,
                    f"matrix {tuple(pixels.shape)} differs from the series "
                    f"{reference_shape}"))
                continue

            plane = pixels.astype(np.float32, copy=True)
            slope = ref.header.get("RescaleSlope")
            intercept = ref.header.get("RescaleIntercept")
            if slope is not None or intercept is not None:
                plane = plane * float(slope or 1.0) + float(intercept or 0.0)
            interpretation = str(ref.header.get("PhotometricInterpretation") or "")
            if interpretation:
                photometric.add(interpretation.upper())
            planes.append(plane)

        if "MONOCHROME1" in photometric:
            warnings.append(
                "PhotometricInterpretation is MONOCHROME1 (higher stored value renders "
                "darker). Stored values were kept as acquired — inverting them is a "
                "display transform and would corrupt quantitative measurements — so a "
                "viewer must apply the inversion, not this layer")
        if len(photometric) > 1:
            warnings.append(f"slices mix photometric interpretations {sorted(photometric)}")
        if corrupt:
            warnings.append(f"{len(corrupt)} slice(s) could not be decoded and were "
                            "excluded from the volume")
        return planes, corrupt

    # -- geometry ------------------------------------------------------------ #
    def _geometry(self, ordered: list[_SliceRef], planes: list[np.ndarray],
                  spacing_stats: dict[str, Any],
                  warnings: list[str]) -> VoxelGeometry:
        """Build the RAS+ affine for the assembled stack."""
        first = ordered[0]
        rows, columns = planes[0].shape
        pixel_spacing = _float_list(first.header.get("PixelSpacing"))
        if not pixel_spacing or len(pixel_spacing) != 2:
            pixel_spacing = [1.0, 1.0]
            warnings.append(
                "PixelSpacing is absent; 1.0 mm was assumed in plane. Every physical "
                "measurement from this volume is unreliable")

        orientation = first.orientation
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            first.header["_OrientationAssumed"] = True
            warnings.append(
                "ImageOrientationPatient is absent; an identity orientation was "
                "assumed, so the world orientation of this volume is unknown")
        position = first.position
        if position is None:
            position = np.zeros(3)
            warnings.append("ImagePositionPatient is absent; the volume origin was set "
                            "to zero and world coordinates are not meaningful")

        last_position = ordered[-1].position if len(ordered) > 1 else None
        slice_spacing = spacing_stats["median"] or _first_float(
            first.header.get("SpacingBetweenSlices"),
            first.header.get("SliceThickness")) or 1.0

        affine = affine_from_dicom(
            orientation, position, pixel_spacing,
            last_position=last_position,
            slice_count=len(ordered),
            slice_spacing=slice_spacing,
        )
        return VoxelGeometry(affine=affine, shape=(columns, rows, len(planes)))

    # ------------------------------------------------------------------ #
    # Enhanced multi-frame
    # ------------------------------------------------------------------ #
    def _build_multiframe(self, key: str, ref: _SliceRef) -> RawSeries:
        """Read an Enhanced MR object: one file holding the whole volume.

        Per-frame geometry lives in ``PerFrameFunctionalGroupsSequence``. When it is
        present the frames are ordered by real position exactly as classic slices are.
        When it is absent, the shared spacing tag is the only geometry available and
        the volume is flagged, because frame order then rests on storage order alone.
        """
        import pydicom

        try:
            dataset = pydicom.dcmread(str(ref.path), force=True)
            pixels = dataset.pixel_array
        except Exception as exc:
            raise CorruptStudy(
                "the enhanced multi-frame MR object could not be decoded",
                detail={"file": ref.path.name}) from exc

        if pixels.ndim != 3:
            raise StudyValidationError(
                f"multi-frame object has shape {tuple(pixels.shape)}; a 3D "
                "(frame, row, column) array was expected")

        warnings: list[str] = []
        positions, orientation, spacing = self._multiframe_geometry(dataset, warnings)
        order = list(range(pixels.shape[0]))
        if positions is not None and orientation is not None:
            normal = np.cross(orientation[:3], orientation[3:])
            norm = float(np.linalg.norm(normal))
            if norm > 1e-8:
                normal = normal / norm
                order = sorted(order, key=lambda i: float(np.dot(positions[i], normal)))
        else:
            warnings.append(
                "per-frame position information is absent; frames were kept in storage "
                "order, which is not guaranteed to be spatial order")

        stacked = pixels[order].astype(np.float32, copy=True)
        first_position = positions[order[0]] if positions is not None else np.zeros(3)
        last_position = positions[order[-1]] if positions is not None else None
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            warnings.append("frame orientation is absent; an identity orientation was "
                            "assumed and world orientation is unknown")

        affine = affine_from_dicom(
            orientation, first_position, spacing[:2],
            last_position=last_position, slice_count=stacked.shape[0],
            slice_spacing=spacing[2])
        array = np.transpose(stacked, (2, 1, 0))
        geometry = VoxelGeometry(affine=affine, shape=tuple(array.shape[:3]))

        header = dict(ref.header)
        header["_SliceCount"] = int(stacked.shape[0])
        if last_position is not None:
            header["_ImagePositionPatientLast"] = [float(v) for v in last_position]

        return RawSeries(
            series_key=key,
            source_format=FileFormat.DICOM,
            voxels=array,
            geometry=geometry,
            header=header,
            integrity=SeriesIntegrity(
                files_found=1,
                slices_loaded=int(stacked.shape[0]),
                median_slice_spacing_mm=float(spacing[2]),
                warnings=tuple(warnings),
            ),
            source_name=str(ref.header.get("SeriesDescription") or ref.path.name),
            contributing_files=(ref.path.name,),
        )

    @staticmethod
    def _multiframe_geometry(dataset: Any, warnings: list[str]
                             ) -> tuple[np.ndarray | None, np.ndarray | None,
                                        list[float]]:
        """Pull per-frame positions and shared spacing out of the functional groups."""
        positions: list[list[float]] = []
        orientation: np.ndarray | None = None
        spacing = [1.0, 1.0, 1.0]

        shared = getattr(dataset, "SharedFunctionalGroupsSequence", None)
        if shared:
            group = shared[0]
            measures = getattr(group, "PixelMeasuresSequence", None)
            if measures:
                pixel_spacing = _float_list(getattr(measures[0], "PixelSpacing", None))
                thickness = _float_list(
                    getattr(measures[0], "SpacingBetweenSlices", None)) or _float_list(
                    getattr(measures[0], "SliceThickness", None))
                if pixel_spacing and len(pixel_spacing) == 2:
                    spacing[0], spacing[1] = pixel_spacing
                if thickness:
                    spacing[2] = thickness[0]
            plane = getattr(group, "PlaneOrientationSequence", None)
            if plane:
                values = _float_list(
                    getattr(plane[0], "ImageOrientationPatient", None))
                if values and len(values) == 6:
                    orientation = np.asarray(values, dtype=float)

        per_frame = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
        if per_frame:
            for frame in per_frame:
                plane_position = getattr(frame, "PlanePositionSequence", None)
                values = _float_list(getattr(plane_position[0],
                                             "ImagePositionPatient", None)) \
                    if plane_position else None
                positions.append(values if values and len(values) == 3 else [0.0] * 3)
                if orientation is None:
                    plane_orientation = getattr(frame, "PlaneOrientationSequence", None)
                    if plane_orientation:
                        iop = _float_list(getattr(plane_orientation[0],
                                                  "ImageOrientationPatient", None))
                        if iop and len(iop) == 6:
                            orientation = np.asarray(iop, dtype=float)
        else:
            warnings.append("the multi-frame object has no PerFrameFunctionalGroups"
                            "Sequence; per-frame geometry is unavailable")

        return (np.asarray(positions, dtype=float) if positions else None,
                orientation, spacing)


def _first_float(*values: Any) -> float | None:
    """First value that converts to a positive float."""
    for value in values:
        items = _float_list(value)
        if items and items[0] > 0:
            return float(items[0])
    return None
