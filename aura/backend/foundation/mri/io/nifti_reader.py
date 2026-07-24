"""NIfTI-1 and NIfTI-2 reader, implemented against the published header layout.

nibabel is not installed in this deployment and adding it was not an option here, but
a brain-MRI foundation layer that cannot read NIfTI is not usable: it is the format
every public dataset (BraTS, IXI, ADNI derivatives, OASIS) and every conversion tool
(dcm2niix, MONAI, nnU-Net) exchanges volumes in.

Scope, stated so nobody has to test to find out:

* NIfTI-1 (``n+1``/``ni1``, 348-byte header) and NIfTI-2 (``n+2``/``ni2``, 540-byte).
* Single-file ``.nii``, gzipped ``.nii.gz``, and the two-file ``.hdr``/``.img`` pair.
* Both byte orders, detected from ``sizeof_hdr`` rather than guessed.
* All real-valued scalar datatypes (int8..int64, uint8..uint64, float32, float64).
* ``scl_slope``/``scl_inter`` applied when the header declares them.
* Affine from ``sform`` when its code is set, else ``qform``, else the ``pixdim``
  fallback — the precedence the standard specifies.
* 3D volumes and 4D series (the fourth dimension is kept, not collapsed).

Deliberately **not** supported, each raising a clear error rather than producing a
wrong array: complex and RGB datatypes, and dimensionality above 4. Those exist in
NIfTI but are not brain-MRI intensity volumes, and guessing at them would be worse
than declining.

Header extensions are skipped: they sit between the header and ``vox_offset``, which
is exactly why data is read from ``vox_offset`` rather than from a fixed position.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.foundation.mri.errors import CorruptStudy, UnsupportedStudyFormat
from backend.foundation.mri.geometry import VoxelGeometry, affine_from_quaternion
from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
from backend.foundation.mri.types import FileFormat

log = get_logger("foundation.mri.io.nifti")

NIFTI1_HEADER_SIZE = 348
NIFTI2_HEADER_SIZE = 540

_NIFTI1_DTYPE = np.dtype([
    ("sizeof_hdr", "i4"), ("data_type", "S10"), ("db_name", "S18"),
    ("extents", "i4"), ("session_error", "i2"), ("regular", "S1"),
    ("dim_info", "u1"), ("dim", "i2", (8,)),
    ("intent_p1", "f4"), ("intent_p2", "f4"), ("intent_p3", "f4"),
    ("intent_code", "i2"), ("datatype", "i2"), ("bitpix", "i2"),
    ("slice_start", "i2"), ("pixdim", "f4", (8,)), ("vox_offset", "f4"),
    ("scl_slope", "f4"), ("scl_inter", "f4"), ("slice_end", "i2"),
    ("slice_code", "u1"), ("xyzt_units", "u1"), ("cal_max", "f4"),
    ("cal_min", "f4"), ("slice_duration", "f4"), ("toffset", "f4"),
    ("glmax", "i4"), ("glmin", "i4"), ("descrip", "S80"), ("aux_file", "S24"),
    ("qform_code", "i2"), ("sform_code", "i2"),
    ("quatern_b", "f4"), ("quatern_c", "f4"), ("quatern_d", "f4"),
    ("qoffset_x", "f4"), ("qoffset_y", "f4"), ("qoffset_z", "f4"),
    ("srow_x", "f4", (4,)), ("srow_y", "f4", (4,)), ("srow_z", "f4", (4,)),
    ("intent_name", "S16"), ("magic", "S4"),
])

_NIFTI2_DTYPE = np.dtype([
    ("sizeof_hdr", "i4"), ("magic", "S8"), ("datatype", "i2"), ("bitpix", "i2"),
    ("dim", "i8", (8,)),
    ("intent_p1", "f8"), ("intent_p2", "f8"), ("intent_p3", "f8"),
    ("pixdim", "f8", (8,)), ("vox_offset", "i8"),
    ("scl_slope", "f8"), ("scl_inter", "f8"), ("cal_max", "f8"), ("cal_min", "f8"),
    ("slice_duration", "f8"), ("toffset", "f8"),
    ("slice_start", "i8"), ("slice_end", "i8"),
    ("descrip", "S80"), ("aux_file", "S24"),
    ("qform_code", "i4"), ("sform_code", "i4"),
    ("quatern_b", "f8"), ("quatern_c", "f8"), ("quatern_d", "f8"),
    ("qoffset_x", "f8"), ("qoffset_y", "f8"), ("qoffset_z", "f8"),
    ("srow_x", "f8", (4,)), ("srow_y", "f8", (4,)), ("srow_z", "f8", (4,)),
    ("slice_code", "i4"), ("xyzt_units", "i4"), ("intent_code", "i4"),
    ("intent_name", "S16"), ("dim_info", "u1"), ("unused_str", "S15"),
])

#: NIfTI datatype code -> numpy base type. Codes absent here are rejected by name.
_DATATYPES: dict[int, str] = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
    256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}
_REJECTED_DATATYPES: dict[int, str] = {
    32: "complex64", 128: "RGB24", 1536: "float128", 1792: "complex128",
    2048: "complex256", 2304: "RGBA32", 0: "unknown",
}

#: xyzt_units spatial codes -> millimetres per stored unit.
_SPATIAL_UNIT_SCALE: dict[int, float] = {0: 1.0, 1: 1000.0, 2: 1.0, 3: 0.001}


def _open_maybe_gzip(path: Path) -> BinaryIO:
    """Open ``path``, transparently decompressing gzip.

    Detected by magic bytes rather than by suffix: plenty of real archives hold
    gzipped volumes named ``.nii``, and the suffix is the least reliable thing about
    a medical image file.
    """
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")            # type: ignore[return-value]
    return open(path, "rb")


def _detect_header(raw: bytes) -> tuple[np.dtype, str]:
    """Identify header version and byte order from ``sizeof_hdr``.

    The first field is the header's own length, which makes it a self-describing
    endianness marker: 348 little-endian is 1543569408 big-endian, so a mismatch is
    unambiguous. Guessing from the magic string instead would fail on the ``.hdr``
    half of a pair written by an older tool.
    """
    if len(raw) < 4:
        raise CorruptStudy("file is too short to contain a NIfTI header")
    for order in ("<", ">"):
        size = int(np.frombuffer(raw[:4], dtype=f"{order}i4")[0])
        if size == NIFTI1_HEADER_SIZE:
            return _NIFTI1_DTYPE.newbyteorder(order), order
        if size == NIFTI2_HEADER_SIZE:
            return _NIFTI2_DTYPE.newbyteorder(order), order
    raise UnsupportedStudyFormat(
        "the file does not start with a NIfTI-1 or NIfTI-2 header",
        detail={"expected_sizeof_hdr": [NIFTI1_HEADER_SIZE, NIFTI2_HEADER_SIZE]},
    )


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).split(b"\x00")[0].decode("ascii", "replace").strip()
    return str(value).strip()


def affine_from_nifti_header(header: np.ndarray) -> tuple[np.ndarray, str]:
    """Derive the voxel-to-world affine, following the standard's precedence.

    Returns ``(affine, source)`` where ``source`` is ``'sform'``, ``'qform'``, or
    ``'pixdim_fallback'``.

    The precedence is not a preference — it is what the format specifies. ``sform``
    is the general affine (it can express shear from a registration); ``qform`` is a
    rigid scanner-anatomical transform. When both are set they describe the same
    volume in different spaces, and tools that pick the wrong one produce volumes that
    are subtly misaligned rather than obviously broken.

    The ``pixdim`` fallback (both codes zero, the "ANALYZE" case) yields spacing but
    no orientation. It is flagged so the quality inspector can warn: a volume whose
    left-right direction is unknown must not be reported on without a human confirming
    it.
    """
    sform_code = int(header["sform_code"])
    qform_code = int(header["qform_code"])
    scale = _SPATIAL_UNIT_SCALE.get(int(header["xyzt_units"]) & 0x07, 1.0)

    if sform_code > 0:
        affine = np.eye(4, dtype=float)
        affine[0, :] = np.asarray(header["srow_x"], dtype=float)
        affine[1, :] = np.asarray(header["srow_y"], dtype=float)
        affine[2, :] = np.asarray(header["srow_z"], dtype=float)
        affine[:3, :] *= scale
        return affine, "sform"

    pixdim = np.asarray(header["pixdim"], dtype=float)
    if qform_code > 0:
        qfac = pixdim[0] if pixdim[0] != 0 else 1.0
        affine = affine_from_quaternion(
            (header["quatern_b"], header["quatern_c"], header["quatern_d"]),
            (header["qoffset_x"], header["qoffset_y"], header["qoffset_z"]),
            pixdim[1:4], float(qfac),
        )
        affine[:3, :] *= scale
        return affine, "qform"

    zooms = np.where(pixdim[1:4] > 0, pixdim[1:4], 1.0) * scale
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = np.diag(zooms)
    return affine, "pixdim_fallback"


class NiftiReader:
    """Reads one NIfTI volume per file. One file is one series."""

    file_format = FileFormat.NIFTI

    #: Recognised suffixes. ``.img`` is not listed: the ``.hdr`` is the entry point
    #: for a pair, and listing both would read the same volume twice.
    SUFFIXES = (".nii", ".nii.gz", ".hdr")

    def can_read(self, path: Path) -> bool:
        name = path.name.lower()
        if not any(name.endswith(s) for s in self.SUFFIXES):
            return False
        try:
            with _open_maybe_gzip(path) as fh:
                head = fh.read(4)
            size = int(np.frombuffer(head, dtype="<i4")[0]) if len(head) == 4 else 0
            swapped = int(np.frombuffer(head, dtype=">i4")[0]) if len(head) == 4 else 0
        except (OSError, ValueError):
            return False
        return {size, swapped} & {NIFTI1_HEADER_SIZE, NIFTI2_HEADER_SIZE} != set()

    def read(self, paths: Sequence[Path], *,
             issues: list[dict[str, Any]] | None = None) -> list[RawSeries]:
        series: list[RawSeries] = []
        for path in paths:
            try:
                series.append(self.read_one(path))
            except (CorruptStudy, UnsupportedStudyFormat) as exc:
                log.warning(
                    "skipping unreadable NIfTI file",
                    extra={"context": {"file": path.name, "reason": exc.reason}},
                )
                if issues is not None:
                    issues.append({"series_key": path.name, "error": exc.code,
                                   "reason": exc.reason})
        return series

    def read_one(self, path: Path) -> RawSeries:
        """Decode a single NIfTI file into a :class:`RawSeries`."""
        with _open_maybe_gzip(path) as fh:
            payload = fh.read()

        dtype, order = _detect_header(payload)
        if len(payload) < dtype.itemsize:
            raise CorruptStudy(
                "the NIfTI header is truncated",
                detail={"expected_bytes": int(dtype.itemsize), "found": len(payload)})
        header = np.frombuffer(payload[:dtype.itemsize], dtype=dtype)[0]

        magic = _text(header["magic"])
        shape, ndim = self._shape(header)
        voxels = self._read_data(path, payload, header, order, shape, magic)
        voxels = self._apply_scaling(voxels, header)

        affine, affine_source = affine_from_nifti_header(header)
        geometry = VoxelGeometry(affine=affine, shape=tuple(shape[:3]))

        warnings: list[str] = []
        if affine_source == "pixdim_fallback":
            warnings.append(
                "the file sets neither sform_code nor qform_code; voxel spacing is "
                "known but the world orientation is not, so left-right cannot be "
                "established from this file alone")
        if ndim == 4 and shape[3] > 1:
            warnings.append(
                f"the file holds {shape[3]} volumes on one grid (4D); a single volume "
                "must be selected before analysis")

        raw_header: dict[str, Any] = {
            "descrip": _text(header["descrip"]),
            "intent_name": _text(header["intent_name"]),
            "magic": magic,
            "nifti_version": 1 if dtype.itemsize == NIFTI1_HEADER_SIZE else 2,
            "byte_order": order,
            "datatype_code": int(header["datatype"]),
            "affine_source": affine_source,
            "scl_slope": float(header["scl_slope"]),
            "scl_inter": float(header["scl_inter"]),
            "xyzt_units": int(header["xyzt_units"]),
            "pixdim": [float(v) for v in np.asarray(header["pixdim"], dtype=float)],
            "dim": [int(v) for v in np.asarray(header["dim"], dtype=int)],
        }

        return RawSeries(
            series_key=path.name,
            source_format=FileFormat.NIFTI,
            voxels=voxels,
            geometry=geometry,
            header=raw_header,
            integrity=SeriesIntegrity(
                files_found=1,
                slices_loaded=int(shape[2]),
                median_slice_spacing_mm=float(geometry.spacing[2]),
                warnings=tuple(warnings),
            ),
            source_name=path.name,
            contributing_files=(path.name,),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _shape(header: np.ndarray) -> tuple[list[int], int]:
        dim = np.asarray(header["dim"], dtype=int)
        ndim = int(dim[0])
        if ndim < 1:
            raise CorruptStudy(f"NIfTI header declares {ndim} dimensions")
        if ndim > 4:
            # 5D+ NIfTI holds vectors or tensors per voxel, not an intensity volume.
            raise UnsupportedStudyFormat(
                f"NIfTI files with {ndim} dimensions are not intensity volumes and "
                "are not supported by the MRI foundation layer",
                detail={"dim": [int(v) for v in dim]})
        shape = [int(v) for v in dim[1:ndim + 1]]
        if any(s < 1 for s in shape):
            raise CorruptStudy("NIfTI header declares a non-positive dimension",
                               detail={"shape": shape})
        while len(shape) < 3:                    # a 2D slice is a 1-slice volume
            shape.append(1)
        return shape, ndim

    def _read_data(self, path: Path, payload: bytes, header: np.ndarray, order: str,
                   shape: list[int], magic: str) -> np.ndarray:
        code = int(header["datatype"])
        if code not in _DATATYPES:
            raise UnsupportedStudyFormat(
                f"NIfTI datatype {code} "
                f"({_REJECTED_DATATYPES.get(code, 'unrecognised')}) is not a real-valued "
                "intensity type and is not supported",
                detail={"datatype_code": code})
        dtype = np.dtype(f"{order}{_DATATYPES[code]}")
        count = int(np.prod(shape))
        offset = int(header["vox_offset"])

        if magic in ("ni1", "ni2"):
            # Two-file form: the header file has no pixel data at all.
            payload = self._read_paired_image(path)
            offset = max(offset, 0)

        end = offset + count * dtype.itemsize
        if len(payload) < end:
            raise CorruptStudy(
                "NIfTI pixel data is truncated: the header declares more voxels than "
                "the file contains",
                detail={"declared_bytes": count * dtype.itemsize,
                        "available_bytes": max(0, len(payload) - offset),
                        "shape": shape})
        flat = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        # NIfTI stores the first index fastest — Fortran order. ``copy=True`` is not
        # incidental: ``frombuffer`` returns a read-only view onto the file bytes, and
        # every later stage expects a writable array it can work in place on.
        return flat.reshape(shape, order="F").astype(np.float32, copy=True)

    @staticmethod
    def _read_paired_image(header_path: Path) -> bytes:
        """Load the ``.img`` half of a ``.hdr``/``.img`` pair."""
        for suffix in (".img", ".img.gz", ".IMG"):
            candidate = header_path.with_suffix(suffix)
            if candidate.exists():
                with _open_maybe_gzip(candidate) as fh:
                    return fh.read()
        raise CorruptStudy(
            "this is the header half of a two-file NIfTI pair and the matching image "
            "file is missing",
            detail={"header_file": header_path.name,
                    "expected_image_file": header_path.with_suffix('.img').name})

    @staticmethod
    def _apply_scaling(voxels: np.ndarray, header: np.ndarray) -> np.ndarray:
        """Apply ``scl_slope``/``scl_inter`` when the header declares them.

        A zero slope means "no scaling" per the standard, *not* "multiply by zero" —
        a distinction that silently blanks a volume when got wrong.
        """
        slope = float(header["scl_slope"])
        inter = float(header["scl_inter"])
        if slope == 0.0 or not np.isfinite(slope):
            return voxels
        if slope == 1.0 and inter == 0.0:
            return voxels
        return voxels * np.float32(slope) + np.float32(inter)
