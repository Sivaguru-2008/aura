"""NRRD reader, implemented against the format specification.

NRRD is 3D Slicer's native format and the interchange format for several public
segmentation datasets, so a foundation layer that reads DICOM and NIfTI but not NRRD
would still turn away real studies. ``pynrrd`` is not installed here; the format is a
plain-text header over a raw, gzipped, or ASCII payload, which makes a correct reader
a bounded amount of work rather than a dependency.

Supported: NRRD magic ``NRRD0001``-``NRRD0005``; ``raw``, ``gzip``/``gz``, and
``ascii``/``text``/``txt`` encodings; attached and detached (``data file:``) payloads;
both byte orders; all real scalar types; ``space``/``space directions``/``space
origin`` geometry with LPS and RAS handling; ``line skip``/``byte skip``.

Not supported, each declined explicitly: ``bzip2`` encoding (no stdlib-only
guarantee that the payload is seekable the way the format allows), block types,
and the multi-file ``LIST`` data-file form. Those appear in tractography and
histology exports, not in brain-MRI volumes.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.foundation.mri.errors import CorruptStudy, UnsupportedStudyFormat
from backend.foundation.mri.geometry import LPS_TO_RAS, VoxelGeometry
from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
from backend.foundation.mri.types import FileFormat

log = get_logger("foundation.mri.io.nrrd")

_MAGIC = re.compile(rb"^NRRD000[1-5]")

#: NRRD type name -> numpy base type. The spec allows several spellings per type and
#: real files use all of them, so every accepted spelling is listed rather than
#: normalised with a fragile regex.
_TYPES: dict[str, str] = {
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "short": "i2", "short int": "i2", "signed short": "i2",
    "signed short int": "i2", "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2",
    "uint16": "u2", "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "longlong": "i8", "long long": "i8", "long long int": "i8",
    "signed long long": "i8", "int64": "i8", "int64_t": "i8",
    "ulonglong": "u8", "unsigned long long": "u8", "uint64": "u8", "uint64_t": "u8",
    "float": "f4", "double": "f8",
}

#: Recognised world spaces and whether they need converting to RAS.
_LPS_SPACES = {"left-posterior-superior", "lps", "left-posterior-superior-time"}
_RAS_SPACES = {"right-anterior-superior", "ras", "right-anterior-superior-time"}
_UNKNOWN_SPACES = {"scanner-xyz", "3d-right-handed", "3d-left-handed"}


def _parse_vector(text: str) -> list[float] | None:
    """Parse a ``(1.0,0,0)`` tuple. ``none`` marks a non-spatial axis."""
    text = text.strip()
    if text.lower() == "none":
        return None
    if not (text.startswith("(") and text.endswith(")")):
        raise CorruptStudy(f"malformed NRRD vector {text!r}")
    return [float(v) for v in text[1:-1].split(",")]


def _split_vectors(text: str) -> list[list[float] | None]:
    """Split a space-directions field into per-axis vectors."""
    return [_parse_vector(tok) for tok in re.findall(r"\([^)]*\)|none|None|NONE", text)]


class NrrdReader:
    """Reads one NRRD volume per header file."""

    file_format = FileFormat.NRRD
    SUFFIXES = (".nrrd", ".nhdr")

    def can_read(self, path: Path) -> bool:
        if path.suffix.lower() not in self.SUFFIXES:
            return False
        try:
            with open(path, "rb") as fh:
                return bool(_MAGIC.match(fh.read(8)))
        except OSError:
            return False

    def read(self, paths: Sequence[Path], *,
             issues: list[dict[str, Any]] | None = None) -> list[RawSeries]:
        out: list[RawSeries] = []
        for path in paths:
            try:
                out.append(self.read_one(path))
            except (CorruptStudy, UnsupportedStudyFormat) as exc:
                log.warning("skipping unreadable NRRD file",
                            extra={"context": {"file": path.name, "reason": exc.reason}})
                if issues is not None:
                    issues.append({"series_key": path.name, "error": exc.code,
                                   "reason": exc.reason})
        return out

    def read_one(self, path: Path) -> RawSeries:
        fields, key_values, header_bytes = self._parse_header(path)
        shape, dtype, payload = self._read_payload(path, fields, header_bytes)
        voxels, spatial_axes = self._to_volume(shape, dtype, payload, fields)
        affine, space_name, warnings = self._affine(fields, spatial_axes, voxels.shape)

        geometry = VoxelGeometry(affine=affine, shape=tuple(voxels.shape[:3]))
        if voxels.ndim == 4 and voxels.shape[3] > 1:
            warnings.append(
                f"the file holds {voxels.shape[3]} volumes on one grid (4D); a single "
                "volume must be selected before analysis")

        raw_header: dict[str, Any] = {
            "content": fields.get("content"),
            "space": space_name,
            # Read by the metadata engine to decide whether left-right is established.
            "affine_source": ("space_directions" if fields.get("space directions")
                              else "assumed"),
            "encoding": fields.get("encoding"),
            "nrrd_type": fields.get("type"),
            "dimension": fields.get("dimension"),
            "kinds": fields.get("kinds"),
            **{k: v for k, v in key_values.items()},
        }
        return RawSeries(
            series_key=path.name,
            source_format=FileFormat.NRRD,
            voxels=voxels,
            geometry=geometry,
            header=raw_header,
            integrity=SeriesIntegrity(
                files_found=1,
                slices_loaded=int(voxels.shape[2]),
                median_slice_spacing_mm=float(geometry.spacing[2]),
                warnings=tuple(warnings),
            ),
            source_name=path.name,
            contributing_files=(path.name,),
        )

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_header(path: Path) -> tuple[dict[str, str], dict[str, str], int]:
        """Read the text header. Returns (fields, key/value pairs, bytes consumed).

        NRRD separates ``field: value`` (spec-defined) from ``key:=value`` (arbitrary
        user metadata). They are kept apart because only the first group may change
        how the payload is decoded — a user key that happened to be named ``sizes``
        must never be able to reshape the volume.
        """
        fields: dict[str, str] = {}
        key_values: dict[str, str] = {}
        consumed = 0
        with open(path, "rb") as fh:
            first = fh.readline()
            consumed += len(first)
            if not _MAGIC.match(first):
                raise UnsupportedStudyFormat("the file does not begin with an NRRD magic")
            while True:
                line = fh.readline()
                if not line:
                    break                      # detached header with no blank line
                consumed += len(line)
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                if not text.strip():
                    break                      # blank line ends the header
                if text.startswith("#"):
                    continue
                if ":=" in text:
                    key, value = text.split(":=", 1)
                    key_values[key.strip()] = value.strip()
                elif ":" in text:
                    key, value = text.split(":", 1)
                    fields[key.strip().lower()] = value.strip()
        if "sizes" not in fields or "type" not in fields:
            raise CorruptStudy(
                "NRRD header is missing a required field",
                detail={"required": ["sizes", "type"],
                        "found": sorted(fields)})
        return fields, key_values, consumed

    # ------------------------------------------------------------------ #
    # Payload
    # ------------------------------------------------------------------ #
    def _read_payload(self, path: Path, fields: dict[str, str],
                      header_bytes: int) -> tuple[list[int], np.dtype, bytes]:
        type_name = fields["type"].strip().lower()
        if type_name not in _TYPES:
            raise UnsupportedStudyFormat(
                f"NRRD type {type_name!r} is not a real-valued scalar type",
                detail={"supported": sorted(set(_TYPES))})
        shape = [int(v) for v in fields["sizes"].split()]
        if not shape or any(s < 1 for s in shape):
            raise CorruptStudy("NRRD 'sizes' field is not a list of positive integers",
                               detail={"sizes": fields.get("sizes")})

        endian = fields.get("endian", "little").strip().lower()
        base = _TYPES[type_name]
        dtype = np.dtype(("<" if endian == "little" else ">") + base) \
            if np.dtype(base).itemsize > 1 else np.dtype(base)

        encoding = fields.get("encoding", "raw").strip().lower()
        data_file = fields.get("data file") or fields.get("datafile")
        if data_file:
            if data_file.strip().upper().startswith("LIST"):
                raise UnsupportedStudyFormat(
                    "multi-file NRRD payloads (the LIST form) are not supported")
            source = (path.parent / data_file.strip()).resolve()
            if not source.exists():
                raise CorruptStudy(
                    "the NRRD header points at a detached data file that is missing",
                    detail={"expected_file": Path(data_file.strip()).name})
            payload = source.read_bytes()
        else:
            payload = path.read_bytes()[header_bytes:]

        payload = self._decode(payload, encoding, fields, shape, dtype)
        return shape, dtype, payload

    @staticmethod
    def _decode(payload: bytes, encoding: str, fields: dict[str, str],
                shape: list[int], dtype: np.dtype) -> bytes:
        line_skip = int(fields.get("line skip") or fields.get("lineskip") or 0)
        byte_skip = int(fields.get("byte skip") or fields.get("byteskip") or 0)
        for _ in range(line_skip):
            index = payload.find(b"\n")
            if index < 0:
                raise CorruptStudy("NRRD 'line skip' exceeds the payload length")
            payload = payload[index + 1:]

        if encoding in ("gzip", "gz"):
            # byte skip -1 means "skip to the end of the gzip stream header", which
            # for a whole-file gzip payload is simply "decompress from the start".
            if byte_skip > 0:
                payload = payload[byte_skip:]
            try:
                payload = gzip.decompress(payload)
            except (OSError, EOFError) as exc:
                raise CorruptStudy("the NRRD gzip payload could not be decompressed"
                                   ) from exc
        elif encoding == "raw":
            if byte_skip > 0:
                payload = payload[byte_skip:]
        elif encoding in ("ascii", "text", "txt"):
            count = int(np.prod(shape))
            try:
                values = np.array(payload.split(), dtype=np.float64)
            except ValueError as exc:
                raise CorruptStudy(
                    "the NRRD ASCII payload contains a non-numeric token") from exc
            if values.size < count:
                raise CorruptStudy(
                    "the NRRD ASCII payload holds fewer values than 'sizes' declares",
                    detail={"expected": count, "found": int(values.size)})
            return values[:count].astype(dtype).tobytes()
        else:
            raise UnsupportedStudyFormat(
                f"NRRD encoding {encoding!r} is not supported",
                detail={"supported": ["raw", "gzip", "ascii"]})

        expected = int(np.prod(shape)) * dtype.itemsize
        if len(payload) < expected:
            raise CorruptStudy(
                "NRRD pixel data is truncated: the header declares more voxels than "
                "the payload contains",
                detail={"declared_bytes": expected, "available_bytes": len(payload)})
        return payload

    @staticmethod
    def _to_volume(shape: list[int], dtype: np.dtype, payload: bytes,
                   fields: dict[str, str]) -> tuple[np.ndarray, list[int]]:
        """Reshape the payload and identify which axes are spatial.

        A diffusion NRRD stores its gradient axis alongside the three spatial ones and
        marks it ``none`` in ``space directions``. Finding the spatial axes from that
        field — rather than assuming the first three — is what stops a 4D DWI from
        being read as a volume with 60 "slices".
        """
        count = int(np.prod(shape))
        flat = np.frombuffer(payload, dtype=dtype, count=count)
        # NRRD stores the fastest axis first, like NIfTI.
        array = flat.reshape(shape, order="F").astype(np.float32, copy=True)

        directions = fields.get("space directions")
        spatial_axes = list(range(min(3, array.ndim)))
        if directions:
            vectors = _split_vectors(directions)
            if len(vectors) == array.ndim:
                spatial_axes = [i for i, v in enumerate(vectors) if v is not None]
        if len(spatial_axes) < 3:
            raise UnsupportedStudyFormat(
                "the NRRD file does not describe three spatial axes and cannot be "
                "read as a volume",
                detail={"spatial_axes": spatial_axes, "shape": shape})
        if len(spatial_axes) > 3:
            spatial_axes = spatial_axes[:3]

        if array.ndim > 3:
            extra = [i for i in range(array.ndim) if i not in spatial_axes]
            array = np.transpose(array, spatial_axes + extra)
            if array.ndim > 4:                 # collapse trailing axes into frames
                array = array.reshape(array.shape[:3] + (-1,))
        return array, spatial_axes

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    @staticmethod
    def _affine(fields: dict[str, str], spatial_axes: list[int],
                shape: tuple[int, ...]) -> tuple[np.ndarray, str, list[str]]:
        """Build the RAS+ affine from ``space directions`` and ``space origin``."""
        warnings: list[str] = []
        space_name = (fields.get("space") or "").strip().lower()
        directions = fields.get("space directions")
        origin_text = fields.get("space origin")

        affine = np.eye(4, dtype=float)
        if directions:
            vectors = _split_vectors(directions)
            spatial = [v for v in vectors if v is not None][:3]
            if len(spatial) == 3 and all(len(v) == 3 for v in spatial):
                for column, vector in enumerate(spatial):
                    affine[:3, column] = vector
            else:
                warnings.append(
                    "'space directions' does not describe three 3-vectors; voxel "
                    "spacing defaults to 1 mm and the orientation is unknown")
        else:
            spacings = fields.get("spacings")
            if spacings:
                values = [float(v) for v in spacings.split()][:3]
                affine[:3, :3] = np.diag(values + [1.0] * (3 - len(values)))
            warnings.append(
                "the file has no 'space directions'; world orientation is unknown and "
                "left-right cannot be established from this file alone")

        if origin_text:
            origin = _parse_vector(origin_text)
            if origin and len(origin) == 3:
                affine[:3, 3] = origin

        if space_name in _LPS_SPACES:
            affine = LPS_TO_RAS @ affine
        elif space_name in _RAS_SPACES:
            pass
        elif space_name in _UNKNOWN_SPACES or not space_name:
            # 3D Slicer and ITK write LPS by default and often omit or genericise the
            # field. Assuming LPS matches the overwhelming majority of real medical
            # NRRD, and the assumption is recorded rather than hidden.
            affine = LPS_TO_RAS @ affine
            warnings.append(
                f"the NRRD 'space' field is {space_name or 'absent'}; LPS was assumed "
                "(the ITK/Slicer default) when converting to RAS")
        else:
            affine = LPS_TO_RAS @ affine
            warnings.append(
                f"unrecognised NRRD space {space_name!r}; LPS was assumed when "
                "converting to RAS")
        return affine, space_name or "unspecified", warnings
