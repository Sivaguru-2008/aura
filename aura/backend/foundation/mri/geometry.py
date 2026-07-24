"""Voxel-to-world geometry: affines, orientation codes, canonical reorientation.

This is the module the rest of the layer is built on, and the one where a sign error
is most expensive: an affine that is wrong by one flip produces a volume that looks
perfectly normal and is left-right mirrored. Radiological left-right confusion is a
real and documented source of wrong-site findings, so every operation here keeps the
array and its affine consistent *by construction* — a function that reorders voxels
also returns the affine that undoes the reordering, and callers never get one without
the other.

Conventions
-----------
* **World frame is RAS+** (see :mod:`backend.foundation.mri.types`). +x → right,
  +y → anterior, +z → superior, in millimetres.
* **Voxel indices are ``[i, j, k]``** with ``array.shape == (ni, nj, nk)``. The affine
  maps ``[i, j, k, 1]`` to world millimetres. This is the NIfTI convention, not
  numpy's image convention — a DICOM ``pixel_array`` is ``(row, column)``, i.e.
  ``(j, i)``, and the DICOM reader transposes on the way in rather than leaving a
  trap for every later module.

The orientation algorithms (``io_orientation`` / ``apply_orientation`` /
``inv_ornt_aff``) follow the standard formulation used across neuroimaging tooling;
they are reimplemented here because nibabel is not a dependency of this deployment,
and are covered by round-trip tests that assert world coordinates are preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from backend.foundation.mri.types import AnatomicalPlane

#: RAS+ axis labels, indexed by world axis then by direction sign (negative, positive).
_AXIS_LABELS: tuple[tuple[str, str], ...] = (("L", "R"), ("P", "A"), ("I", "S"))

#: The orientation array of a volume that is already canonical RAS.
CANONICAL_ORNT: np.ndarray = np.array([[0, 1], [1, 1], [2, 1]], dtype=float)

#: Below this the affine's rotation block is treated as degenerate (a zero-length
#: direction vector, or two collinear ones). 1e-8 in mm-scale units is far below any
#: real acquisition and far above float32 noise.
_SINGULAR_TOL = 1e-8


# --------------------------------------------------------------------------- #
# Affine construction
# --------------------------------------------------------------------------- #
#: DICOM patient coordinates are LPS; NIfTI/RAS flips the first two axes.
LPS_TO_RAS: np.ndarray = np.diag([-1.0, -1.0, 1.0, 1.0])


def affine_from_dicom(
    image_orientation: Sequence[float],
    first_position: Sequence[float],
    pixel_spacing: Sequence[float],
    *,
    last_position: Sequence[float] | None = None,
    slice_count: int = 1,
    slice_spacing: float | None = None,
) -> np.ndarray:
    """Build a RAS+ voxel-to-world affine from DICOM geometry tags.

    Args:
        image_orientation: ``ImageOrientationPatient`` (0020,0037) — six direction
            cosines, the first three along increasing *column* index (our ``i``), the
            next three along increasing *row* index (our ``j``).
        first_position: ``ImagePositionPatient`` (0020,0032) of the first slice, in
            acquisition order.
        pixel_spacing: ``PixelSpacing`` (0028,0030) as ``[row spacing, column
            spacing]``. Note the order: DICOM gives *between-row* spacing first,
            which is the step along ``j``.
        last_position: ``ImagePositionPatient`` of the last slice. When given, the
            slice direction is derived from the two positions, which is exact — it
            captures gantry tilt and any real inter-slice gap.
        slice_count: number of slices, used with ``last_position``.
        slice_spacing: fallback step along ``k`` when only one slice exists or the
            positions are unusable. Falls back to the cross-product normal.

    Returns:
        A ``4x4`` affine mapping ``[i, j, k, 1]`` to RAS millimetres.

    The slice direction is taken from the *positions* rather than from the
    cross-product normal whenever possible, because those disagree in two real cases:
    a tilted gantry, and a series stored in reverse spatial order. Trusting the normal
    there yields a volume that is flipped head-to-foot.
    """
    iop = np.asarray(image_orientation, dtype=float).reshape(6)
    e_i = iop[:3]
    e_j = iop[3:]
    origin = np.asarray(first_position, dtype=float).reshape(3)
    spacing = np.asarray(pixel_spacing, dtype=float).reshape(2)
    d_j, d_i = float(spacing[0]), float(spacing[1])

    if last_position is not None and slice_count > 1:
        last = np.asarray(last_position, dtype=float).reshape(3)
        e_k = (last - origin) / float(slice_count - 1)
    else:
        normal = np.cross(e_i, e_j)
        norm = float(np.linalg.norm(normal))
        normal = normal / norm if norm > _SINGULAR_TOL else np.array([0.0, 0.0, 1.0])
        e_k = normal * float(slice_spacing if slice_spacing else 1.0)

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, 0] = e_i * d_i
    affine_lps[:3, 1] = e_j * d_j
    affine_lps[:3, 2] = e_k
    affine_lps[:3, 3] = origin
    return LPS_TO_RAS @ affine_lps


def affine_from_quaternion(
    quatern: Sequence[float],
    offset: Sequence[float],
    pixdim: Sequence[float],
    qfac: float,
) -> np.ndarray:
    """Rebuild a NIfTI qform affine from its quaternion representation.

    ``quatern`` holds ``(b, c, d)``; ``a`` is recovered from the unit-norm constraint.
    ``qfac`` (``pixdim[0]``, ±1) flips the third axis and exists precisely because a
    quaternion can only express a rotation — it cannot express the left-handed frame
    that a real acquisition sometimes has.
    """
    b, c, d = (float(v) for v in quatern)
    a_sq = 1.0 - (b * b + c * c + d * d)
    # Tiny negatives are float error in the stored quaternion, not a malformed file.
    a = math.sqrt(a_sq) if a_sq > 0.0 else 0.0

    rotation = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ], dtype=float)

    zooms = np.asarray(pixdim, dtype=float).reshape(3).copy()
    zooms[2] *= (-1.0 if qfac < 0 else 1.0)

    affine = np.eye(4, dtype=float)
    affine[:3, :3] = rotation @ np.diag(zooms)
    affine[:3, 3] = np.asarray(offset, dtype=float).reshape(3)
    return affine


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
def io_orientation(affine: np.ndarray) -> np.ndarray:
    """Map each voxel axis to the world axis it most closely runs along.

    Returns a ``(3, 2)`` array where row ``i`` is ``(world_axis, direction)`` for
    voxel axis ``i``, and ``direction`` is ``+1`` or ``-1``.

    The rotation block is orthogonalised by SVD before the axes are read off. That
    matters for oblique acquisitions: on a 20°-oblique coronal series the raw columns
    can be ambiguous enough that a naive ``argmax`` assigns two voxel axes to the same
    world axis, and the resulting "reorientation" silently duplicates one axis.
    """
    aff = np.asarray(affine, dtype=float)
    rzs = aff[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    zooms[zooms < _SINGULAR_TOL] = 1.0
    normalised = rzs / zooms

    try:
        p, _s, qt = np.linalg.svd(normalised)
        rotation = p @ qt
    except np.linalg.LinAlgError:                      # pragma: no cover - defensive
        rotation = normalised

    ornt = np.full((3, 2), np.nan)
    remaining = rotation.copy()
    for voxel_axis in range(3):
        column = remaining[:, voxel_axis]
        if np.all(np.abs(column) < _SINGULAR_TOL):
            continue
        world_axis = int(np.argmax(np.abs(column)))
        ornt[voxel_axis] = (world_axis, 1.0 if column[world_axis] > 0 else -1.0)
        # Claim this world axis so a later voxel axis cannot take it too.
        remaining[world_axis, :] = 0.0
    return ornt


def orientation_to_axcodes(ornt: np.ndarray) -> tuple[str, ...]:
    """Turn an orientation array into letter codes, e.g. ``('R', 'A', 'S')``.

    A voxel axis that could not be assigned (degenerate affine) becomes ``'?'``
    rather than raising — orientation reporting is diagnostic, and a caller inspecting
    a broken study needs the partial answer more than an exception.
    """
    codes: list[str] = []
    for world_axis, direction in np.asarray(ornt, dtype=float):
        if np.isnan(world_axis) or np.isnan(direction):
            codes.append("?")
            continue
        codes.append(_AXIS_LABELS[int(world_axis)][1 if direction > 0 else 0])
    return tuple(codes)


def axis_codes(affine: np.ndarray) -> tuple[str, ...]:
    """Orientation letter codes for ``affine``. ``('R', 'A', 'S')`` is canonical."""
    return orientation_to_axcodes(io_orientation(affine))


def apply_orientation(array: np.ndarray, ornt: np.ndarray) -> np.ndarray:
    """Flip and transpose ``array`` according to ``ornt``.

    Pure index manipulation: no interpolation, no intensity change, and — because
    numpy flips and transposes are views — no copy until something writes.
    """
    result = np.asarray(array)
    ornt = np.asarray(ornt, dtype=float)
    for voxel_axis, direction in enumerate(ornt[:, 1]):
        if direction == -1:
            result = np.flip(result, axis=voxel_axis)
    transpose = np.arange(result.ndim)
    transpose[:ornt.shape[0]] = np.argsort(ornt[:, 0])
    return result.transpose(transpose)


def inverse_orientation_affine(ornt: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Affine that maps *reoriented* voxel indices back to the original ones.

    Post-multiplying the original affine by this keeps world coordinates fixed::

        new_affine = affine @ inverse_orientation_affine(ornt, original_shape)

    which is the invariant :func:`to_canonical` relies on and the tests assert.
    """
    ornt = np.asarray(ornt, dtype=float)
    n = ornt.shape[0]
    dims = np.asarray(shape, dtype=float)[:n]

    undo_reorder = np.eye(n + 1)[list(ornt[:, 0].astype(int)) + [n], :]
    undo_flip = np.diag(list(ornt[:, 1]) + [1.0])
    centre = -(dims - 1) / 2.0
    undo_flip[:n, n] = (ornt[:, 1] * centre) - centre
    return undo_flip @ undo_reorder


def to_canonical(array: np.ndarray, affine: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], bool]:
    """Reorient ``array`` to the closest RAS orientation.

    Returns ``(array, affine, original_axcodes, changed)``. ``changed`` is ``False``
    when the volume was already canonical, which lets the pipeline record a ``no_op``
    step instead of pretending work happened.

    "Closest RAS" is exact only for axis-aligned acquisitions. An oblique series is
    reordered to the nearest axis permutation; its residual obliquity stays in the
    affine, where it belongs, and is reported by
    :attr:`VoxelGeometry.obliquity_degrees`. Removing it would require resampling,
    which is a different stage with a different cost and a different reviewer.
    """
    ornt = io_orientation(affine)
    if np.isnan(ornt).any():
        raise ValueError("affine is degenerate; cannot determine orientation")

    original = orientation_to_axcodes(ornt)
    if np.array_equal(ornt, CANONICAL_ORNT):
        return array, np.asarray(affine, dtype=float), original, False

    reoriented = apply_orientation(array, ornt)
    new_affine = np.asarray(affine, dtype=float) @ inverse_orientation_affine(
        ornt, array.shape)
    return reoriented, new_affine, original, True


# --------------------------------------------------------------------------- #
# Derived measurements
# --------------------------------------------------------------------------- #
def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel size in mm along each voxel axis — the column norms of the affine."""
    rzs = np.asarray(affine, dtype=float)[:3, :3]
    return tuple(float(v) for v in np.sqrt(np.sum(rzs * rzs, axis=0)))  # type: ignore[return-value]


def direction_matrix(affine: np.ndarray) -> np.ndarray:
    """Unit direction cosines of the three voxel axes, as columns."""
    rzs = np.asarray(affine, dtype=float)[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    zooms[zooms < _SINGULAR_TOL] = 1.0
    return rzs / zooms


def slice_normal(affine: np.ndarray) -> np.ndarray:
    """Unit vector along the third voxel axis — the through-plane direction."""
    return direction_matrix(affine)[:, 2]


def obliquity_degrees(affine: np.ndarray) -> float:
    """Angle between the slice normal and the nearest cardinal world axis.

    ``0`` for a perfectly axial/coronal/sagittal acquisition; grows as the plane is
    tilted. Reported rather than corrected — see :func:`to_canonical`.
    """
    normal = slice_normal(affine)
    best = float(np.max(np.abs(normal)))
    return float(math.degrees(math.acos(min(1.0, max(0.0, best)))))


def anatomical_plane(affine: np.ndarray, oblique_threshold_deg: float = 20.0
                     ) -> AnatomicalPlane:
    """Acquisition plane implied by the slice normal.

    Derived from geometry rather than from a description string, because
    ``SeriesDescription`` says "AX T2" on plenty of series that were reformatted or
    acquired obliquely.
    """
    normal = slice_normal(affine)
    if not np.isfinite(normal).all() or float(np.linalg.norm(normal)) < _SINGULAR_TOL:
        return AnatomicalPlane.UNKNOWN
    if obliquity_degrees(affine) > oblique_threshold_deg:
        return AnatomicalPlane.OBLIQUE
    dominant = int(np.argmax(np.abs(normal)))
    return (AnatomicalPlane.SAGITTAL, AnatomicalPlane.CORONAL,
            AnatomicalPlane.AXIAL)[dominant]


def voxel_to_world(affine: np.ndarray, indices: Iterable[float]) -> np.ndarray:
    """Map one voxel index triple to world millimetres."""
    idx = np.asarray(list(indices), dtype=float).reshape(3)
    return (np.asarray(affine, dtype=float) @ np.append(idx, 1.0))[:3]


def world_centre(affine: np.ndarray, shape: Sequence[int]) -> tuple[float, float, float]:
    """World coordinates of the volume's geometric centre, in mm."""
    centre = (np.asarray(shape, dtype=float)[:3] - 1.0) / 2.0
    return tuple(float(v) for v in voxel_to_world(affine, centre))  # type: ignore[return-value]


def field_of_view_mm(affine: np.ndarray, shape: Sequence[int]
                     ) -> tuple[float, float, float]:
    """Physical extent along each voxel axis, in mm."""
    spacing = spacing_from_affine(affine)
    return tuple(float(n * s) for n, s in zip(shape[:3], spacing))  # type: ignore[return-value]


def is_degenerate(affine: np.ndarray) -> bool:
    """True when the affine cannot describe a volume (zero or collinear axes)."""
    aff = np.asarray(affine, dtype=float)
    if not np.isfinite(aff).all():
        return True
    return abs(float(np.linalg.det(aff[:3, :3]))) < _SINGULAR_TOL


# --------------------------------------------------------------------------- #
# Value type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VoxelGeometry:
    """Everything about where a volume's voxels sit in the world.

    Frozen, and always carried *with* the array it describes. Every derived quantity
    is computed from the affine on access rather than stored, so a geometry object
    cannot go stale relative to its own affine — the failure mode where ``spacing``
    says 1 mm and the affine says 0.5 mm simply cannot occur.
    """

    affine: np.ndarray
    shape: tuple[int, int, int]
    space: str = "RAS"

    def __post_init__(self) -> None:
        aff = np.asarray(self.affine, dtype=float)
        if aff.shape != (4, 4):
            raise ValueError(f"affine must be 4x4, got {aff.shape}")
        object.__setattr__(self, "affine", aff)
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))

    # -- derived ------------------------------------------------------------ #
    @property
    def spacing(self) -> tuple[float, float, float]:
        return spacing_from_affine(self.affine)

    @property
    def axis_codes(self) -> tuple[str, ...]:
        return axis_codes(self.affine)

    @property
    def orientation(self) -> str:
        """Orientation as a compact string, e.g. ``'RAS'``."""
        return "".join(self.axis_codes)

    @property
    def is_canonical(self) -> bool:
        return self.orientation == "RAS"

    @property
    def direction(self) -> np.ndarray:
        return direction_matrix(self.affine)

    @property
    def plane(self) -> AnatomicalPlane:
        return anatomical_plane(self.affine)

    @property
    def obliquity_deg(self) -> float:
        return obliquity_degrees(self.affine)

    @property
    def field_of_view_mm(self) -> tuple[float, float, float]:
        return field_of_view_mm(self.affine, self.shape)

    @property
    def centre_world_mm(self) -> tuple[float, float, float]:
        return world_centre(self.affine, self.shape)

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return float(sx * sy * sz)

    @property
    def anisotropy(self) -> float:
        """Ratio of largest to smallest voxel edge. ``1.0`` is isotropic.

        The single most useful number for deciding whether a volume can be fed to a
        3D model: a 0.5 x 0.5 x 5 mm clinical T2 has anisotropy 10 and needs
        resampling that a 1 mm MPRAGE does not.
        """
        spacing = [s for s in self.spacing if s > _SINGULAR_TOL]
        return float(max(spacing) / min(spacing)) if spacing else float("nan")

    @property
    def degenerate(self) -> bool:
        return is_degenerate(self.affine)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary. The affine is included in full — a geometry summary
        without its affine is not reproducible."""
        return {
            "space": self.space,
            "shape": list(self.shape),
            "spacing_mm": [round(v, 6) for v in self.spacing],
            "orientation": self.orientation,
            "plane": self.plane.value,
            "obliquity_deg": round(self.obliquity_deg, 3),
            "field_of_view_mm": [round(v, 3) for v in self.field_of_view_mm],
            "centre_world_mm": [round(v, 3) for v in self.centre_world_mm],
            "voxel_volume_mm3": round(self.voxel_volume_mm3, 6),
            "anisotropy": round(self.anisotropy, 4),
            "affine": [[round(float(v), 8) for v in row] for row in self.affine],
        }
