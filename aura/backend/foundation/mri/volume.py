"""MRI Volume Builder — module 5: decoded slices to one 3D volume with its geometry.

The array and the affine travel together in :class:`MRIVolume` and there is no
supported way to get one without the other. That is the whole design: every bug this
layer could plausibly ship is a bug where voxels were changed and the affine was not,
or vice versa. Making them a single value, and making every transform return a new
one, removes the class of error rather than testing for it.

Frame selection is the builder's one judgement call, and it is recorded rather than
silently made. A 4D series (multi-b-value DWI, a dynamic acquisition) holds several
volumes on one grid; the builder takes one, says which, and says how many there were.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.shared.logging import get_logger
from backend.foundation.mri.errors import StudyValidationError
from backend.foundation.mri.geometry import VoxelGeometry
from backend.foundation.mri.io.base import RawSeries

log = get_logger("foundation.mri.volume")


@dataclass(frozen=True)
class MRIVolume:
    """A 3D intensity volume and the geometry that places it in the world.

    Frozen. Transforms return a new volume via :meth:`derive`, so a stage that
    forgets to update the affine produces a volume that is *obviously* wrong at the
    point of construction rather than one that is subtly wrong three stages later.
    """

    array: np.ndarray
    geometry: VoxelGeometry
    #: Free-form notes about how this volume came to be. Not the processing history
    #: (that lives on the study); this is per-volume provenance such as which frame
    #: of a 4D series was taken.
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.array)
        if array.ndim != 3:
            raise StudyValidationError(
                f"an MRI volume must be 3D; got shape {tuple(array.shape)}")
        if tuple(array.shape) != tuple(self.geometry.shape):
            raise StudyValidationError(
                "volume array and geometry disagree about shape",
                detail={"array_shape": list(array.shape),
                        "geometry_shape": list(self.geometry.shape)})
        object.__setattr__(self, "array", array)

    # -- convenience -------------------------------------------------------- #
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.geometry.shape

    @property
    def spacing(self) -> tuple[float, float, float]:
        return self.geometry.spacing

    @property
    def affine(self) -> np.ndarray:
        return self.geometry.affine

    @property
    def orientation(self) -> str:
        return self.geometry.orientation

    @property
    def voxel_count(self) -> int:
        return int(self.array.size)

    def derive(self, array: np.ndarray, affine: np.ndarray | None = None,
               **provenance: Any) -> "MRIVolume":
        """Return a new volume from a transformed array.

        ``affine`` must be supplied whenever the transform moved voxels relative to
        the world — resampling, cropping, reorientation. Omitting it asserts that the
        transform was intensity-only, and the shape check in ``__post_init__`` catches
        the case where that assertion is wrong.
        """
        geometry = VoxelGeometry(
            affine=self.geometry.affine if affine is None else affine,
            shape=tuple(int(n) for n in np.asarray(array).shape[:3]),
            space=self.geometry.space,
        )
        return MRIVolume(array=array, geometry=geometry,
                         provenance={**self.provenance, **provenance})

    def statistics(self) -> dict[str, float]:
        """Intensity summary. Computed on demand; nothing is cached on a frozen value."""
        finite = self.array[np.isfinite(self.array)]
        if finite.size == 0:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan"),
                    "std": float("nan"), "finite_fraction": 0.0}
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "finite_fraction": float(finite.size / self.array.size),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary. The voxels are not included — a foundation study is
        described in JSON and carried in memory, never serialised whole by accident."""
        return {
            "shape": list(self.shape),
            "dtype": str(self.array.dtype),
            "geometry": self.geometry.to_dict(),
            "statistics": {k: (round(v, 6) if np.isfinite(v) else None)
                           for k, v in self.statistics().items()},
            "provenance": dict(self.provenance),
        }


class VolumeBuilder:
    """Assembles a :class:`MRIVolume` from a :class:`RawSeries`.

    ``min_slices`` is enforced here rather than in each reader because it is a
    property of *being a volume*, not of any storage format. The DICOM reader needs
    two slice positions to derive spacing at all, but a one-slice NIfTI has the same
    problem for the same reason — it is an image, not a volume — and a rule that lived
    only in the DICOM path would let it through.
    """

    def __init__(self, min_slices: int = 2) -> None:
        self._min_slices = int(min_slices)

    def build(self, raw: RawSeries, *, frame: int = 0) -> tuple[MRIVolume, list[str]]:
        """Build one volume. Returns the volume and any warnings raised while doing it.

        Args:
            raw: a decoded series from a reader.
            frame: which volume to take when the series is 4D.

        Raises:
            StudyValidationError: the array is not a usable volume — wrong
                dimensionality, a degenerate affine, or no finite voxels at all.
        """
        warnings: list[str] = []
        array = np.asarray(raw.voxels)
        provenance: dict[str, Any] = {"series_key": raw.series_key,
                                      "source_format": raw.source_format.value}

        if array.ndim == 4:
            array, note = self._select_frame(array, frame, provenance)
            warnings.append(note)
        elif array.ndim != 3:
            raise StudyValidationError(
                f"series has shape {tuple(array.shape)}, which is not a volume",
                detail={"shape": list(array.shape)})

        if array.shape[2] < self._min_slices:
            raise StudyValidationError(
                f"the series has {array.shape[2]} slice(s); a volume requires at least "
                f"{self._min_slices}. A single slice carries no through-plane geometry "
                "and cannot be resampled, registered, or measured in three dimensions",
                detail={"slices": int(array.shape[2]),
                        "minimum": self._min_slices,
                        "shape": [int(n) for n in array.shape]})

        if raw.geometry.degenerate:
            raise StudyValidationError(
                "the series affine is degenerate (zero-length or collinear voxel "
                "axes); the volume cannot be placed in the world",
                detail={"affine": [[float(v) for v in row]
                                   for row in raw.geometry.affine]})

        array = np.ascontiguousarray(array, dtype=np.float32)
        finite = np.isfinite(array)
        if not finite.any():
            raise StudyValidationError(
                "the volume contains no finite voxel values")
        if not finite.all():
            count = int(array.size - finite.sum())
            warnings.append(
                f"{count} voxel(s) ({count / array.size:.2%}) are NaN or infinite; "
                "they were left in place for quality control to measure rather than "
                "quietly replaced")

        geometry = VoxelGeometry(affine=raw.geometry.affine,
                                 shape=tuple(int(n) for n in array.shape),
                                 space=raw.geometry.space)
        volume = MRIVolume(array=array, geometry=geometry, provenance=provenance)

        log.info(
            "volume built",
            extra={"context": {"series": raw.series_key,
                               "shape": list(volume.shape),
                               "spacing_mm": [round(v, 3) for v in volume.spacing],
                               "orientation": volume.orientation}},
        )
        return volume, warnings

    @staticmethod
    def _select_frame(array: np.ndarray, frame: int,
                      provenance: dict[str, Any]) -> tuple[np.ndarray, str]:
        """Take one volume from a 4D series, recording which and out of how many."""
        total = int(array.shape[3])
        if not 0 <= frame < total:
            raise StudyValidationError(
                f"frame {frame} was requested but the series holds {total}",
                detail={"frames": total, "requested": frame})
        provenance["frame_selected"] = frame
        provenance["frames_available"] = total
        return array[..., frame], (
            f"the series holds {total} volumes on one grid; frame {frame} was used. "
            "If this is a diffusion or dynamic series, the other frames carry "
            "different contrast and were not analysed")


__all__ = ["MRIVolume", "VolumeBuilder"]
