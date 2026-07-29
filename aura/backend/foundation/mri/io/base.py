"""What every reader returns, and what every reader promises.

:class:`RawSeries` is deliberately *raw*: the voxels are as decoded (after the
format's own declared scaling, and nothing else), the affine is as recorded, and the
orientation is whatever the scanner wrote. Standardisation happens later, in stages
that log what they did. Keeping the boundary here means a preprocessing bug can
always be localised to one side of it.

:class:`SeriesIntegrity` is the other half of a reader's output and is not optional.
"Did this study arrive complete?" cannot be answered after the fact — once slices are
stacked into an array, a missing slice is indistinguishable from a thicker one. The
reader is the only component that ever sees the evidence, so it is the component that
has to record it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ..geometry import VoxelGeometry
from ..types import FileFormat


@dataclass(frozen=True)
class SliceIssue:
    """One file that could not be used, and why.

    ``name`` is the file's *basename*, never its path: this travels into API
    responses, and a full path leaks the deployment's directory layout and often the
    patient's export folder name.
    """

    name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"file": self.name, "reason": self.reason}


@dataclass
class SeriesIntegrity:
    """Structural findings about one series as it was read.

    Everything here is *measured*, never assumed. ``missing_slices_estimated`` in
    particular is derived from the actual inter-slice distances: a gap that is a clean
    multiple of the series' own median spacing is a dropped slice, and the multiple
    says how many. That is a real inference from real geometry, not a guess.
    """

    files_found: int = 0
    slices_loaded: int = 0
    corrupt_files: tuple[SliceIssue, ...] = ()
    duplicate_positions: int = 0
    missing_slices_estimated: int = 0
    #: Measured inter-slice distances that deviated from the series median, in mm.
    irregular_gaps_mm: tuple[float, ...] = ()
    median_slice_spacing_mm: float | None = None
    spacing_consistent: bool = True
    geometry_consistent: bool = True
    warnings: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True when nothing was lost, dropped, or irregular."""
        return (not self.corrupt_files and self.missing_slices_estimated == 0
                and self.duplicate_positions == 0 and self.spacing_consistent
                and self.geometry_consistent)

    @property
    def loss_fraction(self) -> float:
        """Fraction of expected slices that are absent, in [0, 1]."""
        expected = self.slices_loaded + self.missing_slices_estimated + len(
            self.corrupt_files)
        return 0.0 if expected <= 0 else 1.0 - (self.slices_loaded / expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_found": self.files_found,
            "slices_loaded": self.slices_loaded,
            "corrupt_files": [issue.to_dict() for issue in self.corrupt_files],
            "duplicate_positions": self.duplicate_positions,
            "missing_slices_estimated": self.missing_slices_estimated,
            "irregular_gaps_mm": [round(g, 4) for g in self.irregular_gaps_mm],
            "median_slice_spacing_mm": (
                round(self.median_slice_spacing_mm, 4)
                if self.median_slice_spacing_mm is not None else None),
            "spacing_consistent": self.spacing_consistent,
            "geometry_consistent": self.geometry_consistent,
            "complete": self.complete,
            "loss_fraction": round(self.loss_fraction, 4),
            "warnings": list(self.warnings),
        }


@dataclass
class RawSeries:
    """One decoded series, before any standardisation.

    ``voxels`` is indexed ``[i, j, k]`` (see :mod:`aura.backend.foundation.mri.geometry`)
    and is 4D only when the source genuinely holds several volumes on one grid — a
    multi-b-value DWI, say. It is kept 4D rather than silently collapsed so the volume
    builder has to make an explicit, recorded choice about which frame to use.
    """

    series_key: str
    source_format: FileFormat
    voxels: np.ndarray
    geometry: VoxelGeometry
    #: Allowlisted header values, plain Python types only. Consumed by the metadata
    #: engine; never interpreted by the reader itself.
    header: dict[str, Any] = field(default_factory=dict)
    integrity: SeriesIntegrity = field(default_factory=SeriesIntegrity)
    #: Human-facing origin: a filename or a series description. Never a full path.
    source_name: str = ""
    #: Basenames of the files that contributed, for the audit record.
    contributing_files: tuple[str, ...] = ()

    @property
    def frames(self) -> int:
        """Number of volumes stored on this grid. ``1`` for a plain 3D series."""
        return int(self.voxels.shape[3]) if self.voxels.ndim == 4 else 1

    @property
    def is_multiframe(self) -> bool:
        return self.frames > 1


@runtime_checkable
class StudyReader(Protocol):
    """Contract for a format reader.

    ``can_read`` must be cheap — it is called on every candidate file during
    discovery, and a reader that decodes pixel data to answer it turns a directory
    scan into a full study load.
    """

    #: Format this reader produces.
    file_format: FileFormat

    def can_read(self, path: Path) -> bool:
        """Whether this reader claims ``path``. Must not raise, must stay cheap."""
        ...

    def read(self, paths: Sequence[Path], *,
             issues: list[dict[str, Any]] | None = None) -> list[RawSeries]:
        """Decode every series found among ``paths``.

        A file that fails to decode is recorded in the owning series' integrity
        report rather than aborting the read — one bad slice in a 200-slice series
        should cost that slice, not the study. Raise only when *nothing* is readable.

        ``issues`` is an out-parameter: whole series that were found but could not be
        turned into a volume are appended to it as ``{series_key, error, reason}``.
        An explicit list rather than reader state, because a dropped series must
        reach the caller — a study that silently loses a sequence looks like a study
        that never had it.
        """
        ...
