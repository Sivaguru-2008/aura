"""MRI Study Loader — module 1 of the foundation layer.

Turns a path into a :class:`LoadedStudy`: every series that could be read, each with
its voxels, its affine, its raw header, and its integrity report.

The loader owns the *study-level* view that no individual reader can have:

* **Mixed formats.** A folder holding both the original DICOM and a converted NIfTI
  of the same series would otherwise be loaded twice and analysed twice. When several
  formats are present the loader prefers DICOM — it is the only one that carries
  acquisition parameters, and losing those degrades sequence identification to
  free-text guessing — and records what it set aside.
* **Mixed studies.** More than one ``StudyInstanceUID`` under one root means the
  caller pointed at a patient folder, not a study folder. Loading proceeds, because
  refusing would be unhelpful, but the finding is recorded prominently: cross-study
  comparison of series that were never meant to be compared is a real error mode.
* **Empty or unreadable studies**, which become typed errors rather than an empty
  list — an empty result that looks like a successful load is how a silent failure
  reaches a clinician.

Readers are injected. The default set is DICOM + NIfTI + NRRD; a deployment that adds
a fourth format registers a reader and changes nothing else.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from backend.core.shared.logging import get_logger
from backend.foundation.mri.config import LoaderConfig
from backend.foundation.mri.errors import (
    CorruptStudy,
    StudyNotFound,
    StudyValidationError,
    UnsupportedStudyFormat,
)
from backend.foundation.mri.io.base import RawSeries, StudyReader
from backend.foundation.mri.io.dicom_reader import DicomSeriesReader
from backend.foundation.mri.io.discovery import Discovery, discover
from backend.foundation.mri.io.nifti_reader import NiftiReader
from backend.foundation.mri.io.nrrd_reader import NrrdReader
from backend.foundation.mri.types import FileFormat

log = get_logger("foundation.mri.loader")

#: Preference order when one study directory holds several formats. DICOM first
#: because it is the only format carrying pulse-sequence parameters; HDF5 last because
#: it is a research redistribution format that carries no acquisition header at all.
FORMAT_PREFERENCE: tuple[FileFormat, ...] = (
    FileFormat.DICOM, FileFormat.NIFTI, FileFormat.NRRD, FileFormat.HDF5)


@dataclass
class LoadedStudy:
    """Every series read from one study path, before any processing."""

    study_id: str
    source_name: str
    series: tuple[RawSeries, ...]
    #: Format actually used. Others found are named in ``warnings``.
    source_format: FileFormat = FileFormat.UNKNOWN
    study_instance_uids: tuple[str, ...] = ()
    files_examined: int = 0
    load_seconds: float = 0.0
    warnings: tuple[str, ...] = ()
    #: Series that were found but could not be decoded into a volume, with the reason.
    #: Carried rather than logged and forgotten: a study that quietly loses its FLAIR
    #: is indistinguishable from a study that never had one.
    skipped_series: tuple[dict[str, Any], ...] = ()

    @property
    def series_count(self) -> int:
        return len(self.series)

    @property
    def complete(self) -> bool:
        """True when every series arrived structurally intact."""
        return all(s.integrity.complete for s in self.series)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "source_name": self.source_name,
            "source_format": self.source_format.value,
            "series_count": self.series_count,
            "study_instance_uids": list(self.study_instance_uids),
            "files_examined": self.files_examined,
            "load_seconds": round(self.load_seconds, 4),
            "complete": self.complete,
            "warnings": list(self.warnings),
            "skipped_series": [dict(s) for s in self.skipped_series],
            "series": [
                {"series_key": s.series_key, "source_name": s.source_name,
                 "shape": list(s.voxels.shape), "frames": s.frames,
                 "geometry": s.geometry.to_dict(),
                 "integrity": s.integrity.to_dict()}
                for s in self.series
            ],
        }


class MRIStudyLoader:
    """Reads a complete MRI study from disk. Readers are injected, never imported
    at the point of use."""

    def __init__(self, readers: Sequence[StudyReader] | None = None,
                 config: LoaderConfig | None = None) -> None:
        self._config = config or LoaderConfig()
        self._readers: tuple[StudyReader, ...] = tuple(readers) if readers is not None \
            else (DicomSeriesReader(self._config), NiftiReader(), NrrdReader())

    @property
    def readers(self) -> tuple[StudyReader, ...]:
        return self._readers

    @property
    def supported_formats(self) -> tuple[FileFormat, ...]:
        return tuple(r.file_format for r in self._readers)

    # ------------------------------------------------------------------ #
    def load(self, source: Path | str, *, study_id: str | None = None) -> LoadedStudy:
        """Read every series under ``source``.

        Args:
            source: a study directory, or a single volume file.
            study_id: caller-supplied identifier. Defaults to the path's own name,
                which keeps the identifier stable across re-runs without inventing one.

        Raises:
            StudyNotFound: the path is absent or holds no candidate files.
            UnsupportedStudyFormat: files were found but no reader claimed any.
            CorruptStudy: a reader claimed files and none decoded.
            StudyValidationError: series decoded but none could form a volume.
        """
        started = time.perf_counter()
        root = Path(source)
        discovery = discover(root, self._readers, self._config)

        if not discovery.claimed:
            raise UnsupportedStudyFormat(
                "no file under the study path is a DICOM, NIfTI, or NRRD volume",
                detail={"files_examined": discovery.files_examined,
                        "supported_formats": [f.value for f in self.supported_formats]})

        chosen, warnings = self._choose_format(discovery)
        reader = next(r for r in self._readers if r.file_format is chosen)
        skipped: list[dict[str, Any]] = []
        series = self._read(reader, discovery.claimed[chosen], skipped)

        warnings.extend(self._validate_study(series))
        if skipped:
            warnings.append(
                f"{len(skipped)} series were found but could not be decoded into a "
                "volume; they are listed in skipped_series")
        if discovery.truncated:
            warnings.append(
                f"discovery stopped after {self._config.max_files_scanned} files; the "
                "study may be incomplete and the path may not be a study root")
        if discovery.unclaimed:
            warnings.append(
                f"{len(discovery.unclaimed)} file(s) under the study path were not "
                "recognised by any reader and were ignored")

        study = LoadedStudy(
            study_id=study_id or root.stem or root.name,
            source_name=root.name,
            series=tuple(series),
            source_format=chosen,
            study_instance_uids=self._study_uids(series),
            files_examined=discovery.files_examined,
            load_seconds=time.perf_counter() - started,
            warnings=tuple(warnings),
            skipped_series=tuple(skipped),
        )
        log.info(
            "study loaded",
            extra={"context": {"study_id": study.study_id,
                               "format": chosen.value,
                               "series": study.series_count,
                               "complete": study.complete,
                               "seconds": round(study.load_seconds, 3)}},
        )
        return study

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _choose_format(discovery: Discovery) -> tuple[FileFormat, list[str]]:
        """Pick one format to read. Never read two — that duplicates series."""
        warnings: list[str] = []
        present = [f for f in FORMAT_PREFERENCE if f in discovery.claimed]
        if not present:                                  # pragma: no cover - guarded above
            raise UnsupportedStudyFormat("no supported format present")
        chosen = present[0]
        if len(present) > 1:
            others = ", ".join(f"{f.value} ({len(discovery.claimed[f])} file(s))"
                               for f in present[1:])
            warnings.append(
                f"the study path holds more than one format; {chosen.value} was read "
                f"and {others} ignored. Reading both would load the same anatomy twice")
        return chosen, warnings

    @staticmethod
    def _read(reader: StudyReader, paths: list[Path],
              issues: list[dict[str, Any]]) -> list[RawSeries]:
        series = reader.read(paths, issues=issues)
        if not series:
            raise CorruptStudy(
                f"{len(paths)} file(s) were identified as "
                f"{reader.file_format.value} but none could be decoded into a volume",
                detail={"format": reader.file_format.value, "files": len(paths)})
        return series

    @staticmethod
    def _study_uids(series: Sequence[RawSeries]) -> tuple[str, ...]:
        uids = {str(s.header.get("StudyInstanceUID")) for s in series
                if s.header.get("StudyInstanceUID")}
        return tuple(sorted(uids))

    def _validate_study(self, series: Sequence[RawSeries]) -> list[str]:
        """Study-level checks that no single series can perform."""
        warnings: list[str] = []
        uids = self._study_uids(series)
        if len(uids) > 1:
            warnings.append(
                f"{len(uids)} distinct StudyInstanceUIDs are present; this path holds "
                "more than one study. Series from different studies must not be "
                "compared or fused without an explicit registration step")

        incomplete = [s for s in series if not s.integrity.complete]
        if incomplete:
            warnings.append(
                f"{len(incomplete)} of {len(series)} series did not arrive intact "
                "(missing slices, duplicates, or inconsistent geometry); see each "
                "series' integrity report")

        frames_of_reference = {str(s.header.get("FrameOfReferenceUID"))
                               for s in series if s.header.get("FrameOfReferenceUID")}
        if len(series) > 1 and len(frames_of_reference) > 1:
            warnings.append(
                f"the {len(series)} series span {len(frames_of_reference)} frames of "
                "reference; they are not in a common physical space and must be "
                "registered before any voxel-wise comparison")
        return warnings


def load_study(source: Path | str, *, config: LoaderConfig | None = None,
               study_id: str | None = None) -> LoadedStudy:
    """Convenience wrapper for one-off loads with the default readers.

    The class is the supported entry point for anything that runs more than once —
    it lets readers and configuration be injected — but a script, a notebook, or a
    test that just wants a study should not have to construct one.
    """
    return MRIStudyLoader(config=config).load(source, study_id=study_id)


__all__ = [
    "LoadedStudy", "MRIStudyLoader", "load_study",
    "StudyNotFound", "StudyValidationError", "UnsupportedStudyFormat", "CorruptStudy",
]
