"""Find the readable files under a study path and decide which reader owns each.

Discovery is separated from loading because it is the only step that touches the
filesystem broadly, and it is where a mis-pointed root does damage. Two bounds are
enforced, both configurable: recursion depth and total files examined. A study
directory is a handful of nested folders; anything that blows through those limits is
a caller who passed the wrong path, and failing loudly is far better than silently
walking a drive.

Ownership is decided by asking each reader, in order, whether it claims a file —
never by suffix alone. DICOM is routinely exported with no extension, and a ``.nii``
that is actually gzipped, or a ``.nrrd`` that is truncated, must be identified by
content. Every reader's ``can_read`` is content-sniffing and cheap by contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from backend.core.shared.logging import get_logger
from backend.foundation.mri.config import LoaderConfig
from backend.foundation.mri.errors import StudyNotFound
from backend.foundation.mri.io.base import StudyReader
from backend.foundation.mri.types import FileFormat

log = get_logger("foundation.mri.io.discovery")

#: Names that are never medical images. Skipped before any reader is consulted, which
#: keeps a DICOMDIR index or an OS metadata file out of the corrupt-file report.
_IGNORED_NAMES = frozenset({
    "dicomdir", "dicomdir.", ".ds_store", "thumbs.db", "desktop.ini", "readme",
    "readme.txt", "readme.md", "version", "lockfile",
})
#: Suffixes that are never a volume. ``.gz`` is deliberately absent — ``.nii.gz`` is
#: the single most common way a brain MRI arrives, and it is rescued in
#: :func:`_is_ignorable` before this set is consulted.
_IGNORED_SUFFIXES = frozenset({
    ".txt", ".md", ".json", ".xml", ".csv", ".tsv", ".log", ".pdf", ".html",
    ".zip", ".7z", ".rar", ".png", ".jpg", ".jpeg", ".bmp", ".py", ".sh",
})


@dataclass
class Discovery:
    """What a scan found, per reader."""

    #: Reader file format -> the paths that reader claims.
    claimed: dict[FileFormat, list[Path]] = field(default_factory=dict)
    #: Files examined and claimed by nobody. Diagnostic only.
    unclaimed: list[Path] = field(default_factory=list)
    files_examined: int = 0
    truncated: bool = False

    @property
    def total_claimed(self) -> int:
        return sum(len(paths) for paths in self.claimed.values())

    @property
    def formats(self) -> tuple[FileFormat, ...]:
        return tuple(sorted(self.claimed, key=lambda f: f.value))


def iter_candidate_files(root: Path, config: LoaderConfig) -> tuple[list[Path], bool]:
    """Walk ``root`` and return candidate files plus whether the cap was hit.

    Bounded by ``max_scan_depth`` and ``max_files_scanned``. Symlinked directories are
    not followed: a study export with a self-referential link would otherwise loop,
    and no legitimate export needs one.
    """
    if root.is_file():
        return [root], False

    candidates: list[Path] = []
    truncated = False
    stack: list[tuple[Path, int]] = [(root, 0)]
    examined = 0

    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except (OSError, PermissionError) as exc:
            log.warning("cannot list a directory during study discovery",
                        extra={"context": {"directory": directory.name,
                                           "error": type(exc).__name__}})
            continue
        for entry in entries:
            if entry.is_symlink() and entry.is_dir():
                continue
            if entry.is_dir():
                if depth < config.max_scan_depth:
                    stack.append((entry, depth + 1))
                continue
            examined += 1
            if examined > config.max_files_scanned:
                truncated = True
                break
            if _is_ignorable(entry):
                continue
            candidates.append(entry)
        if truncated:
            break

    if truncated:
        log.warning(
            "study discovery hit the file cap; the path is probably not a study root",
            extra={"context": {"cap": config.max_files_scanned,
                               "root": root.name}})
    return candidates, truncated


def _is_ignorable(path: Path) -> bool:
    name = path.name.lower()
    if name in _IGNORED_NAMES:
        return True
    # ``.nii.gz`` must survive the suffix filter, so compound suffixes are checked
    # before the plain one.
    if name.endswith((".nii.gz", ".img.gz", ".nrrd.gz")):
        return False
    return path.suffix.lower() in _IGNORED_SUFFIXES


def discover(root: Path, readers: Sequence[StudyReader],
             config: LoaderConfig | None = None) -> Discovery:
    """Classify every candidate file under ``root`` by owning reader."""
    config = config or LoaderConfig()
    if not root.exists():
        raise StudyNotFound("the study path does not exist",
                            detail={"name": root.name})

    candidates, truncated = iter_candidate_files(root, config)
    if not candidates:
        raise StudyNotFound("the study path contains no candidate image files",
                            detail={"name": root.name})

    result = Discovery(files_examined=len(candidates), truncated=truncated)
    for path in candidates:
        owner = None
        for reader in readers:
            try:
                if reader.can_read(path):
                    owner = reader
                    break
            except Exception:                       # pragma: no cover - defensive
                log.exception("a reader raised while probing a file",
                              extra={"context": {"file": path.name}})
        if owner is None:
            result.unclaimed.append(path)
        else:
            result.claimed.setdefault(owner.file_format, []).append(path)

    log.info(
        "study discovery complete",
        extra={"context": {"examined": result.files_examined,
                           "claimed": result.total_claimed,
                           "formats": [f.value for f in result.formats],
                           "unclaimed": len(result.unclaimed)}},
    )
    return result
