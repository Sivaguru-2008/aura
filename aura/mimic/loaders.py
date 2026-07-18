"""Step 2 — Production data loaders for the real MIMIC-CXR corpus.

The synthetic world (``ml/data.py``) built patients in memory with ``numpy``.
This module loads **real** patients from the MIMIC-CXR aug CSVs, one longitudinal
record per ``subject_id``, and hands downstream code clean, typed objects.

Design requirements (met):
    * lazy loading        — ``iter_records`` streams; nothing is held whole
    * chunk loading       — CSVs read via ``pandas`` ``chunksize``
    * memory efficient    — ``dtype=str`` + per-chunk parse, junk columns dropped
    * logging             — module logger, progress + reject counts
    * type hints          — everywhere; ``PatientRecord`` is a typed dataclass
    * validation          — every row validated; bad rows counted, not crashed on
    * schema checking      — automatic header check before a single row is parsed

The loader also enforces the Step-1 finding that ~30% of referenced images are
absent: by default it filters image paths to those that exist on disk and drops
patients left with zero images (configurable via ``require_images``).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from mimic.config import MimicPaths, get_mimic_paths
from mimic.parsing import safe_str_list

log = logging.getLogger("mimic.loaders")

CHUNK_ROWS = 512

# View buckets present in the CSV, normalized to AURA's canonical view names.
_VIEW_COLUMNS = ("AP", "PA", "Lateral")


class SchemaError(ValueError):
    """Raised when a CSV header does not match the expected MIMIC-CXR schema."""


@dataclass
class PatientRecord:
    """One real MIMIC-CXR patient — the unit Step 5's Patient Object wraps.

    Image path lists are **relative** to ``images_root``; when ``require_images``
    filtering is on, only on-disk paths survive. ``reports`` holds the real
    radiology text; ``reports_aug`` the parallel paraphrase.
    """

    subject_id: int
    images: list[str] = field(default_factory=list)          # all views, on-disk-filtered
    images_by_view: dict[str, list[str]] = field(default_factory=dict)
    views: list[str] = field(default_factory=list)           # raw per-image view labels
    reports: list[str] = field(default_factory=list)
    reports_aug: list[str] = field(default_factory=list)
    # Bookkeeping for transparency / QC.
    n_images_referenced: int = 0                             # before on-disk filtering
    n_images_present: int = 0                                # after filtering

    @property
    def n_reports(self) -> int:
        return len(self.reports)

    @property
    def has_images(self) -> bool:
        return self.n_images_present > 0


@dataclass
class LoadStats:
    """Counters describing what a load pass did — surfaced for QC and tests."""

    rows_read: int = 0
    records_yielded: int = 0
    dropped_no_images: int = 0
    dropped_bad_subject_id: int = 0
    images_referenced: int = 0
    images_present: int = 0
    seconds: float = 0.0


class MimicCxrLoader:
    """Lazy, chunked, schema-validated loader for one MIMIC-CXR split."""

    def __init__(
        self,
        split: str = "train",
        paths: Optional[MimicPaths] = None,
        *,
        require_images: bool = True,
        filter_to_disk: bool = True,
        chunk_rows: int = CHUNK_ROWS,
    ) -> None:
        """
        Args:
            split: "train" or "validate".
            paths: resolved :class:`MimicPaths` (defaults to :func:`get_mimic_paths`).
            require_images: drop patients left with zero on-disk images.
            filter_to_disk: keep only image paths that resolve to a real file.
            chunk_rows: pandas read chunk size.
        """
        self.paths = paths or get_mimic_paths()
        self.split = split
        self.require_images = require_images
        self.filter_to_disk = filter_to_disk
        self.chunk_rows = chunk_rows
        self.csv_path = self._csv_for_split(split)
        self.stats = LoadStats()

    # ------------------------------------------------------------------ #
    # Setup / schema
    # ------------------------------------------------------------------ #
    def _csv_for_split(self, split: str) -> Path:
        if split == "train":
            return self.paths.train_csv
        if split in ("validate", "val", "validation"):
            return self.paths.validate_csv
        raise ValueError(f"unknown split {split!r} (expected 'train' or 'validate')")

    def validate_schema(self) -> list[str]:
        """Read only the header and assert it matches the expected columns.

        Returns the actual column list on success; raises :class:`SchemaError`
        with a precise diff on mismatch. Cheap — reads a single row.
        """
        if not self.csv_path.is_file():
            raise SchemaError(f"CSV not found: {self.csv_path}")
        head = pd.read_csv(self.csv_path, nrows=0)
        actual = list(head.columns)
        expected = list(self.paths.columns)
        if actual != expected:
            missing = [c for c in expected if c not in actual]
            extra = [c for c in actual if c not in expected]
            raise SchemaError(
                f"schema mismatch in {self.csv_path.name}: "
                f"missing={missing} extra={extra} (got {actual})"
            )
        log.info("schema OK for %s (%d columns)", self.csv_path.name, len(actual))
        return actual

    # ------------------------------------------------------------------ #
    # Row -> record
    # ------------------------------------------------------------------ #
    def _existing(self, rels: list[str]) -> list[str]:
        if not self.filter_to_disk:
            return rels
        return [r for r in rels if self.paths.resolve_image(r).is_file()]

    def _row_to_record(self, row: pd.Series) -> Optional[PatientRecord]:
        """Convert one raw CSV row into a validated :class:`PatientRecord`.

        Returns ``None`` (and bumps a drop counter) for rows that fail validation
        — an unparseable ``subject_id`` or, when ``require_images``, no on-disk image.
        """
        raw_sid = row.get("subject_id")
        try:
            sid = int(float(raw_sid))
        except (TypeError, ValueError):
            self.stats.dropped_bad_subject_id += 1
            return None

        all_imgs = safe_str_list(row.get("image"))
        self.stats.images_referenced += len(all_imgs)
        present = self._existing(all_imgs)
        self.stats.images_present += len(present)

        if self.require_images and not present:
            self.stats.dropped_no_images += 1
            return None

        present_set = set(present)
        by_view: dict[str, list[str]] = {}
        for col in _VIEW_COLUMNS:
            view_imgs = [r for r in safe_str_list(row.get(col)) if r in present_set]
            if view_imgs:
                by_view[col] = view_imgs

        rec = PatientRecord(
            subject_id=sid,
            images=present,
            images_by_view=by_view,
            views=safe_str_list(row.get("view")),
            reports=safe_str_list(row.get("text")),
            reports_aug=safe_str_list(row.get("text_augment")),
            n_images_referenced=len(all_imgs),
            n_images_present=len(present),
        )
        return rec

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def iter_records(self, limit: Optional[int] = None) -> Iterator[PatientRecord]:
        """Lazily stream validated :class:`PatientRecord`s.

        Reads the CSV in chunks; parses/validates each row; yields survivors.
        Memory stays flat regardless of file size. Resets ``self.stats``.
        """
        self.validate_schema()
        self.stats = LoadStats()
        t0 = time.perf_counter()
        reader = pd.read_csv(
            self.csv_path,
            chunksize=self.chunk_rows,
            dtype=str,
            usecols=[c for c in self.paths.columns if c not in self.paths.index_junk_columns],
        )
        for chunk in reader:
            for _, row in chunk.iterrows():
                self.stats.rows_read += 1
                rec = self._row_to_record(row)
                if rec is None:
                    continue
                self.stats.records_yielded += 1
                yield rec
                if limit is not None and self.stats.records_yielded >= limit:
                    self.stats.seconds = time.perf_counter() - t0
                    log.info("iter_records stopped at limit=%d", limit)
                    return
        self.stats.seconds = time.perf_counter() - t0
        log.info(
            "%s: read=%d yielded=%d dropped(no_img=%d bad_id=%d) imgs=%d/%d present in %.1fs",
            self.split, self.stats.rows_read, self.stats.records_yielded,
            self.stats.dropped_no_images, self.stats.dropped_bad_subject_id,
            self.stats.images_present, self.stats.images_referenced, self.stats.seconds,
        )

    def to_frame(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Materialize records into a tidy, one-row-per-patient DataFrame.

        Columns: subject_id, n_images_present, n_images_referenced, n_reports,
        views_available, images, reports. Use only when you need the whole split
        in memory (e.g. building splits in Step 7); prefer ``iter_records`` otherwise.
        """
        rows = []
        for rec in self.iter_records(limit=limit):
            rows.append(
                {
                    "subject_id": rec.subject_id,
                    "n_images_present": rec.n_images_present,
                    "n_images_referenced": rec.n_images_referenced,
                    "n_reports": rec.n_reports,
                    "views_available": ",".join(sorted(rec.images_by_view.keys())),
                    "images": rec.images,
                    "reports": rec.reports,
                }
            )
        df = pd.DataFrame(rows)
        log.info("to_frame(%s): %d patients", self.split, len(df))
        return df

    # ------------------------------------------------------------------ #
    # Parquet cache (Step 14 hook, usable now)
    # ------------------------------------------------------------------ #
    def cache_path(self) -> Path:
        suffix = "all" if not self.require_images else "withimg"
        return self.paths.cache_dir / f"patients_{self.split}_{suffix}.parquet"

    def to_frame_cached(self, limit: Optional[int] = None, rebuild: bool = False) -> pd.DataFrame:
        """Return the patient frame, caching to parquet so repeat loads are instant.

        The 235 MB train CSV takes seconds to stream; the parquet cache turns
        subsequent loads into a sub-second memory-map read.
        """
        cp = self.cache_path()
        if cp.is_file() and not rebuild and limit is None:
            log.info("loading cached frame %s", cp)
            return pd.read_parquet(cp)
        df = self.to_frame(limit=limit)
        if limit is None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            # list-typed columns need pyarrow; fall back to dropping them if absent.
            try:
                df.to_parquet(cp, index=False)
                log.info("wrote parquet cache %s (%d rows)", cp, len(df))
            except (ImportError, ValueError) as e:  # pragma: no cover - env dependent
                log.warning("parquet cache skipped (%s); returning in-memory frame", e)
        return df
