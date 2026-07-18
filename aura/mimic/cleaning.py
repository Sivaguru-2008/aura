"""Step 3 — Data cleaning for the real MIMIC-CXR corpus.

Applies to what this dataset actually contains (images + reports). Each cleaning
concern from the plan is handled or explicitly marked not-applicable:

    missing value handling  -> drop image-less patients (loader) + empty reports
    duplicate removal       -> dedupe repeated image paths and identical reports
    datetime parsing        -> N/A: the aug CSV has no timestamps; study order is
                               the only chronology signal (see timeline, Step 4)
    categorical encoding    -> view labels normalized to {AP, PA, Lateral} + counts
    ICD mapping             -> N/A: no diagnoses_icd table in MIMIC-CXR
    LOINC / unit / lab norm -> N/A: no labevents table in MIMIC-CXR
    invalid value removal   -> drop non-resolving images (loader), blank reports
    outlier detection       -> flag extreme image/report counts per patient

The value-add specific to this corpus: attaching structured labels extracted
from the report text (see :mod:`mimic.labeling`) so every real patient carries a
ground-truth :class:`Diagnosis` — the honest replacement for the synthetic one.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional

from mimic.labeling import ReportLabel, label_patient_reports
from mimic.loaders import MimicCxrLoader, PatientRecord
from schemas.clinical import Diagnosis, Finding

log = logging.getLogger("mimic.cleaning")

# Raw view strings seen in the data -> AURA canonical buckets.
_VIEW_NORMALIZE = {
    "ap": "AP",
    "pa": "PA",
    "lateral": "Lateral",
    "ll": "Lateral",
    "lao": "Lateral",
    "rao": "Lateral",
}
_WS = re.compile(r"\s+")
_PLACEHOLDER = re.compile(r"_{2,}")

# Outlier thresholds (per patient); tunable, deliberately generous.
MAX_IMAGES = 400
MAX_REPORTS = 300


@dataclass
class CleanedPatient:
    """A cleaned, labeled patient — the input Step 5 wraps into the Patient Object."""

    subject_id: int
    images: list[str]                                # deduped, on-disk
    images_by_view: dict[str, list[str]]             # normalized view buckets
    reports: list[str]                               # deduped, whitespace-cleaned
    label: ReportLabel                               # patient-level structured label
    per_report_labels: list[ReportLabel] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    @property
    def diagnosis(self) -> Diagnosis:
        return self.label.diagnosis

    @property
    def view_counts(self) -> dict[str, int]:
        return {v: len(p) for v, p in self.images_by_view.items()}

    @property
    def positive_findings(self) -> list[Finding]:
        return self.label.positive_findings


@dataclass
class CleanStats:
    patients_in: int = 0
    patients_out: int = 0
    dup_images_removed: int = 0
    dup_reports_removed: int = 0
    blank_reports_removed: int = 0
    flagged_outliers: int = 0
    diagnosis_counts: dict[str, int] = field(default_factory=dict)


def _dedup(items: list[str]) -> tuple[list[str], int]:
    """Order-preserving de-duplication; returns (unique, n_removed)."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out, len(items) - len(out)


def _clean_text(text: str) -> str:
    """Collapse whitespace and drop de-identification placeholders."""
    return _WS.sub(" ", _PLACEHOLDER.sub(" ", text)).strip()


def _normalize_views(record: PatientRecord) -> dict[str, list[str]]:
    """Re-bucket the record's per-view image lists into canonical AURA views."""
    buckets: dict[str, list[str]] = {}
    for raw_view, imgs in record.images_by_view.items():
        canon = _VIEW_NORMALIZE.get(raw_view.lower(), raw_view)
        buckets.setdefault(canon, [])
        for p in imgs:
            if p not in buckets[canon]:
                buckets[canon].append(p)
    return buckets


def clean_record(record: PatientRecord) -> tuple[CleanedPatient, CleanStats]:
    """Clean one loaded :class:`PatientRecord` into a labeled :class:`CleanedPatient`."""
    st = CleanStats(patients_in=1)

    images, dup_i = _dedup(record.images)
    st.dup_images_removed += dup_i

    cleaned_reports = [_clean_text(r) for r in record.reports]
    non_blank = [r for r in cleaned_reports if r]
    st.blank_reports_removed += len(cleaned_reports) - len(non_blank)
    reports, dup_r = _dedup(non_blank)
    st.dup_reports_removed += dup_r

    label, per = label_patient_reports(reports)

    flags: list[str] = []
    if len(images) > MAX_IMAGES:
        flags.append(f"image_count_outlier({len(images)})")
    if len(reports) > MAX_REPORTS:
        flags.append(f"report_count_outlier({len(reports)})")
    if not reports:
        flags.append("no_reports")            # images present but no text to label
    if flags:
        st.flagged_outliers += 1

    cp = CleanedPatient(
        subject_id=record.subject_id,
        images=images,
        images_by_view=_normalize_views(record),
        reports=reports,
        label=label,
        per_report_labels=per,
        quality_flags=flags,
    )
    st.patients_out = 1
    st.diagnosis_counts[label.diagnosis.value] = 1
    return cp, st


class DataCleaner:
    """Streams a loader's records through cleaning + labeling, aggregating stats."""

    def __init__(self, loader: MimicCxrLoader) -> None:
        self.loader = loader
        self.stats = CleanStats()

    def iter_clean(self, limit: Optional[int] = None) -> Iterator[CleanedPatient]:
        self.stats = CleanStats()
        for rec in self.loader.iter_records(limit=limit):
            cp, st = clean_record(rec)
            # fold per-record stats into the running aggregate
            self.stats.patients_in += st.patients_in
            self.stats.patients_out += st.patients_out
            self.stats.dup_images_removed += st.dup_images_removed
            self.stats.dup_reports_removed += st.dup_reports_removed
            self.stats.blank_reports_removed += st.blank_reports_removed
            self.stats.flagged_outliers += st.flagged_outliers
            for k, v in st.diagnosis_counts.items():
                self.stats.diagnosis_counts[k] = self.stats.diagnosis_counts.get(k, 0) + v
            yield cp
        log.info(
            "cleaned %d patients | dup_imgs=%d dup_reports=%d blank=%d outliers=%d | dx=%s",
            self.stats.patients_out, self.stats.dup_images_removed,
            self.stats.dup_reports_removed, self.stats.blank_reports_removed,
            self.stats.flagged_outliers, self.stats.diagnosis_counts,
        )
