"""Step 4 — Longitudinal patient timeline.

The plan's canonical timeline (admissions -> transfers -> ICU -> labs -> vitals
-> meds -> diagnoses -> procedures -> notes) assumes MIMIC-IV's relational EHR.
This corpus is MIMIC-CXR: the only longitudinal signal is the sequence of
imaging **studies** and their radiology **reports**. So the timeline here is a
chronological sequence of ``StudyEvent``s — each a study with its images, views,
report, and extracted labels.

Chronology note (important, and honest): the aug CSV carries **no timestamps**.
Real MIMIC-CXR study dates live in ``mimic-cxr-2.0.0-metadata.csv`` (not present
here). We use ``study_id`` order as the chronology proxy — validated on this data
to be already monotonic and 1:1 with reports for 100% of patients. Event index is
the time axis; when true timestamps are dropped in later, only ``sort_key`` changes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from mimic.labeling import ReportLabel, label_report
from mimic.loaders import PatientRecord
from schemas.clinical import Diagnosis, Finding

log = logging.getLogger("mimic.timeline")

_STUDY_RE = re.compile(r"/(s\d+)/")


def _study_of(rel_path: str) -> Optional[str]:
    """Extract the ``sNNNN`` study id from an image path, or None."""
    m = _STUDY_RE.search("/" + rel_path.replace("\\", "/"))
    return m.group(1) if m else None


@dataclass
class StudyEvent:
    """One imaging study in a patient's chronological record."""

    t_index: int                       # 0-based position on the timeline
    study_id: str
    images: list[str] = field(default_factory=list)
    views: dict[str, list[str]] = field(default_factory=dict)   # view -> image paths
    report: str = ""
    label: ReportLabel = field(default_factory=ReportLabel)

    @property
    def diagnosis(self) -> Diagnosis:
        return self.label.diagnosis

    @property
    def positive_findings(self) -> list[Finding]:
        return self.label.positive_findings


@dataclass
class PatientTimeline:
    """A patient's studies in chronological order plus derived trajectory info."""

    subject_id: int
    events: list[StudyEvent] = field(default_factory=list)
    misaligned: bool = False           # study count != report count (best-effort paired)

    @property
    def n_studies(self) -> int:
        return len(self.events)

    @property
    def latest(self) -> Optional[StudyEvent]:
        return self.events[-1] if self.events else None

    @property
    def baseline(self) -> Optional[StudyEvent]:
        return self.events[0] if self.events else None

    @property
    def diagnosis_trajectory(self) -> list[Diagnosis]:
        return [e.diagnosis for e in self.events]

    def finding_first_seen(self, finding: Finding) -> Optional[int]:
        """Timeline index where a finding first turns positive (progression signal)."""
        for e in self.events:
            if e.label.findings.get(finding) == 1:
                return e.t_index
        return None


def _image_to_view(record: PatientRecord) -> dict[str, str]:
    """Invert the record's view buckets into an image-path -> view lookup."""
    m: dict[str, str] = {}
    for view, imgs in record.images_by_view.items():
        for p in imgs:
            m[p] = view
    return m


def build_timeline(record: PatientRecord) -> PatientTimeline:
    """Build a chronological :class:`PatientTimeline` from a raw loaded record.

    Built from the *raw* record (not the deduped clean bag) so per-study
    image/report alignment is preserved. Studies are ordered by ``study_id``;
    reports are paired by position (validated 1:1 on this corpus, but mismatches
    are handled gracefully and flagged).
    """
    img_view = _image_to_view(record)

    # Group on-disk images by study, preserving first-seen order.
    studies: dict[str, list[str]] = {}
    for p in record.images:
        sid = _study_of(p)
        if sid is None:
            continue
        studies.setdefault(sid, []).append(p)

    ordered_ids = sorted(studies.keys())        # study_id as chronology proxy
    reports = record.reports
    misaligned = len(ordered_ids) != len(reports)
    if misaligned:
        log.debug(
            "subject %s: %d studies vs %d reports (best-effort pairing)",
            record.subject_id, len(ordered_ids), len(reports),
        )

    events: list[StudyEvent] = []
    for i, sid in enumerate(ordered_ids):
        imgs = studies[sid]
        views: dict[str, list[str]] = {}
        for p in imgs:
            v = img_view.get(p, "Unknown")
            views.setdefault(v, []).append(p)
        report = reports[i] if i < len(reports) else ""
        events.append(
            StudyEvent(
                t_index=i,
                study_id=sid,
                images=imgs,
                views=views,
                report=report,
                label=label_report(report),
            )
        )

    return PatientTimeline(subject_id=record.subject_id, events=events, misaligned=misaligned)
