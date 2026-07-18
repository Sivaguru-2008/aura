"""Tests for Step 4 — patient timeline construction."""
from __future__ import annotations

import pytest

from mimic.config import get_mimic_paths
from mimic.loaders import MimicCxrLoader, PatientRecord
from mimic.timeline import StudyEvent, build_timeline, _study_of
from schemas.clinical import Diagnosis, Finding

PATHS = get_mimic_paths()
HAS_DATA = PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def _rec() -> PatientRecord:
    return PatientRecord(
        subject_id=1,
        images=[
            "files/p10/p1/s200/a.jpg",   # deliberately out of order
            "files/p10/p1/s100/b.jpg",
            "files/p10/p1/s100/c.jpg",
        ],
        images_by_view={"AP": ["files/p10/p1/s200/a.jpg", "files/p10/p1/s100/b.jpg"],
                        "Lateral": ["files/p10/p1/s100/c.jpg"]},
        reports=["Small pneumothorax.", "No acute cardiopulmonary process."],
        reports_aug=[],
    )


def test_study_of_parses_id():
    assert _study_of("files/p10/p1/s5099/img.jpg") == "s5099"
    assert _study_of("no-study-here.jpg") is None


def test_timeline_orders_by_study_id():
    tl = build_timeline(_rec())
    assert [e.study_id for e in tl.events] == ["s100", "s200"]  # sorted, not input order
    assert tl.n_studies == 2
    assert not tl.misaligned


def test_events_group_images_and_views():
    tl = build_timeline(_rec())
    s100 = tl.events[0]
    assert isinstance(s100, StudyEvent)
    assert set(s100.images) == {"files/p10/p1/s100/b.jpg", "files/p10/p1/s100/c.jpg"}
    assert set(s100.views) == {"AP", "Lateral"}


def test_reports_pair_by_position_and_label():
    tl = build_timeline(_rec())
    # s100 -> report[0] "Small pneumothorax." ; s200 -> report[1] normal
    assert tl.events[0].diagnosis == Diagnosis.PNEUMOTHORAX
    assert tl.events[1].diagnosis == Diagnosis.NORMAL


def test_finding_first_seen():
    rec = PatientRecord(
        subject_id=2,
        images=["files/p/x/s1/a.jpg", "files/p/x/s2/b.jpg"],
        images_by_view={"AP": ["files/p/x/s1/a.jpg", "files/p/x/s2/b.jpg"]},
        reports=["Clear lungs.", "Moderate pleural effusion."],
        reports_aug=[],
    )
    tl = build_timeline(rec)
    assert tl.finding_first_seen(Finding.EFFUSION) == 1
    assert tl.finding_first_seen(Finding.PNEUMOTHORAX) is None


def test_misalignment_flagged_not_crashed():
    rec = PatientRecord(
        subject_id=3,
        images=["files/p/x/s1/a.jpg", "files/p/x/s2/b.jpg"],
        images_by_view={"AP": ["files/p/x/s1/a.jpg", "files/p/x/s2/b.jpg"]},
        reports=["Only one report."],           # 2 studies, 1 report
        reports_aug=[],
    )
    tl = build_timeline(rec)
    assert tl.misaligned is True
    assert tl.events[1].report == ""            # unmatched study -> empty report


@needs_data
def test_real_timelines_are_chronological():
    ld = MimicCxrLoader("validate")
    n = 0
    for rec in ld.iter_records(limit=100):
        tl = build_timeline(rec)
        ids = [e.study_id for e in tl.events]
        assert ids == sorted(ids)               # chronological by construction
        assert all(e.t_index == i for i, e in enumerate(tl.events))
        n += 1
    assert n == 100
