"""Tests for the MIMIC-CXR integration — Step 1 (verify) + Step 2 (loaders).

Pure-unit tests (parsing, schema logic) always run. Tests that need the real
corpus are skipped automatically when it is not mounted, so CI stays green on
machines without the ~50 GB dataset.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from mimic.config import get_mimic_paths
from mimic.loaders import MimicCxrLoader, PatientRecord, SchemaError
from mimic.parsing import safe_list, safe_str_list

PATHS = get_mimic_paths()
HAS_DATA = PATHS.train_csv.is_file() and PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


# --------------------------------------------------------------------------- #
# Pure unit tests — no data required
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("['a', 'b']", ["a", "b"]),
        ("[]", []),
        ("", []),
        ("nan", []),
        ("None", []),
        (None, []),
        (float("nan"), []),
        (["x"], ["x"]),
        ("not a list", []),          # malformed -> [] not crash
    ],
)
def test_safe_list(raw, expected):
    assert safe_list(raw) == expected


def test_safe_str_list_strips_and_drops_blanks():
    assert safe_str_list("['  a ', '', 'b']") == ["a", "b"]


def test_schema_error_on_bad_header(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"subject_id": [1], "foo": [2]}).to_csv(bad, index=False)
    paths = dataclasses.replace(PATHS, train_csv=bad)
    with pytest.raises(SchemaError):
        MimicCxrLoader("train", paths=paths).validate_schema()


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        MimicCxrLoader("nonsense")


def test_row_to_record_drops_bad_subject_id():
    ld = MimicCxrLoader("validate", require_images=False, filter_to_disk=False)
    row = pd.Series({"subject_id": "not-an-int", "image": "[]"})
    assert ld._row_to_record(row) is None
    assert ld.stats.dropped_bad_subject_id == 1


def test_row_to_record_builds_views_filtered_to_present():
    # filter_to_disk off so paths are trusted; require_images off so empty ok.
    ld = MimicCxrLoader("validate", require_images=False, filter_to_disk=False)
    row = pd.Series(
        {
            "subject_id": "42",
            "image": "['a.jpg', 'b.jpg', 'c.jpg']",
            "AP": "['a.jpg']",
            "PA": "['b.jpg']",
            "Lateral": "['c.jpg']",
            "view": "['AP', 'PA', 'LATERAL']",
            "text": "['report one']",
            "text_augment": "['report uno']",
        }
    )
    rec = ld._row_to_record(row)
    assert isinstance(rec, PatientRecord)
    assert rec.subject_id == 42
    assert rec.n_images_present == 3
    assert set(rec.images_by_view) == {"AP", "PA", "Lateral"}
    assert rec.n_reports == 1


# --------------------------------------------------------------------------- #
# Real-data tests — skipped when the corpus is absent
# --------------------------------------------------------------------------- #
@needs_data
def test_validate_schema_real():
    assert MimicCxrLoader("validate").validate_schema() == list(PATHS.columns)


@needs_data
def test_iter_records_filters_to_disk():
    ld = MimicCxrLoader("validate")
    recs = list(ld.iter_records())
    assert len(recs) > 0
    # every yielded image must resolve to a real file
    for rec in recs[:20]:
        assert rec.has_images
        for rel in rec.images:
            assert PATHS.resolve_image(rel).is_file()
    # image accounting is internally consistent
    assert ld.stats.images_present <= ld.stats.images_referenced
    assert ld.stats.records_yielded + ld.stats.dropped_no_images == ld.stats.rows_read


@needs_data
def test_require_images_toggle_changes_count():
    kept = sum(1 for _ in MimicCxrLoader("validate", require_images=True).iter_records())
    all_rows = sum(1 for _ in MimicCxrLoader("validate", require_images=False).iter_records())
    assert all_rows >= kept > 0


@needs_data
def test_lazy_limit_is_cheap():
    # limit must stop early, not materialize the whole split
    ld = MimicCxrLoader("train")
    recs = list(ld.iter_records(limit=5))
    assert len(recs) == 5
    assert ld.stats.rows_read < 5000  # nowhere near the 64k rows


@needs_data
def test_subject_ids_unique_across_records():
    ids = [r.subject_id for r in MimicCxrLoader("validate").iter_records()]
    assert len(ids) == len(set(ids))
