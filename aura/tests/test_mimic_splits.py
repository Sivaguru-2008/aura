"""Tests for Step 7 — patient-level dataset splits (no leakage)."""
from __future__ import annotations

import pytest

from mimic.config import get_mimic_paths
from mimic.splits import (
    DEFAULT_TEST_FRAC,
    DatasetBuilder,
    _hash_frac,
    assign_train_test,
)

PATHS = get_mimic_paths()
HAS_DATA = PATHS.train_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def test_hash_is_deterministic_and_in_range():
    for sid in (1, 42, 10003502, 99999999):
        v = _hash_frac(sid)
        assert 0.0 <= v < 1.0
        assert _hash_frac(sid) == v          # stable across calls


def test_assign_is_deterministic():
    for sid in range(10_000_000, 10_000_050):
        assert assign_train_test(sid) == assign_train_test(sid)


def test_assign_fraction_is_approximately_target():
    n = 20_000
    test = sum(1 for sid in range(n) if assign_train_test(sid, 0.15) == "test")
    assert 0.13 < test / n < 0.17            # ~15% within tolerance


@needs_data
def test_build_has_zero_leakage_and_disjoint_ids():
    frames, stats = DatasetBuilder(test_frac=0.15).build(limit_train=3000)
    assert stats.leakage == 0
    ids = {k: set(df["subject_id"]) for k, df in frames.items() if len(df)}
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["test"].isdisjoint(ids["validation"])


@needs_data
def test_manifest_has_expected_columns():
    frames, _ = DatasetBuilder().build(limit_train=500)
    df = frames["train"]
    for c in ("subject_id", "split", "diagnosis", "n_studies", "n_images"):
        assert c in df.columns
    assert any(c.startswith("label_finding_") for c in df.columns)


@needs_data
def test_written_split_files_exist_and_are_disjoint():
    # Uses the full files written by the Step-7 build (if present).
    import pandas as pd
    out = DatasetBuilder().out_dir()
    files = {s: out / f"{s if s != 'validation' else 'validation'}.csv" for s in
             ("train", "validation", "test")}
    if not all(p.is_file() for p in files.values()):
        pytest.skip("split files not built yet")
    ids = {s: set(pd.read_csv(p)["subject_id"]) for s, p in files.items()}
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["test"].isdisjoint(ids["validation"])
