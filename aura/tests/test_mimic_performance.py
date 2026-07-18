"""Tests for Step 14 — performance utilities."""
from __future__ import annotations

import numpy as np
import pytest

from mimic.config import get_mimic_paths
from mimic.performance import MemmapCache, ParallelFeatureEngineer, gpu_standardize

PATHS = get_mimic_paths()
HAS_DATA = PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def test_memmap_cache_roundtrip(tmp_path):
    import dataclasses
    paths = dataclasses.replace(PATHS, cache_dir=tmp_path)
    mc = MemmapCache(paths)
    X = np.random.default_rng(0).random((100, 12)).astype(np.float32)
    mc.save_matrix("m", X)
    assert mc.exists("m")
    Xm = mc.load_matrix("m", mmap=True)
    assert isinstance(Xm, np.memmap)
    assert np.allclose(np.asarray(Xm), X)


def test_gpu_standardize_normalizes():
    X = np.random.default_rng(1).normal(5, 3, size=(200, 8))
    Xs = gpu_standardize(X)
    assert Xs.shape == X.shape
    assert abs(Xs.mean()) < 1e-4
    assert abs(Xs.std() - 1.0) < 0.1


@needs_data
def test_parallel_matches_serial_rows():
    pfe = ParallelFeatureEngineer()
    df1, _ = pfe.build_frame("validate", limit=60, workers=1)
    df8, _ = pfe.build_frame("validate", limit=60, workers=8)
    assert len(df1) == len(df8)
    # same set of patients regardless of worker count
    assert set(df1["subject_id"]) == set(df8["subject_id"])


@needs_data
def test_parallel_build_produces_valid_features():
    df, timing = ParallelFeatureEngineer().build_frame("validate", limit=40, workers=4)
    from mimic.features import feature_names
    for c in feature_names():
        assert c in df.columns
    assert timing.rows_per_sec > 0
