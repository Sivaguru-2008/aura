"""Tests for Step 6 — feature engineering (leakage-free, schema-complete)."""
from __future__ import annotations

import numpy as np
import pytest

from mimic.config import get_mimic_paths
from mimic.features import (
    CLINICAL_NA_FEATURES,
    FeatureEngineer,
    _clinical_na_features,
    _image_features,
    feature_names,
)

PATHS = get_mimic_paths()
HAS_DATA = PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def test_feature_names_have_no_label_columns():
    assert all(not n.startswith("label_") for n in feature_names())
    assert "subject_id" not in feature_names()


def test_clinical_na_features_flagged_missing():
    f = _clinical_na_features()
    for name in CLINICAL_NA_FEATURES:
        assert f[name] == 0.0
        assert f[f"{name}_missing"] == 1.0


def test_image_features_on_synthetic_array():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    feats = _image_features(img)
    assert "heart_ratio" in feats and "img_mean" in feats
    assert 0.0 <= feats["img_mean"] <= 1.0
    assert all(np.isfinite(v) for v in feats.values())


@needs_data
def test_build_frame_is_leakage_free_and_complete():
    df = FeatureEngineer().build_frame("validate", limit=60)
    feat_cols = feature_names()
    # every declared feature column is present
    for c in feat_cols:
        assert c in df.columns, c
    # no feature column is label-derived
    assert not any(c.startswith("label_") for c in feat_cols)
    # labels present
    assert "label_diagnosis" in df.columns
    assert any(c.startswith("label_finding_") for c in df.columns)
    # clinical-NA indicators all missing on real data too
    for name in CLINICAL_NA_FEATURES:
        assert df[f"{name}_missing"].eq(1.0).all()


@needs_data
def test_image_features_vary_across_patients():
    df = FeatureEngineer().build_frame("validate", limit=80)
    # real images -> non-degenerate image features
    assert df["heart_ratio"].std() > 0.05
    assert df["img_std"].std() > 0.0
