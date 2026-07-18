"""Regression tests for Steps 2/4/6 — evaluation, calibration, performance suites.

The evaluation/calibration tests need real MIMIC-CXR images on disk; they skip
cleanly when the corpus is absent (e.g. CI without the dataset mount).
"""
from __future__ import annotations

import numpy as np
import pytest


def _mimic_available() -> bool:
    try:
        from mimic.config import get_mimic_paths
        from mimic.parsing import safe_str_list
        import pandas as pd

        paths = get_mimic_paths()
        if not paths.validate_csv.is_file():
            return False
        df = pd.read_csv(paths.validate_csv, dtype=str, nrows=5)
        for _, row in df.iterrows():
            for rel in safe_str_list(row["image"]):
                if (paths.images_root / rel).is_file():
                    return True
        return False
    except Exception:
        return False


mimic = pytest.mark.skipif(not _mimic_available(), reason="MIMIC-CXR images not on disk")


def test_binary_ece_bounds():
    from ml.evaluation.clinical_eval import binary_ece

    y = np.array([0, 0, 1, 1])
    # Predicting the empirical rate (0.5) on a balanced set is perfectly calibrated.
    calibrated = np.full(4, 0.5)
    assert binary_ece(calibrated, y) == pytest.approx(0.0, abs=1e-9)
    # Confidently predicting 0.95 when every label is negative → ECE ≈ 0.95.
    confident_wrong = np.full(4, 0.95)
    y_neg = np.array([0, 0, 0, 0])
    assert binary_ece(confident_wrong, y_neg) == pytest.approx(0.95, abs=1e-9)
    # Always in [0, 1].
    rng = np.random.default_rng(0)
    assert 0.0 <= binary_ece(rng.random(50), (rng.random(50) < 0.5).astype(int)) <= 1.0


def test_conformal_evaluation_coverage():
    from ml.evaluation.vision_calibration import conformal_evaluation

    rng = np.random.default_rng(0)
    n, c = 200, 7
    Y = (rng.random((n, c)) < 0.3).astype(int)
    # Reasonably informative probabilities.
    probs = np.clip(Y * 0.7 + rng.random((n, c)) * 0.3, 0, 1)
    out = conformal_evaluation(probs, Y, coverage=0.9)
    assert 0.0 <= out["mean_coverage"] <= 1.0
    assert 1.0 <= out["mean_set_size"] <= 2.0


@mimic
def test_evaluate_validation_small(tmp_path):
    from ml.evaluation.clinical_eval import evaluate_validation

    rep = evaluate_validation(limit=3, n_bootstrap=20, make_plots=False, out_dir=tmp_path)
    assert rep["n_images"] > 0
    assert set(rep["macro"]) >= {"auroc", "auprc", "f1", "sensitivity", "specificity"}
    assert len(rep["per_label"]) == 7
    assert (tmp_path / "metrics.json").exists()


@mimic
def test_calibration_small(tmp_path):
    from ml.evaluation.vision_calibration import run_calibration

    rep = run_calibration(limit=3, make_plots=False, mc_passes=2, out_dir=tmp_path)
    assert "temperature_scaling" in rep and "conformal_prediction" in rep
    assert "mc_dropout" in rep and "test_time_augmentation" in rep
    assert (tmp_path / "calibration.json").exists()


def test_perf_benchmark_smoke(tmp_path):
    from ml.evaluation.perf_benchmark import run

    r = run(iters=2, out_dir=tmp_path)
    assert "cpu_latency" in r
    assert r["cpu_latency"]["mean_ms"] > 0
    assert (tmp_path / "benchmark.json").exists()
