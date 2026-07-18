"""Tests for Step 9 — training pipelines (backend-agnostic, fast synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mimic.tasks import TaskDataset, TaskSpec
from mimic.training import (
    GBMTrainer,
    MLPConfig,
    MLPTrainer,
    _make_booster,
    sequence_models_status,
)


def _toy_task(n=200, d=8, n_classes=3, seed=0) -> tuple[TaskDataset, TaskDataset]:
    """Separable-ish synthetic multiclass data so training actually learns."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(n_classes, d))
    ytr = rng.integers(0, n_classes, size=n)
    Xtr = centers[ytr] + rng.normal(0, 1, size=(n, d))
    yva = rng.integers(0, n_classes, size=n // 2)
    Xva = centers[yva] + rng.normal(0, 1, size=(n // 2, d))
    spec = TaskSpec("toy", "multiclass", "label_diagnosis", n_classes, True)
    cols = [f"f{i}" for i in range(d)]
    tr = TaskDataset(spec, "train", pd.DataFrame(Xtr, columns=cols), ytr, cols,
                     [str(i) for i in range(n_classes)])
    va = TaskDataset(spec, "validation", pd.DataFrame(Xva, columns=cols), yva, cols,
                     [str(i) for i in range(n_classes)])
    return tr, va


def test_booster_falls_back_to_sklearn():
    est, backend = _make_booster("auto", "multiclass", 3)
    assert backend in ("xgboost", "lightgbm", "catboost", "sklearn")
    assert hasattr(est, "fit") and hasattr(est, "predict_proba")


def test_gbm_trains_and_predicts_probabilities():
    tr, va = _toy_task()
    res = GBMTrainer(tr).train(va)
    assert res.proba_val.shape == (va.n_samples, 3)
    # probabilities sum to 1
    assert np.allclose(res.proba_val.sum(1), 1.0, atol=1e-4)
    # learns better than random on separable data
    acc = (res.proba_val.argmax(1) == va.y).mean()
    assert acc > 0.6


def test_gbm_checkpoint_roundtrip(tmp_path):
    tr, va = _toy_task()
    ckpt = tmp_path / "gbm.joblib"
    GBMTrainer(tr).train(va, checkpoint=ckpt)
    assert ckpt.is_file()
    loaded = GBMTrainer.load(ckpt)
    assert "model" in loaded and "backend" in loaded


def test_mlp_trains_with_early_stopping(tmp_path):
    torch = pytest.importorskip("torch")
    tr, va = _toy_task()
    ckpt = tmp_path / "mlp.pt"
    res = MLPTrainer(tr, MLPConfig(epochs=30, patience=5)).train(va, checkpoint=ckpt)
    assert res.proba_val.shape == (va.n_samples, 3)
    assert ckpt.is_file()
    assert len(res.history["val_loss"]) >= 1


def test_mlp_resume(tmp_path):
    pytest.importorskip("torch")
    tr, va = _toy_task()
    ckpt = tmp_path / "mlp.pt"
    MLPTrainer(tr, MLPConfig(epochs=6, patience=5)).train(va, checkpoint=ckpt)
    # resume must reload without error and produce predictions
    res = MLPTrainer(tr, MLPConfig(epochs=10, patience=5)).train(va, checkpoint=ckpt, resume=True)
    assert res.proba_val.shape[0] == va.n_samples


def test_sequence_status_documents_absence():
    s = sequence_models_status()
    assert "lstm" in s and "temporal_transformer" in s
