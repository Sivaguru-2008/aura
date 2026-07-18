"""Tests for Step 12 — uncertainty quantification (fast synthetic models)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from mimic.uncertainty import (
    ConformalPredictor,
    DeepEnsemble,
    TemperatureScaler,
    _softmax,
    _to_logits,
)


def _ensemble_fixture(n=400, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d))
    y = rng.integers(0, 3, size=n)
    models = [
        HistGradientBoostingClassifier(max_iter=60, random_state=s).fit(X, y)
        for s in range(3)
    ]
    return models, X, y


def test_softmax_logits_roundtrip_is_stochastic_vector():
    p = np.array([[0.2, 0.3, 0.5]])
    out = _softmax(_to_logits(p))
    assert np.allclose(out.sum(1), 1.0)
    assert out.shape == p.shape


def test_deep_ensemble_predict_shapes_and_uncertainty():
    models, X, y = _ensemble_fixture()
    ens = DeepEnsemble(models)
    pred = ens.predict(X, n_classes=3)
    assert pred.proba.shape == (len(X), 3)
    assert pred.member_proba.shape == (3, len(X), 3)
    assert pred.predictive_entropy >= 0.0
    assert pred.mutual_information >= -1e-9


def test_deep_ensemble_requires_models():
    with pytest.raises(ValueError):
        DeepEnsemble([])


def test_temperature_scaling_returns_valid_distribution():
    models, X, y = _ensemble_fixture()
    proba = models[0].predict_proba(X)
    ts = TemperatureScaler().fit(proba, y)
    scaled = ts.transform(proba)
    assert np.allclose(scaled.sum(1), 1.0)
    assert ts.T > 0
    gain = ts.calibration_gain(proba, y)
    assert "ece_before" in gain and "ece_after" in gain


def test_conformal_marginal_coverage_is_reasonable():
    models, X, y = _ensemble_fixture(n=600)
    proba = models[0].predict_proba(X)
    half = len(y) // 2
    cp = ConformalPredictor(coverage=0.9).fit(proba[:half], y[:half])
    res = cp.evaluate(proba[half:], y[half:])
    # coverage should be in the right ballpark of the 0.9 target
    assert 0.8 <= res.empirical_coverage <= 1.0
    assert res.avg_set_size >= 1.0


def test_conformal_set_never_empty():
    models, X, y = _ensemble_fixture()
    proba = models[0].predict_proba(X)
    cp = ConformalPredictor(coverage=0.9).fit(proba, y)
    for row in proba[:20]:
        assert len(cp.predict_set(row)) >= 1


def test_conformal_mondrian_runs():
    models, X, y = _ensemble_fixture(n=600)
    proba = models[0].predict_proba(X)
    half = len(y) // 2
    cp = ConformalPredictor(coverage=0.9, mondrian=True).fit(proba[:half], y[:half])
    res = cp.evaluate(proba[half:], y[half:])
    assert res.method == "mondrian"
    assert res.avg_set_size >= 1.0
