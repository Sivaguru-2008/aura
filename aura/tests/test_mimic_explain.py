"""Tests for Step 11 — tabular explainability (fast synthetic models)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from mimic.explain import (
    counterfactual,
    native_importance,
    occlusion_attribution,
    permutation_importance,
    shap_available,
    shap_values,
)


def _fitted(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d))
    # class depends strongly on feature 0 and 1
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=80).fit(X, y)
    names = [f"f{i}" for i in range(d)]
    return model, X, y, names


def test_native_importance_absent_is_empty():
    model, X, y, names = _fitted()
    # HGB has no feature_importances_ -> empty dict, no crash
    assert native_importance(model, names) == {}


def test_permutation_importance_ranks_informative_features():
    model, X, y, names = _fitted()
    pi = permutation_importance(model, X, y, names, n_repeats=3)
    top2 = list(pi)[:2]
    assert set(top2) <= {"f0", "f1"}            # the informative features rank top


def test_occlusion_attribution_shape_and_finite():
    model, X, y, names = _fitted()
    base = X.mean(0)
    attr = occlusion_attribution(model.predict_proba, X[0], base, target=1)
    assert attr.shape == (X.shape[1],)
    assert np.all(np.isfinite(attr))


def test_shap_values_fallback_or_real():
    model, X, y, names = _fitted()
    vals, method = shap_values(model, X[:50], X[:3])
    assert method in ("shap", "occlusion")
    assert vals.shape[0] == 3


def test_counterfactual_can_flip():
    model, X, y, names = _fitted()
    base = X.mean(0)
    # find an instance and push toward baseline; strongly-separable data should flip
    flips = 0
    for i in range(20):
        cf = counterfactual(model, X[i], names, base, max_features=6)
        if cf is not None:
            assert cf.new_class != cf.orig_class
            flips += 1
    assert flips >= 1


def test_shap_available_is_bool():
    assert isinstance(shap_available(), bool)
