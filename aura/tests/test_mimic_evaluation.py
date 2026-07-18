"""Tests for Step 10 — evaluation suite."""
from __future__ import annotations

import numpy as np

from mimic.evaluation import (
    confusion_matrix,
    evaluate_binary,
    evaluate_multiclass,
    evaluate_multilabel,
    write_model_card,
)


def test_confusion_matrix_counts():
    y = np.array([0, 1, 2, 1, 0])
    p = np.array([0, 1, 1, 1, 0])
    m = confusion_matrix(y, p, 3)
    assert m[0, 0] == 2
    assert m[1, 1] == 2
    assert m[2, 1] == 1
    assert m.sum() == 5


def test_evaluate_binary_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    rep = evaluate_binary(scores, y)
    assert rep["auroc"] == 1.0
    assert rep["sensitivity"] == 1.0
    assert rep["specificity"] == 1.0
    assert rep["confusion_matrix"] == [[2, 0], [0, 2]]


def test_evaluate_binary_accepts_2col_proba():
    y = np.array([0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    rep = evaluate_binary(proba, y)
    assert 0.0 <= rep["auroc"] <= 1.0
    assert rep["n"] == 3


def test_evaluate_multiclass_has_confusion_and_core_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, size=120)
    proba = rng.dirichlet(np.ones(6), size=120)
    rep = evaluate_multiclass(proba, y, [str(i) for i in range(6)])
    assert "accuracy" in rep and "macro_auroc" in rep and "ece" in rep
    cm = np.array(rep["confusion_matrix"])
    assert cm.shape == (6, 6)
    assert cm.sum() == 120


def test_evaluate_multilabel_shapes():
    rng = np.random.default_rng(1)
    Y = rng.integers(0, 2, size=(100, 7))
    P = rng.random((100, 7))
    rep = evaluate_multilabel(P, Y, [f"f{i}" for i in range(7)])
    assert len(rep["per_label"]) == 7
    assert "macro_auroc" in rep and "micro_f1" in rep


def test_write_model_card(tmp_path):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 6, size=60)
    proba = rng.dirichlet(np.ones(6), size=60)
    rep = evaluate_multiclass(proba, y, [str(i) for i in range(6)])
    p = write_model_card(rep, tmp_path / "card.md", "Test Card")
    assert p.is_file()
    assert p.with_suffix(".json").is_file()
    assert "Confusion matrix" in p.read_text(encoding="utf-8")
