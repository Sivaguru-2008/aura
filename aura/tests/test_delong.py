"""Tests for the DeLong implementation.

This statistic is now the single implementation in the repository — ``audit_all.py``
delegates to it, and ``ml/evaluation/qkl_full_coverage.py`` reports its p-values in the
manuscript. It used to exist only inside ``audit_all.py``, which is why Section 5.3
could state that AUROC comparisons used DeLong while no packaged evaluation script
was able to call it. Consolidating the two copies removes the drift risk; these tests
are what makes the consolidation safe.
"""
from __future__ import annotations

import numpy as np
import pytest

from aura.ml.evaluation.delong import auc_variance, delong_test, midrank


def test_midrank_averages_ties():
    r = midrank(np.array([1.0, 2.0, 2.0, 3.0]))
    assert r.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_auroc_matches_sklearn():
    """The area DeLong computes must be the ordinary rank-based AUROC."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(60, 400))
        y = rng.integers(0, 2, n).astype(bool)
        if y.all() or not y.any():
            continue
        score = rng.normal(y * 0.9, 1.0)
        auc, var = auc_variance(score, y)
        assert auc == pytest.approx(
            sklearn_metrics.roc_auc_score(y, score), abs=1e-9)
        assert var > 0


def test_identical_models_give_zero_difference():
    """A model compared against itself has no difference and cannot be significant."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200).astype(bool)
    score = rng.normal(y * 0.7, 1.0)
    r = delong_test(score, score, y)
    assert r["delta_auroc"] == 0.0
    assert r["p"] == 1.0


def test_clearly_better_model_is_detected():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500).astype(bool)
    strong = rng.normal(y * 2.0, 1.0)
    weak = rng.normal(y * 0.05, 1.0)
    r = delong_test(strong, weak, y)
    assert r["delta_auroc"] > 0.2
    assert r["p"] < 1e-6
    lo, hi = r["ci95"]
    assert lo > 0, "a significant positive difference must have an interval above zero"
    assert lo < r["delta_auroc"] < hi


def test_interval_and_pvalue_agree():
    """The interval is derived from the same covariance as the p-value.

    Reporting a bootstrap interval beside a DeLong p-value can produce a
    non-significant p with an interval excluding zero, which reads as a
    contradiction. Deriving both from one covariance makes that impossible.
    """
    rng = np.random.default_rng(3)
    for _ in range(10):
        y = rng.integers(0, 2, 300).astype(bool)
        a = rng.normal(y * rng.uniform(0, 1.2), 1.0)
        b = rng.normal(y * rng.uniform(0, 1.2), 1.0)
        r = delong_test(a, b, y)
        lo, hi = r["ci95"]
        excludes_zero = lo > 0 or hi < 0
        assert excludes_zero == (r["p"] < 0.05), (
            f"interval {r['ci95']} and p={r['p']} disagree at the 5% level"
        )


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        delong_test(np.zeros(10), np.zeros(9), np.ones(10, dtype=bool))


def test_requires_both_classes():
    with pytest.raises(ValueError):
        delong_test(np.zeros(10), np.zeros(10), np.ones(10, dtype=bool))


def test_audit_harness_delegates_to_this_module():
    """``audit_all.py`` must not carry a second copy of the algorithm."""
    from pathlib import Path

    audit = Path(__file__).resolve().parents[2] / "audit_all.py"
    if not audit.exists():
        pytest.skip("audit_all.py not present")
    src = audit.read_text(encoding="utf-8")
    assert "from aura.ml.evaluation.delong import" in src, (
        "audit_all.py no longer delegates to the packaged DeLong implementation"
    )
    assert "def _fast_delong" not in src, (
        "audit_all.py has re-acquired its own DeLong implementation; two copies of "
        "one statistic is the drift risk this consolidation removed"
    )
