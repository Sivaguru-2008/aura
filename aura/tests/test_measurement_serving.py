"""QMBA on the serving path.

Measurement-budgeted abstention is the one thing in AURA that a classical model
cannot do at all: because a quantum model's precision is bought with shots, an
unresolved case can be split into *measurement-limited* ("run the circuit longer,
here is how much longer") and *model-limited* ("the top two are tied at infinite
precision — escalate"). A classical softmax of 0.55 means "unsure" and cannot say
which.

It existed as a complete, tested, evidenced module that only `ml/evaluation/`
scripts imported, so the running product never surfaced it. These tests pin it to
the pipeline and pin the properties that make it honest.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from aura.gateway.pipeline import Pipeline
from aura.ml.data import make_sample
from aura.schemas.clinical import DIAGNOSES
from aura.schemas.contracts import MeasurementBudget, StudyInput
from aura.services.fusion import FusionEngine
from aura.services.fusion.qmba import QuantumMeasurementBudget


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline()


def _study(i: int, rng: np.random.Generator) -> StudyInput:
    s = make_sample(DIAGNOSES[i % len(DIAGNOSES)], rng)
    img = np.asarray(s.image, dtype=float)
    return StudyInput(study_id=f"MB-{i}", image=[float(v) for v in img.flatten()],
                      image_shape=tuple(img.shape), priors=s.priors)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_budget_is_active_on_a_quantum_backend(pipeline):
    if pipeline.fusion.backend != "quantum":
        pytest.skip(f"fusion resolved to {pipeline.fusion.backend}; QMBA is quantum-only")
    assert pipeline.measurement_budget is not None


def test_bundle_carries_a_measurement_budget(pipeline):
    if pipeline.fusion.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")
    rng = np.random.default_rng(11)
    bundle = asyncio.run(pipeline.run(_study(3, rng), "MB-CASE-3"))

    m = bundle.measurement
    assert isinstance(m, MeasurementBudget)
    assert m.shots_spent >= 1
    assert m.top in DIAGNOSES and m.runner_up in DIAGNOSES
    assert m.top != m.runner_up
    assert np.isfinite([m.margin, m.margin_std, m.separation_z, m.analytic_margin]).all()
    assert m.trajectory, "the per-stage schedule must be recorded for the console trace"


def test_a_classical_backend_reports_no_budget_rather_than_a_fake_one():
    """The field is None on classical, not a fabricated budget.

    A product-of-experts has no shot noise to sequence. Reporting a measurement
    budget for it would be the same category of fiction as the hardcoded competitor
    rows this repo removed from its benchmark tables.
    """
    p = Pipeline.__new__(Pipeline)              # no engine loading
    p.fusion = type("F", (), {"backend": "classical"})()
    p.measurement_budget = None
    assert Pipeline._measure_budget(p, np.zeros(8)) is None


# --------------------------------------------------------------------------- #
# The properties that make the abstention meaningful
# --------------------------------------------------------------------------- #
def test_abstention_reasons_are_exactly_the_two_actionable_kinds(pipeline):
    if pipeline.fusion.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")
    rng = np.random.default_rng(11)
    seen = set()
    for i in range(10):
        b = asyncio.run(pipeline.run(_study(i, rng), f"MB-CASE-{i}"))
        m = b.measurement
        assert m is not None
        if m.committed:
            assert m.limiting_factor is None
        else:
            # Two values, because they map to two different clinical instructions.
            assert m.limiting_factor in {"measurement", "model"}
            assert m.measurement_limited ^ m.model_limited
        seen.add(m.committed)
    assert seen, "no studies ran"


def test_model_limited_abstention_means_more_shots_would_not_help():
    """The distinction has to be real, not cosmetic.

    A model-limited verdict claims the diagnoses are tied at *infinite* precision.
    That is a statement about the analytic margin, so it must be small — otherwise
    the label is meaningless and the clinician is told to escalate a case that more
    measurement would have settled.
    """
    fe = FusionEngine()
    if fe.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")
    q = QuantumMeasurementBudget(fe.model)
    rng = np.random.default_rng(5)

    checked = 0
    for _ in range(40):
        d = q.decide(rng.random(8))
        if d.model_limited and not d.floor_limited:
            # Ceiling-limited: no achievable budget separates these two.
            assert abs(d.analytic_margin) < 0.15, (
                f"declared model-limited but the analytic margin is {d.analytic_margin:.3f}; "
                f"that gap would resolve with more shots"
            )
            assert d.predicted_shots is None or d.predicted_shots > q.max_shots
            checked += 1
    if checked == 0:
        pytest.skip("no ceiling-limited abstentions in this sample")


def test_measurement_limited_abstention_predicts_a_reachable_budget():
    fe = FusionEngine()
    if fe.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")
    q = QuantumMeasurementBudget(fe.model)
    rng = np.random.default_rng(5)
    for _ in range(40):
        d = q.decide(rng.random(8))
        if d.measurement_limited:
            assert d.predicted_shots is not None and d.predicted_shots > 0
            assert abs(d.analytic_margin) > 0, (
                "measurement-limited claims the answer exists at infinite precision"
            )


def test_the_budget_is_deterministic(pipeline):
    """A clinical decision that changes when you re-run it is not a decision."""
    fe = FusionEngine()
    if fe.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")
    x = np.random.default_rng(2).random(8)
    a = QuantumMeasurementBudget(fe.model, seed=7).decide(x)
    b = QuantumMeasurementBudget(fe.model, seed=7).decide(x)
    assert a.shots_spent == b.shots_spent
    assert a.committed == b.committed
    assert a.limiting_factor == b.limiting_factor
    assert a.margin == pytest.approx(b.margin)


def test_budgeting_never_changes_the_served_posterior(pipeline):
    """QMBA annotates; it must not steer.

    The analytic posterior stays what is served — spending shots buys resolution,
    not correctness, and as n_shots -> inf the estimate converges to the analytic
    value the pipeline already used. If this ever fails, measurement economics have
    started overriding the diagnosis, which is not what they are for.
    """
    if pipeline.fusion.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")
    rng = np.random.default_rng(11)
    b = asyncio.run(pipeline.run(_study(2, rng), "MB-CASE-STEER"))
    served_top = max(b.fusion.posterior, key=b.fusion.posterior.get)
    assert b.measurement.top == served_top


def test_budget_failure_degrades_to_none_rather_than_failing_the_study(monkeypatch, pipeline):
    """Measurement economics are reporting, not a gate."""
    if pipeline.fusion.backend != "quantum":
        pytest.skip("classical backend has no measurement budget")

    class Exploding:
        def decide(self, x):
            raise RuntimeError("simulated QMBA failure")

    monkeypatch.setattr(pipeline, "measurement_budget", Exploding())
    assert pipeline._measure_budget(np.zeros(8)) is None
