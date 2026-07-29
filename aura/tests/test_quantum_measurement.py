"""Tests for the quantum measurement layer: coupling map, shot budget, calibration.

Hermetic: every test builds its own small VQC from a seeded RNG rather than loading
a trained artifact, so the suite runs without model files and cannot silently start
measuring a different model than it was written against.

The load-bearing test in this file is
``test_product_state_ansatz_has_exactly_zero_correlators``. It is what makes every
other claim about the coupling map meaningful: if ``C_ij`` were non-zero on a
separable state, it would be measuring numerical noise or an implementation artefact
rather than entanglement, and the whole evidence-coupling story would be decoration.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pennylane", reason="quantum tests need PennyLane")

from aura.services.fusion.device import make_probs_qnode, make_qnode          # noqa: E402
from aura.services.fusion.qmba import (                                       # noqa: E402
    PLAUSIBLE_SHOT_CEILING,
    QuantumMeasurementBudget,
    shot_sweep,
)
from aura.services.fusion.qmeasure import (                                   # noqa: E402
    CORRELATION_FLOOR,
    coupling_summary,
    measure_entanglement,
)
from aura.services.fusion.quantum import QuantumFusion                        # noqa: E402
from aura.schemas.clinical import DIAGNOSES                                   # noqa: E402

N_QUBITS = 4          # small on purpose: 2**4 states keeps the suite fast
N_LAYERS = 2


def make_model(entangler: str = "ring", seed: int = 3) -> QuantumFusion:
    """A small, seeded VQC. Untrained — the measurements under test are properties of
    the circuit, not of its accuracy."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.8, size=(N_LAYERS, N_QUBITS, 2))
    W = rng.normal(0.0, 0.6, size=(len(DIAGNOSES), N_QUBITS))
    b = rng.normal(0.0, 0.1, size=len(DIAGNOSES))
    return QuantumFusion(theta, W, b, N_QUBITS, N_LAYERS, entangler=entangler)


@pytest.fixture
def evidence() -> np.ndarray:
    return np.array([0.8, 0.2, 0.65, 0.1])


# =========================================================================== #
# Circuit construction
# =========================================================================== #
def test_unknown_entangler_is_rejected():
    for factory in (make_qnode, make_probs_qnode):
        with pytest.raises(ValueError, match="unknown entangler"):
            factory(N_QUBITS, N_LAYERS, entangler="all_to_all")


def test_entangler_choice_changes_the_state(evidence):
    """The ablation control must actually be a different circuit."""
    ring = make_qnode(N_QUBITS, N_LAYERS, entangler="ring")
    product = make_qnode(N_QUBITS, N_LAYERS, entangler="none")
    theta = make_model().theta
    z_ring = np.array([float(v) for v in ring(evidence, theta)])
    z_product = np.array([float(v) for v in product(evidence, theta)])
    assert not np.allclose(z_ring, z_product)


def test_product_and_ring_have_identical_parameter_counts():
    """A fair ablation changes the ansatz and nothing else."""
    ring, product = make_model("ring"), make_model("none")
    assert ring.theta.shape == product.theta.shape
    assert ring.W.shape == product.W.shape


# =========================================================================== #
# Evidence-entanglement map
# =========================================================================== #
def test_product_state_ansatz_has_exactly_zero_correlators(evidence):
    """Without two-qubit gates the register stays separable, so every connected
    correlator must vanish.

    This is the validation the whole coupling map rests on. ``C_ij`` is claimed to
    measure entanglement; on a state that provably has none it must read zero, to
    numerical precision and not merely 'small'.
    """
    result = measure_entanglement(make_model("none"), evidence)
    assert np.allclose(result.correlation, 0.0, atol=1e-12)
    assert result.total_coupling < CORRELATION_FLOOR
    assert result.is_product_state()


def test_ring_ansatz_produces_real_coupling(evidence):
    result = measure_entanglement(make_model("ring"), evidence)
    assert result.total_coupling > 0.01
    assert not result.is_product_state()


def test_correlation_matrix_is_symmetric_with_zero_diagonal(evidence):
    correlation = measure_entanglement(make_model(), evidence).correlation
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(correlation), 0.0, atol=1e-12)


def test_differential_vanishes_against_its_own_reference(evidence):
    """Attributing a state to itself must yield nothing."""
    result = measure_entanglement(make_model(), evidence, reference=evidence)
    assert np.allclose(result.differential, 0.0, atol=1e-12)
    assert result.differential_coupling < CORRELATION_FLOOR
    assert result.entropy_shift_bits == pytest.approx(0.0, abs=1e-9)


def test_raw_coupling_is_not_the_patient_attributable_quantity():
    """The trained rotations entangle the register even with no evidence at all.

    Guards the finding that motivated the differential: reporting raw coupling would
    make an empty study look maximally coupled.
    """
    model = make_model()
    empty = measure_entanglement(model, np.zeros(N_QUBITS))
    assert empty.baseline_coupling == pytest.approx(empty.total_coupling, abs=1e-12)
    assert empty.differential_coupling < CORRELATION_FLOOR
    # The circuit's own coupling is real and non-zero on the empty input.
    assert empty.total_coupling > 0.0


def test_measurement_entropy_is_bounded_by_the_register_width(evidence):
    result = measure_entanglement(make_model(), evidence)
    assert 0.0 <= result.measurement_entropy_bits <= N_QUBITS + 1e-9
    assert result.max_entropy_bits == float(N_QUBITS)
    assert 0.0 <= result.normalised_entropy <= 1.0 + 1e-9


def test_wrong_width_evidence_is_rejected_not_padded():
    with pytest.raises(ValueError, match="channels but the circuit"):
        measure_entanglement(make_model(), np.zeros(N_QUBITS + 2))


def test_top_pairs_preserve_sign_and_rank_by_magnitude(evidence):
    pairs = measure_entanglement(make_model(), evidence).top_pairs(3)
    assert pairs
    magnitudes = [abs(p["correlation"]) for p in pairs]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(p["direction"] in ("aligned", "opposed") for p in pairs)


def test_coupling_summary_never_states_causation(evidence):
    summary = coupling_summary(measure_entanglement(make_model(), evidence))
    assert "not a causal relationship" in summary or "assessed independently" in summary
    for forbidden in ("causes", "because of", "due to"):
        assert forbidden not in summary.lower()


def test_entanglement_map_is_json_serialisable(evidence):
    import json

    json.dumps(measure_entanglement(make_model(), evidence).to_dict())


# =========================================================================== #
# Measurement-budgeted abstention
# =========================================================================== #
def test_margin_uncertainty_falls_as_one_over_root_shots(evidence):
    """The physics the whole schedule rests on: Var[<Z>] = (1 - <Z>^2)/n_shots.

    A 64x increase in shots must shrink the margin's standard deviation by ~8x.
    Checked with a wide tolerance because the spread is itself a Monte-Carlo
    estimate, but a scaling that is not ~1/sqrt(n) would invalidate the extrapolation
    that predicts shots-to-decision.
    """
    rows = {r["shots"]: r for r in shot_sweep(make_model(), evidence,
                                              shot_counts=(64, 4096))}
    ratio = rows[64]["margin_std"] / rows[4096]["margin_std"]
    assert 5.0 < ratio < 12.0, f"expected ~8x, got {ratio:.2f}"


def test_more_shots_never_increases_the_posterior_spread(evidence):
    rows = shot_sweep(make_model(), evidence,
                      shot_counts=(64, 256, 1024, 4096))
    spreads = [r["mean_posterior_std"] for r in rows]
    assert spreads == sorted(spreads, reverse=True)


def test_decision_is_deterministic_for_a_fixed_seed(evidence):
    model = make_model()
    a = QuantumMeasurementBudget(model, seed=11).decide(evidence)
    b = QuantumMeasurementBudget(model, seed=11).decide(evidence)
    assert a.committed == b.committed
    assert a.shots_spent == b.shots_spent
    assert a.separation_z == pytest.approx(b.separation_z)


def test_a_well_separated_case_commits_early_and_cheaply():
    """A confident decision must not pay for precision it does not need."""
    model = make_model()
    # Drive the head to a large margin so the decision is unambiguous.
    model.W = model.W * 8.0
    budget = QuantumMeasurementBudget(model, initial_shots=128)
    decision = budget.decide(np.array([0.9, 0.1, 0.9, 0.1]))
    assert decision.committed
    assert decision.shots_spent <= 512
    assert decision.limiting_factor is None
    assert "committed to" in decision.reason


def _nearly_tied_model(seed: int = 5) -> QuantumFusion:
    """A model whose top two classes are almost indistinguishable.

    The two leading rows of the readout are made nearly identical, so the analytic
    margin is ~1e-5 while shot noise still reaches the posterior. That is the honest
    shape of a hard case: a real tie under measurement, not a degenerate model.
    Zeroing ``W`` instead would remove shot noise altogether and test nothing.
    """
    model = make_model(seed=seed)
    rng = np.random.default_rng(seed)
    # Two steps, both needed. Making rows 0 and 1 near-identical only creates a tie
    # if those rows are also the two the posterior actually ranks highest — the
    # schedule reads the top two by rank, not by index — so the remaining classes are
    # pushed far down first.
    model.b[:] = -12.0
    model.b[0] = model.b[1] = 0.0
    model.W[1] = model.W[0] + 1e-5 * rng.normal(size=model.W.shape[1])
    return model


def test_a_tied_case_abstains_and_is_called_model_limited():
    """When the analytic margin is ~0, no budget helps and the reason must say so."""
    decision = QuantumMeasurementBudget(_nearly_tied_model()).decide(
        np.array([0.5] * N_QUBITS))

    assert not decision.committed
    assert decision.model_limited
    assert decision.floor_limited
    assert decision.analytic_margin == pytest.approx(0.0, abs=1e-3)
    assert "clinical floor" in decision.reason


def test_a_statistically_separated_but_negligible_lead_is_refused():
    """The defect this floor exists for.

    A lead of ~1e-6 is separable from zero to arbitrary confidence if the shot noise
    is smaller still. Committing on it would be statistically defensible and
    clinically absurd, so the analytic margin is checked against a significance floor
    before any measurement is bought.
    """
    decision = QuantumMeasurementBudget(_nearly_tied_model(),
                                        min_margin=0.05).decide(
        np.array([0.5] * N_QUBITS))
    assert not decision.committed
    assert decision.analytic_margin < 0.05
    # Nothing beyond the first probe was spent: precision was never the problem.
    assert decision.shots_spent == decision.trajectory[0].shots
    assert len(decision.trajectory) == 1


def test_disabling_the_floor_restores_the_unsafe_behaviour():
    """Documents exactly what the floor is protecting against."""
    decision = QuantumMeasurementBudget(_nearly_tied_model(),
                                        min_margin=0.0).decide(
        np.array([0.5] * N_QUBITS))
    assert decision.committed
    assert decision.margin < 1e-3, "committed on a negligible lead, as expected"


def test_a_deterministic_tie_is_not_scored_as_certainty():
    """A readout that ignores the quantum state has zero shot noise.

    Dividing the margin by a zero spread would report infinite confidence for an
    arbitrarily small margin. The tie must be recognised as a tie.
    """
    model = make_model()
    model.W = np.zeros_like(model.W)
    model.b = np.zeros_like(model.b)
    decision = QuantumMeasurementBudget(model).decide(np.array([0.5] * N_QUBITS))

    assert not decision.committed
    assert decision.separation_z == 0.0
    assert decision.margin_std == 0.0


def test_limiting_factor_tracks_the_plausible_shot_ceiling(evidence):
    """The measurement/model split is a threshold on achievable device time."""
    model = make_model()
    model.W = model.W * 0.35              # small but non-zero margin
    decision = QuantumMeasurementBudget(model, min_margin=0.0).decide(evidence)
    if not decision.committed:
        assert decision.limiting_factor in ("measurement", "model")
        if decision.predicted_shots is not None:
            expected = ("model" if decision.predicted_shots > PLAUSIBLE_SHOT_CEILING
                        else "measurement")
            assert decision.limiting_factor == expected


def test_abstention_still_reports_what_could_not_be_confirmed():
    decision = QuantumMeasurementBudget(_nearly_tied_model()).decide(
        np.array([0.5] * N_QUBITS))
    assert not decision.committed
    assert decision.top and decision.runner_up  # noqa: PT018
    assert decision.top != decision.runner_up
    assert set(decision.posterior) == {
        d.value if hasattr(d, "value") else str(d) for d in DIAGNOSES}


def test_budget_never_exceeds_its_ceiling(evidence):
    # min_margin=0 so the schedule actually runs; this test is about the ceiling.
    decision = QuantumMeasurementBudget(_nearly_tied_model(), initial_shots=64,
                                        max_shots=512, min_margin=0.0,
                                        required_z=1e9).decide(evidence)
    assert all(stage.shots <= 512 for stage in decision.trajectory)
    assert decision.trajectory[-1].shots == 512


def test_schedule_doubles_until_it_resolves_or_hits_the_cap(evidence):
    decision = QuantumMeasurementBudget(_nearly_tied_model(), initial_shots=32,
                                        max_shots=256, min_margin=0.0,
                                        required_z=1e9).decide(evidence)
    assert [s.shots for s in decision.trajectory] == [32, 64, 128, 256]


def test_invalid_budget_configuration_is_rejected():
    with pytest.raises(ValueError, match="initial_shots"):
        QuantumMeasurementBudget(make_model(), initial_shots=512, max_shots=128)


def test_decision_is_json_serialisable(evidence):
    import json

    json.dumps(QuantumMeasurementBudget(make_model()).decide(evidence).to_dict())


# =========================================================================== #
# Per-backend calibration
# =========================================================================== #
def test_calibration_prefers_the_backend_specific_file(tmp_path, monkeypatch):
    """Serving one backend with another's temperature is the bug this prevents."""
    from aura.services.safety import calibration as calibration_module

    monkeypatch.setattr(calibration_module, "ARTIFACTS", tmp_path)
    Calibration = calibration_module.Calibration

    Calibration(temperature=0.45).save(path=tmp_path / "safety.npz")
    Calibration(temperature=0.99).save(path=tmp_path / "safety_quantum.npz")

    assert Calibration.load(path=tmp_path / "safety.npz").temperature == pytest.approx(0.45)
    assert calibration_module.backend_calibration_path("quantum").name == "safety_quantum.npz"
    assert Calibration.load(backend="quantum").temperature == pytest.approx(0.99)


def test_calibration_falls_back_when_no_backend_file_exists(tmp_path, monkeypatch):
    from aura.services.safety import calibration as calibration_module

    monkeypatch.setattr(calibration_module, "ARTIFACTS", tmp_path)
    Calibration = calibration_module.Calibration
    Calibration(temperature=0.61).save(path=tmp_path / "safety.npz")

    # No safety_learnable.npz — must fall back rather than reset to T=1.0.
    assert Calibration.load(backend="learnable").temperature == pytest.approx(0.61)


def test_saving_with_a_backend_writes_both_files(tmp_path, monkeypatch):
    from aura.services.safety import calibration as calibration_module

    monkeypatch.setattr(calibration_module, "ARTIFACTS", tmp_path)
    calibration_module.Calibration(temperature=0.77).save(
        path=tmp_path / "safety.npz", backend="quantum")

    assert (tmp_path / "safety.npz").exists()
    assert (tmp_path / "safety_quantum.npz").exists()
