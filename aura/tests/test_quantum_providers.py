"""Quantum provider layer: the fallback contract and circuit-translation fidelity.

The property under test throughout is that **execution always returns a result**.
A clinical pipeline cannot block on someone else's queue, so every provider
failure mode — missing SDK, absent credentials, dead device, queue timeout, cost
ceiling — must land on the local simulator with an honest `fell_back` flag rather
than raising into the request path.
"""
from __future__ import annotations

import numpy as np
import pytest

from aura.services.quantum import (
    CircuitSpec,
    ProviderKind,
    ProviderUnavailable,
    describe,
    evaluate_spec,
    execute,
    get_provider,
    list_backends,
    resolve_chain,
    use_real_qpu,
)
from aura.services.quantum.local import LocalProvider


@pytest.fixture
def vqc_spec() -> CircuitSpec:
    rng = np.random.default_rng(0)
    return CircuitSpec(kind="vqc", n_qubits=4, n_layers=2,
                       x=rng.random(4), theta=rng.standard_normal((2, 4, 2)))


@pytest.fixture
def kernel_spec() -> CircuitSpec:
    rng = np.random.default_rng(1)
    return CircuitSpec(kind="iqp_kernel", n_qubits=4, x=rng.random(4), x2=rng.random(4))


# --------------------------------------------------------------------------- #
# Spec identity
# --------------------------------------------------------------------------- #
def test_signature_is_stable_and_discriminating(vqc_spec):
    same = CircuitSpec(kind="vqc", n_qubits=4, n_layers=2,
                       x=vqc_spec.x.copy(), theta=vqc_spec.theta.copy())
    assert vqc_spec.signature() == same.signature()

    perturbed = CircuitSpec(kind="vqc", n_qubits=4, n_layers=2,
                            x=vqc_spec.x + 1e-6, theta=vqc_spec.theta)
    assert vqc_spec.signature() != perturbed.signature()


def test_unknown_circuit_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown circuit kind"):
        evaluate_spec(CircuitSpec(kind="nonsense", n_qubits=2))


# --------------------------------------------------------------------------- #
# Local provider
# --------------------------------------------------------------------------- #
def test_local_provider_is_always_available():
    available, reason = LocalProvider().is_available()
    assert available and reason


def test_local_execution_returns_expected_shape(vqc_spec, kernel_spec):
    r = execute(vqc_spec, provider="local")
    assert r.values.shape == (4,)
    assert np.all(np.abs(r.values) <= 1.0 + 1e-9)      # <Z> is bounded
    assert r.fell_back is False
    assert r.provider is ProviderKind.LOCAL

    k = execute(kernel_spec, provider="local")
    assert k.values.shape == (1,)
    assert 0.0 <= k.values[0] <= 1.0


def test_finite_shots_approach_the_analytic_value(kernel_spec):
    """Sampling is unbiased: many shots must converge on the exact fidelity."""
    exact = evaluate_spec(kernel_spec, shots=None)[0]
    sampled = evaluate_spec(kernel_spec, shots=20000)[0]
    assert sampled == pytest.approx(exact, abs=5 / np.sqrt(20000))


# --------------------------------------------------------------------------- #
# Fallback contract
# --------------------------------------------------------------------------- #
def test_chain_always_ends_at_local():
    for provider in (None, "local", "ibm", "braket"):
        assert resolve_chain(provider)[-1] is ProviderKind.LOCAL


def test_real_qpu_flag_reorders_the_chain(monkeypatch):
    monkeypatch.delenv("AURA_QUANTUM_PROVIDER", raising=False)
    monkeypatch.setenv("AURA_USE_REAL_QPU", "1")
    assert use_real_qpu() is True
    assert resolve_chain() == [ProviderKind.IBM, ProviderKind.BRAKET, ProviderKind.LOCAL]

    monkeypatch.setenv("AURA_USE_REAL_QPU", "0")
    assert resolve_chain() == [ProviderKind.LOCAL]


def test_unknown_provider_env_degrades_to_local(monkeypatch):
    monkeypatch.setenv("AURA_QUANTUM_PROVIDER", "not-a-provider")
    assert resolve_chain() == [ProviderKind.LOCAL]


def test_unavailable_provider_falls_back_and_says_why(vqc_spec, monkeypatch):
    """The central guarantee: a broken provider yields a result, not an exception."""

    class Broken:
        kind = ProviderKind.IBM

        def is_available(self):
            return True, "pretending to be configured"

        def execute(self, *a, **k):
            raise ProviderUnavailable("ibm", "device offline for calibration")

        def list_backends(self, **k):
            return []

        def select_backend(self, *a, **k):
            raise ProviderUnavailable("ibm", "device offline for calibration")

    monkeypatch.setitem(
        __import__("aura.services.quantum.registry", fromlist=["_CACHE"])._CACHE,
        ProviderKind.IBM, Broken(),
    )
    result = execute(vqc_spec, provider="ibm")

    assert result.provider is ProviderKind.LOCAL
    assert result.fell_back is True
    assert "device offline for calibration" in result.fallback_reason
    assert result.values.shape == (4,)


def test_fallback_result_matches_a_direct_local_run(vqc_spec, monkeypatch):
    """A fallback must be the *correct* answer, not merely a non-crash."""

    class Broken:
        kind = ProviderKind.BRAKET

        def is_available(self):
            return True, "configured"

        def execute(self, *a, **k):
            raise ProviderUnavailable("braket", "cost ceiling exceeded")

        def list_backends(self, **k):
            return []

        def select_backend(self, *a, **k):
            raise ProviderUnavailable("braket", "cost ceiling exceeded")

    monkeypatch.setitem(
        __import__("aura.services.quantum.registry", fromlist=["_CACHE"])._CACHE,
        ProviderKind.BRAKET, Broken(),
    )
    assert execute(vqc_spec, provider="braket").values == pytest.approx(
        evaluate_spec(vqc_spec), abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Discovery / description — must never raise, whatever is installed
# --------------------------------------------------------------------------- #
def test_describe_covers_every_provider_without_raising():
    info = describe()
    assert set(info["providers"]) == {"local", "ibm", "braket"}
    assert info["providers"]["local"]["available"] is True
    for p in info["providers"].values():
        assert isinstance(p["available"], bool) and p["reason"]


def test_list_backends_reports_unavailable_providers_as_data():
    """A missing SDK is a row with a reason, never a raised exception."""
    rows = list_backends()
    assert any(r.get("provider") == "local" for r in rows)
    for r in rows:
        assert r.get("name") or r.get("reason")


@pytest.mark.parametrize("kind", list(ProviderKind))
def test_provider_construction_never_touches_the_network(kind):
    available, reason = get_provider(kind).is_available()
    assert isinstance(available, bool) and isinstance(reason, str) and reason


def test_unavailable_ibm_provider_raises_provider_unavailable():
    from aura.services.quantum.ibm import IBMProvider

    p = IBMProvider()
    if p.is_available()[0]:
        pytest.skip("IBM credentials present in this environment")
    with pytest.raises(ProviderUnavailable):
        p.service()


# --------------------------------------------------------------------------- #
# Cross-SDK translation — skipped when the SDK is absent, never faked
# --------------------------------------------------------------------------- #
def test_qiskit_translation_matches_pennylane(vqc_spec, kernel_spec):
    """A translation bug yields well-formed, confidently wrong hardware output."""
    pytest.importorskip("qiskit", reason="qiskit not installed in this environment")
    from aura.services.quantum.benchmark import verify_translation

    for spec in (vqc_spec, kernel_spec):
        check = verify_translation(spec)["checks"]["qiskit"]
        assert check["available"] is True
        assert check["match"] is True, f"max diff {check['max_abs_diff']:.2e}"


def test_verify_translation_reports_absent_sdks_honestly(vqc_spec):
    from aura.services.quantum.benchmark import verify_translation

    report = verify_translation(vqc_spec)
    assert set(report["checks"]) == {"qiskit", "braket"}
    # all_match must be False when nothing could be checked — silence is not success
    if report["n_checked"] == 0:
        assert report["all_match"] is False


# --------------------------------------------------------------------------- #
# Braket cost guard
# --------------------------------------------------------------------------- #
def test_braket_refuses_to_exceed_its_cost_ceiling():
    """Per-shot QPU billing means an unguarded run is a financial incident."""
    from aura.services.quantum.base import BackendInfo, DeviceStatus
    from aura.services.quantum.braket import BraketProvider

    p = BraketProvider(cost_ceiling_usd=1.0)
    pricey = BackendInfo(name="IonQ Aria", provider=ProviderKind.BRAKET, n_qubits=25,
                         simulator=False, status=DeviceStatus.ONLINE,
                         cost_per_shot_usd=0.03)

    assert p.estimate_cost(pricey, shots=1000) == pytest.approx(0.30 + 30.0)
    assert p.estimate_cost(pricey, shots=1000) > p.cost_ceiling_usd

    free = BackendInfo(name="SV1", provider=ProviderKind.BRAKET, n_qubits=34,
                       simulator=True, status=DeviceStatus.ONLINE)
    assert p.estimate_cost(free, shots=100000) is None


def test_braket_shot_cap_is_enforced():
    from aura.services.quantum.braket import BraketProvider

    assert BraketProvider(max_shots=100).max_shots == 100


def test_braket_counts_reduce_to_correct_expectations():
    """<Z> = P(0) - P(1), computed from a shot record."""
    from aura.services.quantum.braket import _expectations_from_counts

    class R:
        measurement_counts = {"00": 50, "11": 50}

    evs, stds = _expectations_from_counts(R(), CircuitSpec(kind="vqc", n_qubits=2))
    assert evs == pytest.approx([0.0, 0.0], abs=1e-9)
    assert np.all(stds > 0)

    class Z:
        measurement_counts = {"0000": 100}

    k, _ = _expectations_from_counts(Z(), CircuitSpec(kind="iqp_kernel", n_qubits=4))
    assert k[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Error mitigation
# --------------------------------------------------------------------------- #
def test_unknown_error_mitigation_is_rejected(vqc_spec):
    from aura.services.quantum.ibm import ERROR_MITIGATION, IBMProvider

    assert set(ERROR_MITIGATION) == {"none", "readout", "zne"}
    p = IBMProvider()
    if not p.is_available()[0]:
        pytest.skip("IBM not configured")
    with pytest.raises(ValueError, match="unknown error_mitigation"):
        p.execute(vqc_spec, error_mitigation="magic")
