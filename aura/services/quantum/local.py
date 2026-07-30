"""Local PennyLane simulator provider — the always-available baseline.

This is the reference implementation of every circuit AURA runs, and the target
that every fallback lands on. Because it is also the ground truth against which
hardware results are compared, the circuits here are the *same functions* the
fusion and QKL engines use (:mod:`aura.services.fusion.device`), not a re-derived
copy that could silently drift.
"""
from __future__ import annotations

import time

import numpy as np

from .base import (
    BackendInfo,
    CircuitSpec,
    DeviceStatus,
    ExecutionResult,
    ProviderKind,
)


def evaluate_spec(spec: CircuitSpec, shots: int | None = None) -> np.ndarray:
    """Evaluate a :class:`CircuitSpec` on the local simulator.

    Shared by every provider's fallback path and by the hardware-vs-analytic
    comparison, so there is exactly one definition of "what the circuit means".
    """
    if spec.kind == "vqc":
        from aura.services.fusion.device import make_qnode

        qnode = make_qnode(
            spec.n_qubits, spec.n_layers, shots=shots, entangler=spec.entangler
        )
        return np.asarray(qnode(np.asarray(spec.x, dtype=float),
                                np.asarray(spec.theta, dtype=float)), dtype=float).ravel()

    if spec.kind == "iqp_kernel":
        from aura.backend.engines.neuro.qkl import QKLClassifier

        clf = QKLClassifier(n_qubits=spec.n_qubits)
        if shots is None:
            return np.asarray([clf.kernel_eval(spec.x, spec.x2)], dtype=float)
        # Finite-shot fidelity estimate: the all-zero string frequency is a
        # binomial estimator of |<phi(x)|phi(x2)>|^2, which is exactly what a
        # hardware SWAP-free adjoint-circuit run measures.
        p = clf.kernel_eval(spec.x, spec.x2)
        rng = np.random.default_rng()
        return np.asarray([rng.binomial(shots, min(max(p, 0.0), 1.0)) / shots], dtype=float)

    raise ValueError(f"unknown circuit kind {spec.kind!r}")


class LocalProvider:
    """PennyLane ``default.qubit``. Always available, never queues."""

    kind = ProviderKind.LOCAL

    def __init__(self, device_name: str = "default.qubit"):
        self.device_name = device_name

    def is_available(self) -> tuple[bool, str]:
        try:
            import pennylane  # noqa: F401
        except ImportError:
            return False, "pennylane is not installed"
        return True, "local statevector simulator"

    def list_backends(self, min_qubits: int = 1) -> list[BackendInfo]:
        return [
            BackendInfo(
                name=self.device_name,
                provider=ProviderKind.LOCAL,
                n_qubits=32,          # practical statevector ceiling on a workstation
                simulator=True,
                status=DeviceStatus.ONLINE,
                pending_jobs=0,
                metadata={"exact": True, "noise_model": None},
            )
        ]

    def select_backend(self, min_qubits: int, prefer: str | None = None) -> BackendInfo:
        return self.list_backends(min_qubits)[0]

    def execute(
        self,
        spec: CircuitSpec,
        shots: int | None = None,
        backend: str | None = None,
        error_mitigation: str = "none",
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        t0 = time.perf_counter()
        values = evaluate_spec(spec, shots=shots)
        return ExecutionResult(
            values=values,
            provider=ProviderKind.LOCAL,
            backend=backend or self.device_name,
            shots=shots,
            wall_seconds=time.perf_counter() - t0,
            queue_seconds=0.0,
            error_mitigation=None,
            metadata={"exact": shots is None, "circuit": spec.signature()},
        )
