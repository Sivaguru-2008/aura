"""Provider-agnostic types for quantum execution.

AURA's quantum work (the fusion VQC and the QKL fidelity kernel) is defined once,
in PennyLane, and must be executable on whatever hardware a deployment can reach:
a local simulator, an IBM Quantum device via Qiskit Runtime, or an AWS Braket
device (IonQ / Rigetti / IQM). This module fixes the vocabulary those providers
share so the engines never import a vendor SDK directly.

Two rules hold everywhere in this package:

1. **No vendor import at module scope.** ``qiskit`` and ``braket`` are optional.
   A deployment without them must import, start, and serve — reporting the
   provider as unavailable, never raising at import time.
2. **Failure degrades, it does not propagate.** A queue timeout, an expired
   credential, or a dead device falls back to the local simulator and says so in
   :attr:`ExecutionResult.fallback_reason`. Clinical inference never blocks on a
   third-party queue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ProviderKind(str, Enum):
    """Execution surfaces AURA knows how to target."""

    LOCAL = "local"
    IBM = "ibm"
    BRAKET = "braket"


class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RETIRED = "retired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BackendInfo:
    """One addressable device (or simulator)."""

    name: str
    provider: ProviderKind
    n_qubits: int
    simulator: bool
    status: DeviceStatus = DeviceStatus.UNKNOWN
    pending_jobs: int | None = None
    basis_gates: tuple[str, ...] = ()
    coupling_map_size: int | None = None
    #: Median two-qubit error, where the provider publishes it. The dominant
    #: term for the CNOT-ring ansatz — more informative than qubit count.
    median_ecr_error: float | None = None
    median_readout_error: float | None = None
    #: Per-shot price in USD where the provider charges per shot (Braket).
    cost_per_shot_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider.value,
            "n_qubits": self.n_qubits,
            "simulator": self.simulator,
            "status": self.status.value,
            "pending_jobs": self.pending_jobs,
            "basis_gates": list(self.basis_gates),
            "coupling_map_size": self.coupling_map_size,
            "median_ecr_error": self.median_ecr_error,
            "median_readout_error": self.median_readout_error,
            "cost_per_shot_usd": self.cost_per_shot_usd,
            "metadata": self.metadata,
        }


@dataclass
class JobHandle:
    """A submitted job, monitorable without blocking on it."""

    job_id: str
    provider: ProviderKind
    backend: str
    submitted_at: float = field(default_factory=time.time)
    status: str = "SUBMITTED"
    queue_position: int | None = None
    _native: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider.value,
            "backend": self.backend,
            "submitted_at": self.submitted_at,
            "elapsed_s": round(time.time() - self.submitted_at, 2),
            "status": self.status,
            "queue_position": self.queue_position,
        }


@dataclass
class ExecutionResult:
    """Expectation values from one circuit execution, with full provenance.

    ``values`` is always populated: on any provider failure it carries the local
    simulator's result and ``fell_back`` is ``True``. Callers therefore never
    need a try/except around execution — but they MUST surface ``fell_back`` in
    any report that claims hardware execution.
    """

    values: np.ndarray
    provider: ProviderKind
    backend: str
    shots: int | None
    stds: np.ndarray | None = None
    job_id: str | None = None
    wall_seconds: float = 0.0
    queue_seconds: float | None = None
    transpiled_depth: int | None = None
    transpiled_two_qubit_gates: int | None = None
    error_mitigation: str | None = None
    fell_back: bool = False
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": [float(v) for v in np.asarray(self.values).ravel()],
            "stds": [float(v) for v in np.asarray(self.stds).ravel()] if self.stds is not None else None,
            "provider": self.provider.value,
            "backend": self.backend,
            "shots": self.shots,
            "job_id": self.job_id,
            "wall_seconds": round(self.wall_seconds, 4),
            "queue_seconds": round(self.queue_seconds, 2) if self.queue_seconds is not None else None,
            "transpiled_depth": self.transpiled_depth,
            "transpiled_two_qubit_gates": self.transpiled_two_qubit_gates,
            "error_mitigation": self.error_mitigation,
            "fell_back": self.fell_back,
            "fallback_reason": self.fallback_reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CircuitSpec:
    """The circuit to execute, described independently of any SDK.

    AURA's two circuits are both fully determined by an encoding vector and a
    parameter tensor, so a provider only needs this plus the observable set.

    ``kind``:
      ``"vqc"``    — fusion variational circuit: RY(pi*x) encode, ``n_layers`` of
                     per-qubit (RY, RZ) plus an optional CNOT ring, ``<Z_i>`` readout.
      ``"iqp_kernel"`` — QKL fidelity kernel: IQP encode of ``x``, adjoint IQP
                     un-encode of ``x2``, probability of the all-zero string.
    """

    kind: str
    n_qubits: int
    n_layers: int = 0
    x: np.ndarray | None = None
    x2: np.ndarray | None = None
    theta: np.ndarray | None = None
    entangler: str = "ring"

    def signature(self) -> str:
        """Stable identity for caching and for logging what actually ran."""
        import hashlib

        h = hashlib.sha256()
        h.update(f"{self.kind}|{self.n_qubits}|{self.n_layers}|{self.entangler}".encode())
        for arr in (self.x, self.x2, self.theta):
            h.update(b"|")
            if arr is not None:
                h.update(np.ascontiguousarray(np.asarray(arr, dtype=float)).tobytes())
        return h.hexdigest()[:16]


class ProviderUnavailable(RuntimeError):
    """Raised inside a provider when its SDK, credentials, or devices are absent.

    Always caught by :mod:`aura.services.quantum.registry`, which falls back.
    """

    def __init__(self, provider: str, reason: str, remedy: str = ""):
        self.provider = provider
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"{provider} unavailable: {reason}" + (f" — {remedy}" if remedy else ""))


@runtime_checkable
class QuantumProvider(Protocol):
    """What every execution surface must offer."""

    kind: ProviderKind

    def is_available(self) -> tuple[bool, str]:
        """``(available, reason)``. Must never raise, and must not block on network."""
        ...

    def list_backends(self, min_qubits: int = 1) -> list[BackendInfo]:
        """Discoverable devices. Raises :class:`ProviderUnavailable` if it cannot."""
        ...

    def select_backend(self, min_qubits: int, prefer: str | None = None) -> BackendInfo:
        """Choose a device — ``prefer`` by name, else the least-busy suitable one."""
        ...

    def execute(
        self,
        spec: CircuitSpec,
        shots: int | None = None,
        backend: str | None = None,
        error_mitigation: str = "none",
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        """Run one circuit and return its expectation values."""
        ...
