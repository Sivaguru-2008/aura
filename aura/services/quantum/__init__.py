"""Vendor-agnostic quantum execution for AURA.

    from aura.services.quantum import CircuitSpec, execute

    result = execute(CircuitSpec(kind="vqc", n_qubits=8, n_layers=3,
                                 x=evidence, theta=model.theta))
    result.fell_back        # True if this did not run where it was asked to
    result.backend          # where it actually ran

Enable hardware with ``AURA_USE_REAL_QPU=1`` (chain: IBM -> Braket -> local) or
pin one surface with ``AURA_QUANTUM_PROVIDER=ibm|braket|local``. Execution always
returns a result; see :mod:`aura.services.quantum.registry` for the fallback
contract.
"""
from .base import (
    BackendInfo,
    CircuitSpec,
    DeviceStatus,
    ExecutionResult,
    JobHandle,
    ProviderKind,
    ProviderUnavailable,
    QuantumProvider,
)
from .braket import BraketProvider
from .ibm import IBMProvider
from .local import LocalProvider, evaluate_spec
from .registry import (
    describe,
    execute,
    get_provider,
    list_backends,
    resolve_chain,
    select_backend,
    use_real_qpu,
)

__all__ = [
    "BackendInfo",
    "BraketProvider",
    "CircuitSpec",
    "DeviceStatus",
    "ExecutionResult",
    "IBMProvider",
    "JobHandle",
    "LocalProvider",
    "ProviderKind",
    "ProviderUnavailable",
    "QuantumProvider",
    "describe",
    "evaluate_spec",
    "execute",
    "get_provider",
    "list_backends",
    "resolve_chain",
    "select_backend",
    "use_real_qpu",
]
