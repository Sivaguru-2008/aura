"""Provider selection and the guaranteed-completion fallback chain.

The contract this module enforces is the one that makes real hardware safe to
enable in a clinical pipeline:

    **Execution always returns a result.**

:func:`execute` tries the requested provider; on *any* failure — SDK missing,
credentials absent, device offline, queue timeout, cost ceiling — it falls back
to the local simulator and returns a result with ``fell_back=True`` and a
human-readable ``fallback_reason``. Inference latency therefore never depends on
a third-party queue, and a report can always state truthfully where its numbers
came from.

Selection order
---------------
1. explicit ``provider=`` argument,
2. ``AURA_QUANTUM_PROVIDER`` (``local`` | ``ibm`` | ``braket``),
3. ``AURA_USE_REAL_QPU=1`` — try IBM, then Braket, then local,
4. local.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from aura.backend.core.shared.logging import get_logger

from .base import (
    BackendInfo,
    CircuitSpec,
    ExecutionResult,
    ProviderKind,
    ProviderUnavailable,
    QuantumProvider,
)
from .braket import BraketProvider
from .ibm import IBMProvider
from .local import LocalProvider, evaluate_spec

log = get_logger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[ProviderKind, QuantumProvider] = {}


def use_real_qpu() -> bool:
    """Whether this deployment has opted in to real hardware."""
    return str(os.environ.get("AURA_USE_REAL_QPU", "")).strip().lower() in {"1", "true", "yes", "on"}


def get_provider(kind: ProviderKind | str) -> QuantumProvider:
    """A cached provider instance. Construction never touches the network."""
    kind = ProviderKind(kind)
    with _LOCK:
        if kind not in _CACHE:
            _CACHE[kind] = {
                ProviderKind.LOCAL: LocalProvider,
                ProviderKind.IBM: IBMProvider,
                ProviderKind.BRAKET: BraketProvider,
            }[kind]()
        return _CACHE[kind]


def resolve_chain(provider: str | None = None) -> list[ProviderKind]:
    """The ordered list of providers to attempt.

    Local is always last, so the chain can never be empty.
    """
    if provider:
        chosen = ProviderKind(provider)
        return [chosen] if chosen is ProviderKind.LOCAL else [chosen, ProviderKind.LOCAL]

    env = os.environ.get("AURA_QUANTUM_PROVIDER", "").strip().lower()
    if env:
        try:
            chosen = ProviderKind(env)
        except ValueError:
            log.warning("unknown AURA_QUANTUM_PROVIDER=%r; using local", env)
            return [ProviderKind.LOCAL]
        return [chosen] if chosen is ProviderKind.LOCAL else [chosen, ProviderKind.LOCAL]

    if use_real_qpu():
        return [ProviderKind.IBM, ProviderKind.BRAKET, ProviderKind.LOCAL]
    return [ProviderKind.LOCAL]


def describe() -> dict[str, Any]:
    """Availability of every provider, for the status endpoint and the CLI.

    Cheap and non-blocking: reports whether SDKs and credentials are present
    without enumerating remote devices.
    """
    out: dict[str, Any] = {
        "use_real_qpu": use_real_qpu(),
        "chain": [k.value for k in resolve_chain()],
        "providers": {},
    }
    for kind in ProviderKind:
        try:
            available, reason = get_provider(kind).is_available()
        except Exception as exc:
            available, reason = False, f"provider construction failed: {exc}"
        out["providers"][kind.value] = {"available": available, "reason": reason}
    return out


def list_backends(provider: str | None = None, min_qubits: int = 1) -> list[dict]:
    """Discover devices on one provider (or every available provider)."""
    kinds = [ProviderKind(provider)] if provider else list(ProviderKind)
    found: list[dict] = []
    for kind in kinds:
        p = get_provider(kind)
        ok, reason = p.is_available()
        if not ok:
            found.append({"provider": kind.value, "available": False, "reason": reason})
            continue
        try:
            found.extend(b.to_dict() for b in p.list_backends(min_qubits=min_qubits))
        except ProviderUnavailable as exc:
            found.append({"provider": kind.value, "available": False, "reason": exc.reason})
        except Exception as exc:
            found.append({"provider": kind.value, "available": False, "reason": str(exc)})
    return found


def select_backend(min_qubits: int, provider: str | None = None,
                   prefer: str | None = None) -> BackendInfo:
    """Pick a device by walking the chain; raises only if every provider fails."""
    errors: list[str] = []
    for kind in resolve_chain(provider):
        try:
            return get_provider(kind).select_backend(min_qubits, prefer=prefer)
        except Exception as exc:
            errors.append(f"{kind.value}: {exc}")
    raise ProviderUnavailable("all", "; ".join(errors))


def execute(
    spec: CircuitSpec,
    shots: int | None = None,
    provider: str | None = None,
    backend: str | None = None,
    error_mitigation: str = "readout",
    timeout_s: float | None = 900.0,
) -> ExecutionResult:
    """Execute ``spec``, falling back to the local simulator on any failure.

    The returned :class:`ExecutionResult` is authoritative about where it ran:
    check ``fell_back`` before describing a number as a hardware measurement.
    """
    chain = resolve_chain(provider)
    reasons: list[str] = []

    for kind in chain:
        p = get_provider(kind)
        ok, why = p.is_available()
        if not ok:
            reasons.append(f"{kind.value}: {why}")
            continue
        try:
            result = p.execute(
                spec, shots=shots, backend=backend,
                error_mitigation=error_mitigation, timeout_s=timeout_s,
            )
            if reasons:      # succeeded, but not on the first choice
                result.fell_back = kind is not chain[0]
                result.fallback_reason = "; ".join(reasons) or None
            return result
        except ProviderUnavailable as exc:
            reasons.append(f"{kind.value}: {exc.reason}")
            log.warning("quantum provider %s unavailable: %s", kind.value, exc.reason)
        except Exception as exc:
            reasons.append(f"{kind.value}: {type(exc).__name__}: {exc}")
            log.warning("quantum provider %s failed: %s", kind.value, exc)

    # Every provider in the chain failed — including, somehow, the local one.
    # Evaluate directly rather than returning nothing to the clinical pipeline.
    import time

    t0 = time.perf_counter()
    values = evaluate_spec(spec, shots=shots)
    return ExecutionResult(
        values=values,
        provider=ProviderKind.LOCAL,
        backend="default.qubit",
        shots=shots,
        wall_seconds=time.perf_counter() - t0,
        fell_back=True,
        fallback_reason="; ".join(reasons) or "no provider available",
        metadata={"circuit": spec.signature()},
    )
