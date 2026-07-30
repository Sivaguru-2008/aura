"""IBM Quantum provider — Qiskit Runtime (EstimatorV2 / SamplerV2).

Credentials are the operator's and are never entered through AURA. Resolution
order, highest priority first:

1. ``QISKIT_IBM_TOKEN`` in the environment (shell export, or ``ibm_quantum.env``
   loaded by :func:`load_credentials`),
2. a saved account (``QiskitRuntimeService.save_account(...)``).

Nothing here writes, logs, or transmits a token.

Everything degrades. Missing ``qiskit-ibm-runtime``, absent credentials, no
operational device, or a queue that outlasts ``timeout_s`` all raise
:class:`ProviderUnavailable`, which :mod:`aura.services.quantum.registry`
converts into a local-simulator run with ``fell_back=True``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .base import (
    BackendInfo,
    CircuitSpec,
    DeviceStatus,
    ExecutionResult,
    JobHandle,
    ProviderKind,
    ProviderUnavailable,
)

_PLACEHOLDER = "PASTE_YOUR_IBM_QUANTUM_TOKEN_HERE"

#: Resilience levels understood by EstimatorV2, plus the local no-op.
ERROR_MITIGATION = {
    "none": 0,
    "readout": 1,      # TREX — twirled readout error extinction
    "zne": 2,          # zero-noise extrapolation (also enables gate twirling)
}


def load_credentials(start: Path | None = None) -> Path | None:
    """Populate ``os.environ`` from the nearest ``ibm_quantum.env``.

    Existing environment variables always win, so a shell-exported token is never
    overwritten by a file. Returns the file used, or ``None``.
    """
    here = Path(__file__).resolve()
    for base in [start or Path.cwd(), *here.parents]:
        f = base / "ibm_quantum.env"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        return f
    return None


def resolved_token() -> str | None:
    tok = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    return tok if tok and tok != _PLACEHOLDER else None


# --------------------------------------------------------------------------- #
# Circuit translation
# --------------------------------------------------------------------------- #
def build_circuit(spec: CircuitSpec):
    """Gate-for-gate Qiskit rebuild of the PennyLane circuit in ``spec``.

    The translation is verified against the statevector simulator by
    :func:`aura.services.quantum.benchmark.verify_translation`; a mismatch means
    hardware would execute a different unitary than the one the model was
    trained on.
    """
    from qiskit import QuantumCircuit

    n = spec.n_qubits
    qc = QuantumCircuit(n)

    if spec.kind == "vqc":
        x, theta = np.asarray(spec.x, float), np.asarray(spec.theta, float)
        for i in range(n):
            qc.ry(float(np.pi * x[i]), i)
        for layer in range(spec.n_layers):
            for i in range(n):
                qc.ry(float(theta[layer][i][0]), i)
                qc.rz(float(theta[layer][i][1]), i)
            if spec.entangler == "ring":
                for i in range(n):
                    qc.cx(i, (i + 1) % n)
        return qc

    if spec.kind == "iqp_kernel":
        x1, x2 = np.asarray(spec.x, float), np.asarray(spec.x2, float)
        for i in range(n):
            qc.h(i)
            qc.rz(float(np.pi * x1[i]), i)
        for i in range(n):
            j = (i + 1) % n
            qc.cx(i, j)
            qc.rz(float(np.pi * x1[i] * x1[j]), j)
            qc.cx(i, j)
        for i in reversed(range(n)):        # adjoint un-encode of x2
            j = (i + 1) % n
            qc.cx(i, j)
            qc.rz(float(-np.pi * x2[i] * x2[j]), j)
            qc.cx(i, j)
        for i in range(n):
            qc.rz(float(-np.pi * x2[i]), i)
            qc.h(i)
        return qc

    raise ValueError(f"unknown circuit kind {spec.kind!r}")


def z_observables(n_qubits: int):
    """``<Z_i>`` per qubit, endian-safe."""
    from qiskit.quantum_info import SparsePauliOp

    return [
        SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits)
        for i in range(n_qubits)
    ]


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class IBMProvider:
    """Qiskit Runtime execution with discovery, monitoring and mitigation."""

    kind = ProviderKind.IBM

    def __init__(self, channel: str = "ibm_quantum_platform", instance: str | None = None):
        self.channel = os.environ.get("QISKIT_IBM_CHANNEL", channel)
        self.instance = instance or os.environ.get("QISKIT_IBM_INSTANCE") or None
        self._service = None

    # -- availability ------------------------------------------------------ #
    def is_available(self) -> tuple[bool, str]:
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except ImportError:
            return False, "qiskit-ibm-runtime not installed (pip install 'aura[ibm]')"
        load_credentials()
        if resolved_token():
            return True, "token resolved from environment"
        try:  # a previously saved account is equally valid
            from qiskit_ibm_runtime import QiskitRuntimeService

            if QiskitRuntimeService.saved_accounts():
                return True, "saved IBM Quantum account"
        except Exception:
            pass
        return False, "no IBM Quantum credentials (set QISKIT_IBM_TOKEN or save an account)"

    def service(self):
        if self._service is None:
            ok, reason = self.is_available()
            if not ok:
                raise ProviderUnavailable(
                    "ibm", reason,
                    "get a free token at https://quantum.ibm.com and export QISKIT_IBM_TOKEN")
            from qiskit_ibm_runtime import QiskitRuntimeService

            kwargs: dict = {"channel": self.channel}
            if (tok := resolved_token()):
                kwargs["token"] = tok
            if self.instance:
                kwargs["instance"] = self.instance
            try:
                self._service = QiskitRuntimeService(**kwargs)
            except Exception as exc:
                raise ProviderUnavailable("ibm", f"could not open a runtime service: {exc}")
        return self._service

    # -- discovery --------------------------------------------------------- #
    def list_backends(self, min_qubits: int = 1, include_simulators: bool = True) -> list[BackendInfo]:
        svc = self.service()
        try:
            backends = svc.backends(min_num_qubits=min_qubits)
        except Exception as exc:
            raise ProviderUnavailable("ibm", f"backend discovery failed: {exc}")

        out: list[BackendInfo] = []
        for b in backends:
            try:
                cfg = b.configuration()
                sim = bool(getattr(cfg, "simulator", False))
                if sim and not include_simulators:
                    continue
                try:
                    status = b.status()
                    pending = int(getattr(status, "pending_jobs", 0))
                    online = bool(getattr(status, "operational", True))
                except Exception:
                    pending, online = None, True
                out.append(
                    BackendInfo(
                        name=b.name,
                        provider=ProviderKind.IBM,
                        n_qubits=int(getattr(cfg, "n_qubits", getattr(b, "num_qubits", 0))),
                        simulator=sim,
                        status=DeviceStatus.ONLINE if online else DeviceStatus.OFFLINE,
                        pending_jobs=pending,
                        basis_gates=tuple(getattr(cfg, "basis_gates", ()) or ()),
                        coupling_map_size=len(getattr(cfg, "coupling_map", []) or []) or None,
                        median_ecr_error=_median_two_qubit_error(b),
                        median_readout_error=_median_readout_error(b),
                        metadata={"processor": str(getattr(cfg, "processor_type", "") or "")},
                    )
                )
            except Exception:
                continue        # a single malformed backend must not kill discovery
        return out

    def select_backend(self, min_qubits: int, prefer: str | None = None) -> BackendInfo:
        """``prefer`` by name, otherwise IBM's own least-busy operational device."""
        svc = self.service()
        if prefer:
            try:
                b = svc.backend(prefer)
                return self._info(b)
            except Exception as exc:
                raise ProviderUnavailable("ibm", f"backend {prefer!r} unavailable: {exc}")
        try:
            b = svc.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
        except Exception as exc:
            raise ProviderUnavailable(
                "ibm", f"no operational device with >= {min_qubits} qubits: {exc}")
        return self._info(b)

    def _info(self, b) -> BackendInfo:
        cfg = b.configuration()
        try:
            st = b.status()
            pending, online = int(getattr(st, "pending_jobs", 0)), bool(getattr(st, "operational", True))
        except Exception:
            pending, online = None, True
        return BackendInfo(
            name=b.name,
            provider=ProviderKind.IBM,
            n_qubits=int(getattr(cfg, "n_qubits", getattr(b, "num_qubits", 0))),
            simulator=bool(getattr(cfg, "simulator", False)),
            status=DeviceStatus.ONLINE if online else DeviceStatus.OFFLINE,
            pending_jobs=pending,
            basis_gates=tuple(getattr(cfg, "basis_gates", ()) or ()),
            coupling_map_size=len(getattr(cfg, "coupling_map", []) or []) or None,
            median_ecr_error=_median_two_qubit_error(b),
            median_readout_error=_median_readout_error(b),
        )

    # -- execution --------------------------------------------------------- #
    def execute(
        self,
        spec: CircuitSpec,
        shots: int | None = None,
        backend: str | None = None,
        error_mitigation: str = "readout",
        timeout_s: float | None = 900.0,
    ) -> ExecutionResult:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import EstimatorV2

        if error_mitigation not in ERROR_MITIGATION:
            raise ValueError(
                f"unknown error_mitigation {error_mitigation!r}; "
                f"expected one of {sorted(ERROR_MITIGATION)}")

        info = self.select_backend(spec.n_qubits, prefer=backend)
        svc = self.service()
        dev = svc.backend(info.name)
        shots = int(shots or 4096)

        qc = build_circuit(spec)
        obs = z_observables(spec.n_qubits)

        # Dynamic qubit mapping: the preset pass manager lays the logical ring onto
        # the device's real coupling graph and routes what it cannot embed.
        #
        # optimization_level=3, not 1. The circuit is an 8-qubit CNOT ring and no IBM
        # device is fully connected, so closing the ring costs SWAPs — which are three
        # CNOTs each, on the error budget's most expensive gate. Level 3 runs the
        # heuristic layout search (SABRE) plus 1q/2q block resynthesis, which is
        # exactly the work that recovers those SWAPs. Levels 0-2 leave them in.
        #
        # The level-1 result is still transpiled, for one reason: it is the honest
        # baseline for the "we optimised the circuit" claim. Both depths and both
        # two-qubit gate counts go into the ExecutionResult, so the improvement is a
        # measured number in the run artifact rather than an assertion in a doc.
        baseline_qc = generate_preset_pass_manager(optimization_level=1, backend=dev).run(qc)
        pm = generate_preset_pass_manager(optimization_level=3, backend=dev, seed_transpiler=7)
        isa_qc = pm.run(qc)
        isa_obs = [o.apply_layout(isa_qc.layout) for o in obs]

        est = EstimatorV2(mode=dev)
        est.options.default_shots = shots
        try:
            est.options.resilience_level = ERROR_MITIGATION[error_mitigation]
        except Exception:
            pass        # older runtimes expose a different options tree; not fatal

        t0 = time.perf_counter()
        job = est.run([(isa_qc, isa_obs)])
        handle = JobHandle(job_id=job.job_id(), provider=ProviderKind.IBM,
                           backend=info.name, _native=job)

        res = self._await(job, handle, timeout_s)
        wall = time.perf_counter() - t0

        values = np.asarray(res.data.evs, dtype=float).ravel()
        stds = np.asarray(getattr(res.data, "stds", np.zeros_like(values)), dtype=float).ravel()
        return ExecutionResult(
            values=values,
            stds=stds,
            provider=ProviderKind.IBM,
            backend=info.name,
            shots=shots,
            job_id=handle.job_id,
            wall_seconds=wall,
            queue_seconds=handle.queue_position and None,
            transpiled_depth=int(isa_qc.depth()),
            transpiled_two_qubit_gates=int(isa_qc.num_nonlocal_gates()),
            error_mitigation=error_mitigation,
            metadata={
                "logical_depth": int(qc.depth()),
                # Transpilation before/after, so "we optimised the circuit" is a
                # number. 2q gates are the figure that matters: they dominate both
                # the error budget and the depth on a non-fully-connected device.
                "transpiler_optimization_level": 3,
                "baseline_optimization_level": 1,
                "baseline_depth": int(baseline_qc.depth()),
                "baseline_two_qubit_gates": int(baseline_qc.num_nonlocal_gates()),
                "two_qubit_gates_saved": int(baseline_qc.num_nonlocal_gates()
                                             - isa_qc.num_nonlocal_gates()),
                "depth_saved": int(baseline_qc.depth() - isa_qc.depth()),
                "backend_qubits": info.n_qubits,
                "median_ecr_error": info.median_ecr_error,
                "circuit": spec.signature(),
            },
        )

    def _await(self, job, handle: JobHandle, timeout_s: float | None):
        """Block on a job while keeping its queue state observable."""
        deadline = None if timeout_s is None else time.time() + timeout_s
        while True:
            try:
                status = str(job.status())
            except Exception:
                status = "UNKNOWN"
            handle.status = status
            if status.upper() in {"DONE", "COMPLETED"}:
                break
            if status.upper() in {"CANCELLED", "ERROR", "FAILED"}:
                raise ProviderUnavailable("ibm", f"job {handle.job_id} ended as {status}")
            if deadline and time.time() > deadline:
                try:
                    job.cancel()
                except Exception:
                    pass
                raise ProviderUnavailable(
                    "ibm",
                    f"job {handle.job_id} exceeded {timeout_s:.0f}s in the queue "
                    f"(last status {status}); cancelled")
            time.sleep(2.0)
        return job.result()[0]

    # -- monitoring -------------------------------------------------------- #
    def job_status(self, job_id: str) -> dict:
        """Poll one job without blocking — for the monitoring dashboard."""
        svc = self.service()
        try:
            job = svc.job(job_id)
        except Exception as exc:
            raise ProviderUnavailable("ibm", f"job {job_id} not found: {exc}")
        try:
            metrics = job.metrics()
        except Exception:
            metrics = {}
        return {
            "job_id": job_id,
            "status": str(job.status()),
            "backend": getattr(job.backend(), "name", "unknown"),
            "queue_seconds": (metrics or {}).get("usage", {}).get("quantum_seconds"),
            "created": str(getattr(job, "creation_date", "")),
            "metrics": metrics,
        }

    def retrieve(self, job_id: str) -> ExecutionResult:
        """Download an already-finished job. Costs no QPU time."""
        svc = self.service()
        job = svc.job(job_id)
        res = job.result()[0]
        values = np.asarray(res.data.evs, dtype=float).ravel()
        stds = np.asarray(getattr(res.data, "stds", np.zeros_like(values)), dtype=float).ravel()
        try:
            backend_name = job.backend().name
        except Exception:
            backend_name = "unknown"
        return ExecutionResult(
            values=values, stds=stds, provider=ProviderKind.IBM, backend=backend_name,
            shots=None, job_id=job_id, metadata={"retrieved_after_the_fact": True},
        )

    def queue_snapshot(self, min_qubits: int = 1) -> list[dict]:
        """Pending-job depth per device — what a scheduler needs to pick a target."""
        return sorted(
            (b.to_dict() for b in self.list_backends(min_qubits, include_simulators=False)),
            key=lambda d: (d["pending_jobs"] is None, d["pending_jobs"] or 0),
        )


def _median_two_qubit_error(backend) -> float | None:
    try:
        props = backend.properties()
        errs = [
            g.parameters[0].value
            for g in props.gates
            if g.gate in {"ecr", "cx", "cz"} and g.parameters
        ]
        return float(np.median(errs)) if errs else None
    except Exception:
        return None


def _median_readout_error(backend) -> float | None:
    try:
        props = backend.properties()
        errs = [props.readout_error(q) for q in range(backend.num_qubits)]
        return float(np.median([e for e in errs if e is not None])) or None
    except Exception:
        return None
