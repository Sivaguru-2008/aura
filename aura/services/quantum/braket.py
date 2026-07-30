"""AWS Braket provider — IonQ, Rigetti, IQM, and Braket's managed simulators.

Optional. Requires ``amazon-braket-sdk`` plus AWS credentials resolvable by boto3
(``~/.aws/credentials``, an instance role, or ``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY`` in the environment). AURA never handles AWS secrets
itself — it only asks boto3 whether a session can be established.

Cost safety
-----------
Braket bills **per shot** on QPUs, so unlike the IBM path this provider will not
silently spend money:

* :attr:`BraketProvider.max_shots` caps any single submission,
* :meth:`estimate_cost` is called before every QPU run, and a run whose estimate
  exceeds ``cost_ceiling_usd`` is refused with :class:`ProviderUnavailable`
  (which falls back to the local simulator),
* the estimate is recorded in :attr:`ExecutionResult.metadata`.

Set ``AURA_BRAKET_COST_CEILING_USD`` to change the ceiling; it defaults to a
deliberately small value so an accidental production run cannot run up a bill.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .base import (
    BackendInfo,
    CircuitSpec,
    DeviceStatus,
    ExecutionResult,
    ProviderKind,
    ProviderUnavailable,
)

#: Device ARNs by short name. Availability varies by AWS region and over time —
#: :meth:`BraketProvider.list_backends` queries the live catalogue rather than
#: trusting this map, which exists so ``--backend ionq`` resolves to something.
KNOWN_DEVICES = {
    "ionq-aria": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    "ionq-forte": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    "rigetti-ankaa": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
    "iqm-garnet": "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "local": "braket_sv",
}

DEFAULT_COST_CEILING_USD = 5.0


class BraketProvider:
    """Execution on AWS Braket devices."""

    kind = ProviderKind.BRAKET

    def __init__(
        self,
        region: str | None = None,
        max_shots: int = 4096,
        cost_ceiling_usd: float | None = None,
    ):
        self.region = region or os.environ.get("AWS_REGION") or "us-east-1"
        self.max_shots = int(max_shots)
        self.cost_ceiling_usd = float(
            cost_ceiling_usd
            if cost_ceiling_usd is not None
            else os.environ.get("AURA_BRAKET_COST_CEILING_USD", DEFAULT_COST_CEILING_USD)
        )

    # -- availability ------------------------------------------------------ #
    def is_available(self) -> tuple[bool, str]:
        try:
            import braket  # noqa: F401
        except ImportError:
            return False, "amazon-braket-sdk not installed (pip install 'aura[braket]')"
        try:
            import boto3

            creds = boto3.Session(region_name=self.region).get_credentials()
            if creds is None:
                return False, "no AWS credentials resolvable by boto3"
        except ImportError:
            return False, "boto3 not installed"
        except Exception as exc:
            return False, f"AWS session could not be established: {exc}"
        return True, f"AWS Braket in {self.region}"

    def _require(self) -> None:
        ok, reason = self.is_available()
        if not ok:
            raise ProviderUnavailable(
                "braket", reason,
                "configure AWS credentials and install amazon-braket-sdk")

    # -- discovery --------------------------------------------------------- #
    def list_backends(self, min_qubits: int = 1) -> list[BackendInfo]:
        self._require()
        from braket.aws import AwsDevice

        try:
            devices = AwsDevice.get_devices()
        except Exception as exc:
            raise ProviderUnavailable("braket", f"device discovery failed: {exc}")

        out: list[BackendInfo] = []
        for d in devices:
            try:
                n_qubits = _device_qubits(d)
                if n_qubits < min_qubits:
                    continue
                dtype = str(getattr(d, "type", "")).upper()
                simulator = "SIMULATOR" in dtype
                status = str(getattr(d, "status", "")).upper()
                out.append(
                    BackendInfo(
                        name=d.name,
                        provider=ProviderKind.BRAKET,
                        n_qubits=n_qubits,
                        simulator=simulator,
                        status=DeviceStatus.ONLINE if status == "ONLINE" else DeviceStatus.OFFLINE,
                        pending_jobs=_queue_depth(d),
                        cost_per_shot_usd=_cost_per_shot(d, simulator),
                        metadata={
                            "arn": d.arn,
                            "provider_name": str(getattr(d, "provider_name", "")),
                            "type": dtype,
                        },
                    )
                )
            except Exception:
                continue
        return out

    def select_backend(self, min_qubits: int, prefer: str | None = None) -> BackendInfo:
        """``prefer`` may be a short name, a full ARN, or ``None`` (least busy)."""
        self._require()
        candidates = self.list_backends(min_qubits)
        if prefer:
            arn = KNOWN_DEVICES.get(prefer, prefer)
            for c in candidates:
                if prefer in (c.name,) or c.metadata.get("arn") in (arn, prefer):
                    return c
            raise ProviderUnavailable(
                "braket",
                f"device {prefer!r} not found among {len(candidates)} available devices")
        online = [c for c in candidates if c.status is DeviceStatus.ONLINE and not c.simulator]
        if not online:
            raise ProviderUnavailable(
                "braket", f"no online QPU with >= {min_qubits} qubits in {self.region}")
        return min(online, key=lambda c: (c.pending_jobs if c.pending_jobs is not None else 1 << 30))

    def estimate_cost(self, backend: BackendInfo, shots: int) -> float | None:
        """USD estimate for one submission, or ``None`` for free simulators."""
        if backend.simulator or backend.cost_per_shot_usd is None:
            return None
        # Braket QPU pricing is a fixed per-task fee plus a per-shot fee.
        return 0.30 + shots * backend.cost_per_shot_usd

    # -- execution --------------------------------------------------------- #
    def execute(
        self,
        spec: CircuitSpec,
        shots: int | None = None,
        backend: str | None = None,
        error_mitigation: str = "none",
        timeout_s: float | None = 900.0,
    ) -> ExecutionResult:
        self._require()
        from braket.aws import AwsDevice

        shots = min(int(shots or 1000), self.max_shots)
        info = self.select_backend(spec.n_qubits, prefer=backend)

        cost = self.estimate_cost(info, shots)
        if cost is not None and cost > self.cost_ceiling_usd:
            raise ProviderUnavailable(
                "braket",
                f"estimated cost ${cost:.2f} for {shots} shots on {info.name} exceeds the "
                f"${self.cost_ceiling_usd:.2f} ceiling",
                "raise AURA_BRAKET_COST_CEILING_USD if this spend is intended")

        circuit = build_braket_circuit(spec)
        device = AwsDevice(info.metadata["arn"])

        t0 = time.perf_counter()
        try:
            task = device.run(circuit, shots=shots)
            result = task.result()
        except Exception as exc:
            raise ProviderUnavailable("braket", f"task failed on {info.name}: {exc}")
        wall = time.perf_counter() - t0

        values, stds = _expectations_from_counts(result, spec)
        return ExecutionResult(
            values=values,
            stds=stds,
            provider=ProviderKind.BRAKET,
            backend=info.name,
            shots=shots,
            job_id=str(getattr(task, "id", "")),
            wall_seconds=wall,
            error_mitigation=None,   # Braket exposes no server-side mitigation here
            metadata={
                "arn": info.metadata.get("arn"),
                "estimated_cost_usd": cost,
                "provider_name": info.metadata.get("provider_name"),
                "circuit": spec.signature(),
            },
        )


def build_braket_circuit(spec: CircuitSpec):
    """Braket rebuild of the same unitary as :func:`aura.services.quantum.ibm.build_circuit`."""
    from braket.circuits import Circuit

    n = spec.n_qubits
    c = Circuit()

    if spec.kind == "vqc":
        x, theta = np.asarray(spec.x, float), np.asarray(spec.theta, float)
        for i in range(n):
            c.ry(i, float(np.pi * x[i]))
        for layer in range(spec.n_layers):
            for i in range(n):
                c.ry(i, float(theta[layer][i][0]))
                c.rz(i, float(theta[layer][i][1]))
            if spec.entangler == "ring":
                for i in range(n):
                    c.cnot(i, (i + 1) % n)
        return c

    if spec.kind == "iqp_kernel":
        x1, x2 = np.asarray(spec.x, float), np.asarray(spec.x2, float)
        for i in range(n):
            c.h(i)
            c.rz(i, float(np.pi * x1[i]))
        for i in range(n):
            j = (i + 1) % n
            c.cnot(i, j)
            c.rz(j, float(np.pi * x1[i] * x1[j]))
            c.cnot(i, j)
        for i in reversed(range(n)):
            j = (i + 1) % n
            c.cnot(i, j)
            c.rz(j, float(-np.pi * x2[i] * x2[j]))
            c.cnot(i, j)
        for i in range(n):
            c.rz(i, float(-np.pi * x2[i]))
            c.h(i)
        return c

    raise ValueError(f"unknown circuit kind {spec.kind!r}")


def _expectations_from_counts(result, spec: CircuitSpec) -> tuple[np.ndarray, np.ndarray]:
    """Reduce Braket measurement counts to the readout the circuit is defined by.

    ``vqc`` needs ``<Z_i>`` per qubit; ``iqp_kernel`` needs the all-zero-string
    probability. Both are computed from the same shot record, with binomial
    standard errors so the caller can see the shot noise.
    """
    counts = result.measurement_counts
    total = sum(counts.values()) or 1
    n = spec.n_qubits

    if spec.kind == "iqp_kernel":
        p = counts.get("0" * n, 0) / total
        return (
            np.asarray([p], dtype=float),
            np.asarray([np.sqrt(max(p * (1 - p), 0.0) / total)], dtype=float),
        )

    evs, stds = np.zeros(n), np.zeros(n)
    for i in range(n):
        p1 = sum(c for bits, c in counts.items() if bits[i] == "1") / total
        evs[i] = 1.0 - 2.0 * p1                       # <Z> = P(0) - P(1)
        stds[i] = 2.0 * np.sqrt(max(p1 * (1 - p1), 0.0) / total)
    return evs, stds


def _device_qubits(device) -> int:
    try:
        paradigm = device.properties.paradigm
        return int(getattr(paradigm, "qubitCount", 0))
    except Exception:
        return 0


def _queue_depth(device) -> int | None:
    try:
        depth = device.queue_depth()
        return int(getattr(depth, "quantum_tasks", {}).get("Normal", 0))
    except Exception:
        return None


def _cost_per_shot(device, simulator: bool) -> float | None:
    if simulator:
        return None
    provider = str(getattr(device, "provider_name", "")).lower()
    # Published per-shot QPU pricing as of writing; refreshed from the AWS
    # pricing page, and used only for the pre-flight ceiling check.
    return {"ionq": 0.03, "rigetti": 0.00035, "iqm": 0.00145, "oqc": 0.00035}.get(
        next((k for k in ("ionq", "rigetti", "iqm", "oqc") if k in provider), ""), None
    )
