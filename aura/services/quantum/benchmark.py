"""Backend benchmarking and circuit-translation verification.

Two jobs, both about trusting a hardware number before acting on it:

:func:`verify_translation`
    Prove the Qiskit / Braket rebuilds compute the *same unitary* as the
    PennyLane circuit the model was trained on. Runs on statevector simulators,
    costs nothing, and must pass before any QPU time is spent — a translation
    bug produces well-formed, confidently wrong expectation values.

:func:`benchmark_backends`
    Compare candidate devices on AURA's real circuit: transpiled depth, two-qubit
    gate count, wall time, and — the number that decides whether a device is
    usable — mean absolute deviation of ``<Z_i>`` from the exact simulator.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aura.common.config import ARTIFACTS

from .base import CircuitSpec, ProviderKind, ProviderUnavailable
from .local import evaluate_spec
from .registry import get_provider

REPORT_PATH = ARTIFACTS / "quantum_backend_benchmark.json"


def served_vqc_spec(index: int | None = None) -> tuple[CircuitSpec, np.ndarray]:
    """The circuit AURA actually serves, on a real held-out evidence vector.

    Benchmarking a toy circuit would measure the device, not the deployment.
    """
    from aura.common.mathx import softmax
    from aura.services.fusion.quantum import QuantumFusion

    model = QuantumFusion.load()
    if model is None:
        raise FileNotFoundError("artifacts/fusion_quantum.npz absent — nothing to benchmark")

    evidence = ARTIFACTS / "quantum_study_evidence.npz"
    if evidence.exists():
        d = np.load(evidence)
        Xte = d["Xte"]
        if index is None:
            conf = [softmax(model.W @ model._expectations(x) + model.b).max() for x in Xte]
            index = int(np.argmax(conf))
        x = Xte[index].astype(float)
    else:
        x = np.full(model.n_qubits, 0.5)

    spec = CircuitSpec(
        kind="vqc", n_qubits=model.n_qubits, n_layers=model.n_layers,
        x=x, theta=model.theta, entangler=model.entangler,
    )
    return spec, model._expectations(x)


def verify_translation(spec: CircuitSpec, tol: float = 1e-9) -> dict[str, Any]:
    """Check every available SDK rebuild against the PennyLane reference."""
    reference = evaluate_spec(spec, shots=None)
    out: dict[str, Any] = {
        "circuit": spec.signature(),
        "kind": spec.kind,
        "n_qubits": spec.n_qubits,
        "reference": reference.tolist(),
        "tolerance": tol,
        "checks": {},
    }

    # --- Qiskit ---------------------------------------------------------- #
    try:
        from qiskit.quantum_info import Statevector

        from .ibm import build_circuit, z_observables

        qc = build_circuit(spec)
        sv = Statevector(qc)
        if spec.kind == "vqc":
            got = np.asarray([sv.expectation_value(o).real for o in z_observables(spec.n_qubits)])
        else:
            got = np.asarray([float(np.abs(sv.data[0]) ** 2)])
        diff = float(np.max(np.abs(got - reference)))
        out["checks"]["qiskit"] = {
            "available": True, "max_abs_diff": diff, "match": diff < tol,
            "logical_depth": int(qc.depth()),
            "two_qubit_gates": int(qc.num_nonlocal_gates()),
        }
    except ImportError:
        out["checks"]["qiskit"] = {"available": False, "reason": "qiskit not installed"}
    except Exception as exc:
        out["checks"]["qiskit"] = {"available": False, "reason": str(exc)}

    # --- Braket ---------------------------------------------------------- #
    try:
        from braket.devices import LocalSimulator

        from .braket import _expectations_from_counts, build_braket_circuit

        circuit = build_braket_circuit(spec)
        shots = 20000
        result = LocalSimulator().run(circuit, shots=shots).result()
        got, _ = _expectations_from_counts(result, spec)
        diff = float(np.max(np.abs(got - reference)))
        # Sampled, so compare against 5x the binomial standard error, not `tol`.
        sampling_tol = 5.0 / np.sqrt(shots)
        out["checks"]["braket"] = {
            "available": True, "max_abs_diff": diff,
            "match": diff < sampling_tol, "shots": shots,
            "sampling_tolerance": sampling_tol,
            "note": "sampled on the Braket local simulator; tolerance is 5 sigma of shot noise",
        }
    except ImportError:
        out["checks"]["braket"] = {"available": False, "reason": "amazon-braket-sdk not installed"}
    except Exception as exc:
        out["checks"]["braket"] = {"available": False, "reason": str(exc)}

    checked = [c for c in out["checks"].values() if c.get("available")]
    out["all_match"] = bool(checked) and all(c.get("match") for c in checked)
    out["n_checked"] = len(checked)
    return out


def benchmark_backends(
    spec: CircuitSpec | None = None,
    reference: np.ndarray | None = None,
    provider: str | None = None,
    backends: list[str] | None = None,
    shots: int = 4096,
    error_mitigation: str = "readout",
    include_local: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run AURA's served circuit on each candidate device and score the results.

    Every entry records whether it executed or why it did not, so an empty
    hardware section is legible rather than silent.
    """
    if spec is None:
        spec, reference = served_vqc_spec()
    if reference is None:
        reference = evaluate_spec(spec, shots=None)

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "circuit": {
            "kind": spec.kind, "n_qubits": spec.n_qubits, "n_layers": spec.n_layers,
            "entangler": spec.entangler, "signature": spec.signature(),
        },
        "reference_analytic": np.asarray(reference).tolist(),
        "translation": verify_translation(spec),
        "results": [],
    }

    if include_local:
        t0 = time.perf_counter()
        sampled = evaluate_spec(spec, shots=shots)
        report["results"].append({
            "provider": "local", "backend": "default.qubit", "simulator": True,
            "executed": True, "shots": shots,
            "wall_seconds": round(time.perf_counter() - t0, 4),
            "mean_abs_error_vs_analytic": float(np.mean(np.abs(sampled - reference))),
            "values": sampled.tolist(),
        })

    kinds = [ProviderKind(provider)] if provider else [ProviderKind.IBM, ProviderKind.BRAKET]
    for kind in kinds:
        p = get_provider(kind)
        ok, why = p.is_available()
        if not ok:
            report["results"].append(
                {"provider": kind.value, "executed": False, "reason": why})
            continue
        try:
            candidates = backends or [
                b.name for b in p.list_backends(min_qubits=spec.n_qubits) if not b.simulator
            ][:2]
        except Exception as exc:
            report["results"].append(
                {"provider": kind.value, "executed": False, "reason": str(exc)})
            continue

        for name in candidates:
            try:
                res = p.execute(spec, shots=shots, backend=name,
                                error_mitigation=error_mitigation)
                values = np.asarray(res.values, dtype=float)
                report["results"].append({
                    "provider": kind.value, "backend": res.backend, "simulator": False,
                    "executed": True, "shots": res.shots, "job_id": res.job_id,
                    "wall_seconds": round(res.wall_seconds, 2),
                    "transpiled_depth": res.transpiled_depth,
                    "transpiled_two_qubit_gates": res.transpiled_two_qubit_gates,
                    "error_mitigation": res.error_mitigation,
                    "mean_abs_error_vs_analytic": float(np.mean(np.abs(values - reference))),
                    "max_abs_error_vs_analytic": float(np.max(np.abs(values - reference))),
                    "values": values.tolist(),
                    "metadata": res.metadata,
                })
            except ProviderUnavailable as exc:
                report["results"].append(
                    {"provider": kind.value, "backend": name, "executed": False,
                     "reason": exc.reason})
            except Exception as exc:
                report["results"].append(
                    {"provider": kind.value, "backend": name, "executed": False,
                     "reason": f"{type(exc).__name__}: {exc}"})

    executed = [r for r in report["results"] if r.get("executed")]
    if executed:
        best = min(executed, key=lambda r: r["mean_abs_error_vs_analytic"])
        report["best_fidelity"] = {"backend": best["backend"],
                                   "mean_abs_error": best["mean_abs_error_vs_analytic"]}
    report["n_executed"] = len(executed)

    if write:
        Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
