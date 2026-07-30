"""The missing rung between an ideal simulator and a real QPU.

The hardware run on ``ibm_marrakesh`` came back with a mean absolute error of 0.186
on the ``<Z_i>`` readout versus the analytic values, and the top-1 diagnosis survived.
Good — but 0.186 is an unattributed number. Two different things produce it:

  * **sampling** — a finite shot budget, which shrinks as 1/sqrt(shots) and is not a
    hardware defect at all; the ideal simulator has it too.
  * **decoherence and gate error** — T1/T2 relaxation and two-qubit gate infidelity,
    which do *not* shrink with more shots and are what "runs on real hardware" is
    actually testing.

Without a middle rung you cannot say which dominates, and therefore cannot say whether
more shots or a shallower circuit is the thing to spend effort on. This script adds
that rung: the same circuit, on the same device's *noise model*, at the same shot
count, locally and for free.

Three rungs, one circuit:

    analytic          exact statevector, infinite shots      (reference)
    shot-noise only   ideal simulator at N shots             (isolates sampling)
    noise model       FakeMarrakesh noise at N shots          (adds decoherence)
    [hardware]        ibm_marrakesh, recorded separately      (artifacts/ibm_hardware_run.json)

Run:  python -m aura.ml.evaluation.noise_rung [--shots 4096]
Out:  aura/artifacts/noise_rung.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aura.common.config import ARTIFACTS, get_settings
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES

BACKEND = "FakeMarrakesh"
SEED = 7


def _label(i: int) -> str:
    d = DIAGNOSES[i]
    return d.value if hasattr(d, "value") else str(d)


def _hardware_case() -> tuple[np.ndarray, dict] | tuple[None, None]:
    """Reuse the exact evidence vector the QPU ran, so the rungs are comparable."""
    path = ARTIFACTS / "ibm_hardware_run.json"
    if not path.exists():
        return None, None
    d = json.loads(path.read_text(encoding="utf-8"))
    x = d.get("case", {}).get("evidence_vector")
    return (np.asarray(x, dtype=float), d) if x else (None, None)


def run(shots: int = 4096) -> dict:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from aura.services.fusion.quantum import QuantumFusion
    from aura.services.quantum.base import CircuitSpec
    from aura.services.quantum.ibm import build_circuit, z_observables

    s = get_settings()
    model = QuantumFusion.load()
    x, hw_doc = _hardware_case()
    if x is None:
        rng = np.random.default_rng(SEED)
        x = rng.random(s.n_qubits)

    # Rung 1 — analytic. What the serving path actually uses.
    z_analytic = np.asarray(model._expectations(x), dtype=float)

    spec = CircuitSpec(kind="vqc", n_qubits=s.n_qubits, n_layers=s.n_layers,
                       x=x, theta=model.theta, entangler="ring")
    qc = build_circuit(spec)
    obs = z_observables(s.n_qubits)

    dev = FakeMarrakesh()
    pm = generate_preset_pass_manager(optimization_level=3, backend=dev, seed_transpiler=SEED)
    isa = pm.run(qc)

    def _estimate(sim, label: str) -> dict:
        from qiskit_ibm_runtime import EstimatorV2

        t0 = time.perf_counter()
        est = EstimatorV2(mode=sim)
        est.options.default_shots = shots
        res = est.run([(isa, [o.apply_layout(isa.layout) for o in obs])]).result()[0]
        z = np.asarray(res.data.evs, dtype=float).ravel()
        err = np.abs(z - z_analytic)
        post = softmax(model.W @ z + model.b)
        return {
            "rung": label,
            "shots": shots,
            "z": [round(float(v), 6) for v in z],
            "mean_abs_z_error_vs_analytic": round(float(err.mean()), 6),
            "max_abs_z_error_vs_analytic": round(float(err.max()), 6),
            "posterior": {_label(i): round(float(p), 6) for i, p in enumerate(post)},
            "top": _label(int(np.argmax(post))),
            "top1_agrees_with_analytic": bool(
                int(np.argmax(post)) == int(np.argmax(softmax(model.W @ z_analytic + model.b)))
            ),
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }

    # Rung 2 — shot noise only. Ideal simulator, no device error.
    ideal = _estimate(AerSimulator(seed_simulator=SEED), "shot_noise_only")
    # Rung 3 — device noise model on top of the same shot budget.
    noisy = _estimate(AerSimulator.from_backend(dev, seed_simulator=SEED), "noise_model")

    post_a = softmax(model.W @ z_analytic + model.b)
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend_model": BACKEND,
        "circuit": {"n_qubits": s.n_qubits, "n_layers": s.n_layers, "entangler": "ring",
                    "transpiled_depth": int(isa.depth()),
                    "transpiled_two_qubit_gates": int(isa.num_nonlocal_gates())},
        "evidence_vector": [round(float(v), 6) for v in x],
        "analytic": {
            "rung": "analytic",
            "z": [round(float(v), 6) for v in z_analytic],
            "posterior": {_label(i): round(float(p), 6) for i, p in enumerate(post_a)},
            "top": _label(int(np.argmax(post_a))),
        },
        "rungs": [ideal, noisy],
    }

    # Attribution — the number this script exists to produce.
    sampling = ideal["mean_abs_z_error_vs_analytic"]
    total_sim = noisy["mean_abs_z_error_vs_analytic"]
    out["attribution"] = {
        "sampling_only": sampling,
        "sampling_plus_device_noise": total_sim,
        "device_noise_component": round(max(0.0, total_sim - sampling), 6),
        "device_noise_share": round(max(0.0, total_sim - sampling) / total_sim, 3) if total_sim else None,
        "note": "Shot noise shrinks as 1/sqrt(shots); the device-noise component does "
                "not. If the device component dominates, buying more shots is the "
                "wrong lever and a shallower circuit is the right one.",
    }
    if hw_doc:
        hw = hw_doc.get("hardware", {})
        out["hardware_reference"] = {
            "backend": hw.get("backend"),
            "job_id": hw.get("job_id"),
            "mean_abs_z_error_vs_analytic": hw.get("mean_abs_z_error_vs_analytic"),
            "top1_agrees_with_analytic": hw.get("top1_agrees_with_analytic"),
            "note": "Measured on the real QPU. The simulated rungs above bracket it; "
                    "a real device also carries drift and crosstalk that a static "
                    "noise model does not reproduce, so the hardware error is "
                    "expected to sit at or above the noise-model rung.",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Noise-model rung between simulator and QPU")
    ap.add_argument("--shots", type=int, default=4096)
    args = ap.parse_args()

    study = run(shots=args.shots)
    path = ARTIFACTS / "noise_rung.json"
    path.write_text(json.dumps(study, indent=1), encoding="utf-8")

    a = study["attribution"]
    print(f"[noise] circuit: depth {study['circuit']['transpiled_depth']}, "
          f"2q gates {study['circuit']['transpiled_two_qubit_gates']}, {args.shots} shots")
    print(f"[noise] analytic top      : {study['analytic']['top']}")
    for r in study["rungs"]:
        print(f"[noise] {r['rung']:<17}: mean|dZ| {r['mean_abs_z_error_vs_analytic']:.4f}  "
              f"top {r['top']}  agrees={r['top1_agrees_with_analytic']}")
    if "hardware_reference" in study:
        h = study["hardware_reference"]
        print(f"[noise] hardware ({h['backend']}): mean|dZ| "
              f"{h['mean_abs_z_error_vs_analytic']:.4f}  agrees={h['top1_agrees_with_analytic']}")
    print(f"[noise] attribution: sampling {a['sampling_only']:.4f}, "
          f"device noise {a['device_noise_component']:.4f} "
          f"({(a['device_noise_share'] or 0) * 100:.0f}% of simulated error)")
    print(f"[noise] artifact -> {path}")


if __name__ == "__main__":
    main()
