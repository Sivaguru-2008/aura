"""How much does transpilation actually buy? Measure it, on a real coupling map.

The serving VQC is an 8-qubit CNOT *ring*. No IBM device is fully connected — the
heavy-hex lattice gives each qubit two or three neighbours — so closing that ring
costs SWAPs, and a SWAP is three CNOTs on the gate the error budget cares about most.
How many of those SWAPs the transpiler recovers is a property of the optimisation
level, and it is measurable without spending a second of QPU time: the layout and
routing passes run locally against the device's published coupling map.

This script transpiles the served circuit at every preset level against the same
backend the hardware run used (``ibm_marrakesh``, via its fake//snapshot backend) and
reports depth and two-qubit gate count for each. That turns "we optimised the circuit"
from a claim into a table.

Run:  python -m aura.ml.evaluation.transpile_study
Out:  aura/artifacts/transpile_study.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from aura.common.config import ARTIFACTS, get_settings

BACKEND = "FakeMarrakesh"          # snapshot of the device the hardware run used
LEVELS = (0, 1, 2, 3)
SEED = 7


def _backend():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    return FakeMarrakesh()


def run() -> dict:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from aura.services.quantum.base import CircuitSpec
    from aura.services.quantum.ibm import build_circuit
    from aura.services.fusion.device import two_qubit_gate_count

    s = get_settings()
    n_qubits, n_layers = s.n_qubits, s.n_layers
    rng = np.random.default_rng(SEED)
    spec = CircuitSpec(
        kind="vqc",
        n_qubits=n_qubits,
        n_layers=n_layers,
        x=rng.random(n_qubits),
        theta=rng.normal(0, 0.3, size=(n_layers, n_qubits, 2)),
        entangler="ring",
    )
    qc = build_circuit(spec)
    dev = _backend()

    logical = {
        "depth": int(qc.depth()),
        "two_qubit_gates": int(qc.num_nonlocal_gates()),
        "expected_two_qubit_gates": two_qubit_gate_count("ring", n_qubits, n_layers),
    }

    rows = []
    for level in LEVELS:
        t0 = time.perf_counter()
        pm = generate_preset_pass_manager(optimization_level=level, backend=dev,
                                          seed_transpiler=SEED)
        isa = pm.run(qc)
        rows.append({
            "optimization_level": level,
            "depth": int(isa.depth()),
            "two_qubit_gates": int(isa.num_nonlocal_gates()),
            "total_gates": int(sum(isa.count_ops().values())),
            "transpile_seconds": round(time.perf_counter() - t0, 3),
        })

    base = next(r for r in rows if r["optimization_level"] == 1)
    served = next(r for r in rows if r["optimization_level"] == 3)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": BACKEND,
        "backend_qubits": dev.num_qubits,
        "circuit": {"n_qubits": n_qubits, "n_layers": n_layers, "entangler": "ring",
                    "encoding": "RY(pi*x_i)", "readout": "<Z_i>"},
        "logical": logical,
        "levels": rows,
        "served_level": 3,
        # Named "saved" rather than "delta": a bare +15 next to "level 3 vs level 1"
        # reads as level 3 costing more, which is the opposite of the result.
        "improvement_vs_level_1": {
            "two_qubit_gates_saved": base["two_qubit_gates"] - served["two_qubit_gates"],
            "two_qubit_gates_saved_pct": round(
                100.0 * (base["two_qubit_gates"] - served["two_qubit_gates"])
                / max(1, base["two_qubit_gates"]), 1),
            "depth_saved": base["depth"] - served["depth"],
            "depth_saved_pct": round(100.0 * (base["depth"] - served["depth"])
                                     / max(1, base["depth"]), 1),
        },
        "note": (
            "Transpiled against the published coupling map of the device the hardware "
            "run used; no QPU time is spent. The logical circuit has "
            f"{logical['two_qubit_gates']} two-qubit gates; every gate above that count "
            "is routing overhead the heavy-hex lattice forces on an 8-qubit ring. "
            "Two-qubit gates are the figure to compare: they dominate the error budget "
            "and the depth, while single-qubit rotations are cheap and high-fidelity."
        ),
    }


def main() -> None:
    study = run()
    out = ARTIFACTS / "transpile_study.json"
    out.write_text(json.dumps(study, indent=1), encoding="utf-8")
    imp = study["improvement_vs_level_1"]
    print(f"[transpile] backend {study['backend']} ({study['backend_qubits']} qubits)")
    print(f"[transpile] logical: depth {study['logical']['depth']}, "
          f"2q gates {study['logical']['two_qubit_gates']}")
    for r in study["levels"]:
        mark = "  <- served" if r["optimization_level"] == study["served_level"] else ""
        print(f"[transpile]   level {r['optimization_level']}: depth {r['depth']:>4}  "
              f"2q {r['two_qubit_gates']:>4}  total {r['total_gates']:>5}  "
              f"{r['transpile_seconds']:.2f}s{mark}")
    print(f"[transpile] level 3 saves {imp['two_qubit_gates_saved']} two-qubit gates "
          f"({imp['two_qubit_gates_saved_pct']:.1f}%) and {imp['depth_saved']} depth "
          f"({imp['depth_saved_pct']:.1f}%) vs level 1")
    print(f"[transpile] artifact -> {out}")


if __name__ == "__main__":
    main()
