"""Run AURA's *served* fusion VQC on a real IBM Quantum device.

This is the exact circuit `services.fusion.quantum.QuantumFusion` serves — same
trained parameters (`artifacts/fusion_quantum.npz`), same 8-qubit / 3-layer
RY-encoding + (RY,RZ)+CNOT-ring ansatz, same <Z_i> readout — rebuilt in Qiskit and
executed on hardware. It exists to turn the strategy bible's "projected, not yet
run" hardware line into a measured result with a job id behind it.

Two modes, on purpose:

    python -m ml.evaluation.run_ibm_hardware --mode local
        Rebuild the circuit and compute <Z_i> with an exact statevector, then diff
        against the PennyLane analytic path. No IBM account needed. This proves the
        Qiskit translation *is* the served circuit before any hardware time is spent.

    python -m ml.evaluation.run_ibm_hardware --mode hardware
        Submit the same circuit to the least-busy real IBM device via EstimatorV2 and
        record <Z_i> (with shot-noise std), the resulting posterior, backend name,
        job id and timestamp to artifacts/ibm_hardware_run.json.

Authentication (mode=hardware) is the user's, never entered here. Set it up once with
your own free IBM Quantum token:

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<YOUR_TOKEN>", overwrite=True)

or export QISKIT_IBM_TOKEN and pass --channel ibm_quantum_platform.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252, which cannot encode the glyphs used below
# (Δ, em-dash). Force UTF-8 so a print never crashes *after* a paid QPU job returns.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PLACEHOLDER = "PASTE_YOUR_IBM_QUANTUM_TOKEN_HERE"


def load_credentials() -> Path | None:
    """Populate os.environ from the first ``ibm_quantum.env`` found at or above cwd
    or the script location. Existing env vars win — a shell-exported token is never
    overwritten by the file. Returns the file used, or None."""
    here = Path(__file__).resolve()
    seen: set[Path] = set()
    for base in [Path.cwd(), *here.parents]:
        f = base / "ibm_quantum.env"
        if f in seen or not f.exists():
            seen.add(f)
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
    """The IBM token from the environment, or None if absent/unfilled."""
    tok = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    return tok or None

from aura.common.config import ARTIFACTS
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.fusion.quantum import QuantumFusion


def _label(i: int) -> str:
    d = DIAGNOSES[i]
    return d.value if hasattr(d, "value") else str(d)


def build_qiskit_circuit(x: np.ndarray, theta: np.ndarray, n_qubits: int,
                         n_layers: int):
    """Gate-for-gate rebuild of `services.fusion.device.make_qnode` (ring entangler)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(float(np.pi * x[i]), i)                 # angle-encode evidence channel i
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(float(theta[layer][i][0]), i)       # trainable RY
            qc.rz(float(theta[layer][i][1]), i)       # trainable RZ
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)              # CNOT ring
    return qc


def z_observables(n_qubits: int):
    """<Z_i> per qubit, endian-safe (Z placed on wire i regardless of convention)."""
    from qiskit.quantum_info import SparsePauliOp

    return [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits)]


def pick_case(model: QuantumFusion, index: int | None) -> tuple[np.ndarray, int, int]:
    """A real held-out evidence vector. Default: the most confident test case, so the
    hardware/analytic agreement is easy to read."""
    d = np.load(ARTIFACTS / "quantum_study_evidence.npz")
    Xte, yte = d["Xte"], d["yte"]
    if index is None:
        confid = np.array([softmax(model.W @ model._expectations(x) + model.b).max()
                           for x in Xte])
        index = int(confid.argmax())
    return Xte[index].astype(float), int(yte[index]), index


def local_check(model: QuantumFusion, x: np.ndarray) -> dict:
    """Exact statevector <Z_i> from the Qiskit circuit vs the PennyLane analytic path."""
    from qiskit.quantum_info import Statevector

    qc = build_qiskit_circuit(x, model.theta, model.n_qubits, model.n_layers)
    sv = Statevector(qc)
    z_qiskit = np.array([sv.expectation_value(ob).real for ob in z_observables(model.n_qubits)])
    z_penny = model._expectations(x)
    max_abs = float(np.max(np.abs(z_qiskit - z_penny)))
    print(f"  qubits={model.n_qubits} layers={model.n_layers} "
          f"depth={qc.depth()} 2q_gates={qc.num_nonlocal_gates()}")
    print(f"  max |<Z>_qiskit - <Z>_pennylane| = {max_abs:.2e}  "
          f"({'MATCH' if max_abs < 1e-9 else 'MISMATCH — circuit translation is wrong'})")
    return {"z_qiskit": z_qiskit.tolist(), "z_pennylane": z_penny.tolist(),
            "max_abs_diff": max_abs, "depth": int(qc.depth()),
            "two_qubit_gates": int(qc.num_nonlocal_gates())}


def hardware_run(model: QuantumFusion, x: np.ndarray, channel: str,
                 shots: int) -> dict:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

    # Token resolution, none of which routes through this chat:
    #   1. QISKIT_IBM_TOKEN in the environment (shell export or ibm_quantum.env)
    #   2. a previously saved account (QiskitRuntimeService.save_account(...))
    service = _make_service(channel)
    backend = service.least_busy(operational=True, simulator=False,
                                 min_num_qubits=model.n_qubits)
    print(f"  least-busy device: {backend.name} ({backend.num_qubits} qubits)")

    qc = build_qiskit_circuit(x, model.theta, model.n_qubits, model.n_layers)
    obs = z_observables(model.n_qubits)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_qc = pm.run(qc)
    isa_obs = [ob.apply_layout(isa_qc.layout) for ob in obs]

    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    print(f"  submitting {shots} shots to {backend.name} ...")
    job = est.run([(isa_qc, isa_obs)])
    print(f"  job id: {job.job_id()}  - waiting for result (may queue)...")
    res = job.result()[0]
    z_hw = np.asarray(res.data.evs, dtype=float).ravel()
    z_std = np.asarray(res.data.stds, dtype=float).ravel()
    return {"backend": backend.name, "backend_qubits": int(backend.num_qubits),
            "job_id": job.job_id(), "shots": int(shots),
            "transpiled_depth": int(isa_qc.depth()),
            "transpiled_two_qubit_gates": int(isa_qc.num_nonlocal_gates()),
            "z_hardware": z_hw.tolist(), "z_hardware_std": z_std.tolist()}


def _make_service(channel: str):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = resolved_token()
    if token == _PLACEHOLDER:
        raise SystemExit(
            "ibm_quantum.env still holds the placeholder token. Paste your real IBM "
            "Quantum token (https://quantum.ibm.com) into ibm_quantum.env and re-run.")
    kwargs: dict = {"channel": os.environ.get("QISKIT_IBM_CHANNEL", channel)}
    if token:
        kwargs["token"] = token
    if os.environ.get("QISKIT_IBM_INSTANCE"):
        kwargs["instance"] = os.environ["QISKIT_IBM_INSTANCE"]
    return QiskitRuntimeService(**kwargs)


def retrieve_run(model: QuantumFusion, job_id: str, channel: str) -> dict:
    """Re-fetch an already-completed hardware job by id. Costs no QPU time — the job
    ran server-side; this only downloads its stored result. Used to recover a run whose
    local process died after the device returned (e.g. a console-encoding crash)."""
    service = _make_service(channel)
    job = service.job(job_id)
    print(f"  retrieving job {job_id}: status={job.status()}")
    res = job.result()[0]
    z_hw = np.asarray(res.data.evs, dtype=float).ravel()
    z_std = np.asarray(res.data.stds, dtype=float).ravel()
    try:
        bname = job.backend().name
        bqubits = int(job.backend().num_qubits)
    except Exception:
        bname, bqubits = "unknown", 0
    return {"backend": bname, "backend_qubits": bqubits, "job_id": job_id,
            "retrieved_after_the_fact": True,
            "z_hardware": z_hw.tolist(), "z_hardware_std": z_std.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["local", "hardware"], default="local")
    ap.add_argument("--index", type=int, default=None,
                    help="held-out test index; default = most confident case")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--job-id", default=None,
                    help="retrieve an already-completed hardware job by id (no new QPU "
                         "cost) instead of submitting a fresh one")
    ap.add_argument("--out", default=str(ARTIFACTS / "ibm_hardware_run.json"))
    args = ap.parse_args()

    cred_file = load_credentials()
    if args.mode == "hardware":
        if cred_file:
            print(f"loaded credentials from {cred_file}")
        tok = resolved_token()
        if not tok or tok == _PLACEHOLDER:
            raise SystemExit(
                "No IBM Quantum token found. Edit ibm_quantum.env (replace the "
                f"'{_PLACEHOLDER}' line with your token) or export QISKIT_IBM_TOKEN, "
                "then re-run with --mode hardware.")

    model = QuantumFusion.load()
    if model is None:
        raise SystemExit("no fusion_quantum.npz — nothing to run")

    x, y_true, index = pick_case(model, args.index)
    z_analytic = model._expectations(x)
    post_analytic = softmax(model.W @ z_analytic + model.b)
    top = int(post_analytic.argmax())
    print(f"case #{index}: true={_label(y_true)}  analytic_top={_label(top)} "
          f"(p={post_analytic[top]:.3f})")

    record: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "circuit": {"n_qubits": model.n_qubits, "n_layers": model.n_layers,
                    "entangler": model.entangler, "encoding": "RY(pi*x_i)",
                    "readout": "<Z_i> -> W z + b -> softmax",
                    "model_version": model.model_version},
        "case": {"test_index": index, "true_label": _label(y_true),
                 "evidence_vector": x.tolist()},
        "analytic": {"z": z_analytic.tolist(),
                     "posterior": {_label(i): float(p) for i, p in enumerate(post_analytic)},
                     "top": _label(top)},
    }

    print("local statevector check:")
    record["local_check"] = local_check(model, x)

    if args.mode == "hardware":
        if args.job_id:
            print(f"retrieving prior hardware job {args.job_id}:")
            hw = retrieve_run(model, args.job_id, args.channel)
        else:
            print("hardware run:")
            hw = hardware_run(model, x, args.channel, args.shots)
        z_hw = np.asarray(hw["z_hardware"], dtype=float)
        post_hw = softmax(model.W @ z_hw + model.b)
        hw["posterior"] = {_label(i): float(p) for i, p in enumerate(post_hw)}
        hw["top"] = _label(int(post_hw.argmax()))
        hw["mean_abs_z_error_vs_analytic"] = float(np.mean(np.abs(z_hw - z_analytic)))
        hw["top1_agrees_with_analytic"] = bool(post_hw.argmax() == top)
        record["hardware"] = hw
        print(f"  hardware top={hw['top']}  agree={hw['top1_agrees_with_analytic']}  "
              f"mean|d<Z>|={hw['mean_abs_z_error_vs_analytic']:.3f}")

    out = Path(args.out)
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
