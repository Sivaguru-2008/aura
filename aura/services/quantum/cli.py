"""Command line for the quantum execution layer.

    python -m aura.services.quantum.cli status
    python -m aura.services.quantum.cli backends --provider ibm --min-qubits 8
    python -m aura.services.quantum.cli verify
    python -m aura.services.quantum.cli benchmark --local-only
    python -m aura.services.quantum.cli run --provider ibm --shots 4096
    python -m aura.services.quantum.cli job <job_id>

``run`` against a QPU consumes real quota, so it refuses to submit unless
``--yes`` is given. ``status``, ``backends``, ``verify`` and ``job`` are
read-only and free.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def cmd_status(args) -> int:
    from . import describe

    info = describe()
    print(f"AURA_USE_REAL_QPU : {info['use_real_qpu']}")
    print(f"fallback chain    : {' -> '.join(info['chain'])}")
    print()
    for name, p in info["providers"].items():
        mark = "OK  " if p["available"] else "--  "
        print(f"  {mark}{name:8s} {p['reason']}")
    return 0


def cmd_backends(args) -> int:
    from . import list_backends

    rows = list_backends(provider=args.provider, min_qubits=args.min_qubits)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        if not r.get("name"):
            print(f"  -- {r.get('provider'):8s} unavailable: {r.get('reason')}")
            continue
        kind = "sim" if r["simulator"] else "QPU"
        print(f"  {r['provider']:7s} {r['name']:24s} {r['n_qubits']:4d}q {kind} "
              f"{r['status']:8s} pending={r['pending_jobs']} "
              f"2q_err={r['median_ecr_error']}")
    return 0


def cmd_verify(args) -> int:
    from .base import CircuitSpec
    from .benchmark import served_vqc_spec, verify_translation

    specs = []
    try:
        spec, _ = served_vqc_spec()
        specs.append(("served fusion VQC", spec))
    except Exception as exc:
        print(f"  (skipping served VQC: {exc})")
    rng = np.random.default_rng(0)
    specs.append(("QKL fidelity kernel",
                  CircuitSpec(kind="iqp_kernel", n_qubits=6, x=rng.random(6), x2=rng.random(6))))

    all_ok, n_checked = True, 0
    for label, spec in specs:
        r = verify_translation(spec)
        print(f"\n{label}  ({spec.n_qubits} qubits, {spec.n_layers} layers)")
        for sdk, c in r["checks"].items():
            if not c.get("available"):
                print(f"  -- {sdk:8s} {c.get('reason')}")
                continue
            n_checked += 1
            verdict = "MATCH" if c["match"] else "MISMATCH"
            print(f"  {'OK ' if c['match'] else 'XX '} {sdk:8s} "
                  f"max|diff|={c['max_abs_diff']:.2e}  {verdict}")
            all_ok &= bool(c["match"])

    if not n_checked:
        # Verifying nothing is not the same as verifying successfully — saying so
        # would be exactly the false assurance this command exists to prevent.
        print("\nNo hardware SDK installed, so NOTHING was verified. Install "
              "qiskit (pip install 'aura[ibm]') or amazon-braket-sdk before "
              "running on hardware.")
        return 2
    print(f"\nAll {n_checked} available SDK translation(s) match the PennyLane reference."
          if all_ok else "\nAT LEAST ONE TRANSLATION DIVERGES — do not run on hardware.")
    return 0 if all_ok else 1


def cmd_benchmark(args) -> int:
    from .benchmark import REPORT_PATH, benchmark_backends

    report = benchmark_backends(
        provider=None if args.local_only else args.provider,
        backends=args.backend,
        shots=args.shots,
        include_local=True,
    )
    for r in report["results"]:
        if not r.get("executed"):
            print(f"  -- {r['provider']:7s} {r.get('backend', ''):22s} {r.get('reason')}")
            continue
        print(f"  OK  {r['provider']:7s} {r['backend']:22s} "
              f"mean|d<Z>|={r['mean_abs_error_vs_analytic']:.4f} "
              f"wall={r['wall_seconds']}s depth={r.get('transpiled_depth')}")
    print(f"\nwrote {REPORT_PATH}")
    return 0


def cmd_run(args) -> int:
    from . import execute
    from .benchmark import served_vqc_spec

    spec, reference = served_vqc_spec(index=args.index)
    is_qpu = args.provider in {"ibm", "braket"}
    if is_qpu and not args.yes:
        print(f"Refusing to submit to {args.provider} without --yes.\n"
              f"This consumes real quota. Circuit: {spec.n_qubits} qubits, "
              f"{spec.n_layers} layers, {args.shots} shots.")
        return 2

    res = execute(spec, shots=args.shots, provider=args.provider,
                  backend=args.backend_name, error_mitigation=args.error_mitigation)
    values = np.asarray(res.values, dtype=float)
    print(f"  ran on   : {res.provider.value}/{res.backend}")
    print(f"  fell back: {res.fell_back}" + (f"  ({res.fallback_reason})" if res.fell_back else ""))
    print(f"  job id   : {res.job_id}")
    print(f"  wall     : {res.wall_seconds:.2f}s   mitigation={res.error_mitigation}")
    print(f"  mean|d<Z>| vs analytic: {np.mean(np.abs(values - reference)):.4f}")
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    return 0


def cmd_job(args) -> int:
    from . import ProviderKind, get_provider

    p = get_provider(ProviderKind.IBM)
    print(json.dumps(p.job_status(args.job_id), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aura-quantum", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="provider availability and fallback chain")
    s.set_defaults(func=cmd_status)

    b = sub.add_parser("backends", help="discover devices")
    b.add_argument("--provider", choices=["local", "ibm", "braket"], default=None)
    b.add_argument("--min-qubits", type=int, default=1)
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_backends)

    v = sub.add_parser("verify", help="check SDK circuit translations (free, no hardware)")
    v.set_defaults(func=cmd_verify)

    bm = sub.add_parser("benchmark", help="compare devices on AURA's served circuit")
    bm.add_argument("--provider", choices=["ibm", "braket"], default=None)
    bm.add_argument("--backend", action="append", default=None, help="repeatable")
    bm.add_argument("--shots", type=int, default=4096)
    bm.add_argument("--local-only", action="store_true", help="simulator only, no QPU time")
    bm.set_defaults(func=cmd_benchmark)

    r = sub.add_parser("run", help="execute the served circuit")
    r.add_argument("--provider", choices=["local", "ibm", "braket"], default="local")
    r.add_argument("--backend-name", default=None)
    r.add_argument("--shots", type=int, default=4096)
    r.add_argument("--index", type=int, default=None, help="held-out test case index")
    r.add_argument("--error-mitigation", choices=["none", "readout", "zne"], default="readout")
    r.add_argument("--yes", action="store_true", help="confirm real QPU submission")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_run)

    j = sub.add_parser("job", help="poll an IBM job by id (free)")
    j.add_argument("job_id")
    j.set_defaults(func=cmd_job)
    return ap


def main(argv: list[str] | None = None) -> int:
    _utf8()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
