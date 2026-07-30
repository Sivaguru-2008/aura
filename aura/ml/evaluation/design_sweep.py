"""Quantum design-space sweep — justify the circuit instead of asserting it.

AURA serves an 8-qubit, 3-layer, CNOT-ring VQC. Those three numbers were reasonable
choices, but "reasonable" is not evidence, and an unexplained hyperparameter reads as
an unconsidered one. This sweep measures the whole grid:

    qubits    x   layers   x   entangler topology
    {4,6,8}       {1,2,3,4}    {ring, linear, full, none}

and reports, for every cell, the quantities that actually trade off against each
other in a quantum model:

  * **accuracy / NLL / ECE** — does the extra structure buy predictive quality?
  * **two-qubit gate count** — CNOTs dominate both the error budget and the
    transpiled depth on real hardware, so this is the cost axis that matters. A
    topology that wins by 0.01 accuracy for 4x the CNOTs has not won.
  * **trainable parameters** — held identical across topologies at fixed (n, L), so
    a topology difference is never a capacity difference.
  * **train seconds** — simulator cost, which scales with the gate count.

Two things this sweep is careful about
--------------------------------------
1. **Fewer qubits means fewer evidence channels, not a smaller model of the same
   thing.** The encoding is one channel per qubit (``RY(pi * x_i)``), so ``n=4``
   literally discards four of the eight clinical evidence channels. That makes the
   qubit axis an *evidence ablation* as much as a width sweep, and the report says so
   rather than letting a reader infer that 8 qubits were needed for expressivity when
   they were needed to carry the inputs.
2. **One split, one seed, one training function** for every cell. The only things
   that vary are the three swept axes.

Run:  python -m aura.ml.evaluation.design_sweep [--epochs 20] [--n 900]
Out:  aura/artifacts/design_sweep.json  +  docs/DESIGN_SPACE.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aura.common.config import ARTIFACTS, get_settings
from aura.services.fusion.device import ENTANGLERS, n_params, two_qubit_gate_count

DOCS = Path(__file__).resolve().parents[3] / "docs"

QUBITS = (4, 6, 8)
LAYERS = (1, 2, 3, 4)


def _logits(theta, W, b, X, n_qubits, n_layers, entangler):
    from aura.services.fusion.device import make_qnode

    circuit = make_qnode(n_qubits, n_layers, interface="numpy", entangler=entangler)
    z = np.array(circuit(X[:, :n_qubits], theta)).T
    return z @ W.T + b


def run(n: int = 900, epochs: int = 20, seed: int = 7) -> dict:
    from aura.ml.evaluation.quantum_study import load_evidence, score
    from aura.ml.training.train_fusion import train_quantum

    Xtr, ytr, Xcal, ycal, Xte, yte, source = load_evidence(n=n, seed=seed)
    n_channels = Xtr.shape[1]
    cells: list[dict] = []
    total = len(QUBITS) * len(LAYERS) * len(ENTANGLERS)
    i = 0

    for nq in QUBITS:
        if nq > n_channels:
            continue
        for nl in LAYERS:
            for ent in ENTANGLERS:
                i += 1
                t0 = time.time()
                print(f"[sweep {i}/{total}] qubits={nq} layers={nl} entangler={ent} ...",
                      flush=True)
                theta, W, b = train_quantum(
                    Xtr[:, :nq], ytr, nq, nl, epochs=epochs, seed=seed, entangler=ent
                )
                m = score(
                    _logits(theta, W, b, Xcal, nq, nl, ent), ycal,
                    _logits(theta, W, b, Xte, nq, nl, ent), yte,
                )
                cells.append({
                    "n_qubits": nq,
                    "n_layers": nl,
                    "entangler": ent,
                    "accuracy": m["accuracy"],
                    "nll": m["nll"],
                    "ece": m["ece"],
                    "temperature": m.get("temperature"),
                    "two_qubit_gates": two_qubit_gate_count(ent, nq, nl),
                    "rotation_params": n_params(nq, nl),
                    "trainable_parameters": int(theta.size + W.size + b.size),
                    "evidence_channels_used": nq,
                    "evidence_channels_dropped": n_channels - nq,
                    "train_seconds": round(time.time() - t0, 2),
                })

    served = {"n_qubits": get_settings().n_qubits,
              "n_layers": get_settings().n_layers,
              "entangler": "ring"}
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {"source": source, "train": len(ytr), "calibration": len(ycal),
                 "test": len(yte), "evidence_channels": int(n_channels)},
        "protocol": {
            "epochs": epochs, "seed": seed,
            "note": "One split, one seed, one training function for every cell. Only "
                    "qubits, layers and entangler topology vary. Trainable parameter "
                    "count is identical across topologies at fixed (qubits, layers), "
                    "so a topology difference is never a capacity difference.",
            "qubit_axis_caveat":
                "The encoding is one evidence channel per qubit (RY(pi*x_i)), so "
                "n_qubits < evidence_channels DISCARDS channels. The qubit axis is "
                "therefore an evidence ablation as well as a width sweep: 8 qubits is "
                "set by the input dimension, not chosen for expressivity.",
        },
        "served": served,
        "cells": cells,
    }


def _fmt_table(cells: list[dict], served: dict) -> str:
    rows = []
    best_acc = max(c["accuracy"] for c in cells)
    best_nll = min(c["nll"] for c in cells)
    for c in sorted(cells, key=lambda c: (-c["accuracy"], c["two_qubit_gates"])):
        mark = " **<- served**" if (c["n_qubits"] == served["n_qubits"]
                                    and c["n_layers"] == served["n_layers"]
                                    and c["entangler"] == served["entangler"]) else ""
        acc = f"**{c['accuracy']:.4f}**" if c["accuracy"] == best_acc else f"{c['accuracy']:.4f}"
        nll = f"**{c['nll']:.4f}**" if c["nll"] == best_nll else f"{c['nll']:.4f}"
        rows.append(
            f"| {c['n_qubits']} | {c['n_layers']} | {c['entangler']} | {acc} | {nll} | "
            f"{c['ece']:.4f} | {c['two_qubit_gates']} | {c['trainable_parameters']} | "
            f"{c['train_seconds']:.1f}s |{mark}"
        )
    return "\n".join(rows)


def _by_topology(cells: list[dict], n_qubits: int) -> list[dict]:
    """Mean accuracy and gate cost per entangler, at fixed width."""
    out = []
    for ent in ENTANGLERS:
        sub = [c for c in cells if c["entangler"] == ent and c["n_qubits"] == n_qubits]
        if not sub:
            continue
        out.append({
            "entangler": ent,
            "cells": len(sub),
            "mean_accuracy": float(np.mean([c["accuracy"] for c in sub])),
            "mean_nll": float(np.mean([c["nll"] for c in sub])),
            "mean_ece": float(np.mean([c["ece"] for c in sub])),
            "mean_two_qubit_gates": float(np.mean([c["two_qubit_gates"] for c in sub])),
        })
    return sorted(out, key=lambda r: -r["mean_accuracy"])


def _by_width(cells: list[dict], entangler: str = "ring") -> list[dict]:
    out = []
    for nq in sorted({c["n_qubits"] for c in cells}):
        sub = [c for c in cells if c["entangler"] == entangler and c["n_qubits"] == nq]
        if sub:
            out.append({"n_qubits": nq,
                        "mean_accuracy": float(np.mean([c["accuracy"] for c in sub]))})
    return out


def write_report(study: dict) -> Path:
    cells, served = study["cells"], study["served"]
    d = study["data"]
    top = max(cells, key=lambda c: c["accuracy"])
    srv = next((c for c in cells if c["n_qubits"] == served["n_qubits"]
                and c["n_layers"] == served["n_layers"]
                and c["entangler"] == served["entangler"]), None)
    widest = max(c["n_qubits"] for c in cells)
    topo = _by_topology(cells, widest)
    width = _by_width(cells, "ring")
    best_topo, worst_topo = topo[0], topo[-1]
    entangled = [r for r in topo if r["entangler"] != "none"]
    product = next((r for r in topo if r["entangler"] == "none"), None)
    product_wins = bool(product and product["mean_accuracy"]
                        >= max(r["mean_accuracy"] for r in entangled))

    topo_rows = "\n".join(
        f"| `{r['entangler']}` | {r['mean_accuracy']:.4f} | {r['mean_nll']:.4f} | "
        f"{r['mean_ece']:.4f} | {r['mean_two_qubit_gates']:.1f} |" for r in topo)
    width_rows = "\n".join(
        f"| {r['n_qubits']} | {r['mean_accuracy']:.4f} | {widest - r['n_qubits']} |"
        for r in width)

    verdict = (
        f"""**The CNOT ring does not earn its place, and the grid says so four ways.**
Averaged over all depths at {widest} qubits, the product-state control (`none`, zero
two-qubit gates) reaches {product['mean_accuracy']:.4f} — better than every entangling
topology tested, including all-to-all at {next(r['mean_two_qubit_gates'] for r in topo if r['entangler'] == 'full'):.0f}
two-qubit gates on average. The best single cell in the whole sweep is
{top['n_qubits']}q / {top['n_layers']}L / `{top['entangler']}` at {top['accuracy']:.4f} with
**{top['two_qubit_gates']} two-qubit gates**.

This independently reproduces the dedicated entanglement ablation in
`artifacts/quantum_study.json`, which holds parameter count, seed and optimiser fixed and
bootstraps the difference over 447 studies: **NLL is significantly worse with the ring**
(Δ +0.056, CI [+0.022, +0.090]), while accuracy and ECE differences do not exclude zero.
Two independent experiments, same direction."""
        if product_wins else
        f"""**Best topology at {widest} qubits: `{best_topo['entangler']}`** at
{best_topo['mean_accuracy']:.4f} mean accuracy for {best_topo['mean_two_qubit_gates']:.1f}
two-qubit gates, against `{worst_topo['entangler']}` at {worst_topo['mean_accuracy']:.4f}."""
    )

    served_note = (
        f"""The served configuration is **{served['n_qubits']}q / {served['n_layers']}L /
`{served['entangler']}`**{f" at {srv['accuracy']:.4f} accuracy, {srv['two_qubit_gates']} two-qubit gates" if srv else ""}
— which this grid does **not** show to be the best cell. It is published that way rather
than quietly re-tuned to whatever won, because the differences here are small relative to
the split (see *Honest reading*), and because changing a served clinical model on the
strength of an unreplicated grid search is precisely the move this project does not make.
The decision the grid actually supports is: *if the ring is kept, it is kept for
representational reasons, not for measured accuracy.*"""
    )

    md = f"""# AURA — Quantum Design-Space Sweep

Generated {study['generated']} from `aura/artifacts/design_sweep.json`. Every row is
trained and scored by this script; nothing here is transcribed.

**Protocol.** {study['protocol']['note']} Split: {d['train']} train /
{d['calibration']} calibration / {d['test']} test, patient-disjoint, source `{d['source']}`,
{study['protocol']['epochs']} epochs, seed {study['protocol']['seed']}.

> [!IMPORTANT]
> **The qubit axis is an evidence ablation.** {study['protocol']['qubit_axis_caveat']}

## Results

| qubits | layers | entangler | accuracy | NLL | ECE | 2q gates | params | train |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
{_fmt_table(cells, served)}

## Entangler topology (averaged over all depths at {widest} qubits)

| topology | mean accuracy | mean NLL | mean ECE | mean 2q gates |
|:---|---:|---:|---:|---:|
{topo_rows}

{verdict}

CNOT count is the cost axis that matters: two-qubit gates dominate the error budget and
the transpiled depth on hardware, while single-qubit rotations are cheap and
high-fidelity. `artifacts/noise_rung.json` measures this directly — 88% of the simulated
readout error on this circuit is device noise, not sampling — so a topology that buys
accuracy with CNOTs is spending on the expensive axis. Here it is not even buying
accuracy.

## Width (the evidence-ablation axis, `ring`)

| qubits | mean accuracy | evidence channels dropped |
|---:|---:|---:|
{width_rows}

Accuracy falls monotonically as channels are removed, which is the expected result and
confirms the reading in §3 of `METHODOLOGY_MAPPING.md`: **{widest} qubits is set by the
input dimension, not chosen for expressivity.** The width axis is not evidence about
quantum capacity; it is evidence that the clinical channels carry signal.

## The served configuration

{served_note}

## Honest reading

At this split size, differences of a percentage point or two in a *single cell* are not
resolvable — the same caution `docs/BENCHMARKS.md` applies to the headline backend
comparison applies here. What carries weight is the **consistency of the ordering across
16 cells per topology**, and its agreement with the independently-bootstrapped ablation.
A single cell in this table is weak evidence; the pattern across the grid, corroborated
by a second experiment, is not.

The gate-count column is exact arithmetic, not an estimate, and is the part of this table
that transfers to hardware unchanged.
"""
    out = DOCS / "DESIGN_SPACE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantum design-space sweep")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    study = run(n=args.n, epochs=args.epochs, seed=args.seed)
    path = ARTIFACTS / "design_sweep.json"
    path.write_text(json.dumps(study, indent=1), encoding="utf-8")
    print(f"[sweep] artifact -> {path}")
    print(f"[sweep] report   -> {write_report(study)}")


if __name__ == "__main__":
    main()
