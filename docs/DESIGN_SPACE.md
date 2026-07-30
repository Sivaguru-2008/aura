# AURA — Quantum Design-Space Sweep

Generated 2026-07-30T18:35:53Z from `aura/artifacts/design_sweep.json`. Every row is
trained and scored by this script; nothing here is transcribed.

**Protocol.** One split, one seed, one training function for every cell. Only qubits, layers and entangler topology vary. Trainable parameter count is identical across topologies at fixed (qubits, layers), so a topology difference is never a capacity difference. Split: 1338 train /
446 calibration / 447 test, patient-disjoint, source `mimic-cxr`,
20 epochs, seed 7.

> [!IMPORTANT]
> **The qubit axis is an evidence ablation.** The encoding is one evidence channel per qubit (RY(pi*x_i)), so n_qubits < evidence_channels DISCARDS channels. The qubit axis is therefore an evidence ablation as well as a width sweep: 8 qubits is set by the input dimension, not chosen for expressivity.

## Results

| qubits | layers | entangler | accuracy | NLL | ECE | 2q gates | params | train |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 8 | 1 | none | **0.5749** | **1.1996** | 0.0655 | 0 | 70 | 23.0s |
| 8 | 1 | full | 0.5638 | 1.2078 | 0.0592 | 28 | 70 | 34.1s |
| 8 | 3 | none | 0.5615 | 1.2489 | 0.0604 | 0 | 102 | 49.7s |
| 6 | 4 | none | 0.5593 | 1.2596 | 0.0916 | 0 | 90 | 32.5s |
| 8 | 2 | none | 0.5593 | 1.2052 | 0.0555 | 0 | 86 | 39.1s |
| 8 | 4 | none | 0.5593 | 1.2237 | 0.0615 | 0 | 118 | 66.8s |
| 8 | 4 | ring | 0.5548 | 1.2423 | 0.0614 | 32 | 118 | 72.3s |
| 8 | 2 | full | 0.5548 | 1.2282 | 0.0465 | 56 | 86 | 48.3s |
| 8 | 4 | full | 0.5548 | 1.2424 | 0.0464 | 112 | 118 | 95.7s |
| 8 | 3 | ring | 0.5481 | 1.2794 | 0.0708 | 24 | 102 | 64.2s | **<- served**
| 6 | 2 | none | 0.5436 | 1.2593 | 0.0782 | 0 | 66 | 18.5s |
| 6 | 1 | full | 0.5436 | 1.2698 | 0.0536 | 15 | 54 | 16.6s |
| 8 | 2 | ring | 0.5414 | 1.3040 | 0.0667 | 16 | 86 | 40.2s |
| 8 | 2 | linear | 0.5391 | 1.3046 | 0.0712 | 14 | 86 | 34.8s |
| 6 | 3 | linear | 0.5391 | 1.2854 | 0.0414 | 15 | 78 | 27.1s |
| 6 | 3 | ring | 0.5369 | 1.2833 | 0.0498 | 18 | 78 | 29.2s |
| 6 | 4 | linear | 0.5369 | 1.2757 | 0.0649 | 20 | 90 | 37.9s |
| 6 | 1 | none | 0.5347 | 1.2528 | 0.0523 | 0 | 54 | 12.9s |
| 6 | 2 | ring | 0.5347 | 1.2913 | 0.0631 | 12 | 66 | 23.0s |
| 8 | 4 | linear | 0.5347 | 1.2593 | 0.0385 | 28 | 118 | 67.3s |
| 6 | 2 | full | 0.5347 | 1.2801 | 0.0822 | 30 | 66 | 23.3s |
| 8 | 3 | linear | 0.5302 | 1.2733 | 0.0756 | 21 | 102 | 55.9s |
| 6 | 3 | none | 0.5280 | 1.2970 | 0.0471 | 0 | 78 | 27.0s |
| 6 | 4 | ring | 0.5235 | 1.2904 | 0.0509 | 24 | 90 | 40.7s |
| 6 | 4 | full | 0.5235 | 1.2762 | 0.0502 | 60 | 90 | 37.7s |
| 8 | 3 | full | 0.5213 | 1.2826 | 0.0386 | 84 | 102 | 76.0s |
| 8 | 1 | linear | 0.5190 | 1.3604 | 0.0744 | 7 | 70 | 27.0s |
| 6 | 1 | ring | 0.5168 | 1.3664 | 0.0874 | 6 | 54 | 13.1s |
| 6 | 3 | full | 0.5145 | 1.2809 | 0.0590 | 45 | 78 | 35.8s |
| 6 | 1 | linear | 0.5101 | 1.3725 | 0.0910 | 5 | 54 | 14.9s |
| 8 | 1 | ring | 0.5101 | 1.3567 | 0.0662 | 8 | 70 | 24.8s |
| 6 | 2 | linear | 0.5056 | 1.3080 | 0.0802 | 10 | 66 | 20.2s |
| 4 | 3 | linear | 0.4676 | 1.4004 | 0.0636 | 9 | 54 | 16.8s |
| 4 | 3 | full | 0.4653 | 1.4056 | 0.0488 | 18 | 54 | 18.3s |
| 4 | 1 | ring | 0.4631 | 1.4319 | 0.0666 | 4 | 38 | 14.5s |
| 4 | 2 | linear | 0.4631 | 1.4020 | 0.0522 | 6 | 46 | 12.9s |
| 4 | 2 | full | 0.4631 | 1.4128 | 0.0500 | 12 | 46 | 13.5s |
| 4 | 4 | linear | 0.4631 | 1.4094 | 0.0627 | 12 | 62 | 21.5s |
| 4 | 4 | full | 0.4631 | 1.4030 | 0.0769 | 24 | 62 | 22.9s |
| 4 | 2 | none | 0.4586 | 1.4034 | 0.0652 | 0 | 46 | 11.2s |
| 4 | 2 | ring | 0.4586 | 1.4318 | 0.0535 | 8 | 46 | 13.8s |
| 4 | 3 | none | 0.4564 | 1.4081 | 0.0800 | 0 | 54 | 15.0s |
| 4 | 1 | full | 0.4564 | 1.4226 | 0.0560 | 6 | 38 | 8.9s |
| 4 | 3 | ring | 0.4564 | 1.4154 | 0.0738 | 12 | 54 | 17.4s |
| 4 | 4 | none | 0.4541 | 1.4111 | 0.0817 | 0 | 62 | 15.7s |
| 4 | 1 | none | 0.4497 | 1.4146 | 0.0785 | 0 | 38 | 7.9s |
| 4 | 4 | ring | 0.4474 | 1.4141 | 0.0505 | 16 | 62 | 21.9s |
| 4 | 1 | linear | 0.4407 | 1.4537 | 0.0439 | 3 | 38 | 9.0s |

## Entangler topology (averaged over all depths at 8 qubits)

| topology | mean accuracy | mean NLL | mean ECE | mean 2q gates |
|:---|---:|---:|---:|---:|
| `none` | 0.5637 | 1.2193 | 0.0607 | 0.0 |
| `full` | 0.5487 | 1.2403 | 0.0477 | 70.0 |
| `ring` | 0.5386 | 1.2956 | 0.0663 | 20.0 |
| `linear` | 0.5308 | 1.2994 | 0.0649 | 17.5 |

**The CNOT ring does not earn its place, and the grid says so four ways.**
Averaged over all depths at 8 qubits, the product-state control (`none`, zero
two-qubit gates) reaches 0.5637 — better than every entangling
topology tested, including all-to-all at 70
two-qubit gates on average. The best single cell in the whole sweep is
8q / 1L / `none` at 0.5749 with
**0 two-qubit gates**.

This independently reproduces the dedicated entanglement ablation in
`artifacts/quantum_study.json`, which holds parameter count, seed and optimiser fixed and
bootstraps the difference over 447 studies: **NLL is significantly worse with the ring**
(Δ +0.056, CI [+0.022, +0.090]), while accuracy and ECE differences do not exclude zero.
Two independent experiments, same direction.

CNOT count is the cost axis that matters: two-qubit gates dominate the error budget and
the transpiled depth on hardware, while single-qubit rotations are cheap and
high-fidelity. `artifacts/noise_rung.json` measures this directly — 88% of the simulated
readout error on this circuit is device noise, not sampling — so a topology that buys
accuracy with CNOTs is spending on the expensive axis. Here it is not even buying
accuracy.

## Width (the evidence-ablation axis, `ring`)

| qubits | mean accuracy | evidence channels dropped |
|---:|---:|---:|
| 4 | 0.4564 | 4 |
| 6 | 0.5280 | 2 |
| 8 | 0.5386 | 0 |

Accuracy falls monotonically as channels are removed, which is the expected result and
confirms the reading in §3 of `METHODOLOGY_MAPPING.md`: **8 qubits is set by the
input dimension, not chosen for expressivity.** The width axis is not evidence about
quantum capacity; it is evidence that the clinical channels carry signal.

## The served configuration

The served configuration is **8q / 3L /
`ring`** at 0.5481 accuracy, 24 two-qubit gates
— which this grid does **not** show to be the best cell. It is published that way rather
than quietly re-tuned to whatever won, because the differences here are small relative to
the split (see *Honest reading*), and because changing a served clinical model on the
strength of an unreplicated grid search is precisely the move this project does not make.
The decision the grid actually supports is: *if the ring is kept, it is kept for
representational reasons, not for measured accuracy.*

## Honest reading

At this split size, differences of a percentage point or two in a *single cell* are not
resolvable — the same caution `docs/BENCHMARKS.md` applies to the headline backend
comparison applies here. What carries weight is the **consistency of the ordering across
16 cells per topology**, and its agreement with the independently-bootstrapped ablation.
A single cell in this table is weak evidence; the pattern across the grid, corroborated
by a second experiment, is not.

The gate-count column is exact arithmetic, not an estimate, and is the part of this table
that transfers to hardware unchanged.
