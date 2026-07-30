# AURA — System & Model Benchmarks

This document records the quantitative results of AURA's performance benchmarks and the head-to-head comparison between classical and quantum evidence fusion backends.

---

## 0. What the quantum layer buys — and what it doesn't

Read this before the accuracy table, because the table alone answers the wrong question.

**No speed-up is claimed.** At 8 qubits this task is classically simulable. What the quantum
formulation provides is a different *resource model*: precision costs measurements,
`Var[⟨Z⟩] = (1 − ⟨Z⟩²) / n_shots`. Sequencing that budget splits "unresolved" into two states
with opposite clinical actions:

| verdict | meaning | action | count (447 studies) |
|---|---|---|---:|
| committed | margin separated from zero | report the diagnosis | 386 (86.4%) |
| **measurement-limited** | margin is real, precision was under-bought | run longer — AURA predicts how many shots | **1** |
| **model-limited** | tied at *infinite* precision | escalate to a human | **60** |

Median spend: **128 shots** against a fixed 512-shot budget. No classical fusion model can make
this distinction, because classical inference has no measurement budget to vary — a softmax of
0.55 means "unsure" and cannot say which kind. Served in `gateway/pipeline.py`.

**Circuit cost is engineered, not assumed.** Transpiled against the real `ibm_marrakesh`
coupling map (`artifacts/transpile_study.json`):

| optimization level | depth | two-qubit gates |
|---:|---:|---:|
| logical (pre-transpile) | 31 | 24 |
| 1 (previously served) | 185 | 75 |
| **3 (served)** | **146** | **60** |

A 20% reduction in the gate class that dominates the error budget. That this is the right lever
is measured, not assumed (`artifacts/noise_rung.json`): at 4096 shots, sampling contributes
mean \|Δ⟨Z⟩\| **0.0074** while device noise contributes **0.0544** — **88% of the error is
decoherence**, so shortening the circuit beats buying shots. Real hardware lands at 0.1858, ~3×
the static noise model, as drift and crosstalk are not in a calibration snapshot. The top-1
diagnosis survives all three rungs.

See [METHODOLOGY_MAPPING.md](METHODOLOGY_MAPPING.md) for the full problem → method mapping and
[DESIGN_SPACE.md](DESIGN_SPACE.md) for the qubits × layers × topology sweep.

---

## 1. Classical vs. Quantum Evidence Fusion

AURA evaluates evidence fusion models on a held-out test split of the real MIMIC-CXR evidence distribution. To make the comparison fair, **each backend is temperature-scaled on its own calibration split** before evaluation.

The results, reproducible via `py -m aura.aura_cli bench`, are summarized below:

All figures below are read directly from `aura/artifacts/benchmark.json` (`metrics_full`), the
file `bench` writes — they are not transcribed by hand. `n_eval = 69`.

| Backend | Accuracy | ECE (Calibration) | Macro AUROC | Correct / n | 95% CI (accuracy) |
|---|---|---|---|---|---|
| **Classical PoE** (Bayesian) | **0.6957** | 0.2194 | **0.7875** | 48 / 69 | [0.581, 0.795] |
| **Quantum VQC** (8-qubit) | 0.6377 | 0.2381 | 0.7696 | 44 / 69 | [0.520, 0.744] |
| **Learnable Head** (Linear) | 0.6232 | **0.1879** | 0.7306 | 43 / 69 | [0.506, 0.731] |
| Ensemble (quantum + classical) | 0.6957 | 0.2267 | 0.7798 | 48 / 69 | [0.581, 0.795] |

*Intervals are Jeffreys (Beta) intervals on the accuracy column, computed from the same
`n_eval = 69` the table is scored on. They are reproduced here rather than derived by the
reader because the sample is small enough that the point estimates alone mislead.*

### Analysis

* **The ordering is not statistically resolvable at this sample size.** Classical PoE
  leads the quantum VQC by **four cases out of 69** (48 vs 44 correct). The accuracy
  intervals above overlap across nearly their entire range, and a paired McNemar test
  cannot reach significance under *any* assignment of the discordant pairs — even the
  most favourable one, where all four break the same way, gives **p = 0.125**. Read the
  table as "these three backends perform comparably on this split", not as a ranking.
* **Why classical is nonetheless the served fair-accuracy reference.** Interpretability
  and cost, not a demonstrated accuracy advantage: the Product-of-Experts model exposes
  a per-evidence likelihood a clinician can inspect, and runs without a simulator. That
  is a design argument, and it does not depend on the four-case gap holding up.
* **No quantum advantage is claimed.** The 8-qubit VQC is competitive and correctly
  calibrated, but this task is low-dimensional and classically simulable, so there is no
  mechanism by which it *should* win. The dashboard and `artifacts/registry.json` say
  the same thing.
* **Per-class figures are directional only.** Support is thin — `malignancy` n=4 (macro
  sensitivity 0.000), `copd` n=4, `heart_failure` n=7, `pneumonia` n=8. Five of the six
  classes are in single digits.

> [!NOTE]
> Widening this split is the single highest-value change to these benchmarks. At n=69
> nothing short of a ~15-point accuracy gap would be detectable, so the current numbers
> can only support "comparable", never "better".

---

## 2. CNN Inference Latency & Throughput

Inference performance was measured on a standard evaluation platform:
* **Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (6GB VRAM), Intel i7 CPU (8-cores), Windows 11.
* **Model**: DenseNet-121 (7-finding multilabel), input size $1 \times 1 \times 224 \times 224$.

### Latency (Single Image)

| Device | Mean | p50 | p95 | Throughput |
|---|---|---|---|---|
| **GPU (RTX 5050)** | 29.1 ms | 26.1 ms | 42.9 ms | 34.4 images / sec |
| **CPU (8-core i7)** | 83.2 ms | 80.0 ms | 107.1 ms | 12.0 images / sec |

### Batch Throughput (GPU)

| Batch Size | Throughput (img/s) | Latency per Batch (ms) |
|---|---|---|
| 1 | 26.5 | 37.8 |
| 8 | 300.6 | 26.6 |
| 16 | 524.3 | 30.5 |
| **32** | **618.1** | **51.8** |
| 64 | 571.4 | 112.0 |

*Throughput peaks at batch size 32. At batch size 64, performance is memory-bandwidth limited.*

---

## 3. GPU Memory Footprint

| Metric | Value |
|---|---|
| **Peak GPU VRAM Allocated** | 693 MB |
| **Peak GPU VRAM Reserved** | 1,283 MB |

*The model runs comfortably within a 2 GB VRAM budget, permitting multiple server replicas on a single GPU.*

---

## 4. Execution Command

To run the latency and throughput benchmark suite locally:

```bash
cd aura-main   # the repository root: `aura` is the import root, so -m resolves from here
venv\Scripts\python.exe -m aura.aura_cli benchmark --iters 50
```
