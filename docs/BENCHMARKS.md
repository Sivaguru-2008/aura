# AURA — System & Model Benchmarks

This document records the quantitative results of AURA's performance benchmarks and the head-to-head comparison between classical and quantum evidence fusion backends.

---

## 1. Classical vs. Quantum Evidence Fusion

AURA evaluates evidence fusion models on a held-out test split of the real MIMIC-CXR evidence distribution. To make the comparison fair, **each backend is temperature-scaled on its own calibration split** before evaluation.

The results, reproducible via `py -m aura_cli bench`, are summarized below:

| Backend | Accuracy | ECE (Calibration) | Macro AUROC |
|---|---|---|---|
| **Classical PoE** (Bayesian) | **0.710** | **0.215** | **0.782** |
| **Quantum VQC** (8-qubit) | 0.667 | 0.237 | 0.765 |
| **Learnable Head** (Linear) | 0.652 | 0.189 | 0.747 |

### Analysis
* **Classical Superiority**: On the current real MIMIC-CXR distribution, the classical Product-of-Experts model slightly outperforms the 8-qubit quantum variational circuit (VQC) in accuracy, calibration, and classification performance.
* **Calibrated Parity**: The quantum model is competitive and valid, but does not yield a "quantum advantage" on this classically-simulable, low-dimensional task. This is stated honestly in the system dashboard and registry, avoiding marketing exaggeration.

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
cd aura
venv\Scripts\python.exe -m aura_cli benchmark --iters 50
```
