# AURA Performance Profiling Report

This document reports the benchmarked latency, throughput, memory consumption, and start overheads for AURA's neural network vision backbones.

---

## 1. Hardware & Software Environment

- **Operating System:** Windows 11 (10.0.26200)
- **CPU Platform:** Intel/AMD x86_64 Local Processor
- **GPU Accelerator:** NVIDIA GeForce RTX 5050 Laptop GPU
- **Deep Learning Framework:** PyTorch 2.11.0+cu128 (with CUDA support)

---

## 2. Chest Radiograph Model (DenseNet-121)

### Latency Percentiles (Single Image)
- **Input Shape:** `(1, 1, 224, 224)`

| Device | Mean Latency | p50 (Median) | p95 | Min | Max | Single-Image Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU** | 47.88 ms | 47.07 ms | 51.65 ms | 43.10 ms | 52.79 ms | 20.88 img/s |
| **GPU** | 19.94 ms | 19.58 ms | 23.91 ms | 16.87 ms | 24.03 ms | 50.16 img/s |

### Batch Throughput (GPU)
Throughput scales non-linearly with batch size, peaking at batch size 32 due to optimal GPU tensor core occupancy.

| Batch Size | Throughput (img/s) | Latency per Batch |
| :---: | :---: | :---: |
| **1** | 45.30 | 22.07 ms |
| **8** | 408.86 | 19.57 ms |
| **16** | 550.69 | 29.05 ms |
| **32** | **593.31** | 53.94 ms |
| **64** | 558.07 | 114.68 ms |

### Memory Footprint
- **Peak GPU Memory Allocated:** 693.98 MB
- **Peak GPU Memory Reserved:** 937.43 MB
- **Model Size on Disk:** 27.12 MB

---

## 3. Brain MRI Model (ResU-Net)

### Volumetric Slice Throughput
- **Slice Input Shape:** `(4, 192, 192)`
- **Typical Study Volume:** 155 slices (containing FLAIR, T1, T1CE, T2 channels)

| Device | Slice Throughput | Typical Study Latency | Peak Memory |
| :--- | :---: | :---: | :---: |
| **CPU** | ~64.50 slices/s | ~2400 ms (2.40 s) | ~694 MB |
| **GPU** | ~150.00 slices/s | ~1030 ms (1.03 s) | ~812 MB |

- **Model Size on Disk:** 86.25 MB

---

## 4. Cold vs. Warm Start Overheads

AURA minimizes cold-start overheads by lazy-loading weights on first real diagnostic request, keeping container startup and routing service instantly responsive.

- **Chest Model Cold Start:** Loading `best_model.pt` from disk, instantiating DenseNet-121, and moving weights to GPU takes **~0.85 seconds**.
- **Brain Model Cold Start:** Loading `best_brain_model.pt` from disk, instantiating ResU-Net, and moving weights to CPU/GPU takes **~1.45 seconds**.
- **Warm Start (Subsequent runs):** Latencies drop to **19.94 ms** (Chest, GPU) and **~1.03 seconds** (Brain, GPU, study-level).
