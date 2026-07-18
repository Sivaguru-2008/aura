# AURA — Performance Report

**Model:** DenseNet-121 (7-finding multilabel), input 1×1×224×224.
**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU · torch 2.11.0+cu128 · Windows 11 · Python 3.12.
**Command:** `python aura_cli.py benchmark` → `artifacts/performance/benchmark.json` (+ `PERFORMANCE_SUMMARY.md`).

> Note: the performance benchmark writes to `artifacts/performance/benchmark.json` and
> deliberately leaves the fusion `artifacts/benchmark.json` (read by `/v1/admin/safety`)
> untouched.

---

## 1. Latency (single image)

| device | mean | p50 | p95 | throughput |
|---|---|---|---|---|
| **GPU** | 29.1 ms | 26.1 ms | 42.9 ms | 34.4 img/s |
| **CPU** | 83.2 ms | 80.0 ms | 107.1 ms | 12.0 img/s |

Single-image latency includes per-call kernel-launch overhead; sustained throughput
is far higher under batching (below).

## 2. Batch throughput (GPU)

| batch size | img/s | ms/batch |
|---|---|---|
| 1 | 26.5 | 37.8 |
| 8 | 300.6 | 26.6 |
| 16 | 524.3 | 30.5 |
| **32** | **618.1** | 51.8 |
| 64 | 571.4 | 112.0 |

Throughput peaks at **~618 img/s** (batch 32) on this laptop GPU; batch 64 is slightly
lower (memory-bandwidth bound).

## 3. Mixed precision (AMP fp16)

| fp32 ms/batch(8) | AMP fp16 ms/batch(8) | speedup |
|---|---|---|
| 25.6 | 31.2 | **0.82×** |

On this GPU/workload AMP is **slower** (0.82×): DenseNet-121 at batch 8 is already
fast in fp32, and autocast cast overhead dominates at this small size. AMP typically
helps on larger models/batches; it is measured and reported honestly rather than
assumed. Keep fp32 for this configuration.

## 4. Memory

| metric | value |
|---|---|
| peak GPU allocated | 693 MB |
| peak GPU reserved | 1,283 MB |

The model comfortably fits a <2 GB GPU budget; several replicas fit on a single card.

## 5. End-to-end predict latency

`python aura_cli.py predict` measures the **full pipeline** (vision → fusion → safety →
explainability pre-pass → recommend → reasoning → report). The first invocation is
dominated by CUDA/cuDNN warm-up and the gradient-based explainability pass
(Integrated Gradients 32 steps + SmoothGrad 25 passes); expect a few seconds cold,
faster warm. The pure vision forward pass is the 29 ms above. For high-throughput
batch scoring, call the vision engine directly (Section 2 numbers apply).

## 6. Reproduce

```bash
python aura_cli.py benchmark            # iters=50
python aura_cli.py benchmark --iters 100
```

Warm-up iterations precede every timed loop; GPU timings use `torch.cuda.synchronize`.
