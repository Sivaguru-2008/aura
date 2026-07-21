"""Performance benchmark for the production vision model (Step 6).

Measures the numbers an MLOps reviewer asks for before a deploy:

    * CPU latency (single image)          * batch throughput (several batch sizes)
    * GPU latency (single image)          * single-image throughput
    * peak GPU / process memory           * mixed-precision (AMP) speedup

Writes ``artifacts/performance/benchmark.json``. It deliberately does **not** touch
``artifacts/benchmark.json`` (the quantum-vs-classical fusion benchmark the admin API
reads) so nothing downstream changes.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

from common.config import ARTIFACTS
from schemas.clinical import FINDINGS

PERF_DIR = ARTIFACTS / "performance"


def _percentiles(times: list[float]) -> dict:
    a = np.array(times) * 1000.0  # ms
    return {
        "mean_ms": round(float(a.mean()), 3),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
    }


def _build_model(device: str):
    import torch

    from ml.vision_cxr.model import DenseNet121CXR

    model = DenseNet121CXR(num_classes=len(FINDINGS))
    ckpt = ARTIFACTS / "best_model.pt"
    if ckpt.exists():
        state = torch.load(str(ckpt), map_location=device, weights_only=True)  # safe unpickler (audit §11.5)
        model.load_state_dict(state.get("model_state_dict", state))
    return model.to(device).eval()


def _time_latency(model, device, iters: int, warmup: int = 5) -> dict:
    import torch

    x = torch.randn(1, 1, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    d = _percentiles(times)
    d["throughput_img_per_s"] = round(1.0 / (d["mean_ms"] / 1000.0), 2)
    return d


def _batch_throughput(model, device, batch_sizes, iters: int = 20) -> dict:
    import torch

    out = {}
    for b in batch_sizes:
        try:
            x = torch.randn(b, 1, 224, 224, device=device)
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            imgs = b * iters
            out[str(b)] = {
                "img_per_s": round(imgs / dt, 2),
                "ms_per_batch": round(dt / iters * 1000, 3),
            }
        except RuntimeError as e:  # OOM at large batch — record and stop growing
            out[str(b)] = {"error": str(e)[:80]}
            break
    return out


def _amp_speedup(model, device, iters: int = 30) -> dict:
    import torch

    if device != "cuda":
        return {"available": False, "note": "AMP speedup measured on GPU only"}
    x = torch.randn(8, 1, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        fp32 = time.perf_counter() - t0
        for _ in range(5):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
        amp = time.perf_counter() - t0
    return {
        "available": True,
        "fp32_ms_per_batch": round(fp32 / iters * 1000, 3),
        "amp_fp16_ms_per_batch": round(amp / iters * 1000, 3),
        "speedup": round(fp32 / amp, 3),
    }


def _memory(device) -> dict:
    import torch

    out = {}
    if device == "cuda":
        out["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)
        out["gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 2)
        out["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        import psutil

        out["process_rss_mb"] = round(psutil.Process().memory_info().rss / 1e6, 2)
    except Exception:
        out["process_rss_mb"] = None
    return out


def run(iters: int = 50, out_dir: Optional[Path] = None) -> dict:
    import torch

    out_dir = Path(out_dir) if out_dir else PERF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    result = {
        "torch_version": torch.__version__,
        "cuda_available": has_cuda,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model": "DenseNet121CXR (7-finding multilabel)",
        "input_shape": [1, 1, 224, 224],
    }

    # CPU latency
    print("[benchmark] CPU latency ...")
    cpu_model = _build_model("cpu")
    result["cpu_latency"] = _time_latency(cpu_model, "cpu", iters=min(iters, 20))
    result["memory_after_cpu"] = _memory("cpu")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
        print("[benchmark] GPU latency + throughput + AMP ...")
        gpu_model = _build_model("cuda")
        result["gpu_latency"] = _time_latency(gpu_model, "cuda", iters=iters)
        result["batch_throughput"] = _batch_throughput(gpu_model, "cuda", [1, 8, 16, 32, 64])
        result["mixed_precision"] = _amp_speedup(gpu_model, "cuda")
        result["memory"] = _memory("cuda")
        result["single_image_throughput_img_per_s"] = result["gpu_latency"]["throughput_img_per_s"]
    else:
        result["batch_throughput"] = _batch_throughput(cpu_model, "cpu", [1, 4, 8])
        result["mixed_precision"] = {"available": False}
        result["memory"] = _memory("cpu")
        result["single_image_throughput_img_per_s"] = result["cpu_latency"]["throughput_img_per_s"]

    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _summary_md(result, out_dir / "PERFORMANCE_SUMMARY.md")
    _print_table(result)
    print(f"[benchmark] artifacts -> {path}")
    return result


def _print_table(r: dict) -> None:
    print("\n  Performance benchmark")
    print("  " + "-" * 46)
    if "cpu_latency" in r:
        print(f"  CPU latency (1 img)   {r['cpu_latency']['mean_ms']:.2f} ms  "
              f"({r['cpu_latency']['throughput_img_per_s']} img/s)")
    if "gpu_latency" in r:
        print(f"  GPU latency (1 img)   {r['gpu_latency']['mean_ms']:.2f} ms  "
              f"({r['gpu_latency']['throughput_img_per_s']} img/s)")
    if r.get("mixed_precision", {}).get("available"):
        print(f"  AMP fp16 speedup      {r['mixed_precision']['speedup']}x")
    if "batch_throughput" in r:
        best = max((v.get("img_per_s", 0) for v in r["batch_throughput"].values()), default=0)
        print(f"  Peak batch throughput {best} img/s")
    if r.get("memory", {}).get("gpu_peak_mb"):
        print(f"  Peak GPU memory       {r['memory']['gpu_peak_mb']} MB")
    print()


def _summary_md(r: dict, path: Path) -> None:
    L = ["# Performance Benchmark", "",
         f"- **Device:** {r.get('memory', {}).get('gpu_name', 'CPU')} · torch {r['torch_version']}",
         f"- **Platform:** {r['platform']}", ""]
    if "cpu_latency" in r:
        c = r["cpu_latency"]
        L.append(f"- **CPU latency (1 img):** {c['mean_ms']} ms "
                 f"(p95 {c['p95_ms']} ms, {c['throughput_img_per_s']} img/s)")
    if "gpu_latency" in r:
        g = r["gpu_latency"]
        L.append(f"- **GPU latency (1 img):** {g['mean_ms']} ms "
                 f"(p95 {g['p95_ms']} ms, {g['throughput_img_per_s']} img/s)")
    if r.get("mixed_precision", {}).get("available"):
        mp = r["mixed_precision"]
        L.append(f"- **Mixed precision:** {mp['speedup']}x "
                 f"({mp['fp32_ms_per_batch']} → {mp['amp_fp16_ms_per_batch']} ms/batch of 8)")
    if r.get("memory", {}).get("gpu_peak_mb"):
        L.append(f"- **Peak GPU memory:** {r['memory']['gpu_peak_mb']} MB")
    if "batch_throughput" in r:
        L += ["", "## Batch throughput", "", "| batch | img/s | ms/batch |", "|---|---|---|"]
        for b, v in r["batch_throughput"].items():
            if "img_per_s" in v:
                L.append(f"| {b} | {v['img_per_s']} | {v['ms_per_batch']} |")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    run()
