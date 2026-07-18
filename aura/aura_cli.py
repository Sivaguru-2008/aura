"""AURA command-line entry point.

    py -m aura_cli train [n]       fit vision detectors + fusion models + calibration
    py -m aura_cli train-cnn [arch] fine-tune a CNN vision backbone on GPU (timm)
    py -m aura_cli bench [n]       quantum-vs-classical fusion benchmark + full metrics
    py -m aura_cli serve [port]    start gateway + dashboard
    py -m aura_cli demo            train (if needed) then serve

Run from the aura/ directory.  train-cnn arch ∈ {densenet121, efficientnetv2,
convnext, swin}; pass a manifest CSV via AURA_CNN_MANIFEST to train on real CXRs.
"""
from __future__ import annotations

import os
import sys

from common.config import ARTIFACTS


def _need_training() -> bool:
    return not (ARTIFACTS / "fusion_quantum.npz").exists()


def cmd_train(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 700
    from ml.training import train_vision, train_fusion
    train_vision.run(n)
    train_fusion.run(n)


def cmd_train_cnn(argv: list[str]) -> None:
    from ml.training.train_cnn import run, TrainConfig
    arch = argv[0] if argv else "densenet121"
    manifest = os.environ.get("AURA_CNN_MANIFEST")
    run(manifest=manifest, synthetic=manifest is None, cfg=TrainConfig(arch=arch))


def cmd_bench(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 500
    from ml.evaluation import benchmark
    benchmark.run(n)


def cmd_serve(argv: list[str]) -> None:
    port = int(argv[0]) if argv else 8000
    import uvicorn
    if _need_training():
        print("[aura] no trained models found — training first ...")
        cmd_train([])
    uvicorn.run("gateway.app:app", host="127.0.0.1", port=port, log_level="info")


def cmd_demo(argv: list[str]) -> None:
    if _need_training():
        cmd_train([])
        cmd_bench([])
    cmd_serve(argv)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    dispatch = {"train": cmd_train, "train-cnn": cmd_train_cnn, "bench": cmd_bench,
                "serve": cmd_serve, "demo": cmd_demo}
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    dispatch[cmd](argv)


if __name__ == "__main__":
    main()
