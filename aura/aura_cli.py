"""AURA command-line entry point.

    py -m aura_cli train [n]     fit vision detectors + fusion models + calibration
    py -m aura_cli bench [n]     quantum-vs-classical fusion benchmark
    py -m aura_cli serve [port]  start gateway + dashboard
    py -m aura_cli demo          train (if needed) then serve

Run from the aura/ directory.
"""
from __future__ import annotations

import sys

from common.config import ARTIFACTS


def _need_training() -> bool:
    return not (ARTIFACTS / "fusion_quantum.npz").exists()


def cmd_train(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 700
    from ml.training import train_vision, train_fusion
    train_vision.run(n)
    train_fusion.run(n)


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
    dispatch = {"train": cmd_train, "bench": cmd_bench,
                "serve": cmd_serve, "demo": cmd_demo}
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    dispatch[cmd](argv)


if __name__ == "__main__":
    main()
