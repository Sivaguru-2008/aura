"""Command line for the Brain Vision Engine.

    python -m backend.vision.brain.cli ingest   --corpus <path> [--max-subjects N]
    python -m backend.vision.brain.cli train    [--epochs N] [--batch-size N] ...
    python -m backend.vision.brain.cli evaluate [--split test]
    python -m backend.vision.brain.cli smoke    --corpus <path>
    python -m backend.vision.brain.cli info

``smoke`` runs ingest, training, and evaluation end to end on six subjects with a
three-stage network in about a minute. It exists so that "is the pipeline wired
correctly" is a question that can be answered without an afternoon of GPU time, and so
CI can answer it too.

The ``if __name__ == "__main__"`` guard is not decoration: on Windows, dataloader
workers are spawned rather than forked, and a module-level training call would be
re-executed in every worker.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from backend.core.shared.logging import get_logger
from backend.vision.brain.config import (
    BrainVisionConfig,
    IngestConfig,
    PathsConfig,
    smoke_config,
)
from backend.vision.brain.types import SplitName

log = get_logger("vision.brain.cli")


def build_parser() -> argparse.ArgumentParser:
    # The options every subcommand shares live on a parent parser rather than on the
    # top-level one. With argparse, a flag declared only at the top level must be
    # written *before* the subcommand — `cli --corpus X ingest`, never
    # `cli ingest --corpus X` — which is the opposite of what everyone types and of
    # what the documentation shows. Sharing them through `parents=` accepts the
    # natural order.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path, default=None,
                        help="BraTS2020 HDF5 corpus directory "
                             "(default: $AURA_BRATS_ROOT)")
    common.add_argument("--artifacts", type=Path, default=None,
                        help="output root (default: aura/artifacts/brain)")
    common.add_argument("--device", default=None, help="cuda | cpu")
    common.add_argument("--run-name", default=None)
    common.add_argument("--json", action="store_true",
                        help="print the result as JSON instead of a summary")

    parser = argparse.ArgumentParser(
        prog="brain-vision", parents=[common],
        description="AURA NeuroMind Brain Vision Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common],
                            help="build the standardised study cache")
    ingest.add_argument("--max-subjects", type=int, default=None)
    ingest.add_argument("--overwrite", action="store_true")

    train = sub.add_parser("train", parents=[common],
                           help="train the multi-task network")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--samples-per-epoch", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--grad-accum", type=int, default=None)
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--no-ema", action="store_true")
    train.add_argument("--no-curriculum", action="store_true")
    train.add_argument("--no-hard-mining", action="store_true")
    train.add_argument("--no-resume", action="store_true")

    evaluate = sub.add_parser("evaluate", parents=[common],
                              help="score a split with the best checkpoint")
    evaluate.add_argument("--split", choices=[s.value for s in SplitName],
                          default=SplitName.TEST.value)
    evaluate.add_argument("--full", action="store_true",
                          help="score the whole split, ignoring validation.max_batches")

    sub.add_parser("smoke", parents=[common],
                   help="tiny end-to-end run over a handful of subjects")
    sub.add_parser("info", parents=[common],
                   help="describe the cache, the checkpoints, and the registry")
    return parser


def config_from_args(args: argparse.Namespace) -> BrainVisionConfig:
    paths = PathsConfig()
    if args.corpus is not None:
        paths = replace(paths, corpus_root=Path(args.corpus))
    if args.artifacts is not None:
        paths = replace(paths, artifacts_root=Path(args.artifacts))

    config = BrainVisionConfig(paths=paths, device=args.device)
    if args.run_name:
        config = config.with_overrides(run_name=args.run_name)

    if args.command == "ingest":
        config = config.with_overrides(ingest=IngestConfig(
            max_subjects=args.max_subjects, overwrite=bool(args.overwrite)))
    elif args.command == "train":
        optim = config.optim
        sampling = config.sampling
        updates: dict[str, Any] = {}
        if args.epochs is not None:
            updates["epochs"] = args.epochs
        if args.batch_size is not None:
            updates["batch_size"] = args.batch_size
        if args.lr is not None:
            updates["lr"] = args.lr
        if args.num_workers is not None:
            updates["num_workers"] = args.num_workers
        if args.grad_accum is not None:
            updates["grad_accum"] = args.grad_accum
        if args.no_amp:
            updates["amp"] = False
        if args.no_ema:
            updates["ema"] = False
        if args.no_resume:
            updates["auto_resume"] = False
        if updates:
            optim = replace(optim, **updates)
        if args.samples_per_epoch is not None or args.no_hard_mining:
            sampling = replace(
                sampling,
                samples_per_epoch=args.samples_per_epoch or sampling.samples_per_epoch,
                hard_mining=not args.no_hard_mining)
        curriculum = (replace(config.curriculum, enabled=False) if args.no_curriculum
                      else config.curriculum)
        config = config.with_overrides(optim=optim, sampling=sampling,
                                       curriculum=curriculum)
    return config


# --------------------------------------------------------------------------- #
def command_ingest(config: BrainVisionConfig) -> dict[str, Any]:
    from backend.vision.brain.ingest import BrainCorpusIngestor

    manifest = BrainCorpusIngestor(config).run()
    return {
        "subjects": len(manifest.subjects),
        "slices": sum(s.slices_cached for s in manifest.subjects),
        "splits": {split.value: len(manifest.by_split(split)) for split in SplitName},
        "channel_verification": manifest.channel_verification,
        "cache": str(config.paths.cache_dir),
    }


def command_train(config: BrainVisionConfig) -> dict[str, Any]:
    from backend.vision.brain.train import BrainVisionTrainer

    trainer = BrainVisionTrainer(config)
    history = trainer.fit()
    best = history.best or {}
    test = trainer.evaluate(SplitName.TEST)
    result = {
        "epochs_completed": len(history.records),
        "best_epoch": best.get("epoch"),
        "best_value": best.get("monitor_value"),
        "monitor": config.optim.monitor,
        "test": test.to_dict(),
        "checkpoints": {
            "best": str(config.paths.best_model_path),
            "latest": str(config.paths.latest_model_path),
            "encoder": str(config.paths.encoder_path),
            "decoder": str(config.paths.decoder_path),
            "embedding_head": str(config.paths.embedding_head_path),
            "training_state": str(config.paths.training_state_path),
        },
        "model_card": str(config.paths.model_card_path),
    }
    (config.paths.report_dir / "test_report.json").write_text(
        json.dumps(test.to_dict(), indent=2, default=str), encoding="utf-8")
    return result


def command_evaluate(config: BrainVisionConfig, split: str,
                     full: bool = False) -> dict[str, Any]:
    import torch

    from backend.vision.brain.checkpoint import load_network_checkpoint
    from backend.vision.brain.dataset import build_datasets
    from backend.vision.brain.losses import MultiTaskLoss
    from backend.vision.brain.model.network import build_network
    from backend.vision.brain.validate import BrainValidator
    from torch.utils.data import DataLoader

    if full:
        config = config.with_overrides(
            validation=replace(config.validation, max_batches=None))
    device = torch.device(config.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets, _, _, _ = build_datasets(config)
    network = build_network(config.model).to(device)
    meta = load_network_checkpoint(config.paths.best_model_path, network,
                                   device=str(device))
    criterion = MultiTaskLoss(config.loss, heads=config.model.heads).to(device)
    loader = DataLoader(datasets[SplitName(split)],
                        batch_size=config.validation.batch_size, shuffle=False,
                        num_workers=config.optim.num_workers)
    report = BrainValidator(config, network, criterion, device).run(
        loader, epoch=meta.epoch, split=split)
    (config.paths.report_dir / f"{split}_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return report.to_dict()


def command_smoke(config: BrainVisionConfig) -> dict[str, Any]:
    from backend.vision.brain.ingest import BrainCorpusIngestor
    from backend.vision.brain.train import BrainVisionTrainer

    manifest = BrainCorpusIngestor(config).run()
    trainer = BrainVisionTrainer(config)
    history = trainer.fit()
    report = trainer.evaluate(SplitName.VAL)
    return {
        "subjects": len(manifest.subjects),
        "epochs": len(history.records),
        "val_composite_dice_mean":
            report.segmentation.get("composite_dice_mean"),
        "artifacts": str(config.paths.artifacts_root),
    }


def command_info(config: BrainVisionConfig) -> dict[str, Any]:
    from backend.vision.brain.model import available_encoders, declared_architectures

    info: dict[str, Any] = {
        "brain_vision_version": __import__(
            "backend.vision.brain.types", fromlist=["BRAIN_VISION_VERSION"]
        ).BRAIN_VISION_VERSION,
        "artifacts_root": str(config.paths.artifacts_root),
        "corpus_root": str(config.paths.corpus_root) if config.paths.corpus_root
                       else None,
        "registry": {"encoders": list(available_encoders()),
                     "declared": declared_architectures()},
        "checkpoints": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in (
                ("best", config.paths.best_model_path),
                ("latest", config.paths.latest_model_path),
                ("encoder", config.paths.encoder_path),
                ("decoder", config.paths.decoder_path),
                ("embedding_head", config.paths.embedding_head_path),
                ("training_state", config.paths.training_state_path))},
    }
    if config.paths.manifest_path.exists():
        from backend.vision.brain.ingest import load_manifest

        manifest = load_manifest(config)
        info["cache"] = {
            "created_at": manifest.created_at,
            "subjects": len(manifest.subjects),
            "slices": sum(s.slices_cached for s in manifest.subjects),
            "splits": {split.value: len(manifest.by_split(split))
                       for split in SplitName},
            "channel_verification": manifest.channel_verification,
            "caveats": manifest.caveats,
        }
    else:
        info["cache"] = None
    return info


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        config = smoke_config(args.corpus, args.artifacts)
        if args.device:
            config = config.with_overrides(device=args.device)
    else:
        config = config_from_args(args)

    handlers = {
        "ingest": lambda: command_ingest(config),
        "train": lambda: command_train(config),
        "evaluate": lambda: command_evaluate(config, args.split,
                                             full=bool(getattr(args, "full", False))),
        "smoke": lambda: command_smoke(config),
        "info": lambda: command_info(config),
    }
    result = handlers[args.command]()

    if args.json or args.command == "info":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
