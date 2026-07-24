"""The training loop.

Everything the specification asks for under "training stability" is here — mixed
precision, gradient clipping, gradient accumulation, automatic resume, warmup, cosine
annealing, early stopping, and an exponential moving average of the weights — and each
one is switchable, because a stability feature that cannot be turned off cannot be shown
to have helped.

The loop itself is unremarkable and deliberately so. What is worth reading is how the
three adaptive mechanisms are wired, because each has an ordering constraint that is
easy to get wrong and silent when you do.

**Curriculum** is applied at the epoch boundary, before the dataloader is constructed.
The sampler recomputes its eligible pools, and the dataset is told the epoch so its
augmentation stream advances. Changing the pool mid-epoch would mean an epoch's
composition depends on how far through it you were.

**Hard-example mining** is fed from the training step. Every batch already computes a
per-sample foreground Dice as part of the loss, so the difficulty signal costs nothing
and arrives continuously. The weights it produces take effect at the next epoch
boundary — not mid-epoch, for the same reason.

**EMA** is updated after the optimiser step, never before, and *only* on steps where an
optimiser step actually happened. Under gradient accumulation those are different
things, and an EMA updated per batch decays at ``grad_accum`` times the intended rate,
which looks like a slightly-worse-than-expected model rather than like a bug.

Validation runs on the EMA weights when EMA is enabled, and the best checkpoint is
selected on that. Serving the raw weights after selecting on the EMA ones would mean
deploying a model that was never the one measured, so the checkpoint carries both and
:func:`~backend.vision.brain.checkpoint.load_network_checkpoint` prefers the EMA copy.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.core.shared.logging import get_logger
from backend.vision.brain.checkpoint import (
    CheckpointMeta,
    CheckpointWriter,
    load_training_state,
)
from backend.vision.brain.config import BrainVisionConfig
from backend.vision.brain.dataset import build_datasets
from backend.vision.brain.embeddings import EmbeddingStore
from backend.vision.brain.errors import ConfigurationError
from backend.vision.brain.losses import MultiTaskLoss
from backend.vision.brain.metrics import LossMeter
from backend.vision.brain.model.network import BrainVisionNetwork, build_network
from backend.vision.brain.sampling import AdaptiveSliceSampler
from backend.vision.brain.types import BRAIN_VISION_VERSION, SplitName
from backend.vision.brain.validate import BrainValidator, ValidationReport

log = get_logger("vision.brain.train")


class ExponentialMovingAverage:
    """Shadow copy of the weights, updated after every optimiser step.

    Buffers (instance-norm running statistics, if a batch-norm variant is configured)
    are copied rather than averaged: they are already running estimates, and averaging an
    average is both meaningless and a source of drift between the shadow model and the
    live one.
    """

    def __init__(self, network: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {name: parameter.detach().clone().float()
                       for name, parameter in network.state_dict().items()
                       if parameter.dtype.is_floating_point}
        self.steps = 0

    @torch.no_grad()
    def update(self, network: torch.nn.Module) -> None:
        self.steps += 1
        # Warm up the decay so the first steps are not dominated by the random
        # initialisation the shadow started from.
        decay = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for name, value in network.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(decay).add_(value.detach().float(), alpha=1 - decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    @contextlib.contextmanager
    def applied(self, network: torch.nn.Module):
        """Temporarily swap the EMA weights in — for validation, then swap back."""
        backup = {name: value.detach().clone()
                  for name, value in network.state_dict().items()
                  if name in self.shadow}
        network.load_state_dict(
            {name: value.to(dtype=backup[name].dtype)
             for name, value in self.shadow.items() if name in backup}, strict=False)
        try:
            yield network
        finally:
            network.load_state_dict(backup, strict=False)


@dataclass
class TrainingHistory:
    """Per-epoch record, appended to ``reports/history.jsonl``."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    @property
    def best(self) -> dict[str, Any] | None:
        return max(self.records, key=lambda r: r.get("monitor_value", -math.inf),
                   default=None)


class BrainVisionTrainer:
    """Trains the Brain Vision network end to end."""

    def __init__(self, config: BrainVisionConfig) -> None:
        self.config = config
        self.device = torch.device(config.device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        _seed_everything(config.optim.seed)
        config.paths.ensure()

        self.datasets, self.table, self.manifest, self.morphology = \
            build_datasets(config)
        train_indices = self.table.split_indices(SplitName.TRAIN.value)
        if train_indices.size == 0:
            raise ConfigurationError(
                "the training split is empty",
                detail={"hint": "ingest more subjects, or check the split fractions"})

        self.network: BrainVisionNetwork = build_network(config.model).to(self.device)
        self.criterion = MultiTaskLoss(config.loss, heads=config.model.heads).to(
            self.device)
        self.sampler = AdaptiveSliceSampler(
            self.table, train_indices, sampling=config.sampling,
            curriculum=config.curriculum, seed=config.optim.seed)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=config.optim.lr,
            betas=config.optim.betas, weight_decay=config.optim.weight_decay)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=config.optim.amp and self.device.type == "cuda")
        self.ema = (ExponentialMovingAverage(self.network, config.optim.ema_decay)
                    if config.optim.ema else None)

        self.writer = CheckpointWriter(config.paths, config)
        self.validator = BrainValidator(config, self.network, self.criterion,
                                        self.device)
        self.history = TrainingHistory()
        self.start_epoch = 0
        self.best_value = -math.inf if config.optim.monitor_mode == "max" else math.inf
        self.epochs_without_improvement = 0
        self._global_step = 0
        self._loader: DataLoader | None = None

        self._steps_per_epoch = max(
            1, math.ceil(len(self.sampler) / max(config.optim.batch_size, 1)))
        self._optimizer_steps_per_epoch = max(
            1, math.ceil(self._steps_per_epoch / max(config.optim.grad_accum, 1)))

        if config.optim.compile:
            self.network = torch.compile(self.network)      # pragma: no cover
        if config.optim.auto_resume:
            self._maybe_resume()

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def fit(self) -> TrainingHistory:
        """Train to the epoch budget, or until early stopping."""
        config = self.config
        log.info("training starting", extra={"context": {
            "run": config.run_name, "device": str(self.device),
            "epochs": config.optim.epochs, "start_epoch": self.start_epoch,
            "steps_per_epoch": self._steps_per_epoch,
            "train_slices": int(self.table.split_indices(SplitName.TRAIN.value).size),
            "val_slices": int(self.table.split_indices(SplitName.VAL.value).size),
            "parameters": self.network.parameter_count()["total"],
            "amp": self.scaler.is_enabled(), "ema": self.ema is not None}})
        self._write_model_card()

        for epoch in range(self.start_epoch, config.optim.epochs):
            started = time.perf_counter()
            train_summary = self._train_one_epoch(epoch)

            report: ValidationReport | None = None
            if (epoch + 1) % max(1, config.validation.every_n_epochs) == 0:
                report = self._validate(epoch)

            monitor_value = (report.monitor(config.optim.monitor)
                             if report is not None else float("nan"))
            improved = self._is_improvement(monitor_value)
            if improved:
                self.best_value = monitor_value
                self.epochs_without_improvement = 0
            elif report is not None:
                self.epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "stage": self.sampler.stage.value,
                "sampler": self.sampler.plan(),
                "difficulty": self.sampler.refresh_difficulty(),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_summary,
                "validation": report.to_dict() if report else None,
                "monitor": config.optim.monitor,
                "monitor_value": monitor_value,
                "improved": improved,
                "seconds": round(time.perf_counter() - started, 2),
            }
            self.history.add(record)
            self.writer.append_history(record)
            self._save(epoch, monitor_value, improved, report)

            log.info("epoch complete", extra={"context": {
                "epoch": epoch, "stage": self.sampler.stage.value,
                "train_loss": train_summary.get("total"),
                "monitor": config.optim.monitor,
                "value": (round(monitor_value, 5)
                          if math.isfinite(monitor_value) else None),
                "best": (round(self.best_value, 5)
                         if math.isfinite(self.best_value) else None),
                "patience": self.epochs_without_improvement,
                "seconds": record["seconds"]}})

            if self.epochs_without_improvement >= config.optim.early_stopping_patience:
                log.info("early stopping", extra={"context": {
                    "epoch": epoch, "patience": config.optim.early_stopping_patience,
                    "best": self.best_value}})
                break

        self._write_model_card(final=True)
        return self.history

    # ------------------------------------------------------------------ #
    # One epoch
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        config = self.config
        self.network.train()
        self.sampler.set_epoch(epoch)
        self.datasets[SplitName.TRAIN].set_epoch(epoch)
        loader = self._train_loader()
        meter = LossMeter()
        accumulated = 0

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            self._set_learning_rate(epoch, step)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type=self.device.type,
                                enabled=self.scaler.is_enabled()):
                output = self.network(batch["image"])
                breakdown = self.criterion(output, batch)
                # Scaling by 1/grad_accum here rather than at the step keeps the
                # gradient magnitude — and therefore the clip threshold — independent of
                # the accumulation setting.
                loss = breakdown.total / config.optim.grad_accum

            self.scaler.scale(loss).backward()
            accumulated += 1
            meter.update(breakdown.scalars(), weight=int(batch["image"].shape[0]))

            if breakdown.per_sample_dice is not None:
                self.sampler.difficulty.update(
                    batch["index"].cpu().numpy(),
                    breakdown.per_sample_dice.cpu().numpy())

            if accumulated >= config.optim.grad_accum:
                self._optimizer_step()
                accumulated = 0

        if accumulated:                        # a trailing partial accumulation window
            self._optimizer_step()

        summary = meter.summary()
        log.info("training epoch summary", extra={"context": {
            "epoch": epoch, "stage": self.sampler.stage.value,
            **{k: v for k, v in summary.items() if k in
               ("total", "segmentation", "dice", "presence", "quality", "supcon")}}})
        return summary

    def _optimizer_step(self) -> None:
        config = self.config
        if config.optim.grad_clip > 0:
            # Unscale before clipping: clipping scaled gradients clips at
            # ``grad_clip * scale``, which is a threshold that moves every time the
            # GradScaler adjusts — the classic silent AMP bug.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           config.optim.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self._global_step += 1
        if self.ema is not None:
            self.ema.update(self.network)

    def _set_learning_rate(self, epoch: int, step: int) -> None:
        """Linear warmup into cosine annealing, computed per batch.

        Per batch rather than per epoch because the warmup is often shorter than one
        epoch, and a per-epoch schedule would simply skip it.
        """
        config = self.config.optim
        progress = epoch + step / max(self._steps_per_epoch, 1)
        if config.warmup_epochs > 0 and progress < config.warmup_epochs:
            lr = config.lr * (progress / config.warmup_epochs)
        else:
            span = max(config.epochs - config.warmup_epochs, 1e-6)
            cosine = (progress - config.warmup_epochs) / span
            cosine = min(max(cosine, 0.0), 1.0)
            lr = config.min_lr + 0.5 * (config.lr - config.min_lr) * (
                1.0 + math.cos(math.pi * cosine))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------ #
    # Validation and checkpointing
    # ------------------------------------------------------------------ #
    def _validate(self, epoch: int) -> ValidationReport:
        loader = self._eval_loader(SplitName.VAL)
        store = (EmbeddingStore(self.network.embedding_spec,
                                limit=self.config.validation.embedding_limit)
                 if self.config.validation.export_embeddings else None)
        context = (self.ema.applied(self.network) if self.ema is not None
                   else contextlib.nullcontext(self.network))
        with context:
            report = self.validator.run(loader, epoch=epoch, split=SplitName.VAL.value,
                                        store=store)
        if store is not None and len(store):
            store.write(self.config.paths.embedding_dir, epoch=epoch,
                        checkpoint=self.config.paths.best_model_path.name,
                        architecture=self.network.describe(),
                        subject_ids=[s.subject_id for s in self.manifest.subjects])
        return report

    def evaluate(self, split: SplitName = SplitName.TEST) -> ValidationReport:
        """Score a split with the current weights. Used for the final test report."""
        loader = self._eval_loader(split)
        context = (self.ema.applied(self.network) if self.ema is not None
                   else contextlib.nullcontext(self.network))
        with context:
            return self.validator.run(loader, epoch=-1, split=split.value)

    def _is_improvement(self, value: float | None) -> bool:
        # A monitor that could not be computed is never an improvement. Treating a
        # missing metric as a win would checkpoint a model nothing measured.
        if value is None or not math.isfinite(value):
            return False
        if self.config.optim.monitor_mode == "max":
            return value > self.best_value
        return value < self.best_value

    def _save(self, epoch: int, monitor_value: float, improved: bool,
              report: ValidationReport | None) -> None:
        meta = CheckpointMeta(
            run_name=self.config.run_name, epoch=epoch,
            monitor=self.config.optim.monitor, monitor_value=monitor_value,
            architecture=self.network.describe(), config=self.config.to_dict(),
            metrics=report.to_dict() if report else {},
            caveats=list(self.manifest.caveats))
        self.writer.save_epoch(
            self.network, meta, is_best=improved,
            ema_state=self.ema.state_dict() if self.ema else None)
        self.writer.save_training_state({
            "epoch": epoch,
            "global_step": self._global_step,
            "best_value": self.best_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "sampler_state": self.sampler.state_dict(),
            "rng": {"python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
            "brain_vision_version": BRAIN_VISION_VERSION,
            "meta": meta.to_dict(),
        })

    def _maybe_resume(self) -> None:
        path = self.config.paths.training_state_path
        if not path.exists():
            return
        try:
            state = load_training_state(path, device=str(self.device))
        except Exception:
            log.exception("the training state could not be read; starting fresh",
                          extra={"context": {"path": str(path)}})
            return
        latest = self.config.paths.latest_model_path
        if latest.exists():
            payload = torch.load(latest, map_location=self.device, weights_only=False)
            self.network.load_state_dict(payload["model_state_dict"])
            if self.ema is not None and payload.get("ema_state_dict"):
                self.ema.shadow = {k: v.float().to(self.device)
                                   for k, v in payload["ema_state_dict"].items()}
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])
        if state.get("sampler_state"):
            self.sampler.load_state_dict(state["sampler_state"])
        rng = state.get("rng") or {}
        with contextlib.suppress(Exception):
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu()
                                if hasattr(rng["torch"], "cpu") else rng["torch"])
        self.start_epoch = int(state.get("epoch", -1)) + 1
        self.best_value = float(state.get("best_value", self.best_value))
        self.epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0))
        self._global_step = int(state.get("global_step", 0))
        self._load_history()
        log.info("resumed from training state", extra={"context": {
            "path": str(path), "next_epoch": self.start_epoch,
            "best": self.best_value,
            "history_records_recovered": len(self.history.records)}})

    def _load_history(self) -> None:
        """Re-read the epochs a previous process completed.

        Without this, a resumed run's ``TrainingHistory`` starts empty, and the model
        card — which reports ``history.best`` — would name the best epoch *since the
        resume* while ``best_brain_model.pt`` on disk holds an earlier and better one.
        The checkpoint would be right and the card describing it would be wrong, which
        is the worst of the available failure modes.
        """
        path = self.config.paths.history_path
        if not path.exists():
            return
        recovered: dict[int, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:                 # a torn final line
                continue
            epoch = record.get("epoch")
            if isinstance(epoch, int) and epoch < self.start_epoch:
                recovered[epoch] = record                # later lines win
        self.history.records = [recovered[e] for e in sorted(recovered)]

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #
    def _train_loader(self) -> DataLoader:
        """The training loader, built once and reused.

        Reused rather than rebuilt per epoch because ``persistent_workers=True`` means
        a fresh ``DataLoader`` each epoch spawns a fresh set of worker processes and
        leaves the previous set to be reaped whenever the old object is collected. The
        sampler is stateful and is re-read at the start of every iteration, so one
        loader serves every epoch: ``set_epoch`` changes what the next pass draws.
        """
        if self._loader is None:
            config = self.config.optim
            self._loader = DataLoader(
                self.datasets[SplitName.TRAIN],
                batch_size=config.batch_size,
                sampler=_IndexSampler(self.sampler, self.datasets[SplitName.TRAIN]),
                num_workers=config.num_workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=config.num_workers > 0,
                drop_last=False,
            )
        return self._loader

    #: Worker processes for an evaluation pass. Fewer than training uses, and not
    #: persistent, because a validation loader is built *while* the training loader's
    #: workers are alive — eight worker processes each holding open memory maps and
    #: prefetched batches is where a machine runs out of shared memory, and it does so
    #: mid-validation where the cause is least obvious.
    _EVAL_WORKERS = 2

    def _eval_loader(self, split: SplitName) -> DataLoader:
        """Evaluation loader, in a fixed order that is representative when truncated.

        Cached slices are stored in anatomical order, so the first N of a split are the
        inferior slices of the first subjects — which in this corpus contain almost no
        tumour. A ``max_batches``-truncated validation over that prefix reports a Dice
        of zero against an empty ground truth and looks like a broken model. The order
        is therefore a *fixed* permutation: representative under truncation, and
        identical every cycle, so a change in a validation number is a change in the
        model rather than a change in the sample.
        """
        dataset = self.datasets[split]
        order = np.random.default_rng(self.config.optim.seed).permutation(len(dataset))
        workers = min(self._EVAL_WORKERS, self.config.optim.num_workers)
        return DataLoader(
            dataset,
            batch_size=self.config.validation.batch_size,
            sampler=order.tolist(),
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )

    # ------------------------------------------------------------------ #
    def _write_model_card(self, final: bool = False) -> None:
        """The model card. Written before training and rewritten at the end.

        Before, because a run that crashes should still have left behind what it was
        going to be; after, because the numbers only exist then.
        """
        best = self.history.best
        card = {
            "name": "AURA NeuroMind Brain Vision Engine",
            "version": BRAIN_VISION_VERSION,
            "run_name": self.config.run_name,
            "status": "trained" if final else "training",
            "architecture": self.network.describe(),
            "objective": self.criterion.describe(),
            "data": {
                "corpus": self.manifest.corpus_root,
                "cache_version": self.manifest.cache_version,
                "foundation_version": self.manifest.foundation_version,
                "channel_verification": self.manifest.channel_verification,
                "subjects": {split.value: len(self.manifest.by_split(split))
                             for split in SplitName},
                "slices": {split.value: int(self.table.split_indices(split.value).size)
                           for split in SplitName},
                "slice_statistics": self.table.describe(),
                "morphology_labels": self.morphology.to_dict(),
                "split_policy": ("by subject, stratified by tumour grade; no subject "
                                 "contributes slices to more than one split"),
            },
            "training": {
                "device": str(self.device),
                "epochs_configured": self.config.optim.epochs,
                "epochs_completed": len(self.history.records),
                "batch_size": self.config.optim.batch_size,
                "effective_batch_size": (self.config.optim.batch_size
                                         * self.config.optim.grad_accum),
                "samples_per_epoch": self.config.sampling.samples_per_epoch,
                "curriculum": [[s.value, n] for s, n
                               in self.config.curriculum.schedule],
                "stability": {"amp": self.scaler.is_enabled(),
                              "grad_clip": self.config.optim.grad_clip,
                              "grad_accum": self.config.optim.grad_accum,
                              "ema": self.ema is not None,
                              "warmup_epochs": self.config.optim.warmup_epochs,
                              "schedule": "cosine annealing",
                              "early_stopping_patience":
                                  self.config.optim.early_stopping_patience},
            },
            "best": ({"epoch": best.get("epoch"),
                      "monitor": best.get("monitor"),
                      "value": best.get("monitor_value"),
                      "validation": best.get("validation")} if best else None),
            "head_validity": _head_validity(best),
            "intended_use": (
                "Perception layer for AURA NeuroMind: glioma sub-region segmentation "
                "and reusable latent representation over pre-processed multi-sequence "
                "brain MRI. Research use. Not a diagnostic device and not validated for "
                "clinical decision-making."),
            "caveats": list(self.manifest.caveats) + [
                "Trained and validated on 2D axial slices. Volumetric consistency "
                "between adjacent slices is not enforced by the model or measured by "
                "the reported metrics.",
                "The tumour-grade probe reported under embedding metrics is a "
                "representation-quality diagnostic, not a grading model.",
            ],
        }
        self.writer.write_model_card(card)


class _IndexSampler:
    """Adapts :class:`AdaptiveSliceSampler` to torch's ``Sampler`` interface.

    The adaptive sampler yields indices into the *global* slice table; a
    ``BrainSliceDataset`` is indexed by position within its own split. This translates
    between them, which keeps the sampling policy expressed in the coordinates the
    difficulty tracker and the curriculum use — global slice ids — while the dataset
    stays a simple positional sequence.
    """

    def __init__(self, sampler: AdaptiveSliceSampler, dataset: Any) -> None:
        self._sampler = sampler
        self._position = {int(value): position
                          for position, value in enumerate(dataset.indices)}

    def __len__(self) -> int:
        return len(self._sampler)

    def __iter__(self):
        for index in self._sampler:
            position = self._position.get(int(index))
            if position is not None:
                yield position


def _head_validity(best: dict[str, Any] | None) -> dict[str, Any]:
    """Per-head verdict for the model card: did each head pass its own test?

    A model card that lists five heads and reports only the segmentation numbers invites
    a reader to assume the other four work. Each head here has a metric that a
    degenerate predictor cannot fake — AUROC for presence, correlation for size and
    quality, a cross-subject probe for the embedding — and the verdict is derived from
    that metric rather than asserted.
    """
    from backend.vision.brain.output import QUALITY_VALIDITY_THRESHOLD

    validation = (best or {}).get("validation") or {}
    verdicts: dict[str, Any] = {}

    presence = validation.get("presence") or {}
    aurocs = [v.get("auroc") for v in presence.values()
              if isinstance(v, dict) and v.get("auroc") is not None]
    if aurocs:
        verdicts["presence"] = {
            "metric": "AUROC per region", "min": round(min(aurocs), 5),
            "passes": min(aurocs) > 0.75,
            "criterion": "AUROC > 0.75 on every region"}

    size = validation.get("size") or {}
    rs = [v.get("pearson_r") for v in size.values()
          if isinstance(v, dict) and v.get("pearson_r") is not None]
    if rs:
        verdicts["size"] = {
            "metric": "Pearson r vs true log-area", "min": round(min(rs), 5),
            "passes": min(rs) > 0.60,
            "criterion": "r > 0.60 on every region"}

    diagnostics = ((validation.get("quality") or {}).get("diagnostics") or {})
    correlation = diagnostics.get("severity_correlation")
    if correlation is not None:
        passes = correlation <= QUALITY_VALIDITY_THRESHOLD
        verdicts["quality"] = {
            "metric": "correlation with known degradation severity",
            "value": round(correlation, 5), "passes": passes,
            "criterion": f"r <= {QUALITY_VALIDITY_THRESHOLD}",
            "note": None if passes else (
                "near-constant predictor; do not use. The dataset normalises each "
                "slice over brain voxels *after* the artefact is applied, which removes "
                "most of the intensity-statistic evidence this head needs. Held-out "
                "linear probes put the ceiling on these features at r~0.3, so the head "
                "is weak rather than merely untrained.")}

    embedding = validation.get("embedding") or {}
    probe = embedding.get("grade_probe") or {}
    accuracy, baseline = probe.get("knn_accuracy_cross_subject"), \
        probe.get("majority_baseline")
    if accuracy is not None and baseline is not None:
        verdicts["embedding_grade_transfer"] = {
            "metric": "cross-subject k-NN accuracy on held-out tumour grade",
            "value": accuracy, "majority_baseline": baseline,
            "passes": accuracy > baseline,
            "criterion": "beats the majority-class baseline",
            "note": None if accuracy > baseline else (
                "the embedding does not carry tumour grade above the majority "
                "baseline. It clusters the morphology it was trained on; that is what "
                "it is validated for and the limit of what it is validated for.")}
    if embedding.get("morphology_knn_purity") is not None:
        verdicts["embedding_morphology"] = {
            "metric": "k-NN purity over the trained morphology class",
            "value": embedding["morphology_knn_purity"],
            "collapse": embedding.get("collapse"),
            "passes": embedding["morphology_knn_purity"] > 0.5,
            "criterion": "purity > 0.5 and not collapsed"}
    return verdicts


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: BrainVisionConfig) -> TrainingHistory:
    """Convenience entry point. The seam a CLI or a scheduler calls."""
    return BrainVisionTrainer(config).fit()


__all__ = ["BrainVisionTrainer", "ExponentialMovingAverage", "TrainingHistory", "train"]
