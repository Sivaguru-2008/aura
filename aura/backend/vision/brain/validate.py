"""The validation cycle.

Runs the network over a held-out split and reports everything the specification asks
for — Dice, IoU, Hausdorff, precision, recall, sensitivity, specificity, inference time,
GPU memory, validation loss, per-class performance — plus the three things that are
specific to this being a representation learner rather than a segmenter: whether the
embedding space has collapsed, whether it clusters what it was trained to cluster, and
whether it carries something it was never told (tumour grade).

The quality head gets its own scrutiny. A regression metric alone cannot distinguish a
head that predicts image quality from one that predicts the *mean* image quality, so the
report includes the predicted standard deviation and the correlation against known
degradation severities. Those two numbers are the test; the MAE is a description.

Everything here is deterministic. Validation samples are degraded on a fixed
per-sample seed with a fixed artefact rotation (see
:meth:`~backend.vision.brain.dataset.BrainSliceDataset._forced_artifact`), so a change
in a validation number between epochs is a change in the model and not a change in the
draw.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch

from backend.core.shared.logging import get_logger
from backend.vision.brain.config import BrainVisionConfig
from backend.vision.brain.dataset import TARGET_REGIONS, decode_size
from backend.vision.brain.degradations import ARTIFACT_CLASSES, ARTIFACT_ORDER
from backend.vision.brain.embeddings import EmbeddingBatch, EmbeddingStore
from backend.vision.brain.losses import MultiTaskLoss
from backend.vision.brain.metrics import (
    ClassificationMeter,
    EmbeddingMeter,
    LossMeter,
    PerformanceMeter,
    RegressionMeter,
    SegmentationMeter,
)
from backend.vision.brain.model.network import BrainVisionNetwork
from backend.vision.brain.types import HeadName

log = get_logger("vision.brain.validate")


@dataclass
class ValidationReport:
    """Everything one validation cycle measured."""

    epoch: int
    split: str
    loss: dict[str, float] = field(default_factory=dict)
    segmentation: dict[str, Any] = field(default_factory=dict)
    presence: dict[str, Any] = field(default_factory=dict)
    size: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    samples: int = 0
    seconds: float = 0.0

    def monitor(self, name: str) -> float:
        """Value of the monitored metric, by dotted or bare name.

        Bare names are looked up in the segmentation summary first because that is where
        every sensible monitor lives; a dotted name addresses any section explicitly.
        """
        if "." in name:
            section, _, key = name.partition(".")
            return float(getattr(self, section, {}).get(key, float("nan")))
        for section in (self.segmentation, self.loss, self.presence, self.quality):
            if name in section:
                value = section[name]
                if isinstance(value, (int, float)):
                    return float(value)
        return float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch, "split": self.split, "samples": self.samples,
            "seconds": round(self.seconds, 3), "loss": dict(self.loss),
            "segmentation": dict(self.segmentation), "presence": dict(self.presence),
            "size": dict(self.size), "quality": dict(self.quality),
            "embedding": dict(self.embedding), "performance": dict(self.performance),
        }

    def headline(self) -> str:
        """One line for the log. A metric that was not computed prints ``n/a``.

        Not ``0.0000``: a reader scanning epoch lines cannot tell a Dice of zero from a
        Dice that does not exist, and the two mean opposite things.
        """
        composite = self.segmentation.get("per_composite", {})

        def show(value: Any) -> str:
            return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

        return (f"epoch {self.epoch} [{self.split}] "
                f"loss {show(self.loss.get('total'))} | "
                f"Dice WT {show(composite.get('whole_tumor', {}).get('dice'))} "
                f"TC {show(composite.get('tumor_core', {}).get('dice'))} "
                f"ET {show(composite.get('enhancing_tumor', {}).get('dice'))} | "
                f"mean {show(self.segmentation.get('composite_dice_mean'))}")


class BrainValidator:
    """Runs one validation pass and produces a :class:`ValidationReport`."""

    def __init__(self, config: BrainVisionConfig, network: BrainVisionNetwork,
                 criterion: MultiTaskLoss, device: torch.device) -> None:
        self.config = config
        self.network = network
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def run(self, loader: Iterable[dict[str, torch.Tensor]], *, epoch: int,
            split: str = "val", store: EmbeddingStore | None = None
            ) -> ValidationReport:
        validation = self.config.validation
        heads = set(self.config.model.heads)
        self.network.eval()

        segmentation = SegmentationMeter(
            compute_hausdorff=validation.compute_hausdorff,
            percentile=validation.hausdorff_percentile)
        presence = ClassificationMeter(TARGET_REGIONS)
        size = RegressionMeter(TARGET_REGIONS)
        quality = RegressionMeter(("image_quality",))
        embedding = EmbeddingMeter(probe_grade=validation.probe_grade)
        performance = PerformanceMeter()
        losses = LossMeter()

        severities: list[np.ndarray] = []
        quality_predictions: list[np.ndarray] = []
        degraded_flags: list[np.ndarray] = []
        artifact_targets: list[np.ndarray] = []
        artifact_predictions: list[np.ndarray] = []
        samples = 0
        started = time.perf_counter()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for position, batch in enumerate(loader):
            if validation.max_batches is not None and position >= validation.max_batches:
                break
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            count = int(batch["image"].shape[0])
            samples += count

            timed = time.perf_counter()
            with torch.autocast(device_type=self.device.type,
                                enabled=self.config.optim.amp
                                        and self.device.type == "cuda"):
                output = self.network(batch["image"], need_features=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            performance.record(time.perf_counter() - timed, count)

            breakdown = self.criterion(output, batch)
            losses.update(breakdown.scalars(), weight=count)

            if HeadName.SEGMENTATION in heads and output.segmentation:
                predicted = output.logits.float().argmax(dim=1).cpu().numpy()
                segmentation.update(predicted, batch["label"].cpu().numpy())
            if HeadName.PRESENCE in heads and output.presence is not None:
                presence.update(torch.sigmoid(output.presence.float()).cpu().numpy(),
                                batch["presence"].cpu().numpy())
            if HeadName.SIZE in heads and output.size is not None:
                size.update(output.size.float().cpu().numpy(),
                            batch["size"].cpu().numpy())
            if HeadName.QUALITY in heads and output.quality is not None:
                predicted_quality = output.quality.float().cpu().numpy()
                quality.update(predicted_quality, batch["quality"].cpu().numpy())
                quality_predictions.append(predicted_quality.reshape(-1))
                severities.append(batch["severity"].cpu().numpy().reshape(-1))
                degraded_flags.append(batch["degraded"].cpu().numpy().reshape(-1))
                if "artifact" in batch:
                    artifact_targets.append(batch["artifact"].cpu().numpy())
                    artifact_predictions.append(
                        output.artifact.float().argmax(dim=1).cpu().numpy()
                        if output.artifact is not None
                        else np.full(count, -1, dtype=np.int64))
            if HeadName.EMBEDDING in heads and output.embedding is not None:
                vectors = output.embedding.float().cpu().numpy()
                embedding.update(vectors, batch["morphology"].cpu().numpy(),
                                 batch["grade"].cpu().numpy(),
                                 batch["subject_index"].cpu().numpy())
                if store is not None and not store.full:
                    store.add(EmbeddingBatch(
                        embedding=vectors,
                        slice_index=batch["index"].cpu().numpy(),
                        subject_index=batch["subject_index"].cpu().numpy(),
                        cache_z=batch["cache_z"].cpu().numpy(),
                        morphology=batch["morphology"].cpu().numpy(),
                        grade=batch["grade"].cpu().numpy(),
                        tumor_area=(batch["label"] > 0).sum(dim=(1, 2)).cpu().numpy(),
                        quality=batch["quality"].cpu().numpy().reshape(-1)))

        if self.device.type == "cuda":
            performance.note_memory(torch.cuda.max_memory_allocated(self.device))

        report = ValidationReport(
            epoch=epoch, split=split, samples=samples,
            seconds=time.perf_counter() - started,
            loss=losses.summary(),
            segmentation=segmentation.summary() if HeadName.SEGMENTATION in heads else {},
            presence=presence.summary() if HeadName.PRESENCE in heads else {},
            size=self._size_summary(size) if HeadName.SIZE in heads else {},
            quality=self._quality_summary(quality, quality_predictions, severities,
                                          degraded_flags, artifact_targets,
                                          artifact_predictions)
                    if HeadName.QUALITY in heads else {},
            embedding=embedding.summary() if HeadName.EMBEDDING in heads else {},
            performance=performance.summary() if validation.measure_performance else {},
        )
        log.info(report.headline(), extra={"context": {
            "samples": samples, "seconds": round(report.seconds, 1)}})
        return report

    # ------------------------------------------------------------------ #
    def _size_summary(self, meter: RegressionMeter) -> dict[str, Any]:
        """Size-head errors, reported in pixels as well as in the scaled log space.

        The loss is computed on the scaled log-area because that is what makes the
        objective well-behaved; a mean absolute error of 0.03 in that space means
        nothing to a reader. The pixel figure is what a report would quote.
        """
        summary = meter.summary()
        pixels = float(np.prod(self.config.model.input_size))
        for name, values in summary.items():
            mae = values.get("mae")
            if mae is None:
                continue
            # The log scale makes the pixel error depend on where on the curve it sits,
            # so the conversion is quoted at the midpoint of the target range rather
            # than presented as exact.
            values["approx_pixel_error_at_median"] = round(
                float(decode_size(np.asarray([0.5 + mae]), pixels)[0]
                      - decode_size(np.asarray([0.5]), pixels)[0]), 1)
        return summary

    def _quality_summary(self, meter: RegressionMeter,
                         predictions: list[np.ndarray], severities: list[np.ndarray],
                         degraded: list[np.ndarray],
                         artifact_targets: list[np.ndarray],
                         artifact_predictions: list[np.ndarray]) -> dict[str, Any]:
        """Quality-head metrics, with the constant-predictor test made explicit.

        ``pearson_r`` against the target and ``predicted_std`` are the two numbers that
        matter. A head that has learned the mean has ``pearson_r`` near zero and
        ``predicted_std`` near zero while its MAE looks respectable, so reporting the
        MAE alone would let that pass as a working head.

        ``severity_correlation`` is the sharper version of the same test: over degraded
        samples only, how well does the predicted quality track the severity we chose?
        It should be strongly negative.
        """
        summary = meter.summary()
        if not predictions:
            return summary
        predicted = np.concatenate(predictions)
        severity = np.concatenate(severities)
        flags = np.concatenate(degraded) > 0.5

        detail: dict[str, Any] = {
            "degraded_fraction": round(float(flags.mean()), 4),
            "predicted_range": [round(float(predicted.min()), 4),
                                round(float(predicted.max()), 4)],
        }
        if flags.sum() >= 8 and severity[flags].std() > 1e-6:
            detail["severity_correlation"] = round(
                float(np.corrcoef(predicted[flags], severity[flags])[0, 1]), 5)
            detail["severity_correlation_note"] = (
                "predicted quality vs known degradation severity, on degraded samples "
                "only; strongly negative is correct")
        if flags.sum() >= 4 and (~flags).sum() >= 4:
            detail["mean_predicted_clean"] = round(float(predicted[~flags].mean()), 5)
            detail["mean_predicted_degraded"] = round(float(predicted[flags].mean()), 5)
            detail["separation"] = round(
                float(predicted[~flags].mean() - predicted[flags].mean()), 5)

        # Per artefact, because the pooled number averages incompatible things. Measured
        # recoverability from texture statistics alone ranges from r=0.97 (Rician noise)
        # to r=0.23 (bias field, most of which per-slice z-scoring removes), so one
        # pooled correlation hides both the head's best and its worst behaviour.
        if artifact_targets:
            kinds = np.concatenate(artifact_targets)
            per_artifact: dict[str, Any] = {}
            for index, name in enumerate(ARTIFACT_ORDER):
                mask = (kinds == index) & flags
                if mask.sum() >= 8 and severity[mask].std() > 1e-6:
                    per_artifact[name] = {
                        "n": int(mask.sum()),
                        "severity_correlation": round(
                            float(np.corrcoef(predicted[mask],
                                              severity[mask])[0, 1]), 5)}
            if per_artifact:
                detail["per_artifact"] = per_artifact
            if artifact_predictions:
                guessed = np.concatenate(artifact_predictions)
                valid = guessed >= 0
                if valid.any():
                    detail["artifact_type_accuracy"] = round(
                        float((guessed[valid] == kinds[valid]).mean()), 5)
                    detail["artifact_type_chance"] = round(1.0 / ARTIFACT_CLASSES, 5)
        summary["diagnostics"] = detail
        return summary


__all__ = ["BrainValidator", "ValidationReport"]
