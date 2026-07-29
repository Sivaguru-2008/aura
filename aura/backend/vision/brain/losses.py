"""The multi-task objective.

Five heads, one scalar. Every component is separately weighted and separately
switchable, and :meth:`MultiTaskLoss.forward` returns the breakdown alongside the total
so a training log can show which head is actually moving — a combined loss that only
reports its sum makes a silently dead head indistinguishable from a converged one.

Segmentation
------------
Dice plus weighted cross-entropy is the standard pairing for BraTS and it is standard
for a reason: cross-entropy is well-behaved everywhere but is dominated by the 97% of
voxels that are background, and Dice attends to the foreground but has almost no
gradient when a class is absent from both prediction and label. Together they cover each
other. Focal and boundary terms are available and default to zero weight, so switching
one on is an ablation with a baseline rather than an unfalsifiable choice.

Deep supervision applies the same segmentation loss at each supervised decoder level
against a nearest-neighbour-downsampled label, with geometrically decaying weights.

Embedding
---------
Supervised contrastive over the morphology class, plus the variance and covariance terms
from VICReg. The anti-collapse terms are not optional garnish: a contrastive loss sharing
an encoder with four other heads can satisfy itself by collapsing every embedding onto a
point, at which point the SupCon term is minimised, the segmentation head is unaffected,
and the exported embeddings are worthless in a way no segmentation metric would reveal.
The variance term makes collapse expensive; the covariance term stops the 128 dimensions
from becoming 3 dimensions repeated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .config import LossConfig
from .model.network import NetworkOutput, downsample_label
from .types import HeadName, TumorRegion


@dataclass
class LossBreakdown:
    """The total and every component that produced it."""

    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Per-sample foreground Dice, detached. Feeds hard-example mining.
    per_sample_dice: torch.Tensor | None = None

    def scalars(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.components.items()}


# --------------------------------------------------------------------------- #
# Segmentation terms
# --------------------------------------------------------------------------- #
def soft_dice(logits: torch.Tensor, target: torch.Tensor, *, num_classes: int,
              smooth: float = 1.0, ignore_background: bool = True,
              reduce: bool = True) -> torch.Tensor:
    """Soft Dice over a softmax prediction.

    Computed in float32 regardless of the autocast dtype. Under mixed precision the
    intersection of a 192x192 map in float16 accumulates to a value whose relative error
    is visible in the third decimal of the loss, and the resulting gradient noise looks
    exactly like a learning-rate problem.
    """
    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    start = 1 if ignore_background else 0
    probabilities = probabilities[:, start:]
    one_hot = one_hot[:, start:]

    dims = (2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    cardinality = probabilities.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean() if reduce else dice


def per_sample_foreground_dice(logits: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Hard Dice over the union of the foreground classes, one value per sample.

    The difficulty signal for hard-example mining. Whole-tumour rather than per class,
    because a sample where the model finds the tumour but confuses oedema with core is
    not the same kind of hard as one where it finds nothing, and the sampler should be
    chasing the second.
    """
    with torch.no_grad():
        predicted = logits.argmax(dim=1) > 0
        truth = target > 0
        dims = (1, 2)
        intersection = (predicted & truth).sum(dims).float()
        cardinality = predicted.sum(dims).float() + truth.sum(dims).float()
        # A slice with no tumour that was predicted empty is a perfect result, and
        # 0/0 must read as 1.0 rather than as 0.0 — otherwise every negative slice
        # looks maximally hard and the miner spends the epoch on empty anatomy.
        return torch.where(cardinality > 0, 2.0 * intersection / cardinality,
                           torch.ones_like(cardinality))


def focal_loss(logits: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy down-weighted on already-confident voxels."""
    log_probabilities = F.log_softmax(logits.float(), dim=1)
    log_pt = log_probabilities.gather(1, target[:, None]).squeeze(1)
    loss = -((1.0 - log_pt.exp()) ** gamma) * log_pt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, *,
                  num_classes: int, smooth: float = 1.0) -> torch.Tensor:
    """Dice restricted to a one-pixel band around each class boundary.

    This is a *boundary-band* loss, not Kervadec's distance-map boundary loss, and the
    difference is worth naming rather than blurring: the published formulation
    integrates the prediction against a signed distance transform of the label, which
    has to be computed on CPU per sample per class and costs more than the rest of the
    step. This version extracts boundaries with a dilation-minus-erosion on the GPU and
    scores agreement there. It gives the same qualitative pressure — errors at the edge
    of a region cost more than errors in its interior — at a fraction of the price, and
    it does not claim to be the same objective.
    """
    with torch.no_grad():
        one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        dilated = F.max_pool2d(one_hot, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-one_hot, 3, stride=1, padding=1)
        band = (dilated - eroded) > 0                     # (B, K, H, W)
    probabilities = torch.softmax(logits.float(), dim=1)
    band = band[:, 1:]
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:]

    dims = (2, 3)
    masked_prediction = probabilities * band
    masked_truth = one_hot * band
    intersection = (masked_prediction * masked_truth).sum(dims)
    cardinality = masked_prediction.sum(dims) + masked_truth.sum(dims)
    return 1.0 - ((2.0 * intersection + smooth) / (cardinality + smooth)).mean()


class SegmentationLoss(nn.Module):
    """Weighted combination of Dice, cross-entropy, focal, and boundary terms."""

    def __init__(self, config: LossConfig, num_classes: int = len(TumorRegion)) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        weights = torch.as_tensor(config.class_weights, dtype=torch.float32)
        if weights.numel() != num_classes:
            weights = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        config = self.config
        components: dict[str, torch.Tensor] = {}
        total = logits.new_zeros(())

        if config.dice_weight > 0:
            value = soft_dice(logits, target, num_classes=self.num_classes,
                              smooth=config.dice_smooth,
                              ignore_background=config.dice_ignore_background)
            components["dice"] = value
            total = total + config.dice_weight * value
        if config.cross_entropy_weight > 0:
            value = F.cross_entropy(logits.float(), target,
                                    weight=self.class_weights.to(logits.device))
            components["cross_entropy"] = value
            total = total + config.cross_entropy_weight * value
        if config.focal_weight > 0:
            value = focal_loss(logits, target, gamma=config.focal_gamma,
                               weight=self.class_weights.to(logits.device))
            components["focal"] = value
            total = total + config.focal_weight * value
        if config.boundary_weight > 0:
            value = boundary_loss(logits, target, num_classes=self.num_classes,
                                  smooth=config.dice_smooth)
            components["boundary"] = value
            total = total + config.boundary_weight * value
        return total, components


# --------------------------------------------------------------------------- #
# Embedding terms
# --------------------------------------------------------------------------- #
def supervised_contrastive(embeddings: torch.Tensor, labels: torch.Tensor, *,
                           temperature: float = 0.1) -> torch.Tensor:
    """SupCon over in-batch positives, single view.

    Two samples are positives when they share a morphology class. A second augmented
    view per sample is not generated: with real labels the positives already exist in
    the batch, and doubling the forward pass to manufacture more would double the cost
    of every step for the benefit of one head out of five.

    Samples whose class is unique in the batch have no positive and are excluded from
    the mean rather than contributing a zero — including them would make the loss a
    function of batch composition as much as of the embedding.
    """
    embeddings = F.normalize(embeddings.float(), dim=1, eps=1e-6)
    similarity = embeddings @ embeddings.t() / max(temperature, 1e-6)
    # Subtracting the row maximum before exponentiating is the standard numerically
    # stable log-sum-exp; without it a temperature of 0.1 overflows float16.
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    batch = embeddings.shape[0]
    identity = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    positives = (labels[:, None] == labels[None, :]) & ~identity

    exponentiated = torch.exp(similarity) * (~identity)
    log_probability = similarity - torch.log(exponentiated.sum(dim=1, keepdim=True)
                                             + 1e-12)
    positive_counts = positives.sum(dim=1)
    usable = positive_counts > 0
    if not bool(usable.any()):
        return embeddings.new_zeros(())
    per_sample = ((log_probability * positives).sum(dim=1)[usable]
                  / positive_counts[usable].float())
    return -per_sample.mean()


def variance_covariance(embeddings: torch.Tensor, *, target_std: float = 1.0
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg's anti-collapse pair: hinge on per-dimension std, plus off-diagonal
    covariance."""
    embeddings = embeddings.float()
    if embeddings.shape[0] < 2:
        zero = embeddings.new_zeros(())
        return zero, zero
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    std = torch.sqrt(centred.var(dim=0) + 1e-6)
    variance = F.relu(target_std - std).mean()

    batch, dimension = centred.shape
    covariance = (centred.t() @ centred) / (batch - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_term = off_diagonal.pow(2).sum() / dimension
    return variance, covariance_term


class EmbeddingLoss(nn.Module):
    """SupCon plus variance/covariance regularisation."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        config = self.config
        contrastive = supervised_contrastive(embeddings, labels,
                                             temperature=config.supcon_temperature)
        variance, covariance = variance_covariance(
            embeddings, target_std=config.variance_target)
        total = (contrastive
                 + config.variance_weight * variance
                 + config.covariance_weight * covariance)
        return total, {"supcon": contrastive, "embed_variance": variance,
                       "embed_covariance": covariance}


# --------------------------------------------------------------------------- #
# The combined objective
# --------------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    """Segmentation with deep supervision, plus presence, size, quality, and embedding."""

    def __init__(self, config: LossConfig, *, heads: Sequence[HeadName],
                 num_classes: int = len(TumorRegion)) -> None:
        super().__init__()
        self.config = config
        self.heads = set(heads)
        self.segmentation = SegmentationLoss(config, num_classes)
        self.embedding = EmbeddingLoss(config)
        self.num_classes = num_classes

    def forward(self, output: NetworkOutput,
                batch: Mapping[str, torch.Tensor]) -> LossBreakdown:
        config = self.config
        components: dict[str, torch.Tensor] = {}
        device = output.pooled.device if output.pooled is not None else None
        total = torch.zeros((), device=device)
        per_sample: torch.Tensor | None = None

        if HeadName.SEGMENTATION in self.heads and output.segmentation:
            label = batch["label"]
            weights = _supervision_weights(config.deep_supervision_weights,
                                           len(output.segmentation))
            segmentation_total = torch.zeros((), device=output.logits.device)
            for level, (logits, weight) in enumerate(zip(output.segmentation, weights)):
                level_target = (label if logits.shape[-2:] == label.shape[-2:]
                                else downsample_label(label, logits.shape[-2:]))
                value, parts = self.segmentation(logits, level_target)
                segmentation_total = segmentation_total + weight * value
                if level == 0:
                    components.update(parts)
            # Normalising by the weight sum keeps ``segmentation_weight`` meaning the
            # same thing whether deep supervision uses one level or four.
            segmentation_total = segmentation_total / sum(weights)
            components["segmentation"] = segmentation_total
            total = total + config.segmentation_weight * segmentation_total
            per_sample = per_sample_foreground_dice(output.logits, label)

        if HeadName.PRESENCE in self.heads and output.presence is not None:
            value = F.binary_cross_entropy_with_logits(
                output.presence.float(), batch["presence"].float())
            components["presence"] = value
            total = total + config.presence_weight * value

        if HeadName.SIZE in self.heads and output.size is not None:
            # Smooth L1 rather than plain L2: the size target is a log-area, so a
            # mislabelled or partially cropped slice produces a large residual that an
            # L2 term would let dominate the head's gradient.
            value = F.smooth_l1_loss(output.size.float(), batch["size"].float(),
                                     beta=0.1)
            components["size"] = value
            total = total + config.size_weight * value

        if HeadName.QUALITY in self.heads and output.quality is not None:
            value = F.mse_loss(output.quality.float(), batch["quality"].float())
            components["quality"] = value
            total = total + config.quality_weight * value
            if output.artifact is not None and "artifact" in batch:
                artifact = F.cross_entropy(output.artifact.float(), batch["artifact"])
                components["quality_artifact"] = artifact
                total = total + config.quality_artifact_weight * artifact

        if HeadName.EMBEDDING in self.heads and output.embedding is not None:
            value, parts = self.embedding(output.embedding, batch["morphology"])
            components.update(parts)
            components["embedding"] = value
            total = total + config.embedding_weight * value

        components["total"] = total
        return LossBreakdown(total=total, components=components,
                             per_sample_dice=per_sample)

    def describe(self) -> dict[str, Any]:
        config = self.config
        return {
            "heads": sorted(h.value for h in self.heads),
            "segmentation_terms": {
                "dice": config.dice_weight, "cross_entropy": config.cross_entropy_weight,
                "focal": config.focal_weight, "boundary": config.boundary_weight},
            "head_weights": {
                "segmentation": config.segmentation_weight,
                "presence": config.presence_weight, "size": config.size_weight,
                "quality": config.quality_weight,
                "embedding": config.embedding_weight},
            "deep_supervision_weights": list(config.deep_supervision_weights),
            "class_weights": list(config.class_weights),
            "embedding_objective": {
                "supcon_temperature": config.supcon_temperature,
                "variance_weight": config.variance_weight,
                "covariance_weight": config.covariance_weight},
        }


def _supervision_weights(configured: Sequence[float], levels: int) -> list[float]:
    """Weights for the supervised decoder levels, extended or truncated to ``levels``."""
    weights = list(configured[:levels])
    while len(weights) < levels:
        weights.append(weights[-1] / 2.0 if weights else 1.0)
    return weights


__all__ = [
    "EmbeddingLoss", "LossBreakdown", "MultiTaskLoss", "SegmentationLoss",
    "boundary_loss", "focal_loss", "per_sample_foreground_dice", "soft_dice",
    "supervised_contrastive", "variance_covariance",
]
