"""The five prediction heads.

Four of them read the pooled encoder bottleneck and one reads the decoder pyramid. That
split is the architectural expression of the whole point of this module: segmentation is
the thing that needs full spatial resolution, and *description* — is there a tumour, how
big, how good is this image, what does this brain look like — does not. Keeping the
description heads on the bottleneck means they can be evaluated without running the
decoder at all, which is what makes an embedding cheap enough for a downstream module to
compute over a longitudinal series of forty studies.

The embedding head in particular takes the bottleneck and *only* the bottleneck. It
could be made better by feeding it decoder features; it is not, because an embedding
that depends on the decoder cannot be computed without segmenting, and the requirement
is that future modules reuse the representation independently of the segmentation task.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .blocks import make_activation


class SegmentationHead(nn.Module):
    """Per-level 1x1 classifiers over the decoder pyramid — deep supervision.

    Index 0 is the full-resolution prediction; higher indices are progressively coarser
    and are supervised against a downsampled label. Only the finest is ever returned to
    a caller outside training.
    """

    def __init__(self, feature_channels: Sequence[int], num_classes: int, *,
                 levels: int = 4) -> None:
        super().__init__()
        levels = max(1, min(int(levels), len(feature_channels)))
        self.levels = levels
        self.num_classes = int(num_classes)
        self.classifiers = nn.ModuleList([
            nn.Conv2d(int(feature_channels[i]), self.num_classes, kernel_size=1)
            for i in range(levels)])

    def forward(self, decoder_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [classifier(decoder_features[i])
                for i, classifier in enumerate(self.classifiers)]


class _MLPHead(nn.Module):
    """Shared shape for the pooled-feature heads: one hidden layer, then an output."""

    def __init__(self, in_features: int, out_features: int, *, hidden: int = 256,
                 dropout: float = 0.0, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            make_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_features))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class PresenceHead(_MLPHead):
    """Multi-label tumour presence: whole tumour, then each of the three subregions.

    Multi-label rather than a single binary output because "is there a tumour" and "is
    there an enhancing component" are different clinical questions with different
    prevalence, and a downstream triage rule needs both. Logits are returned; the
    sigmoid lives in the loss and in the inference wrapper so that training never
    computes it twice.
    """

    def __init__(self, in_features: int, *, regions: int = 4, hidden: int = 256,
                 dropout: float = 0.0) -> None:
        super().__init__(in_features, regions, hidden=hidden, dropout=dropout)
        self.regions = regions


class SizeHead(_MLPHead):
    """Tumour extent, as scaled log-area for the same four regions.

    Regression on a log scale, not a linear one: tumour areas in this corpus span from
    tens to thousands of pixels, and a linear target makes the objective indifferent to
    everything below a few hundred — which is most of the range where a size estimate
    would actually change management. The output is passed through a softplus so a
    negative area is not representable.
    """

    def __init__(self, in_features: int, *, regions: int = 4, hidden: int = 256,
                 dropout: float = 0.0) -> None:
        super().__init__(in_features, regions, hidden=hidden, dropout=dropout)
        self.regions = regions

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(super().forward(pooled))


class QualityHead(nn.Module):
    """Predicted image quality in [0, 1], plus the artefact class it thinks it sees.

    The v1 version of this head read the pooled encoder bottleneck through a LayerNorm
    MLP and learned a constant (severity correlation -0.11 against a -0.30 validity
    threshold). Two measured reasons, and both are fixed here rather than papered over.

    **The encoder destroys texture.** Every stage is ``Conv -> InstanceNorm ->
    LeakyReLU``, and instance normalisation removes each channel's per-sample mean and
    variance — which is exactly where "this image is noisy" or "this image is blurred"
    lives. Whatever survives is then flattened by global *average* pooling. So this head
    gets its own :class:`QualityBranch`: a shallow convolutional path over the input with
    **no normalisation layers anywhere**, read out with mean *and standard deviation*
    pooling. Standard-deviation pooling is the point — it measures texture energy
    directly, and average pooling cannot.

    **Severity across five artefacts is not one regression.** Blur lowers
    high-frequency energy and noise raises it, so a single scalar "severity" head has to
    represent two opposite directions with one output. Measured on hand-made texture
    statistics of the normalised slice: pooled severity is recoverable at r=0.53, but
    *within* an artefact type it is r=0.97 for noise and r=0.77 for blur, and the type
    itself is 68% recoverable from crude features alone. So the head also classifies the
    artefact, which gives it a representation in which severity is a conditional
    quantity rather than an averaged contradiction.

    **It reads its own branch and nothing else.** The obvious design concatenates the
    encoder's 320-d pooled bottleneck alongside the 128-d texture vector, on the theory
    that knowing the anatomy helps calibrate the quality judgement. Measured, it does the
    opposite: the bottleneck is larger, carries a segmentation-shaped signal, and is
    still moving early in training, and the head collapses onto predicting the mean —
    artefact accuracy settles at exactly the clean-class prevalence. The same branch
    trained alone reaches r=0.65 against quality and -0.59 against severity. So the
    bottleneck is left out, which also makes this head genuinely independent of the
    encoder: it can be lifted, retrained, or replaced without touching anything else.

    Returns ``(quality, artifact_logits)``. The quality scalar remains the head's
    product; the artefact class is an auxiliary that shapes it and is reported.
    """

    def __init__(self, pooled_features: int = 0, *, texture_channels: int = 64,
                 artifact_classes: int = 6, hidden: int = 128,
                 dropout: float = 0.0, in_channels: int = 4) -> None:
        super().__init__()
        self.artifact_classes = int(artifact_classes)
        self.branch = QualityBranch(texture_channels, in_channels=in_channels)
        width = self.branch.out_features
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden),
            # No LayerNorm. It normalises across the hidden units of each sample, which
            # removes that sample's overall activation magnitude — the same class of
            # mistake as the InstanceNorm above, one layer later.
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.quality = nn.Linear(hidden, 1)
        self.artifact = nn.Linear(hidden, self.artifact_classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Float32 regardless of autocast. The branch has no normalisation anywhere by
        # design, so its activations are unbounded, and the spatial standard deviation
        # it depends on is exactly the reduction that loses precision worst in float16.
        with torch.autocast(device_type=image.device.type, enabled=False):
            features = self.trunk(self.branch(image.float()))
            return torch.sigmoid(self.quality(features)), self.artifact(features)


class QualityBranch(nn.Module):
    """Texture-sensitive readout of the input. Deliberately un-normalised.

    Three strided convolutions and a mean+std pooling. Small (~30k parameters) because
    image quality is a texture statistic, not a semantic property: it does not need
    depth, it needs the scale information that the segmentation encoder throws away.
    """

    def __init__(self, channels: int = 64, in_channels: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels // 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels // 4, channels // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True))
        self.out_features = channels * 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        maps = self.features(image)
        mean = maps.mean(dim=(2, 3))
        # Standard deviation over the spatial field. The single most informative
        # statistic for noise and blur, and the one average pooling cannot express.
        std = maps.std(dim=(2, 3))
        return torch.cat([mean, std], dim=1)


class EmbeddingHead(nn.Module):
    """Projects the pooled bottleneck to the exported latent representation.

    Two outputs, and the distinction is the standard contrastive-learning one that is
    easy to get wrong. ``projection`` is what the contrastive loss is computed on;
    ``embedding`` is what is exported and what downstream modules consume. They are the
    same vector here — the projector *is* the representation this module publishes,
    because a downstream consumer needs a stable, documented, L2-normalised space rather
    than the raw bottleneck whose width changes with the encoder configuration.

    L2-normalised because every consumer will compare embeddings with cosine similarity
    or Euclidean distance, and on a normalised space those two induce the same ordering.
    Leaving normalisation to the consumer means half of them will forget.
    """

    def __init__(self, in_features: int, dimension: int = 128, *, hidden: int = 256,
                 normalize: bool = True) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.normalize = bool(normalize)
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, self.dimension))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        embedding = self.projector(pooled)
        if self.normalize:
            embedding = nn.functional.normalize(embedding, dim=1, eps=1e-6)
        return embedding


__all__ = ["EmbeddingHead", "PresenceHead", "QualityBranch", "QualityHead",
           "SegmentationHead", "SizeHead"]
