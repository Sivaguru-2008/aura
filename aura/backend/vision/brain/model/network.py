"""The multi-task network: one encoder, one decoder, five heads.

    image -> encoder -> pyramid ------------> decoder -> segmentation logits (x levels)
                          \\
                           pooled bottleneck -> presence
                                             -> size
                                             -> quality
                                             -> embedding

:class:`NetworkOutput` is an *internal* container of tensors. It never leaves this
package: :class:`~aura.backend.vision.brain.output.BrainVisionOutput` is the public object,
and it holds numpy arrays and plain Python. The separation exists because a tensor that
escapes carries an autograd graph and a CUDA context with it, and something downstream
will eventually hold one alive for the lifetime of a request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

from aura.backend.core.shared.logging import get_logger
from ..config import ModelConfig
from ..degradations import ARTIFACT_CLASSES, ARTIFACT_ORDER

# Load-bearing despite looking unused: importing these modules is what registers
# "residual_unet2d" and "unet2d" with the architecture registry, and the network
# resolves both by *string* rather than by class. Remove them and every build fails
# with "no encoder named 'residual_unet2d' is registered".
from . import decoder as _register_decoders  # noqa: F401
from . import encoder as _register_encoders  # noqa: F401
from .heads import (
    EmbeddingHead,
    PresenceHead,
    QualityHead,
    SegmentationHead,
    SizeHead,
)
from .registry import build_decoder, build_encoder
from ..types import (
    BRAIN_VISION_VERSION,
    EmbeddingSpec,
    HeadName,
    ModalitySpec,
    TumorRegion,
)

log = get_logger("vision.brain.network")


@dataclass
class NetworkOutput:
    """Raw multi-task output. Internal to this package — tensors, by design."""

    #: Segmentation logits, finest first. Index 0 is full resolution.
    segmentation: list[torch.Tensor] = field(default_factory=list)
    presence: torch.Tensor | None = None
    size: torch.Tensor | None = None
    quality: torch.Tensor | None = None
    #: Artefact-class logits from the quality head. Auxiliary: it shapes the quality
    #: scalar by giving the network a representation in which severity is conditional
    #: on artefact type rather than averaged across five incompatible ones.
    artifact: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    #: Pooled bottleneck, before the embedding projector.
    pooled: torch.Tensor | None = None
    #: Encoder and decoder pyramids, populated only when a caller asks. Holding them
    #: unconditionally would keep every intermediate activation alive through the
    #: backward pass for no reason.
    encoder_features: list[torch.Tensor] = field(default_factory=list)
    decoder_features: list[torch.Tensor] = field(default_factory=list)

    @property
    def logits(self) -> torch.Tensor:
        """Full-resolution segmentation logits."""
        if not self.segmentation:
            raise RuntimeError("the network produced no segmentation output")
        return self.segmentation[0]


class BrainVisionNetwork(nn.Module):
    """The Brain Vision Engine's network.

    Built entirely from configuration through the architecture registry, so replacing
    the 2D residual U-Net with a 3D one or a transformer is a string change plus a
    registered factory — see
    :mod:`aura.backend.vision.brain.model.registry`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.modalities: tuple[ModalitySpec, ...] = tuple(config.modalities)
        self.num_classes = len(TumorRegion)

        self.encoder = build_encoder(
            config.encoder,
            modalities=self.modalities,
            stage_channels=config.stage_channels,
            blocks_per_stage=config.blocks_per_stage,
            norm=config.norm, activation=config.activation, dropout=config.dropout)
        self.decoder = build_decoder(
            config.decoder,
            encoder_channels=self.encoder.feature_channels,
            norm=config.norm, activation=config.activation, dropout=config.dropout)

        heads = set(config.heads)
        pooled_width = int(self.encoder.embedding_channels)

        self.segmentation_head = SegmentationHead(
            self.decoder.feature_channels, self.num_classes,
            levels=config.deep_supervision_levels) \
            if HeadName.SEGMENTATION in heads else None
        self.presence_head = PresenceHead(pooled_width, hidden=config.embedding_hidden,
                                          dropout=config.dropout) \
            if HeadName.PRESENCE in heads else None
        self.size_head = SizeHead(pooled_width, hidden=config.embedding_hidden,
                                  dropout=config.dropout) \
            if HeadName.SIZE in heads else None
        self.quality_head = QualityHead(
            texture_channels=config.quality_texture_channels,
            artifact_classes=ARTIFACT_CLASSES,
            hidden=max(64, config.embedding_hidden // 2),
            dropout=config.dropout, in_channels=len(self.modalities)) \
            if HeadName.QUALITY in heads else None
        self.embedding_head = EmbeddingHead(pooled_width, config.embedding_dim,
                                            hidden=config.embedding_hidden) \
            if HeadName.EMBEDDING in heads else None

        log.info("network built", extra={"context": self.describe()})

    # ------------------------------------------------------------------ #
    @property
    def embedding_spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(dimension=int(self.config.embedding_dim),
                             source="encoder_bottleneck", normalized=True)

    @property
    def deep_supervision_levels(self) -> int:
        return self.segmentation_head.levels if self.segmentation_head else 0

    def forward(self, image: torch.Tensor, *,
                availability: torch.Tensor | None = None,
                need_features: bool = False,
                need_segmentation: bool = True) -> NetworkOutput:
        """Run the network.

        ``need_segmentation=False`` skips the decoder entirely — the path a downstream
        module takes when it only wants an embedding, and the reason the description
        heads hang off the bottleneck rather than off decoder features.
        """
        encoder_features = self.encoder(image, availability)
        pooled = self.encoder.pool(encoder_features)

        output = NetworkOutput(pooled=pooled)
        if need_features:
            output.encoder_features = list(encoder_features)

        if need_segmentation and self.segmentation_head is not None:
            decoder_features = self.decoder(encoder_features)
            output.segmentation = self.segmentation_head(decoder_features)
            if need_features:
                output.decoder_features = list(decoder_features)

        if self.presence_head is not None:
            output.presence = self.presence_head(pooled)
        if self.size_head is not None:
            output.size = self.size_head(pooled)
        if self.quality_head is not None:
            # The only head that sees the image itself, and the only one that does not
            # read the encoder at all. Both are deliberate: instance normalisation
            # removes the per-sample intensity statistics image quality is made of, and
            # concatenating the bottleneck made the head collapse onto the mean. See
            # QualityHead.
            output.quality, output.artifact = self.quality_head(image)
        if self.embedding_head is not None:
            output.embedding = self.embedding_head(pooled)
        return output

    @torch.no_grad()
    def embed(self, image: torch.Tensor,
              availability: torch.Tensor | None = None) -> torch.Tensor:
        """Latent embedding alone, without running the decoder."""
        return self.forward(image, availability=availability,
                            need_segmentation=False).embedding    # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    def head_modules(self) -> dict[HeadName, nn.Module]:
        """The heads that were actually built, by name."""
        mapping = {
            HeadName.SEGMENTATION: self.segmentation_head,
            HeadName.PRESENCE: self.presence_head,
            HeadName.SIZE: self.size_head,
            HeadName.QUALITY: self.quality_head,
            HeadName.EMBEDDING: self.embedding_head,
        }
        return {name: module for name, module in mapping.items() if module is not None}

    def parameter_count(self) -> dict[str, int]:
        def count(module: nn.Module | None) -> int:
            return sum(p.numel() for p in module.parameters()) if module else 0

        heads = {name.value: count(module)
                 for name, module in self.head_modules().items()}
        return {"total": sum(p.numel() for p in self.parameters()),
                "encoder": count(self.encoder), "decoder": count(self.decoder),
                **{f"head_{k}": v for k, v in heads.items()}}

    def describe(self) -> dict[str, Any]:
        return {
            "version": BRAIN_VISION_VERSION,
            "encoder": self.config.encoder,
            "decoder": self.config.decoder,
            "stage_channels": list(self.config.stage_channels),
            "classes": {int(r.value): r.label for r in TumorRegion},
            "heads": [h.value for h in self.head_modules()],
            "deep_supervision_levels": self.deep_supervision_levels,
            "embedding": self.embedding_spec.to_dict(),
            "quality_artifact_classes": list(ARTIFACT_ORDER) + ["clean"],
            "modalities": [m.to_dict() for m in self.modalities],
            "input_size": list(self.config.input_size),
            "parameters": self.parameter_count(),
        }


def build_network(config: ModelConfig) -> BrainVisionNetwork:
    """Factory. The seam a service layer or a test wires through."""
    return BrainVisionNetwork(config)


def downsample_label(label: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Nearest-neighbour label downsampling, for a deep-supervision level.

    Nearest rather than area or bilinear: an averaged label is not a label. A small
    enhancing focus that occupies three pixels at full resolution either survives to the
    coarse grid or does not, and pretending it is 40% present at 24x24 teaches the coarse
    classifier a target it can never match.
    """
    return torch.nn.functional.interpolate(
        label[:, None].float(), size=tuple(int(s) for s in size),
        mode="nearest")[:, 0].long()


__all__ = ["BrainVisionNetwork", "NetworkOutput", "build_network", "downsample_label"]
