"""The shared encoder — the part of this network that is meant to outlive it.

Everything else in the Brain Vision Engine is replaceable: the decoder is one way to
turn features into a mask, the heads are five ways to read a description off them. The
encoder is the actual asset. It is what ``brain_encoder.pt`` holds, what a future
progression model or digital twin loads instead of training from scratch, and what the
latent embedding is computed from.

Two consequences for the design.

**The input stem is per-modality.** Each sequence gets its own first convolution before
anything is mixed, and the results are summed. A shared 4-channel stem would tie the
learned filters to the exact channel set the network was trained with, so adding PET or
a diffusion map later would mean reinitialising the first layer and losing the
pretrained weights it sits on top of. Per-modality stems mean a new modality is a new
stem — the rest of the encoder transfers untouched. It also makes a *missing* sequence
representable: drop its stem's contribution, and the network sees the sum of what is
actually there rather than a zero-filled channel it will interpret as signal.

**The bottleneck exposes a pooled representation explicitly.** ``embedding_channels`` is
part of the encoder's declared contract, so the embedding head can be built and loaded
against an encoder without either knowing about the other.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from backend.vision.brain.model.blocks import ConvNormAct, ResidualStage
from backend.vision.brain.model.registry import register_encoder
from backend.vision.brain.types import DEFAULT_MODALITIES, ModalitySpec


class ModalityStem(nn.Module):
    """One convolution per input sequence, summed.

    ``availability`` lets a caller mark a sequence as absent for a given sample. The
    corresponding stem contributes nothing and the sum is rescaled by the number of
    present modalities, so a study with three of four sequences produces features on the
    same scale rather than features that are uniformly 25% quieter — which the following
    normalisation would partly hide and partly turn into a systematic bias.
    """

    def __init__(self, modalities: Sequence[ModalitySpec], out_channels: int, *,
                 norm: str = "instance", activation: str = "leaky_relu") -> None:
        super().__init__()
        self.modality_keys = tuple(m.key for m in modalities)
        self.stems = nn.ModuleList([
            ConvNormAct(1, out_channels, kernel_size=3, norm=norm,
                        activation=activation)
            for _ in modalities])

    def forward(self, x: torch.Tensor,
                availability: torch.Tensor | None = None) -> torch.Tensor:
        outputs = [stem(x[:, index:index + 1]) for index, stem in
                   enumerate(self.stems)]
        stacked = torch.stack(outputs, dim=0)             # (M, B, C, H, W)
        if availability is None:
            return stacked.mean(dim=0)
        weights = availability.to(stacked.dtype).clamp(0.0, 1.0)
        weights = weights.t().reshape(len(self.stems), -1, 1, 1, 1)
        present = weights.sum(dim=0).clamp(min=1.0)
        return (stacked * weights).sum(dim=0) / present


class ResidualUNetEncoder2D(nn.Module):
    """Five-stage residual encoder. Satisfies
    :class:`~backend.vision.brain.model.registry.EncoderBackbone`."""

    def __init__(self, *, modalities: Sequence[ModalitySpec] = DEFAULT_MODALITIES,
                 stage_channels: Sequence[int] = (32, 64, 128, 256, 320),
                 blocks_per_stage: Sequence[int] = (1, 2, 2, 2, 2),
                 norm: str = "instance", activation: str = "leaky_relu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        channels = tuple(int(c) for c in stage_channels)
        blocks = tuple(int(b) for b in blocks_per_stage)
        if len(blocks) < len(channels):
            blocks = blocks + (blocks[-1],) * (len(channels) - len(blocks))

        self.modalities = tuple(modalities)
        self.stem = ModalityStem(self.modalities, channels[0], norm=norm,
                                 activation=activation)

        stages: list[nn.Module] = [
            ResidualStage(channels[0], channels[0], blocks[0], stride=1, norm=norm,
                          activation=activation, dropout=dropout)]
        for level in range(1, len(channels)):
            stages.append(ResidualStage(
                channels[level - 1], channels[level], blocks[level], stride=2,
                norm=norm, activation=activation, dropout=dropout))
        self.stages = nn.ModuleList(stages)

        self.feature_channels: tuple[int, ...] = channels
        self.strides: tuple[int, ...] = tuple(2 ** i for i in range(len(channels)))
        self.embedding_channels: int = channels[-1]

    def forward(self, x: torch.Tensor,
                availability: torch.Tensor | None = None) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        out = self.stem(x, availability)
        for stage in self.stages:
            out = stage(out)
            features.append(out)
        return features

    def pool(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pooled bottleneck representation, the embedding head's input.

        Average and maximum pooling concatenated would double the width; average alone
        is used because the heads that consume it are predicting extensive quantities
        (is a tumour present, how large is it) for which the mean over the field is the
        natural statistic, and because a wider vector here would make
        ``brain_embedding_head.pt`` incompatible with an encoder whose pooling changed.
        """
        return torch.flatten(nn.functional.adaptive_avg_pool2d(features[-1], 1), 1)


@register_encoder("residual_unet2d")
def _build_residual_unet2d(**kwargs) -> ResidualUNetEncoder2D:
    return ResidualUNetEncoder2D(**kwargs)


__all__ = ["ModalityStem", "ResidualUNetEncoder2D"]
