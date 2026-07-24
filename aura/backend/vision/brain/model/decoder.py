"""The decoder — features back up to full resolution, at every scale on the way.

It returns feature *maps*, not logits, and it returns one per level rather than only the
finest. Both are for deep supervision, which is the reason this is not a plain U-Net
decoder: the segmentation head attaches a classifier to as many of these levels as the
configuration asks for, and each gets its own loss term against a downsampled label.

Why supervise the intermediate levels at all. A single loss at full resolution has to
travel back through four upsampling stages before it reaches the encoder's bottleneck,
and early in training it arrives there heavily attenuated — the coarse layers spend
several epochs learning very little. A direct loss at 24x24 says "this region is
tumour" to the bottleneck immediately. The measurable effect is on convergence speed and
on the coarse features themselves, which is exactly what a module whose main product is
a reusable encoder should care about.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from backend.vision.brain.model.blocks import UpsampleBlock
from backend.vision.brain.model.registry import register_decoder


class UNetDecoder2D(nn.Module):
    """Skip-connected upsampling path. Satisfies
    :class:`~backend.vision.brain.model.registry.DecoderBackbone`."""

    def __init__(self, *, encoder_channels: Sequence[int],
                 blocks_per_stage: int = 2, norm: str = "instance",
                 activation: str = "leaky_relu", dropout: float = 0.0) -> None:
        super().__init__()
        channels = tuple(int(c) for c in encoder_channels)
        if len(channels) < 2:
            raise ValueError("the decoder needs at least two encoder levels")

        # Built coarse to fine: level i consumes the previous decoder output and the
        # encoder skip at the same resolution.
        blocks: list[UpsampleBlock] = []
        current = channels[-1]
        for level in range(len(channels) - 2, -1, -1):
            blocks.append(UpsampleBlock(current, channels[level], channels[level],
                                        blocks=blocks_per_stage, norm=norm,
                                        activation=activation, dropout=dropout))
            current = channels[level]
        self.blocks = nn.ModuleList(blocks)

        # Declared finest first, matching the encoder's convention.
        self.feature_channels: tuple[int, ...] = tuple(channels[:-1])
        self.strides: tuple[int, ...] = tuple(2 ** i for i in range(len(channels) - 1))

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """``features`` is the encoder pyramid, finest first. Returns finest first."""
        if len(features) != len(self.blocks) + 1:
            raise ValueError(
                f"the decoder was built for {len(self.blocks) + 1} encoder levels but "
                f"received {len(features)}")
        out = features[-1]
        produced: list[torch.Tensor] = []
        for position, block in enumerate(self.blocks):
            skip = features[len(features) - 2 - position]
            out = block(out, skip)
            produced.append(out)
        produced.reverse()                                # coarse-to-fine -> finest first
        return produced


@register_decoder("unet2d")
def _build_unet2d(**kwargs) -> UNetDecoder2D:
    return UNetDecoder2D(**kwargs)


__all__ = ["UNetDecoder2D"]
