"""Convolutional building blocks shared by the encoder and the decoder.

Instance normalisation rather than batch normalisation is the one choice worth
explaining. Segmentation batches are small — 16 slices at 192x192 on an 8 GB card — and
batch statistics over 16 samples are noisy enough to hurt; more importantly, MR
intensity has no absolute scale, so a normaliser that couples samples together is
normalising across scans whose intensities mean different things. Instance norm treats
each sample's each channel on its own, which is what nnU-Net settled on for the same
reasons.
"""
from __future__ import annotations

import torch
from torch import nn


def make_norm(kind: str, channels: int) -> nn.Module:
    """Normalisation layer by name. ``group`` uses 8 channels per group."""
    kind = kind.lower()
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(max(1, channels // 8), channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalisation {kind!r}")


def make_activation(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {kind!r}")


class ConvNormAct(nn.Sequential):
    """Convolution, normalisation, activation — the unit everything else is built from."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3,
                 stride: int = 1, norm: str = "instance",
                 activation: str = "leaky_relu") -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            make_norm(norm, out_channels),
            make_activation(activation),
        )


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection.

    Residual rather than plain: the encoder is the artefact every future NeuroMind
    module is meant to reuse, so it needs to stay trainable if someone deepens it, and
    a plain stack of five stages is already at the depth where optimisation starts to
    be the limiting factor rather than capacity.
    """

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1,
                 norm: str = "instance", activation: str = "leaky_relu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, stride=stride, norm=norm,
                                 activation=activation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_norm(norm, out_channels))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = make_activation(activation)
        # Projection only when the shape actually changes: an identity skip is both
        # cheaper and better-conditioned, so it is kept wherever it is valid.
        self.project: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                make_norm(norm, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.project(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.activation(out + identity)


class ResidualStage(nn.Sequential):
    """``blocks`` residual blocks, the first of which may downsample."""

    def __init__(self, in_channels: int, out_channels: int, blocks: int, *,
                 stride: int = 1, norm: str = "instance",
                 activation: str = "leaky_relu", dropout: float = 0.0) -> None:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride, norm=norm,
                                activation=activation, dropout=dropout)]
        layers += [ResidualBlock(out_channels, out_channels, norm=norm,
                                 activation=activation, dropout=dropout)
                   for _ in range(max(0, blocks - 1))]
        super().__init__(*layers)


class UpsampleBlock(nn.Module):
    """Transposed convolution up, concatenate the skip, then residual blocks.

    Transposed convolution rather than interpolate-then-convolve. The checkerboard
    artefact that transposed convolutions are criticised for appears when the kernel
    size is not divisible by the stride; 2x2 with stride 2 is exactly divisible, and the
    learned upsampling is worth having at the boundary of a small enhancing focus, where
    bilinear interpolation systematically blurs the class away.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *,
                 blocks: int = 2, norm: str = "instance",
                 activation: str = "leaky_relu", dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.blocks = ResidualStage(out_channels + skip_channels, out_channels, blocks,
                                    norm=norm, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # Odd input sizes make the upsampled map differ by a pixel. Padding the
            # smaller one is safe and exact; cropping the skip would discard a real
            # column of anatomy at the image edge.
            x = _pad_to(x, skip.shape[-2:])
        return self.blocks(torch.cat([x, skip], dim=1))


def _pad_to(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    diff_h = int(size[0]) - x.shape[-2]
    diff_w = int(size[1]) - x.shape[-1]
    return nn.functional.pad(
        x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])


__all__ = ["ConvNormAct", "ResidualBlock", "ResidualStage", "UpsampleBlock",
           "make_activation", "make_norm"]
