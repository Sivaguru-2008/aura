"""Architecture registry — the seam that makes 3D U-Net, SwinUNETR, and nnU-Net
additions rather than rewrites.

The training pipeline never names a class. It asks the registry for an encoder and a
decoder by string, and everything downstream — the heads, the losses, the deep
supervision schedule, the checkpoint layout — is expressed against the two protocols
below rather than against any particular implementation. A new architecture is a new
module with a ``@register_encoder`` decorator and a matching decoder; nothing in
``train.py``, ``losses.py``, or ``inference.py`` changes.

The contract, in full:

* An **encoder** takes ``(B, C, H, W)`` (or ``(B, C, D, H, W)``, for a 3D
  implementation) and returns a list of feature maps, *finest first*. It declares
  ``feature_channels`` and ``strides`` so a decoder can be built against it without
  running a forward pass, and ``embedding_channels`` so the embedding head knows its own
  input width. The last entry of the list is the bottleneck.
* A **decoder** takes that list and returns its own list of feature maps, again finest
  first, with ``feature_channels`` declared. It does not produce logits — the
  segmentation head does, at as many levels as deep supervision asks for.

Declared-but-unimplemented architectures
----------------------------------------
``unet3d``, ``swin_unetr``, and ``nnunet`` are registered as *declarations*. Asking for
one raises :class:`~aura.backend.vision.brain.errors.ArchitectureUnavailable` naming what
would be needed. This is the same posture the MRI Foundation Layer takes with N4 bias
correction and skull stripping, and for the same reason: a roadmap entry that raises is
honest, while an alias that quietly returns the 2D network would produce a model card
claiming an architecture that never ran.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from aura.backend.core.shared.logging import get_logger
from ..errors import ArchitectureUnavailable

log = get_logger("vision.brain.registry")


@runtime_checkable
class EncoderBackbone(Protocol):
    """Feature extractor. The one component every future NeuroMind model reuses."""

    #: Channel width of each returned feature map, finest first.
    feature_channels: tuple[int, ...]
    #: Cumulative spatial stride of each feature map relative to the input.
    strides: tuple[int, ...]
    #: Width of the pooled representation the embedding head consumes.
    embedding_channels: int

    def forward(self, x: Any) -> list[Any]:
        """Return feature maps finest first; the last is the bottleneck."""
        ...


@runtime_checkable
class DecoderBackbone(Protocol):
    """Upsampling path. Returns feature maps, never logits."""

    feature_channels: tuple[int, ...]
    strides: tuple[int, ...]

    def forward(self, features: Sequence[Any]) -> list[Any]:
        ...


_ENCODERS: dict[str, Callable[..., Any]] = {}
_DECODERS: dict[str, Callable[..., Any]] = {}
#: name -> what it would take to implement it.
_DECLARED: dict[str, str] = {}


def register_encoder(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
        _ENCODERS[name] = factory
        return factory

    return decorate


def register_decoder(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
        _DECODERS[name] = factory
        return factory

    return decorate


def declare_architecture(name: str, requirement: str) -> None:
    """Record an architecture the pipeline is designed for but does not implement."""
    _DECLARED[name] = requirement


def available_encoders() -> tuple[str, ...]:
    return tuple(sorted(_ENCODERS))


def available_decoders() -> tuple[str, ...]:
    return tuple(sorted(_DECODERS))


def declared_architectures() -> dict[str, str]:
    return dict(_DECLARED)


def build_encoder(name: str, **kwargs: Any) -> Any:
    return _build(_ENCODERS, name, "encoder", **kwargs)


def build_decoder(name: str, **kwargs: Any) -> Any:
    return _build(_DECODERS, name, "decoder", **kwargs)


def _build(registry: dict[str, Callable[..., Any]], name: str, kind: str,
           **kwargs: Any) -> Any:
    factory = registry.get(name)
    if factory is not None:
        return factory(**kwargs)
    if name in _DECLARED:
        raise ArchitectureUnavailable(
            f"the {kind} {name!r} is declared as a supported extension point but is "
            f"not implemented in this deployment: {_DECLARED[name]}",
            detail={"requested": name, "requirement": _DECLARED[name],
                    "implemented": sorted(registry)})
    raise ArchitectureUnavailable(
        f"no {kind} named {name!r} is registered",
        detail={"requested": name, "implemented": sorted(registry),
                "declared": sorted(_DECLARED)})


# --------------------------------------------------------------------------- #
# The roadmap, as executable declarations rather than a comment.
# --------------------------------------------------------------------------- #
declare_architecture(
    "unet3d",
    "a volumetric encoder/decoder pair plus a patch-based sampler. The cache already "
    "stores whole volumes as (Z, C, H, W) memmaps, so the dataset is the only piece "
    "that would change; the losses, heads, metrics, and checkpoints are dimension-"
    "agnostic.")
declare_architecture(
    "swin_unetr",
    "a Swin transformer encoder. It satisfies the EncoderBackbone protocol as written "
    "— feature maps finest first with declared channels and strides — so it plugs in "
    "at the registry, but the weights and the windowed-attention implementation are "
    "not vendored here.")
declare_architecture(
    "nnunet",
    "nnU-Net's self-configuring planner, which chooses patch size, spacing, and "
    "topology from a dataset fingerprint. The fingerprint it needs is exactly what the "
    "ingest cache manifest records; the planner itself is a separate dependency.")
declare_architecture(
    "vit",
    "a plain vision-transformer encoder with a convolutional decoder. Needs patch "
    "embedding at the input stem and positional encodings sized to the training grid.")


__all__ = [
    "DecoderBackbone", "EncoderBackbone", "available_decoders", "available_encoders",
    "build_decoder", "build_encoder", "declare_architecture", "declared_architectures",
    "register_decoder", "register_encoder",
]
