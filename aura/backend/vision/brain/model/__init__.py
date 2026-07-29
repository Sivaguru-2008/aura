"""The Brain Vision network and its extension points.

Importing this package registers the implemented architectures and the declared-but-
unimplemented ones, so :func:`available_encoders` and :func:`declared_architectures`
answer correctly from the first import.
"""
from .blocks import (
    ConvNormAct,
    ResidualBlock,
    ResidualStage,
    UpsampleBlock,
)
from .decoder import UNetDecoder2D
from .encoder import ModalityStem, ResidualUNetEncoder2D
from .heads import (
    EmbeddingHead,
    PresenceHead,
    QualityHead,
    SegmentationHead,
    SizeHead,
)
from .network import (
    BrainVisionNetwork,
    NetworkOutput,
    build_network,
    downsample_label,
)
from .registry import (
    DecoderBackbone,
    EncoderBackbone,
    available_decoders,
    available_encoders,
    build_decoder,
    build_encoder,
    declare_architecture,
    declared_architectures,
    register_decoder,
    register_encoder,
)

__all__ = [
    "BrainVisionNetwork", "NetworkOutput", "build_network", "downsample_label",
    "ResidualUNetEncoder2D", "ModalityStem", "UNetDecoder2D",
    "SegmentationHead", "PresenceHead", "SizeHead", "QualityHead", "EmbeddingHead",
    "ConvNormAct", "ResidualBlock", "ResidualStage", "UpsampleBlock",
    "EncoderBackbone", "DecoderBackbone", "register_encoder", "register_decoder",
    "build_encoder", "build_decoder", "available_encoders", "available_decoders",
    "declare_architecture", "declared_architectures",
]
