"""Modality adapters — pluggable strategy objects that decouple gateway
intake from modality-specific pre-processing and analysis.

Each adapter encapsulates the inspect / standardize / analyze sequence for one
imaging modality, replacing the procedural branches that previously lived in
``app.py`` and ``pipeline.py``.

Step 1 v1.1: Plugin architecture — new modalities register as fully decoupled
``BaseModalityPlugin`` subclasses via ``PluginRegistry``.
"""
from aura.gateway.adapters.base import (
    InspectionResult,
    ModalityAdapter,
    StandardizedAsset,
    EngineOutput,
)
from aura.gateway.adapters.registry import get_adapter, get_adapter_for_modality, register_adapter
from aura.gateway.adapters.base_plugin import BaseModalityPlugin, PixelSignature
from aura.gateway.adapters.plugin_registry import (
    get_plugin,
    plugin_modalities,
    register_plugin,
    registered_plugins,
    resolve_plugin,
)

__all__ = [
    "InspectionResult",
    "ModalityAdapter",
    "StandardizedAsset",
    "EngineOutput",
    "get_adapter",
    "get_adapter_for_modality",
    "register_adapter",
    "BaseModalityPlugin",
    "PixelSignature",
    "get_plugin",
    "plugin_modalities",
    "register_plugin",
    "registered_plugins",
    "resolve_plugin",
]
