"""Pluggable Modality Plugin Registry.

Replaces the legacy adapter registry with a fully decoupled plugin system.
New modalities register their ``BaseModalityPlugin`` subclass and the gateway
auto-discovers them — no changes to routes or orchestrators required.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from aura.gateway.adapters.base_plugin import BaseModalityPlugin, PixelSignature

# Global plugin registry
_plugin_registry: dict[str, BaseModalityPlugin] = {}


def register_plugin(plugin: BaseModalityPlugin) -> BaseModalityPlugin:
    """Register a modality plugin instance.

    Can be used as a decorator::

        @register_plugin
        class ECGPlugin(BaseModalityPlugin):
            modality = "ecg"
    """
    if isinstance(plugin, type):
        plugin = plugin()
    _plugin_registry[plugin.modality] = plugin
    return plugin


def get_plugin(modality: str) -> BaseModalityPlugin | None:
    """Look up a registered plugin by modality string."""
    return _plugin_registry.get(modality)


def registered_plugins() -> dict[str, BaseModalityPlugin]:
    """Return a copy of all registered plugins."""
    return dict(_plugin_registry)


def plugin_modalities() -> list[str]:
    """Return sorted list of registered modality keys."""
    return sorted(_plugin_registry.keys())


def resolve_plugin(file_bytes: bytes | None = None, filename: str = "",
                   content_type: str = "") -> BaseModalityPlugin | None:
    """Auto-detect which plugin should handle an incoming asset.

    Tries each registered plugin's ``validate_signature`` in registration
    order and returns the first match.
    """
    for plugin in _plugin_registry.values():
        if plugin.validate_signature(file_bytes, filename, content_type):
            return plugin
    return None


def _auto_discover_plugins() -> None:
    """Import and register built-in plugins.

    This scans the gateway/adapters/ directory for any module whose name
    ends with ``_plugin`` or ``plugin`` and imports it, triggering its
    registration via the ``@register_plugin`` decorator.
    """
    from aura.gateway.adapters.thorax_plugin import ThoraxPlugin
    from aura.gateway.adapters.neuro_plugin import NeuroPlugin

    register_plugin(ThoraxPlugin())
    register_plugin(NeuroPlugin())


_auto_discover_plugins()
