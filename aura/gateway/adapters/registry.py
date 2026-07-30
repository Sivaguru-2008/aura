"""Modality adapter registry — the factory that maps modality strings to adapters.

The router resolves a modality string; the registry resolves the concrete adapter
that handles inspection, standardization, and analysis for that modality.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.gateway.adapters.base import ModalityAdapter

_registry: dict[str, type[ModalityAdapter]] = {}


def register_adapter(adapter_cls: type[ModalityAdapter]) -> type[ModalityAdapter]:
    """Register an adapter class by its ``modality`` attribute.

    Can be used as a decorator::

        @register_adapter
        class MyAdapter(ModalityAdapter):
            modality = "chest_xray"
    """
    key = adapter_cls.modality
    _registry[key] = adapter_cls
    return adapter_cls


def get_adapter(modality: str) -> ModalityAdapter | None:
    """Return a *new instance* of the adapter for ``modality``, or ``None``."""
    cls = _registry.get(modality)
    if cls is None:
        return None
    return cls()


def get_adapter_for_modality(modality: str) -> ModalityAdapter | None:
    """Alias for :func:`get_adapter` — resolve modality string to adapter instance."""
    return get_adapter(modality)


def registered_modalities() -> list[str]:
    """Return all registered modality keys."""
    return sorted(_registry.keys())


def _auto_register_builtins() -> None:
    """Register the built-in adapters on first import."""
    from aura.gateway.adapters.thorax import ThoraxAdapter
    from aura.gateway.adapters.neuro import NeuroAdapter

    register_adapter(ThoraxAdapter)
    register_adapter(NeuroAdapter)


_auto_register_builtins()
