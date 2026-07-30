"""Pluggable Modality Architecture — BaseModalityPlugin.

New modalities (ECG, CT, Ultrasound, etc.) register as fully decoupled plugins
that bring their own pixel signatures, engine adapters, and pipeline hooks.
The FastAPI gateway routes and orchestrators remain untouched when new modalities
are added — they just call ``PluginRegistry.resolve()``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PixelSignature:
    """Fingerprint that identifies whether raw bytes belong to this modality.

    ``mime_types``  — accepted MIME types (e.g. "application/dicom").
    ``extensions``  — file extensions (e.g. ".dcm", ".nii").
    ``magic_bytes`` — optional prefix bytes for format sniffing.
    ``min_dims``    — minimum array dimensions (e.g. 2 for images, 3 for volumes).
    ``max_dims``    — maximum array dimensions.
    """
    mime_types: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    magic_bytes: bytes = b""
    min_dims: int = 2
    max_dims: int = 3


class BaseModalityPlugin(ABC):
    """Abstract base for a modality plugin.

    Subclass this, set ``modality`` and ``display_name``, implement the three
    hooks, then register with ``PluginRegistry.register(MyPlugin)``.
    """

    modality: str
    display_name: str
    pixel_signature: PixelSignature = PixelSignature()

    @abstractmethod
    def create_inspector(self) -> Any:
        """Return a callable that inspects raw bytes → accepted/rejected."""

    @abstractmethod
    def create_standardizer(self) -> Any:
        """Return a callable that decodes bytes → StandardizedAsset."""

    @abstractmethod
    def create_engine(self) -> Any:
        """Return the modality-specific analysis engine."""

    def pipeline_hooks(self) -> dict[str, Any]:
        """Optional hooks that the pipeline orchestrator calls at known stages.

        Returns a dict keyed by hook name (e.g. "pre_reasoning", "post_safety").
        Plugins can inject custom logic without modifying the pipeline.
        """
        return {}

    def validate_signature(self, file_bytes: bytes | None = None,
                           filename: str = "",
                           content_type: str = "") -> bool:
        """Check whether the incoming asset matches this plugin's pixel signature."""
        sig = self.pixel_signature
        if filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if sig.extensions and ext not in sig.extensions:
                return False
        if content_type and sig.mime_types:
            if not any(content_type.startswith(m) for m in sig.mime_types):
                return False
        if file_bytes and sig.magic_bytes:
            if not file_bytes[:len(sig.magic_bytes)] == sig.magic_bytes:
                return False
        return True
