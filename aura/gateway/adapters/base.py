"""Abstract base for modality adapters.

A ``ModalityAdapter`` is the gateway-level strategy that replaces the
hard-coded procedural branches in ``app.py`` and ``pipeline.py``.  Each
concrete adapter wraps one modality's intake gate, standardization, and
analysis call, presenting a uniform three-phase interface:

1. **inspect** — decide whether this asset belongs to the adapter's domain
   (cheap, no model load).
2. **standardize** — decode / resample / normalise into a model-ready form.
3. **analyze** — run the modality-specific pipeline and return a unified
   ``EngineOutput``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InspectionResult:
    """Verdict from :meth:`ModalityAdapter.inspect`."""

    accepted: bool
    reason: str = ""
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardizedAsset:
    """Output of :meth:`ModalityAdapter.standardize` — a model-ready payload.

    ``payload`` is adapter-owned and opaque to the platform: a chest adapter
    puts a ``StudyInput`` here, a neuro adapter puts a foundation-pipeline
    result.  Surrounding fields are platform-visible metadata.
    """

    study_id: str
    case_id: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineOutput:
    """Unified output of :meth:`ModalityAdapter.analyze`.

    Wraps the case bundle (or whatever the engine produced) plus provenance
    metadata so callers don't have to know which engine ran.
    """

    case_id: str
    study_id: str
    bundle: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class ModalityAdapter(ABC):
    """Strategy interface for modality-specific intake, standardization, and analysis.

    Concrete adapters are registered with :func:`~gateway.adapters.registry.register_adapter`
    and looked up by modality string at dispatch time.
    """

    modality: str
    display_name: str

    @abstractmethod
    def inspect(self, asset_path: str, asset_meta: dict[str, Any] | None = None,
                **kwargs) -> InspectionResult:
        """Decide whether this adapter should handle the given asset.

        Must not raise on bad input — a rejection is returned, not thrown.
        """

    @abstractmethod
    def standardize(self, asset_path: str, asset_meta: dict[str, Any] | None = None,
                    **kwargs) -> StandardizedAsset:
        """Decode / normalise the asset into a model-ready form.

        Raise on unreadable input; the caller converts to an HTTP error.
        """

    @abstractmethod
    async def analyze(self, standardized: StandardizedAsset,
                      pipeline: Any, store: Any,
                      on_case_created: Any | None = None,
                      **kwargs) -> EngineOutput:
        """Run the modality-specific analysis and return unified output.
        """
