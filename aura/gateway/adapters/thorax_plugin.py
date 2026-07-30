"""Thorax (Chest X-ray) modality plugin."""
from __future__ import annotations

from typing import Any

from aura.gateway.adapters.base_plugin import BaseModalityPlugin, PixelSignature


class ThoraxPlugin(BaseModalityPlugin):
    modality = "chest_xray"
    display_name = "Chest X-ray"
    pixel_signature = PixelSignature(
        mime_types=("image/", "application/dicom"),
        extensions=(".png", ".jpg", ".jpeg", ".dcm", ".dicom"),
        min_dims=2,
        max_dims=2,
    )

    def create_inspector(self) -> Any:
        from aura.gateway.adapters.thorax import ThoraxAdapter
        adapter = ThoraxAdapter()
        return adapter.inspect

    def create_standardizer(self) -> Any:
        from aura.gateway.adapters.thorax import ThoraxAdapter
        adapter = ThoraxAdapter()
        return adapter.standardize

    def create_engine(self) -> Any:
        from aura.gateway.adapters.thorax import ThoraxAdapter
        adapter = ThoraxAdapter()
        return adapter.analyze

    def pipeline_hooks(self) -> dict[str, Any]:
        return {
            "intake_gate": "services.vision.xray_gate.validate_cxr",
        }
