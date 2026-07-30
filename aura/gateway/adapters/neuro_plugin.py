"""Brain MRI modality plugin."""
from __future__ import annotations

from typing import Any

from aura.gateway.adapters.base_plugin import BaseModalityPlugin, PixelSignature


class NeuroPlugin(BaseModalityPlugin):
    modality = "brain_mri"
    display_name = "Brain MRI"
    pixel_signature = PixelSignature(
        mime_types=("image/", "application/dicom", "application/nifti"),
        extensions=(".nii", ".nii.gz", ".nrrd", ".dcm", ".dicom", ".png", ".jpg", ".jpeg"),
        min_dims=2,
        max_dims=4,
    )

    def create_inspector(self) -> Any:
        from aura.gateway.adapters.neuro import NeuroAdapter
        adapter = NeuroAdapter()
        return adapter.inspect

    def create_standardizer(self) -> Any:
        from aura.gateway.adapters.neuro import NeuroAdapter
        adapter = NeuroAdapter()
        return adapter.standardize

    def create_engine(self) -> Any:
        from aura.gateway.adapters.neuro import NeuroAdapter
        adapter = NeuroAdapter()
        return adapter.analyze

    def pipeline_hooks(self) -> dict[str, Any]:
        return {
            "mri_gate": "services.vision.mri_gate.validate_mri",
        }
