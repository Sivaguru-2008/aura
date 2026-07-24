"""AURA NeuroView: MRI-only visualization payloads for NeuroMind cases.

NeuroView is deliberately downstream of diagnosis. It consumes the standardized MRI
study and the :class:`BrainVisionOutput` already produced by NeuroMind, then builds a
browser-ready volume on the same grid as the segmentation head's output. It does not
read chest artefacts, does not infer missing anatomy, and marks unavailable layers as
unavailable rather than filling them with plausible-looking masks.
"""
from __future__ import annotations

import base64
from typing import Any, Sequence

import numpy as np

from backend.engines.neuro.multisequence import MultiSequenceStudy
from backend.vision.brain.dataset import fit_to_grid, normalize_slice
from backend.vision.brain.types import ModalitySpec, TumorRegion

UNAVAILABLE = "Unable to compute from current MRI study."
NOT_AVAILABLE = "Not Available"
NEUROVIEW_VERSION = "1.0.0"


def build_neuroview_payload(
    *,
    case_id: str,
    study: Any,
    output: Any,
    modalities: Sequence[ModalitySpec],
    min_brain_fraction: float,
) -> dict[str, Any]:
    """Build the MRI visualization artifact for a completed NeuroMind result."""
    prepared = _model_grid_volume(
        study=study,
        output=output,
        modalities=modalities,
        min_brain_fraction=min_brain_fraction,
    )
    if prepared is None:
        return _unavailable(case_id, getattr(output, "study_id", ""), UNAVAILABLE)

    volume, source, spacing, brain_mask = prepared
    encoded = _encode_volume(volume)
    if encoded is None:
        return _unavailable(
            case_id,
            getattr(output, "study_id", ""),
            "Unable to compute from current MRI study.",
        )

    masks = _segmentation_layers(output, volume.shape, brain_mask)
    return {
        "neuroview_version": NEUROVIEW_VERSION,
        "case_id": case_id,
        "study_id": getattr(output, "study_id", ""),
        "status": "available",
        "message": "",
        "renderer": {
            "preferred": "vtk.js",
            "fallback": "three.js",
            "gpu_required": True,
            "controls": [
                "rotate",
                "zoom",
                "pan",
                "reset_camera",
                "axial_slice",
                "coronal_slice",
                "sagittal_slice",
                "opacity",
                "window_level",
            ],
        },
        "source": source,
        "metadata": {
            "spacing_mm": list(spacing) if spacing is not None else None,
            "orientation": source.get("orientation") or NOT_AVAILABLE,
            "model_version": getattr(output, "model_version", NOT_AVAILABLE),
            "diagnosis": {
                "tumor_probability": getattr(output, "tumor_probability", None),
                "tumor_present": getattr(output, "tumor_present", None),
            },
            "processing": output.processing.to_dict()
            if getattr(output, "processing", None) is not None
            else {},
        },
        "volume": encoded,
        "layers": masks,
        "unavailable_text": NOT_AVAILABLE,
    }


def _unavailable(case_id: str, study_id: str, message: str) -> dict[str, Any]:
    return {
        "neuroview_version": NEUROVIEW_VERSION,
        "case_id": case_id,
        "study_id": study_id,
        "status": "unavailable",
        "message": message,
        "renderer": {"preferred": "vtk.js", "fallback": "three.js"},
        "source": {},
        "metadata": {},
        "volume": None,
        "layers": [],
        "unavailable_text": NOT_AVAILABLE,
    }


def _model_grid_volume(
    *,
    study: Any,
    output: Any,
    modalities: Sequence[ModalitySpec],
    min_brain_fraction: float,
) -> tuple[np.ndarray, dict[str, Any], tuple[float, float, float] | None, np.ndarray | None] | None:
    volumes, source, spacing, brain_mask = _assemble_source_volume(study, output, modalities)
    if volumes is None:
        return None

    brain = np.any(volumes > 0, axis=0)
    per_slice = brain.reshape(-1, brain.shape[2]).mean(axis=0)
    keep = np.flatnonzero(per_slice >= float(min_brain_fraction))
    if keep.size == 0:
        keep = np.arange(volumes.shape[3])

    input_size = tuple(getattr(output.processing, "input_size", ()) or ())
    if len(input_size) != 2 or input_size[0] <= 0 or input_size[1] <= 0:
        seg = getattr(output, "segmentation", None)
        if seg is None or np.asarray(seg).ndim != 3:
            return None
        input_size = tuple(int(v) for v in np.asarray(seg).shape[1:3])

    selected_channel = int(source["channel_index"])
    planes: list[np.ndarray] = []
    mask_planes: list[np.ndarray] = []
    for z in keep:
        z_int = int(z)
        slice_channels = np.ascontiguousarray(volumes[..., z_int], dtype=np.float32)
        normalised = normalize_slice(slice_channels, slice_channels.max(axis=0) > 0)
        fitted, _ = fit_to_grid(
            normalised,
            np.zeros(slice_channels.shape[1:], dtype=np.uint8),
            input_size,
        )
        planes.append(np.asarray(fitted[selected_channel], dtype=np.float32))

        if brain_mask is not None:
            dummy = np.zeros((1, brain_mask.shape[0], brain_mask.shape[1]), dtype=np.float32)
            _, fitted_mask = fit_to_grid(
                dummy,
                np.asarray(brain_mask[:, :, z_int], dtype=np.uint8),
                input_size,
            )
            mask_planes.append(fitted_mask.astype(bool))

    if not planes:
        return None

    volume = np.stack(planes, axis=2).astype(np.float32)
    fitted_brain = np.stack(mask_planes, axis=2) if mask_planes else None
    return volume, source, spacing, fitted_brain


def _assemble_source_volume(
    study: Any,
    output: Any,
    modalities: Sequence[ModalitySpec],
) -> tuple[np.ndarray | None, dict[str, Any], tuple[float, float, float] | None, np.ndarray | None]:
    used = tuple(getattr(output.processing, "sequences_used", ()) or ())
    selected_key = used[0] if used else (modalities[0].key if modalities else "")

    if isinstance(study, MultiSequenceStudy):
        volumes = np.asarray(study.volumes, dtype=np.float32)
        keys = tuple(study.sequence_keys)
        index = keys.index(selected_key) if selected_key in keys else 0
        return volumes, {
            "type": "mri_model_input",
            "sequence": keys[index] if keys else NOT_AVAILABLE,
            "sequence_label": keys[index].upper() if keys else NOT_AVAILABLE,
            "channel_index": index,
            "orientation": "RAS",
            "volume_source": "NeuroMind multi-sequence MRI preprocessing output",
        }, study.spacing_mm, None

    if not hasattr(study, "first"):
        return None, {}, None, None

    planes: list[np.ndarray | None] = []
    reference: tuple[int, int, int] | None = None
    spacing: tuple[float, float, float] | None = None
    selected_index = 0
    selected_series = None

    for index, spec in enumerate(modalities):
        series = study.first(spec.sequence)
        if series is None:
            planes.append(None)
            continue
        array = np.asarray(series.volume.array, dtype=np.float32)
        if reference is None:
            reference = array.shape
            spacing = tuple(float(v) for v in series.spacing)
        if array.shape != reference:
            planes.append(None)
            continue
        if spec.key == selected_key:
            selected_index = index
            selected_series = series
        planes.append(array)

    if reference is None:
        return None, {}, None, None
    if selected_series is None:
        selected_series = next((study.first(spec.sequence) for spec in modalities
                                if study.first(spec.sequence) is not None), None)

    volumes = np.stack([
        plane if plane is not None else np.zeros(reference, dtype=np.float32)
        for plane in planes
    ], axis=0)
    mask = _true_brain_mask(selected_series)
    label = _sequence_label(modalities[selected_index]) if modalities else selected_key
    return volumes, {
        "type": "mri_model_input",
        "sequence": selected_key or NOT_AVAILABLE,
        "sequence_label": label,
        "channel_index": selected_index,
        "orientation": getattr(selected_series, "orientation", NOT_AVAILABLE),
        "series_id": getattr(selected_series, "series_id", NOT_AVAILABLE),
        "volume_source": "MRI Foundation Layer standardized volume prepared for NeuroMind",
    }, spacing, mask


def _sequence_label(spec: ModalitySpec) -> str:
    return spec.label or spec.key.upper()


def _true_brain_mask(series: Any | None) -> np.ndarray | None:
    slot = getattr(series, "brain_mask", None)
    if slot is None or not getattr(slot, "is_brain_mask", False):
        return None
    mask = getattr(slot, "mask", None)
    return np.asarray(mask, dtype=bool) if mask is not None else None


def _encode_volume(volume: np.ndarray) -> dict[str, Any] | None:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return None
    low = float(finite.min())
    high = float(finite.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    scaled = np.clip((volume - low) / (high - low), 0.0, 1.0)
    uint16 = np.rint(scaled * np.iinfo(np.uint16).max).astype("<u2")
    return {
        "dims": [int(v) for v in volume.shape],
        "encoding": "base64",
        "dtype": "uint16",
        "endianness": "little",
        "data": _b64(uint16),
        "intensity": {
            "min": low,
            "max": high,
            "window": high - low,
            "level": (high + low) / 2.0,
            "window_level_supported": True,
        },
    }


def _segmentation_layers(
    output: Any,
    volume_shape: tuple[int, int, int],
    brain_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    seg = getattr(output, "segmentation", None)
    if seg is None:
        return [_unavailable_layer("tumor"), _unavailable_layer("edema"),
                _unavailable_layer("necrosis"), _unavailable_layer("healthy_tissue")]
    seg = np.asarray(seg)
    if seg.ndim != 3 or seg.shape[0] != volume_shape[2] or tuple(seg.shape[1:]) != volume_shape[:2]:
        return [_unavailable_layer("tumor"), _unavailable_layer("edema"),
                _unavailable_layer("necrosis"), _unavailable_layer("healthy_tissue")]

    aligned = np.transpose(seg.astype(np.uint8), (1, 2, 0))
    layers = [
        _mask_layer("tumor", "Tumor", aligned > 0, "#ff5d5d",
                    "Derived from NeuroMind segmentation classes > 0."),
        _mask_layer("edema", "Edema", aligned == TumorRegion.EDEMA.value, "#f4b64e",
                    "Derived from the NeuroMind edema segmentation class."),
        _mask_layer(
            "necrosis",
            "Necrosis",
            aligned == TumorRegion.NECROTIC_CORE.value,
            "#8b7cf7",
            "Derived from the NeuroMind necrotic/non-enhancing core segmentation class.",
        ),
    ]
    if brain_mask is None or brain_mask.shape != volume_shape:
        layers.append(_unavailable_layer("healthy_tissue"))
    else:
        layers.append(_mask_layer(
            "healthy_tissue",
            "Healthy tissue",
            brain_mask & (aligned == TumorRegion.BACKGROUND.value),
            "#4be1c3",
            "Derived from an existing brain mask minus NeuroMind tumor segmentation.",
        ))
    return layers


def _mask_layer(
    key: str,
    label: str,
    mask: np.ndarray,
    color: str,
    provenance: str,
) -> dict[str, Any]:
    data = np.ascontiguousarray(mask.astype(np.uint8))
    return {
        "key": key,
        "label": label,
        "available": True,
        "message": "",
        "color": color,
        "opacity": 0.45,
        "dims": [int(v) for v in data.shape],
        "encoding": "base64",
        "dtype": "uint8",
        "data": _b64(data),
        "voxel_count": int(data.sum()),
        "provenance": provenance,
    }


def _unavailable_layer(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": key.replace("_", " ").title(),
        "available": False,
        "message": NOT_AVAILABLE,
        "color": "",
        "opacity": None,
        "dims": [],
        "encoding": None,
        "dtype": None,
        "data": None,
        "voxel_count": None,
        "provenance": UNAVAILABLE,
    }


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.asfortranarray(array).tobytes(order="F")).decode("ascii")


__all__ = ["NEUROVIEW_VERSION", "build_neuroview_payload"]
