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

    volume, source, spacing, brain_mask, keep = prepared
    encoded = _encode_volume(volume)
    if encoded is None:
        return _unavailable(
            case_id,
            getattr(output, "study_id", ""),
            "Unable to compute from current MRI study.",
        )

    # Calculate NeuroInsight post-processing clinical intelligence
    from backend.engines.neuro.neuroinsight import compute_neuroinsight
    insight = compute_neuroinsight(study, output)
    insight_payload = {k: v for k, v in insight.items() if k != "labeled_mask"}

    # Reconstruct segmentations back to original volume grid
    seg = getattr(output, "segmentation", None)
    if seg is not None and keep is not None:
        seg = np.asarray(seg)
        if seg.ndim == 3 and seg.shape[0] == len(keep):
            full_seg = _reconstruct_segmentation(seg, keep, volume.shape)
            
            masks = [
                _mask_layer("tumor", "Tumor", full_seg > 0, "#ff5d5d",
                            "Derived from NeuroMind segmentation classes > 0."),
                _mask_layer("edema", "Edema", full_seg == TumorRegion.EDEMA.value, "#f4b64e",
                            "Derived from the NeuroMind edema segmentation class."),
                _mask_layer(
                    "necrosis",
                    "Necrosis",
                    full_seg == TumorRegion.NECROTIC_CORE.value,
                    "#8b7cf7",
                    "Derived from the NeuroMind necrotic/non-enhancing core segmentation class.",
                ),
            ]
            if brain_mask is None or brain_mask.shape != volume.shape:
                masks.append(_unavailable_layer("healthy_tissue"))
            else:
                masks.append(_mask_layer(
                    "healthy_tissue",
                    "Healthy tissue",
                    brain_mask & (full_seg == TumorRegion.BACKGROUND.value),
                    "#4be1c3",
                    "Derived from an existing brain mask minus NeuroMind tumor segmentation.",
                ))
        else:
            masks = [_unavailable_layer("tumor"), _unavailable_layer("edema"),
                     _unavailable_layer("necrosis"), _unavailable_layer("healthy_tissue")]
    else:
        masks = [_unavailable_layer("tumor"), _unavailable_layer("edema"),
                 _unavailable_layer("necrosis"), _unavailable_layer("healthy_tissue")]

    # Reconstruct connected component labels layer
    if insight.get("status") == "available" and "labeled_mask" in insight and keep is not None:
        labeled_mask = insight["labeled_mask"]
        if labeled_mask.ndim == 3 and labeled_mask.shape[0] == len(keep):
            reconstructed_labels = _reconstruct_segmentation(labeled_mask, keep, volume.shape)
            masks.append(_labeled_layer(
                "lesion_labels",
                "Lesion Labels",
                reconstructed_labels,
                "Derived from AURA NeuroInsight connected components."
            ))
        else:
            masks.append(_unavailable_layer("lesion_labels"))
    else:
        masks.append(_unavailable_layer("lesion_labels"))

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
        "neuroinsight": insight_payload,
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
        "neuroinsight": {
            "status": "unavailable",
            "lesion_count": 0,
            "lesions": [],
            "message": message,
        },
    }


def _reconstruct_segmentation(seg: np.ndarray, keep: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Map the resized/cropped segmentation slices back to the original volume grid."""
    H, W, Z = target_shape
    full_seg = np.zeros((H, W, Z), dtype=np.uint8)
    
    for i, z in enumerate(keep):
        slice_2d = seg[i]
        
        target_h, target_w = H, W
        height, width = slice_2d.shape
        
        if target_h > height:
            pad_h = target_h - height
            before_h = pad_h // 2
            slice_2d = np.pad(slice_2d, ((before_h, pad_h - before_h), (0, 0)))
        elif target_h < height:
            offset_h = (height - target_h) // 2
            slice_2d = slice_2d[offset_h:offset_h + target_h, :]
            
        if target_w > width:
            pad_w = target_w - width
            before_w = pad_w // 2
            slice_2d = np.pad(slice_2d, ((0, 0), (before_w, pad_w - before_w)))
        elif target_w < width:
            offset_w = (width - target_w) // 2
            slice_2d = slice_2d[:, offset_w:offset_w + target_w]
            
        full_seg[:, :, int(z)] = slice_2d
        
    return full_seg


def _model_grid_volume(
    *,
    study: Any,
    output: Any,
    modalities: Sequence[ModalitySpec],
    min_brain_fraction: float,
) -> tuple[np.ndarray, dict[str, Any], tuple[float, float, float] | None, np.ndarray | None, np.ndarray] | None:
    volumes, source, spacing, brain_mask = _assemble_source_volume(study, output, modalities)
    if volumes is None:
        return None

    # Calculate slices kept by model inference based on min_brain_fraction
    brain = np.any(volumes > 0, axis=0)
    per_slice = brain.reshape(-1, brain.shape[2]).mean(axis=0)
    keep = np.flatnonzero(per_slice >= float(min_brain_fraction))
    if keep.size == 0:
        keep = np.arange(volumes.shape[3])

    selected_channel = int(source["channel_index"])
    # Return the raw original channel volume (H, W, Z)
    channel_volume = volumes[selected_channel]
    
    # Ensure there are no NaNs/Infs
    channel_volume = np.nan_to_num(channel_volume, nan=0.0, posinf=0.0, neginf=0.0)

    return channel_volume, source, spacing, brain_mask, keep


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


def _labeled_layer(
    key: str,
    label_name: str,
    labels: np.ndarray,
    provenance: str,
) -> dict[str, Any]:
    data = np.ascontiguousarray(labels.astype(np.uint8))
    return {
        "key": key,
        "label": label_name,
        "available": True,
        "message": "",
        "color": "",
        "opacity": 0.0,  # Functional layer, not rendered as a mask
        "dims": [int(v) for v in data.shape],
        "encoding": "base64",
        "dtype": "uint8",
        "data": base64.b64encode(np.asfortranarray(data).tobytes(order="F")).decode("ascii"),
        "voxel_count": int(np.count_nonzero(data > 0)),
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
