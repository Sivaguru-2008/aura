from __future__ import annotations

import pytest
import base64

pytest.importorskip("torch")
import json

import numpy as np

from aura.backend.engines.neuro.multisequence import MultiSequenceStudy
from aura.backend.engines.neuro.neuroview import build_neuroview_payload
from aura.backend.vision.brain.output import (
    BrainVisionOutput,
    FeatureMaps,
    ProcessingMetadata,
    QualityMetadata,
    build_regions,
)
from aura.backend.vision.brain.types import DEFAULT_MODALITIES


def _output(segmentation: np.ndarray) -> BrainVisionOutput:
    confidence = np.full(segmentation.shape, 0.8, dtype=np.float32)
    presence = np.array([0.9, 0.2, 0.7, 0.4], dtype=np.float32)
    return BrainVisionOutput(
        study_id="STU-MR-view",
        segmentation=segmentation,
        confidence=confidence,
        tumor_probability=0.9,
        regions=build_regions(segmentation, confidence, presence, None, (1.0, 1.0, 1.0)),
        embedding=None,
        embedding_spec=None,
        features=FeatureMaps(maps=None, pooled=None, stride=8),
        processing=ProcessingMetadata(
            study_id="STU-MR-view",
            slices_processed=segmentation.shape[0],
            sequences_used=("flair", "t1", "t1ce", "t2"),
            sequences_missing=(),
            device="cpu",
            inference_ms=1.0,
            input_size=segmentation.shape[1:],
            spacing_mm=(1.0, 1.0, 1.0),
        ),
        quality=QualityMetadata(predicted_score=None),
        model_version="brain@test",
    )


def _study() -> MultiSequenceStudy:
    volumes = np.zeros((4, 4, 4, 3), dtype=np.float32)
    for z in range(3):
        plane = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0 + z
        volumes[0, :, :, z] = plane
        volumes[1, :, :, z] = plane * 0.5
        volumes[2, :, :, z] = plane * 0.25
        volumes[3, :, :, z] = plane * 0.75
    return MultiSequenceStudy(
        volumes=volumes,
        sequence_keys=("flair", "t1", "t1ce", "t2"),
        spacing_mm=(1.0, 1.0, 1.0),
        order_source="test",
        order_endorsement={"available": False},
    )


def test_neuroview_uses_mri_volume_and_existing_segmentation_masks():
    segmentation = np.zeros((3, 4, 4), dtype=np.uint8)
    segmentation[0, 1, 1] = 1
    segmentation[1, 1, 2] = 2
    segmentation[2, 2, 2] = 3

    payload = build_neuroview_payload(
        case_id="CASE-MR-1",
        study=_study(),
        output=_output(segmentation),
        modalities=DEFAULT_MODALITIES,
        min_brain_fraction=0.0,
    )

    json.dumps(payload)
    assert payload["status"] == "available"
    assert payload["volume"]["dims"] == [4, 4, 3]
    raw = base64.b64decode(payload["volume"]["data"])
    assert len(raw) == 4 * 4 * 3 * np.dtype(np.uint16).itemsize
    layers = {layer["key"]: layer for layer in payload["layers"]}
    assert layers["tumor"]["voxel_count"] == 3
    assert layers["edema"]["voxel_count"] == 1
    assert layers["necrosis"]["voxel_count"] == 1
    assert layers["healthy_tissue"]["available"] is False
    assert layers["healthy_tissue"]["message"] == "Not Available"


def test_neuroview_does_not_fabricate_masks_when_grids_do_not_align():
    segmentation = np.zeros((2, 4, 4), dtype=np.uint8)
    segmentation[0, 1, 1] = 2

    payload = build_neuroview_payload(
        case_id="CASE-MR-2",
        study=_study(),
        output=_output(segmentation),
        modalities=DEFAULT_MODALITIES,
        min_brain_fraction=0.0,
    )

    assert payload["status"] == "available"
    for layer in payload["layers"]:
        assert layer["available"] is False
        assert layer["data"] is None
        assert layer["message"] == "Not Available"
