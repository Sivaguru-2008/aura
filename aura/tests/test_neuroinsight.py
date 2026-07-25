from __future__ import annotations

import numpy as np
import pytest

from backend.engines.neuro.neuroinsight import (
    compute_neuroinsight,
    get_anatomical_localization,
    get_study_affine,
)
from backend.vision.brain.output import (
    BrainVisionOutput,
    FeatureMaps,
    ProcessingMetadata,
    QualityMetadata,
    build_regions,
)


class MockStudy:
    def __init__(self, spacing_mm=(1.0, 1.0, 1.0)):
        self.spacing_mm = spacing_mm
        self.series = []


def _mock_output(segmentation: np.ndarray, confidence: np.ndarray | None = None) -> BrainVisionOutput:
    if confidence is None:
        confidence = np.full(segmentation.shape, 0.95, dtype=np.float32)
    presence = np.array([0.9, 0.2, 0.7, 0.4], dtype=np.float32)
    return BrainVisionOutput(
        study_id="STU-MR-test-insight",
        segmentation=segmentation,
        confidence=confidence,
        tumor_probability=0.9,
        regions=build_regions(segmentation, confidence, presence, None, (1.0, 1.0, 1.0)),
        embedding=None,
        embedding_spec=None,
        features=FeatureMaps(maps=None, pooled=None, stride=8),
        processing=ProcessingMetadata(
            study_id="STU-MR-test-insight",
            slices_processed=segmentation.shape[0],
            sequences_used=("flair", "t1", "t1ce", "t2"),
            sequences_missing=(),
            device="cpu",
            inference_ms=1.0,
            input_size=segmentation.shape[1:],
            spacing_mm=(1.0, 1.0, 1.0),
        ),
        quality=QualityMetadata(predicted_score=None),
        model_version="brain@test-insight",
    )


def test_neuroinsight_handles_empty_segmentation():
    segmentation = np.zeros((5, 10, 10), dtype=np.uint8)
    study = MockStudy()
    output = _mock_output(segmentation)
    
    res = compute_neuroinsight(study, output)
    assert res["status"] == "unavailable"
    assert res["lesion_count"] == 0
    assert len(res["lesions"]) == 0
    assert "Measurement unavailable" in res["message"]


def test_neuroinsight_computes_mathematical_metrics():
    # Construct a simple 3D volume with 2 distinct lesion components (connected components)
    segmentation = np.zeros((10, 20, 20), dtype=np.uint8)
    # Lesion 1: a 2x2x2 cube
    segmentation[2:4, 2:4, 2:4] = 1 # necrotic/core
    # Lesion 2: a single voxel
    segmentation[7, 12, 12] = 3 # enhancing

    study = MockStudy(spacing_mm=(2.0, 2.0, 2.0))
    output = _mock_output(segmentation)

    res = compute_neuroinsight(study, output)
    assert res["status"] == "available"
    assert res["lesion_count"] == 2
    assert len(res["lesions"]) == 2

    # Verify Lesion 1 metrics (voxel count, volume, centroid)
    # Lesion 1: 8 voxels
    # Spacing is 2.0, so voxel volume is 2 * 2 * 2 = 8.0 mm3
    # Total volume = 8 * 8.0 = 64.0 mm3
    l1 = res["lesions"][0]
    assert l1["voxel_count"] == 8
    assert l1["volume_mm3"] == 64.0
    
    # Bounding box in voxel indices
    assert l1["bbox_voxel"] == [2, 2, 2, 3, 3, 3] # [xmin, ymin, zmin, xmax, ymax, zmax]
    
    # Centroid in voxel coordinates should be 2.5
    assert l1["centroid_voxel"] == [2.5, 2.5, 2.5]
    
    # Lesion dimensions: (xmax-xmin+1)*sx = (3-2+1)*2 = 4.0
    assert l1["dimensions_mm"] == [4.0, 4.0, 4.0]

    # Verify Lesion 2 metrics
    l2 = res["lesions"][1]
    assert l2["voxel_count"] == 1
    assert l2["volume_mm3"] == 8.0
    assert l2["centroid_voxel"] == [12.0, 12.0, 7.0] # cx=12, cy=12, cz=7
    assert l2["dimensions_mm"] == [2.0, 2.0, 2.0]


def test_neuroinsight_surface_area_exactness():
    # Create a 1x2x3 block
    segmentation = np.zeros((10, 10, 10), dtype=np.uint8)
    segmentation[2:5, 3:5, 4:5] = 1 # z: 2..4 (size 3), y: 3..4 (size 2), x: 4..4 (size 1)
    
    # voxel count = 1 * 2 * 3 = 6
    # Faces:
    # faces perp to X (sz * sy = 1 * 1 = 1): 2 exposed faces for each of the 6 voxels?
    # No, let's calculate exposed boundary faces directly.
    # For a block of size 1x2x3:
    # perp to X: 2 faces of area 2x3 = 4 faces? No, X has 1 voxel width, so there are 2 ends of area 2*3 = 6. Total X-faces = 12.
    # perp to Y: 2 ends of area 1x3 = 3. Total Y-faces = 6.
    # perp to Z: 2 ends of area 1x2 = 2. Total Z-faces = 4.
    # Total surface area for 1x1x1 spacing: 2*(1*2) + 2*(1*3) + 2*(2*3) = 4 + 6 + 12 = 22.
    
    study = MockStudy(spacing_mm=(1.0, 1.0, 1.0))
    output = _mock_output(segmentation)
    res = compute_neuroinsight(study, output)
    
    assert res["lesion_count"] == 1
    l = res["lesions"][0]
    assert l["voxel_count"] == 6
    assert l["surface_area_mm2"] == 22.0


def test_get_anatomical_localization():
    # Frontal Lobe
    loc_frontal = get_anatomical_localization(20.0, 40.0, 10.0)
    assert loc_frontal["lobe"] == "Frontal Lobe"
    assert "Frontal" in loc_frontal["structure"]
    assert loc_frontal["laterality"] == "Right"
    assert loc_frontal["atlas_registration"] == "Approximate Anatomical Localization"

    # Occipital Lobe
    loc_occipital = get_anatomical_localization(-10.0, -85.0, 5.0)
    assert loc_occipital["lobe"] == "Occipital Lobe"
    assert "Calcarine" in loc_occipital["structure"]
    assert loc_occipital["laterality"] == "Left"

    # Cerebellum
    loc_cerebellar = get_anatomical_localization(30.0, -60.0, -25.0)
    assert loc_cerebellar["lobe"] == "Cerebellum"
    
    # Brainstem
    loc_stem = get_anatomical_localization(0.0, -20.0, -25.0)
    assert loc_stem["lobe"] == "Brainstem"
