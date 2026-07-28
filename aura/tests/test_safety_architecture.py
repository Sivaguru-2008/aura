from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from schemas.clinical import Diagnosis, Finding
from schemas.contracts import CaseBundle, CaseState
from services.safety.controller import ClinicalSafetyController, ClinicalSafetyException
from services.safety.readiness import ClinicalDecisionReadinessEngine
from gateway.storage import compute_provenance_hash

def test_safety_controller_data_integrity():
    controller = ClinicalSafetyController()
    
    # 1. Valid CXR image (aspect ratio 1.0, high std)
    valid_img = np.zeros((100, 100))
    valid_img[:50, :] = 1.0 # Standard deviation is 0.5, well above 0.04
    
    study_cxr = MagicMock()
    study_cxr.modality.value = "CXR"
    
    ok, reason, checks = controller.inspect_data_integrity(study_cxr, valid_img)
    assert ok is True
    assert checks["aspect_ratio"] == 1.0
    assert checks["gray_std"] > 0.04

    # 2. Invalid CXR aspect ratio (height/width = 5.0)
    invalid_aspect_img = np.ones((500, 100))
    ok, reason, checks = controller.inspect_data_integrity(study_cxr, invalid_aspect_img)
    assert ok is False
    assert "aspect ratio" in reason or "proportions" in reason

    # 3. Invalid CXR gray std (nearly uniform, std = 0)
    flat_img = np.ones((100, 100))
    ok, reason, checks = controller.inspect_data_integrity(study_cxr, flat_img)
    assert ok is False
    assert "uniform" in reason or "tonal" in reason or "gray_std" in checks
    assert checks["gray_std"] < 0.04

def test_readiness_engine_coverage():
    engine = ClinicalDecisionReadinessEngine()
    
    # Normal diagnosis has no imaging/symptoms template requirements, but has spo2 labs
    findings_map = {}
    
    study = MagicMock()
    study.multimodal = MagicMock()
    study.multimodal.labs = MagicMock()
    study.multimodal.labs.spo2 = 98.0
    
    coverage = engine.evaluate_coverage(Diagnosis.NORMAL, findings_map, study)
    assert coverage == 1.0 # spo2 is known, total requirements = 1, coverage = 1.0

    # Test missing spo2 lab
    study_missing_labs = MagicMock()
    study_missing_labs.multimodal = MagicMock()
    study_missing_labs.multimodal.labs = MagicMock()
    study_missing_labs.multimodal.labs.spo2 = None
    
    coverage_missing = engine.evaluate_coverage(Diagnosis.NORMAL, findings_map, study_missing_labs)
    assert coverage_missing == 0.0

def test_provenance_hash_is_stable():
    bundle = CaseBundle(
        case_id="CASE-TEST-123",
        study_id="STU-TEST-123",
        state=CaseState.READY,
        priority_score=0.85,
        image=[1.0, 2.0, 3.0, 4.0],
        image_shape=(2, 2)
    )
    
    h1 = compute_provenance_hash(bundle)
    h2 = compute_provenance_hash(bundle)
    
    assert isinstance(h1, str)
    assert len(h1) == 64 # SHA-256 length
    assert h1 == h2 # stable
