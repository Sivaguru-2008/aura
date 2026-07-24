"""Integration tests for clinical honesty, hard modality isolation, and safety gating."""
from __future__ import annotations

import zipfile
import tempfile
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from gateway.app import app
from backend.core.shared.errors import UnsupportedModality, ModalityConflict
from schemas.contracts import CaseBundle, CaseState, AbstentionReason
from backend.services.reasoning.progression import LongitudinalAnalyzer
from backend.services.reasoning.tracking import TumorTracker

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def create_dummy_dicom(
    path: Path,
    series_description: str,
    patient_id: str = "PAT001",
    patient_name: str = "Patient One",
    modality: str = "MR",
    study_uid: str = "1.2.826.0.1.3680043.2.1125.0",
    series_uid: str = "1.2.826.0.1.3680043.2.1125.1",
    instance_number: int = 1,
    z_position: float = -60.0,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = series_description
    ds.InstanceNumber = instance_number
    ds.BodyPartExamined = "BRAIN" if modality == "MR" else "CHEST"
    ds.ProtocolName = "BRAIN ROUTINE" if modality == "MR" else "CHEST AP"
    ds.Rows = 64
    ds.Columns = 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.PixelSpacing = [2.0, 2.0]
    ds.SliceThickness = 2.0
    ds.SpacingBetweenSlices = 2.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [-100.0, -100.0, z_position]
    ds.PixelData = np.random.randint(10, 100, size=(64, 64), dtype=np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=False)

def create_study_zip(
    zip_path: Path,
    series_definitions: list[dict],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for idx, s_def in enumerate(series_definitions):
                # Write slice 1
                file_path1 = tmp_path / f"slice_{idx}_1.dcm"
                create_dummy_dicom(
                    file_path1,
                    series_description=s_def.get("desc", "AX FLAIR"),
                    patient_id=s_def.get("pid", "PAT001"),
                    patient_name=s_def.get("pname", "Patient One"),
                    modality=s_def.get("modality", "MR"),
                    series_uid=s_def.get("series_uid", f"1.2.826.0.1.3680043.2.1125.{idx}"),
                    instance_number=1,
                    z_position=-60.0,
                )
                zip_file.write(file_path1, file_path1.name)
                
                # Write slice 2
                file_path2 = tmp_path / f"slice_{idx}_2.dcm"
                create_dummy_dicom(
                    file_path2,
                    series_description=s_def.get("desc", "AX FLAIR"),
                    patient_id=s_def.get("pid", "PAT001"),
                    patient_name=s_def.get("pname", "Patient One"),
                    modality=s_def.get("modality", "MR"),
                    series_uid=s_def.get("series_uid", f"1.2.826.0.1.3680043.2.1125.{idx}"),
                    instance_number=2,
                    z_position=-58.0,
                )
                zip_file.write(file_path2, file_path2.name)

# 1. Test Hard Modality Isolation & Routing
def test_modality_isolation_rejects_mismatch(client, tmp_path):
    # Create a brain MRI DICOM
    mri_path = tmp_path / "brain_mri.dcm"
    create_dummy_dicom(mri_path, "AX T2 TSE", modality="MR")

    # Try uploading with declared_modality="chest_xray" (Thorax engine)
    # This should trigger ModalityConflict or reject because MR cannot go to Thorax
    with open(mri_path, "rb") as f:
        response = client.post(
            "/v1/studies/analyze",
            files={"file": (mri_path.name, f, "application/dicom")},
            params={"declared_modality": "chest_xray"}
        )
    assert response.status_code == 422
    assert "conflict" in response.json()["error"]

# 2. Test Brain Sequence Validation - Incomplete Study
def test_sequence_validation_incomplete(client, tmp_path):
    # Only FLAIR and T1, missing T2 and T1CE
    zip_path = tmp_path / "incomplete_study.zip"
    create_study_zip(zip_path, [
        {"desc": "AX FLAIR"},
        {"desc": "AX T1"},
    ])

    with open(zip_path, "rb") as f:
        response = client.post(
            "/v1/studies/analyze",
            files={"file": (zip_path.name, f, "application/zip")},
            params={"declared_modality": "brain_mri"}
        )
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "failed"
    assert "missing" in res["message"].lower()

# 3. Test Brain Sequence Validation - Duplicate Sequence
def test_sequence_validation_duplicate(client, tmp_path):
    # 2 FLAIRs, T1, T2, T1CE (duplicate FLAIR)
    zip_path = tmp_path / "duplicate_study.zip"
    create_study_zip(zip_path, [
        {"desc": "AX FLAIR", "series_uid": "1.2.826.0.1.3680043.2.1125.10"},
        {"desc": "AX FLAIR", "series_uid": "1.2.826.0.1.3680043.2.1125.11"},
        {"desc": "AX T1"},
        {"desc": "AX T2"},
        {"desc": "AX T1CE"},
    ])

    with open(zip_path, "rb") as f:
        response = client.post(
            "/v1/studies/analyze",
            files={"file": (zip_path.name, f, "application/zip")},
            params={"declared_modality": "brain_mri"}
        )
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "failed"
    assert "duplicate" in res["message"].lower()

# 4. Test Brain Sequence Validation - Mixed Patients
def test_sequence_validation_mixed_patients(client, tmp_path):
    zip_path = tmp_path / "mixed_patients.zip"
    create_study_zip(zip_path, [
        {"desc": "AX FLAIR", "pid": "PAT001", "pname": "Patient One"},
        {"desc": "AX T1", "pid": "PAT002", "pname": "Patient Two"},
        {"desc": "AX T2"},
        {"desc": "AX T1CE"},
    ])

    with open(zip_path, "rb") as f:
        response = client.post(
            "/v1/studies/analyze",
            files={"file": (zip_path.name, f, "application/zip")},
            params={"declared_modality": "brain_mri"}
        )
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "failed"
    assert "mixed patients" in res["message"].lower()

# 5. Test Longitudinal Analysis calculation and status
def test_longitudinal_analyzer():
    from schemas.clinical import Diagnosis
    from schemas.contracts import CaseState, SafetyAssessment, VisionResult
    
    # Create two stub bundles
    prev_bundle = CaseBundle(
        case_id="CASE-MR-1",
        study_id="STU-MR-1",
        state=CaseState.READY,
        priority_score=0.5,
        image_shape=(64, 64),
        vision=VisionResult(study_id="STU-MR-1", findings=[], embedding=[], model_version="1.0"),
        volumes={
            "whole_tumor": {"volume_mm3": 1000.0},
            "tumor_core": {"volume_mm3": 500.0},
            "enhancing_tumor": {"volume_mm3": 200.0},
            "edema": {"volume_mm3": 300.0},
        }
    )

    # 1. Test Stable: 1050 mm3 (5% growth, threshold is 20%)
    curr_bundle_stable = CaseBundle(
        case_id="CASE-MR-2",
        study_id="STU-MR-2",
        state=CaseState.READY,
        priority_score=0.5,
        image_shape=(64, 64),
        vision=VisionResult(study_id="STU-MR-2", findings=[], embedding=[], model_version="1.0"),
        volumes={
            "whole_tumor": {"volume_mm3": 1050.0},
            "tumor_core": {"volume_mm3": 500.0},
            "enhancing_tumor": {"volume_mm3": 200.0},
            "edema": {"volume_mm3": 350.0},
        }
    )

    report_stable = LongitudinalAnalyzer.compare(prev_bundle, curr_bundle_stable)
    assert report_stable.status == "Stable"
    assert report_stable.metrics.whole_tumor_change_pct == 0.05

    # 2. Test Progressive Disease: 1250 mm3 (25% growth, threshold is 20%)
    curr_bundle_pd = CaseBundle(
        case_id="CASE-MR-3",
        study_id="STU-MR-3",
        state=CaseState.READY,
        priority_score=0.5,
        image_shape=(64, 64),
        vision=VisionResult(study_id="STU-MR-3", findings=[], embedding=[], model_version="1.0"),
        volumes={
            "whole_tumor": {"volume_mm3": 1250.0},
            "tumor_core": {"volume_mm3": 600.0},
            "enhancing_tumor": {"volume_mm3": 300.0},
            "edema": {"volume_mm3": 350.0},
        }
    )

    report_pd = LongitudinalAnalyzer.compare(prev_bundle, curr_bundle_pd)
    assert report_pd.status == "Progressive Disease"
    assert report_pd.metrics.whole_tumor_change_pct == 0.25

    # 3. Test Regression: 750 mm3 (-25% change, threshold is -20%)
    curr_bundle_reg = CaseBundle(
        case_id="CASE-MR-4",
        study_id="STU-MR-4",
        state=CaseState.READY,
        priority_score=0.5,
        image_shape=(64, 64),
        vision=VisionResult(study_id="STU-MR-4", findings=[], embedding=[], model_version="1.0"),
        volumes={
            "whole_tumor": {"volume_mm3": 750.0},
            "tumor_core": {"volume_mm3": 350.0},
            "enhancing_tumor": {"volume_mm3": 100.0},
            "edema": {"volume_mm3": 300.0},
        }
    )

    report_reg = LongitudinalAnalyzer.compare(prev_bundle, curr_bundle_reg)
    assert report_reg.status == "Regression"
    assert report_reg.metrics.whole_tumor_change_pct == -0.25
