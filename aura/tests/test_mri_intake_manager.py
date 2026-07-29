import os
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import pytest

from aura.backend.foundation.mri.intake_manager import MRIIntakeManager
from aura.backend.foundation.mri.errors import StudyValidationError
from aura.backend.foundation.mri.types import SequenceType
from aura.backend.engines.neuro.multisequence import MultiSequenceStudy
from .test_mri_foundation import write_nifti1, ras_affine, head_phantom

def create_phantom_study(temp_dir: Path, shapes=None, spacings=None, affines=None, file_names=None):
    if file_names is None:
        file_names = {s: f"{s}.nii" for s in ["flair", "t1", "t1ce", "t2"]}
    if shapes is None:
        shapes = {}
    if spacings is None:
        spacings = {}
    if affines is None:
        affines = {}

    paths = {}
    for s, name in file_names.items():
        clean_s = "t1ce" if "t1ce" in s else ("t1" if "t1" in s else ("t2" if "t2" in s else "flair"))
        
        shape = shapes.get(s)
        if shape is None:
            shape = shapes.get(clean_s)
        if shape is None:
            shape = (40, 40, 30)

        spacing = spacings.get(s)
        if spacing is None:
            spacing = spacings.get(clean_s)
        if spacing is None:
            spacing = (2.0, 2.0, 2.0)

        affine = affines.get(s)
        if affine is None:
            affine = affines.get(clean_s)
        if affine is None:
            affine = ras_affine(spacing)

        arr = head_phantom(shape, spacing)
        p = temp_dir / name
        write_nifti1(p, arr, affine)
        paths[s] = p
    return paths

def test_intake_manager_success():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        create_phantom_study(temp_path)

        manager = MRIIntakeManager()
        study = manager.process(temp_path)

        assert isinstance(study, MultiSequenceStudy)
        assert study.volumes.shape == (4, 40, 40, 30)
        assert study.sequence_keys == ("flair", "t1", "t1ce", "t2")
        assert study.spacing_mm == (2.0, 2.0, 2.0)
        assert "stacking" in study.order_source

def test_intake_manager_missing_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        # Omit T1ce
        create_phantom_study(temp_path, file_names={"flair": "flair.nii", "t1": "t1.nii", "t2": "t2.nii"})

        manager = MRIIntakeManager()
        with pytest.raises(StudyValidationError) as exc:
            manager.process(temp_path)
        assert "Missing T1ce sequence" in str(exc.value)

def test_intake_manager_duplicate_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        # Create T1ce duplicates
        create_phantom_study(temp_path, file_names={
            "flair": "flair.nii",
            "t1": "t1.nii",
            "t1ce": "t1ce.nii",
            "t2": "t2.nii",
            "t1ce_dup": "t1ce_post.nii"
        })

        manager = MRIIntakeManager()
        with pytest.raises(StudyValidationError) as exc:
            manager.process(temp_path)
        assert "Duplicate T1ce detected" in str(exc.value)

def test_intake_manager_dimension_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        shapes = {
            "flair": (40, 40, 30),
            "t1": (40, 40, 30),
            "t1ce": (40, 40, 30),
            "t2": (40, 40, 25), # Z is different
        }
        create_phantom_study(temp_path, shapes=shapes)

        manager = MRIIntakeManager()
        with pytest.raises(StudyValidationError) as exc:
            manager.process(temp_path)
        assert "T2 volume dimensions do not match T1" in str(exc.value)

def test_intake_manager_spacing_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        spacings = {
            "flair": (2.0, 2.0, 2.0),
            "t1": (2.0, 2.0, 2.0),
            "t1ce": (2.0, 2.0, 2.5), # Z spacing is different
            "t2": (2.0, 2.0, 2.0),
        }
        create_phantom_study(temp_path, spacings=spacings)

        manager = MRIIntakeManager()
        with pytest.raises(StudyValidationError) as exc:
            manager.process(temp_path)
        assert "T1ce voxel spacing does not match T1" in str(exc.value)

def test_intake_manager_affine_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        affines = {
            "flair": ras_affine((2.0, 2.0, 2.0)),
            "t1": ras_affine((2.0, 2.0, 2.0)),
            "t1ce": ras_affine((2.0, 2.0, 2.0), origin=(-90.0, -100.0, -60.0)), # offset origin
            "t2": ras_affine((2.0, 2.0, 2.0)),
        }
        create_phantom_study(temp_path, affines=affines)

        manager = MRIIntakeManager()
        with pytest.raises(StudyValidationError) as exc:
            manager.process(temp_path)
        assert "Affine matrices are inconsistent" in str(exc.value)

def test_intake_manager_zip_file():
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        # Create study files
        create_phantom_study(temp_path)

        # Create zip archive
        zip_filepath = temp_path / "study.zip"
        with zipfile.ZipFile(zip_filepath, "w") as zf:
            for name in ["flair.nii", "t1.nii", "t1ce.nii", "t2.nii"]:
                zf.write(temp_path / name, name)
                # Remove file after zipping to ensure manager only sees ZIP
                (temp_path / name).unlink()

        manager = MRIIntakeManager()
        study = manager.process(zip_filepath)

        assert isinstance(study, MultiSequenceStudy)
        assert study.volumes.shape == (4, 40, 40, 30)
        assert study.spacing_mm == (2.0, 2.0, 2.0)

def test_intake_manager_4d_nifti_success():
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        # Create a 4D array of shape (40, 40, 30, 4)
        vol = np.stack([head_phantom((40, 40, 30)) for _ in range(4)], axis=-1)
        affine = ras_affine((2.0, 2.0, 2.0))
        nii_path = temp_path / "study_4d.nii"
        write_nifti1(nii_path, vol, affine)

        mock_endorsement = {
            "available": True,
            "assumed_order": ["flair", "t1", "t1ce", "t2"],
            "predicted_order": ["flair", "t1", "t1ce", "t2"],
            "endorsing": 4,
            "required": 3,
            "checker_accuracy_per_channel": 0.9,
            "checker_study_endorsement_rate": 0.9,
            "slices_voted": 9,
        }

        with patch("aura.backend.engines.neuro.multisequence._check_order", return_value=mock_endorsement):
            manager = MRIIntakeManager()
            study = manager.process(nii_path)

        assert isinstance(study, MultiSequenceStudy)
        assert study.volumes.shape == (4, 40, 40, 30)
        assert study.spacing_mm == (2.0, 2.0, 2.0)


def test_upload_guard_valid_nifti_gz():
    from aura.gateway.security import validate_mri_content
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp)
        arr = head_phantom((40, 40, 30))
        affine = ras_affine((2.0, 2.0, 2.0))
        p = temp_path / "valid.nii.gz"
        write_nifti1(p, arr, affine, gzipped=True)
        payload = p.read_bytes()
        
        # Should pass without raising any exception
        validate_mri_content(payload, "valid.nii.gz")


def test_upload_guard_corrupted_gzip():
    from aura.gateway.security import validate_mri_content
    from fastapi import HTTPException
    
    payload = b"not a gzip archive at all"
    with pytest.raises(HTTPException) as exc:
        validate_mri_content(payload, "corrupted.nii.gz")
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "Corrupted MRI archive"


def test_upload_guard_arbitrary_gzip():
    from aura.gateway.security import validate_mri_content
    from fastapi import HTTPException
    import gzip
    
    payload = gzip.compress(b"a" * 1000) # Arbitrary large gzip file
    with pytest.raises(HTTPException) as exc:
        validate_mri_content(payload, "arbitrary.nii.gz")
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "Unsupported MRI format"


def test_upload_guard_invalid_nifti():
    from aura.gateway.security import validate_mri_content
    from fastapi import HTTPException
    
    payload = b"just some text, not a nifti"
    with pytest.raises(HTTPException) as exc:
        validate_mri_content(payload, "invalid.nii")
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "Invalid NIfTI file"
