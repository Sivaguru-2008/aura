"""Demo Dataset Generator for AURA.

Generates 9 structured cases demonstrating routing, safety, rejections,
and analysis pathways, along with expected_output.json for each.
"""
from __future__ import annotations

import json
import os
import gzip
from pathlib import Path
import numpy as np

# Set paths
AURA_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEMO_DIR = AURA_DIR / "demo_data"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Structured NIfTI-1 Header Dtype definition
# --------------------------------------------------------------------------- #
_NIFTI1_DTYPE = np.dtype([
    ("sizeof_hdr", "i4"), ("data_type", "S10"), ("db_name", "S18"),
    ("extents", "i4"), ("session_error", "i2"), ("regular", "S1"),
    ("dim_info", "u1"), ("dim", "i2", (8,)),
    ("intent_p1", "f4"), ("intent_p2", "f4"), ("intent_p3", "f4"),
    ("intent_code", "i2"), ("datatype", "i2"), ("bitpix", "i2"),
    ("slice_start", "i2"), ("pixdim", "f4", (8,)), ("vox_offset", "f4"),
    ("scl_slope", "f4"), ("scl_inter", "f4"), ("slice_end", "i2"),
    ("slice_code", "u1"), ("xyzt_units", "u1"), ("cal_max", "f4"),
    ("cal_min", "f4"), ("slice_duration", "f4"), ("toffset", "f4"),
    ("glmax", "i4"), ("glmin", "i4"), ("descrip", "S80"), ("aux_file", "S24"),
    ("qform_code", "i2"), ("sform_code", "i2"),
    ("quatern_b", "f4"), ("quatern_c", "f4"), ("quatern_d", "f4"),
    ("qoffset_x", "f4"), ("qoffset_y", "f4"), ("qoffset_z", "f4"),
    ("srow_x", "f4", (4,)), ("srow_y", "f4", (4,)), ("srow_z", "f4", (4,)),
    ("intent_name", "S16"), ("magic", "S4"),
])

# --------------------------------------------------------------------------- #
# Helpers to generate synthetic image files
# --------------------------------------------------------------------------- #
def write_dummy_nifti(path: Path, array: np.ndarray, descrip: bytes = b"") -> Path:
    hdr_array = np.zeros(1, dtype=_NIFTI1_DTYPE)[0]
    hdr_array["sizeof_hdr"] = 348
    hdr_array["regular"] = b'r'
    hdr_array["dim"][0] = array.ndim
    for i in range(array.ndim):
        hdr_array["dim"][i+1] = array.shape[i]
    hdr_array["datatype"] = 16 # float32
    hdr_array["bitpix"] = 32
    hdr_array["vox_offset"] = 352
    hdr_array["magic"] = b'n+1'
    hdr_array["qform_code"] = 1
    hdr_array["sform_code"] = 1
    hdr_array["srow_x"] = [1.0, 0.0, 0.0, 0.0]
    hdr_array["srow_y"] = [0.0, 1.0, 0.0, 0.0]
    hdr_array["srow_z"] = [0.0, 0.0, 1.0, 0.0]
    hdr_array["pixdim"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    hdr_array["descrip"] = descrip
    
    # write header + padding + voxel data
    payload = hdr_array.tobytes() + b"\x00" * 4 + np.asfortranarray(array.astype(np.float32)).tobytes(order="F")
    path.write_bytes(payload)
    return path

def generate_normal_volume(shape=(40, 40, 30)) -> np.ndarray:
    # Ellipsoid representing head/brain boundary
    grid = np.indices(shape).astype(np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    radii = np.asarray([shape[0] * 0.38, shape[1] * 0.34, shape[2] * 0.40], dtype=np.float32)
    distance = sum(((grid[i] - centre[i]) / radii[i]) ** 2 for i in range(3))
    
    volume = np.abs(np.random.normal(2.0, 0.5, size=shape)).astype(np.float32)
    volume[distance <= 1.0] = 100.0 + np.random.normal(0.0, 2.0, size=int((distance <= 1.0).sum())).astype(np.float32)
    volume[distance <= 0.45] = 130.0
    return volume

def generate_tumor_volume(shape=(40, 40, 30), tumor_radius=0.15, edema=False) -> np.ndarray:
    grid = np.indices(shape).astype(np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    radii = np.asarray([shape[0] * 0.38, shape[1] * 0.34, shape[2] * 0.40], dtype=np.float32)
    distance = sum(((grid[i] - centre[i]) / radii[i]) ** 2 for i in range(3))
    
    volume = np.abs(np.random.normal(2.0, 0.5, size=shape)).astype(np.float32)
    volume[distance <= 1.0] = 100.0 + np.random.normal(0.0, 2.0, size=int((distance <= 1.0).sum())).astype(np.float32)
    
    # Tumor center (offset from brain center)
    t_center = centre + np.asarray([4, 4, 2], dtype=np.float32)
    t_radii = np.asarray([shape[0] * tumor_radius, shape[1] * tumor_radius, shape[2] * tumor_radius], dtype=np.float32)
    t_dist = sum(((grid[i] - t_center[i]) / t_radii[i]) ** 2 for i in range(3))
    
    # Enhancing core
    volume[t_dist <= 0.8] = 200.0 + np.random.normal(0.0, 5.0, size=int((t_dist <= 0.8).sum())).astype(np.float32)
    # Necrotic core
    volume[t_dist <= 0.4] = 80.0
    
    if edema:
        # Edema region around tumor
        edema_radii = t_radii * 1.8
        edema_dist = sum(((grid[i] - t_center[i]) / edema_radii[i]) ** 2 for i in range(3))
        edema_mask = (edema_dist <= 1.0) & (t_dist > 0.8) & (distance <= 1.0)
        volume[edema_mask] = 150.0 + np.random.normal(0.0, 3.0, size=int(edema_mask.sum())).astype(np.float32)
        
    return volume

def write_dicom_with_patient(path: Path, patient_id: str, modality: str = "MR", body_part: str = "HEAD") -> Path:
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
    except ImportError:
        # Return empty file if pydicom is not present
        path.write_bytes(b"DUMMY_DICOM_BYTES")
        return path

    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 190
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = f'1.2.826.0.1.3680043.8.498.1.1.1'
    file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.1.1.2'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1' # Explicit VR Little Endian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.PatientID = patient_id
    ds.PatientName = f"Patient^{patient_id}"
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    
    # Hash patient ID to create valid numeric UIDs
    pid_hash = abs(hash(patient_id)) % 1000000
    ds.StudyInstanceUID = f"1.2.826.0.1.3680043.8.498.2.1"
    ds.SeriesInstanceUID = f"1.2.826.0.1.3680043.8.498.2.2.{pid_hash}"
    ds.SOPInstanceUID = f"1.2.826.0.1.3680043.8.498.2.3.{pid_hash}"
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PixelData = np.zeros((64, 64), dtype=np.uint16).tobytes()
    pydicom.dcmwrite(str(path), ds, write_like_original=True)
    return path

# --------------------------------------------------------------------------- #
# Case generation
# --------------------------------------------------------------------------- #
def main():
    print("[AURA Demo Data] Preparing structured demo dataset...")

    # Case 1: Normal Brain MRI
    print("Generating Case 1: Normal Brain MRI...")
    c1_dir = DEMO_DIR / "1_normal_brain_mri"
    c1_dir.mkdir(exist_ok=True)
    normal_vol = generate_normal_volume()
    write_dummy_nifti(c1_dir / "flair.nii", normal_vol, b"FLAIR")
    write_dummy_nifti(c1_dir / "t1.nii", normal_vol, b"T1")
    write_dummy_nifti(c1_dir / "t1ce.nii", normal_vol, b"T1CE")
    write_dummy_nifti(c1_dir / "t2.nii", normal_vol, b"T2")
    
    expected_c1 = {
        "case_description": "Normal brain MRI study with all four standard sequences.",
        "routing": {
            "detected_modality": "brain_mri",
            "selected_engine": "neuromind",
            "requires_review": False
        },
        "report": {
            "impression": "No evidence of intracranial mass effect, abnormal enhancing mass, or surrounding vasogenic edema.",
            "findings": "The ventricles and sulci are within normal limits. No abnormal signal intensities are seen on FLAIR or T2 sequences.",
            "differential": "Normal study."
        },
        "safety": {
            "abstained": False,
            "top_probability": 0.05,
            "conformal_set": ["normal"]
        },
        "audit_trail": {
            "events": ["case.uploaded", "modality.routed", "inference.completed", "report.generated"]
        }
    }
    (c1_dir / "expected_output.json").write_text(json.dumps(expected_c1, indent=2))

    # Case 2: Small Tumor Brain MRI
    print("Generating Case 2: Small Tumor Brain MRI...")
    c2_dir = DEMO_DIR / "2_small_tumor_brain_mri"
    c2_dir.mkdir(exist_ok=True)
    small_vol = generate_tumor_volume(tumor_radius=0.10, edema=False)
    write_dummy_nifti(c2_dir / "flair.nii", small_vol, b"FLAIR")
    write_dummy_nifti(c2_dir / "t1.nii", small_vol, b"T1")
    write_dummy_nifti(c2_dir / "t1ce.nii", small_vol, b"T1CE")
    write_dummy_nifti(c2_dir / "t2.nii", small_vol, b"T2")
    
    expected_c2 = {
        "case_description": "Brain MRI study demonstrating a focal small enhancing mass in the right hemisphere.",
        "routing": {
            "detected_modality": "brain_mri",
            "selected_engine": "neuromind",
            "requires_review": False
        },
        "report": {
            "impression": "Focal small enhancing lesion in the right hemisphere. Appearance is suspicious for a low-grade glioma or early metastasis.",
            "findings": "There is a well-circumscribed, small enhancing lesion measuring approximately 1.2 cm in the right frontoparietal region.",
            "differential": "Low-grade glioma, metastasis, demyelinating plaque."
        },
        "safety": {
            "abstained": False,
            "top_probability": 0.68,
            "conformal_set": ["malignancy"]
        },
        "audit_trail": {
            "events": ["case.uploaded", "modality.routed", "inference.completed", "report.generated"]
        }
    }
    (c2_dir / "expected_output.json").write_text(json.dumps(expected_c2, indent=2))

    # Case 3: Large Tumor Brain MRI
    print("Generating Case 3: Large Tumor Brain MRI...")
    c3_dir = DEMO_DIR / "3_large_tumor_brain_mri"
    c3_dir.mkdir(exist_ok=True)
    large_vol = generate_tumor_volume(tumor_radius=0.22, edema=True)
    write_dummy_nifti(c3_dir / "flair.nii", large_vol, b"FLAIR")
    write_dummy_nifti(c3_dir / "t1.nii", large_vol, b"T1")
    write_dummy_nifti(c3_dir / "t1ce.nii", large_vol, b"T1CE")
    write_dummy_nifti(c3_dir / "t2.nii", large_vol, b"T2")
    
    expected_c3 = {
        "case_description": "Brain MRI study demonstrating a large necrotic tumor with extensive mass effect.",
        "routing": {
            "detected_modality": "brain_mri",
            "selected_engine": "neuromind",
            "requires_review": False
        },
        "report": {
            "impression": "Large ring-enhancing intra-axial mass in the right hemisphere with central necrosis, substantial surrounding vasogenic edema, and mass effect.",
            "findings": "A large heterogeneously enhancing lesion measuring 4.5 cm is noted in the right hemisphere. Significant surrounding T2/FLAIR hyperintensity is present.",
            "differential": "Glioblastoma Multiforme (GBM), solitary metastasis."
        },
        "safety": {
            "abstained": False,
            "top_probability": 0.94,
            "conformal_set": ["malignancy"]
        },
        "audit_trail": {
            "events": ["case.uploaded", "modality.routed", "inference.completed", "report.generated"]
        }
    }
    (c3_dir / "expected_output.json").write_text(json.dumps(expected_c3, indent=2))

    # Case 4: High Edema Brain MRI
    print("Generating Case 4: High Edema Brain MRI...")
    c4_dir = DEMO_DIR / "4_high_edema_brain_mri"
    c4_dir.mkdir(exist_ok=True)
    edema_vol = generate_tumor_volume(tumor_radius=0.12, edema=True)
    write_dummy_nifti(c4_dir / "flair.nii", edema_vol, b"FLAIR")
    write_dummy_nifti(c4_dir / "t1.nii", edema_vol, b"T1")
    write_dummy_nifti(c4_dir / "t1ce.nii", edema_vol, b"T1CE")
    write_dummy_nifti(c4_dir / "t2.nii", edema_vol, b"T2")
    
    expected_c4 = {
        "case_description": "Brain MRI study showing a moderate tumor surrounded by disproportionately extensive edema.",
        "routing": {
            "detected_modality": "brain_mri",
            "selected_engine": "neuromind",
            "requires_review": False
        },
        "report": {
            "impression": "Focal enhancing lesion with disproportionately extensive surrounding vasogenic edema.",
            "findings": "A 2.1 cm enhancing lesion is present. Surrounding vasogenic edema extends through the right frontal lobe white matter.",
            "differential": "Metastasis, meningioma (if extra-axial), abscess."
        },
        "safety": {
            "abstained": False,
            "top_probability": 0.82,
            "conformal_set": ["malignancy"]
        },
        "audit_trail": {
            "events": ["case.uploaded", "modality.routed", "inference.completed", "report.generated"]
        }
    }
    (c4_dir / "expected_output.json").write_text(json.dumps(expected_c4, indent=2))

    # Case 5: Out of Distribution (OOD) Case
    print("Generating Case 5: Out of Distribution (OOD) Case...")
    c5_dir = DEMO_DIR / "5_ood_case"
    c5_dir.mkdir(exist_ok=True)
    from PIL import Image
    noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(noise).save(c5_dir / "non_medical_noise.png")
    
    expected_c5 = {
        "case_description": "A random noise PNG upload representing non-medical images.",
        "routing": {
            "detected_modality": "unknown",
            "selected_engine": None,
            "requires_review": True,
            "reason": "file could not be decoded as an image or is not a medical radiograph / slice"
        },
        "safety": {
            "abstained": True,
            "abstention_reason": "unrecognised_image"
        },
        "audit_trail": {
            "events": ["case.upload_rejected", "modality.rejected"]
        }
    }
    (c5_dir / "expected_output.json").write_text(json.dumps(expected_c5, indent=2))

    # Case 6: Missing Sequence
    print("Generating Case 6: Missing Sequence...")
    c6_dir = DEMO_DIR / "6_missing_sequence"
    c6_dir.mkdir(exist_ok=True)
    seq_vol = generate_normal_volume()
    # Missing FLAIR sequence, only T1, T1CE, and T2 are present
    write_dummy_nifti(c6_dir / "t1.nii", seq_vol, b"T1")
    write_dummy_nifti(c6_dir / "t1ce.nii", seq_vol, b"T1CE")
    write_dummy_nifti(c6_dir / "t2.nii", seq_vol, b"T2")
    
    expected_c6 = {
        "case_description": "Brain MRI study missing the critical FLAIR sequence.",
        "routing": {
            "detected_modality": "brain_mri",
            "selected_engine": "neuromind",
            "requires_review": True
        },
        "safety": {
            "abstained": True,
            "abstention_reason": "incomplete_study",
            "detail": "Incomplete study: missing required sequences (missing: FLAIR)."
        },
        "audit_trail": {
            "events": ["case.uploaded", "modality.routed", "engine.rejected_preprocess"]
        }
    }
    (c6_dir / "expected_output.json").write_text(json.dumps(expected_c6, indent=2))

    # Case 7: Mixed Patient
    print("Generating Case 7: Mixed Patient...")
    c7_dir = DEMO_DIR / "7_mixed_patient"
    c7_dir.mkdir(exist_ok=True)
    write_dicom_with_patient(c7_dir / "slice1.dcm", "PATIENT_A")
    write_dicom_with_patient(c7_dir / "slice2.dcm", "PATIENT_B")
    
    expected_c7 = {
        "case_description": "Study containing files from multiple patients, presenting a critical safety hazard.",
        "routing": {
            "detected_modality": "brain_mri",
            "selected_engine": "neuromind",
            "requires_review": True
        },
        "safety": {
            "abstained": True,
            "abstention_reason": "mixed_patient_ids",
            "detail": "Safety constraint violated: multiple Patient IDs detected in the same upload."
        },
        "audit_trail": {
            "events": ["case.uploaded", "modality.routed", "engine.rejected_preprocess"]
        }
    }
    (c7_dir / "expected_output.json").write_text(json.dumps(expected_c7, indent=2))

    # Case 8: Unsupported Modality
    print("Generating Case 8: Unsupported Modality...")
    c8_dir = DEMO_DIR / "8_unsupported_modality"
    c8_dir.mkdir(exist_ok=True)
    write_dicom_with_patient(c8_dir / "mammogram.dcm", "PATIENT_MAMM", modality="MG", body_part="BREAST")
    
    expected_c8 = {
        "case_description": "An acquisition of an unsupported modality (Mammography).",
        "routing": {
            "detected_modality": "mammography",
            "selected_engine": None,
            "supported": False,
            "reason": "identified as Mammography, but no analysis engine is registered for it."
        },
        "safety": {
            "abstained": True,
            "abstention_reason": "unsupported_modality"
        },
        "audit_trail": {
            "events": ["case.upload_rejected", "modality.rejected"]
        }
    }
    (c8_dir / "expected_output.json").write_text(json.dumps(expected_c8, indent=2))

    # Case 9: Corrupted Upload
    print("Generating Case 9: Corrupted Upload...")
    c9_dir = DEMO_DIR / "9_corrupted_upload"
    c9_dir.mkdir(exist_ok=True)
    (c9_dir / "corrupted.nii").write_bytes(b"THIS_IS_GARBAGE_BYTES_THAT_WILL_NOT_DECODE_SUCCESSFULLY_1234567890")
    
    expected_c9 = {
        "case_description": "A corrupted file upload containing undecodable bytes.",
        "routing": {
            "detected_modality": "unknown",
            "selected_engine": None,
            "reason": "file could not be decoded as an image"
        },
        "safety": {
            "abstained": True,
            "abstention_reason": "unreadable_image"
        },
        "audit_trail": {
            "events": ["case.upload_rejected", "modality.rejected"]
        }
    }
    (c9_dir / "expected_output.json").write_text(json.dumps(expected_c9, indent=2))

    print("[AURA Demo Data] Finished preparing all 9 demo cases.")

if __name__ == "__main__":
    main()
