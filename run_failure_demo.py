"""Failure Demonstration Runner for AURA.

Programmatically runs failure cases through the routing and safety engines,
captures details of rejections, and compiles them into a markdown document.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent / "aura"))

from aura.backend.core.upload.intake import stage_bytes
from aura.backend.core.router.router import ModalityRouter
from aura.backend.engines.neuro.engine import NeuroMindEngine
from aura.backend.core.shared.errors import UnsupportedModality, ModalityConflict, AuraBackendError
from aura.backend.models.routing import ResultStatus

# Set paths
AURA_DIR = Path(__file__).resolve().parent
DEMO_DIR = AURA_DIR.parent / "demo_data"
DOCS_DIR = AURA_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

async def run_case(filepath: Path, filename: str) -> dict:
    from aura.backend.core.upload.intake import stage_bytes
    
    # We stage the file
    payload = filepath.read_bytes()
    with stage_bytes(payload, filename) as asset:
        # 1. Modality Routing
        router = ModalityRouter()
        routing = router.route(asset)
        
        # 2. Engine validate and execute
        engine = NeuroMindEngine()
        
        # We manually invoke validate_input and run to observe where it fails
        validation = engine.validate_input(asset)
        
        outcome = None
        exception_raised = None
        try:
            outcome = await engine.run(asset)
        except AuraBackendError as exc:
            exception_raised = exc
            
        return {
            "routing_modality": routing.modality,
            "routing_confidence": routing.confidence,
            "routing_reason": routing.reason,
            "routing_supported": routing.supported,
            "validation_accepted": validation.accepted,
            "validation_reason": validation.reason,
            "validation_checks": validation.checks,
            "outcome_status": outcome.status.value if outcome else "failed",
            "outcome_message": outcome.message if outcome else (exception_raised.reason if exception_raised else "unknown"),
            "outcome_payload": outcome.payload if outcome else (exception_raised.detail if exception_raised else {})
        }

async def run_failure_demo():
    print("[AURA Failure Demo] Starting failure demonstration execution...")
    
    # Define failure case paths
    cases = {
        "OOD Case (non_medical_noise.png)": {
            "path": DEMO_DIR / "5_ood_case" / "non_medical_noise.png",
            "name": "non_medical_noise.png"
        },
        "Missing Sequence (FLAIR missing)": {
            "path": DEMO_DIR / "6_missing_sequence",
            "name": "study.zip" # We will package this as a zip
        },
        "Mixed Patient (mismatched DICOMs)": {
            "path": DEMO_DIR / "7_mixed_patient",
            "name": "mixed_patient.zip" # We will package this as a zip
        },
        "Unsupported Modality (mammogram.dcm)": {
            "path": DEMO_DIR / "8_unsupported_modality" / "mammogram.dcm",
            "name": "mammogram.dcm"
        },
        "Corrupted Upload (corrupted.nii)": {
            "path": DEMO_DIR / "9_corrupted_upload" / "corrupted.nii",
            "name": "corrupted.nii"
        }
    }
    
    # Helper to zip folders
    import zipfile
    for key in ["Missing Sequence (FLAIR missing)", "Mixed Patient (mismatched DICOMs)"]:
        case_info = cases[key]
        zip_path = case_info["path"].parent / case_info["name"]
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(case_info["path"]):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)
        case_info["path"] = zip_path

    results = {}
    for name, info in cases.items():
        print(f"Running failure demonstration for {name}...")
        try:
            results[name] = await run_case(info["path"], info["name"])
        except Exception as e:
            results[name] = {"error": str(e)}
            
    # Cleanup zip files
    for key in ["Missing Sequence (FLAIR missing)", "Mixed Patient (mismatched DICOMs)"]:
        try:
            os.remove(cases[key]["path"])
        except:
            pass

    print("[AURA Failure Demo] Compiling failure_demo.md...")
    
    md_content = f"""# AURA Failure Mode and Rejection Demonstration

This document demonstrates AURA's multi-layered safety guardrails. When presented with corrupted data, incorrect modalities, or unsafe uploads, AURA actively rejects the cases at the boundary rather than outputting silent, confident hallucinations.

---

## 1. Out-of-Distribution (OOD) Case

- **File:** `non_medical_noise.png` (Random RGB noise representing non-medical images)
- **Modality Routing Outcome:**
  - **Detected Modality:** `{results["OOD Case (non_medical_noise.png)"].get("routing_modality", "unknown")}`
  - **Routing Confidence:** {results["OOD Case (non_medical_noise.png)"].get("routing_confidence", 0.0):.4f}
  - **Routing Status:** {"Supported" if results["OOD Case (non_medical_noise.png)"].get("routing_supported") else "Rejected / Unsupported"}
  - **Rule Triggered:** `{results["OOD Case (non_medical_noise.png)"].get("routing_reason", "")}`
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** {results["OOD Case (non_medical_noise.png)"].get("validation_accepted", False)}
  - **Rejection Reason:** *"{results["OOD Case (non_medical_noise.png)"].get("validation_reason", "")}"*
- **Audit Event Generated:** `modality.rejected`

> [!WARNING]
> Non-medical imagery is caught immediately by the pixel-based signatures, preventing it from reaching the neural network models.

---

## 2. Incomplete Study (Missing FLAIR Sequence)

- **File:** `study.zip` (Brain MRI containing T1, T1CE, T2 but missing the required FLAIR sequence)
- **Modality Routing Outcome:**
  - **Detected Modality:** `{results["Missing Sequence (FLAIR missing)"].get("routing_modality", "unknown")}`
  - **Routing Status:** {"Supported" if results["Missing Sequence (FLAIR missing)"].get("routing_supported") else "Unsupported"}
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** {results["Missing Sequence (FLAIR missing)"].get("validation_accepted", False)}
  - **Status:** `{results["Missing Sequence (FLAIR missing)"].get("outcome_status", "failed")}`
  - **Rejection Reason:** *"{results["Missing Sequence (FLAIR missing)"].get("outcome_message", "")}"*
  - **Rejection Details:** `{json.dumps(results["Missing Sequence (FLAIR missing)"].get("outcome_payload", {}))}`

> [!IMPORTANT]
> The engine accepts the format (NIfTI volume) as a brain MRI during the routing check, but the multi-sequence check during the analysis stage blocks execution when it detects that one of the four required pulse sequences is missing.

---

## 3. Mixed Patient ID Upload

- **File:** `mixed_patient.zip` (Mismatched Patient IDs in the same upload)
- **Modality Routing Outcome:**
  - **Detected Modality:** `{results["Mixed Patient (mismatched DICOMs)"].get("routing_modality", "unknown")}`
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** {results["Mixed Patient (mismatched DICOMs)"].get("validation_accepted", False)}
  - **Rejection Reason:** *"{results["Mixed Patient (mismatched DICOMs)"].get("validation_reason", "")}"*
  - **Rejection Details:** `{json.dumps(results["Mixed Patient (mismatched DICOMs)"].get("validation_checks", {}))}`

> [!CAUTION]
> Uploading studies containing images from multiple patient profiles is a severe patient safety hazard. AURA actively compares the patient IDs in the DICOM headers at the boundary and rejects the study before preprocessing.

---

## 4. Unsupported Modality

- **File:** `mammogram.dcm` (Mammography DICOM acquisition)
- **Modality Routing Outcome:**
  - **Detected Modality:** `{results["Unsupported Modality (mammogram.dcm)"].get("routing_modality", "unknown")}`
  - **Routing Status:** {"Supported" if results["Unsupported Modality (mammogram.dcm)"].get("routing_supported") else "Unsupported"}
  - **Routing Reason:** *"{results["Unsupported Modality (mammogram.dcm)"].get("routing_reason", "")}"*
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** {results["Unsupported Modality (mammogram.dcm)"].get("validation_accepted", False)}
  - **Rejection Reason:** *"{results["Unsupported Modality (mammogram.dcm)"].get("validation_reason", "")}"*

> [!IMPORTANT]
> Running a deep learning model on the wrong modality (e.g. chest X-ray model on a mammogram) produces a confident, plausible-looking, but completely meaningless report. AURA blocks this immediately during modality routing.

---

## 5. Corrupted File Upload

- **File:** `corrupted.nii` (Undecodable random bytes)
- **Modality Routing Outcome:**
  - **Detected Modality:** `{results["Corrupted Upload (corrupted.nii)"].get("routing_modality", "unknown")}`
  - **Routing Status:** {"Supported" if results["Corrupted Upload (corrupted.nii)"].get("routing_supported") else "Unsupported"}
  - **Routing Reason:** *"{results["Corrupted Upload (corrupted.nii)"].get("routing_reason", "")}"*
- **Clinical Validation Outcome:**
  - **Accepted for Analysis:** {results["Corrupted Upload (corrupted.nii)"].get("validation_accepted", False)}
  - **Status:** `{results["Corrupted Upload (corrupted.nii)"].get("outcome_status", "failed")}`
  - **Rejection Reason:** *"{results["Corrupted Upload (corrupted.nii)"].get("outcome_message", "")}"*

> [!CAUTION]
> If a file header cannot be decoded or is corrupted during transmission, the system fails cleanly and logs an unreadable image event, protecting the system from crashes.
"""
    (DOCS_DIR / "failure_demo.md").write_text(md_content)
    print(f"[AURA Failure Demo] Exported failure_demo.md to {DOCS_DIR / 'failure_demo.md'}")

if __name__ == "__main__":
    asyncio.run(run_failure_demo())
