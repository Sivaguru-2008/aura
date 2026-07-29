"""AURA Demo Launcher.

Loads AURA models, runs the 5 prepared cases through the full pipeline using
the production DispatchService, generates report/overlay files, and exits.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent / "aura"))

from aura.backend.core.upload.intake import stage_bytes
from aura.backend.services.dispatch import DispatchService
from aura.backend.models.routing import ResultStatus
from aura.backend.core.shared.errors import UnsupportedModality, ModalityConflict

# Set paths
AURA_DIR = Path(__file__).resolve().parent
DEMO_DATA_DIR = AURA_DIR.parent / "demo_data"
RESULTS_DIR = AURA_DIR / "demo_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Bootstrap production engines and dispatch service
# --------------------------------------------------------------------------- #
from aura.backend.bootstrap import install_router
from aura.gateway.pipeline import Pipeline
from aura.gateway.storage import Store
from aura.common.config import DB_PATH

class MockApp:
    def include_router(self, router):
        pass

print("[AURA Demo] Initializing production dispatch service and loading models...")
store = Store(DB_PATH)
pipeline = Pipeline(store=store)
dispatch_service = install_router(
    app=MockApp(),
    pipeline=pipeline,
    store=store
)

# --------------------------------------------------------------------------- #
# Helper to zip a folder for NeuroMind ingestion
# --------------------------------------------------------------------------- #
def package_zip(folder_path: Path) -> Path:
    zip_path = folder_path.parent / (folder_path.name + ".zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
    return zip_path

async def run_demo_case(case_name: str, folder_name: str, declared_modality: str | None, is_zipped: bool = True):
    print(f"\n--- Running Demo Case: {case_name} ---")
    case_path = DEMO_DATA_DIR / folder_name
    
    if not case_path.exists():
        print(f"Error: Demo data path {case_path} does not exist. Run prepare_demo_data.py first.")
        return
        
    temp_zip = None
    if is_zipped:
        temp_zip = package_zip(case_path)
        payload = temp_zip.read_bytes()
        filename = temp_zip.name
    else:
        # A single file upload (e.g. OOD png)
        filepath = next(case_path.glob("*.png"))
        payload = filepath.read_bytes()
        filename = filepath.name

    try:
        # Stage bytes in the AURA intake system
        with stage_bytes(payload, filename) as asset:
            print(f"Dispatching with declared modality: {declared_modality}")
            
            try:
                # Run the entire pipeline (inspect, route, calibrate, analyze, safety, etc.)
                envelope = await dispatch_service.dispatch(asset, declared_modality=declared_modality)
                
                routing = envelope.routing
                outcome = envelope.result
                
                print(f"Routed Modality: {routing.modality} (Confidence: {routing.confidence:.4f})")
                print(f"Pipeline Outcome: {outcome.status.value}")
                
                if outcome.status == ResultStatus.COMPLETED:
                    print("Inference completed successfully.")
                elif outcome.status == ResultStatus.UNSUPPORTED:
                    print(f"Refusal Triggered: {outcome.message}")
                else:
                    print(f"Analysis Failed/Abstained: {outcome.message}")
                    
                # Save results
                save_case_results(folder_name, outcome.payload, outcome.payload.get("segmentation_volume"))
                
            except (UnsupportedModality, ModalityConflict) as exc:
                print(f"Safety Rejection Triggered: {exc.reason}")
                actual = {
                    "routing": {
                        "modality": "unknown",
                        "supported": False,
                        "reason": exc.reason
                    },
                    "safety": {"abstained": True, "reason": exc.reason}
                }
                save_case_results(folder_name, actual, None)
            
    finally:
        # Cleanup temp zip file
        if temp_zip and temp_zip.exists():
            try:
                os.remove(temp_zip)
            except:
                pass

def save_case_results(folder_name: str, payload: dict, seg_volume: np.ndarray | None):
    case_res_dir = RESULTS_DIR / folder_name
    case_res_dir.mkdir(exist_ok=True)
    
    # Save Report
    report = payload.get("report", {})
    if report:
        report_md = f"""# AURA Diagnostic Report - {folder_name.upper()}

## Findings
{report.get("findings_text", "No clinical abnormalities detected.")}

## Impression
{report.get("impression_text", "Normal study.")}

## Differential Diagnosis
{report.get("differential_text", "None.")}

## Safety Status
- **Abstained:** {payload.get("safety", {}).get("abstained", False)}
- **Reason:** {payload.get("safety", {}).get("abstention_reason", "N/A")}
- **Conformal Set:** {payload.get("safety", {}).get("conformal_set", [])}
"""
        (case_res_dir / "report.md").write_text(report_md)
        print(f"Saved report.md to {case_res_dir}")
        
    # Save raw JSON payload (excluding large segmentation volumes)
    json_payload = {k: v for k, v in payload.items() if k != "segmentation_volume"}
    (case_res_dir / "actual_output.json").write_text(json.dumps(json_payload, indent=2, default=str))
    
    # Save Segmentation Mask if present
    if seg_volume is not None:
        # Save WT mask as a 3D NIfTI file
        wt_mask = (seg_volume > 0).astype(np.float32)
        write_nifti_helper(case_res_dir / "tumor_segmentation.nii", wt_mask)
        print(f"Saved tumor_segmentation.nii mask to {case_res_dir}")

def write_nifti_helper(path: Path, array: np.ndarray):
    header = bytearray(348)
    header[0:4] = (348).to_bytes(4, byteorder='little')
    header[38:39] = b'r'
    header[40:42] = (array.ndim).to_bytes(2, byteorder='little')
    for i in range(array.ndim):
        idx = 42 + 2 * i
        header[idx:idx+2] = (array.shape[i]).to_bytes(2, byteorder='little')
    header[70:72] = (16).to_bytes(2, byteorder='little')
    header[72:74] = (32).to_bytes(2, byteorder='little')
    header[108:112] = int(352).to_bytes(4, byteorder='little')
    header[344:347] = b'n+1'
    
    payload = bytes(header) + b"\x00" * 4 + np.asfortranarray(array.astype(np.float32)).tobytes(order="F")
    path.write_bytes(payload)

async def main():
    print("==============================================================")
    print("AURA DEMO LAUNCHER: RUNNING PREPARED CLINICAL CASES")
    print("==============================================================")
    
    # Run the 5 prepared cases
    await run_demo_case("Normal Brain MRI", "1_normal_brain_mri", declared_modality="brain_mri", is_zipped=True)
    await run_demo_case("Small Tumor Brain MRI", "2_small_tumor_brain_mri", declared_modality="brain_mri", is_zipped=True)
    await run_demo_case("Large Tumor Brain MRI", "3_large_tumor_brain_mri", declared_modality="brain_mri", is_zipped=True)
    await run_demo_case("High Edema Brain MRI", "4_high_edema_brain_mri", declared_modality="brain_mri", is_zipped=True)
    await run_demo_case("Out of Distribution (OOD) Anomaly", "5_ood_case", declared_modality=None, is_zipped=False)
    
    print("\n==============================================================")
    print(f"DEMO MODE EXECUTION COMPLETED. RESULTS EXPORTED TO {RESULTS_DIR}")
    print("==============================================================")

if __name__ == "__main__":
    asyncio.run(main())
