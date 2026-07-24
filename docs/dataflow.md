# AURA Data Flow Map

This document describes the step-by-step lifecycle of studies uploaded to AURA, mapping out inputs, processing stages, boundaries, and outputs.

---

## 1. Step-by-Step Data Flow Diagram

```
[ Upload File ]
       │
       ▼
1. INTAKE & STORAGE BOUNDARY (gateway.security / backend.core.upload)
  ├── Enforce type allowlist (DICOM, PNG, JPG, JPEG, TIFF, BMP, NIfTI, ZIP)
  ├── Stream payload with a hard size cap (max_upload_mb) to prevent memory exhaustion
  ├── Write binary to temporary path (aura-intake-*.bin/nii)
  ├── Compute SHA-256 content hash (retained as unique case ID/audit key)
  └── Best-effort extraction of DICOM header tags (Modality, PatientID, StudyInstanceUID)
       │
       ▼
2. INTELLIGENT MODALITY ROUTER (backend.core.router)
  ├── Extract ImageFingerprint (DICOM headers + tone/aspect-ratio pixel features)
  ├── Run Modality Signatures (ChestRadiographSignature, BrainMRISignature, HeadCTSignature)
  ├── Compare highest score against commit threshold (0.55) and margin (0.10)
  ├── Cross-check with client-declared modality:
  │     ├── Agree/Silence -> Route proceeds
  │     └── Contradict -> Raise ModalityConflict (422) and audit `modality.rejected`
  └── Select appropriate engine from EngineRegistry
       │
       ▼
3. ENGINES INTEGRATION LAYER (backend.engines)
  ├── Dispatches ImageAsset to the selected engine adapter (Thorax or NeuroMind)
  │
  ├── [A] NeuroMind Adapter (Brain MRI)
  │     ├── Volumetric Series Discovery: verifies if folder/ZIP contains valid files
  │     ├── Pre-processing (MRI Foundation): canonical orientation, voxel resampling to 1.0mm,
  │     │   intensity normalisation (percentile robust clip/zscore)
  │     ├── Neural Network Inference: multi-task ResU-Net segments tumor zones, computes
  │     │   tumour presence probability, size estimates, and artifact quality scores
  │     ├── Platt Calibration: scales raw tumor presence probability
  │     └── Safety checks: validates sequence completeness (blocks if FLAIR/T1/T1CE/T2 is missing)
  │
  └── [B] Thorax Adapter (Chest X-Ray)
        ├── Resize image to 224x224 grid (matching training resolution)
        ├── Neural Network Inference: DenseNet-121 computes 7-finding probabilities + embedding
        ├── Platt Calibration: maps output probabilities using calibration curve
        ├── Evidence Fusion: fuses chest findings with priors (quantum VQC or classical PoE)
        └── Clinical Reasoner: adjusts posteriors based on guidelines and priors (Guideline LRs)
             │
             ▼
4. SAFETY & CALIBRATION ENGINE (services.safety)
  ├── Temperature-scale logits to fit the model's calibration split (restores calibration honesty)
  ├── Compute conformal prediction sets (marginal or class-conditional Mondrian sets)
  ├── Compute Out-of-Distribution (OOD) energy score and flags OOD if z-score > 3.0
  └── Epistemic Uncertainty Decomposition: splits predictive entropy into aleatoric and epistemic
       │
       ▼
5. CLINICAL REPORT & EXPLANATION GENERATION (services.report / services.explain)
  ├── Explanation: computes occlusion saliency maps highlighting decision areas
  ├── Report Compose: generates report sections (findings, impression, differential, recommendations)
  ├── Recommendations: identifies missing multimodal evidence (e.g. labs/scans) to resolve entropy
  └── Memory indexing: indexes cases in the database (local SQLite audit and vector similarity memory)
       │
       ▼
6. PERSISTENCE & HISTORY PORTAL (gateway.storage)
  ├── Save structured CaseBundle in the SQLite database (aura.db)
  └── Write audit log row: logs action, actor, timestamp, and metadata (no patient identifiers)
```

---

## 2. Audits, Events, and Monitoring

Every upload and state transition is recorded in the SQLite audit log. Key audit events include:

| Event Name | Source Component | Triggers when... | Logged Fields |
| :--- | :--- | :--- | :--- |
| **`case.upload_rejected`** | Intake | File size exceeds cap or type is not on the allowlist | `filename`, `reason` |
| **`modality.rejected`** | Modality Router | Input modality does not match serving capability or contradicts declared modality | `reason`, `detected`, `declared`, `confidence` |
| **`case.uploaded`** | API Gateway | Study successfully routed and processed | `case_id`, `top_diagnosis`, `abstained`, `inference_time_s` |
| **`feedback.recorded`** | SPA Feedback | Clinician reviews and accepts/corrects the case | `case_id`, `verdict`, `correction`, `confirmed_diagnosis` |
| **`conformal.updated`** | ACI Service | Feedback confirmed outcome is recorded; updates the online ACI threshold | `case_id`, `qhat`, `covered`, `localized_coverage` |
| **`report.signed`** | SPA Signoff | Clinician signs off the report | `case_id`, `signed_by` |
