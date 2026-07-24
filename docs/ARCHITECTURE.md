# AURA System Architecture and Design Diagrams

This document details the software architecture, pipeline flow, and safety mechanisms of the AURA Clinical Intelligence Copilot.

---

## 1. Overall System Architecture

```mermaid
graph TD
    Client[Client / Web SPA] -->|POST /analyze| Gateway[FastAPI Gateway]
    Client -->|HTTP GET/POST| Auth[OIDC/RBAC Gateway Security]
    
    Gateway -->|1. Stream & Hash| Intake[Upload Intake]
    Intake -->|ImageAsset| Router[Intelligent Modality Router]
    
    Router -->|Support Check| Registry[Engine Registry]
    Router -->|Select Engine| Dispatch[Dispatch Service]
    
    subgraph Thorax Engine (Chest X-Ray)
        Dispatch -->|Route: chest_xray| Thorax[Thorax Engine Adapter]
        Thorax -->|224x224 Grid| DenseNet[DenseNet-121 Feature Extractor]
        DenseNet -->|Logits| tempScaleC[Temperature Scaling]
        tempScaleC -->|Calibrated Logits| Fusion[Evidence Fusion Backend]
        Fusion -->|Adjusted Posterior| Reasoner[Clinical Reasoner]
        Reasoner -->|multimodal Posterior| SafetyC[Safety Engine: Conformal + OOD]
        SafetyC -->|Assess| Explain[Saliency Explainer]
        Explain -->|Grounded| ReportC[Grounded Report Engine]
    end
    
    subgraph NeuroMind Engine (Brain MRI)
        Dispatch -->|Route: brain_mri| NeuroMind[NeuroMind Engine Adapter]
        NeuroMind -->|Volumetric Series| Found[MRI Foundation Pipeline]
        Found -->|Resample + Normalise| ResUNet[Multi-Task ResU-Net]
        ResUNet -->|Segmentation Mask| SliceProj[2D Visual Slice Projector]
        ResUNet -->|Tumour Presence| tempScaleB[Platt Calibration]
        ResUNet -->|Quality & Size| SafetyB[Neuro Safety Engine]
        tempScaleB -->|Calibrated Prob| SafetyB
        SafetyB -->|Report Grounding| ReportB[Brain Report Engine]
    end
    
    ReportC -->|Save Bundle| DB[(SQLite Store & Conformal Log)]
    ReportB -->|Save Bundle| DB
    ReportC -->|Embeddings| Mem[(Memory Similar Index)]
    
    DB -->|Audit Rows| Admin[Admin Monitoring Dashboard]
```

---

## 2. Modality Routing and Decision Flow

```mermaid
flowchart TD
    Start[Upload Staged] --> ParseHeader{DICOM Header Present?}
    
    ParseHeader -->|Yes| DicomModality{Modality MR/CT/US/MG?}
    DicomModality -->|MR / CT / US / MG| MapModality[Map Modality & Extract Keywords]
    DicomModality -->|Other / Chest| RunGate[Run Chest X-Ray Intake Gate]
    
    ParseHeader -->|No| CheckPixel{Grayscale & High Tonal Entropy?}
    CheckPixel -->|Yes| HeadGeom{Axial Head Geometry Score > 0.85?}
    CheckPixel -->|No| RunGate
    
    HeadGeom -->|Yes| MatchMR[Match Brain MRI - Capped 0.91]
    HeadGeom -->|No| RunGate
    
    RunGate -->|CXR Confirmed| MatchCXR[Match Chest X-Ray - 0.88]
    RunGate -->|Unconfirmed| RejectImage[Unrecognised Image - Reject]
    
    MapModality --> MatchMapped[Match via Header Keyword - 0.97]
    
    MatchMR --> Resolve[Modality Router Decision]
    MatchCXR --> Resolve
    MatchMapped --> Resolve
    RejectImage --> Resolve
    
    Resolve --> CheckDeclaration{Client Declared Modality?}
    CheckDeclaration -->|No| FinalRoute[Resolve Engine from Registry]
    CheckDeclaration -->|Yes| Conflict{Declaration Contradicts Detector?}
    Conflict -->|Yes| Refuse[Refuse / ModalityConflict Error]
    Conflict -->|No| FinalRoute
```

---

## 3. Thorax Diagnostic Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant SPA as Web Dashboard
    participant API as FastAPI Gateway
    participant Router as Modality Router
    participant Vision as DenseNet-121 Vision Engine
    participant Fusion as Evidence Fusion Engine
    participant Safety as Safety Engine
    participant Reasoner as Clinical Reasoner
    participant DB as SQLite Audit Trail
    
    SPA->>API: POST /v1/studies/analyze (File + Priors)
    API->>Router: route(asset)
    Note over Router: Identifies modality, validates against client statement
    Router-->>API: RoutingMetadata (chest_xray)
    API->>Vision: analyze(img)
    Note over Vision: Computes 7-finding probabilities + anatomical embeddings
    Vision-->>API: VisionResult
    API->>Fusion: fuse_vector(evidence)
    Note over Fusion: Computes diagnostic logits from quantum/classical model
    Fusion-->>API: Logits
    API->>Reasoner: reason(posterior, labs/symptoms)
    Note over Reasoner: Fuses imaging posterior with clinical context using Guideline LRs
    Reasoner-->>API: Adjusted Posterior
    API->>Safety: assess(final_posterior, online_qhat)
    Note over Safety: Runs conformal prediction + energy OOD z-score + deep ensemble MI
    Safety-->>API: SafetyAssessment
    API->>DB: save_case(CaseBundle)
    API-->>SPA: CaseBundle (report, conformal_set, saliency_overlay)
```

---

## 4. NeuroMind Diagnostic Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant SPA as Web Dashboard
    participant API as FastAPI Gateway
    participant Router as Modality Router
    participant Found as MRI Foundation Pipeline
    participant ResUNet as Multi-Task ResU-Net
    participant Safety as Neuro Safety Engine
    participant DB as SQLite Audit Trail
    
    SPA->>API: POST /v1/studies/analyze (NIfTI / ZIP)
    API->>Router: route(asset)
    Note over Router: Verifies NIfTI/NRRD format, bypasses CXR gate
    Router-->>API: RoutingMetadata (brain_mri)
    API->>Found: run(study_path)
    Note over Found: Voxel resampling, orientation canonical, intensity normalisation
    Found-->>API: FoundationStudy
    API->>ResUNet: analyze(FoundationStudy)
    Note over ResUNet: Segment tumor regions (WT, TC, ET) + compute presence + size + quality
    ResUNet-->>API: NeuroVisionOutput
    API->>Safety: assess(presence_raw, quality_score, sequence_mapping)
    Note over Safety: Platt calibration, checks sequence completeness, flags motion artifacts
    Safety-->>API: NeuroSafetyAssessment
    API->>DB: save_case(BrainCaseBundle)
    API-->>SPA: BrainCaseBundle (segmentation_overlay, volumes_table, report)
```

---

## 5. Safety Engine Guardrails

```mermaid
graph TD
    Logits[Raw Diagnostic Logits] --> temp[Temperature Scaling]
    temp --> Calibrated[Calibrated Probabilities]
    
    subgraph Conformal Prediction
        Calibrated --> marginal{ACI qhat Available?}
        marginal -->|Yes| ACI[Set conformal threshold to ACI qhat]
        marginal -->|No| Static[Set conformal threshold to split-conformal qhat]
        ACI --> ConformalSet[Compute Conformal Prediction Set]
        Static --> ConformalSet
    end
    
    subgraph Out-of-Distribution Safety
        Logits --> Energy[Compute Energy Score]
        Energy --> ZScore[Compute Z-Score against In-Distribution Mean/Std]
        ZScore --> Threshold{Z-Score > Threshold?}
        Threshold -->|Yes| OOD[Flag OOD & Abstain]
        Threshold -->|No| Clean[Clear OOD Check]
    end
    
    subgraph Epistemic Uncertainty
        Logits --> Ensemble{Deep Ensemble Present?}
        Ensemble -->|Yes| MI[mutual Information between members]
        Ensemble -->|No| Perturb[Perturb input evidence & measure posterior variance]
        MI --> Epistemic[Quantify Aleatoric vs Epistemic Uncertainty]
        Perturb --> Epistemic
    end
    
    ConformalSet --> Assessment[Assemble Safety Assessment]
    OOD --> Assessment
    Epistemic --> Assessment
```

---

## 6. Longitudinal Progression and Tracking

```mermaid
flowchart LR
    Prev[Previous MRI Case] --> Reg[Spatial Registration]
    Curr[Current MRI Case] --> Reg
    
    Reg --> Align[Co-register previous & current volumes]
    Align --> Segment[Extract tumor segmentations WT, TC, ET]
    
    Segment --> VolumeCalc[Calculate volume in mm3 per region]
    VolumeCalc --> Diff[Calculate absolute & percentage difference]
    
    Diff --> Grow{Growth > 10%?}
    Diff --> Regress{Regression > 10%?}
    
    Grow -->|Yes| Progress[Flag Tumor Progression]
    Regress -->|Yes| Response[Flag Treatment Response / Regression]
    
    Grow -->|No| Stable[Flag Stable Disease]
    Regress -->|No| Stable
    
    Progress --> Report[Generate Longitudinal Progress Report & Timeline]
    Response --> Report
    Stable --> Report
```
