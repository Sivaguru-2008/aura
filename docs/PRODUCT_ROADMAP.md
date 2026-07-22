# AURA — Product Vision & Strategic Roadmap

This document outlines the product strategy, clinical applicability, and long-term roadmap for AURA (Adaptive Uncertainty-aware Reasoning Assistant) as it evolves from a single-study clinical copilot into a comprehensive **Clinical reasoning Operating System**.

---

## 1. The Core Strategic Problem: loop Closure

Most medical AI solutions on the market today focus on image classification: *"What disease is in this image?"*
While perception AI has matured, it has inadvertently exacerbated a massive clinical safety bottleneck: **loop opening**.

A chest classifier flags a nodule at minute zero of a diagnostic process. That process may run for three weeks to three years, crossing multiple clinicians, handoffs, and IT systems. Every detected finding is an open loop — a follow-up to schedule, a test to order, a differential to resolve. In US clinical care, **30–70% of imaging follow-up recommendations are never completed**, resulting in diagnostic errors, delayed cancer diagnoses, and significant malpractice liability.

```
Classifier detects finding ──► Open loop (recommend follow-up)
                                   │
                                   ├──► [30–70% LOST] ──► Delayed diagnosis & malpractice
                                   │
                                   └──► [AURA intervention] ──► commitment ledger & tracking
```

AURA is designed as the **system of record for clinical reasoning**. While EHRs record billing/actions and PACS record images, AURA records what the care team *thinks*: the differential, the calibrated confidence, what evidence is missing, and who owes the next action.

---

## 2. Product Architecture Evolution

AURA's product evolution moves from single-study evaluations to longitudinal patient tracking:

### Phase 0: Single-Study Epistemic Copilot (Current Baseline)
* Calibrated confidence, epistemic/aleatoric uncertainty split, and conformal prediction sets.
* Explicit abstention policy ("I don't know") with explanations and next-best-test recommendations.

### Phase 1: Longitudinal Diagnostic Loops (V2 OS)
* **Diagnostic Loop**: A managed process for a specific clinical question (e.g., "Is this lung nodule malignant?"), tracked across encounters.
* **Temporal Belief State**: The diagnostic posterior is treated as a filter updated as evidence arrives, maintaining historical context.
* **Commitment Ledger**: Tracks accepted next-test recommendations as owned obligations with deadlines, preventing critical recommendations from being lost.
* **Air Traffic Control Console**: A hospital-wide dashboard displaying all active diagnostic loops, sorted by risk of silent failure.

---

## 3. strategic Roadmap & Milestones

The roadmap leverages clinical validation and on-premise security to build a defensible product moat:

```
                  ┌──────────────────────┐
                  │ PHASE 0: HACKATHON   │
                  │ current baseline,    │
                  │ simulator validated  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ PHASE 1: PILOT       │
                  │ clinical pilots,     │
                  │ prior timelines,     │
                  │ PACS integration     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ PHASE 2: ENTERPRISE  │
                  │ multimodal fusion,   │
                  │ FDA SaMD validation, │
                  │ billing integration  │
                  └──────────────────────┘
```

### Phase 0: Hackathon & Simulator Validation (Current State)
* End-to-end P0 pipeline running offline.
* Real MIMIC-CXR-trained DenseNet-121 backbone with Platt serving calibration.
* Integrated variational quantum fusion benchmarked against classical product-of-experts.

### Phase 1: Clinical Pilots & Prior Comparison (3–6 Months)
* **Longitudinal Memory**: Enable prior image retrieval and delta overlays to monitor changes (e.g. pneumothorax resolution or nodule growth).
* **PACS / RIS Integration**: DICOM C-STORE routing and HL7 FHIR `DiagnosticReport` exports.
* **Academic Research Pilots**: Deploy to research sites to compile clinical-utility and time-saving metrics.

### Phase 2: Enterprise Scaling & Multimodal Fusion (6–18 Months)
* **Multimodal Fusion**: Incorporate laboratory values (BNP, troponin) and patient symptoms directly into the learned fusion circuit.
* **FDA SaMD Clearance**: Pursue FDA 510(k) clearance under CADe/CADx classification categories.
* **Malpractice Carrier Integration**: Partner with insurance providers to offer premium discounts to hospitals using AURA's commitment ledger.
