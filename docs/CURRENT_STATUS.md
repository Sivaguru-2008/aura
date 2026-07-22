# AURA — Current Capabilities & Build Status

This document captures the implementation status of AURA's components, differentiating between built production logic, simulation/feature fallbacks, and designed-only features.

---

## 1. Subsystem Snapshot

| Component | Status | Description | Implementation File |
|---|---|---|---|
| **API Gateway** | ✅ Built | FastAPI endpoints for case retrieval, feedback submission, study upload, model registration, and admin metrics. | [`gateway/app.py`](file:///e:/AURA/aura-main/aura/gateway/app.py) |
| **Pipeline Orchestrator** | ✅ Built | Process-wide async event bus execution tracing through all nine stages. | [`gateway/pipeline.py`](file:///e:/AURA/aura-main/aura/gateway/pipeline.py) |
| **Vision Engine** | ✅ Built | Production-grade DenseNet-121 (trained on MIMIC-CXR) with per-finding Platt calibration. Falls back to a heuristic feature model if configuration overrides are set. | [`services/vision/engine.py`](file:///e:/AURA/aura-main/aura/services/vision/engine.py) |
| **Evidence Encoder** | ✅ Built | Maps 7 vision findings + clinical priors into a unified $x \in [0, 1]^8$ vector. | [`services/fusion/evidence.py`](file:///e:/AURA/aura-main/aura/services/fusion/evidence.py) |
| **Quantum Fusion (Q1)** | ✅ Built | 8-qubit variational circuit (PennyLane) angle-encoding evidence, outputting a 6-diagnosis posterior. | [`services/fusion/quantum.py`](file:///e:/AURA/aura-main/aura/services/fusion/quantum.py) |
| **Classical Fusion** | ✅ Built | Bayesian Product-of-Experts serving as the primary baseline and fallback. | [`services/fusion/classical.py`](file:///e:/AURA/aura-main/aura/services/fusion/classical.py) |
| **Conflict Guard** | ✅ Built | Wasserstein distance monitor that switches the serving backend to classical if VQC and PoE diverge. | [`services/fusion/conflict.py`](file:///e:/AURA/aura-main/aura/services/fusion/conflict.py) |
| **Clinical Reasoning** | ✅ Built | Nudges posterior probabilities using likelihood ratios matching clinical guidelines. | [`services/reasoning/engine.py`](file:///e:/AURA/aura-main/aura/services/reasoning/engine.py) |
| **Safety & Conformal** | ✅ Built | Performs Platt/temperature scaling, Mondrian (class-conditional) conformal set generation, and checks a 4-reason abstention policy. | [`services/safety/engine.py`](file:///e:/AURA/aura-main/aura/services/safety/engine.py) |
| **OOD Detector** | ✅ Built | Computes energy z-score against calibrated in-distribution statistics. | [`services/safety/uncertainty.py`](file:///e:/AURA/aura-main/aura/services/safety/uncertainty.py) |
| **Adaptive Conformal (ACI)**| ✅ Built | Updates the conformal threshold based on signed clinician feedback and updates SQLite state. | [`gateway/storage.py`](file:///e:/AURA/aura-main/aura/gateway/storage.py) |
| **Explainability** | ✅ Built | Grad-CAM++ regional heatmaps, leave-one-out feature attributions, and counterfactual predictions. | [`services/explain/engine.py`](file:///e:/AURA/aura-main/aura/services/explain/engine.py) |
| **Recommender** | ✅ Built | Greedy Expected Information Gain (EIG) evaluation of candidates (CT, labs, prior). | [`services/recommend/engine.py`](file:///e:/AURA/aura-main/aura/services/recommend/engine.py) |
| **Grounded Report** | ✅ Built | Generates report sections mapped directly to supporting evidence nodes. | [`services/report/engine.py`](file:///e:/AURA/aura-main/aura/services/report/engine.py) |
| **Memory Index** | 🟡 Partial | In-memory cosine similarity index on DenseNet embeddings (not persisted across runs). | [`services/memory/engine.py`](file:///e:/AURA/aura-main/aura/services/memory/engine.py) |
| **Model Registry** | 🟡 Partial | Simple version listing and JSON-based calibration/metrics catalog. | [`services/models/registry.py`](file:///e:/AURA/aura-main/aura/services/models/registry.py) |
| **Doctor Dashboard** | ✅ Built | Zero-dependency SPA client console rendered directly from the gateway static files. | [`apps/web/`](file:///e:/AURA/aura-main/aura/apps/web) |
| **Audit Ledger** | ✅ Built | SQLite append-only audit trail logging feedback, case histories, and conformal updates. | [`gateway/storage.py`](file:///e:/AURA/aura-main/aura/gateway/storage.py) |
| **Test Suite** | ✅ Built | 135+ tests validating contracts, inference pathways, calibration math, and security gates. | [`tests/`](file:///e:/AURA/aura-main/aura/tests) |
| **Quantum Stack Q2-Q6** | ⚪ Designed | Mix-state belief updates, similarity kernels, and QAOA optimization. Described in architecture but not coded. | [`docs/ARCHITECTURE.md`](file:///e:/AURA/aura-main/docs/ARCHITECTURE.md) |

*Legend: ✅ Built (running in production paths) · 🟡 Partial (functional but simplified) · ⚪ Designed (docs only, no code).*

---

## 2. Genuinely Built Capabilities

* **End-to-End Image Upload**: Radiographs (PNG/JPG/DICOM) uploaded to `/v1/studies/upload` undergo validation and run through the full 9-stage analysis pipeline.
* **Calibrated Operating Points**: Vision model thresholding is driven by Platt calibration parameters fitted on $n=2,099$ held-out MIMIC cases rather than arbitrary cutoffs.
* **Tight Safety Loops**: The conflict guard, conformal sets, and clinical reasoning are fully wired together. The final diagnosis shown to the clinician cannot contradict an upstream safety override.
* **Clinician Feedback updates ACI**: Confirming case outcomes on the dashboard writes back to SQLite, causing the marginal conformal threshold to dynamically self-correct under distribution shift.
