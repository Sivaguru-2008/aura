# AURA — System & Product Architecture

This document describes the Product & System Architecture for **AURA** (*Adaptive Uncertainty-aware Reasoning Assistant*). It outlines the modular microservice layout, the 9-stage clinical inference pipeline, the event bus orchestration, and the integrated Classical-Quantum Stack.

---

## 1. Product Vision: Epistemic Clinical Intelligence

Every diagnostic AI on the market today answers one question: *"What disease is in this image?"*
AURA is designed to answer a different, safety-critical question: **"What is known, how confidently is it known, what evidence is missing, and what should be done next?"**

AURA creates a new category: **Clinical Epistemic Intelligence** — software whose core job is to model clinical uncertainty, manage incomplete evidence, and provide an auditable reasoning path. The core primitive is the **Evidence Graph**, where every case is a graph of evidence nodes (imaging findings, prior studies, structured priors, clinician annotations) carrying calibrated probabilities and uncertainty bounds.

---

## 2. System Topology & Event-Driven Pipeline

AURA's operations run as an event-driven analysis pipeline. The FastAPI gateway orchestrates the execution of independent service engines over an async event bus:

```
study.received ──► VisionEngine ──► evidence.encode ──► FusionEngine
                        │                                    │
                  (7 findings)                      (Quantum/Classical)
                        │                                    │
                        ▼                                    ▼
                ClinicalReasoner ◄──────────────────── resolved_logits
                        │
                  (Adjusted dx)
                        │
                        ▼
                  SafetyEngine ◄─── aci_qhat (conformal, OOD, abstention)
                        │
                        ├──► ExplainEngine (Grad-CAM++ / attribution)
                        ├──► RecommendEngine (greedy EVOI next-best-test)
                        └──► ReportEngine ──► MemoryEngine (longitudinal index)
```

### The 9-Stage Runtime Pipeline

1. **Intake & Gate**: The study (image file or DICOM) is loaded, checked for format validity, normalized, and resized to $224 \times 224$ (using area-averaging).
2. **Vision Engine**: Runs a fine-tuned DenseNet-121 model on the chest radiograph to yield 7 independent finding probabilities and a 1024-dimensional feature embedding.
3. **Evidence Encoding**: Compresses findings + structured clinical priors (age, smoking status, etc.) into an 8-channel vector $x \in [0, 1]^8$.
4. **Evidence Fusion**: Fuses the evidence vector into a joint diagnostic posterior over 6 diagnoses. Supports:
   - **Quantum Fusion (Q1)**: Parameterized 8-qubit variational circuit.
   - **Classical Fusion**: Bayesian product-of-experts (the baseline).
   - **Wasserstein Conflict Guard**: Tie-breaker that defers to classical if the quantum and classical posteriors diverge beyond a threshold, resolving contradiction risks.
5. **Clinical Reasoning**: Intercepts the fusion logits and incorporates multimodal laboratory/symptom history via guideline likelihood ratios.
6. **Safety & Calibration**: Applies per-finding Platt/temperature scaling, builds a Mondrian class-conditional conformal prediction set, evaluates OOD energy score, and checks the 4-reason abstention policy (Low Confidence, Large Conformal Set, High Epistemic Uncertainty, Out of Distribution).
7. **Explainability**: Computes Grad-CAM++ heatmaps for regional localization, leave-one-out feature attribution, and counterfactuals.
8. **Next-Step Recommendation**: Ranks diagnostic actions (e.g. CT, lateral film, blood tests) by Expected Information Gain (EIG) per unit cost/risk.
9. **Report Generation & Memory**: Prepares a structured findings report where every sentence is grounded in evidence, indexes the study embedding in the similarity memory, and logs clinician feedback to the SQLite audit database.

---

## 3. The Classical-Quantum Stack

AURA separates classical perception (computer vision) from quantum-native reasoning. We apply quantum computing strictly to the low-dimensional, correlation-rich clinical evidence vector ($8\text{--}16$ features).

### The Honesty Contract & Verification Tiers

To ensure scientific integrity, every quantum capability is labeled under one of three tiers:
* **[A] Measured**: Runs today on a simulator; compared head-to-head against a classical twin on real data via the benchmark harness.
* **[B] Grounded**: Implemented on a simulator; quantum benefit is structural or representational (e.g. density-matrix coherences representing joint uncertainty), backed by peer-reviewed literature.
* **[C] Vision**: Requires fault-tolerant quantum hardware (FTQC); represents future roadmap capabilities.

### Quantum Services (Q0–Q6)

* **Q0 — Quantum Feature-Map Registry [A]**: Governs the evidence-to-state mapping (`RY` rotation). Supports feedback-driven angle training to align the feature representation.
* **Q1 — Quantum Evidence Fusion [A - Active]**: models $k$-th order interactions between evidence channels using a variational quantum classifier (VQC) with CNOT-ring entanglement. It outputs a 6-diagnosis posterior and finite-shot uncertainty.
* **Q2 — Quantum Belief Engine [B]**: Represents belief states as density matrices $\rho$ on mixed-state simulators to track correlated ambiguity (off-diagonal coherences) and analyze clinician anchoring risk via order-sensitivity re-evaluation.
* **Q3 — Quantum Similarity Kernels [B]**: Re-ranks historical cases using state-fidelity kernels for rare disease matching.
* **Q4 — QAOA Diagnostic Planner [C]**: Resolves next-test panels as a Quadratic Unconstrained Binary Optimization (QUBO) problem using QAOA.
* **Q5 — Quantum Trajectory Engine [B]**: Maps multi-state longitudinal progression models as unitary evolutions over time.
* **Q6 — Quantum Uncertainty Service [A]**: Optimizes shot budgeting and Born sampling over diagnostic outcomes.

---

## 4. Folder Structure (Monorepo)

The repository layout is organized to support microservice isolation and clean integration boundaries:

```
aura-main/
├── docs/                          # Project documentation hierarchy
│   ├── ARCHITECTURE.md            # System & Product Architecture
│   ├── CURRENT_STATUS.md          # Real capabilities snapshot
│   └── ...                        # Guides, APIs, and benchmarks
├── aura/
│   ├── apps/                      # Doctor dashboard (HTML/CSS/JS SPA)
│   ├── artifacts/                 # Serialized model weights & configurations
│   ├── common/                    # Shared utilities, configuration, event bus
│   ├── gateway/                   # FastAPI application, database schemas, and pipeline orchestration
│   ├── mimic/                     # MIMIC-CXR dataset parsers, labeling rules, and splits
│   ├── ml/                        # Training pipelines, Platt calibration, and benchmark execution
│   ├── schemas/                   # Pydantic data models (single source of truth)
│   ├── services/                  # Business logic engines (vision, fusion, safety, etc.)
│   └── tests/                     # 135+ automated unit and integration tests
├── datasets/                      # Gitignored; MIMIC-CXR validation and train subsets
└── README.md                      # Main project entrance
```
