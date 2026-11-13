# CLINICAL VALIDITY AND MEDICAL DATASET AUDIT (TRIPOD+AI EVALUATION)

**Role:** Senior Medical AI Clinician-Scientist & Trustworthy Healthcare AI Reviewer  
**Repository State:** `E:/AURA/aura-main`  
**Audit Date:** October 11, 2026

---

## 1. Executive Summary: Construct Validity and Clinical Framing

A major hazard in medical machine learning papers is misrepresenting retrospective heuristic benchmark emulations as "clinical diagnostic systems" or "deployed decision support."

Our audit reveals that while the technical implementation and code quality of AURA are exceptional, the **clinical construct validity** requires rigorous qualification before external peer review:

1. **The Diagnostic Target is a Heuristic Rule, Not an Adjudicated Medical Ground Truth:**
   The multi-class target is derived deterministically from chest radiograph report labelers (CheXbert / NegBio) using an additive scoring heuristic (`_map_diagnosis`). The fusion model is emulating a clinical triage algorithm rather than discovering underlying pathophysiology.
2. **"Deployed System" vs Retrospective Test Harness:**
   The system was served locally via FastAPI / Docker (`aura/gateway`), but was never evaluated in a live clinical workflow with real-time patient care. Calling this a "deployed clinical decision support system" risks immediate rejection by medical AI reviewers.
3. **Data Governance & Licensing Strictness:**
   All trained weights are governed by `AURA_COMMERCIAL_MODE=0` (Research Only under PhysioNet Credentialed Health Data License and Stanford CheXpert Research Terms).

---

## 2. Dataset-by-Dataset Audit

### 2.1 MIMIC-CXR-JPG (Beth Israel Deaconess Medical Center, Boston, MA)
- **Cohort Size in Fusion Benchmarks:** 540 train, 446 calibration, 447 test studies.
- **Selection Rule:** One study per patient (highest-numbered frontal view).
- **Construct Verification:**
  - Finding probabilities (pneumothorax, nodule, edema, cardiomegaly, effusion, pneumonia, consolidation, opacity) are predicted by DenseNet-121.
  - The 6-class diagnosis (Pneumothorax, Malignancy, Heart Failure, Pneumonia, COPD, Normal) is computed via:
    - Pneumothorax if pneumothorax $>0$
    - Malignancy if nodule $>0$
    - Heart Failure if $0.7 \times \text{edema} + 0.5 \times \text{cardiomegaly} + 0.2 \times \text{effusion} > \text{threshold}$
    - Pneumonia if $0.9 \times \text{pneumonia} + 0.6 \times \text{consolidation} + 0.3 \times \text{opacity} > \text{threshold}$
    - Normal if no findings present.
- **Audit Verdict:** The task is **rule emulation**. This is scientifically valuable as an audit of fusion heads, but must be described as *rule-derived triage classification*, not clinical diagnostic certainty.

### 2.2 CheXpert-v1.0-Small (Stanford Hospital, Stanford, CA)
- **Cohort Size:** 1,600 development studies, 56,534 confirmatory test studies.
- **Selection Rule:** Strictly frontal views, one study per patient, ordered by file appearance.
- **Domain Shift Confounders:**
  1. *Scanner & Population Shift:* Significant differences in imaging hardware, contrast distributions, patient age, and disease severity.
  2. *Label Schema Mismatch (COPD Unreachable):* CheXpert does not annotate *hyperinflation*. In MIMIC, COPD is reached via hyperinflation. Consequently, in CheXpert, COPD has zero reachable cases by design (`scores[Diagnosis.COPD] = 0.0`).
  3. *Uncertainty Handling:* CheXpert includes explicit `-1.0` (uncertain) annotations. Two pre-registered policies were evaluated:
     - `y_u0`: Uncertain mapped to negative (conservative primary policy).
     - `y_u1`: Uncertain mapped to positive (sensitivity analysis).
- **Audit Verdict:** External degradation on CheXpert reflects composite domain shift (population, scanner, and label mapping truncation), which cannot be attributed purely to model capacity without controlling for these domain factors.

### 2.3 BraTS 2020 (Multicentric Glioma Grading)
- **Cohort Size:** 369 official subjects (293 HGG, 76 LGG).
- **Held-Out Isolation:** 111 subjects strictly held out from encoder training and feature extraction (`qkl_brats2020_official_heldout111.json`).
- **Audit Verdict:** Upstream leakage was completely audited and resolved following the September 2026 Erratum. The finding that Quantum Kernel Learning does not outperform classical RBF kernels on 3D MRI feature embeddings is robust.

---

## 3. TRIPOD+AI Reporting Checklist Assessment

| TRIPOD+AI Item | Description | Compliance Status | Audit Recommendation |
|---|---|:---:|---|
| **Item 1** | Title identifies study as model evaluation / audit | **YES** | Title in v6 clearly states evaluation and audit scope. |
| **Item 2** | Abstract presents structured / clear summary | **PARTIAL** | Unstructured abstract ($\le 150$ words) required for *npj Digital Medicine*. |
| **Item 4** | Source of data clearly specified | **YES** | MIMIC-CXR, CheXpert, BraTS 2020 fully cited with version numbers. |
| **Item 5** | Eligibility criteria and participant selection | **YES** | One-study-per-patient rule and frontal view filtering documented. |
| **Item 6** | Predictor definitions and handling | **YES** | 8-channel DenseNet probability vector explicitly detailed. |
| **Item 7** | Outcome definition and derivation | **CRITICAL UPDATE** | Must state plainly that outcome is derived via algorithmic heuristic rules. |
| **Item 10** | Sample size determination | **YES** | $n=447$ for development, $n=56,534$ for confirmatory test power. |
| **Item 11** | Handling of missing data / uncertainty | **YES** | `y_u0` vs `y_u1` pre-registered and reported. |
| **Item 14** | Calibration evaluation alongside discrimination | **YES** | ECE, Reliability diagrams, and Temperature Scaling evaluated. |
| **Item 19** | Distinction between internal and external validation | **YES** | CheXpert treated strictly as external institution validation. |
