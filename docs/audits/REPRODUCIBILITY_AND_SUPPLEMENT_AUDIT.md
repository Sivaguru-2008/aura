# Reproducibility Framework & Supplementary Information Audit Report

**Manuscript Title:** Symmetric evaluation of shallow quantum machine learning in healthcare  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Final Submission Freeze)  
**Lead Auditor Role:** Clinical ML Reproducibility & Open Science Specialist  

---

## 1. Executive Summary & Audit Mandate

In accordance with *npj Digital Medicine* reproducibility standards, this audit evaluated:
1. Whether Supplementary Notes 1 and 7 contain all necessary operational protocols and technical details.
2. The operational distinction and execution status across **Level A**, **Level B**, and **Level C** reproduction tiers.
3. The empirical verification of all executed reproduction commands against published numbers and cryptographic hashes.

---

## 2. Three-Tiered Reproducibility Framework

To prevent ambiguous reproduction claims, the repository implements three strictly separated reproduction tiers:

### Level A: Checkpoint & Metric Verification (Quick Tier, $\sim$10--20 seconds)
- **Scope:** Verifies file integrity, regenerates all display figures directly from saved JSON metrics, verifies cryptographic SHA-256 hashes against `manifest.json`, and executes 31 numerical drift tests asserting that every published decimal matches its underlying artifact.
- **Data Requirement:** Only the open-source repository artifacts. Does not require access to raw clinical image datasets (MIMIC-CXR or CheXpert).
- **Execution Command:**
  ```bash
  python reproduce_paper.py
  ```
- **Audit Execution Result:** **PASSED** (Executed in 10s on host environment; 5/5 steps passed, 31/31 drift tests passed, 8/8 manifest artifacts bit-for-bit verified).

### Level B: Statistical Re-Scoring & Bootstrap Resampling ($\sim$5 minutes)
- **Scope:** Loads full-length prediction probability arrays on the 56,534 held-out patients, re-computes all Two One-Sided Tests (TOST) statistics, re-runs $10,000$ hierarchical and patient-level bootstrap resamples, and computes DeLong ROC tests.
- **Data Requirement:** Pre-computed prediction tensors (`.npz` files in `artifacts/v6/` and `bench/revision_r4/`).
- **Execution Command:**
  ```bash
  pytest aura/tests/test_doc_numbers.py
  python bench/revision_r4/run_a3_class_analysis.py
  ```
- **Audit Execution Result:** **PASSED** (26/26 documentation tests passed; class-conditional bootstrap intervals matched manuscript values).

### Level C: Full End-to-End Retraining & Feature Extraction ($\sim$16 hours compute)
- **Scope:** Trains all models from scratch: 36 quantum configurations, 36 classical PoE configurations, 36 Gated Head configurations, 10 random initialization seeds across C1, C1b, and C2, and fits classical Fourier surrogates.
- **Data Requirement:** Requires credentialed PhysioNet access for MIMIC-CXR (`AURA_MIMIC_ROOT`) and Stanford AIMI access for CheXpert (`chexpert_external.CHEXPERT_ROOT`).
- **Execution Command:**
  ```bash
  python reproduce_paper.py --full
  ```
- **Runtime Requirement:** $\sim$16 hours total compute on modern multi-core CPU/GPU workstation ($\sim$4.16 hours for the quantum 36-trial search alone).

---

## 3. Supplementary Notes Forensic Review

### Supplementary Note 1: 17-Item Checklist and Protocol Rules
- **Contents:** Full formalization of Rules P1 through P7, objective operational definitions for all 17 checklist items, and complete pseudo-code for the Symmetric Evaluation Algorithm.
- **Audit Finding:** Fully aligned with main text and PRISMA standards. Equations for TOST displayed with clean formatting and correct degrees of freedom.

### Supplementary Note 7: Search Spaces, Budgets, and Retraining Protocol
- **Contents:**
  - Complete 36-configuration hyperparameter search grids for VQC, PoE, and Gated Head.
  - Wall-clock runtime logs and compute asymmetry accounting (92.8-fold training runtime difference).
  - Selection robustness analysis across 20 resampled development splits (`a2_selection_robustness.json`).
  - Dual-use disclosure of the 1,600 calibration patients.
  - Full Level C step-by-step retraining instructions, environment specifications, random seed list (`[7, 11, 13, 17, 19, 23, 29, 31, 37, 41]`), expected output files, and execution commands.
- **Audit Finding:** Comprehensive, mathematically rigorous, and fully auditable.

---

## 4. Execution Log of Level A Verification

The Level A verification suite was executed in `aura-main`:
```
[1/5] Input provenance: pinned SHA-256 of the fusion evidence and predictions
ok  quantum_study_evidence.npz
ok  fusion_matched_optimiser_predictions.npz

[2/5] Figure 1: forest plot (every evaluation choice; published claims re-tested)
Generated: docs/figures/fig_v6_forest.png, v6_forest.json

[3/5] Figure 2: mechanism (interaction-order spectrum; kernel bandwidth sweep)
Generated: docs/figures/fig_v6_mechanism.png

[4/5] Drift tests: every manuscript number equals its artifact
...............................                                          [100%]

[5/5] Revision R4: Level A integrity verification of all new artifacts in results/revision_r4/manifest.json
Verifying 8 Revision R4 artifacts against pinned SHA-256...
  [OK] classical_seeds_summary.json (7349 bytes, SHA-256: 04fff45a11eb...)
  [OK] a1_quantum_seeds_summary.json (9395 bytes, SHA-256: 07c7778c35d9...)
  [OK] a2_selection_robustness.json (48226 bytes, SHA-256: 301193f7a10a...)
  [OK] a3_class_analysis.json (6331 bytes, SHA-256: ee0d9ea07ef0...)
  [OK] domain_shift_analysis.json (24637 bytes, SHA-256: 12e548be2377...)
  [OK] m1_effective_order.json (11763 bytes, SHA-256: efeb24b60d03...)
  [OK] m2_completeness.json (1583 bytes, SHA-256: 5a6dd9958ef5...)
  [OK] ckd_protocol_grid.json (48608 bytes, SHA-256: 57c4f1ea7682...)

Level A Verification PASSED: All 8 artifacts intact and bit-for-bit verified.
All 5 steps passed in 10s.
```

---

## 5. Audit Conclusion

The manuscript and supplementary materials adhere to the highest open-science standards. Reproduction tiers are clearly distinguished, preventing misleading claims of automated end-to-end retraining, while providing instantaneous verification of published claims from frozen artifacts.
