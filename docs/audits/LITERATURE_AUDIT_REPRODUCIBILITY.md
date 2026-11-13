# Systematic Literature Audit Reproducibility Report & PRISMA Accounting

**Manuscript Title:** Symmetric evaluation of shallow quantum machine learning in healthcare  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Final Submission Freeze)  
**Lead Auditor Role:** Clinical ML Reproducibility & Systematic Review Specialist  

---

## 1. Executive Summary & Audit Mandate

The manuscript reports a systematic literature audit of empirical quantum machine learning (QML) claims in digital healthcare:
- **231 PubMed records** identified and screened.
- **129 health-related records** evaluated.
- **41 records claiming empirical quantum advantage**.
- **8 eligible peer-reviewed publications** meeting inclusion criteria.
- **5 reconstructible publications spanning 6 distinct clinical prediction tasks** (plus 3 non-reconstructible publications).
- **17-item checklist** operationalizing symmetric evaluation across all included studies.

This document provides the complete, independently auditable PRISMA-compliant search strategy, arithmetic count reconciliation, objective binary scoring operational definitions, study-level scoring matrix, and single-rater disclosure.

---

## 2. Reconstructed PubMed Search Protocol & Strategy

### 2.1 Database & Execution Parameters
- **Database Searched:** National Library of Medicine PubMed / MEDLINE.
- **Search Execution Date:** October 15, 2026 (updating prior query cycles spanning January 1, 2020 through October 15, 2026).
- **Language Restrictions:** English language only.
- **Exact Search Syntax:**
  ```sql
  ("quantum machine learning" OR "variational quantum" OR "quantum neural network" 
   OR "quantum support vector" OR "quantum kernel") 
  AND 
  ("clinical" OR "medical" OR "diagnosis" OR "radiology" OR "healthcare" 
   OR "biomedical" OR "pathology" OR "triage")
  AND 
  ("2020/01/01"[Date - Publication] : "2026/10/15"[Date - Publication])
  ```

### 2.2 Screening Criteria & Multi-Stage Filtration
1. **Title and Abstract Screening ($n = 231$):**
   - *Included ($n = 129$):* Addressed human health, biomedicine, clinical diagnosis, or medical image/signal analysis.
   - *Excluded ($n = 102$):* Purely theoretical physics, abstract quantum computing, pulse engineering, quantum chemistry of materials, or non-biomedical benchmark tasks.
2. **Health-Related Full-Text Review ($n = 129$):**
   - *Advantage-Claiming Publications ($n = 41$):* Explicitly claimed that a quantum machine learning model outperformed classical baselines in accuracy, AUROC, sensitivity, specificity, or F1-score.
   - *Excluded ($n = 88$):* Reviews, perspectives, and surveys lacking original empirical modeling ($n = 18$); or empirical studies reporting exploratory feasibility without claiming quantum superiority ($n = 70$).
3. **Eligibility Assessment ($n = 41$):**
   - *Eligible Published Studies ($n = 8$):* Evaluated parameterized quantum circuits on tabular, electrophysiological, or imaging clinical healthcare data against at least one classical comparator.
   - *Excluded ($n = 33$):* Purely synthetic datasets (e.g., concentric circles, two moons, synthetic Gaussian blobs; $n = 14$); lack of any classical machine learning comparator baseline ($n = 11$); conceptual perspectives lacking quantitative data ($n = 8$).
4. **Reconstructibility Assessment ($n = 8$ Eligible Studies):**
   - *Computationally Reconstructible ($n = 5$ publications, 6 clinical tasks):* Open publicly accessible datasets and sufficient architectural specification to reproduce the pipeline.
   - *Excluded from Reconstruction ($n = 3$ publications):*
     * *Jahin et al. (Comput. Biol. Med. 2025):* 3-million parameter custom hybrid convolutional network with private unreleased layers and non-public training scripts.
     * *Chen et al. (Proc. SPIE 2026):* Proprietary radiomic features derived from undisclosed institutional CT scans.
     * *Choi et al. (Comput. Meth. Prog. Bio. 2026):* Multi-omics genomic tensors restricted by institutional data-use agreements prohibiting external re-execution.

---

## 3. PRISMA Flow Diagram & Arithmetic Reconciliation

```
================================================================================
                           PRISMA 2020 FLOW DIAGRAM
================================================================================

                      PubMed Database Query (2020 - 2026)
                             Records Identified
                                  (n = 231)
                                      |
                                      v
                         Title & Abstract Screening
                                  (n = 231)
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
               Health-Related Records     Excluded Records (n = 102)
                     (n = 129)            - Theoretical physics / methods
                         |                - Hardware / pulse engineering
                         |                - Non-biomedical applications
                         |
                         +------------+------------+
                         |                         |
                         v                         v
               Claimed QML Advantage      Excluded Records (n = 88)
                     (n = 41)             - Surveys & reviews (n = 18)
                         |                - No advantage claimed (n = 70)
                         |
                         +------------+------------+
                         |                         |
                         v                         v
               Eligible Studies           Excluded Full-Text (n = 33)
                    (n = 8)               - Synthetic benchmarks (n = 14)
                         |                - No classical baseline (n = 11)
                         |                - Perspectives/conceptual (n = 8)
                         |
                         +------------+------------+
                         |                         |
                         v                         v
             Reconstructible Studies     Inaccessible Studies (n = 3)
                     (n = 5)              - Proprietary layers (Jahin 2025)
               (6 Clinical Tasks)         - Private radiomics (Chen 2026)
                                          - Restricted multi-omics (Choi 2026)
================================================================================
```

### Exact Arithmetic Reconciliation:
- Total Screened: $231$
- Stage 1 Excluded: $102 \implies 231 - 102 = 129$ health-related records.
- Stage 2 Excluded: $88 \implies 129 - 88 = 41$ advantage-claiming records.
- Stage 3 Excluded: $33 \implies 41 - 33 = 8$ eligible studies.
- Stage 4 Breakdown: $5$ reconstructible ($6$ clinical tasks) $+ 3$ inaccessible $= 8$ eligible studies.
- Net arithmetic discrepancy: **$0$ records**.

---

## 4. Comprehensive Inventory of the Eight Eligible Publications

| Study Identifier | Venue | Clinical Task | Qubits | Claimed Quantum Advantage | Asymmetric Factor Identified | Re-Test Verdict Under Symmetric Protocol |
|---|---|---|:---:|---|:---:|---|
| **Parthasarathi et al. (2025)** [15] | *PeerJ CS* | Chronic Kidney Disease (UCI CKD, $n=400$) | 8 | VQC 99.2% accuracy beats Random Forest (90.7%) and RBF SVM (96.3%) | Baseline untuned; LASSO and imputer fitted globally on full pooled dataset. | **NOT REPRODUCED:** Reimplemented VQC achieves 83.50% on reported features; classical models achieve 97.40%; $\Delta = -9.65\%$, $p_{\text{NB}} = 0.0009$. |
| **Daga et al. (2026a)** [16] | *Sci Rep* | Parkinson's Disease Voice (UCI Voice, $n=195$, 31 subjects) | 10 | QKL Stacking achieves 81.3% specificity vs SVM 58.5% (claim in abstract; up to 99.1% on naive split) | PCA fitted globally before split; multiple recordings from same subject crossed train/test splits. | **INCONCLUSIVE:** Under subject-grouped nested CV with Nadeau–Bengio correction, $\Delta = +0.33\%$, 95% CI $[-24.97\%, +25.64\%]$, $p_{\text{NB}} = 0.98$. |
| **Daga et al. (2026b)** [16] | *Sci Rep* | Type 2 Diabetes Onset (Pima Indian Diabetes, $n=768$) | 8 | QKL Stacking achieves 62.1% sensitivity vs SVM 55.3% (up to 88.5% on author split) | Global PCA on pooled dataset; classical SVM evaluated under arbitrary default hyperparameters. | **INCONCLUSIVE:** Under nested CV with tuned baselines, $\Delta = -5.30\%$, 95% CI $[-11.00\%, +0.39\%]$, $p_{\text{NB}} = 0.067$. |
| **Aksoy & Cattan (2024)** [17] | *J Med Syst* | Adolescent Schizophrenia (MSU EEG, $n=84$ subjects) | 10 | Quantum SVM achieves 100.0% diagnostic classification accuracy | 2-second temporal EEG epochs from the same subject randomly shuffled into training and test splits. | **COLLAPSED TO CHANCE:** Drops to 52.14% accuracy under leak-free subject-grouped partitioning ($p = 0.32$). |
| **Akpinar & Oduncuoglu (2025)** [18] | *Sci Rep* | Glioma Tumor Grading (TCGA $n=839$, BraTS $n=369$) | 5 | VQC 83.0% accuracy outperforms classical baselines (75%–82%) | Feature ranking and selection conducted across all samples prior to cross-validation. | **INCONCLUSIVE:** On TCGA, $\Delta = +0.19\%$, 95% CI $[-1.22\%, +1.60\%]$, $p = 0.78$; on BraTS 2020, $\Delta\text{AUROC} = -0.022$, $p = 0.19$. |
| **Ghisoni et al. (2026)** [19] | *Sci Rep* | Cardiovascular Disease Risk (UCI Heart, $n=69$ low-data) | 11 | Quantum Neural Network 82.72% beats classical NN 80.51% | Early stopping evaluated on test set loss; classical NN undertuned; naive Student's $t$ reported. | **INCONCLUSIVE:** Observed difference $\Delta = -1.10\%$; Nadeau–Bengio corrected 95% CI spans $34.7$ percentage points ($[-18.45\%, +16.25\%]$, $p = 0.90$). |
| **Jahin et al. (2025)** [20] | *Comput Biol Med* | Pulmonary Radiography ($n=1,200$) | 4 | Hybrid KACQ-DCNN 92.0% beats CatBoost (86.4%) | Untuned baseline; SMOTE oversampling applied prior to cross-validation splits. | *Eligible, Not Reconstructible:* 3-million parameter custom hybrid model; proprietary source code withheld. |
| **Chen et al. (2026)** [21] | *Proc SPIE Med Imag* | Oncology Biomarkers ($n=312$) | 8 | Quantum SVM 0.799 AUROC beats classical SVM-RBF (0.608) | Feature extraction proprietary and withheld. | *Eligible, Not Reconstructible:* Image features non-reproducible from published text. |
| **Choi et al. (2026)** [22] | *Comput Meth Prog Bio* | Multi-Omics Pan-Cancer ($n=540$) | 6 | Quantum SVM comparable to or exceeds tuned classical SVM | Multi-omics raw tensors restricted by institutional data agreements. | *Eligible, Not Reconstructible:* Genomic tensors inaccessible without institutional sponsorship. |

---

## 5. Objective Operational Definitions for the 17-Item Checklist

Every item in our 17-item checklist is governed by an explicit, deterministic binary scoring rule (1 = Compliant, 0 = Non-Compliant):

### Category A: Preprocessing and Partition Integrity (Rules P3, P7)
1. **Item 1 (Leak-Free Partitioning):** All preprocessing transformations (imputation, standardization, normalization, PCA) must be fitted strictly inside training folds. Global fitting on pooled data scores 0.
2. **Item 2 (Subject-Grouped Splitting):** Partitioning must group all longitudinal, multi-recording, or temporal observations from the same human participant into the same fold ($|S_{\text{train}} \cap S_{\text{test}}| = 0$). Random shuffling of repeated records scores 0.
3. **Item 3 (Temporal Leakage Isolation):** Time-series or physiological epoch data must be partitioned chronologically or by participant. Shuffling contiguous epochs scores 0.
4. **Item 4 (In-Fold Feature Selection):** Supervised or unsupervised feature filtering (e.g., LASSO, ANOVA, mutual information) must occur strictly inside training folds. Global ranking scores 0.

### Category B: Hyperparameter Optimization and Model Matching (Rule P2)
5. **Item 5 (Equal Search Budget):** Classical comparator arms must receive a hyperparameter trial budget equal to or greater than the quantum arm. Arbitrary default baselines score 0.
6. **Item 6 (Search Space Transparency):** Exact hyperparameter search grids, distributions, and sampling schedules must be published for all arms. Concealed classical tuning scores 0.
7. **Item 7 (Matched Optimization Algorithms):** If second-order or adaptive optimizers (e.g., L-BFGS, Adam) are applied to quantum circuits, classical baselines must receive matched optimization. Comparing tuned quantum to unoptimized Adam scores 0.
8. **Item 8 (Validation Split Independence):** Winning hyperparameters must be selected on validation folds separate from test evaluation folds. Test-set peek selection scores 0.

### Category C: Probability Calibration and Output Scoring (Rule P1)
9. **Item 9 (Symmetric Calibration):** Post-hoc probability calibration (Platt temperature scaling, isotonic regression) must be applied symmetrically across all arms. Calibrating only quantum models scores 0.
10. **Item 10 (Independent Calibration Split):** Calibration parameters must be fitted on validation folds independent of test partitions. In-sample calibration scores 0.
11. **Item 11 (Proper Scoring Rule Reporting):** Models must be evaluated using proper scoring rules (NLL, Brier score) and Expected Calibration Error (ECE), not threshold-dependent accuracy alone. Accuracy-only reporting scores 0.

### Category D: Statistical Inference and Reproducibility (Rules P4, P5, P6)
12. **Item 12 (Dependency-Corrected Resampling):** Resampled cross-validation comparisons must apply variance corrections (e.g., Nadeau–Bengio). Naive Student's $t$-tests across folds score 0.
13. **Item 13 (Cluster-Adjusted Standard Errors):** Multi-patient clinical datasets must calculate standard errors adjusted for intra-patient clustering. Independent assumption on repeated data scores 0.
14. **Item 14 (Multiple Testing Control):** Family-wise error rate or false discovery rate corrections (e.g., Holm–Bonferroni) must be applied across multiple comparative endpoints. Unadjusted p-values score 0.
15. **Item 15 (Equivalence Margin Pre-Specification):** Where parity or non-inferiority is claimed, equivalence bounds ($\Delta_{\text{equiv}}$) must be prespecified and tested via TOST. Informal acceptance of the null scores 0.
16. **Item 16 (Open Code and Artifact Provenance):** Complete executable code, dependencies, and environment configurations must be publicly archived. Code-withheld studies score 0.
17. **Item 17 (Public Checkpoint Verification):** Trained model weights, prediction tensors, and evaluation splits must be available for independent verification. Missing weights score 0.

---

## 6. Study-Level Audit Scoring Matrix (17 Items)

| Checklist Item | Parthasarathi [15] | Daga (Voice) [16] | Daga (Pima) [16] | Aksoy & Cattan [17] | Akpinar [18] | Ghisoni [19] | Jahin [20] | Chen [21] | Choi [22] | Our Evaluation (AURA) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. Leak-Free Preprocessing | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | **1** |
| 2. Subject Grouping | N/A | 0 | N/A | 0 | N/A | N/A | N/A | N/A | N/A | **1** |
| 3. Temporal Isolation | N/A | N/A | N/A | 0 | N/A | N/A | N/A | N/A | N/A | **1** |
| 4. In-Fold Feature Select | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | **1** |
| 5. Equal Search Budget | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **1** |
| 6. Search Transparency | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **1** |
| 7. Matched Optimizers | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | **1** |
| 8. Validation Independence | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | **1** |
| 9. Symmetric Calibration | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 10. Cal Split Independence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 11. Proper Scoring Rules | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 12. Corrected Resampling | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 13. Clustered SEs | N/A | 0 | N/A | 0 | N/A | N/A | N/A | N/A | N/A | **1** |
| 14. Multiple Testing Adjust | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 15. Equivalence Pre-spec | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 16. Open Code Provenance | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| 17. Public Checkpoints | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |
| **Total Score (/17)** | **1 / 14** | **2 / 17** | **2 / 14** | **4 / 17** | **2 / 14** | **2 / 14** | **0 / 14** | **1 / 14** | **4 / 14** | **17 / 17** |

---

## 7. Methodological Disclosure: Single-Rater Scoring

In accordance with scientific integrity guidelines:
- **Single Investigator Scoring:** The 17-item checklist scores reported in the manuscript were compiled by a single investigator (S.R.M.).
- **No Manufactured Second Reviewer:** We explicitly disclose that no independent second reviewer scored the literature checklist. No synthetic inter-rater reliability metrics (such as a simulated Cohen's kappa) have been generated or reported.
- **Auditability:** To compensate for single-reviewer evaluation, the operational criteria for each checklist item were strictly operationalized into unambiguous, objective binary conditions. Every checklist score assigned to the audited publications is fully documented with page and section citations in [`audit/08_literature_replication.md`](file:///E:/AURA/audit/08_literature_replication.md), enabling external reviewers and readers to independently audit and verify every judgment.
