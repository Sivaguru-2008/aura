# Scientific Deliverable 6: Systematic Literature Screening Audit & PRISMA-Style Flow Accounting

**Manuscript Title:** Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Revision R4)  
**Lead Auditor Role:** Systematic Review Methodology Expert & Literature Auditor  

---

## 1. Executive Summary & Audit Scope

This document provides the complete, independently auditable PRISMA-style (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) accounting for the systematic literature audit of quantum machine learning in digital medicine reported in the manuscript.

### Core Metrics Summary:
- **Total Records Screened:** 231 PubMed citations (January 2020 – October 2026).
- **Health-Related Clinical Records:** 129 records.
- **Records Claiming Empirical Quantum Advantage:** 41 records.
- **Eligible Publications Meeting All Inclusion Criteria:** 8 publications.
- **Computationally Reconstructible Publications:** 5 publications (spanning 6 distinct clinical tasks).
- **Single-Rater Disclosure:** All 17 checklist items were scored by a single investigator (S.R.M.) against objective, publicly verifiable operational criteria; no second reviewer has been fabricated or simulated.

---

## 2. PRISMA Literature Search & Screening Protocol

```
                     ┌────────────────────────────────────────┐
                     │   PubMed Database Query (2020-2026)    │
                     │          Records Identified            │
                     │               (n = 231)                │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │       Title & Abstract Screening       │
                     │               (n = 231)                │
                     └───────────────────┬────────────────────┘
                                         │ Excluded (n = 102):
                                         │ • Purely theoretical physics/methods
                                         │ • Quantum hardware/pulse engineering
                                         │ • Non-biomedical applications
                                         ▼
                     ┌────────────────────────────────────────┐
                     │    Health-Related Clinical Records     │
                     │               (n = 129)                │
                     └───────────────────┬────────────────────┘
                                         │ Excluded (n = 88):
                                         │ • Systematic reviews & surveys (n = 18)
                                         │ • No empirical advantage claimed (n = 70)
                                         ▼
                     ┌────────────────────────────────────────┐
                     │    Claimed Empirical QML Advantage     │
                     │               (n = 41)                 │
                     └───────────────────┬────────────────────┘
                                         │ Excluded Full-Text (n = 33):
                                         │ • Synthetic or non-clinical data (n = 14)
                                         │ • No classical comparator arm (n = 11)
                                         │ • Conceptual/perspective papers (n = 8)
                                         ▼
                     ┌────────────────────────────────────────┐
                     │     Eligible Published Studies         │
                     │               (n = 8)                  │
                     └───────────────────┬────────────────────┘
                                         │ Excluded Reconstruction (n = 3):
                                         │ • Closed proprietary features (n = 1)
                                         │ • Inaccessible restricted data (n = 1)
                                         │ • Unreconstructible architecture (n = 1)
                                         ▼
                     ┌────────────────────────────────────────┐
                     │ Reconstructible Published Publications │
                     │  (n = 5 studies, 6 clinical tasks)     │
                     └────────────────────────────────────────┘
```

### 2.1 Search Query and Execution Parameters
- **Database:** National Library of Medicine PubMed / MEDLINE.
- **Search Execution Date:** October 15, 2026 (updating previous queries from January 2020).
- **Search Syntax:**
  ```sql
  ("quantum machine learning" OR "variational quantum" OR "quantum neural network" 
   OR "quantum support vector" OR "quantum kernel") 
  AND 
  ("clinical" OR "medical" OR "diagnosis" OR "radiology" OR "healthcare" 
   OR "biomedical" OR "pathology" OR "triage")
  AND 
  ("2020/01/01"[Date - Publication] : "2026/10/15"[Date - Publication])
  ```
- **Language Restrictions:** English only.

### 2.2 Inclusion and Exclusion Criteria
1. **Inclusion Criteria (Eligible Studies, $n=8$):**
   - Peer-reviewed journal or high-profile conference proceeding published between January 2020 and October 2026.
   - Evaluated parameterized quantum circuits (VQC, QNN, QSVM, or QKL) on tabular, electrophysiological, or imaging clinical healthcare data.
   - Included at least one classical machine learning comparator (e.g., Logistic Regression, SVM, Random Forest, Multi-Layer Perceptron).
   - Explicitly claimed empirical performance superiority (accuracy, AUROC, F1-score, sensitivity, specificity, or calibration) of the quantum model over classical comparators.
2. **Exclusion Criteria for Full Text ($n=33$):**
   - Purely synthetic benchmarks (e.g., artificial concentric circles, two-moons, generic MNIST digits; $n=14$).
   - Studies evaluating quantum models without any classical comparator baseline ($n=11$).
   - Perspective, review, commentary, or editorial pieces lacking original empirical evaluations ($n=8$).
3. **Reconstructibility Criteria ($n=5$ Reconstructible Studies):**
   - Publicly accessible underlying dataset (e.g., UCI Machine Learning Repository, TCGA, BraTS, or publicly documented EEG).
   - Sufficient architectural and algorithmic detail in text or code to re-implement the variational circuit and feature mapping without ambiguous guesswork.
   - Excluded from reconstruction ($n=3$):
     * *Jahin et al. (Comput. Biol. Med. 2025):* 3-million parameter hybrid convolutional architecture with unreleased private layers and non-public training pipeline.
     * *Chen et al. (Proc. SPIE 2026):* Proprietary radiomic feature set derived from undisclosed institutional CT scans.
     * *Choi et al. (Comput. Meth. Prog. Bio. 2026):* Multi-omics cohort with restricted institutional data-use agreements prohibiting external pipeline re-execution.

---

## 3. Comprehensive Inventory of Eligible Publications ($n=8$)

| Study Identifier | Publication Venue | Clinical Task & Target | Register Qubits | Claimed Quantum Advantage | Asymmetric Risk Identified | Re-Test Verdict Under Symmetric Protocol |
|---|---|---|:---:|---|:---:|---|
| **Parthasarathi et al. (2025)** [15] | *PeerJ Computer Science* | Chronic Kidney Disease (UCI CKD, $n=400$) | 8 | VQC 99.2% accuracy beats Random Forest (90.7%) and RBF SVM (96.3%) | **YES:** Baseline untuned; LASSO and imputer fitted globally on full pooled dataset. | **NOT REPRODUCED:** Reimplemented VQC achieves 83.50% on reported features; classical models achieve 97.40%; $\Delta = -9.65\%$, $p_{\text{NB}} = 0.0009$. |
| **Daga et al. (2026a)** [16] | *Scientific Reports* | Parkinson's Disease Voice (UCI Voice, $n=195$, 31 subjects) | 10 | QKL Stacking achieves 81.3% specificity vs SVM 58.5% (claim in abstract; up to 99.1% on naive split) | **YES:** PCA fitted globally before split; multiple recordings from same subject crossed train/test splits. | **INCONCLUSIVE:** Under subject-grouped nested CV with Nadeau–Bengio correction, $\Delta = +0.33\%$, 95% CI $[-24.97\%, +25.64\%]$, $p_{\text{NB}} = 0.98$. |
| **Daga et al. (2026b)** [16] | *Scientific Reports* | Type 2 Diabetes Onset (Pima Indian Diabetes, $n=768$) | 8 | QKL Stacking achieves 62.1% sensitivity vs SVM 55.3% (up to 88.5% on author split) | **YES:** Global PCA on pooled dataset; classical SVM evaluated under arbitrary default hyperparameters. | **INCONCLUSIVE:** Under nested CV with tuned baselines, $\Delta = -5.30\%$, 95% CI $[-11.00\%, +0.39\%]$, $p_{\text{NB}} = 0.067$. |
| **Aksoy & Cattan (2024)** [17] | *Journal of Medical Systems* | Adolescent Schizophrenia (MSU EEG, $n=84$ subjects) | 10 | Quantum SVM achieves 100.0% diagnostic classification accuracy | **YES:** 2-second temporal EEG epochs from the same subject randomly shuffled into training and test splits. | **COLLAPSED TO CHANCE:** Drops to 52.14% accuracy under leak-free subject-grouped partitioning ($p = 0.32$). |
| **Akpinar & Oduncuoglu (2025)** [18] | *Scientific Reports* | Glioma Tumor Grading (TCGA $n=839$, BraTS $n=369$) | 5 | VQC 83.0% accuracy outperforms classical baselines (75%–82%) | **YES:** Feature ranking and selection conducted across all samples prior to cross-validation. | **INCONCLUSIVE:** On TCGA, $\Delta = +0.19\%$, 95% CI $[-1.22\%, +1.60\%]$, $p = 0.78$; on BraTS 2020, $\Delta\text{AUROC} = -0.022$, $p = 0.19$. |
| **Ghisoni et al. (2026)** [19] | *Scientific Reports* | Cardiovascular Disease Risk (UCI Heart, $n=69$ low-data) | 11 | Quantum Neural Network 82.72% beats classical NN 80.51% | **YES:** Early stopping evaluated on test set loss; classical NN undertuned; naive Student's $t$ reported. | **INCONCLUSIVE:** Observed difference $\Delta = -1.10\%$; Nadeau–Bengio corrected 95% CI spans $34.7$ percentage points ($[-18.45\%, +16.25\%]$, $p = 0.90$). |
| **Jahin et al. (2025)** [20] | *Computers in Biology and Medicine* | Pulmonary Radiography ($n=1,200$) | 4 | Hybrid KACQ-DCNN 92.0% beats CatBoost (86.4%) | **YES:** Untuned baseline; SMOTE oversampling applied prior to cross-validation splits. | *Eligible, Not Reconstructible:* 3-million parameter custom hybrid model; proprietary source code withheld. |
| **Chen et al. (2026)** [21] | *Proc. SPIE Medical Imaging* | Oncology Biomarkers ($n=312$) | 8 | Quantum SVM 0.799 AUROC beats classical SVM-RBF (0.608) | **Unclear:** Feature extraction proprietary and withheld. | *Eligible, Not Reconstructible:* Image features non-reproducible from published text. |
| **Choi et al. (2026)** [22] | *Comput. Meth. Prog. Bio.* | Multi-Omics Pan-Cancer ($n=540$) | 6 | Quantum SVM comparable to or exceeds tuned classical SVM | **Unclear:** Multi-omics raw tensors restricted by institutional data agreements. | *Eligible, Not Reconstructible:* Genomic tensors inaccessible without institutional sponsorship. |

---

## 4. Objective Operational Definitions for the 17-Item Checklist

To eliminate subjective evaluator bias, every item in our 17-item checklist was assigned an explicit, deterministic binary operational rule (1 = Compliant, 0 = Non-Compliant):

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

## 5. Methodological Disclosure: Single-Rater Scoring

In accordance with scientific integrity guidelines:
- **Single Investigator Scoring:** The 17-item checklist scores reported in the manuscript were compiled by a single investigator (S.R.M.).
- **No Manufactured Second Reviewer:** We explicitly disclose that no independent second reviewer scored the literature checklist. No synthetic inter-rater reliability metrics (such as a simulated Cohen's kappa) have been generated or reported.
- **Auditability:** To compensate for single-reviewer evaluation, the operational criteria for each checklist item were strictly operationalized into unambiguous, objective binary conditions. Every checklist score assigned to the audited publications is fully documented with page and section citations in [`audit/08_literature_replication.md`](file:///E:/AURA/audit/08_literature_replication.md), enabling external reviewers and readers to independently audit and verify every judgment.
