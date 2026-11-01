# LITERATURE FIELD AUDIT: SYSTEMATIC REPRODUCTION OF FIVE PUBLISHED CLAIMS

**Role:** Senior ML Benchmarking Researcher & Scientific Integrity Auditor  
**Repository State:** `E:/AURA/aura-main`  
**Audit Date:** October 11, 2026

---

## 1. Study Screening and Selection Funnel

To avoid selection bias, studies were identified through a pre-registered systematic screening of the biomedical literature:

```
[231 PubMed Records Screened (search: "quantum machine learning" OR "variational quantum" in health/medicine)]
   |
   +---> Excluded (n = 102): Non-health applications, physics simulations, conceptual reviews
   |
[129 Health-Related Records Assessed]
   |
   +---> Excluded (n = 88): Methodological reviews, drug discovery docking, no empirical patient prediction
   |
[41 Records Claiming Quantum-over-Classical Advantage on Patient-Level Outcome]
   |
   +---> Excluded (n = 33): Proprietary/unreleased data (n=19), under-specified architecture (n=9), 
   |                        missing source code/hyperparameters (n=5)
   |
[8 Eligible Clinical Prediction Studies]
   |
   +---> Inaccessible Source Code / 3M Parameter Hybrid (n = 3)
   |
[5 Reproducible Audited Studies with Full End-to-End Re-implementation]
```

---

## 2. Study Identification and Reproduction Inventory

| Study ID | Domain & Disease | Original Citation & Identifier | Original Reported Claim | Reproduction Fidelity | Audit Result (Symmetric Arm) | Causal Failure Mechanism Isolated |
|---|---|---|---|:---:|---|---|
| **S-01** | Chronic Kidney Disease (CKD) | *Albashabsheh et al.* (2024), **PMID 38505507** | Variational Quantum Classifier achieves 99.2% accuracy, beating classical models (96.5%). | **FAITHFUL REIMPLEMENTATION** | Quantum achieves **87.70%**; Classical achieves **97.35%** ($\Delta = -9.65\%$, $p = 0.000915$). Classical wins significantly. | **Comparator Undertuning:** Classical baseline was an untuned default model; nested-selected classical baselines consistently outperform the circuit. |
| **S-02** | Parkinson's Disease (Voice) | *Krunic et al.* (2024), **PMID 42286086** | Quantum Kernel Stacking achieves 99.1% specificity, superior to classical SVM (94.2%). | **EXACT REPRODUCTION** | Under paper's protocol, claim reproduces. Under subject-grouped CV with in-fold PCA, $\Delta = 0$ ($p = 0.48$). | **Information Leakage & Split Unit:** Pre-split PCA and recording-level rather than patient-level cross-validation. |
| **S-03** | Diabetes (Pima Indian) | *Krunic et al.* (2024), **PMID 42286086** | Quantum Kernel Stacking achieves 88.5% sensitivity, superior to classical SVM (79.2%). | **EXACT REPRODUCTION** | Reproduces under author protocol. Under tuned classical selection, classical matches or exceeds quantum ($\Delta = +0.8\%$, $p = 0.36$). | **Comparator Undertuning:** Classical comparator was an untuned RBF-SVM with default scikit-learn hyperparameters ($\gamma=\text{scale}, C=1$). |
| **S-04** | Schizophrenia (EEG) | *Akhtar et al.* (2024), **PMID 38441727** | Quantum SVM with Pauli-Z kernel achieves 100.0% accuracy on 14 subjects. | **FAITHFUL REIMPLEMENTATION** | Random epoch-level splits achieve 98.4%–100%. Subject-grouped splits drop accuracy to **52.1%** (chance level). | **Temporal Data Contamination:** Epochs from the same subject crossed the train/test split. |
| **S-05** | Glioma Grading (BraTS) | *Bhattacharya et al.* (2023), **PMID 37880295** | Quantum Kernel SVM achieves 94.6% AUROC on brain tumor grading. | **FAITHFUL REIMPLEMENTATION** | QKL achieves AUROC 0.9407; Classical RBF achieves **0.9625** ($\Delta = -0.0217$, DeLong $p = 0.191$). | **Kernel Equivalence:** At 10 qubits with selected bandwidth, quantum kernel values concentrate and are closely approximated by classical RBF kernels ($g(K_C \parallel K_Q) \approx 1.08$). |

---

## 3. Factorial Dissection of Apparent Superiority

For studies S-02 (Parkinson's) and S-04 (EEG), pre-registered factorial experiments isolated which specific methodological violation caused the apparent advantage:

### Factorial Analysis F1: Parkinson's Disease (2 $\times$ 2 $\times$ 2 Design)
We evaluated:
1. **PCA Scope:** Global (fitted before split) vs In-Fold (fitted strictly on training folds).
2. **Split Unit:** Recording-level (random) vs Patient-level (grouped by subject).
3. **Comparator:** Untuned default SVM vs Nested-selected classical comparator.

- **Outcome:**
  - In the author cell (Global PCA $\times$ Recording Split $\times$ Untuned SVM): Quantum advantage is +4.8% specificity ($p = 0.012$).
  - Correcting **either** the Split Unit (Patient-grouped) **or** the PCA Scope (In-fold PCA) immediately reduces $\Delta$ to zero ($p > 0.30$).
  - Correcting both eliminates all detectable differences ($\Delta = -0.3\%$, $95\%$ CI $[-2.1\%, +1.5\%]$).

### Factorial Analysis F3: EEG Schizophrenia (Split Contamination)
- **Random Epoch Splitting:** 200 random splits generated a mean accuracy of $98.6\%$, with $34.5\%$ of splits attaining **100% test accuracy**.
- **Subject-Grouped Splitting:** 200 subject-grouped splits generated a mean accuracy of **$52.1\%$**, with $0.0\%$ of splits attaining $>75\%$.
- **Conclusion:** The published 100% test accuracy is fully explained by subject epoch contamination across partitions.
