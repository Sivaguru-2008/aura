# Quantum and Classical Comparison Fairness Audit Report

**Manuscript Title:** Symmetric evaluation of shallow quantum machine learning in healthcare  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Final Submission Freeze)  
**Lead Auditor Role:** Quantum Machine Learning Researcher & Statistical Methodology Auditor  

---

## 1. Executive Summary & Audit Mandate

In scientific comparisons between quantum and classical machine learning models, claims of "fairness" frequently confound distinct experimental dimensions. This audit investigated the exact computational, parametric, and architectural matching implemented in the manuscript:
- **36 Hyperparameter Configurations Evaluated Per Arm.**
- **Quantum Grid Coverage:** 26.67% ($\sim 27\%$) of the 135-configuration search space.
- **Classical Grid Coverage:** 60.0% of the 60-configuration search space.
- **Computational Runtime Asymmetry:** **92.8-fold difference** in total training wall-clock time (14,969.4 seconds for VQC vs 161.4 seconds for PoE).
- **Parameter Capacity:** 118 parameters for tuned VQC vs 54 for PoE and 126 for the nonlinear Gated Head.

---

## 2. Parameter and Search Space Specifications

### 2.1 Model Capacities & Trainable Parameter Counts
- **Analysis C1 (Default Matched Arms):**
  - Classical Product-of-Experts (PoE): **54 parameters** ($8 \times 6$ weights $+ 6$ biases).
  - Logistic Regression: **54 parameters**.
  - Classical Attention-Gated Head: **126 parameters** ($64$ gating weights $+ 8$ gating biases $+ 54$ readout parameters).
  - Variational Quantum Circuit (VQC, $L=3$): **102 parameters** ($48$ variational angles $+ 54$ linear readout weights).
- **Analysis C2 (Tuned Winning Checkpoints):**
  - Tuned Product-of-Experts: **54 parameters**.
  - Tuned Attention-Gated Head: **126 parameters**.
  - Optimal Tuned VQC ($L=4$): **118 parameters** ($64$ variational angles $+ 54$ readout weights).
- **Analysis M2 (Classical Fourier Surrogates):**
  - Order-1 Trigonometric Logistic Regression: **90 parameters** ($14$ trigonometric features $+ 1$ bias $\times 6$ classes).
  - Order-2 Trigonometric Logistic Regression: **678 parameters** ($112$ pairwise features $+ 1$ bias $\times 6$ classes).
  - Polynomial Logistic Regression ($x, x^2$): **90 parameters** ($14$ polynomial features $+ 1$ bias $\times 6$ classes).

### 2.2 Search Space Definitions and Sampling Methodology
- **Variational Quantum Circuit Grid ($N = 135$ Total Configurations):**
  - Depth $L \in \{1, 2, 3, 4, 6\}$ layers.
  - Encoding scale $s \in \{0.5, 1.0, 2.0\}$.
  - Initialization standard deviation $\sigma \in \{0.01, 0.1, 1.0\}$.
  - Learning rate $\eta \in \{0.01, 0.05, 0.1\}$.
  - *Sampling:* Exactly 36 configurations drawn without replacement via `default_rng(7)` ($36 / 135 = 26.67\%$).
- **Classical Product-of-Experts Grid ($N = 60$ Total Configurations):**
  - Learning rate $\eta \in \{0.003, 0.01, 0.05, 0.1, 0.3\}$.
  - Weight decay $\lambda \in \{0.0, 10^{-5}, 10^{-4}, 10^{-3}\}$.
  - Initialization standard deviation $\sigma \in \{0.01, 0.1, 1.0\}$.
  - *Sampling:* Exactly 36 configurations drawn without replacement via `default_rng(7)` ($36 / 60 = 60.0\%$).

---

## 3. Computational Effort and Wall-Clock Runtime Asymmetry

To eliminate concealed computational advantages, all 36 configurations per arm were trained on identical host hardware:
- **Product-of-Experts (PoE):** Total runtime = **161.4 seconds** (mean $4.5\text{ s}$ / config; range $1.2$ to $14.3\text{ s}$).
- **Nonlinear Gated Head:** Total runtime = **228.1 seconds** (mean $6.3\text{ s}$ / config; range $2.1$ to $14.7\text{ s}$).
- **Variational Quantum Circuit (VQC):** Total runtime = **14,969.4 seconds** (4.16 hours; mean $415.8\text{ s}$ / config; range $114.6$ to $1642.6\text{ s}$).

### Critical Methodological Distinction:
1. **Search-Trial Matching:** Matched identically at $k = 36$ hyperparameter configurations evaluated.
2. **Computational Budget Matching:** Highly asymmetric. Because quantum statevector simulation requires dense matrix-vector multiplications across $2^8 = 256$ state amplitudes per sample and layer, VQC training consumed **92.8 times more wall-clock compute** than classical PoE.

---

## 4. Sensitivity Analysis: Full 60-Configuration Classical Grid

To verify whether classical models were disadvantaged by sampling only 36 of 60 grid points, all 60 classical configurations were evaluated in `results/revision_r4/a2_selection_robustness.json`:
- **36-Trial Classical Winner:** Calibration NLL = **1.06504**, Test NLL = **1.11744**.
- **Full-60 Classical Winner:** Calibration NLL = **1.06504**, Test NLL = **1.11744**.
- **Observed Difference:** Exactly **0.00000 nats** on calibration and test. The 36-configuration draw successfully discovered the global optimum of the classical grid.

---

## 5. Summary of Matched and Unmatched Experimental Dimensions

| Experimental Dimension | Status | Audit Verification & Evidence |
|---|:---:|---|
| **Training Cohort Size ($n=6,400$)** | **MATCHED** | Identical patient representations and ground-truth labels across all arms. |
| **Validation Selection Split ($n=1,600$)** | **MATCHED** | Winning checkpoints selected by minimum calibration NLL under identical patience. |
| **Confirmatory Test Cohort ($N=56,534$)** | **MATCHED** | Scored on the exact same untouched clinical test patients. |
| **Evaluated Configuration Budget ($k=36$)** | **MATCHED** | Exactly 36 configurations evaluated per arm. |
| **Early Stopping Criterion (Patience = 20)** | **MATCHED** | Identical early stopping logic across all iterative runs. |
| **Probability Calibration Protocol** | **MATCHED** | Post-hoc Platt temperature scaling fitted on validation split via L-BFGS. |
| **Hyperparameter Grid Coverage** | **UNMATCHED** | 26.67% of quantum space sampled vs 60.0% of classical space. |
| **Computational Wall-Clock Compute** | **UNMATCHED** | VQC consumed 92.8-fold more training compute than PoE. |
| **Target Representation Encoding** | **UNMATCHED** | VQC evaluates trigonometric Fourier basis terms ($RY(\pi x_i)$); standard PoE lacks periodic terms (resolved in Analysis M2). |

---

## 6. Audit Conclusion & Journal Readiness

The manuscript transparently and accurately presents both what was matched and what was unmatched. It explicitly refrains from claiming compute-budget parity, correctly identifying that VQC consumed 92.8-fold more runtime, and notes that elementary classical Fourier surrogates outperforming tuned VQC (Analysis M2) prove that the minor apparent advantage of VQC over linear PoE was driven by trigonometric basis expansion rather than quantum computational superiority.
