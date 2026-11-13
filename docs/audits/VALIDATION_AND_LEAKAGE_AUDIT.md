# Validation, Calibration, and Data-Leakage Forensic Audit

**Manuscript Title:** Symmetric evaluation of shallow quantum machine learning in healthcare  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Final Submission Freeze)  
**Lead Auditor Role:** Clinical ML Reproducibility & Data Integrity Specialist  

---

## 1. Executive Summary & Audit Mandate

The manuscript discloses that during model development on CheXpert, the same 1,600-patient development validation partition was used for two distinct operations:
1. Monitoring validation loss for early-stopping checkpoint selection (with patience of 20 epochs).
2. Fitting the post-hoc scalar Platt temperature scaling parameter $T^*$ via L-BFGS.

This audit investigated:
- Whether this validation reuse introduced optimistic selection bias into the reported metrics.
- Whether any test-set information from the 56,534 held-out patients leaked into feature extraction, hyperparameter tuning, checkpoint selection, or calibration.
- How selection stability behaves across 20 resampled development partitions.

---

## 2. Partition Architecture and Data Provenance

```
================================================================================
                    CHEXPERT DATA PARTITION ARCHITECTURE
================================================================================

Total Unique Frontal Patients: N = 64,534
(Deterministic rule: 1 frontal radiograph per patient, highest study ID)

  +------------------------------------------------------------------------+
  |              DEVELOPMENT COHORT (Initial 8,000 Patients)               |
  |                                                                        |
  |  +----------------------------------+  +----------------------------+  |
  |  | Training Split (n = 6,400)       |  | Calibration Split (n=1,600)|  |
  |  | - Preprocessing parameter fit    |  | - Early stopping selection |  |
  |  | - Backpropagation gradient steps |  | - Platt temperature fit   |  |
  |  +----------------------------------+  +----------------------------+  |
  +------------------------------------------------------------------------+
                                      |
                      STRICT PARTITION FIREWALL (0 OVERLAP)
                                      |
  +------------------------------------------------------------------------+
  |             CONFIRMATORY TEST COHORT (56,534 Patients)                 |
  |                                                                        |
  |  - Completely sealed during development                                |
  |  - Zero gradient updates, zero hyperparameter peeks                    |
  |  - Evaluated only once with all weights and temperatures frozen        |
  +------------------------------------------------------------------------+
================================================================================
```

### 2.1 Verification of Patient Independence
We verified patient-level deduplication:
- Development patients: Exactly indices $0$ to $7,999$ ($n_{\text{dev}} = 8,000$).
- Confirmatory test patients: Exactly indices $8,000$ to $64,533$ ($n_{\text{test}} = 56,534$).
- Patient overlap: $|S_{\text{dev}} \cap S_{\text{test}}| = 0$.
- Intra-patient clustering: Frontal selection rule extracted exactly one radiograph per human participant, completely eliminating repeated-measure clustering in the confirmatory evaluation.

---

## 3. Investigation of Development Calibration Reuse

### 3.1 Mechanism of Validation Reuse
In `aura/ml/evaluation/v6_confirmatory.py`:
1. During iterative training across epochs, cross-entropy loss was evaluated on the 1,600 calibration patients at the conclusion of each epoch. The checkpoint with the lowest calibration loss was saved (`best_epoch`).
2. After early stopping terminated training, the saved checkpoint weights were reloaded. Logits were computed on the same 1,600 calibration patients ($L_{\text{cal}}$), and Platt temperature parameter $T^*$ was optimized by minimizing negative log-likelihood via L-BFGS:
   $$T^* = \arg\min_T \left[ -\sum_{i=1}^{1600} \ln \sigma\left(\frac{L_{\text{cal}, i}}{T}\right)_{y_i} \right]$$
3. No model was refit on the pooled 8,000 development studies prior to calibration.

### 3.2 Impact on Model-Selection Optimism
- **On Development Metrics:** Dual-use of the same 1,600 samples produces slight optimistic selection bias in the reported development calibration NLL ($1.1326$ for VQC, $1.1369$ for PoE).
- **On Confirmatory Test Metrics:** Because the 56,534 test patients were strictly sealed and played zero role in selecting checkpoints or fitting temperatures, test-set performance ($1.1144$ VQC, $1.1170$ PoE) remains an honest, out-of-sample evaluation.

### 3.3 Selection Stability Across 20 Resampled Development Partitions
To assess whether checkpoint selection was an artifact of the specific $80/20$ split, we audited `results/revision_r4/a2_selection_robustness.json`, where 20 independent permutations of the 8,000 development cases were drawn (seeds 100 to 119):
- **VQC Selected Configuration Test NLL:** $\text{Mean} = \mathbf{1.11444}$ ($\text{SD} = 0.00241$, median $1.11375$, IQR $0.00144$, range $1.11248$ to $1.12119$).
- **PoE Selected Configuration Test NLL:** $\text{Mean} = \mathbf{1.11397}$ ($\text{SD} = 0.00092$, median $1.11356$, IQR $0.00056$, range $1.11319$ to $1.11638$).
- **Paired Difference (VQC vs PoE):** $\text{Mean} = \mathbf{+0.00047\text{ nats}}$ ($\text{SD} = 0.00223$, median $-0.00002$, range $-0.00301$ to $+0.00546$).

Across all 20 resampled splits, the paired difference remained strictly within the $\pm 0.010$-nat operational equivalence margin. This demonstrates that validation reuse on development data did not induce spurious quantum advantage or compromise test-set conclusions.

---

## 4. Verification of Zero Test-Set Leakage

We systematically audited all preprocessing transformations, hyperparameter pipelines, and scoring code:
1. **Feature Extraction:** DenseNet-121 image feature representations were pre-extracted using ImageNet and standard ChestX-ray pre-trained backbones without fine-tuning on the test set.
2. **Standardization / Imputation:** All tabular transformations were fitted solely on development training folds.
3. **Hyperparameter Selection:** The 36 configurations per arm were ranked solely on the 1,600 development calibration cases. Test set data were never accessed during search.
4. **Frozen Evaluation:** Model weights and temperature parameters were saved to disk before scoring on the 56,534 test patients began.

---

## 5. Audit Conclusion & Disclosure Language

- **Verdict:** Zero test-set data leakage. The confirmatory evaluation on 56,534 patients is completely independent, unselected, and leak-free.
- **Disclosed Limitation:** The retrospective dual-use of the 1,600 development patients for both early stopping and temperature scaling is explicitly retained and transparently disclosed in Methods, Results, and Supplementary Note 7 as a development constraint.
