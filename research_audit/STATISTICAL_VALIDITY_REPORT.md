# STATISTICAL VALIDITY REPORT: ADVERSARIAL RECONSTRUCTION

**Role:** Computational Statistician & Adversarial Methodology Reviewer  
**Repository State:** `E:/AURA/aura-main`  
**Audit Date:** October 11, 2026

---

## 1. Experimental Units, Sampling Units, and Dependence Structures

A frequent failure mode in clinical AI and machine learning benchmarks is the violation of statistical independence assumptions: treating repeated measurements on the same patient, multiple cross-validation folds, or multiple random optimizer seeds as independent observations.

We audited every statistical inference pipeline in the manuscript:

### 1.1 MIMIC-CXR Fusion Cohort (n=447)
- **Sampling Unit:** Unique patient (`subject_id`).
- **Experimental Unit:** Frontal chest radiograph study (`study_id`).
- **Independence Audit:** Verified. `build_real_evidence_dataset.py` filters to strictly **one study per patient**. Therefore, study-level resampling is mathematically equivalent to patient-level resampling.
- **Verification Code:** Pinned in `revision_v51.json`:
  ```json
  "resampling_unit_note": "one study per patient (build_real_evidence_dataset), so study-level resampling is patient-level resampling"
  ```

### 1.2 CheXpert External Cohort
- **Sampling Unit:** Unique patient.
- **Selection Rule:** Highest-numbered study with a frontal view per patient (`chexpert_external.select_studies`).
- **Independence Audit:** Verified. Guaranteed patient-disjoint across all 1,600 development and 56,534 confirmatory test cases.

### 1.3 BraTS 2020 Glioma Cohort
- **Sampling Unit:** Unique patient / subject (`BraTS20_Training_XXX`).
- **Experimental Unit:** 3D multi-parametric MRI volume.
- **Independence Audit:** Verified. 5-fold cross-validation partitioned strictly at the subject level (369 subjects total; 111 held-out subjects isolated prior to any hyperparameter selection).

### 1.4 Field Audit Re-implementations
- **Parkinson's Voice (PMID 42286086):**
  - *Original Publication Flaw:* 195 recordings from 31 patients. Cross-validation was performed by randomly splitting recordings, resulting in **patient-level data leakage** across train and test folds.
  - *Audit Correction:* Grouped 10-fold cross-validation by patient identifier (`factors_parkinsons.py`), proving that apparent quantum superiority disappears when patient clusters are respected.
- **EEG Schizophrenia (PMID 38441727):**
  - *Original Publication Flaw:* 14 subjects (7 schizophrenia, 7 healthy), sliced into short epochs. Random splitting of epochs placed adjacent time-windows of the same subject into both train and test splits, producing an artificial "100% accuracy" claim.
  - *Audit Correction:* Subject-level grouping drops accuracy to ~52% across 200 random splits (`eeg_qsvm.json`), confirming the 100% claim was an artifact of split contamination.

---

## 2. Statistical Tests and Hypothesis Families

### 2.1 Paired Likelihood and Accuracy Comparisons
- **Endpoint:** Per-study paired difference $d_i = \text{NLL}_{A, i} - \text{NLL}_{B, i}$ and classification correctness indicator $c_i = \mathbb{I}(\hat{y}_{A, i} = y_i) - \mathbb{I}(\hat{y}_{B, i} = y_i)$.
- **Tests Used:**
  - *Likelihood:* Wilcoxon signed-rank test (non-parametric paired median difference) and paired Student's $t$-test.
  - *Accuracy:* McNemar's test for paired binary proportions.
  - *Multiple Testing Correction:* Holm-Bonferroni step-down procedure applied over the family of all pairwise backend contrasts ($\binom{4}{2} = 6$ comparisons: Classical vs Quantum, Classical vs Learnable, Classical vs Logistic, Quantum vs Learnable, Quantum vs Logistic, Learnable vs Logistic).

### 2.2 Repeated Cross-Validation Inference (Nadeau-Bengio Correction)
Standard Student's $t$-tests over repeated cross-validation folds vastly underestimate variance due to fold overlap.
- **Correction Formula:**
  $$\sigma^2_{\text{corrected}} = \left( \frac{1}{K \cdot R} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) S^2$$
  where $K$ is folds, $R$ is repeats, and $S^2$ is the sample variance across the $K \cdot R$ estimates.
- **Application:** Applied throughout the field audit (`bench/field_audit/protocol.py`, `factors.py`). This prevents false-positive claims of model superiority across repeated runs.

---

## 3. Equivalence Testing and Margin Justification

The manuscript pre-registered a Two One-Sided Test (TOST) framework to evaluate whether the Quantum VQC and Product-of-Experts are practically equivalent:

- **Hypotheses:**
  $$H_0: |\mu_{\Delta\text{NLL}}| \ge \Delta_{\text{equiv}} \quad \text{vs} \quad H_1: -\Delta_{\text{equiv}} < \mu_{\Delta\text{NLL}} < \Delta_{\text{equiv}}$$
- **Equivalence Margin:** $\Delta_{\text{equiv}} = \pm 0.010$ nats.
- **Clinical & Information-Theoretic Justification:**
  - $0.010$ nats corresponds to a relative likelihood ratio difference of $\exp(0.01) - 1 \approx 1.005$ (0.5% shift in odds).
  - In clinical diagnostic triage across 6 classes, a log-likelihood difference $< 0.010$ is clinically imperceptible and smaller than the calibration uncertainty introduced by binning errors.

### Empirical TOST Results
1. **On MIMIC-CXR (n=447):**
   - Mean $\Delta\text{NLL} = +0.00335$, $\text{SE} = 0.01843$.
   - $90\%$ Confidence Interval: $[-0.02704, +0.03373]$.
   - $p_{\text{TOST}} = \max(p_{\text{lower}}, p_{\text{upper}}) = 0.35921$.
   - **Verdict:** **EQUIVALENCE NOT ESTABLISHED**. The sample size $n=447$ has insufficient power to resolve a $\pm 0.010$ margin. The manuscript correctly states: *"we say only that no difference was detected; equivalence is not claimed on this cohort."*
2. **On Confirmatory Cohort (n=56,534):**
   - Expected standard error: $\text{SE} \approx \frac{0.21}{\sqrt{56534}} \approx 0.00088$.
   - Expected $90\%$ CI half-width: $\approx 1.645 \times 0.00088 \approx \pm 0.00145$ nats.
   - This provides $>99\%$ statistical power to verify whether $|\mu_{\Delta\text{NLL}}| < 0.010$.

---

## 4. Clustered and Hierarchical Bootstrap Analysis

To rigorously separate seed-level variation from study-level sampling uncertainty, the audit executed:

1. **Study-Level Clustered Bootstrap (10,000 resamples):**
   - Point estimate: $\Delta\text{NLL} = 0.0250$.
   - Bootstrap Standard Error: $0.01195$.
   - $95\%$ Percentile Confidence Interval: $[0.00158, 0.04828]$.
   - Two-sided empirical $p$-value: $0.0360$.
2. **Hierarchical Seed $\times$ Study Bootstrap (10,000 resamples):**
   - Accounts for simultaneous random draws of optimizer seed and patient study.
   - Point estimate: $0.0250$.
   - Bootstrap Standard Error: $0.01356$.
   - $95\%$ Confidence Interval: $[-0.00110, 0.05218]$.
   - Two-sided empirical $p$-value: $0.0618$.
- **Statistical Interpretation:** While seed-averaged likelihood on this fixed cohort reliably favors the unentangled arm ($p=0.036$), the hierarchical bootstrap crosses zero ($p=0.062$), establishing that when *both* seed initialization and patient sampling variance are modeled simultaneously, the entanglement penalty is marginally significant. The manuscript must report this hierarchical uncertainty transparently.
