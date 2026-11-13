# Statistical Forensic Audit Report: Hypothesis Testing, Equivalence Inference, and Resampling Integrity

**Manuscript Title:** Symmetric evaluation of shallow quantum machine learning in healthcare  
**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Author:** Sivaguru R. M. (Independent Researcher)  
**Audit Date:** November 2026 (Final Submission Freeze)  
**Lead Auditor Role:** Senior Biostatistician & Clinical ML Methodology Auditor  

---

## 1. Executive Summary & Audit Mandate

This forensic report verifies every primary statistical claim, test statistic, degree of freedom, cumulative distribution function, bootstrap confidence interval, and multiple testing correction in the manuscript.

### Core Verified Assertions:
1. **Analysis C1 Confirmatory Test ($N = 56,534$):** Default VQC underperforms matched Product-of-Experts by $\Delta\text{NLL} = \mathbf{+0.0205\text{ nats}}$ ($95\%$ CI $[0.0186, 0.0224]$, $t = 21.034$, $p = 7.65 \times 10^{-98}$; TOST $p = 1.000$). Across 10 random seeds, mean paired gap is $\mathbf{+0.0142\text{ nats}}$ ($95\%$ Student's $t$ CI $[0.0106, 0.0178]$, $t = 9.014$, $\text{df} = 9$, $p = 8.43 \times 10^{-6}$).
2. **Analysis C2 Capacity-Matched Search ($N = 56,534$):** Selected VQC checkpoint achieves $\Delta\text{NLL} = \mathbf{-0.0026\text{ nats}}$ vs tuned PoE ($95\%$ CI $[-0.0036, -0.0016]$, $t = -5.10$, $p = 3.07 \times 10^{-7}$; TOST $p < 10^{-6}$). Across 10 initialization seeds, mean paired difference is $\mathbf{-0.00067\text{ nats}}$ vs PoE ($95\%$ Student's $t$ CI $[-0.00288, +0.00153]$, $t = -0.69$, $\text{df} = 9$, $p = 0.506$; Student's $t$ TOST $p = 2.57 \times 10^{-6}$) and $\mathbf{-0.00064\text{ nats}}$ vs Gated Head ($p_{\text{TOST}} = 1.50 \times 10^{-5}$).
3. **Class-Balanced Macro Breakdown:** While population-weighted difference is $-0.00235$ nats ($p_{\text{TOST}} < 10^{-6}$), class-balanced macro-averaging decisively fails operational equivalence ($\Delta\text{NLL}_{\text{macro}} = \mathbf{+0.03892\text{ nats}}$, $95\%$ bootstrap CI $[+0.03692, +0.04098]$, $p_{\text{TOST}} > 0.99$). VQC exhibits severe degradation on Malignancy ($+0.1771$ nats), Heart Failure ($+0.0753$ nats), and Pneumothorax ($+0.0549$ nats).
4. **Institutional Domain Transfer ($n = 4,800$):** Under external transfer from MIMIC-CXR to CheXpert, frozen VQC degrades severely ($\Delta\text{NLL} = \mathbf{+0.1589\text{ nats}}$ vs PoE, $95\%$ CI $[0.1422, 0.1757]$; and $\mathbf{+0.1967\text{ nats}}$ vs LR, $95\%$ CI $[0.1785, 0.2149]$; both $p < 10^{-15}$).

---

## 2. Mathematical Verification of Two One-Sided Tests (TOST)

### 2.1 Formal Hypothesis Definition
Let $d_i = \ell_{\text{quantum}}(x_i, y_i) - \ell_{\text{classical}}(x_i, y_i)$ denote the paired difference in Negative Log-Likelihood (NLL) for patient $i$. Let $\mu_d = \mathbb{E}[d_i]$ denote the expected mean paired difference:
- $\mu_d > 0 \implies$ Classical model achieves lower loss (Classical Superiority).
- $\mu_d < 0 \implies$ Quantum model achieves lower loss (Quantum Superiority).

Schuirmann's Two One-Sided Tests (TOST) evaluates the composite null hypothesis of non-equivalence against the alternative of equivalence within a pre-specified margin $\Delta_{\text{equiv}} > 0$:
$$H_0: \mu_d \le -\Delta_{\text{equiv}} \quad \text{or} \quad \mu_d \ge +\Delta_{\text{equiv}}$$
$$H_1: -\Delta_{\text{equiv}} < \mu_d < +\Delta_{\text{equiv}}$$

### 2.2 Test Statistics and Directional p-values
Let $\bar{d} = \frac{1}{N} \sum_{i=1}^N d_i$ and $\text{SE}(\bar{d}) = \frac{s_d}{\sqrt{N}}$.
1. **Lower Bound Test ($H_{01}: \mu_d \le -\Delta_{\text{equiv}}$ vs $H_{11}: \mu_d > -\Delta_{\text{equiv}}$):**
   $$t_1 = \frac{\bar{d} - (-\Delta_{\text{equiv}})}{\text{SE}(\bar{d})} = \frac{\bar{d} + \Delta_{\text{equiv}}}{\text{SE}(\bar{d})}$$
   $$p_1 = 1 - F_t(t_1; \nu)$$
2. **Upper Bound Test ($H_{02}: \mu_d \ge +\Delta_{\text{equiv}}$ vs $H_{12}: \mu_d < +\Delta_{\text{equiv}}$):**
   $$t_2 = \frac{\bar{d} - (+\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}$$
   $$p_2 = F_t(t_2; \nu)$$
3. **Overall TOST Equivalence p-value:**
   $$p_{\text{TOST}} = \max(p_1, p_2)$$
   Equivalence at significance level $\alpha = 0.05$ holds if and only if $p_{\text{TOST}} < 0.05$.

### 2.3 Degrees of Freedom Verification
- **Cohort-Level Inference ($N = 56,534$ Patients):**
  Here, $\nu = 56,534 - 1 = 56,533$. The difference between $F_t(t; 56533)$ and the standard normal CDF $\Phi(t)$ is less than $10^{-8}$ across all evaluated test values ($|F_t - \Phi| < 5 \times 10^{-6}$). Thus, asymptotic normal approximation is mathematically valid.
- **Ten-Seed Initialization Distribution ($S = 10$ Seeds):**
  Here, $\nu = 10 - 1 = 9$. Evaluating Student's $t$ with 9 degrees of freedom is strictly required. For the observed mean paired difference vs Tuned PoE ($\bar{d} = -0.00067$ nats, $\text{SE} = 0.000974$ nats):
  $$t_1 = \frac{-0.00067 - (-0.010)}{0.000974} = \frac{+0.00933}{0.000974} = 9.573$$
  $$t_2 = \frac{-0.00067 - (+0.010)}{0.000974} = \frac{-0.01067}{0.000974} = -10.957$$
  $$p_1 = 1 - F_t(9.573; 9) = 2.57 \times 10^{-6}$$
  $$p_2 = F_t(-10.957; 9) = 8.32 \times 10^{-7}$$
  $$p_{\text{TOST}} = \max(p_1, p_2) = \mathbf{2.57 \times 10^{-6}}$$
  *(Note: Applying a naive normal approximation $\Phi(9.573)$ would yield an artificially anti-conservative $p \sim 5 \times 10^{-22}$. Our implementation strictly reports the correct Student's $t$ p-value $2.57 \times 10^{-6}$.)*

### 2.4 Equivalence Margin Sensitivity Analysis
The pre-specified operational equivalence margin is $\Delta_{\text{equiv}} = \pm 0.010$ nats. We tested sensitivity across narrower and wider bounds for the 10-seed capacity-matched evaluation:
- At $\Delta = \pm 0.020\text{ nats}$: $t_1 = 19.86$, $p_{\text{TOST}} = 1.1 \times 10^{-9}$ (Decisively Equivalent).
- At $\Delta = \pm 0.010\text{ nats}$ (Pre-specified): $t_1 = 9.57$, $p_{\text{TOST}} = \mathbf{2.57 \times 10^{-6}}$ (Statistically Equivalent).
- At $\Delta = \pm 0.005\text{ nats}$: $t_1 = 4.44$, $p_{\text{TOST}} = \mathbf{0.0008}$ (Statistically Equivalent).
- At $\Delta = \pm 0.002\text{ nats}$: $t_1 = 1.36$, $p_{\text{TOST}} = \mathbf{0.103}$ (Inconclusive).

---

## 3. Class-Conditional Log-Likelihood Decomposition

Under symmetric evaluation, the 56,534 held-out patients break down across the 5 observed positive diagnostic classes as follows:

| Acuity Class | Clinical Finding | Patient Count ($n$) | Prevalence | Mean VQC NLL | Mean PoE NLL | Paired $\Delta\text{NLL}$ (VQC $-$ PoE) | $95\%$ Bootstrap CI | Clinical Impact |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **0** | Normal | 17,692 | 31.29% | 0.73786 | 0.75768 | $\mathbf{-0.01981}$ | $[-0.02167, -0.01795]$ | Quantum lower loss |
| **1** | Pneumonia | 17,052 | 30.16% | 1.02629 | 1.11918 | $\mathbf{-0.09289}$ | $[-0.09429, -0.09152]$ | Quantum lower loss |
| **2** | Heart Failure | 16,490 | 29.17% | 1.09257 | 1.01724 | $\mathbf{+0.07533}$ | $[+0.07417, +0.07650]$ | **VQC degraded** (PoE catches 1,187 more cases) |
| **4** | Malignancy | 2,192 | 3.88% | 2.89900 | 2.72190 | $\mathbf{+0.17710}$ | $[+0.16807, +0.18612]$ | **Severe VQC degradation** ($+0.1771$ nats) |
| **5** | Pneumothorax | 3,108 | 5.50% | 2.57049 | 2.51561 | $\mathbf{+0.05488}$ | $[+0.05096, +0.05863]$ | **VQC degraded** (Acute tension risk) |
| **Total / Agg.** | **All Patients** | **56,534** | **100.0%** | **1.11438** | **1.11698** | --- | --- | --- |

### Aggregation Comparison:
- **Prevalence-Weighted Population Mean:**  
  $$\Delta\text{NLL}_{\text{weighted}} = \sum_{c} w_c \Delta\text{NLL}_c = \mathbf{-0.00235\text{ nats}} \quad (95\%\text{ CI } [-0.00326, -0.00146])$$
  Falls strictly within the pre-specified equivalence band ($p_{\text{TOST}} < 10^{-6}$).
- **Class-Balanced Macro Mean:**  
  $$\Delta\text{NLL}_{\text{macro}} = \frac{1}{5} \sum_{c=1}^5 \Delta\text{NLL}_c = \mathbf{+0.03892\text{ nats}} \quad (95\%\text{ bootstrap CI } [+0.03692, +0.04098])$$
  Decisively **fails operational equivalence** ($p_{\text{TOST}} > 0.99$; lower bound $+0.03692 > +0.010$).

This proves that apparent aggregate equivalence was driven by high-prevalence categories (Normal and Pneumonia), masking substantial degradation on critical minority diagnoses.

---

## 4. Multiple Testing Control and Resampling Variance Corrections

1. **Holm–Bonferroni Step-Down Procedure:** Applied across all secondary hypotheses (C1b ablation, M2 surrogates, positive controls, and class-conditional contrasts). Family-wise error rate is controlled at $\alpha = 0.05$.
2. **Nadeau–Bengio Resampled Variance Correction:** For $J \times K$ repeated cross-validation ($5 \times 5 = 25$ folds):
   $$\sigma^2_{\text{NB}} = \left(\frac{1}{JK} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2$$
   Uncorrected Student's $t$ underestimates variance by up to $10.8\times$ because folds share training samples. In our field audit of Daga et al., uncorrected $p = 0.012$ widened to Nadeau–Bengio $p = 0.9785$, proving the apparent gain was an artifact of uncorrected cross-validation variance.
3. **DeLong Test:** Applied to evaluate correlated receiver operating characteristic (ROC) curves on held-out multimodal test splits (e.g., BraTS 2020 glioma grading), correctly accounting for covariance between model scores evaluated on the same subjects ($Z = 1.31, p = 0.1912$).
