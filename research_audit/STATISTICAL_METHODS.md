# COMPREHENSIVE STATISTICAL METHODS AND INFERENCE PROTOCOL

**Role:** Computational Statistician & Research Methodology Lead  
**Repository State:** `E:/AURA/aura-main`  
**Audit Date:** October 11, 2026

---

## 1. Primary Inference: Paired Likelihood and Equivalence Testing

### 1.1 Paired Negative Log-Likelihood
Let $y_i \in \{0, 1, \dots, C-1\}$ be the true ground-truth diagnosis for study $i \in \{1, \dots, N\}$, and let $\hat{p}_{M, i}(c)$ be the predicted probability assigned to class $c$ by model $M$ after temperature calibration. The per-study negative log-likelihood is:
$$\ell_{M, i} = -\ln \hat{p}_{M, i}(y_i)$$

The paired per-study likelihood contrast between the Quantum Variational Circuit ($Q$) and the Product-of-Experts ($C$) is defined as:
$$d_i = \ell_{Q, i} - \ell_{C, i}$$
where $d_i > 0$ indicates that the classical model achieved a superior (lower) negative log-likelihood on study $i$.

The sample mean and sample standard error are:
$$\bar{d} = \frac{1}{N} \sum_{i=1}^N d_i, \qquad \text{SE}(\bar{d}) = \sqrt{\frac{1}{N(N-1)} \sum_{i=1}^N (d_i - \bar{d})^2}$$

### 1.2 Two One-Sided Test (TOST) Equivalence Formulation
To test the hypothesis that the quantum model and the classical baseline are equivalent within a pre-registered equivalence margin $\Delta_{\text{equiv}} = \pm 0.010$ nats:

- **Null Hypothesis ($H_0$):** Non-equivalence
  $$H_{01}: \mu_d \le -\Delta_{\text{equiv}} \quad \text{or} \quad H_{02}: \mu_d \ge +\Delta_{\text{equiv}}$$
- **Alternative Hypothesis ($H_1$):** Practical Equivalence
  $$H_1: -\Delta_{\text{equiv}} < \mu_d < +\Delta_{\text{equiv}}$$

The test statistics are two one-sided $t$-ratios:
$$t_1 = \frac{\bar{d} - (-\Delta_{\text{equiv}})}{\text{SE}(\bar{d})} = \frac{\bar{d} + \Delta_{\text{equiv}}}{\text{SE}(\bar{d})}$$
$$t_2 = \frac{\Delta_{\text{equiv}} - \bar{d}}{\text{SE}(\bar{d})}$$

With degrees of freedom $\nu = N - 1$, the individual $p$-values under Student's $t$-distribution are:
$$p_1 = P(T_\nu \le t_1), \qquad p_2 = P(T_\nu \le t_2)$$
The overall equivalence $p$-value is:
$$p_{\text{TOST}} = \max(p_1, p_2)$$

Equivalence at significance level $\alpha = 0.05$ is established if and only if $p_{\text{TOST}} < \alpha$, which is mathematically equivalent to the two-sided $100(1 - 2\alpha)\% = 90\%$ confidence interval $[\bar{d} - t_{0.95, \nu}\text{SE}, \bar{d} + t_{0.95, \nu}\text{SE}]$ falling strictly inside $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$.

---

## 2. Repeated Cross-Validation Inference: Nadeau-Bengio Correction

When comparing models across $K$-fold cross-validation repeated across $R$ seeds, individual fold estimates $\hat{\delta}_{k, r}$ violate mutual independence because training folds share $1 - \frac{1}{K}$ of their data.

The standard $t$-statistic suffers from severe Type I error inflation (false positive rates frequently exceeding $30-50\%$). We implement the Nadeau-Bengio corrected resampled $t$-test:

$$\text{Var}_{\text{NB}}(\bar{\delta}) = \left( \frac{1}{K \cdot R} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) S^2_{\delta}$$
where:
$$S^2_{\delta} = \frac{1}{K \cdot R - 1} \sum_{r=1}^R \sum_{k=1}^K (\hat{\delta}_{k, r} - \bar{\delta})^2$$
The corrected test statistic:
$$t_{\text{NB}} = \frac{\bar{\delta}}{\sqrt{\text{Var}_{\text{NB}}(\bar{\delta})}} \sim t_{K \cdot R - 1}$$

This test is strictly enforced across all field-audit benchmark re-implementations (`bench/field_audit/protocol.py`).

---

## 3. Clustered and Hierarchical Bootstrap Framework

### 3.1 Patient-Level Clustered Bootstrap
To account for patient-level clustering where each patient $j \in \{1, \dots, J\}$ contributes a set of studies $\mathcal{S}_j$:
1. Sample $J$ patient clusters uniformly with replacement: $j_1^*, \dots, j_J^*$.
2. Aggregate all studies belonging to selected patient clusters: $\mathcal{D}^* = \bigcup_{k=1}^J \mathcal{S}_{j_k^*}$.
3. Compute the bootstrap statistic $\hat{\theta}^* = g(\mathcal{D}^*)$.
4. Repeat for $B = 10,000$ resamples.
5. Compute the $95\%$ percentile interval $[\hat{\theta}^*_{0.025}, \hat{\theta}^*_{0.975}]$.

### 3.2 Hierarchical Seed $\times$ Patient Bootstrap
To evaluate the stability of model comparisons across both optimization initialization seeds $s \in \{1, \dots, S\}$ and patient cohorts $j \in \{1, \dots, J\}$:
1. Resample $S$ optimizer seeds with replacement: $s_1^*, \dots, s_S^*$.
2. Resample $J$ patient clusters with replacement: $j_1^*, \dots, j_J^*$.
3. Compute the two-way bootstrap difference:
   $$\bar{d}^* = \frac{1}{S \cdot |\mathcal{D}^*|} \sum_{s \in \{s^*\}} \sum_{i \in \mathcal{D}^*} d_{i}^{(s)}$$
4. Repeat across $B = 10,000$ iterations to derive hierarchical standard errors and intervals.

---

## 4. Calibration: Temperature Scaling and Expected Calibration Error (ECE)

### 4.1 Symmetric Platt / Temperature Scaling
For uncalibrated logit vector $z \in \mathbb{R}^C$, temperature scaling introduces a single scalar parameter $T > 0$ such that calibrated class probabilities are:
$$\hat{p}(c; T) = \frac{\exp(z_c / T)}{\sum_{k=1}^C \exp(z_k / T)}$$
$T$ is fitted strictly on the held-out calibration partition by minimizing empirical cross-entropy:
$$T^* = \arg\min_{T > 0} -\sum_{i \in \text{Cal}} \ln \hat{p}(y_i; T)$$
No test set observations are ever accessed during calibration fitting.

### 4.2 Expected Calibration Error (ECE)
Predictions are partitioned into $M = 15$ equally spaced confidence bins $B_1, \dots, B_M \subset [0, 1]$ based on $\hat{p}_i = \max_c \hat{p}_{i}(c)$:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i), \qquad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
The Expected Calibration Error is computed as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

Confidence intervals for ECE are computed using 2,000 bootstrap resamples.

---

## 5. Kernel Expressivity and Geometric Difference

To quantify the expressivity and separation between the quantum feature map kernel $K_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$ and classical RBF kernel $K_C(x, x') = \exp(-\gamma \|x - x'\|^2)$, we compute:

### 5.1 Effective Dimension
Given trace-normalized Gram matrix $\tilde{K} = K / \text{Tr}(K)$ with eigenvalues $\lambda_1, \dots, \lambda_n$:
$$d_{\text{eff}}(K) = \frac{(\sum_{i=1}^n \lambda_i)^2}{\sum_{i=1}^n \lambda_i^2} = \frac{1}{\text{Tr}(\tilde{K}^2)}$$

### 5.2 Geometric Difference (Huang et al., 2021)
The geometric distance quantifying potential quantum kernel advantage over classical kernels is defined as:
$$g(K_C \parallel K_Q) = \sqrt{ \left\| K_Q^{1/2} (K_C + \lambda I)^{-1} K_Q^{1/2} \right\|_\infty }$$
where $\lambda = 10^{-3}$ is a Tikhonov regularizer. If $g(K_C \parallel K_Q)$ is small (close to 1), classical RBF kernels can efficiently approximate the quantum feature space, precluding quantum kernel advantage.
