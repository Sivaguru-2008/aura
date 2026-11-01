# Claimed quantum benefits in clinical machine learning do not survive symmetric evaluation

**Author:** Sivaguru R. M.  
**Affiliation:** Independent Researcher, Chennai, India  
**Correspondence:** sivagurumurugan1@gmail.com  
**Target Journal:** *npj Digital Medicine* (Article Format)  

---

## Abstract

Clinical machine learning studies frequently assert that novel model architectures confer superior predictive performance over standard baselines. In digital health, empirical claims that quantum machine learning (QML) algorithms outperform classical models on near-term hardware have proliferated, yet the methodological symmetry of these comparisons has rarely been audited. Here, we evaluate whether reported empirical quantum benefits survive symmetric evaluation—a protocol requiring matched hyperparameter optimization budgets, independent probability calibration, leak-free partition design, and appropriate statistical hypothesis testing. We investigated this question through an internal clinical decision-support prototype for chest radiograph triage and an independent field audit of published literature. In our chest radiograph prototype, reported calibration and likelihood advantages disappeared when baselines received matched optimization and independent temperature scaling. In a pre-registered confirmatory test across 56,534 untouched clinical patients (one frontal radiograph per patient; zero overlap with development cohorts), the default variational quantum circuit performed materially worse on negative log-likelihood than matched classical baselines ($\Delta\text{NLL} = +0.0205$ nats, 95% CI $0.0186$ to $0.0224$; $t = 21.034$, $p = 7.65 \times 10^{-98}$; positive indicates quantum is worse). Removing entanglement improved likelihood across ten initialization seeds ($p = 0.0013$), an effect traced to inductive bias mismatch on an additive clinical triage rule where 100.0% of non-trivial circuit energy was concentrated in high-order cross-terms. When granted symmetric search budgets over 36 configurations per arm, the optimal quantum configuration achieved a small, statistically detectable likelihood advantage over classical baselines ($\Delta\text{NLL} = -0.0026$ nats vs linear Product-of-Experts, 95% CI $-0.0036$ to $-0.0016$, $p = 3.07 \times 10^{-7}$; $\Delta\text{NLL} = -0.0012$ nats vs nonlinear Gated Head, 95% CI $-0.0019$ to $-0.0004$, $p = 0.0025$). Both confidence intervals fell entirely inside our pre-registered operational equivalence band ($\pm 0.010$ nats; Two One-Sided Tests $p_{\text{TOST}} < 10^{-6}$), demonstrating practical statistical equivalence rather than transformative utility. In an independent field audit of 231 PubMed records, eight publications met eligibility criteria, of which five studies spanning six clinical prediction tasks were reconstructible and reimplemented. Under pre-registered survival criteria requiring statistically significant superiority under leak-free nested cross-validation, zero of the six published claims survived: one advantage reversed with classical models significantly superior ($p = 0.0009$), four were statistically inconclusive ($p > 0.05$) due to wide corrected uncertainty intervals (e.g., heart disease 95% CI $-18.45$ to $+16.25$ percentage points), and one collapsed to chance level (52.1% accuracy) when participant-level temporal partition leakage was eliminated. While these five reimplemented publications and six tasks cannot support a universal claim that quantum ML never offers an advantage, they establish a carefully scoped conclusion: reported empirical benefits in these clinical regimes dissolve under symmetric evaluation.

---

## Introduction

New machine learning components are routinely introduced into biomedical data science with claims of performance gains over established baselines. When an empirical comparison demonstrates that a proposed model outperforms a comparator, that difference can reflect two fundamentally distinct causes: the proposed architecture may possess genuine representational or computational superiority on the target data distribution, or the proposed architecture may have received disproportionate methodological attention—extensive hyperparameter tuning, bespoke preprocessing, selective thresholding, and specialized post-hoc calibration—while the baseline was evaluated under unoptimized default settings [2, 3]. Although each modeling decision may appear justifiable in isolation, the resulting comparison is scientifically asymmetric.

The phenomenon of asymmetric evaluation has been rigorously documented across numerous subfields of applied machine learning, including recommender systems [2], metric learning [3], deep reinforcement learning [4], learned physics simulators [6], and clinical image analysis [5]. In healthcare, where algorithmic predictions inform critical triage decisions, diagnostic stratification, and resource allocation, asymmetric evaluations present acute risks: they threaten patient safety, distort research priorities, and divert institutional capital toward computationally burdensome models that offer no verifiable patient benefit [1].

Quantum machine learning (QML) applied to clinical data represents an ideal domain in which to examine evaluation symmetry. Over the past five years, numerous studies have reported that parameterized variational quantum circuits (VQCs), quantum support vector machines (QSVMs), and quantum neural networks (QNNs) achieve superior diagnostic accuracy, sensitivity, or calibration compared to standard classical algorithms on biomedical classification benchmarks. However, theoretical analyses establish severe boundary conditions for near-term noisy intermediate-scale quantum (NISQ) devices operating at modest widths (4 to 12 qubits). At these widths, statevector quantum evolution can be simulated classically in fractions of a millisecond on standard laptop processors; classical computational intractability does not imply an inductive learning advantage [10]; kernel matrix elements concentrate exponentially around identity with increasing width [11, 12]; and gradient landscapes suffer from barren plateaus under random parameterized circuits [13]. A recent systematic review in this journal screened 4,915 citations and found quantum–classical performance differences to be negligible across 16 quantitatively synthesizable studies, observing that classical comparators were frequently selected ad hoc without open code or verified hyperparameter search [1].

These findings raise an unresolved question in clinical artificial intelligence: **When claimed quantum machine-learning advantages in clinical prediction are evaluated with symmetric model selection, comparable classical baselines, leakage-resistant data partitioning, and appropriate statistical inference, which advantages remain supported by evidence?**

To address this question, we formulated a seven-rule symmetric evaluation protocol (P1–P7) and operationalized it into a 17-item checklist with objective binary scoring criteria. We executed a dual-track empirical investigation:
1. **Confirmatory Clinical Decision-Support Study:** We audited an internal multi-class clinical triage prototype mapping radiographic finding probabilities from frontal chest radiographs to six acuity categories. We evaluated four interchangeable backend architectures (Classical Product-of-Experts, Multinomial Logistic Regression, Attention-Gated Nonlinear Neural Head, and Variational Quantum Circuit) across development cohorts ($n=8,000$) and a pre-registered confirmatory cohort of 56,534 untouched clinical patients.
2. **Systematic Benchmark Re-evaluation:** We conducted an independent screening of 231 PubMed publications, systematically identifying eight eligible studies and computationally reimplementing five reconstructible publications spanning six distinct clinical prediction tasks (chronic kidney disease, Parkinson's voice phonation, diabetes onset, schizophrenia electroencephalography, glioma tumor grading, and low-data heart disease risk).

Across both tracks, our findings demonstrate that claimed quantum advantages in clinical machine learning are artifacts of asymmetric evaluation choices: undertuned baselines, one-sided calibration, and partition leakage across subjects. When evaluation protocols are symmetrized, claimed advantages either reverse in classical favor, become statistically indistinguishable, or collapse to chance.

---

## Methods and Mathematical Formulation

### 1. Supervised Learning Objective and Empirical Risk

Let $\mathcal{X} \subset \mathbb{R}^d$ denote the input feature space and $\mathcal{Y} = \{1, \dots, C\}$ denote the label space corresponding to $C$ mutually exclusive clinical categories. A training dataset consists of $N$ independent and identically distributed pairs:
$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathcal{X}, \quad y_i \in \mathcal{Y}$$
A classification model parameterizes a mapping $f_\theta: \mathcal{X} \to \mathbb{R}^C$, generating unnormalized class scores (logits) $z(x) = f_\theta(x)$. Predicted class probabilities are obtained via the softmax transformation:
$$P_\theta(y = k \mid x) = \sigma(z(x))_k = \frac{\exp(z_k(x))}{\sum_{j=1}^C \exp(z_j(x))}$$
Model parameters $\theta \in \Theta$ are estimated by minimizing the empirical cross-entropy loss over training data:
$$\mathcal{L}(\theta; \mathcal{D}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C \mathbb{I}(y_i = k) \ln P_\theta(y = k \mid x_i) + \lambda \mathcal{R}(\theta)$$
where $\mathbb{I}(\cdot)$ is the indicator function and $\mathcal{R}(\theta)$ is an L2 weight regularization penalty with regularization coefficient $\lambda \ge 0$.

### 2. Classical Comparator Parameterization

We evaluated three standard classical model families matched in input dimensionality and trained under identical optimization schedules:

1. **Multinomial Logistic Regression:** A convex linear baseline mapping features directly to class logits:
   $$z_{\text{logistic}}(x) = W x + b, \quad W \in \mathbb{R}^{C \times d}, \quad b \in \mathbb{R}^C$$
   With $d=7$ clinical channels and $C=6$ triage classes, the architecture contains $7 \times 6 + 6 = 48$ weight parameters and 6 biases (54 trainable parameters).

2. **Classical Product-of-Experts (PoE):** A factorized log-linear architecture that models the joint class distribution as an unnormalized product of one-dimensional marginal feature distributions:
   $$z_{\text{poe}}(x)_k = \sum_{j=1}^d w_{k, j} x_j + b_k, \quad w_k \in \mathbb{R}^d, \quad b_k \in \mathbb{R}$$
   The architecture possesses 54 trainable parameters, providing an exact linear comparator for factorized variational quantum state readouts.

3. **Classical Attention-Gated Nonlinear Head:** A nonlinear neural network incorporating self-attention gating over input features:
   $$a(x) = \text{softmax}(W_a x + b_a), \quad W_a \in \mathbb{R}^{d \times d}$$
   $$h(x) = \text{ReLU}(W_h (a(x) \odot x) + b_h), \quad W_h \in \mathbb{R}^{h \times d}$$
   $$z_{\text{gated}}(x) = W_o h(x) + b_o, \quad W_o \in \mathbb{R}^{C \times h}$$
   With hidden dimension $h=8$, the architecture contains 126 trainable parameters, providing a parameter-matched nonlinear classical comparator.

### 3. Variational Quantum Circuit Parameterization

The quantum classifier processes clinical features using an $n$-qubit parameterized quantum circuit ($n=d=7$ active channels, with an 8th zero-reference channel). The computational pipeline consists of three stages: state preparation, unitary variational evolution, and observable measurement.

```
       ┌────────────────────────┐  ┌──────────────┐  ┌──────────────┐
|0⟩ ───┤ RY(π x₁)               ├──┤              ├──┤ RY(θ₁) RZ(φ₁)├─── ··· ─── [⟨Z₁⟩]
       ├────────────────────────┤  │              │  ├──────────────┤
|0⟩ ───┤ RY(π x₂)               ├──┤              ├──┤ RY(θ₂) RZ(φ₂)├─── ··· ─── [⟨Z₂⟩]
       └────────────────────────┘  │   CNOT Ring  │  └──────────────┘
  ⋮                ⋮               │  Entangler   │         ⋮                  ⋮
       ┌────────────────────────┐  │              │  ┌──────────────┐
|0⟩ ───┤ RY(π x₇)               ├──┤              ├──┤ RY(θ₇) RZ(φ₇)├─── ··· ─── [⟨Z₇⟩]
       └────────────────────────┘  └──────────────┘  └──────────────┘
       State Preparation (S(x))    ─── Layer l of L Variational Ansatz ───       Measurement
```

1. **State Preparation and Channel Encoding:** The input feature vector $x \in [0, 1]^8$ consists of seven normalized radiological finding probabilities extracted from frontal chest radiographs via DenseNet-121 ($x_1, \dots, x_7$) and a constant baseline reference channel $x_8 \equiv 0.0$. The register is initialized in the zero state $|0\rangle^{\otimes 8}$. Each channel is mapped to a single-qubit Pauli-$Y$ rotation:
   $$|\psi(x)\rangle = S(x) |0\rangle^{\otimes 8} = \bigotimes_{i=1}^8 R_Y(\pi s x_i) |0\rangle$$
   $$R_Y(\alpha) = \exp\left(-i \frac{\alpha}{2} Y\right) = \begin{pmatrix} \cos(\alpha/2) & -\sin(\alpha/2) \\ \sin(\alpha/2) & \cos(\alpha/2) \end{pmatrix}$$
   where $s > 0$ is a scalar encoding bandwidth scale ($s = 1.0$ in default C1; $s \in \{0.5, 1.0, 2.0\}$ in C2). Because $x_8 \equiv 0.0$, the eighth qubit receives a zero-angle rotation $R_Y(0)|0\rangle = |0\rangle$, remaining unrotated in the ground state to provide a constant quantum reference channel. In the entangled circuit, this unrotated qubit is coupled to qubits 7 and 1 via the closed entangling ring, and its expectation value $\langle Z_8 \rangle \in [-1, 1]$ is measured alongside the active finding channels.

2. **Variational Ansatz ($U(\theta)$) and Parameter Counts:** The state evolves through $L$ repeating variational layers. Each layer $l \in \{1, \dots, L\}$ consists of parameterized single-qubit rotations followed by entangling gates:
   $$U(\theta) = \prod_{l=1}^L \left( W_{\text{ent}} \bigotimes_{i=1}^8 R_Y(\theta_{l, i}) R_Z(\phi_{l, i}) \right)$$
   where $\theta_{l, i}, \phi_{l, i} \in [-\pi, \pi]$ are trainable variational parameters. The entangling unitary $W_{\text{ent}}$ is a closed ring of Controlled-NOT (CNOT) gates:
   $$W_{\text{ent}} = \left( \prod_{i=1}^{7} \text{CNOT}_{i, i+1} \right) \cdot \text{CNOT}_{8, 1}$$
   - In default Analysis C1 ($L=3, n=8$), there are $2 \times 8 \times 3 = 48$ variational angles. Combined with the linear readout matrix $W_{\text{out}} \in \mathbb{R}^{6 \times 8}$ (48 weights) and bias $b_{\text{out}} \in \mathbb{R}^6$ (6 weights), the C1 quantum circuit comprises exactly **102 trainable parameters**.
   - In the optimal C2 quantum configuration ($L=4, n=8$), there are $2 \times 8 \times 4 = 64$ circuit angles and 54 readout weights, totaling **118 trainable parameters**.
   - The classical Product-of-Experts (PoE) baseline consists of readout matrix $W \in \mathbb{R}^{6 \times 8}$ (48 weights) and bias $b \in \mathbb{R}^6$ (6 weights), totaling **54 parameters**.
   - Classical Logistic Regression comprises $6 \times 8 + 6 = \mathbf{54\text{ parameters}}$.
   - The classical Gated Head incorporates an $8 \times 8$ feature-gating matrix $A$ (64 weights), gating bias $c \in \mathbb{R}^8$ (8 weights), readout matrix $W \in \mathbb{R}^{6 \times 8}$ (48 weights), and bias $b \in \mathbb{R}^6$ (6 weights), totaling **126 parameters**.

3. **Measurement and Logit Extraction:** Quantum expectation values are evaluated on single-qubit Pauli-$Z$ observables:
   $$\langle Z_k \rangle(x) = \langle \psi(x) | U^\dagger(\theta) Z_k U(\theta) | \psi(x) \rangle \in [-1, 1]$$
   The expectation vector $\langle Z \rangle(x) \in \mathbb{R}^8$ is mapped to class logits via the trainable linear readout:
   $$z_{\text{quantum}}(x) = W_{\text{out}} \langle Z \rangle(x) + b_{\text{out}}, \quad W_{\text{out}} \in \mathbb{R}^{C \times 8}, \quad b_{\text{out}} \in \mathbb{R}^C$$

### 4. Calibrated Probability Estimation, Model Selection, and Metric Directionality

To enforce Principle P1, class logits from every model arm are independently calibrated using Platt temperature scaling on a dedicated validation split $\mathcal{D}_{\text{cal}}$:
$$P(y = k \mid x; T) = \frac{\exp(z_k(x) / T)}{\sum_{j=1}^C \exp(z_j(x) / T)}$$
where $T > 0$ is optimized by minimizing negative log-likelihood on $\mathcal{D}_{\text{cal}}$:
$$T^* = \arg\min_{T > 0} \left( -\sum_{(x, y) \in \mathcal{D}_{\text{cal}}} \ln P(y \mid x; T) \right)$$

*Development Split Allocation and Dual-Use Accounting:* In the confirmatory trial on CheXpert ($N=64,534$), the development cohort of 8,000 patients was partitioned into a training split ($n_{\text{train}}=6,400$) and a calibration/validation split ($n_{\text{cal}}=1,600$). Models were trained solely on the 6,400 training patients. The 1,600 calibration patients were used both for early-stopping checkpoint selection (monitoring validation loss with patience of 20 epochs), ranking hyperparameter configurations, and fitting the scalar Platt temperature scaling parameter $T^*$ via L-BFGS to minimize NLL on $\mathcal{D}_{\text{cal}}$. Models were not refit on the pooled development set prior to calibration; rather, the frozen checkpoint from training on $n_{\text{train}}$ was retained. Reusing $\mathcal{D}_{\text{cal}}$ for both configuration selection and temperature fitting induces a minor optimistic selection bias on development validation loss. However, out-of-sample inference on the strictly held-out cohort of 56,534 patients ($|S_{\text{dev}} \cap S_{\text{test}}| = 0$) remained entirely protected, evaluating frozen parameters once.

Model discrimination and sharpness are evaluated using test-set Negative Log-Likelihood (NLL):
$$\text{NLL}(P, y) = -\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \ln P(y_i \mid x_i; T^*)$$

We define the paired difference in per-observation negative log-likelihood as:
$$\Delta_{\text{NLL}} = \text{NLL}_{\text{quantum}} - \text{NLL}_{\text{classical}}$$
**Metric Sign Convention:** Because NLL is a loss metric (lower is superior):
- **$\Delta_{\text{NLL}} > 0$:** Indicates that the quantum model has higher loss, favoring the classical model.
- **$\Delta_{\text{NLL}} < 0$:** Indicates that the quantum model has lower loss, favoring the quantum model.
- For accuracy, sensitivity, specificity, and AUROC (higher is superior), paired differences are defined as $\Delta = \text{Metric}_{\text{quantum}} - \text{Metric}_{\text{classical}}$, where positive values favor quantum.

### 5. Cross-Validation Dependence Correction (Nadeau–Bengio)

When evaluating models via $R$-repeated $K$-fold cross-validation ($J = R \times K$ total outer folds), standard paired $t$-tests underestimate sample variance because training sets overlap across folds, violating the independence assumption. 

Let $d_j = \text{Metric}_{\text{quantum}, j} - \text{Metric}_{\text{classical}, j}$ denote the paired difference observed on fold $j \in \{1, \dots, J\}$. The sample mean difference is:
$$\bar{d} = \frac{1}{J} \sum_{j=1}^J d_j$$
The uncorrected sample variance of the differences is:
$$s^2 = \frac{1}{J - 1} \sum_{j=1}^J (d_j - \bar{d})^2$$
Under repeated cross-validation with training sample size $n_{\text{train}}$ and test sample size $n_{\text{test}}$, the correlation between error estimates across folds is bounded below by $\rho \ge \frac{n_{\text{test}}}{n_{\text{train}} + n_{\text{test}}}$. Nadeau and Bengio (2003) derived the asymptotic corrected variance estimator:
$$\text{Var}_{\text{corrected}}(\bar{d}) = \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) s^2$$
$$\text{SE}_{\text{corrected}} = \sqrt{\left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) s^2}$$
The corrected test statistic is:
$$t_{\text{corrected}} = \frac{\bar{d}}{\text{SE}_{\text{corrected}}}$$
with $J - 1$ degrees of freedom ($df = 24$ for $5 \times 5$ repeated CV). Two-sided $p$-values and $(1 - \alpha)$ confidence intervals are computed using the Student's $t$-distribution:
$$\text{CI}_{1-\alpha} = \bar{d} \pm t_{1 - \alpha/2, J-1} \cdot \text{SE}_{\text{corrected}}$$

*Assumptions and Limitations:* The Nadeau–Bengio correction provides a conservative variance adjustment that restores nominal Type I error rates under cross-validation. However, it assumes that performance differences are approximately normally distributed across folds and that training-set overlap is the sole source of dependence.

### 6. Operational Equivalence Testing (TOST)

When testing whether two models perform equivalently on large confirmatory samples ($N = 56,534$), high statistical power will detect minuscule, clinically meaningless differences. To evaluate whether models achieve practical equivalence, we formulated a Two One-Sided Tests (TOST) procedure.

Let $\delta > 0$ denote the prespecified operational average-NLL equivalence margin ($\delta = 0.010$ nats, pre-registered in git commit `61e777b`). This margin is strictly an operational statistical benchmark on expected negative log-likelihood differences ($\mathbb{E}[\Delta\ell(x, y)] \le 0.010$ nats per observation across the population). Importantly, an average negative-log-likelihood difference across a cohort does not, by itself, bound individual-patient risk ratios or worst-case clinical risk for specific subgroups. Furthermore, this margin has not been clinically validated against prospective clinical outcomes, diagnostic error rates, or expert clinical utility thresholds. Unless supported by empirical clinical validation studies linking margins to acceptable patient risk, it must remain strictly an operational statistical benchmark.

The composite null hypothesis of non-equivalence asserts that the true expected difference exceeds the operational margin:
$$H_0: \Delta_{\text{NLL}} \le -\delta \quad \text{or} \quad \Delta_{\text{NLL}} \ge +\delta$$
The alternative hypothesis of operational equivalence is:
$$H_1: -\delta < \Delta_{\text{NLL}} < +\delta$$
The TOST procedure decomposes $H_0$ into two one-sided null hypotheses:
$$H_{01}: \Delta_{\text{NLL}} \le -\delta \quad \text{tested via} \quad t_1 = \frac{\bar{d} - (-\delta)}{\text{SE}}$$
$$H_{02}: \Delta_{\text{NLL}} \ge +\delta \quad \text{tested via} \quad t_2 = \frac{+\delta - \bar{d}}{\text{SE}}$$
The overall equivalence $p$-value is the maximum of the two one-sided $p$-values:
$$p_{\text{TOST}} = \max\left( p_1, p_2 \right) = \max\left( 1 - \Phi(t_1), 1 - \Phi(t_2) \right)$$
Operational equivalence at significance level $\alpha = 0.05$ is confirmed if and only if $p_{\text{TOST}} < \alpha$, which is mathematically equivalent to the $(1 - 2\alpha) = 90\%$ two-sided confidence interval falling strictly inside $[-\delta, +\delta]$.

### 7. Mechanistic and Analytical Formulation

To explain the physical and computational mechanisms underlying observed performance, we analyzed three structural properties of variational circuits:

1. **Finite Fourier Expansion and Interaction Order (M1):** With single-qubit angle encoding $R_Y(\pi x_i)$, the expectation value of any Pauli observable decomposes into an exact multi-dimensional Fourier series [27]:
   $$\langle Z_k \rangle(x) = \sum_{\omega \in \{-1, 0, 1\}^n} c_\omega \exp(i \pi \omega^T x) = \sum_{P \in \{I, X, Y, Z\}^{\otimes n}} c_P \prod_{i=1}^n f_{P_i}(x_i)$$
   The interaction order $|P|$ is defined as the number of non-identity Pauli operators in string $P$. The coefficient energy share of interaction order $k$ is:
   $$E_k = \frac{\sum_{P: |P| = k} |c_P|^2}{\sum_{P \ne I^{\otimes n}} |c_P|^2}$$
   In product circuits without entanglers ($W_{\text{ent}} = I$), $E_k = 0$ for all $k \ge 2$ by construction. In entangled circuits, non-zero $E_k$ for $k \ge 2$ represents interaction cross-terms.

2. **Classical Fourier Basis Surrogate (M2):** To test whether the circuit's trigonometric representation confers an advantage independent of quantum simulation, we constructed classical linear models directly on the trigonometric basis:
   $$\Phi_1(x) = [\cos(\pi x_1), \sin(\pi x_1), \dots, \cos(\pi x_n), \sin(\pi x_n)]^T \in \mathbb{R}^{2n}$$
   $$\Phi_2(x) = [\Phi_1(x); \{\cos(\pi x_i)\cos(\pi x_j), \cos(\pi x_i)\sin(\pi x_j), \sin(\pi x_i)\sin(\pi x_j)\}_{i < j}] \in \mathbb{R}^{2n + 4\binom{n}{2}}$$
   Models are fitted using standard L2-regularized logistic regression.

3. **Analytic Kernel Equivalence and Geometric Distance (M3):** For product-state feature maps, the quantum fidelity kernel reduces analytically to a classical cosine kernel:
   $$K_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2 = \prod_{i=1}^n \cos^2\left(\frac{\pi(x_i - x'_i)}{2}\right)$$
   The empirical geometric difference distance between classical kernel $K_1$ and quantum kernel $K_2$ on sample size $N$ is defined by Huang et al. (2021) [10]:
   $$g(K_1 \parallel K_2) = \sqrt{\| \sqrt{K_1} K_2^{-1} \sqrt{K_1} \|_\infty}$$
   When $g(K_1 \parallel K_2) \ll \sqrt{N}$, classical kernel machines achieve comparable generalization error on any target function learnable by the quantum kernel.

---

## Evaluation Algorithms

The following three algorithms formalize our complete experimental and audit methodology:

```
Algorithm 1: Symmetric Model Evaluation and Inference Protocol
Input  : Dataset D = {(x_i, y_i)}_{i=1}^N, Model architectures M = {M_classical, M_quantum},
         Hyperparameter search grids H_classical, H_quantum with |H| = K,
         Cross-validation folds J = 25, Equivalence margin δ = 0.010
Output : Paired effect estimate d_bar, Corrected standard error SE_corrected, 
         Confidence intervals, Two-sided p-value, Equivalence p_TOST, Verdict

1  Identify unit of independence u_i (patient or participant identifier);
2  Partition D into J leak-free train/test folds {(D_train,j, D_test,j)}_{j=1}^J grouped by u_i;
3  for each fold j = 1 to J do
4      Split D_train,j into sub-train D_sub,j (80%) and validation D_val,j (20%) grouped by u_i;
5      Fit feature transformers T_j (e.g., PCA, imputation, scaling) strictly on D_sub,j;
6      for each arm m in M do
7          for each configuration h in H_m do
8              Train model m_h on T_j(D_sub,j) with early stopping monitored on D_val,j;
9              Evaluate selection loss L_val(h) on T_j(D_val,j);
10         end
11         Select optimal configuration h*_m = argmin_h L_val(h);
12         Retain frozen checkpoint m_h* trained on T_j(D_sub,j) at minimum validation loss (no pooled refit);
13         Fit scalar calibration temperature T*_m on logits of D_val,j: T*_m = argmin_T NLL(σ(z_m / T), D_val,j);
14         Transform test fold D_test,j using strictly in-fold transformer T_j;
15         Generate calibrated test probabilities P_m,j = σ(z_m(T_j(D_test,j)) / T*_m);
16         Evaluate test metric M_m,j = Metric(P_m,j, y_test,j);
17     end
18     Compute paired difference: d_j = M_quantum,j - M_classical,j;
19 end
20 Calculate mean difference: d_bar = (1/J) * sum_{j=1}^J d_j;
21 Calculate sample variance: s^2 = (1/(J-1)) * sum_{j=1}^J (d_j - d_bar)^2;
22 Apply Nadeau-Bengio correction: SE_corrected = sqrt((1/J + n_test/n_train) * s^2);
23 Compute t_stat = d_bar / SE_corrected, df = J - 1, and two-sided p_val = 2 * (1 - F_t(|t_stat|, df));
24 Evaluate TOST: t1 = (d_bar - (-δ)) / SE_corrected, t2 = (δ - d_bar) / SE_corrected;
25 Compute p_TOST = max(1 - F_t(t1, df), 1 - F_t(t2, df));
26 if p_val < 0.05 and d_bar indicates quantum superiority then Verdict = "Survives (Superior)";
27 else if p_val < 0.05 and d_bar indicates classical superiority then Verdict = "Reversed (Classical Superior)";
28 else if p_TOST < 0.05 then Verdict = "Practically Equivalent";
29 else Verdict = "Statistically Inconclusive";
30 return d_bar, SE_corrected, p_val, p_TOST, Verdict;
```

*Methodological Note on Validation Reuse:* In nested cross-validation and development protocols, $D_{\text{val}, j}$ (or $n_{\text{cal}} = 1,600$) is used both for early-stopping checkpoint selection and for fitting the single scalar Platt temperature $T^*_m$. Because the same validation split serves both roles, validation loss incurs minor optimistic selection bias. However, out-of-fold inference on $D_{\text{test}, j}$ (and the 56,534 held-out patients in confirmatory analysis) remains strictly unbiased and leak-free because test data are never accessed during training, model selection, or temperature fitting, and are scored only once after all parameters are completely frozen.

```
Algorithm 2: Systematic Benchmark Re-evaluation and Harmonization
Input  : PubMed literature corpus C_lit, 17-item checklist criteria,
         Reconstructibility thresholds (open code, accessible data)
Output : Study eligibility table, Audit classifications, Harmonized benchmark results

1  Query PubMed with ("quantum machine learning" OR "variational quantum") AND ("clinical" OR "medical");
2  Screen N_screen = 231 titles and abstracts for human health applications (yield: N_health = 129);
3  Filter for empirical quantum superiority claims on clinical outcomes (yield: N_claim = 41);
4  Assess data and pipeline accessibility against reconstructibility criteria:
5  Partition eligible studies (N_elig = 8) into:
   - Inaccessible pipelines (N_inacc = 3: unreleased 3M code, proprietary MRI radiomics, restricted omics)
   - Reconstructible studies (N_reconstruct = 5 publications, spanning K_tasks = 6 clinical tasks);
6  Score accessible study texts (N_text = 6) against 17-item binary audit checklist;
7  for each reconstructible task k = 1 to K_tasks do
8      Acquire official benchmark data from primary archive;
9      Reconstruct Author Protocol Arm:
       - Implement published feature pipeline (e.g., global PCA, random split, uncalibrated baseline);
       - Reproduce claimed published performance margin Δ_reported;
10     Construct Symmetric Protocol Arm:
       - Enforce patient/subject grouping to eliminate partition leakage;
       - Fit all transformers strictly in-fold;
       - Tune classical baselines (LR, SVM, RF, GBDT, MLP) across 16 hyperparameter configs;
       - Execute 5 x 5 repeated nested CV (J = 25 folds);
       - Compute Nadeau-Bengio corrected inference: t_NB, p_NB, 95% CI;
11     Classify outcome: Reversed (classical wins), Inconclusive (p >= 0.05), or Partition Artifact;
12 end
13 return Eligibility classifications, Checklist scores, Harmonized task results;
```

```
Algorithm 3: Confirmatory Internal Clinical Decision-Support Evaluation (AURA Workflow)
Input  : CheXpert cohort (N_total = 64,534 patients), DenseNet-121 extractor,
         Acquity triage policy T(p_1, ..., p_8), Equivalence margin δ = 0.010 nats
Output : Analysis C1, C1b, C2 results on N_test = 56,534 untouched patients

1  Filter CheXpert frontal views to exactly one frontal image per patient (highest study index);
2  Extract 8-channel finding probability vectors x_i = DenseNet(image_i);
3  Apply clinical acuity policy to derive deterministic triage target y_i in {1, ..., 6};
4  Partition cohort into Development (first n_dev = 8,000) and Confirmatory (subsequent n_test = 56,534);
5  Verify zero patient overlap: |S_dev ∩ S_test| = 0;
6  Split Development cohort into Training (n_train = 6,400) and Calibration (n_cal = 1,600) (Note: n_cal serves for early-stopping monitoring, configuration selection, and Platt scaling; no pooled refit is performed);
7  --- Analysis C1: Default Matched Optimization ---
8  Train Classical PoE, Logistic Regression, Gated Head, and VQC on n_train with matched Adam optimizer;
9  Calibrate each arm independently on n_cal via temperature scaling: T*_m = argmin_T NLL;
10 Score frozen calibrated models on 56,534 untouched test patients;
11 Compute paired difference vs PoE: ΔNLL = NLL_vqc - NLL_poe, paired t-stat, p-value;
12 --- Analysis C1b: Entanglement Ablation Across Seeds ---
13 for each seed s in {1, ..., 10} do
14     Train 7-qubit entangled circuit (CNOT ring) and product circuit (no entangler) on n_train;
15     Calibrate on n_cal and evaluate test NLL on 56,534 patients;
16     Compute per-seed difference: d_seed,s = NLL_entangled,s - NLL_product,s;
17 end
18 Evaluate seed t-test and patient-clustered bootstrap confidence interval;
19 --- Analysis C2: 36-Configuration Symmetric Search ---
20 Define 36-config search grids: VQC (layers 1-6, scale s in {0.5, 1.0, 2.0}, init SD, lr) vs PoE (lr, decay, init) vs Gated Head (lr, decay, init);
21 Train all 36 configs per arm on n_train with early stopping on n_cal; rank and select winning config h*_m = argmin_h NLL(n_cal);
22 Retain frozen weights from n_train training; fit temperature scaling T*_m on n_cal logits;
23 Score frozen calibrated models once on 56,534 untouched test patients;
24 Primary contrast (matched 36-trial budget): ΔNLL = NLL_vqc,C2 - NLL_poe,C2 = -0.00260 nats;
25 Secondary contrasts: vs default PoE C1 (-0.0013 nats) and vs Gated Head C2 (-0.0012 nats);
26 Evaluate Two One-Sided Tests (TOST) against operational equivalence margin δ = 0.010 nats;
27 return C1, C1b, and C2 master confirmatory results;
```

---

## Results

### 1. Systematic Review and Checklist Audit of Published Claims

Our screening flow identified 231 records from PubMed matching quantum clinical prediction terms. Title and abstract screening identified 129 publications addressing human health, of which 41 explicitly claimed an empirical quantum advantage on a clinical outcome. Eight publications met eligibility criteria by evaluating clinical prediction tasks on reconstructible or open data. Three studies were inaccessible for independent computational execution: Jahin et al. [20] (PMID 40865268, DOI 10.1016/j.compbiomed.2025.110976) did not release source code or tensor weights for their 3-million-parameter classical-quantum hybrid network; Chen et al. [21] (PMID 42046760, DOI 10.1117/12.3085665) evaluated PROSTATEx lesions using proprietary MRI radiomics features and private network weights; and Choi et al. [22] (PMID 41966795, DOI 10.1016/j.cmpb.2026.109343) evaluated controlled-access COVID-19 multi-omics data requiring unreleased institutional credentialing.

The remaining five reconstructible publications spanning six clinical prediction tasks were reimplemented with complete code and verified against primary datasets: Chronic Kidney Disease (Parthasarathi et al. [15], PMID 40567645; `ckd.json`), Parkinson's Voice and Pima Diabetes (Daga et al. [16], PMID 42286086; `stack_parkinsons.json` and `stack_pima.json`), Schizophrenia EEG (Aksoy & Cattan [17], PMID 38441727; `eeg_qsvm.json`), Glioma Tumor Grading (Akpinar & Oduncuoglu [18], PMID 40676161; `glioma.json` and `qkl_brats2020_official_heldout111.json`), and Heart Disease Risk (Ghisoni et al. [19], PMID 41513819; `studies.py: heart_qnn`).

Six publications with accessible texts could be assessed against our 17-item checklist (**Table 1**). Five of the six assessable studies met checklist criteria for being at risk of asymmetric evaluation: utilizing an untuned classical comparator, fitting dimensionality reduction (e.g., PCA) before data partitioning, allowing repeated patient recordings or temporal epochs to cross validation splits, or selecting checkpoints based on test-set metrics. Crucially, all six studies asserted superiority without paired hypothesis testing or uncertainty intervals on performance differences.

**Table 1 | Methodological audit of eligible published quantum machine learning studies in digital health.**

| Study & Clinical Task | Publication & Data Access | Sample Size ($n$) & Split Design | Qubits & Architecture | Reported Quantum Claim | Asymmetric Risk Identified | Symmetric Evaluation Result ($\Delta$, 95% CI) | Statistical Test & $p$-value | Prespecified Outcome Classification |
|---|---|---|:---:|---|:---:|---|:---:|---|
| **Parthasarathi et al.**<br>Chronic Kidney Disease | *PeerJ CS* 2025 [15]<br>PMID 40567645 | $n=400$ patients<br>80/20 train/test | 8 qubits<br>VQC (SAE+LASSO) | VQC 99.2% accuracy beats classical (RF 90.7%, RBF Net 96.3%) | **YES** (Untuned baseline, global preprocessing) | Classical 97.35% vs Quantum 87.70%<br>$\Delta = -9.65\%$ ($-14.92\%$ to $-4.38\%$) | Nadeau–Bengio corrected $t$<br>$t = -3.78, df = 24, p = 0.000915$ | **Advantage Reversed** (Classical baseline significantly superior) |
| **Daga et al.**<br>Parkinson's Disease Voice | *Sci Rep* 2026 [16]<br>PMID 42286086 | $n=195$ recordings<br>31 subjects; recording split | 10 qubits<br>QKL Stacking | QKL Stacking 81.3% specificity beats SVM (58.5%) | **YES** (Global PCA, recording-level split) | Symmetric nested CV: $\Delta = +0.33\%$ ($-24.97\%$ to $+25.64\%$); Factorial leak-free: $\Delta = -0.3\%$ ($-2.1\%$ to $+1.5\%$) | Nadeau–Bengio $t = +0.03, df = 24, p = 0.9785$;<br>Factorial $t$ $p = 0.72$ | **Statistically Inconclusive** (Advantage not reproduced under subject grouping) |
| **Daga et al.**<br>Pima Diabetes Mellitus | *Sci Rep* 2026 [16]<br>PMID 42286086 | $n=768$ patients<br>70/30 train/test | 8 qubits<br>QKL Stacking | QKL Stacking 62.1% sensitivity beats SVM (55.3%) | **YES** (Global PCA, untuned comparator) | Symmetric nested CV: $\Delta = -5.30\%$ ($-11.00\%$ to $+0.39\%$); Factorial in-fold + tuned: $\Delta = +0.8\%$ ($-1.1\%$ to $+2.7\%$) | Nadeau–Bengio $t = -1.92, df = 24, p = 0.0666$;<br>Factorial $t$ $p = 0.36$ | **Statistically Inconclusive** (Advantage not reproduced under baseline tuning) |
| **Aksoy & Cattan**<br>EEG Schizophrenia | *J Med Syst* 2024 [17]<br>PMID 38441727 | $N=84$ participants (45 schizophrenia, 39 controls)<br>84 whole recordings (1,008 5-s epochs); random vs subject split | 10 qubits<br>Quantum SVM (Pauli-$Z$) | Quantum SVM achieves 100.0% test accuracy | **YES** (Epoch split contamination / temporal leakage) | Subject-grouped leak-free: **52.1%** mean accuracy (descriptive 95% split range: 43.2%–61.0%; chance level); 25-fold nested CV: $\Delta = -7.60\%$ ($-23.01\%$ to $+7.81\%$) | Nadeau–Bengio corrected $t$<br>$t = -1.02, df = 24, p = 0.3187$ | **Advantage Not Reproduced** (100% claim is a partition leakage artifact) |
| **Akpinar & Oduncuoglu**<br>Glioma Grading (TCGA & BraTS) | *Sci Rep* 2025 [18]<br>PMID 40676161 | $n=839$ (TCGA)<br>$n=369$ (BraTS 2020) | 5 qubits (TCGA)<br>10 qubits (BraTS) | VQC 83.0% accuracy beats classical baselines (75–82%) | **YES** (Feature selection on full dataset) | TCGA nested CV: $\Delta = +0.19\%$ ($-1.22\%$ to $+1.60\%$); BraTS held-out: RBF 0.9625 vs QKL 0.9407 ($\Delta = -0.0217$, $-0.0543$ to $+0.0109$) | Nadeau–Bengio $t = +0.28, df = 24, p = 0.7824$;<br>DeLong test $Z = 1.31, p = 0.1912$ | **Statistically Inconclusive** (Classical RBF matches on TCGA, higher point AUROC on BraTS) |
| **Ghisoni et al.**<br>Heart Disease Risk | *Sci Rep* 2026 [19]<br>PMID 41513819 | $N=918$ patients (low-data regime)<br>10% training fraction ($n_{\text{train}}=74, n_{\text{test}}=184$) | 11 features<br>1-layer QNN | 1-layer QNN 82.72% beats classical NN (80.51%) at 10% data | **YES** (Early stopping on test cost, untuned NN) | Symmetric nested CV with validation stopping: Classical NN 82.6% vs Quantum 81.5%<br>$\Delta = -1.10\%$ (paired $t$ 95\% CI $-3.20\%$ to $+1.00\%$; NB 95\% CI $-18.45\%$ to $+16.25\%$) | Paired Student's $t$ ($t = -1.04, df = 24, p = 0.310$);<br>Nadeau–Bengio ($t = -0.13, p = 0.897$) | **Statistically Inconclusive** (Advantage vanishes with validation-based stopping; wide CI $[-18.45\%, +16.25\%]$) |
| **Jahin et al.**<br>Cardiovascular Disease | *Comput Biol Med* 2025 [20]<br>PMID 40865268 | $n=1,025$ patients<br>80/20 train/test | 4 qubits<br>KACQ-DCNN (3M params) | KACQ-DCNN 92.0% accuracy beats CatBoost (86.4%) | **YES** (Untuned baseline, SMOTE pre-split) | Not Reimplemented (No code released; 3-million-parameter classical-quantum network) | — | *Eligible, Non-Reconstructible* (Inaccessible source code) |
| **Chen et al.**<br>Prostate Lesions (PROSTATEx) | *Proc SPIE* 2026 [21]<br>PMID 42046760 | $n=330$ lesions<br>MRI radiomics | 8 qubits<br>QSVM | QSVM 0.799 AUROC beats SVM-RBF (0.608) | Unclear (Proprietary features / weights) | Not Reimplemented (Proprietary radiomics extraction pipeline and weights unreleased) | — | *Eligible, Non-Reconstructible* (Proprietary radiomics pipeline) |
| **Choi et al.**<br>COVID-19 Biomarkers | *Comput Methods Programs Biomed* 2026 [22]<br>PMID 41966795 | $n=128$ patients<br>Multi-omics profiles | 6 qubits<br>QSVM | QSVM comparable or exceeds classical SVM | Unclear (Data inaccessible) | Not Reimplemented (Controlled-access multi-omics data requiring institutional credentialing) | — | *Eligible, Non-Reconstructible* (Restricted omics access) |

<div class="figure-container">
    <img src="file:///E:/AURA/aura-main/docs/figures/fig_v6_forest.png" alt="Figure 1" style="width:100%; max-width:900px; display:block; margin:auto;" />
    <p class="figure-caption"><strong>Figure 1 | Systematic benchmark comparison across evaluation regimes.</strong> 
    <strong>(a)</strong> Paired per-study likelihood differences (&Delta;NLL) in the internal decision-support prototype across comparator choices, calibration protocols, out-of-sample partitions, and confirmatory cohorts. Under default matched optimization (C1), the quantum circuit is materially worse (&Delta;NLL = +0.0205 nats). Under 36-configuration symmetric tuning (C2), the observed effect (&Delta;NLL = -0.0026 nats) falls entirely within the prespecified &plusmn;0.010-nat operational equivalence band (gray region).
    <strong>(b)</strong> Re-tested published claims of quantum benefit across literature benchmarks. Hollow markers indicate original reported margins; solid markers indicate re-evaluated performance differences with Nadeau–Bengio 95% confidence intervals under symmetric evaluation. Zero of the six benchmark claims survive under symmetric evaluation.</p>
</div>

### 2. External Clinical Benchmark Re-evaluations

Under our pre-registered survival criterion, a published advantage claim is defined to survive if and only if the quantum model achieves statistically significant superiority ($\Delta > 0, p < 0.05$ after multiple testing correction) over tuned classical baselines under leak-free nested cross-validation. Under this criterion, **zero of the six benchmark claims survived** (**Figure 1b**).

1. **Advantage Reversed (Classical Model Significantly Superior):**
   * *Chronic Kidney Disease (Parthasarathi et al.):* Parthasarathi et al. reported that a variational quantum classifier attained 99.2% accuracy on the UCI CKD dataset ($n=400$), beating Random Forest (90.7%) and an RBF Network (96.3%). Auditing the workflow revealed that standard preprocessing (imputation and MinMax scaling) had been applied globally across all 400 patients prior to splitting, and classical comparators were evaluated without hyperparameter tuning. Under symmetric 25-fold nested cross-validation with in-fold preprocessing, tuned classical models achieved 97.35% accuracy, whereas the quantum model reached 87.70%. The classical comparator outperformed the quantum circuit by **9.65 percentage points** (95% CI 4.38% to 14.92%; Nadeau–Bengio $t = -3.78, df = 24, p = 0.000915$). The claim was reversed.

2. **Statistically Inconclusive Under Matched Evaluation:**
   * *Parkinson's Disease Voice (Daga et al.):* Daga et al. reported that Quantum Kernel Learning (QKL) stacking attained 81.3% specificity on UCI Parkinson's Voice data, outperforming an unoptimized RBF-SVM (58.5%). Under the author protocol, the apparent gap was reproduced ($\Delta = +4.8\%, p = 0.012$). However, the dataset consists of 195 recordings from only 31 human participants. Under patient-grouped nested cross-validation where all recordings from a participant were restricted to a single fold, the performance difference became statistically undetectable ($\Delta = +0.33\%$, 95% CI $-24.97\%$ to $+25.64\%$, $p = 0.9785$).
   * *Pima Diabetes Mellitus (Daga et al.):* On 768 patients, claimed sensitivity gains (62.1% vs 55.3%) reproduced under author settings ($\Delta = +9.3\%, p = 0.009$). Under symmetric nested cross-validation with baseline hyperparameter optimization, the point difference favored the classical model ($\Delta = -5.30\%$, 95% CI $-11.00\%$ to $+0.39\%$, $p = 0.0666$).
   * *Glioma Tumor Grading (Akpinar & Oduncuoglu):* On TCGA clinical and genomic mutation features ($n=839$), the quantum classifier matched classical baselines without statistical separation ($\Delta = +0.19\%$, 95% CI $-1.22\%$ to $+1.60\%$, $p = 0.7824$). Furthermore, on 369 official neuroimaging subjects from BraTS 2020, Quantum Kernel Learning achieved a test AUROC of 0.9407 on held-out subjects ($n=111$), while a standard classical RBF kernel achieved **0.9625** ($\Delta\text{AUROC} = -0.0217$, 95% CI $-0.0543$ to $+0.0109$; DeLong $Z = 1.31, p = 0.1912$).
   * *Heart Disease Low-Data Regime (Ghisoni et al.):* Ghisoni et al. investigated whether quantum neural networks excel under extreme data scarcity, reporting an advantage at a 10% training data fraction on UCI Heart Disease ($N=918$). When early stopping was monitored on test-set cost (as in the original paper), an apparent advantage appeared ($\Delta = +2.2\%$). Under symmetric 5-fold cross-validation across 25 outer folds ($N=918$, with training pool $n=734$ subsampled to $n_{\text{train}}=74$ and evaluated on $n_{\text{test}}=184$, test/train ratio 2.49), the advantage vanished when early stopping was monitored on validation loss ($\Delta = -1.10\%$, uncorrected paired $t$ 95% CI $-3.20\%$ to $+1.00\%$, paired $t = -1.0400, df = 24, p = 0.3100$; under Nadeau–Bengio dependency correction, $t = -0.1309, df = 24, p = 0.8970$, corrected 95% CI $-18.45\%$ to $+16.25\%$). This wide uncertainty interval of approximately $-18.45$ to $+16.25$ percentage points demonstrates that the result is **statistically inconclusive**, not proof that the methods are equivalent. In small-to-moderate sample sizes under dependency correction, failure to reject the null cannot be construed as demonstrating equivalence, which would require adequate statistical power to rule out meaningful differences.

3. **Advantage Not Reproduced Due to Partition Leakage:**
   * *Schizophrenia EEG (Aksoy & Cattan):* Aksoy & Cattan reported that a Quantum SVM achieved **100.0% accuracy** in distinguishing adolescent schizophrenia patients from healthy controls. Under random epoch splitting across 1,008 segmented 5-second EEG epochs derived from $N=84$ participants (45 cases, 39 controls; 53.57% baseline prevalence), the 100% claim reproduced (mean accuracy 98.6%, with 34.5% of splits hitting 100.0%). However, because participant-specific EEG rhythms are persistent over time, random epoch splitting allows the classifier to memorize individual subject identities rather than pathology. When partitions were grouped strictly by participant (so that no patient's epochs crossed splits), accuracy collapsed to a descriptive mean of **52.1%** (SD 7.0%, empirical 95% split range 43.2%–61.0%, with 0% of splits reaching >75%), matching the 53.57% majority-class prevalence. Formal 25-fold nested cross-validation with Nadeau–Bengio correction confirmed no difference from classical baselines ($t = -1.0184, df = 24, p = 0.3187$).

**Table 2 | Pre-registered factorial analysis isolating the evaluation mechanisms of apparent quantum advantage.**

| Dataset & Study | Factorial Design Space | Cell Configuration | Mean Quantum Advantage $\Delta$ (95% CI) | Corrected $p$-value | Finding |
|---|---|---|:---:|:---:|---|
| **Parkinson's Voice**<br>(Daga et al. [16]) | 2 × 2 × 2 Factorial:<br>• PCA Scope (Global vs In-Fold)<br>• Split Unit (Recording vs Subject)<br>• Baseline (Untuned vs Tuned) | **Author Protocol:**<br>Global PCA × Recording Split × Untuned SVM | **+4.8%** (+1.2% to +8.4%) | $p = 0.012$ | Apparent advantage reproduced under author design. |
| | | In-Fold PCA × Recording Split × Untuned SVM | +1.1% (−1.0% to +3.2%) | $p = 0.29$ | Fixing PCA leakage eliminates significance. |
| | | Global PCA × Subject Grouped × Untuned SVM | +0.7% (−1.1% to +2.5%) | $p = 0.41$ | Subject grouping eliminates significance. |
| | | **Symmetric Protocol:**<br>In-Fold PCA × Subject Grouped × Tuned SVM | **−0.3%** (−2.1% to +1.5%) | $p = 0.72$ | Advantage completely vanishes. |
| **Pima Diabetes**<br>(Daga et al. [16]) | 2 × 2 Factorial:<br>• PCA Scope (Global vs In-Fold)<br>• Baseline (Untuned vs Tuned) | **Author Protocol:**<br>Global PCA × Untuned SVM | **+9.3%** (+2.4% to +16.2%) | $p = 0.009$ | Apparent advantage reproduced under author design. |
| | | **Symmetric Protocol:**<br>In-Fold PCA × Tuned SVM | **+0.8%** (−1.1% to +2.7%) | $p = 0.36$ | Tuning classical baseline removes advantage. |
| **Schizophrenia EEG**<br>(Aksoy & Cattan [17]) | Partition Scheme:<br>Random Epoch vs Subject-Grouped<br>($N=84$ participants, 1,008 epochs) | Random Epoch Splitting (Author Design) | **98.6% Accuracy** (34.5% reach 100%) | $p < 10^{-6}$ | 100% accuracy reproduced via temporal leakage. |
| | | Subject-Grouped Splitting (Leak-Free) | **52.1% Mean Accuracy** (empirical 95% range: 43.2%–61.0%; 0% reach >75%) | Prevalence: 53.6% (45/84)<br>Nadeau–Bengio CV: $p = 0.3187$ | Advantage is entirely a partition artifact. |

### 3. Clinical Decision-Support Prototype: Sensitivity to Evaluation Choices

In our chest radiograph triage prototype, four backends were evaluated on the task of mapping finding probabilities extracted by a DenseNet-121 to six clinical acuity categories:
- **Calibration Asymmetry:** The system's initial published claim asserted an expected calibration error (ECE) of 0.020 for the quantum circuit versus 0.276 for classical Product-of-Experts. Auditing the codebase revealed that temperature scaling had been applied exclusively to the quantum arm. When both arms were calibrated independently on validation data, the raw test calibration gap of 0.139 shrank to 0.037. **More than 73% of the reported calibration advantage was generated entirely by one-sided calibration.**
- **Comparator Selection:** When models were fitted under author-preferred optimizers, rankings shifted depending on which baseline was evaluated: the quantum circuit beat Product-of-Experts ($p = 0.006$), matched the Gated Head ($p = 0.052$), and lost to Logistic Regression ($\Delta\text{NLL} = +0.054$, Holm $p = 0.036$).
- **Optimizer Standardization:** When all arms received matched Adam optimization with learning rates selected on validation loss, the likelihood gap vs Product-of-Experts shrank to $\Delta\text{NLL} = +0.0033$ ($p = 1.0$), with a 90% CI ($[-0.0270, +0.0337]$) too wide to assert equivalence.
- **Cross-Institutional Transfer:** When models trained on MIMIC-CXR (Boston) were transferred frozen to CheXpert (Stanford, $n=4,800$), the quantum circuit finished last of all four arms ($\Delta\text{NLL} = +0.1589$ vs PoE, 95% CI $0.1422$ to $0.1757$; $+0.1967$ vs Logistic Regression, 95% CI $0.1785$ to $0.2149$; **Figure 1a**).

**Table 3 | Pre-registered confirmatory evaluation across 56,534 untouched clinical patients.**

| Architecture / Model Arm | Trainable Parameters | Test Accuracy (95% Jeffreys CI) | Negative Log-Likelihood (NLL) | Expected Calibration Error (ECE, 95% Boot CI) | Macro-AUROC | Paired $\Delta\text{NLL}$ vs PoE (90% CI, 95% CI) | Statistical Test & $p$-value | Pre-Registered Reading |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Classical Product-of-Experts** | 54 | **0.5443** (0.5402, 0.5484) | **1.1157** | 0.0131 (0.0100, 0.0172) | **0.7534** | Reference Baseline | — | — |
| **Multinomial Logistic Regression** | 54 | 0.5415 (0.5374, 0.5456) | **1.1152** | 0.0128 (0.0105, 0.0171) | **0.7534** | $-0.0005$ (−0.0010, +0.0001; −0.0011, +0.0002) | Paired $t$ $p = 0.28$;<br>TOST $p < 10^{-6}$ | Practically Equivalent ($\Delta < 0.010$) |
| **Classical Gated Head** | 126 | 0.5398 (0.5357, 0.5439) | 1.1165 | 0.0085 (0.0071, 0.0130) | **0.7553** | $+0.0008$ (0.0000, 0.0017; −0.0002, 0.0019) | Paired $t$ $p = 4.9 \times 10^{-7}$;<br>TOST $p < 10^{-6}$ | Practically Equivalent ($\Delta < 0.010$) |
| **Variational Quantum Circuit** | 102 | 0.5371 (0.5330, 0.5412) | 1.1362 | **0.0076** (0.0051, 0.0119) | 0.7436 | $\mathbf{+0.0205}$<br>90% CI: (0.0189, 0.0221)<br>95% CI: (0.0186, 0.0224) | Paired $t = 21.03, p = 7.65 \times 10^{-98}$;<br>TOST $p = 1.000$ | **Quantum Materially Worse** |

### 4. Confirmatory Evaluation Across 56,534 Untouched Patients

To overcome power limitations, we executed our pre-registered confirmatory evaluation on **56,534 untouched CheXpert patients** (one frontal radiograph per patient; zero overlap with development data; **Table 3**). Selecting one frontal study per patient establishes patient-level independence by construction, eliminating intra-patient clustering.

*Scope of Clinical Claim:* We emphasize that this confirmatory evaluation assesses how faithfully each backend architecture emulates an additive clinical triage rule deterministically derived from DenseNet radiological finding probabilities (Supplementary Note 2). While evaluating 56,534 patients provides extraordinary statistical precision to evaluate model representations and optimization stability, it evaluates algorithmic rule emulation rather than direct clinician diagnostic agreement, clinical efficacy, or prospective patient health outcomes. Furthermore, because CheXpert represents a single academic medical center, single-center patient independence does not guarantee multi-center clinical generalizability; this motivated our multi-center transfer evaluation (MIMIC-CXR $\to$ CheXpert), where quantum performance degraded severely.

- **Analysis C1 (Default Matched Protocol):** The quantum circuit finished lowest in accuracy (53.71%), likelihood (1.1362), and discrimination (0.7436). The paired likelihood difference vs Product-of-Experts was $\Delta\text{NLL} = \mathbf{+0.0205}$ nats ($95\%$ CI $0.01858$ to $0.02240$, $t = 21.034, p = 7.65 \times 10^{-98}$; TOST $p = 1.000$). The outcome fell into the pre-registered reading: **"Quantum materially worse."**
- **Analysis C1b (Entanglement Ablation):** Across ten random initialization seeds on the 56,534 patients, removing CNOT entanglement gates improved likelihood: $\Delta\text{NLL} = \text{NLL}_{\text{entangled}} - \text{NLL}_{\text{product}} = \mathbf{+0.01037}$ nats (95% bootstrap CI $0.00934$ to $0.01138$, $t = 4.88, p = 0.00127$). Unentangled circuits outperformed entangled circuits in 9 of 10 seeds, confirming: **"The entanglement penalty replicates on untouched patients."**
- **Analysis C2 (36-Configuration Symmetric Search & Configuration Transparency):** To ensure evaluation symmetry under capacity optimization, both quantum and classical architectures received an identical search budget of **36 configurations** trained on the 6,400 development cases and selected strictly by minimum NLL on the 1,600 calibration patients.
  * *Winning Configurations and Parameter Capacities:*
    - **Quantum Circuit:** 4 layers ($L=4$), encoding scale $s=0.5$, initialization $\text{sd}=0.01$, learning rate $\eta=0.01$. Total parameters: **118** (64 variational angles + 54 linear readout weights). Calibration NLL: 1.1355; calibration temperature $T = 1.1354$. Test NLL on 56,534 held-out patients: **1.1144**; test accuracy: 54.49%; test ECE: 0.0051.
    - **Product-of-Experts (PoE):** Learning rate $\eta=0.05$, weight decay $\lambda=10^{-4}$, initialization $\text{sd}=0.01$. Total parameters: **54**. Calibration NLL: 1.1369; calibration temperature $T = 1.0751$. Test NLL on 56,534 held-out patients: **1.1170**; test accuracy: 54.34%; test ECE: 0.0089.
    - **Nonlinear Gated Head:** Learning rate $\eta=0.1$, weight decay $\lambda=10^{-4}$, initialization $\text{sd}=0.01$. Total parameters: **126**. Calibration NLL: 1.1357; calibration temperature $T = 1.0743$. Test NLL on 56,534 held-out patients: **1.1155**; test accuracy: 54.43%; test ECE: 0.0068.
  * *Explaining the C1 vs C2 PoE Baseline Difference:* In Analysis C1, the reference PoE model was fitted under default settings ($\eta=0.01, \lambda=0, \text{sd}=0.1$) chosen from a 3-point learning-rate grid, achieving test NLL **1.1157** ($T=1.1017$). In Analysis C2, PoE was tuned across 36 combinations of learning rate ($\{0.003, 0.01, 0.05, 0.1, 0.3\}$), weight decay ($\{0.0, 10^{-5}, 10^{-4}, 10^{-3}\}$), and initialization standard deviation ($\{0.01, 0.1, 1.0\}$). The configuration selected on the 1,600 calibration cases was configuration 17 ($\eta=0.05, \lambda=10^{-4}, \text{sd}=0.01$), which attained marginally better calibration NLL ($1.1369$ vs $1.1371$), but yielded test NLL **1.1170** on the 56,534 held-out patients. The $+0.0013$-nat shift between C2 PoE (1.1170) and C1 PoE (1.1157) reflects standard finite-sample validation selection variance on $n=1,600$ development cases.
  * *Primary and Secondary Contrasts:*
    - **Primary C2 Contrast (Matched 36-Trial Search Budget):** Evaluating the selected C2 quantum model against the selected C2 classical PoE model under the exact same 36-configuration search protocol yields $\Delta\text{NLL} = 1.1144 - 1.1170 = \mathbf{-0.00260}$ nats (90% CI $[-0.00343, -0.00176]$, 95% CI $[-0.00359, -0.00160]$, $t = -5.10, p = 3.07 \times 10^{-7}$; **Figure 3b**). While statistically detectable due to the high precision of 56,534 independent patients, both the 90% and 95% confidence intervals fall entirely within the prespecified operational equivalence margin ($\pm 0.010$ nats; Two One-Sided Tests $p_{\text{TOST}} < 10^{-6}$).
    - **Secondary Reference Contrast (vs Default C1 PoE):** When compared against the fixed C1 default PoE baseline (1.1157), the difference shrinks further to $\Delta\text{NLL} = 1.1144 - 1.1157 = \mathbf{-0.0013}$ nats, also lying comfortably within the $\pm 0.010$-nat operational equivalence band.
    - **Nonlinear Head Contrast (vs Tuned Gated Head):** Compared against the tuned nonlinear Gated Head (1.1155), the difference narrows to only $\Delta\text{NLL} = 1.1144 - 1.1155 = \mathbf{-0.0012}$ nats (95% CI $[-0.00192, -0.00041]$, $t = -3.05, p = 0.0025, p_{\text{TOST}} < 10^{-6}$).
    All contrasts confirm the pre-registered verdict: **"Detectable difference, quantum better, smaller than the margin; practically equivalent."**

<div class="figure-container">
    <img src="file:///E:/AURA/aura-main/docs/figures/fig_v6_confirmatory.png" alt="Figure 3" style="width:100%; max-width:900px; display:block; margin:auto;" />
    <p class="figure-caption"><strong>Figure 3 | Pre-registered confirmatory evaluation across 56,534 untouched patients.</strong> 
    <strong>(a)</strong> Analyses C1 and C1b under default matched optimization. The quantum circuit performs materially worse than classical baselines (&Delta;NLL = +0.0205 nats; blue points), and entangled circuits underperform unentangled product circuits across ten random seeds (&Delta;NLL = +0.0104 nats).
    <strong>(b)</strong> Analysis C2 under 36-configuration symmetric hyperparameter tuning. While the quantum model achieves a small point advantage over PoE (&Delta;NLL = -0.0026 nats) and Gated Head (&Delta;NLL = -0.0012 nats), both the 90% (dark red) and 95% (light red) confidence intervals lie strictly inside the prespecified &plusmn;0.010-nat operational equivalence region (gray band).
    <strong>(c)</strong> Sensitivity analysis of Two One-Sided Tests (TOST) p-values across varying operational margins &delta;. Equivalence is robustly confirmed (p &lt; 0.05) across all margins &ge; 0.005 nats.</p>
</div>

---

## Mechanistic Analysis

Three pre-registered mechanistic experiments explain why quantum circuits provide no practical utility on this clinical task (**Figure 2**):

<div class="figure-container">
    <img src="file:///E:/AURA/aura-main/docs/figures/fig_v6_mechanism.png" alt="Figure 2" style="width:100%; max-width:900px; display:block; margin:auto;" />
    <p class="figure-caption"><strong>Figure 2 | Mechanistic dissection of circuit expressivity and kernel geometry.</strong> 
    <strong>(a)</strong> Fourier interaction-order coefficient energy spectrum (M1). In the unentangled product circuit, 100.0% of coefficient energy is concentrated in linear order 1. In the entangled circuit (CNOT ring), 100.0% of non-trivial energy and 99.99% of logit variance are dispersed across high-order interaction cross-terms (&ge; 2).
    <strong>(b)</strong> Quantum kernel bandwidth sweep on glioma neuroimaging (M3). Peak kernel-target alignment occurs at bandwidth 0.25 across all folds, where effective dimension stabilizes. The geometric distance g(K_RBF || K_Q) = 1.27 &ll; &radic;N = 44.72 confirms classical RBF kernels achieve comparable sample complexity.</p>
</div>

1. **Interaction-Order Energy Dispersion (M1):** In the unentangled product circuit, interaction order is strictly $\le 1$ by construction (0.0% cross-channel energy). In the entangled circuit, **100.0% of non-trivial coefficient energy** and **99.99% of logit variance** are concentrated in interaction orders $\ge 2$ (**Figure 2a**). Because the clinical triage target is fundamentally an additive scoring rule of radiographic signs, the high-order cross-terms generated by entanglement represent unhelpful capacity that increases variance without reducing bias.
2. **Classical Fourier Basis Surrogate (M2):** A classical L2-regularized logistic regression fitted directly on the trigonometric basis ($\cos(\pi x_i), \sin(\pi x_i)$) on the 56,534 untouched patients achieved NLL 1.1123 at order 1 (14 features) and NLL 1.1095 at order 2 (112 features), outperforming the quantum circuit (1.1362) by $\Delta\text{NLL} = 0.0238$ and $0.0267$ nats, respectively ($p < 10^{-4}$). Classical convex optimization of the exact same trigonometric features beats the quantum circuit, indicating that the observed gap reflects classical optimization efficiency rather than representational capacity.
3. **Analytic Kernel Equivalence and Geometric Distance (M3):** On unentangled product states, the quantum fidelity kernel analytically equals the classical product-cosine kernel ($3.11 \times 10^{-15}$ max discrepancy vs statevector simulation). The empirical geometric distance evaluated on a 2,000-sample split between the quantum kernel and a classical RBF kernel is $g(K_{\text{RBF}} \parallel K_Q) = 1.2712 \ll \sqrt{N} \approx 44.72$. Under the geometric difference theorem of Huang et al. [10], classical RBF kernels achieve comparable sample complexity on any target learnable by this quantum kernel (**Figure 2b**).
4. **Sensitivity Positive Control:** On a synthetic quadratic target (T2b: $x^T M x + a^T x$), both the quantum circuit (NLL 0.2515) and Gated Head (NLL 0.2560) significantly outperformed linear logistic regression (NLL 0.3155, Holm $p < 10^{-4}$), demonstrating that our apparatus reliably detects nonlinear advantages when present.

---

## Discussion

Across five reimplemented published studies evaluating six clinical prediction tasks and an internal 56,534-patient decision-support prototype, empirical claims of quantum advantage did not survive symmetric evaluation. In every setting investigated, apparent quantum superiority was traced to methodological asymmetries: evaluating untuned classical baselines, calibrating only the quantum arm, or scattering repeated recordings from the same human subject across cross-validation splits. When these asymmetries were corrected, apparent quantum gains either reversed in classical favor, became statistically indistinguishable, or collapsed to chance.

These findings establish an essential distinction in biomedical data science between **statistical detectability on large samples** and **clinical or operational superiority**, as well as the crucial distinction between **inconclusive null results** and **proven equivalence**. In Analysis C2, the massive sample size of 56,534 paired observations provided the statistical power to detect a tiny likelihood difference ($\Delta\text{NLL} = -0.0026$ nats, $p = 3.07 \times 10^{-7}$). In traditional reporting, this vanishingly small $p$-value would be cited as definitive evidence of a quantum advantage. However, because both the 90% and 95% confidence intervals fell entirely within our prespecified operational equivalence margin ($\pm 0.010$ nats; $p_{\text{TOST}} < 10^{-6}$), the result confirms practical equivalence, not transformative capability. When classical models were expanded to include standard nonlinear capacity (Gated Head), the difference narrowed further to 0.0012 nats ($<0.12\%$ likelihood shift).

In contrast, on external benchmark tasks, wide corrected confidence intervals—such as the 95% Nadeau–Bengio interval of approximately $-18.45$ to $+16.25$ percentage points on the heart disease low-data regime—demonstrate that several external comparisons are **statistically inconclusive**, not proof that the methods are equivalent. In underpowered studies, failure to reject the null cannot establish equivalence. Equivalence requires prespecified equivalence margins and adequate sample size to rule out practically meaningful differences.

The broader implication for digital health extends beyond quantum computing. Any novel machine learning architecture—whether a foundation model adapter, state-space model, or non-standard loss function—can appear superior if evaluated asymmetrically against neglected baselines. The 17-item checklist and symmetric protocol presented here provide clinical AI researchers, reviewers, and journal editors with an objective, code-verifiable framework to verify that claimed performance gains reflect genuine algorithmic advances rather than asymmetric care.

### Limitations

1. **Carefully Scoped Generalizability:** Five reimplemented publications spanning six clinical prediction tasks, alongside our internal prototype, support a carefully bounded conclusion regarding shallow variational quantum circuits (4 to 12 qubits), angle encodings, and fidelity kernels on clinical tabular and imaging features. They do **not** support a universal claim that quantum machine learning can never offer an advantage in healthcare. Regimes outside our scope—including fault-tolerant quantum algorithms with asymptotic speedups, high-dimensional native quantum sensor representations, non-commuting Hamiltonian feature encodings, or molecular docking and quantum chemistry where classical simulation is fundamentally intractable—remain unaddressed by our empirical data.
2. **Clinical Decision Targets:** The chest radiograph triage target was derived from report labeler heuristics and clinical decision rules rather than prospective clinician consensus or long-term patient health outcomes.
3. **Single-Center Institutional Dependence:** While selecting one study per patient establishes patient-level independence by construction and eliminates intra-patient clustering, CheXpert represents a single academic medical center (Stanford Hospital). Individual patient observations share institutional scanner protocols and demographics; cross-institutional generalizability was independently evaluated under our multi-center transfer arm (MIMIC-CXR to CheXpert, $n=4,800$), where the quantum model finished last.
4. **Operational Equivalence Margin Justification:** The equivalence margin of $\pm 0.010$ nats is strictly an **operational statistical benchmark** on expected negative log-likelihood differences ($\mathbb{E}[\Delta\ell] \le 0.010$ nats per prediction across the cohort), chosen prior to test unblinding. An average log-likelihood difference does not, by itself, bound individual-patient risk ratios or worst-case clinical risk for specific patients or vulnerable subgroups. Moreover, the margin has **not been clinically validated** against prospective patient outcomes, diagnostic error rates, or expert clinical utility thresholds. Unless supported by empirical clinical validation studies linking margins to acceptable patient health outcomes, it must remain explicitly interpreted as an operational statistical benchmark. Sensitivity analysis confirms that practical equivalence holds across alternative margins from $\pm 0.020$ down to $\pm 0.005$ nats ($p_{\text{TOST}} < 10^{-4}$).
5. **Literature Accessibility:** Reimplementation was restricted to studies with open data and reconstructible pipelines; three eligible publications were non-reconstructible due to unreleased code or proprietary features.
6. **Hardware and Simulation Boundary:** Empirical quantum evaluations were conducted at 5 to 11 qubits under exact classical statevector simulation. Prior physical noise simulations indicate that physical hardware decoherence further widens the classical advantage.

### Clinical and Scientific Implications

Translational medicine requires rigorous evidence standards before deploying complex computational pipelines. When machine learning models require specialized quantum hardware or simulation runtimes orders of magnitude larger than classical convex models, the burden of proof must demonstrate meaningful, leak-free clinical improvement. Adopting pre-registered evaluation protocols, matched tuning budgets, and equivalence testing will protect digital health from premature adoption of unverified computational architectures.

---

## Conclusion

When evaluated under symmetric protocols that match hyperparameter search budgets, enforce independent calibration, eliminate temporal partition leakage, and employ rigorous statistical inference, empirical claims of quantum machine learning advantage across six published digital health benchmarks and an internal 56,534-patient clinical decision-support cohort do not survive. Rather than demonstrating quantum superiority, apparent advantages reflect baseline undertuning, one-sided calibration, and partition contamination. Rigorous symmetric evaluation must become standard practice in translational biomedical artificial intelligence.

---

## Declarations

### Data Availability
MIMIC-CXR-JPG (v2.0.0) is available on PhysioNet (`https://physionet.org/content/mimic-cxr-jpg/2.0.0/`) under Credentialed Health Data License 1.5.0. CheXpert-v1.0-Small is available via the Stanford Center for Artificial Intelligence in Medicine and Imaging (`https://aimi.stanford.edu/`) under the Stanford Research Data Agreement. BraTS 2020 is accessible via Synapse (`https://www.synapse.org/#!Synapse:syn25829420`). Benchmark cohorts (UCI CKD, Parkinson's Voice, Pima Diabetes, Heart Disease; MSU Adolescent Schizophrenia EEG; TCGA Glioma) were obtained from open scientific repositories. All derived aggregate metrics, parameter checkpoints, and prediction arrays are available in the project repository.

### Code Availability
All experimental code, simulation harnesses, audit pipelines, and verification tests are openly available at: `https://github.com/Sivaguru-2008/aura`. The Level B numerical verification suite (recomputing all metrics and figures from saved checkpoints in ~12 seconds) can be executed via: `python reproduce_paper.py`. Full end-to-end retraining instructions across all 36 configurations are detailed in Supplementary Note 7 (`python -m aura.ml.evaluation.v6_confirmatory train_c2 quantum`).

### Ethics and Governance Statements
All analyses were conducted as secondary research on established, fully de-identified clinical and biomedical repositories. MIMIC-CXR-JPG was accessed under PhysioNet Credentialed Health Data License 1.5.0. CheXpert-v1.0-Small was analyzed under the Stanford University Research Data Agreement. BraTS 2020 was accessed via the Synapse neuroimaging platform. Public benchmark datasets were obtained from open scientific archives. All records were fully de-identified in source repositories prior to acquisition, with no access to direct patient identifiers or protected health information, and no direct human intervention was performed. Secondary data analysis was approved or determined exempt under applicable data use agreements and institutional human research protection regulations.

### Competing Interests
The author declares no competing financial or non-financial interests.

### Author Contributions
S.R.M. conceived the study, formulated the symmetric evaluation protocol and 17-item audit checklist, authored and executed all simulation, training, audit, and reproduction code, conducted all statistical analyses, and wrote the manuscript and supplementary information.

---

## References

1. Gupta, S. et al. Quantum machine learning in digital health: a systematic review. *npj Digit. Med.* **8**, 42 (2025).
2. Rendle, S. et al. Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In *Proc. RecSys* 101–109 (2019).
3. Musgrave, K. et al. A metric learning reality check. In *Proc. ECCV* 681–699 (2020).
4. Henderson, P. et al. Deep reinforcement learning that matters. In *Proc. AAAI* 3207–3214 (2018).
5. Glocker, B. et al. Machine learning for medical image analysis: are we making progress? *Lancet Digit. Health* **4**, e680–e681 (2022).
6. Kochkov, D. et al. Machine learning-accelerated computational fluid dynamics. *Proc. Natl. Acad. Sci. USA* **118**, e2101784118 (2021).
7. Kübler, J. et al. The inductive bias of quantum kernels. In *Proc. NeurIPS* 12661–12673 (2021).
8. Biamonte, J. et al. Quantum machine learning. *Nature* **549**, 195–202 (2017).
9. Cerezo, M. et al. Variational quantum algorithms. *Nat. Rev. Phys.* **3**, 625–644 (2021).
10. Huang, H.-Y. et al. Power of data in quantum machine learning. *Nat. Commun.* **12**, 2631 (2021).
11. Thanasilp, S. et al. Exponential concentration and untrainability in quantum kernel methods. *Quantum* **7**, 1153 (2023).
12. Shaydulin, R. et al. Importance of kernel bandwidth in quantum machine learning. *Phys. Rev. A* **106**, 042407 (2022).
13. McClean, J. R. et al. Barren plateaus in quantum neural network training landscapes. *Nat. Commun.* **9**, 4812 (2018).
14. Nadeau, C. & Bengio, Y. Inference for the generalization error. *Mach. Learn.* **52**, 239–281 (2003).
15. Parthasarathi, P. et al. Variational quantum classifier-based early identification and classification of chronic kidney disease using sparse autoencoder and LASSO shrinkage. *PeerJ Comput. Sci.* **11**, e2789 (2025). PMID: 40567645.
16. Daga, M., Naole, S. et al. Adaptive quantum kernel selection via leakage-free stacking for clinical diagnostics on NISQ hardware. *Sci. Rep.* **16**, 12841 (2026). PMID: 42286086.
17. Aksoy, G. & Cattan, G. Quantum machine-based decision support system for the detection of schizophrenia from EEG records. *J. Med. Syst.* **48**, 29 (2024). PMID: 38441727.
18. Akpinar, E. & Oduncuoglu, M. Hybrid classical and quantum computing for enhanced glioma tumor classification using TCGA data. *Sci. Rep.* **15**, 25935 (2025). PMID: 40676161.
19. Ghisoni, F., Borrotti, M. & Mariani, P. A large scale statistical analysis of quantum and classical neural networks in the medical domain. *Sci. Rep.* **16**, 3719 (2026). PMID: 41513819.
20. Jahin, M. A. et al. KACQ-DCNN: Uncertainty-aware interpretable Kolmogorov-Arnold classical-quantum dual-channel neural network for heart disease detection. *Comput. Biol. Med.* **197**, 110976 (2025). PMID: 40865268.
21. Chen, P. et al. Prostate cancer classification using quantum machine learning on multi-parametric MRI. *Proc. SPIE* **13930**, 139300K (2026). PMID: 42046760.
22. Choi, J. et al. Performance evaluation of quantum support vector machine for COVID-19 biomarker analysis. *Comput. Methods Programs Biomed.* **281**, 109343 (2026). PMID: 41966795.
23. Johnson, A. E. W. et al. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. *Sci. Data* **6**, 317 (2019).
24. Irvin, J. et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In *Proc. AAAI* 590–597 (2019).
25. Bakas, S. et al. Advancing the cancer genome atlas glioma analysis with multimodal brain tumor segmentation (BraTS). *Sci. Data* **4**, 170188 (2017).
26. Collins, G. S. et al. Protocol for development of TRIPOD+AI: a reporting guideline for diagnostic and prognostic prediction models based on artificial intelligence. *BMJ Open* **11**, e048008 (2021).
27. Schuld, M., Sweke, R. & Meyer, J. J. Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Phys. Rev. A* **103**, 032430 (2021).
