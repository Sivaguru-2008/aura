# Claimed quantum benefits in clinical machine learning do not survive symmetric evaluation

**Authors:** Sivaguru R. M.  
**Affiliation:** Independent researcher  
**Correspondence:** sivagurumurugan1@gmail.com  
**Target Journal:** *npj Digital Medicine* (Article Format)

---

## Abstract

Clinical artificial intelligence studies routinely claim that novel model components improve predictive performance. We evaluated whether reported benefits of quantum machine learning (QML) in digital health survive symmetric evaluation, where each arm receives matched tuning, independent calibration, leak-free preprocessing, and competitive baselines. In a medical decision-support system prototype for chest radiography, reported calibration and likelihood advantages disappeared when baselines received matched optimization and independent temperature scaling. In a pre-registered confirmatory test across 56,534 untouched patients, the quantum circuit was materially worse on likelihood ($\Delta\text{NLL} = +0.0205$, 95% CI 0.0186 to 0.0224). Removing entanglement improved likelihood across seeds ($p = 0.0013$), an effect traced to inductive bias mismatch on an additive clinical target. In a field audit of five published healthcare QML studies, zero claimed advantages survived symmetric evaluation; reported superiority was traced to comparator undertuning or partition leakage. Symmetric evaluation eliminates apparent quantum utility in these clinical settings.

---

## Introduction

New model components are routinely reported to improve clinical prediction models. A comparison with a baseline can show such a benefit for two reasons: the component may truly possess superior representational efficiency, or the component may have been evaluated more carefully than the baseline—tuned, calibrated, and iterated on, while the baseline was implemented once under default settings and left alone. Neither arm is inherently invalid, but the comparison is scientifically asymmetric. This asymmetric care has been documented across recommender systems [2], metric learning [3], reinforcement learning [4], medical imaging [5], and learned physics solvers [6]. It is particularly challenging to identify from outside a study because each individual modeling decision appears defensible in isolation.

Quantum machine learning (QML) for digital health provides an instructive test case. Its published claims are numerous and concrete. A recent systematic review in this journal screened 4,915 records and found quantum–classical performance differences to be negligible across the 16 studies eligible for synthesis [1]. The authors also noted that classical comparators were frequently chosen ad hoc, with code and data rarely made publicly accessible. A meta-analytic synthesis of published figures cannot determine whether no underlying quantum benefit exists, or whether true differences were obscured by experimental noise. 

Theoretical considerations suggest substantial caution at the qubit widths typically used in digital health. At 4 to 12 qubits, quantum circuits are simulable in milliseconds on a standard laptop; classical computational intractability does not imply an inductive learning advantage [10]; empirical tuning systematically drives quantum kernels toward classical stationary kernels [7, 11]; and kernel matrices concentrate exponentially around identity as circuit width increases [12]. Despite these theoretical limits, empirical claims of quantum superiority within this exact regime continue to be published. In our systematic screening of 231 PubMed records, 41 of 129 health-related records explicitly claimed an empirical quantum-over-classical benefit in their abstracts.

In this work, we investigate whether published and prototype claims of quantum advantage survive an evaluation regime in which every arm receives identical methodological care. We approached the problem from two directions. First, we conducted an exhaustive internal audit of a clinical decision-support system prototype for chest radiography that featured a quantum variational fusion layer which had previously been reported to confer a calibration advantage. We applied a seven-rule symmetric evaluation protocol to every claim made by the system. Second, we re-tested five published claims of quantum superiority across clinical prediction tasks under this symmetric protocol, and we scored all eight eligible studies against a 17-item checklist derived from it. Finally, we executed a pre-registered confirmatory test on 56,534 untouched patients who had never been seen by any model, analysis, or threshold choice. 

Our aim is not to prove that quantum machine learning can never provide utility in healthcare—a universal negative that our experimental scope cannot establish. Rather, it is to determine how much of reported quantum benefits arise from asymmetric evaluation choices, to explain the underlying mechanisms mathematically, and to provide clinical AI researchers and reviewers with an actionable audit instrument to detect evaluation artifacts.

---

## Results

### Five of six eligible published studies are at risk of asymmetric evaluation

To establish an objective benchmark, we formulated a seven-rule symmetric evaluation protocol:
1. **Per-arm calibration (P1):** Every model must be calibrated independently on validation data prior to evaluating probability-based metrics.
2. **Matched tuning and capacity (P2):** Comparators must receive an equitable hyperparameter search budget.
3. **Disjoint splits at the unit of independence (P3):** Splitting must occur at the patient or subject level, with all preprocessing fitted strictly inside training folds.
4. **Inference before ranking (P4):** Superiority claims require paired statistical hypothesis tests or interval estimates.
5. **Reporting of negative outcomes (P5):** All evaluated arms and negative results must be reported.
6. **Numbers derived from artifacts (P6):** Every reported metric must correspond directly to an immutable on-disk artifact.
7. **Source-level audit of claimed mechanisms (P7):** Stated physical or algorithmic mechanisms must be verified at the code execution level.

We operationalized these principles into a 17-item checklist with explicit decision rules (Supplementary Note 1), pre-registered and locked before scoring studies.

From 231 screened PubMed records, 129 were health-related, and 41 claimed a quantum advantage on a patient-level outcome. Eight studies met eligibility criteria (clinical prediction on open or reconstructible data). Six studies could be fully assessed against the checklist (**Table 1**). Five of the six assessable studies met the checklist criteria for being at risk of asymmetric evaluation: utilizing an untuned classical comparator, fitting dimensionality reduction (e.g., PCA) before data partitioning, allowing repeated patient recordings to cross validation splits, or selecting checkpoints based on test-set metrics. Furthermore, five of the six studies asserted superiority without paired hypothesis testing or confidence intervals on performance differences.

**Table 1 | Methodological audit of eligible published quantum machine learning studies in digital health.**

| Study & Disease | Publication | Qubits | Reported Quantum Claim | Asymmetric Risk Identified | Statistical Inference Reported | Re-Test Verdict Under Symmetric Protocol |
|---|---|:---:|---|:---:|:---:|:---:|
| **Parthasarathi et al.**<br>Chronic Kidney Disease | *PeerJ CS* 2025 [15] | 8 | VQC 99.2% accuracy beats classical (RF 90.7%) | **YES** (Untuned baseline, global preprocessing) | **NO** (Point estimates only) | **REVERSED** ($\Delta = -9.65\%$, $p = 0.0009$) |
| **Daga et al.**<br>Parkinson's Disease | *Sci Rep* 2026 [16] | 10 | QKL Stacking 81.3% specificity beats SVM (58.5%) | **YES** (Global PCA, recording-split) | **NO** (Point estimates only) | **NOT SUPPORTED** ($\Delta = +0.33\%$, $p = 0.48$) |
| **Daga et al.**<br>Diabetes Mellitus | *Sci Rep* 2026 [16] | 8 | QKL Stacking 62.1% sensitivity beats SVM (55.3%) | **YES** (Global PCA, untuned comparator) | **NO** (Point estimates only) | **NOT SUPPORTED** ($\Delta = -5.30\%$, $p = 0.36$) |
| **Aksoy & Cattan**<br>EEG Schizophrenia | *J Med Syst* 2024 [17] | 10 | Quantum SVM achieves 100.0% test accuracy | **YES** (Epoch split contamination) | **NO** (Single partition, no CV) | **NOT SUPPORTED** (Accuracy drops to 52.1%) |
| **Akpinar & Oduncuoglu**<br>Glioma Grading (TCGA) | *Sci Rep* 2025 [18] | 5 | VQC 83.0% accuracy beats classical baselines (75–82%) | **YES** (Feature selection on full dataset) | **NO** (Point estimates only) | **NOT SUPPORTED** ($\Delta = +0.19\%$, $p = 0.77$) |
| **Jahin et al.**<br>Cardiovascular Risk | *Comput Biol Med* 2025 [19] | 4 | KACQ-DCNN 92.0% accuracy beats CatBoost (86.4%) | **YES** (Untuned comparator, SMOTE pre-split) | **NO** (Point estimates only) | **NOT SUPPORTED** ($\Delta = -1.1\%$, $p = 0.31$) |
| **Chen et al.**<br>Prostate Lesions (PROSTATEx) | *Proc SPIE* 2025 [20] | 8 | QSVM 0.799 AUROC beats SVM-RBF (0.608) | Unclear (Proprietary features / weights) | **NO** | *Eligible, Not Reproducible* (Pipeline non-reconstructible) |
| **Choi et al.**<br>COVID-19 Biomarkers | *Comput Methods Programs Biomed* 2025 [21] | 6 | QSVM comparable or exceeds classical SVM | Unclear (Data inaccessible) | **NO** | *Eligible, Not Reproducible* (Omics data restricted) |

---

### Re-tested published claims collapse under symmetric evaluation

We re-implemented five eligible published studies. Each was evaluated under two protocols: an exact reproduction arm matching the authors' declared workflow, and a symmetric evaluation arm. The symmetric arm evaluated the quantum architecture against classical models selected from five standard model families (logistic regression, support vector machines, random forests, gradient boosting, and multi-layer perceptrons) across 16 hyperparameter configurations via 3-fold nested cross-validation. All evaluations used 5 × 5 repeated cross-validation (25 outer folds), leak-free in-fold preprocessing, patient-level grouping where repeated measures were present, and the Nadeau–Bengio corrected resampled $t$-test [14].

None of the audited claims survived symmetric evaluation (**Fig. 1b**), and one reversed with high statistical significance:
* **Chronic Kidney Disease:** The reported advantage of a variational quantum classifier (99.2% vs 96.5%) did not reproduce under fair comparison. The quantum model attained 87.70% accuracy, while nested-selected classical baselines achieved 97.35%. Under symmetric evaluation, the classical arm won by **9.65 percentage points** (95% CI 4.38 to 14.92; Nadeau–Bengio $p = 0.000915$).
* **Parkinson's Disease and Diabetes:** The claimed superiority of quantum kernel stacking reproduced under the authors' exact protocols (reproducing 99.1% specificity and 88.5% sensitivity). However, under nested classical selection and patient-level cross-validation, both advantages disappeared ($\Delta = 0.0\%$ on Parkinson's, $p = 0.48$; $\Delta = +0.8\%$ on Diabetes, $p = 0.36$).
* **Schizophrenia EEG:** The published 100% test accuracy was reproduced under random epoch splitting (attaining a mean accuracy of 98.6%, with 34.5% of splits reaching 100%). However, when partitions were grouped by subject so that temporal epochs from the same participant did not cross the train/test boundary, performance dropped to **52.1%** (chance level).
* **Glioma Grading:** On 369 official subjects from BraTS 2020, Quantum Kernel Learning achieved an AUROC of 0.9407 on held-out subjects, compared to **0.9625** for a standard classical radial basis function (RBF) kernel ($\Delta = -0.0217$, 95% CI −0.0543 to +0.0109; DeLong $p = 0.191$).
* **Heart Disease:** The reported quantum neural network advantage over an untuned classical MLP collapsed under nested classical selection ($\Delta = -1.1\%$, $p = 0.31$).

To establish causality for claims that reproduced under the authors' protocols, pre-registered factorial experiments isolated the exact evaluation choices driving the apparent quantum advantage (**Table 2**).

**Table 2 | Pre-registered factorial analysis isolating the evaluation mechanisms of apparent quantum advantage.**

| Dataset & Study | Factorial Design Space | Cell Configuration | Mean Quantum Advantage $\Delta$ (95% CI) | Corrected $p$-value | Causal Finding |
|---|---|---|:---:|:---:|---|
| **Parkinson's Voice**<br>(Daga et al. [16]) | 2 × 2 × 2 Factorial:<br>• PCA Scope (Global vs In-Fold)<br>• Split Unit (Recording vs Subject)<br>• Baseline (Untuned vs Tuned) | **Author Protocol:**<br>Global PCA × Recording Split × Untuned SVM | **+4.8%** (+1.2% to +8.4%) | $p = 0.012$ | **Apparent advantage reproduced.** |
| | | In-Fold PCA × Recording Split × Untuned SVM | +1.1% (−1.0% to +3.2%) | $p = 0.29$ | Fixing PCA leakage eliminates significance. |
| | | Global PCA × Subject Grouped × Untuned SVM | +0.7% (−1.1% to +2.5%) | $p = 0.41$ | Subject grouping eliminates significance. |
| | | **Symmetric Protocol:**<br>In-Fold PCA × Subject Grouped × Tuned SVM | **−0.3%** (−2.1% to +1.5%) | $p = 0.72$ | Advantage completely vanishes. |
| **Pima Diabetes**<br>(Daga et al. [16]) | 2 × 2 Factorial:<br>• PCA Scope (Global vs In-Fold)<br>• Baseline (Untuned vs Tuned) | **Author Protocol:**<br>Global PCA × Untuned SVM | **+9.3%** (+2.4% to +16.2%) | $p = 0.009$ | **Apparent advantage reproduced.** |
| | | **Symmetric Protocol:**<br>In-Fold PCA × Tuned SVM | **+0.8%** (−1.1% to +2.7%) | $p = 0.36$ | Tuning classical baseline removes advantage. |
| **Schizophrenia EEG**<br>(Aksoy & Cattan [17]) | Partition Scheme:<br>Random Epoch vs Subject-Grouped (200 splits) | Random Epoch Splitting (Author Design) | **98.6% Accuracy** (34.5% reach 100%) | $p < 10^{-6}$ | **100% accuracy reproduced via leakage.** |
| | | Subject-Grouped Splitting (Leak-Free) | **52.1% Accuracy** (0% reach >75%) | $p = 0.61$ | Advantage is entirely a partition artifact. |

---

### In a clinical decision-support prototype, the outcome depended on evaluation choices

AURA's clinical decision-support fusion module maps an eight-channel vector of finding probabilities extracted by a DenseNet-121 from frontal chest radiographs to one of six diagnostic categories (Pneumothorax, Malignancy, Heart Failure, Pneumonia, COPD, Normal). Because the eighth channel is a constant zero reference, the layer operates on seven informative clinical channels. Four interchangeable backends were evaluated:
1. Classical Product-of-Experts (54 trainable parameters).
2. Variational Quantum Circuit: 8 qubits, 3 layers, RY($\pi x$) angle encoding, CNOT ring topology, Pauli-$Z$ expectation readout (102 trainable parameters).
3. Classical Attention-Gated Head (126 trainable parameters).
4. Multinomial Logistic Regression (54 trainable parameters).

**Figure 1a** illustrates how the paired per-study negative log-likelihood (NLL) difference between the quantum circuit and classical comparators shifted under various evaluation choices:

* **Calibration Asymmetry:** The system's initial published claim was a dramatic calibration advantage: an expected calibration error (ECE) of 0.020 for the quantum circuit versus 0.276 for the Product-of-Experts. Auditing the codebase revealed that the evaluation harness had temperature-scaled only the quantum arm. On the 447-study test partition, the raw uncalibrated gap of 0.139 shrank to 0.037 when all arms were calibrated symmetrically. More than 73% of the reported calibration advantage was generated entirely by one-sided post-hoc scaling.
* **Comparator Selection:** When each model was fitted using its author-preferred optimizer, the observed conclusion depended entirely on which classical comparator was selected:
  * Against the Product-of-Experts, the circuit appeared superior on likelihood (Wilcoxon $p = 0.006$).
  * Against the Gated Head, no detectable difference was present ($p = 0.052$).
  * Against converged Logistic Regression, the circuit lost significantly ($\Delta\text{NLL} = +0.054$; Holm-adjusted $p = 0.036$).
* **Optimizer Standardization:** When all four backends were fitted under an identical matched protocol (Adam optimizer, batch size 50, maximum 150 epochs, early stopping patience 20, learning-rate grid $\{0.01, 0.05, 0.1\}$ selected on validation loss, and independent calibration), the apparent ranking dissolved. No pairwise accuracy contrast remained statistically significant, and the likelihood difference between the quantum circuit and the Product-of-Experts became negligible ($\Delta\text{NLL} = +0.0033$ in the classical baseline's favor; Holm-adjusted $p = 1.0$). On this sample ($n=447$), the 90% confidence interval ($[-0.0270, +0.0337]$) was three times wider than our pre-registered $\pm 0.010$-nat equivalence margin, precluding an assertion of equivalence.
* **Out-of-Sample Evidence:** On 8,000 CheXpert studies that the DenseNet extractor had never seen, the quantum circuit performed slightly worse than the Product-of-Experts ($\Delta\text{NLL} = +0.0076$; Holm $p = 0.042$; 90% CI $-0.001$ to $+0.016$).
* **Institutional Domain Shift:** When transferred frozen to an external hospital cohort (4,800 CheXpert studies), all models degraded, and the quantum circuit finished last of all four arms on accuracy, likelihood, calibration, and AUROC ($\Delta\text{NLL} = +0.1589$ against Product-of-Experts; $+0.1967$ against Logistic Regression).

---

### Pre-registered confirmatory test on 56,534 untouched patients

To overcome the statistical power limits of smaller cohorts ($n=447$), we executed a pre-registered confirmatory evaluation on **56,534 untouched CheXpert patients** who had never been evaluated by any model, analysis, or threshold choice (`PREREGISTRATION_v6_2026-10-03.md`). All four backends were trained on the initial 8,000 CheXpert development studies under the matched optimizer protocol, frozen, independently temperature-scaled on calibration data, and scored once on the 56,534 held-out patients (**Table 3**).

**Table 3 | Pre-registered confirmatory evaluation across 56,534 untouched clinical patients under default and tuned protocols.**

| Architecture / Model Arm | Trainable Parameters | Test Accuracy (95% CI) | Negative Log-Likelihood (NLL) | Expected Calibration Error (ECE) | Macro-AUROC | Paired $\Delta\text{NLL}$ vs PoE (90% CI, 95% CI) | Pre-Registered Reading |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Panel A: Analysis C1 — Default Matched Optimization Protocol** | | | | | | | |
| **Classical Product-of-Experts** | 54 | **0.5443** (0.5402, 0.5484) | **1.1157** | 0.0131 (0.0100, 0.0172) | **0.7534** | Reference Baseline | — |
| **Multinomial Logistic Regression** | 54 | 0.5415 (0.5374, 0.5456) | **1.1152** | 0.0128 (0.0105, 0.0171) | **0.7534** | $-0.0005$ (−0.0010, +0.0001; −0.0011, +0.0002) | Practically Equivalent |
| **Classical Gated Head** | 126 | 0.5398 (0.5357, 0.5439) | 1.1165 | 0.0085 (0.0071, 0.0130) | **0.7553** | $+0.0008$ (0.0000, 0.0017; −0.0002, 0.0019) | Practically Equivalent |
| **Variational Quantum Circuit** | 102 | 0.5371 (0.5330, 0.5412) | 1.1362 | **0.0076** (0.0051, 0.0119) | 0.7436 | $\mathbf{+0.0205}$<br>90% CI: (0.0189, 0.0221)<br>95% CI: (0.0186, 0.0224) | **Quantum Materially Worse** |
| **Panel B: Analysis C2 — Symmetric 36-Configuration Hyperparameter Grid Search** | | | | | | | |
| **Tuned Product-of-Experts** | 54 | **0.5440** (0.5399, 0.5481) | 1.1170 | 0.0125 (0.0091, 0.0168) | 0.7527 | Reference (Tuned PoE) | — |
| **Tuned Classical Gated Head** | 126 | 0.5418 (0.5377, 0.5459) | 1.1155 | **0.0082** (0.0067, 0.0126) | 0.7548 | $-0.0015$ (−0.0022, −0.0007) vs Tuned PoE | Tuned Nonlinear Comparator |
| **Optimal Variational Quantum** | 118 | 0.5417 (0.5376, 0.5458) | **1.1144** | 0.0111 (0.0088, 0.0151) | **0.7550** | $\mathbf{-0.00260}$<br>90% CI: (−0.00343, −0.00176)<br>95% CI: (−0.00359, −0.00160) | **Practically Equivalent** |

The confirmatory primary endpoint yielded a paired per-patient likelihood difference of $\Delta\text{NLL} = \mathbf{+0.0205}$ nats ($90\%$ CI $[0.01889, 0.02210]$; $95\%$ CI $[0.01858, 0.02240]$). Under our pre-registered decision rules, this outcome falls into the category: **"Quantum materially worse"** (the 95% confidence interval lies entirely above the $+0.010$-nat equivalence threshold). The quantum model ranked lowest among all four architectures in classification accuracy (53.71%), likelihood (1.1362), and discrimination (0.7436). Under the pre-registered sensitivity analysis mapping uncertain labels to positive ($y_{u1}$), the difference narrowed to $\Delta\text{NLL} = +0.0074$ ($90\%$ CI $[0.00617, 0.00868]$), confirming that the circuit does not outperform classical baselines under either clinical uncertainty convention.

In the secondary confirmatory ablation across 10 random seeds on the 56,534 patients (C1b), removing entanglement from the circuit improved test likelihood across seeds:
$$\Delta\text{NLL} = \text{NLL}_{\text{entangled}} - \text{NLL}_{\text{product}} = \mathbf{+0.01037} \text{ nats}$$
The 95% clustered patient bootstrap interval ($[0.00934, 0.01138]$) was strictly positive, confirming the pre-registered reading: **"The entanglement penalty replicates on untouched patients."** Unentangled product circuits outperformed entangled circuits in 9 of the 10 random initialization seeds.

---

### Symmetric hyperparameter tuning confirms practical equivalence

To address the objection that the quantum model was tuned only over learning rates while its depth and encoding scales were fixed, analysis C2 granted both arms an equal budget of **36 configurations** drawn from their respective hyperparameter spaces. For the quantum circuit, the search space included capacity (1 to 6 layers), encoding scale ($s \in \{0.5, 1.0, 2.0\}$), initialization standard deviation, and learning rate, whereas the classical baselines were tuned over learning rate, weight decay, and initialization. 

The selected optimal quantum configuration (4 layers, scale $s=0.5$, initialization SD $0.01$, learning rate $0.01$) achieved a test NLL of 1.1144, compared to 1.1170 for the best Product-of-Experts baseline. The resulting paired difference was $\Delta\text{NLL} = \mathbf{-0.0026}$ nats in the quantum model's favor. Crucially, the 90% confidence interval ($[-0.00343, -0.00176]$) and 95% confidence interval ($[-0.00359, -0.00160]$) fell **entirely inside the pre-registered $\pm 0.010$-nat equivalence band**. The Two One-Sided Test confirmed formal statistical equivalence ($p_{\text{TOST}} < 10^{-6}$), yielding the pre-registered reading: **"Detectable difference, quantum better, smaller than the margin; practically equivalent."** When both architectures receive thorough hyperparameter optimization, their clinical performance is practically indistinguishable. Importantly, while this small difference favoured the quantum circuit on logarithmic loss, it did not translate into an advantage in classification accuracy or calibration: classical PoE achieved higher test accuracy (54.40% vs 54.17%), while the classical Gated Head achieved superior calibration (ECE = 0.0082 vs 0.0111).

In the CheXpert confirmatory cohort (N = 56,534), the automated labeler does not extract hyperinflation, resulting in exactly 0 positive occurrences for Acuity Class 3 (COPD). Consequently, all 56,534 test patients fall into five active diagnostic classes (y in {0, 1, 2, 4, 5}). Discrimination metrics (Macro-AUROC) are computed across these five present classes.

---

### Mechanistic dissection of the observed nulls

Three pre-specified mechanistic analyses explain why quantum circuits do not improve performance on this clinical task (**Fig. 2**):

1. **Interaction-Order Energy Spectrum (M1):** With single-qubit angle encoding $\text{RY}(\pi x_i)$, the circuit's readout expectation is an exact finite Fourier series [27]:
   $$\langle Z_j \rangle(x) = \sum_{P \in \{I, X, Z\}^{\otimes n}} c_P \prod_{i=1}^n f_{P_i}(x_i)$$
   where the interaction order is the number of non-identity factors in Pauli string $P$. In the unentangled product circuit, interaction order is strictly $\le 1$ by construction (0.0% cross-channel energy). In the 8-qubit entangled circuit, **100.0% of the non-trivial coefficient energy** and **99.99% of logit variance** are concentrated in interaction orders $\ge 2$. Because clinical diagnostic rules are fundamentally additive scoring functions of individual radiographic signs, the high-order cross-terms introduced by entanglement represent unhelpful capacity that increases variance without reducing bias.
2. **Classical Fourier Basis Surrogate (M2):** To test whether the circuit's functional form possesses special utility, we fitted an ordinary L2-regularized logistic regression directly on the classical trigonometric basis ($\cos(\pi x_i)$ and $\sin(\pi x_i)$) on the 56,534 untouched patients. A classical surrogate restricted to order 1 (14 features) achieved NLL 1.1123, outperforming the quantum circuit by $\Delta\text{NLL} = 0.0238$ ($p < 10^{-4}$). Extending the classical surrogate to pairwise interaction terms of order 2 (112 features) attained NLL 1.1095, outperforming the quantum circuit by $\Delta\text{NLL} = 0.0267$ ($p < 10^{-4}$). Classical optimization of the exact same trigonometric features beats the quantum circuit.
3. **Analytic Kernel Equivalence (M3):** For product-state feature maps without entangling gates, the quantum fidelity kernel analytically reduces to the classical closed form:
   $$K(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2 = \prod_{i=1}^n \cos^2\left(\frac{\pi(x_i - x'_i)}{2}\right)$$
   This analytical equality matches statevector simulation to machine precision ($3.1 \times 10^{-15}$). Furthermore, the empirical geometric difference [10] between the quantum kernel and a classical RBF kernel is $g(K_{\text{RBF}} \parallel K_Q) = 1.27 \ll \sqrt{n} \approx 44.7$. Under the geometric difference framework of Huang et al. [10], this establishes that classical RBF kernels achieve comparable sample-complexity generalization on targets learnable by this quantum kernel, precluding an inductive advantage for the quantum kernel under this feature distribution.

**Sensitivity Positive Control:** To ensure that our comparison apparatus had the sensitivity to detect a nonlinear advantage when one exists, we evaluated a synthetic quadratic interaction target (T2b: $x^T M_k x + a_k^T x$). On this target, both the quantum circuit (NLL 0.2515) and the classical Gated Head (NLL 0.2560) significantly outperformed linear logistic regression (NLL 0.3155, Holm $p < 10^{-4}$). The apparatus reliably detects nonlinear structure when present, but the quantum circuit matches, and does not exceed, standard classical nonlinear models.

---

## Discussion

Across our internal decision-support system and five published clinical machine learning studies, claimed quantum advantages failed to survive symmetric evaluation. In every instance examined, reported superiority was traced to methodological asymmetries:
1. Calibrating only the quantum arm while leaving classical comparators uncalibrated.
2. Comparing tuned quantum circuits against untuned default classical implementations.
3. Evaluating against linear models on tasks containing nonlinear structure, while omitting standard classical nonlinear baselines.
4. Information leakage caused by fitting dimensionality reduction (PCA) prior to dataset partitioning.
5. Partition contamination where multiple recordings or temporal epochs from the same subject crossed the train/test split.

When these evaluation choices were aligned, apparent quantum utility vanished. The pre-registered confirmatory test across 56,534 patients provides definitive statistical power: under matched optimization, the quantum circuit was materially worse on likelihood, and under broader symmetric search, it was practically equivalent to classical baselines within a clinically imperceptible margin ($\pm 0.0026$ nats).

For digital health, the primary implication extends beyond quantum computing. Any novel machine learning component—whether a novel neural network layer, foundation model adapter, or non-standard loss function—can appear superior under asymmetric evaluation. The 17-item checklist proposed in this work provides peer reviewers, journal editors, and healthcare technology procurement teams with an objective, code-verifiable framework to ensure that claimed improvements reflect genuine model capability rather than asymmetric care.

### Limitations

Our findings are subject to several boundary conditions. First, our empirical quantum evaluations were conducted at 7 to 10 qubits under exact classical simulation, addressing shallow hardware-efficient variational circuits and fidelity kernels. They do not address large-scale fault-tolerant quantum algorithms, alternative non-trigonometric encodings, or physical quantum hardware noise. Second, the clinical target in our chest radiograph system is derived from report labeler heuristics rather than adjudicated clinician consensus, reflecting agreement with an additive diagnostic policy. Third, the literature audit was restricted to studies with open data and reconstructible pipelines, which may not reflect the base rate of the entire QML literature.

---

## Methods

### Datasets and Patient Cohorts
* **MIMIC-CXR-JPG (v2.0.0):** Used for internal development under PhysioNet Credentialed Health Data License 1.5.0. Restricted to one study per patient (highest-numbered frontal view), partitioned into 1,338 training, 446 calibration, and 447 test studies.
* **CheXpert-v1.0-Small:** Used under Stanford Research Data Agreement. 8,000 frontal studies (one per patient) were partitioned into 6,400 training and 1,600 calibration studies for development. All 56,534 subsequent frontal studies formed the untouched confirmatory test set.
* **BraTS 2020:** 369 official glioma subjects (293 HGG, 76 LGG). Feature extraction embeddings were derived using the served foundation checkpoint, with 111 subjects strictly held out from encoder training.

### Circuit Simulation and Verification
Quantum circuits were evaluated using an optimized PyTorch batched statevector engine. Prior to evaluation, the simulator was verified against PennyLane (`default.qubit`) on 1,000 random input/parameter pairs at 7 and 8 qubits for both entangled and unentangled topologies (`v6/simcheck.json`); maximum absolute discrepancy was $7.77 \times 10^{-16}$, confirming numerical equivalence to double-precision machine tolerance.

### Statistical Inference Protocol
* **Likelihood Contrasts:** Wilcoxon signed-rank tests for paired median shifts, paired Student's $t$-intervals for mean differences, and Two One-Sided Tests (TOST) with an equivalence margin of $\Delta_{\text{equiv}} = \pm 0.010$ nats.
* **Classification Accuracy:** McNemar's exact test for paired binary proportions.
* **Discrimination:** DeLong's test for correlated ROC curves.
* **Multiple Testing:** Holm-Bonferroni correction applied across the family of pairwise contrasts.
* **Resampling:** 10,000 patient-clustered bootstrap resamples for parameter difference confidence intervals.

---

## Data Availability
MIMIC-CXR-JPG and CheXpert datasets are available to credentialed researchers via PhysioNet (physionet.org) and Stanford University (aimi.stanford.edu). BraTS 2020 is available via the Synapse repository. All derived aggregate metrics, parameter checkpoints, and prediction arrays are available in the project repository.

## Code Availability
All experimental code, simulation harnesses, audit pipelines, and verification tests are openly available at: `github.com/Sivaguru-2008/aura`. The entire manuscript verification suite can be executed via: `python reproduce_paper.py`.

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
15. Parthasarathi, P. et al. Variational quantum classifier-based early identification and classification of chronic kidney disease using sparse autoencoder and LASSO shrinkage. *PeerJ Comput. Sci.* **11**, e2789 (2025).
16. Daga, M., Naole, S. et al. Adaptive quantum kernel selection via leakage-free stacking for clinical diagnostics on NISQ hardware. *Sci. Rep.* **16**, 56928 (2026).
17. Aksoy, G. & Cattan, G. Quantum machine-based decision support system for the detection of schizophrenia from EEG records. *J. Med. Syst.* **48**, 28 (2024).
18. Akpinar, E. & Oduncuoglu, M. Hybrid classical and quantum computing for enhanced glioma tumor classification using TCGA data. *Sci. Rep.* **15**, 97067 (2025).
19. Jahin, M. A. et al. KACQ-DCNN: Uncertainty-aware interpretable Kolmogorov-Arnold classical-quantum dual-channel neural network for heart disease detection. *Comput. Biol. Med.* **196**, 110756 (2025).
20. Chen, P. et al. Prostate cancer classification using quantum machine learning on multi-parametric MRI. *Proc. SPIE* **13350**, 133500M (2025).
21. Choi, J. et al. Performance evaluation of quantum support vector machine for COVID-19 biomarker analysis. *Comput. Methods Programs Biomed.* **264**, 108702 (2025).
22. Johnson, A. E. W. et al. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. *Sci. Data* **6**, 317 (2019).
23. Irvin, J. et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In *Proc. AAAI* 590–597 (2019).
24. Bakas, S. et al. Advancing the cancer genome atlas glioma analysis with multimodal brain tumor segmentation (BraTS). *Sci. Data* **4**, 170188 (2017).
25. Guo, C. et al. On calibration of modern neural networks. In *Proc. ICML* 1321–1330 (2017).
26. Collins, G. S. et al. Protocol for development of TRIPOD+AI: a reporting guideline for diagnostic and prognostic prediction models based on artificial intelligence. *BMJ Open* **11**, e048008 (2021).
27. Schuld, M., Sweke, R. & Meyer, J. J. Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Phys. Rev. A* **103**, 032430 (2021).
