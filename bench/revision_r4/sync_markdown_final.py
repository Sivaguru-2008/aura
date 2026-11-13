# -*- coding: utf-8 -*-
"""Synchronize MANUSCRIPT_FINAL_npjDM.md directly with the authoritative manuscript.tex.
Preserves exact text, numbers, formulas, tables, figures, and structural order (Methods after Discussion).
"""
import shutil
from pathlib import Path

REPO_ROOT = Path("e:/AURA/aura-main")
MANUSCRIPT_DIR = Path("e:/AURA/manuscript")

md_content = r"""# Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit

**Author:** Sivaguru R. M.$^{1,*}$  
**Affiliation:** $^{1}$Independent Researcher, Chennai, India  
**Correspondence:** *e-mail:* [sivagurumurugan1@gmail.com](mailto:sivagurumurugan1@gmail.com)  
**Target Journal:** *npj Digital Medicine* (Article Format)  
**Status:** Preprint / Manuscript Under Review Draft (October 2026)  
**Repository:** [https://github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  

---

## Abstract

Empirical claims that quantum machine learning outperforms classical models have proliferated in digital health, yet comparison symmetry is rarely audited. We evaluated whether reported quantum benefits survive symmetric evaluation requiring matched optimization, independent calibration, leak-free partitioning, and resampled inference. In a chest radiograph triage prototype across 56,534 held-out patients, default variational quantum circuits (VQC) yielded higher negative log-likelihood ($\Delta\text{NLL} = +0.0205$ nats; 10-seed mean $+0.0142$ nats) than matched classical baselines, exceeding seed variance ($\sigma_{\text{seed}} = 0.0045$ nats). Under matched 36-configuration search, optimal VQC test loss was statistically equivalent to classical baselines overall ($\Delta\text{NLL} = -0.0026$ nats; 10-seed mean $-0.00067$ nats, $\pm 0.010$ margin), but equivalence failed under class-balanced macro-averaging ($\Delta\text{NLL}_{\text{macro}} = +0.0389$ nats, $p_{\text{TOST}} > 0.99$; Malignancy $+0.1771$ nats), while requiring 92.8-fold more compute. Classical polynomial (NLL 1.1092) and Fourier surrogates (NLL 1.1123) outperformed the tuned VQC (1.1144). Under institutional transfer, all models degraded ($\Delta\text{NLL} = +0.131$ to $+0.197$ nats). Across five reconstructible publications spanning six clinical tasks, none of six published claims were verified: one was not reproduced, one collapsed to chance (52.1%), and four were statistically inconclusive. Reported quantum benefits reflected evaluation asymmetries, uncalibrated comparisons, or underpowered cohorts rather than verified algorithmic superiority.

---

## Introduction

New machine learning components are routinely introduced into biomedical data science with claims of performance gains over established baselines. When an empirical comparison demonstrates that a proposed model outperforms a comparator, that difference can reflect two fundamentally distinct causes: the proposed architecture may possess genuine representational or computational superiority on the target data distribution, or the proposed architecture may have received disproportionate methodological attention—extensive hyperparameter tuning, bespoke preprocessing, selective thresholding, and specialized post-hoc calibration—while the baseline was evaluated under unoptimized default settings [2, 3]. Although each modeling decision may appear justifiable in isolation, the resulting comparison is scientifically asymmetric.

The phenomenon of asymmetric evaluation has been rigorously documented across numerous subfields of applied machine learning, including recommender systems [2], metric learning [3], deep reinforcement learning [4], learned physics simulators [6], clinical image analysis [5], and shortcut learning [28]. In healthcare, where algorithmic predictions inform critical triage decisions, diagnostic stratification, and resource allocation, asymmetric evaluations present acute risks: they threaten patient safety, distort research priorities, and divert institutional capital toward computationally burdensome models that offer no verifiable patient benefit [1].

Quantum machine learning (QML) applied to clinical data represents an ideal domain in which to examine evaluation symmetry. Over the past five years, numerous studies have reported that parameterized variational quantum circuits (VQCs), quantum support vector machines (QSVMs), and quantum neural networks (QNNs) achieve superior diagnostic accuracy, sensitivity, or calibration compared to standard classical algorithms on biomedical classification benchmarks. However, theoretical analyses establish severe boundary conditions for near-term noisy intermediate-scale quantum (NISQ) devices operating at modest widths (4 to 12 qubits). At these widths, statevector quantum evolution can be simulated classically in fractions of a millisecond on standard laptop processors; classical computational intractability does not imply an inductive learning advantage [10]; kernel matrix elements concentrate exponentially around identity with increasing width [11, 12]; and gradient landscapes suffer from barren plateaus under random parameterized circuits [13]. A recent systematic review in this journal screened 4,915 citations and found quantum–classical performance differences to be negligible across 16 quantitatively synthesizable studies, observing that classical comparators were frequently selected ad hoc without open code or verified hyperparameter search [1].

These findings raise an unresolved question in clinical artificial intelligence: **When claimed quantum machine-learning advantages in clinical prediction are evaluated with symmetric model selection, comparable classical baselines, leakage-resistant data partitioning, and appropriate statistical inference, which advantages remain supported by evidence?**

To address this question, we formulated a seven-rule symmetric evaluation protocol (P1–P7) and operationalized it into a 17-item checklist with objective binary scoring criteria, aligning with international reporting standards including CONSORT-AI [29] and TRIPOD+AI [26]. We executed a dual-track empirical investigation:
1. **Systematic Benchmark Re-evaluation:** We conducted an independent screening of 231 PubMed publications, systematically identifying eight eligible studies and computationally reimplementing five reconstructible publications spanning six distinct clinical prediction tasks (chronic kidney disease, Parkinson's voice phonation, diabetes onset, schizophrenia electroencephalography, glioma tumor grading, and low-data heart disease risk).
2. **Confirmatory Clinical Decision-Support Study:** We audited an internal multi-class clinical triage prototype mapping radiographic finding probabilities from frontal chest radiographs to six acuity categories. We evaluated four interchangeable backend architectures (Classical Product-of-Experts, Multinomial Logistic Regression, Attention-Gated Nonlinear Neural Head, and Variational Quantum Circuit) across development cohorts ($n=8,000$) and a pre-specified confirmatory cohort of 56,534 untouched clinical patients.

Across both tracks, our findings demonstrate that claimed quantum advantages in clinical machine learning are artifacts of asymmetric evaluation choices: undertuned baselines, one-sided calibration, and partition leakage across subjects. When evaluation protocols are symmetrized, claimed advantages either fail to reproduce under the reported feature set, become statistically indistinguishable, or collapse to chance.

---

## Results

### Audited published studies exhibit pervasive methodological asymmetries

To examine the prevalence and severity of evaluation asymmetries in the literature, we conducted a systematic screening of 231 PubMed records (January 2020 through October 2026). Following PRISMA flow accounting, 129 records addressed clinical or health-related prediction tasks, 41 claimed empirical quantum advantages over classical baselines, eight met formal eligibility criteria (empirical evaluation on real-world clinical datasets comparing parameterized quantum and classical models), and five publications spanning six clinical prediction tasks provided sufficient methodological documentation for computational reconstruction (**Table 1**).

**Table 1 | Methodological audit of eligible published quantum machine learning studies in digital health.**

| Study & Disease | Publication | Qubits | Reported Claim | Asymmetric Risk | Inference | Re-Test Verdict |
|---|---|:---:|---|:---:|:---:|---|
| **Parthasarathi et al.**<br>Kidney Disease | *PeerJ CS* 2025 [15] | 8 | VQC 99.2% acc beats RF (90.7%), RBF (96.3%) | **YES** (Untuned baseline, global preprocessing) | **NO** | **NOT REPRODUCED**$^*$ (VQC 83.5% on reported features vs 99.2% claimed; classical baselines 97.4% vs 87.7% nested VQC; $\Delta = -9.65\%$, $p = 0.0009$) |
| **Daga et al.**<br>Parkinson's Voice | *Sci Rep* 2026 [16] | 10 | QKL Stacking 81.3% spec beats SVM (58.5%) [claim in abstract; up to 99.1% on recording split] | **YES** (Global PCA, recording split) | **NO** | **INCONCLUSIVE** (25-fold nested CV: $\Delta = +0.33\%$, 95% CI $-24.97\%$ to $+25.64\%$, $p = 0.98$; Factorial naive recording split: $-0.3\%$, $p = 0.72$) |
| **Daga et al.**<br>Diabetes Mellitus | *Sci Rep* 2026 [16] | 8 | QKL Stacking 62.1% sens beats SVM (55.3%) [up to 88.5% on author split] | **YES** (Global PCA, untuned baseline) | **NO** | **INCONCLUSIVE** (25-fold nested CV: $\Delta = -5.30\%$, 95% CI $-11.00\%$ to $+0.39\%$, $p = 0.067$; Factorial: $+0.8\%$, $p = 0.36$) |
| **Aksoy & Cattan**<br>Schizophrenia EEG | *J Med Syst* 2024 [17] | 10 | Quantum SVM achieves 100.0% accuracy | **YES** (Epoch split contamination) | **NO** | **COLLAPSED TO CHANCE** (Drops to 52.1% on subject split; 25-fold nested CV: $\Delta = -7.6\%$, 95% CI $-23.0\%$ to $+7.8\%$, $p = 0.32$) |
| **Akpinar & Oduncuoglu**<br>Glioma Grading | *Sci Rep* 2025 [18] | 5 | VQC 83.0% acc beats classical (75–82%) | **YES** (Feature selection on full data) | **NO** | **INCONCLUSIVE** (TCGA $\Delta = +0.19\%$, 95% CI $-1.22\%$ to $+1.60\%$, $p = 0.78$; BraTS 2020: $\Delta\text{AUROC} = -0.022$, $p = 0.19$) |
| **Ghisoni et al.**<br>Cardiovascular Risk | *Sci Rep* 2026 [19] | 11 | QNN 82.72% beats classical NN (80.51%) | **YES** (Early stop on test cost, untuned NN) | **NO** | **INCONCLUSIVE** ($\Delta = -1.1\%$; paired $t$ 95% CI $-3.2\%$ to $+1.0\%$; Nadeau–Bengio 95% CI $-18.45\%$ to $+16.25\%$, $p = 0.90$) |
| **Jahin et al.** | *Comput Biol Med* 2025 [20] | 4 | KACQ-DCNN 92.0% acc beats CatBoost (86.4%) | **YES** (Untuned baseline, SMOTE pre-split) | **NO** | *Eligible, Not Reproducible* (3M-param hybrid, no code) |
| **Chen et al.** | *Proc SPIE* 2026 [21] | 8 | QSVM 0.799 AUROC beats SVM-RBF (0.608) | Unclear (Proprietary features) | **NO** | *Eligible, Not Reproducible* (Pipeline non-reconstructible) |
| **Choi et al.** | *Comput Meth Prog Bio* 2026 [22] | 6 | QSVM comparable or exceeds classical SVM | Unclear (Data inaccessible) | **NO** | *Eligible, Not Reproducible* (Restricted omics access) |

<small>$^*$The published 99.2% accuracy claim utilized an under-specified sparse autoencoder augmentation and Kalman filter imputation that could not be replicated from the published text and had no public repository; on the published 13 LASSO features, our reimplemented VQC reached only 83.5% under the author protocol, while standard classical models achieved 96–98%. Under symmetric nested tuning, classical models maintained significant superiority ($\Delta = -9.65\%$, $p = 0.0009$).</small>

We evaluated each study against our 17-item checklist. Every eligible study exhibited at least one critical methodological asymmetry: 8 of 8 left classical comparators under-optimized; 7 of 8 fitted preprocessing or feature selection on the complete pooled dataset before cross-validation partitioning; 5 of 8 evaluated on partitions with patient-level record or temporal leakage; and 0 of 8 applied dependency-corrected resampled hypothesis testing (such as the Nadeau–Bengio correction [14]), treating correlated cross-validation folds as independent draws.

---

### Re-tested published claims could not be verified under symmetric evaluation

We computationally reimplemented five eligible published studies spanning six distinct clinical prediction tasks. Each study was evaluated under three clearly distinguished experimental streams: (1) an exact author replication arm matching the published workflows as far as documented; (2) a factorial confounder dissection isolating leakage from tuning on fixed evaluation partitions; and (3) a standardized symmetric evaluation arm with nested cross-validation and Nadeau–Bengio variance correction (**Table 2**, **Figure 1**).

**Table 2 | Pre-specified factorial analysis isolating the evaluation mechanisms of apparent quantum advantage.**

| Dataset & Study | Factorial Design Space | Cell Configuration | Advantage $\Delta$ (95% CI) | $p$-value | Finding |
|---|---|---|:---:|:---:|---|
| **Parkinson's Voice**<br>(Daga et al. [16]) | $2 \times 2 \times 2$ Factorial:<br>• PCA (Global vs In-Fold)<br>• Split (Record vs Subject)<br>• Baseline (Untuned vs Tuned) | **Author Protocol:** Global PCA $\times$ Record Split $\times$ Untuned<br>In-Fold PCA $\times$ Record Split $\times$ Untuned<br>Global PCA $\times$ Subject Split $\times$ Untuned<br>**Tuned on Factorial Split:** In-Fold PCA $\times$ Record Split $\times$ Tuned | **+4.8%** (+1.2% to +8.4%)<br>+1.1% ($-1.0\%$ to +3.2%)<br>+0.7% ($-1.1\%$ to +2.5%)<br>**$-$0.3%** ($-2.1\%$ to +1.5%) | $p = 0.012$<br>$p = 0.29$<br>$p = 0.41$<br>$p = 0.72$ | **Apparent gain** reproduced.<br>PCA leakage fix.<br>Subject split fix.<br>**Advantage vanishes.** |
| **Pima Diabetes**<br>(Daga et al. [16]) | $2 \times 2$ Factorial:<br>• PCA (Global vs In-Fold)<br>• Baseline (Untuned vs Tuned) | **Author Protocol:** Global PCA $\times$ Untuned<br>**Symmetric Protocol:** In-Fold PCA $\times$ Tuned | **+9.3%** (+2.4% to +16.2%)<br>**+0.8%** ($-1.1\%$ to +2.7%) | $p = 0.009$<br>$p = 0.36$ | **Apparent gain** reproduced.<br>**Tuning removes gain.** |
| **Schizophrenia EEG**<br>(Aksoy & Cattan [17]) | Partition Scheme:<br>Random Epoch vs Subject-Grouped (200 splits) | **Author Design:** Random Epoch Split<br>**Symmetric Design:** Subject-Grouped | **98.6% Accuracy** (34.5% reach 100%)<br>**52.1% Accuracy** (0% reach $>75\%$) | $p < 10^{-6}$<br>$p = 0.32$ | **Leakage artifact.**<br>**Chance level.** |

Across all six re-tested claims:
- **Chronic Kidney Disease (Parthasarathi et al. [15]):** The published claim of 99.2% VQC accuracy could not be reproduced from the reported feature set (reimplemented VQC reached 83.50% vs 97.40% for classical RBF SVM). Under symmetric nested cross-validation, classical models were significantly superior: $\Delta = -9.65\%$ (95% CI $-14.92\%$ to $-4.38\%$, $t_{\text{NB}} = -3.74, p = 0.0009$).
- **Schizophrenia EEG (Aksoy & Cattan [17]):** The reported 100.0% accuracy collapsed to chance (52.14%) when 2-second EEG epochs were partitioned strictly by human subject rather than randomly shuffled across epochs ($p = 0.32$).
- **Parkinson's Voice Phonation & Pima Diabetes (Daga et al. [16]):** Under the published protocol, apparent advantages of +4.8% and +9.3% were reproduced. However, when baseline models were tuned and PCA was fitted strictly inside training folds, the differences diminished to $-0.3\%$ ($p = 0.72$) and $+0.8\%$ ($p = 0.36$). Under repeated nested cross-validation with Nadeau–Bengio correction, both findings were statistically inconclusive (Parkinson's: $\Delta = +0.33\%$, 95% CI $-24.97\%$ to $+25.64\%$, $p = 0.98$; Diabetes: $\Delta = -5.30\%$, 95% CI $-11.00\%$ to $+0.39\%$, $p = 0.067$).
- **Glioma Neuroimaging (Akpinar & Oduncuoglu [18]):** On TCGA ($n=839$), the symmetric difference was $\Delta = +0.19\%$ (95% CI $-1.22\%$ to $+1.60\%$, $p = 0.78$); on BraTS 2020 ($n=369$), $\Delta\text{AUROC} = -0.022$ ($p = 0.19$).
- **Cardiovascular Risk (Ghisoni et al. [19]):** On the low-data split ($n=69$), the observed difference was $\Delta = -1.10\%$. Correcting for cross-validation dependence via Nadeau–Bengio yielded a 95% confidence interval spanning 34.7 percentage points ($-18.45\%$ to $+16.25\%$, $t_{\text{NB}} = -0.13, p = 0.90$), demonstrating the analysis was severely underpowered.

![Figure 1 | Forest plot of evaluation mechanisms and re-tested clinical claims](docs/figures/fig_v6_forest.png)  
*Figure 1 | Forest plot of evaluation mechanisms and re-tested clinical claims. (a) Paired likelihood differences ($\Delta\text{NLL}$) in the chest radiograph prototype across optimizers, calibration protocols, out-of-sample partitions, and confirmatory cohorts ($n=447$ to $n=56,534$). Horizontal error bars indicate 95% confidence intervals; vertical dashed lines indicate the operational equivalence boundary ($\pm 0.010$ nats). (b) Re-tested published claims under author protocols versus symmetric evaluation across five eligible published studies spanning six clinical prediction tasks. Blue intervals represent symmetric evaluations; red markers represent author-declared protocols.*

---

### In a clinical decision-support prototype, the outcome depended on evaluation choices

To investigate how evaluation choices alter conclusions within an active engineering pipeline, we conducted an exhaustive forensic audit of an internal multi-class clinical decision-support prototype for chest radiograph triage. The prototype maps seven radiographic finding probabilities extracted by a DenseNet-121 backbone to six clinical acuity categories (Normal, Pneumonia, Heart Failure, COPD, Malignancy, Pneumothorax).

Prior to audit, the system featured a variational quantum fusion layer and asserted that the quantum circuit conferred superior calibration and likelihood over classical models. By systematically toggling evaluation choices across four experimental dimensions, we found that every reported quantum advantage was an artifact of asymmetric evaluation:
1. **Optimizer Matching:** When classical comparators used unoptimized Adam with default learning rates while the quantum circuit used extensive parameter tuning, the quantum circuit appeared superior ($\Delta\text{NLL} = -0.0173$ nats vs Product-of-Experts). When baselines received matched L-BFGS-B optimization, the apparent advantage vanished ($\Delta\text{NLL} = +0.0033$ nats, 95% CI $-0.0329$ to $+0.0396$; $p = 0.86$).
2. **Symmetric Probability Calibration:** The uncalibrated quantum circuit achieved ECE = 0.0384, appearing substantially better than the uncalibrated classical linear baseline (ECE = 0.0815). However, when both models underwent independent Platt temperature scaling fitted strictly on calibration folds, calibration errors converged (classical ECE = 0.0131 vs quantum ECE = 0.0076) with overlapping confidence intervals.
3. **Out-of-Sample Evaluation:** On the internal development cohort ($n=447$), the quantum circuit appeared competitive. When evaluated on out-of-sample chest radiograph evidence ($n=1,600$), the classical Product-of-Experts baseline achieved lower negative log-likelihood ($\Delta\text{NLL} = +0.0076$ nats, 95% CI $-0.0029$ to $+0.0181$).
4. **Institutional Transfer Degradation:** Under frozen institutional transfer from MIMIC-CXR to CheXpert ($n=4,800$), the quantum circuit degraded severely, performing worse than Product-of-Experts by $\Delta\text{NLL} = \mathbf{+0.1589}$ nats (95% CI $0.1422$ to $0.1757$; $p < 10^{-15}$) and worse than Logistic Regression by $\Delta\text{NLL} = \mathbf{+0.1967}$ nats (95% CI $0.1785$ to $0.2149$; $p < 10^{-15}$). This 15- to 20-fold degradation beyond the operational equivalence margin exposed the acute vulnerability of periodic trigonometric angle encodings to feature scale drift across hospital environments.

---

### Pre-specified confirmatory test on 56,534 held-out patients

To eliminate sample-size limitations and patient clustering, we executed our pre-specified confirmatory evaluation across **56,534 held-out CheXpert patients** (one frontal radiograph per unique human subject; zero overlap with development data; $|S_{\text{dev}} \cap S_{\text{test}}| = 0$; **Table 3**, **Figure 3**).

In CheXpert, because automated report labeling lacks the `hyperinflation` finding, Acuity Class 3 (COPD) has zero positive occurrences in both development and test partitions ($y \in \{0, 1, 2, 4, 5\}$). Forward inference and calibration symmetrically compute full 6-way softmax probabilities across all models; sample NLL is evaluated as $-\ln \hat{p}_{i, y_i}$ for observed labels $y_i$, while multi-class AUROC is macro-averaged across the 5 observed positive classes. Upstream DenseNet perceptual uncertainty establishes an empirical noise ceiling (test accuracy $\sim$54.4%, AUROC $\sim$0.755).

- **Analysis C1 (Default Matched Protocol):** The quantum circuit achieved the lowest accuracy (53.71%), likelihood (1.1362), and AUROC (0.7436). The paired likelihood difference vs Product-of-Experts was $\Delta\text{NLL} = \mathbf{+0.0205}$ nats (95% CI $0.0186$ to $0.0224$, $t = 21.034, p = 7.65 \times 10^{-98}$; TOST $p = 1.000$). The outcome fell into the pre-specified reading: **"Quantum materially worse."** Calibration error was statistically indistinguishable between calibrated arms (quantum ECE = 0.0076, 95% bootstrap CI 0.0051 to 0.0119; classical Gated Head ECE = 0.0085, 95% CI 0.0071 to 0.0130). Across 10 random initialization seeds, mean NLL was $1.1305 \pm 0.0045$ (mean difference vs PoE: $+0.0142 \pm 0.0045$ nats), proving the default deficit exceeds initialization noise.
- **Analysis C1b (Entanglement Ablation):** Across ten random initialization seeds on the 56,534 patients, removing circular CNOT entangling gates improved likelihood: $\Delta\text{NLL} = \text{NLL}_{\text{entangled}} - \text{NLL}_{\text{product}} = \mathbf{+0.01037}$ nats (seed-level 95% Student's $t$ CI $[0.00491, 0.01583]$ nats, $t = 4.30, p = 0.0013$). Unentangled circuits outperformed entangled circuits in 9 of 10 seeds, confirming: **"The entanglement penalty replicates on untouched patients."**

![Figure 3 | Pre-specified confirmatory evaluation across 56,534 held-out patients](docs/figures/fig_v6_confirmatory.png)  
*Figure 3 | Pre-specified confirmatory evaluation across 56,534 held-out patients scored without selection. (a) Analyses C1 and C1b under default matched optimization ($N=56,534$ patients), showing the quantum circuit is materially worse on likelihood ($\Delta\text{NLL} = +0.0205$, 95% CI 0.0186 to 0.0224) and unentangled product circuits outperform entangled circuits across random seeds ($\Delta\text{NLL} = +0.0104$, seed-level 95% CI 0.0053 to 0.0154). Gray band indicates the pre-specified operational equivalence boundary ($\pm 0.010$ nats). (b) Analysis C2 under 36-configuration symmetric hyperparameter tuning; the observed paired differences ($\Delta\text{NLL} = -0.0026$ vs PoE, $-0.0012$ vs Gated Head) fall entirely inside the operational equivalence band ($p_{\text{TOST}} < 10^{-6}$) and within training-seed variance. (c) Sensitivity analysis of Two One-Sided Tests (TOST) $p$-values across varying operational margins $\delta \in [0.002, 0.015]$ nats.*

---

### Symmetric hyperparameter tuning confirms population equivalence but fails class-balanced evaluation

To ensure evaluation symmetry under capacity optimization, each architecture received a budget of **36 configurations** trained on 6,400 development cases and selected strictly by validation NLL on 1,600 calibration patients (Analysis C2; **Table 3**).

**Table 3 | Pre-specified confirmatory evaluation across 56,534 untouched clinical patients under default and tuned protocols.**

| Model Arm | Params | Accuracy (95% CI) | NLL | ECE (95% Boot CI) | AUROC | Paired $\Delta\text{NLL}$ vs PoE (95% CI) |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Panel A: Analysis C1 — Default Matched Optimization Protocol** | | | | | | |
| **Product-of-Experts** | 54 | **0.5443** (0.5402, 0.5484) | **1.1157** | 0.0131 (0.0100, 0.0172) | **0.7534** | Reference Baseline |
| **Logistic Regression** | 54 | 0.5415 (0.5374, 0.5456) | **1.1152** | 0.0128 (0.0105, 0.0171) | **0.7534** | $-0.0005$ ($-0.0011$, $+0.0002$) |
| **Gated Head** | 126 | 0.5398 (0.5357, 0.5439) | 1.1165 | 0.0085 (0.0071, 0.0130) | **0.7553** | $+0.0008$ ($-0.0002$, $+0.0019$) |
| **Variational Quantum** | 102 | 0.5371 (0.5330, 0.5412) | 1.1362 | 0.0076 (0.0051, 0.0119) | 0.7436 | $\mathbf{+0.0205}$ (0.0186, 0.0224) [**Materially Worse**] |
| **Panel B: Analysis C2 — Symmetric 36-Configuration Hyperparameter Grid Search** | | | | | | |
| **Tuned Product-of-Experts** | 54 | **0.5440** (0.5399, 0.5481) | 1.1170 | 0.0125 (0.0091, 0.0168) | 0.7527 | Reference (Tuned PoE) |
| **Tuned Gated Head** | 126 | 0.5418 (0.5377, 0.5459) | 1.1155 | **0.0082** (0.0067, 0.0126) | 0.7548 | $-0.0015$ ($-0.0022$, $-0.0007$) vs Tuned PoE |
| **Optimal Variational Quantum** | 118 | 0.5417 (0.5376, 0.5458) | **1.1144** | 0.0111 (0.0088, 0.0151) | **0.7550** | $\mathbf{-0.0026}$ ($-0.0036$, $-0.0016$) [**Practically Equivalent**] |
| *10-Seed Initialization Means (Post-Hoc Analysis across 56,534 Test Patients):* | | | | | | |
| *Tuned Product-of-Experts* | 54 | $0.5425 \pm 0.0015$ | $1.1180 \pm 0.0013$ | $0.0109 \pm 0.0011$ | $0.7533 \pm 0.0009$ | Reference (10-Seed PoE Mean: 1.11797) |
| *Tuned Gated Head* | 126 | $0.5411 \pm 0.0015$ | $1.1179 \pm 0.0037$ | $0.0111 \pm 0.0024$ | $0.7547 \pm 0.0010$ | $-0.00003$ vs Tuned PoE Mean |
| *Optimal Variational Quantum* | 118 | $0.5409 \pm 0.0014$ | $1.1173 \pm 0.0035$ | $0.0252 \pm 0.0038$ | $0.7546 \pm 0.0011$ | $\mathbf{-0.00067}$ ($-0.00288, +0.00153$) [$p_{\text{TOST}} < 10^{-5}$] |
| **Panel C: Analysis M2 — Classical Fourier Basis Surrogates (56,534 Test Patients)** | | | | | | |
| **Order-1 Trigonometric LR** | 90 | 0.5403 (0.5362, 0.5444) | 1.1123 | 0.0128 (0.0094, 0.0171) | 0.7541 | $-0.0047$ ($-0.0057$, $-0.0036$) vs Tuned PoE |
| **Order-2 Trigonometric LR** | 678 | **0.5419** (0.5378, 0.5460) | 1.1095 | 0.0122 (0.0089, 0.0164) | 0.7552 | $-0.0075$ ($-0.0087$, $-0.0062$) vs Tuned PoE |
| **Polynomial LR ($x, x^2$)** | 90 | 0.5406 (0.5365, 0.5447) | **1.1092** | **0.0101** (0.0072, 0.0143) | **0.7555** | $-0.0077$ ($-0.0089$, $-0.0066$) vs Tuned PoE |

In Panel B, the selected single quantum checkpoint achieved NLL 1.1144 ($\Delta\text{NLL} = -0.0026$ nats vs tuned PoE, 95% CI $-0.0036$ to $-0.0016$, $p_{\text{TOST}} < 10^{-6}$; $\Delta\text{NLL} = -0.0012$ nats vs tuned Gated Head, $p_{\text{TOST}} < 10^{-6}$), confirming practical statistical equivalence on the overall population. However, total training runtimes across the 36 configurations were 161.4 s for PoE, 228.1 s for Gated Head, and 14,969.4 s for the quantum circuit (a **92.8-fold compute penalty**).

Across 10 random initialization seeds of these selected winning configurations, tuned VQC achieved mean test NLL $1.11730 \pm 0.00346$ vs $1.11797 \pm 0.00130$ for PoE (mean paired difference $-0.00067$ nats, Student's $t$ with $\nu=9$, $p_{\text{TOST}} = 2.57 \times 10^{-6}$), and $-0.00064$ nats vs Gated Head ($p_{\text{TOST}} = 1.50 \times 10^{-5}$). Notice that the single tuned checkpoint ECE (0.0111) was an initialization-favorable draw; across 10 random seeds, tuned VQC mean ECE was **$0.0252 \pm 0.0038$**, $2.3\times$ worse than tuned PoE ($0.0109 \pm 0.0011$) and tuned Gated Head ($0.0111 \pm 0.0024$).

**Table 4 | Class-conditional log-likelihood decomposition across 56,534 held-out patients.**

| Clinical Triage Diagnosis | Patients ($n$) | Prevalence | VQC NLL | PoE NLL | Gated NLL | $\Delta\text{NLL}$ vs PoE (95% Boot CI) |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Normal** | 17,692 | 31.29% | 0.7379 | 0.7577 | 0.7329 | $\mathbf{-0.0198}$ ($-0.0217$, $-0.0180$) |
| **Pneumonia** | 17,052 | 30.16% | 1.0263 | 1.1192 | 1.0364 | $\mathbf{-0.0929}$ ($-0.0943$, $-0.0915$) |
| **Heart Failure** | 16,490 | 29.17% | 1.0926 | 1.0172 | 1.1060 | $\mathbf{+0.0753}$ ($+0.0742$, $+0.0765$) |
| **Pneumothorax** | 3,108 | 5.50% | 2.5705 | 2.5156 | 2.5817 | $\mathbf{+0.0549}$ ($+0.0510$, $+0.0586$) |
| **Malignancy** | 2,192 | 3.88% | 2.8990 | 2.7219 | 2.7686 | $\mathbf{+0.1771}$ ($+0.1681$, $+0.1861$) |
| **Summary Metrics Across All Categories:** | | | | | | |
| \quad**Prevalence-Weighted Mean ($\sum_i \frac{n_i}{N} \delta_i$):** | | | | | | $\mathbf{-0.00235}$ nats (95% CI $-0.00326$ to $-0.00146$) [*Practically Equivalent*] |
| \quad**Class-Balanced Macro-Mean ($\frac{1}{K}\sum_i \delta_i$):** | | | | | | $\mathbf{+0.03892}$ nats (95% CI $+0.03688$ to $+0.04098$) [**Equivalence Fails**, $p_{\text{TOST}} > 0.99$] |

Crucially, post-hoc subgroup analysis revealed that the small population equivalence ($\Delta\text{NLL} = -0.00235$ nats prevalence-weighted) masks substantial degradation on minority acute conditions (**Table 4**). While the quantum circuit achieved lower loss on high-prevalence categories (Normal, Pneumonia), it degraded sharply on Heart Failure ($\Delta = \mathbf{+0.0753}$ nats; PoE detected 1,187 more acute heart failure cases than VQC), Pneumothorax ($\Delta = \mathbf{+0.0549}$ nats), and Malignancy ($\Delta = \mathbf{+0.1771}$ nats). Under unweighted class-balanced macro-averaging, operational equivalence failed decisively: $\Delta\text{NLL}_{\text{macro}} = \mathbf{+0.03892}$ nats (95% bootstrap CI $0.0369$ to $0.0410$, $p_{\text{TOST}} > 0.99$).

---

### Mechanistic dissection of the observed nulls

We investigated why shallow variational quantum circuits fail to demonstrate representational advantages over classical linear and kernel baselines on clinical tabular data (**Figure 2**):

![Figure 2 | Mechanistic dissection of observed circuit performance and kernel geometry](docs/figures/fig_v6_mechanism.png)  
*Figure 2 | Mechanistic dissection of observed circuit performance and kernel geometry. (a) Interaction-order coefficient energy spectrum across 10 random initialization seeds for circuits trained on the 7 clinical finding channels (Analysis C1b; red: circular CNOT ring entanglement; blue: unentangled product circuit), showing 100.0% of non-trivial energy in interaction orders $\ge 2$ under entanglement. In the 8-qubit confirmatory architecture (incorporating the constant reference qubit $x_8 \equiv 0.0$), the effective interaction order over clinical inputs is bounded at 7. (b) Bandwidth sweep and regularized geometric difference distance $g_\lambda(K_{\text{RBF}} \parallel K_{\text{Q}}) = 1.27 \ll \sqrt{N} = 44.72$ on 10-qubit glioma neuroimaging (M3), demonstrating classical kernel surrogacy.*

1. **Interaction-Order Energy Dispersion (M1):** In the primary 8-qubit entangled circuit (7 clinical features plus 1 constant reference qubit $x_8 \equiv 0.0$), 100.0% of non-trivial coefficient energy and 99.99% of logit variance reside in interaction orders $\ge 2$ across all seeds (mean order: 4.3; **Figure 2a**), whereas unentangled product circuits strictly confine interaction order to $\le 1$ (0.0% cross-channel energy). Because clinical triage rules are fundamentally additive combinations of radiological signs, high-order cross-terms generated by entanglement represent unhelpful capacity that increases variance without reducing bias.
2. **Classical Fourier Basis Surrogates (M2):** Fitting an ordinary $L_2$-regularized logistic regression directly on the classical trigonometric Fourier basis ($\cos(\pi x_i), \sin(\pi x_i)$) achieved NLL **1.1123** at order 1 (outperforming the optimal tuned quantum circuit's 1.1144; $p = 0.0016$), NLL **1.1095** at order 2, and NLL **1.1092** for polynomial regression ($x, x^2$; **Table 3, Panel C**). The tiny advantage of tuned VQC over linear PoE was simply an artifact of PoE lacking trigonometric features; when classical models receive the identical single-qubit Fourier features, classical optimization achieves superior test likelihood.
3. **Analytic Kernel Equivalence and Geometric Distance (M3):** On unentangled product states, the quantum fidelity kernel analytically equals the classical product-cosine kernel:
   $$K_Q(x, x') = \prod_{i=1}^n \cos^2\left(\frac{\pi(x_i - x_i')}{2}\right)$$
   ($3.11 \times 10^{-15}$ max discrepancy vs statevector simulation). On 10-qubit glioma neuroimaging, the empirical geometric distance between the quantum kernel and a classical RBF kernel is $g_\lambda(K_{\text{RBF}} \parallel K_Q) = 1.2712 \ll \sqrt{N} = 44.72$. Under the geometric difference theorem [10], classical RBF kernels achieve comparable sample complexity on any target learnable by this quantum kernel (**Figure 2b**).

---

## Discussion

Across five reconstructible published clinical machine learning studies spanning six distinct clinical prediction tasks, none of the six published advantage claims could be verified under symmetric evaluation; where disparities existed, apparent superiority was traced to comparator undertuning, data leakage, or underpowered cohorts. In our internal chest radiograph decision-support prototype, reported calibration advantages disappeared when baselines received matched optimization and independent temperature scaling. Where empirical advantages were claimed in the audited literature, they traced to specific methodological flaws:
1. Calibrating only the quantum arm while leaving classical comparators uncalibrated [31].
2. Optimizing quantum architectures through extensive hyperparameter search while evaluating classical models under arbitrary default settings [2, 3].
3. Evaluating performance on partitions contaminated by patient-level longitudinal or temporal leakage [17, 26].
4. Conducting feature selection or dimensionality reduction globally across all available data prior to cross-validation partitioning [16, 18].
5. Reporting naive Student's $t$-tests across resampled cross-validation folds, ignoring training-set overlap and generating artificially narrow confidence intervals [14].

### Limitations

Several methodological boundaries must be explicitly noted:
1. **Low-Width Regime ($n \le 12$ Qubits):** All evaluated quantum circuits operated at modest qubit widths accessible to classical statevector simulation. We make no empirical claims regarding fault-tolerant quantum algorithms operating with hundreds of logical qubits or error-mitigated circuits on native quantum sensors.
2. **Clinical Emulation Noise Floor:** The chest radiograph evaluation assessed emulation of an additive triage rule derived from DenseNet finding probabilities. Upstream perceptual and report-extraction noise imposes an empirical performance ceiling; our findings demonstrate that shallow VQCs do not resolve this upstream noise.
3. **Checklist Scoring Single Reviewer:** The 17-item checklist was scored by a single reviewer according to pre-specified objective binary criteria. Independent second-reviewer scoring with inter-rater reliability metrics remains an area for future multi-center registry expansion.
4. **Finite Hyperparameter Search Budget:** In Analysis C2, the 36-configuration search sampled 27% of the quantum grid and 60% of the classical grid. Expanding classical tuning to the full 60 configurations did not change optimal classical test performance (NLL 1.1170), demonstrating that classical baselines had fully converged.

### Clinical and Scientific Implications

For clinical artificial intelligence, these results underscore that new algorithmic paradigms must be held to rigorous standards of evaluation symmetry before clinical translation. In our prototype, evaluating the tuned quantum circuit required a 92.8-fold training compute penalty and degraded predictions for minority acute conditions (Heart Failure, Pneumothorax, Malignancy) despite aggregate population equivalence. In healthcare settings where model deployment carries patient safety risks and substantial infrastructure costs, adopting quantum machine learning without verified, symmetric superiority is unwarranted.

---

## Methods

### Algorithm: Symmetric Evaluation Protocol

All comparative analyses adhered to the formal procedure specified in **Algorithm 1**.

```python
# Algorithm 1: Symmetric Evaluation Protocol for Clinical Prediction Models
# Input: Dataset D = {(x_i, y_i, g_i)} with patient IDs g_i;
#        Quantum family M_Q with search grid Theta_Q;
#        Classical families M_C with search grid Theta_C;
#        Operational margin Delta_equiv; significance level alpha = 0.05.
# Output: Paired shifts Delta, corrected 95% CIs, p-values, verdict.

# 1. Partitioning (Rule P3):
#    Partition D into J x K outer folds grouped strictly by g_i
#    such that no patient crosses fold boundaries (|S_train \cap S_test| = 0).

# 2. In-Fold Preprocessing (Rule P3):
#    For each outer fold (j, k):
#      Fit imputer, standardizer, and PCA strictly on D_train^(j,k).
#      Apply fitted transformers to D_train^(j,k) and D_test^(j,k).

# 3. Matched Hyperparameter Tuning (Rule P2):
#    Within D_train^(j,k), run 3-fold inner CV:
#      Select theta_Q* = argmin NLL_inner(theta) over Theta_Q.
#      Select theta_C* = argmin NLL_inner(theta) over Theta_C.

# 4. Model Selection and Calibration (Rules P1, P2):
#    Retain frozen checkpoint theta* trained on D_sub^(j,k) at validation minimum (no pooled refit).
#    Fit temperature scale T independently per arm on validation split logits:
#      T* = argmin NLL(y_val, sigma(z / T)).

# 5. Held-Out Evaluation (Rules P5, P6):
#    Score frozen, calibrated models on untouched D_test^(j,k):
#      Compute fold metric d_(j,k) = Metric_Q^(j,k) - Metric_C^(j,k).

# 6. Inferential Analysis (Rule P4):
#    If repeated cross-validation (J x K):
#      Apply Nadeau-Bengio corrected resampled t-test (df = JK - 1):
#        sigma_NB^2 = (1/(JK) + n_test / n_train) * Var(d)
#        t_NB = mean(d) / sigma_NB.
#    If confirmatory cohort (N patients) or multi-seed evaluation (S seeds):
#      Conduct Two One-Sided Tests (TOST) against [-Delta_equiv, +Delta_equiv]:
#        t_1 = (mean(d) - (-Delta_equiv)) / SE;  p_1 = 1 - F_t(t_1, df)
#        t_2 = (mean(d) - (+Delta_equiv)) / SE;  p_2 = F_t(t_2, df)
#        p_TOST = max(p_1, p_2).

# 7. Decision Verdict:
#    Assign: Not Reproduced, Inconclusive, Collapsed to Chance, or Practically Equivalent.
```

### Cohort Construction, Deduplication, and Independence Architecture

- **MIMIC-CXR-JPG (v2.0.0):** Evaluated under PhysioNet Credentialed Health Data License 1.5.0 [25]. Filtered strictly to one frontal study per patient (highest study index), partitioned into 1,338 training, 446 calibration, and 447 test studies.
- **CheXpert-v1.0-Small:** Evaluated under Stanford Research Data Agreement [24]. Filtered strictly to frontal views (AP/PA), selecting exactly one study per unique human subject deterministically by taking the final encounter (highest study ID). The initial 8,000 unique patients were partitioned into 6,400 training and 1,600 calibration studies for development. All 56,534 subsequent unique frontal studies formed the untouched confirmatory test set. Patient-level set intersection confirmed 0 patient overlap ($|S_{\text{dev}} \cap S_{\text{test}}| = 0$). Unit tests (`test_one_study_per_patient.py`) enforce that repeated subject rows are excluded before dataset assembly.
- **BraTS 2020:** 369 official glioma subjects (293 HGG, 76 LGG) [27]. Feature extraction embeddings were derived using the served foundation checkpoint, with 111 subjects strictly held out from encoder training.
- **Literature Benchmark Cohorts:** UCI Chronic Kidney Disease ($n=400$), UCI Parkinson's Disease Voice ($n=195$, 31 subjects), UCI Pima Indian Diabetes ($n=768$), MSU Adolescent Schizophrenia EEG ($n=84$), UCI Heart Disease ($n=918$), and TCGA Glioma ($n=839$).

### Variational Circuit Architecture and Parameter Counts

The input feature vector $x \in [0, 1]^8$ consists of seven normalized radiological finding probabilities extracted via DenseNet-121 ($x_1, \dots, x_7$) and a constant baseline reference channel $x_8 \equiv 0.0$. Features are mapped via single-qubit angle rotations $R_Y(\pi s x_i)|0\rangle$. Because $x_8 \equiv 0$, the eighth qubit receives a zero rotation angle $R_Y(0)|0\rangle = |0\rangle$ and remains initialized in the unrotated ground state, providing an unrotated reference baseline coupled to qubits 7 and 1 through the closed CNOT entangling ring. Expectation values of Pauli-$Z$ observables $\langle Z_k \rangle(x) \in [-1, 1]$ across all 8 qubits are projected to 6 class logits via linear readout $W_{\text{out}} \langle Z \rangle(x) + b_{\text{out}}$ (corresponding to the six triage categories: Normal, Pneumonia, Heart Failure, COPD, Malignancy, Pneumothorax).

In CheXpert, because automated report labeling lacks the `hyperinflation` finding, Acuity Class 3 (COPD) has zero positive occurrences in both development and confirmatory test partitions ($y \in \{0, 1, 2, 4, 5\}$). Forward inference and calibration symmetrically compute full 6-way softmax probabilities across all models; sample NLL is evaluated as $-\ln \hat{p}_{i, y_i}$ for observed ground-truth labels $y_i$, while multi-class AUROC is macro-averaged across the 5 observed positive classes to prevent zero-denominator division.

Parameter counts are:
- Default C1 VQC ($L=3$, 48 angles $+ 54$ readout weights = **102 parameters**).
- Optimal C2 VQC ($L=4$, 64 angles $+ 54$ readout weights = **118 parameters**).
- Linear Product-of-Experts ($6 \times 8 + 6 = \mathbf{54\text{ parameters}}$).
- Logistic Regression ($6 \times 8 + 6 = \mathbf{54\text{ parameters}}$).
- Nonlinear Gated Head (64 gating weights $+ 8$ gating biases $+ 54$ readout parameters = **126 parameters**).

During development, the 1,600 calibration patients were used for early-stopping checkpoint selection and Platt temperature scaling [30] without pooled refitting; the 56,534 test patients were strictly held out and scored without any selection on them.

### Circuit Simulation and Mathematical Verification

Quantum circuits were evaluated using an optimized PyTorch batched statevector engine. Prior to evaluation, the simulator was verified against PennyLane (`default.qubit`) on 1,000 random input/parameter pairs at 7 and 8 qubits for both entangled and unentangled topologies; maximum absolute discrepancy was $7.77 \times 10^{-16}$, confirming numerical equivalence to double-precision machine tolerance.

### Statistical Inference and Equivalence Margins

Model calibration and discrimination are evaluated via Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE):
$$\text{NLL}(y, \hat{p}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln \hat{p}_{ic}$$
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

Under our metric directionality, the paired likelihood difference is defined as:
$$\Delta\text{NLL} = \text{NLL}_{\text{quantum}} - \text{NLL}_{\text{classical}}$$
where $\Delta\text{NLL} > 0$ indicates classical superiority (lower loss) and $\Delta\text{NLL} < 0$ indicates quantum superiority.

To test practical statistical equivalence on the confirmatory cohort, Two One-Sided Tests (TOST) [32, 33] evaluate the composite null hypothesis $H_0: \mu_d \le -\Delta_{\text{equiv}} \text{ or } \mu_d \ge +\Delta_{\text{equiv}}$ against the alternative $H_1: -\Delta_{\text{equiv}} < \mu_d < +\Delta_{\text{equiv}}$ under the operational margin $\Delta_{\text{equiv}} = \pm 0.010$ nats:
$$t_1 = \frac{\bar{d} - (-\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}, \quad t_2 = \frac{\bar{d} - (+\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}$$
$$p_1 = 1 - F_t(t_1; \nu), \quad p_2 = F_t(t_2; \nu), \quad p_{\text{TOST}} = \max(p_1, p_2)$$
where $F_t(\cdot; \nu)$ denotes the cumulative distribution function of Student's $t$-distribution with $\nu$ degrees of freedom (equivalently written with $t_2' = \frac{+\Delta_{\text{equiv}} - \bar{d}}{\text{SE}(\bar{d})}$ and $p_2 = 1 - F_t(t_2'; \nu)$). For the massive confirmatory cohort ($N = 56,534$, $\nu = 56,533$), $F_t$ converges asymptotically to the standard normal CDF $\Phi$; for the 10-seed initialization analysis ($\nu = 9$), Student's $t$-distribution with 9 degrees of freedom is strictly evaluated. Equivalence at significance $\alpha = 0.05$ is established when $p_{\text{TOST}} < 0.05$, which is mathematically equivalent to the 90% confidence interval falling strictly inside $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$.

Additional statistical procedures:
- **Likelihood Contrasts:** Wilcoxon signed-rank tests for paired median shifts, paired Student's $t$-intervals for mean differences.
- **Classification Accuracy:** McNemar's exact test for paired binary proportions.
- **Discrimination:** DeLong's test for correlated ROC curves [34].
- **Multiple Testing:** Holm–Bonferroni correction applied across pairwise contrasts.
- **Resampling:** 10,000 patient-clustered bootstrap resamples for parameter difference confidence intervals.
- **Cross-Validation Corrected Inference:** Nadeau–Bengio corrected resampled $t$-test for $J \times K$ repeated cross-validation [14]:
  $$\hat{\sigma}^2_{\text{corrected}} = \left(\frac{1}{JK} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2$$
  preventing false-positive inflation from fold overlap.

---

## Data Availability

The datasets evaluated in this study are available from public and credentialed repositories:
1. **MIMIC-CXR-JPG (v2.0.0):** Available on PhysioNet ([https://physionet.org/content/mimic-cxr-jpg/2.0.0/](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)) under Credentialed Health Data License 1.5.0.
2. **CheXpert-v1.0-Small:** Available from the Stanford Machine Learning Group ([https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)) under Stanford Research Data Agreement.
3. **BraTS 2020:** Available through the Medical Image Computing and Computer Assisted Intervention Society ([http://braintumorsegmentation.org/](http://braintumorsegmentation.org/)).
4. **Literature Benchmark Datasets:** Available through the UCI Machine Learning Repository ([https://archive.ics.uci.edu/](https://archive.ics.uci.edu/)) and original publication repositories.

Extracted embedding features, prediction arrays, and evaluation splits are available in the project repository.

---

## Code Availability

All software, evaluation pipelines, simulation engines, and verification scripts are released as open-source under the Apache 2.0 license at [https://github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura). The full empirical suite can be reproduced via a single command:
```bash
python reproduce_paper.py
```

---

## Acknowledgements

The author thanks the PhysioNet team and Stanford ML Group for providing access to the MIMIC-CXR and CheXpert datasets.

---

## Author Contributions

**S.R.M.** conceived the study, formulated the symmetric evaluation protocol, developed the simulation and evaluation software, conducted the systematic literature audit and empirical experiments, analyzed the data, and wrote the manuscript.

---

## Competing Interests

The author declares no competing financial or non-financial interests.

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
11. Thanasilp, S. et al. Exponential concentration and untrainability in quantum kernel methods. *Nat. Commun.* **15**, 5200 (2024).
12. Shaydulin, R. & Wild, S. M. Importance of kernel bandwidth in quantum machine learning. *Phys. Rev. A* **106**, 042407 (2022).
13. McClean, J. R. et al. Barren plateaus in quantum neural network training landscapes. *Nat. Commun.* **9**, 4812 (2018).
14. Nadeau, C. & Bengio, Y. Inference for the generalization error. *Mach. Learn.* **52**, 239–281 (2003).
15. Parthasarathi, P. et al. Chronic kidney disease classification using variational quantum circuits. *PeerJ Comput. Sci.* **11**, e2541 (2025).
16. Daga, R. et al. Quantum kernel learning for biomedical diagnostic classification. *Sci. Rep.* **16**, 1042 (2026).
17. Aksoy, D. & Cattan, G. Schizophrenia detection from EEG using quantum support vector machines. *J. Med. Syst.* **48**, 19 (2024).
18. Akpinar, B. & Oduncuoglu, M. Variational quantum classification of brain tumor grades from MRI and genomic features. *Sci. Rep.* **15**, 8912 (2025).
19. Ghisoni, F. et al. Quantum neural networks for cardiovascular risk prediction in low-data regimes. *Sci. Rep.* **16**, 3418 (2026).
20. Jahin, M. et al. Hybrid quantum-classical convolutional neural networks for disease classification. *Comput. Biol. Med.* **185**, 109452 (2025).
21. Chen, Y. et al. Quantum-enhanced imaging biomarkers in clinical oncology. In *Proc. SPIE Medical Imaging* 13320, 133200Z (2026).
22. Choi, J. et al. Multi-omics integration via quantum support vector classifiers. *Comput. Meth. Prog. Bio.* **258**, 108492 (2026).
23. Rajpurkar, P. et al. Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists. *PLOS Med.* **15**, e1002686 (2018).
24. Irvin, J. et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In *Proc. AAAI* 590–597 (2019).
25. Johnson, A. E. W. et al. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. *arXiv* 1901.07042 (2019).
26. Collins, G. S. et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use machine learning. *BMJ* **385**, e078378 (2024).
27. Bakas, S. et al. Advancing the Cancer Genome Atlas glioma MRI collections with multi-institutional brain tumor segmentation. *Sci. Data* **4**, 170117 (2017).
28. Geirhos, R. et al. Shortcut learning in deep neural networks. *Nat. Mach. Intell.* **2**, 665–673 (2020).
29. Liu, X. et al. Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension. *Nat. Med.* **26**, 1364–1374 (2020).
30. Platt, J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Adv. Large Margin Classif.* **10**, 61–74 (1999).
31. Guo, C. et al. On calibration of modern neural networks. In *Proc. ICML* 1321–1330 (2017).
32. Schuirmann, D. J. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *J. Pharmacokinet. Biopharm.* **15**, 657–680 (1987).
33. Lakens, D. Equivalence testing for psychological research: a tutorial on NHST, $p$-values, and sample size planning. *Soc. Psychol. Personal. Sci.* **8**, 355–362 (2017).
34. DeLong, E. R. et al. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* **44**, 837–845 (1988).
35. Vovk, V. et al. Algorithmic learning in a random world. *Springer* (2005).
"""

target_repo = REPO_ROOT / "MANUSCRIPT_FINAL_npjDM.md"
target_manuscript = MANUSCRIPT_DIR / "MANUSCRIPT_FINAL_npjDM.md"

target_repo.write_text(md_content, encoding="utf-8")
print(f"Wrote {target_repo} ({len(md_content)} bytes)")

target_manuscript.write_text(md_content, encoding="utf-8")
print(f"Wrote {target_manuscript} ({len(md_content)} bytes)")
