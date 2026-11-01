# Supplementary Methods: Symmetric Evaluation Protocol, Mathematical Derivations, and Audit Ledger

**Manuscript Title:** Claimed quantum benefits in clinical machine learning do not survive symmetric evaluation  
**Target Journal:** *npj Digital Medicine*  
**Author:** Sivaguru R. M.  
**Affiliation:** Independent researcher  
**Correspondence:** sivagurumurugan1@gmail.com  

---

## Supplementary Note 1: The 17-Item Symmetric Evaluation Checklist

The checklist operationalizes the seven protocol rules (P1–P7) into concrete criteria for auditing clinical AI studies claiming architectural advantage:

```
Section A: Calibration and Probability Evaluation (Rule P1)
[ ] Item 1:  Are predicted probabilities calibrated independently per arm?
             Rule: Each evaluated model must be calibrated separately using a dedicated 
             validation/calibration partition prior to probability-based scoring.
[ ] Item 2:  Is calibration fitted strictly on validation/calibration data without test-set access?
             Rule: No test-set labels or features may be used during temperature scaling or Platt scaling.
[ ] Item 3:  Are calibration metrics (ECE, Brier score) reported alongside discrimination (AUROC)?
             Rule: Studies claiming clinical utility must evaluate both probability fidelity and ranking.

Section B: Optimization and Capacity Fairness (Rule P2)
[ ] Item 4:  Do classical and novel arms receive matched or equitable tuning budgets?
             Rule: The classical baseline must receive an equivalent number of hyperparameter trials 
             spanning standard model families.
[ ] Item 5:  Is hyperparameter selection performed via nested cross-validation or isolated validation splits?
             Rule: Test data must never influence architecture selection, learning rates, or early stopping.
[ ] Item 6:  Are classical baselines competitive (including strong nonlinear models when tasks contain nonlinear structure)?
             Rule: Novel models must be compared against modern competitive nonlinear baselines, not 
             deliberately crippled or untuned linear baselines alone.
[ ] Item 7:  Are parameter counts, circuit depths, and convergence criteria explicitly reported?
             Rule: Model capacity, training time, and optimizer termination conditions must be stated.

Section C: Partition Integrity and Leakage Prevention (Rule P3)
[ ] Item 8:  Are train, validation, and test splits strictly patient-disjoint?
             Rule: No patient records may cross partition boundaries.
[ ] Item 9:  Do repeated measurements (e.g., temporal EEG epochs, voice recordings, MRI slices) remain within their patient partition?
             Rule: Cross-validation must group by patient or subject identifier.
[ ] Item 10: Are all data-dependent transformations (PCA, scaling, imputations) fitted strictly inside training folds?
             Rule: Global dimensionality reduction on the full dataset prior to splitting is prohibited.

Section D: Statistical Inference and Error Quantification (Rule P4)
[ ] Item 11: Are superiority claims supported by paired statistical hypothesis tests or difference intervals?
             Rule: Point estimate comparisons without uncertainty intervals or hypothesis tests are invalid.
[ ] Item 12: Is repeated cross-validation analyzed using appropriate dependency corrections (e.g., Nadeau–Bengio)?
             Rule: Conventional t-tests that treat cross-validation folds as independent are prohibited.
[ ] Item 13: Are multiple comparisons adjusted across the full family of evaluated model contrasts?
             Rule: Holm-Bonferroni or Benjamini-Hochberg multiplicity corrections must be applied.
[ ] Item 14: Are equivalence claims supported by formal equivalence tests (TOST) rather than failure to reject the null?
             Rule: A non-significant p-value cannot be reported as proof of equivalence without a pre-specified margin.

Section E: Transparency and Provenance (Rules P5–P7)
[ ] Item 15: Are all evaluated architectures, seeds, and negative findings reported?
             Rule: Omission of poorly performing configurations or unconfirmed trials is prohibited.
[ ] Item 16: Are reported metrics tied directly to public, machine-readable binary prediction artifacts?
             Rule: Every decimal number in the text and tables must match an on-disk prediction file.
[ ] Item 17: Are claimed physical mechanisms (e.g., finite-shot execution, hardware speedup) verified at the source code level?
             Rule: Stated algorithmic or physical mechanisms must be verified against actual execution code.
```

---

## Supplementary Note 2: Multi-Class Chest Radiograph Triage Mapping & Bayes Optimality

### Diagnostic Category Formulation
The multi-class diagnostic target $y \in \{0, 1, 2, 3, 4, 5\}$ represents a clinical triage category derived deterministically from the predicted probabilities of eight radiological findings extracted by DenseNet-121 from frontal chest radiographs:
1. Pneumothorax ($p_1$)
2. Nodule / Lung Lesion ($p_2$)
3. Edema ($p_3$)
4. Cardiomegaly ($p_4$)
5. Pleural Effusion ($p_5$)
6. Pneumonia ($p_6$)
7. Consolidation ($p_7$)
8. Lung Opacity ($p_8$)

The category scores are computed according to an additive clinical policy:
$$\text{Score}(\text{Pneumothorax}) = 1.0 \times \mathbb{I}(p_1 \ge \tau_1)$$
$$\text{Score}(\text{Malignancy}) = 1.0 \times \mathbb{I}(p_2 \ge \tau_2)$$
$$\text{Score}(\text{Heart Failure}) = 0.7 \times \mathbb{I}(p_3 \ge \tau_3) + 0.5 \times \mathbb{I}(p_4 \ge \tau_4) + 0.2 \times \mathbb{I}(p_5 \ge \tau_5)$$
$$\text{Score}(\text{Pneumonia}) = 0.9 \times \mathbb{I}(p_6 \ge \tau_6) + 0.6 \times \mathbb{I}(p_7 \ge \tau_7) + 0.3 \times \mathbb{I}(p_8 \ge \tau_8)$$

If all category scores are zero, the study is classified as Normal ($y=5$). When positive scores exist, ties are resolved using the clinical acuity hierarchy:
$$\text{Pneumothorax} \succ \text{Malignancy} \succ \text{Heart Failure} \succ \text{Pneumonia} \succ \text{COPD} \succ \text{Normal}$$

### Structural Bayes Optimality
Because the decision rule is an argmax over an additive weighted sum of binary indicator features:
$$y = \arg\max_{c} \sum_{i=1}^8 w_{c, i} x_i$$
the theoretical Bayes-optimal decision boundary is linear in the underlying indicator space. Consequently, classical linear models (Product-of-Experts, Multinomial Logistic Regression) possess zero structural approximation bias on this task. Any model family that introduces high-order cross-terms (such as quantum circuits with entangling layers) increases representational capacity without reducing bias, leading to overfitting and an empirical generalization penalty.

### Crucial Distinction: Rule Emulation vs Clinical Outcome
Success on this benchmark measures how faithfully a fusion architecture recovers this additive clinical decision-support rule. It does **not** represent agreement with independently adjudicated expert consensus or prospective patient diagnostic benefit.

---

## Supplementary Note 3: Analytic Derivation of the Interaction-Order Spectrum (Analysis M1)

For an $n$-qubit variational quantum circuit with single-qubit angle encoding $\text{RY}(\pi x_i)$ on each qubit $i \in \{1, \dots, n\}$, the input state prior to parameter layers is:
$$\rho_0(x) = \bigotimes_{i=1}^n |\psi(x_i)\rangle\langle\psi(x_i)|, \quad |\psi(x_i)\rangle = \cos\left(\frac{\pi x_i}{2}\right)|0\rangle + \sin\left(\frac{\pi x_i}{2}\right)|1\rangle$$

In density matrix notation:
$$\rho_0(x) = \bigotimes_{i=1}^n \frac{1}{2} \left[ I + \cos(\pi x_i) Z + \sin(\pi x_i) X \right]$$

Let $U(\theta)$ represent the unitary transformation parameterized by the variational layers. The expectation value of measurement operator $Z_j$ is:
$$\langle Z_j \rangle(x) = \text{Tr}\left( U^\dagger(\theta) Z_j U(\theta) \rho_0(x) \right)$$

Expanding the Heisenberg-evolved observable in the $n$-qubit Pauli basis $P \in \{I, X, Y, Z\}^{\otimes n}$:
$$U^\dagger Z_j U = \sum_{P} c_P P, \quad c_P = \frac{1}{2^n} \text{Tr}\left( U^\dagger Z_j U P \right)$$

Because $\rho_0(x)$ is purely real and has zero expectation on Pauli-$Y$ ($\text{Tr}(Y \rho_0) = 0$), all terms containing $Y$ vanish:
$$\langle Z_j \rangle(x) = \sum_{P \in \{I, X, Z\}^{\otimes n}} c_P \prod_{i=1}^n f_{P_i}(x_i)$$
where:
$$f_I(x_i) = 1, \quad f_Z(x_i) = \cos(\pi x_i), \quad f_X(x_i) = \sin(\pi x_i)$$

The **interaction order** of Pauli string $P$ is the number of non-identity factors:
$$\text{Order}(P) = \sum_{i=1}^n \mathbb{I}(P_i \ne I)$$

- **Unentangled Product Circuits ($U = \bigotimes_i U_i$):** $U^\dagger Z_j U$ acts solely on qubit $j$. Hence, $c_P = 0$ for all $|P| \ge 2$. The circuit output is strictly additive over single-feature Fourier modes (Order $\le 1$).
- **Entangled Circuits (CNOT Ring Topology):** Entangling gates propagate information across qubits, generating non-zero coefficients for $|P| \ge 2$. Empirical measurement M1 confirms that $100.0\%$ of non-trivial coefficient energy and $99.99\%$ of logit variance are concentrated in interaction orders $\ge 2$. On additive targets, this represents unhelpful variance that degrades likelihood.

---

## Supplementary Note 4: Closed-Form Fidelity Kernel Reduction & Geometric Difference (Analysis M3)

### Closed-Form Product State Kernel
When no entangling gates are present, the quantum fidelity kernel reduces to the inner product between product states:
$$K_Q(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2 = \prod_{i=1}^n |\langle \psi(x_i) | \psi(x'_i) \rangle|^2$$
Using the trigonometric identity:
$$|\langle \psi(x_i) | \psi(x'_i) \rangle|^2 = \left[ \cos\left(\frac{\pi x_i}{2}\right)\cos\left(\frac{\pi x'_i}{2}\right) + \sin\left(\frac{\pi x_i}{2}\right)\sin\left(\frac{\pi x'_i}{2}\right) \right]^2 = \cos^2\left(\frac{\pi(x_i - x'_i)}{2}\right)$$
Therefore, the exact classical closed form is:
$$K_Q(x, x') = \prod_{i=1}^n \cos^2\left(\frac{\pi(x_i - x'_i)}{2}\right)$$
This analytical form matches statevector simulation to double-precision tolerance ($3.11 \times 10^{-15}$).

### Geometric Kernel Distance Formulation
Following Huang et al. (*Nat. Commun.* 12, 2631, 2021), the geometric difference between a classical kernel $K_1$ and a quantum kernel $K_2$ is defined as:
$$g(K_1 \parallel K_2) = \sqrt{ \left\| \sqrt{K_2} (K_1 + \lambda I)^{-1} \sqrt{K_2} \right\|_\infty }$$
where kernels are trace-normalized ($\tilde{K} = K \cdot N / \text{Tr}(K)$) and regularized with $\lambda = 10^{-3}$.

On a 2,000-patient sample from the development set:
- $\sqrt{N} = \sqrt{2000} \approx 44.72$
- $g(K_{\text{RBF}} \parallel K_Q) = 1.2712 \ll 44.72$

Under the theoretical bounds of Huang et al., when $g(K_1 \parallel K_2) \ll \sqrt{N}$, any target function learnable by $K_2$ with low sample complexity can also be learned by classical kernel $K_1$ with sample complexity bounded by $g^2 \cdot s_2 \approx 1.62 \cdot s_2$. This empirical calculation demonstrates that on this feature distribution, classical RBF kernels provide comparable sample complexity, precluding an inductive advantage.

---

## Supplementary Note 5: Pre-Registration & Git Commit Ledger

| Protocol & Addendum | Git Commit Hash | Date | Scope & Decisions Locked |
|---|:---:|:---:|---|
| **CheXpert External Transfer** | `fa8c2b6` | 2026-08-19 | Pre-registered CheXpert sampling, freeze/refit boundary, and label mapping. |
| **BraTS 2020 Official Re-embedding** | `21e9a4f` | 2026-09-21 | Pre-registered re-embedding of 369 official subjects and 111-subject held-out split. |
| **MIMIC Revision Analyses** | `d4052fb` | 2026-10-02 | Pre-registered TOST margin ($\pm 0.010$ nats) and patient-level bootstrap. |
| **CheXpert-Internal Protocol** | `3a54488` | 2026-10-02 | Pre-registered out-of-sample evidence E1 and E2 splits. |
| **Confirmatory v6 Design** | `61e777b` | 2026-10-03 | Pre-registered confirmatory evaluation on 56,534 untouched CheXpert patients. |
| **Field Audit Re-implementations** | `c72945e` | 2026-10-03 | Pre-registered protocol for re-testing published literature claims. |
| **Confirmatory Execution & Mechanism** | Working Tree | 2026-10-11 | Executed confirmatory scoring and mechanistic analyses M1–M3. |

---

## Supplementary Note 6: Systematic Review Search Strategy & PRISMA Master Ledger

### PubMed Search String (Executed October 2026)
```
("quantum machine learning"[Title/Abstract] OR "quantum neural network"[Title/Abstract] OR 
"variational quantum"[Title/Abstract] OR "quantum kernel"[Title/Abstract] OR 
"quantum support vector"[Title/Abstract]) AND 
("health"[Title/Abstract] OR "medical"[Title/Abstract] OR "clinical"[Title/Abstract] OR 
"disease"[Title/Abstract] OR "patient"[Title/Abstract] OR "diagnosis"[Title/Abstract])
```

### Screening Stages & Record-Level Audit Trail
1. **Initial Search & Retrieval (`all_pmids.txt`, `pubmed_records.json`):** 231 records identified via PubMed search on 21 September 2026.
2. **Stage S1 Medical & Empirical Filtering (`screen.py`, `stage1_medical.json`):** Deliberately broad medical term regular expression matching (`medic|clinic|patient|disease|cancer|...`) excluding review/perspective articles. **129 records** passed S1 into `stage1_medical.json` (102 records excluded as physics simulations, non-human health, molecular docking, or review articles).
3. **Stage S2 Abstract Clinical Claim Assessment (`s2_decisions.csv`):** 129 records independently screened for empirical claims of quantum advantage on patient-level clinical prediction or diagnostic outcomes. **41 records** passed S2 (`s2 == 'include'`), and **88 records** were excluded (`s2 == 'exclude'`).
4. **Stage S3 Full-Text Eligibility & Reconstruction Assessment (`s3_decisions.csv`):** 41 records assessed at full text for open/reconstructible data, detailed circuit architecture, and comparative evaluation against classical baselines:
   - **8 Eligible Included Publications:** Formally met eligibility criteria (clinical prediction on open or reconstructible data).
   - **29 Excluded Publications:** S3(a) private hospital cohorts without data release (19 records); S3(b) quantum component underspecified or embedded in deep classical network without circuit specification (7 records); S3(c) classical models omitted or not empirical (3 records).
   - **4 Not Assessable Publications:** Full text behind publisher paywall where no open preprint was retrievable (PMIDs: 41517992, 41728802, 41523782, 41850494).

### Synthesis and Re-Evaluation Breakdown
From the 8 eligible published studies:
- **3 Non-reconstructible for computational execution:**
  1. *Jahin et al.* (Comput Biol Med 2025; PMID 40865268): 3M-parameter hybrid KACQ-DCNN; no source code or tensor dimensions released.
  2. *Chen et al.* (Proc SPIE 2026; PMID 42046760): PROSTATEx radiomics; proprietary feature extraction pipeline and weights unreleased.
  3. *Choi et al.* (Comput Methods Programs Biomed 2026; PMID 41966795): COVID-19 omics; controlled-access data requiring unreleased credentialing.
- **6 Checklist-assessable from published text:** *Parthasarathi et al.*, *Akpinar & Oduncuoglu*, *Daga et al.*, *Ghisoni et al.*, *Aksoy & Cattan*, and *Jahin et al.* (Jahin et al. published complete methodological descriptions allowing 17-item checklist scoring, but did not release source code). Five of six met criteria for asymmetric evaluation risk; all six lacked paired statistical hypothesis testing.
- **5 Reimplemented publications:** *Parthasarathi et al.*, *Akpinar & Oduncuoglu*, *Daga et al.*, *Ghisoni et al.*, *Aksoy & Cattan*.
- **6 Benchmark clinical prediction tasks:**
  1. Chronic Kidney Disease (Parthasarathi et al., VQC vs RF/RBF)
  2. Parkinson's Disease voice phonation (Daga et al., QKL stacking vs SVM)
  3. Pima Diabetes mellitus (Daga et al., QKL stacking vs SVM)
  4. Schizophrenia EEG classification (Aksoy & Cattan, QSVM)
  5. Glioma tumor grading (Akpinar & Oduncuoglu, VQC & BraTS 2020 QKL)
  6. Heart Disease low-data risk prediction (Ghisoni et al., 1-layer QNN vs classical NN)
- **Survival Decision Criterion:** A published quantum advantage claim is declared to **survive** if and only if the quantum model achieves statistically significant superiority ($\Delta > 0, p < 0.05$ after multiple testing correction) over tuned classical baselines under leak-free nested cross-validation. Under this prespecified rule, **zero of the six benchmark claims survived** (Task 1 reversed with classical significantly superior, $p = 0.0009$; Tasks 2–6 showed no statistically detectable difference, $p > 0.05$).

---

## Supplementary Note 7: Hyperparameter Grids and Optimization Details

### Confirmatory Arm C1 (Matched Optimization Protocol)
- Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.0$).
- Batch Size: 50.
- Maximum Epochs: 150.
- Early Stopping: Patience 20 epochs monitored on calibration cross-entropy loss.
- Learning Rate Grid: $\text{lr} \in \{0.01, 0.05, 0.10\}$ selected on calibration split.
- Temperature Scaling: Single parameter $T > 0$ fitted via NLL minimization on calibration split.

### Confirmatory Arm C2 (36-Configuration Symmetric Grid)
- **Quantum Search Space:**
  - Layers $L \in \{1, 2, 3, 4, 6\}$
  - Angle Scale $s \in \{0.5, 1.0, 2.0\}$
  - Initialization Standard Deviation $\text{sd} \in \{0.01, 0.1, 1.0\}$
  - Learning Rate $\text{lr} \in \{0.01, 0.05, 0.10\}$
  - Random draw of 36 configurations (seed 7).
- **Classical Linear (PoE) & Gated Search Spaces:**
  - Learning Rate $\text{lr} \in \{0.003, 0.01, 0.05, 0.1, 0.3\}$
  - Weight Decay $\text{wd} \in \{0.0, 10^{-5}, 10^{-4}, 10^{-3}\}$
  - Initialization Standard Deviation $\text{sd} \in \{0.01, 0.1, 1.0\}$
  - Random draw of 36 configurations (seed 7).

---

## Supplementary Note 8: Statistical Formulations

### Inconclusive Comparisons vs. Formal Equivalence Testing: Biostatistical Distinction
A fundamental principle of clinical biostatistics is that "absence of evidence is not evidence of absence" (Altman & Bland, 1995). A common misinterpretation in machine learning benchmarks is to treat a non-significant superiority test ($p > 0.05$) as proof that two methods are equivalent. This assertion is mathematically invalid unless evaluated using formal equivalence testing (such as Two One-Sided Tests) backed by adequate statistical power to rule out practically meaningful differences.

This distinction is prominently illustrated in our literature benchmark evaluations. For instance, in the low-data heart disease benchmark (Ghisoni et al., 2026; PMID 41513819), the original study evaluated a 1-layer quantum neural network against classical baselines on a 10% training fraction ($n_{\text{train}} = 74, n_{\text{test}} = 184$ per outer fold across 25 outer folds of $5 \times 5$ repeated CV). Under validation-based early stopping, the observed point difference was $\Delta = -1.10\%$. When analyzed using a conventional paired Student's $t$-test (which erroneously treats CV folds as independent), the 95% CI appears deceptively narrow: $[-3.20\%, +1.00\%]$ ($p = 0.310$). However, when evaluated under the rigorous Nadeau–Bengio correction accounting for outer-fold data overlap with the actual low-data ratio $n_{\text{test}} / n_{\text{train}} = 184 / 74 = 2.486$, the corrected standard error widens significantly:
$$\text{SE}_{\text{NB}} = \sqrt{\left(\frac{1}{25} + 2.486\right) S^2} = \sqrt{2.526} \cdot S \approx 1.589 \cdot S$$
yielding a corrected 95% confidence interval of **$-18.45$ to $+16.25$ percentage points** ($t = -0.13, df = 24, p = 0.897$).

An uncertainty interval of nearly 35 percentage points spans substantial clinical superiority in either direction. Consequently, this outcome supports an **inconclusive result**, not proof that the methods are equivalent. The benchmark lacked the sample size to detect whether the models differ by 5, 10, or 15 percentage points. Similar wide uncertainty intervals characterize other external benchmarks evaluated under dependency correction (e.g., Parkinson's Voice 95% CI: $[-24.97\%, +25.64\%]$).

In stark contrast, our confirmatory evaluation on CheXpert was powered on **56,534 independent patients**. The resulting 95% confidence interval was extremely narrow ($[-0.00359, -0.00160]$ nats), lying entirely inside our pre-specified operational equivalence band ($\pm 0.010$ nats) and rejecting non-equivalence via Two One-Sided Tests ($p_{\text{TOST}} < 10^{-6}$). The paper carefully distinguishes between these two regimes: external tasks were statistically inconclusive, whereas the confirmatory trial formally confirmed practical equivalence.

### Scope and Generalizability Boundaries
The empirical evidence provided by our five reimplemented publications and six clinical tasks supports a carefully bounded conclusion: published claims of quantum advantage in these specific digital health settings collapse when subjected to symmetric optimization, independent calibration, and leak-free validation. However, this finding **cannot be generalized to a universal claim** that quantum computing can never offer an advantage in machine learning or clinical healthcare. 

The empirical boundary of our work is explicitly defined by:
1. **Model Families:** Shallow hardware-efficient variational circuits (VQC, QNN, 4–12 qubits) and fidelity quantum kernels.
2. **Data Encodings:** Single-qubit Pauli-$Y$ angle encodings $R_Y(\pi x_i)$.
3. **Data Representations:** Tabular clinical features and classical deep CNN feature embeddings.
4. **Computational Paradigm:** Exact classical statevector simulation.

Alternative quantum paradigms---including fault-tolerant algorithms with asymptotic polynomial or exponential speedups (e.g., HHL algorithm, quantum phase estimation), native quantum states from quantum sensors, multi-qubit interactive Hamiltonian feature maps that cannot be classically simulated, and biological domains where physical classical simulation is intractable (e.g., molecular dynamics, drug-target binding, and quantum chemistry)---fall entirely outside our experimental boundary and may hold genuine quantum potential.

### Paired Negative Log-Likelihood & Equivalence Testing (TOST)
Let $p_{i, y_i}^{(A)}$ and $p_{i, y_i}^{(B)}$ denote predicted probabilities for true label $y_i$ of patient $i \in \{1, \dots, n\}$.
$$d_i = -\log p_{i, y_i}^{(A)} - \left( -\log p_{i, y_i}^{(B)} \right)$$
Mean difference $\bar{d} = \frac{1}{n} \sum_{i=1}^n d_i$, Standard error $\text{SE}(\bar{d}) = \frac{s_d}{\sqrt{n}}$.

Two One-Sided Tests (TOST) evaluate the null hypothesis of non-equivalence against margin $\pm \Delta_{\text{equiv}}$ ($\Delta_{\text{equiv}} = 0.010$ nats):
$$H_{0, 1}: \bar{d} \le -\Delta_{\text{equiv}}, \quad t_1 = \frac{\bar{d} - (-\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}$$
$$H_{0, 2}: \bar{d} \ge +\Delta_{\text{equiv}}, \quad t_2 = \frac{+\Delta_{\text{equiv}} - \bar{d}}{\text{SE}(\bar{d})}$$
$$p_{\text{TOST}} = \max\left( \text{Pr}(T \le t_1), \text{Pr}(T \le t_2) \right)$$
Equivalence is declared at level $\alpha = 0.05$ if $p_{\text{TOST}} < 0.05$, which occurs if and only if the $90\%$ confidence interval $[\bar{d} - t_{0.95} \text{SE}, \bar{d} + t_{0.95} \text{SE}]$ falls entirely within $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$.

*Operational Margin Formulation & Git Provenance:* The margin $\Delta_{\text{equiv}} = \pm 0.010$ nats was pre-specified in `PREREGISTRATION_ADDENDUM_2026-10.md` (Oct 2, 2026) and locked in Git commit `61e777b` on Sat Oct 3, 2026, eight days prior to the extraction and evaluation of the confirmatory test cohort on Oct 11, 2026. As documented in the preregistration, this margin represents an operational statistical benchmark on expected negative log-likelihood differences ($\mathbb{E}[\Delta\ell(x, y)] \le 0.010$ nats per observation across the population). Importantly, an average negative-log-likelihood difference across a cohort does NOT, by itself, mathematically bound individual-patient risk ratios or worst-case clinical risk for specific patients. Furthermore, this margin has NOT been clinically validated against prospective clinical outcomes (such as patient mortality, readmission, or diagnostic error thresholds). Unless supported by empirical clinical validation studies linking margins to acceptable patient risk, it must remain explicitly interpreted as an operational statistical threshold.

### Sensitivity Analysis Across Alternative Equivalence Margins
To evaluate how conclusions depend on the choice of operational boundary, we evaluated the C2 confirmatory contrasts across a range of tighter and wider margins:

| Equivalence Margin ($\Delta_{\text{equiv}}$) | Nominal Likelihood Ratio ($\exp(\Delta)$)* | Quantum vs Linear PoE ($95\%$ CI: $[-0.00359, -0.00160]$) | Quantum vs Nonlinear Gated ($95\%$ CI: $[-0.00192, -0.00041]$) | Reading / Conclusion |
|:---:|:---:|:---:|:---:|---|
| **$\pm 0.020$ nats** | $\approx 1.020$ | $p_{\text{TOST}} = 2.4 \times 10^{-146}$ (Equivalent) | $p_{\text{TOST}} = 5.8 \times 10^{-198}$ (Equivalent) | Fully equivalent under wide margin |
| **$\pm 0.010$ nats** (Preregistered) | $\approx 1.010$ | $p_{\text{TOST}} = 8.5 \times 10^{-49}$ (Equivalent) | $p_{\text{TOST}} = 3.6 \times 10^{-78}$ (Equivalent) | **Preregistered baseline: Practical equivalence** |
| **$\pm 0.005$ nats** | $\approx 1.005$ | $p_{\text{TOST}} = 1.3 \times 10^{-6}$ (Equivalent) | $p_{\text{TOST}} = 3.8 \times 10^{-24}$ (Equivalent) | Robust equivalence under tight margin |
| **$\pm 0.002$ nats** | $\approx 1.002$ | $p_{\text{TOST}} = 0.882$ (**Not Equivalent**) | $p_{\text{TOST}} = 0.013$ (**Equivalent**) | Advantage over linear PoE exceeds 0.002 nats; matches nonlinear Gated Head |

*\*Note: Geometric-mean likelihood ratio across the population; an average NLL difference does not bound individual-patient risk ratios.*

### Architecture Parameter Capacities and Channel Encoding
- **8th Reference Channel:** The input vector $x \in [0, 1]^8$ consists of seven normalized DenseNet finding probabilities ($x_1, \dots, x_7$) and a constant baseline reference channel $x_8 \equiv 0.0$. Because $x_8 = 0.0$, the eighth qubit receives rotation $R_Y(0)|0\rangle = |0\rangle$, remaining unrotated in the ground state. It is coupled to qubits 7 and 1 through the closed CNOT ring, and its expectation $\langle Z_8 \rangle \in [-1, 1]$ is measured into the linear readout.
- **Model Parameter Counts:**
  - C1 VQC: 8 qubits, 3 layers ($8 \times 2 \times 3 = 48$ angles) + linear readout ($6 \times 8 + 6 = 54$) = **102 parameters**.
  - C2 VQC: 8 qubits, 4 layers ($8 \times 2 \times 4 = 64$ angles) + linear readout ($54$) = **118 parameters**.
  - Linear Product-of-Experts: $6 \times 8 + 6 = \mathbf{54\text{ parameters}}$.
  - Logistic Regression: $6 \times 8 + 6 = \mathbf{54\text{ parameters}}$.
  - Nonlinear Gated Head: $8 \times 8 + 8 = 72$ gating + $54$ readout = **126 parameters**.
- **C1 vs C2 PoE Selection Variance:** In C1, default PoE achieved test NLL 1.1157 ($\eta=0.01, \lambda=0, \text{sd}=0.1$). In C2, PoE was tuned over 36 configurations; configuration 17 ($\eta=0.05, \lambda=10^{-4}, \text{sd}=0.01$) had slightly better calibration NLL on $n=1,600$ ($1.1369$ vs $1.1371$), but test NLL 1.1170 on the 56,534 held-out patients (+0.0013 validation selection variance). The primary C2 contrast is the matched 36-trial budget comparison ($\Delta\text{NLL} = -0.00260$, $p = 3.07 \times 10^{-7}$, $p_{\text{TOST}} < 10^{-6}$).
- **Development Calibration Split Dual Use:** The 1,600 development calibration cases were used both for early-stopping checkpoint selection and for fitting the single scalar Platt temperature $T^*$ via L-BFGS. No model was refit on the pooled development set. While this dual use introduces a minor optimistic bias on development validation loss, the confirmatory evaluation on 56,534 held-out patients remained strictly leak-free and untouched.

### Sampling Unit & Dependence Audit
- **Unit of Observation:** The 56,534 test examples represent **56,534 distinct human patients**. Exactly one frontal chest radiograph study is selected per patient (the highest-numbered frontal study on record in CheXpert-v1.0-small, selected via `fresh_rows()`).
- **Patient-Level Independence by Construction:** Automated unit tests in `aura/tests/test_one_study_per_patient.py` verify that repeated studies from the same subject are skipped before dataset emission. Selecting exactly one study per patient establishes patient-level independence by construction, strictly eliminating intra-patient repeated-measures clustering.
- **Statistical Significance vs Clinical Utility:** With $N = 56,534$ paired patient observations (patient-level independence by construction), the standard error of the mean difference is extremely small ($\text{SE} \approx 0.0005$ nats). Consequently, minute shifts ($\Delta = 0.0026$ nats) achieve overwhelming statistical significance ($p = 3.07 \times 10^{-7}$). We caution readers and reviewers that statistical detectability is an artifact of massive sample size and must not be conflated with transformative clinical efficacy.
- **Institutional Single-Center Dependencies:** Although patient-level selection guarantees zero repeated-measures clustering (no subject appears more than once across development or test sets), CheXpert is a single-center cohort collected from Stanford Hospital. As such, observations share hospital-wide imaging protocols, scanner calibration, radiology report formatting, and regional demographic characteristics. Complete statistical independence across healthcare environments cannot be assumed from single-center cohorts alone. This motivated our multi-center domain transfer experiment (training on MIMIC-CXR from Beth Israel Deaconess Medical Center, Boston, and testing frozen on CheXpert, Stanford across $n=4,800$ studies), where the quantum model finished last among all architectures ($\Delta\text{NLL} = +0.1589$ against Product-of-Experts, $+0.1967$ against Logistic Regression; 95% CIs exclude zero).
- **Descriptive Split Distributions vs Inferential Hypothesis Testing (EEG Benchmark):** For the adolescent schizophrenia EEG benchmark (Aksoy & Cattan: $N = 84$ unique adolescent human participants: 45 schizophrenia cases, 39 healthy controls; majority class prevalence $45/84 = 53.57\%$), the 200 repeated `GroupShuffleSplit` subject-level partitions are reported descriptively to demonstrate the contrast between random epoch splitting (mean 98.6%, 34.5% hitting 100%) and subject-grouped splitting (mean 52.1%, SD 7.0%, empirical 2.5th–97.5th percentile range across splits: 43.2%–61.0%, with 0% of splits reaching >75%, matching the 53.57% majority class prevalence). Because repeated random splits resample the same 84 human participants across training and test subsets, treating the 200 splits as independent observations to compute naive standard errors or one-sample $t$-tests against chance would induce pseudo-replication. Therefore, all formal inferential hypothesis testing against the classical baseline relies strictly on the pre-registered 25-fold nested cross-validation with Nadeau–Bengio dependency correction ($n_{\text{train}}=68, n_{\text{test}}=16, J=25, df=24$), yielding $\Delta = -7.60\%$ in the classical model's favor ($t = -1.018, p = 0.3187, 95\%$ CI $[-23.01\%, +7.81\%]$).

### Nadeau–Bengio Corrected Resampled $t$-Test
For repeated cross-validation with $R$ repeats and $K$ folds ($J = R \times K$ runs) with training set size $n_{\text{train}}$ and test set size $n_{\text{test}}$:
$$\text{SE}_{\text{corrected}} = \sqrt{\left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s^2}$$
$$t_{\text{corrected}} = \frac{\bar{d}}{\text{SE}_{\text{corrected}}}$$
with degrees of freedom $df = J - 1$. The test operates on paired fold-level performance differences $d_j = \text{Metric}_{\text{quantum}, j} - \text{Metric}_{\text{classical}, j}$, accounting for the correlation induced by training-set overlap.

The parameters and exact inferential statistics for the five computationally reimplemented literature studies across 25 outer folds ($5 \times 5$ repeated CV, $df = 24$) are summarized below:

| Study & Clinical Task | Target Metric | Folds ($J$) | $n_{\text{train}}$ | $n_{\text{test}}$ | Quantum Mean (SD) | Classical Mean (SD) | Paired $\Delta$ (95% CI) | Corrected $t$-stat ($df=24$) | Two-Sided $p$-value | Bonferroni Survival ($\alpha = 0.05/6$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chronic Kidney Disease** (Parthasarathi et al.) | Accuracy | 25 | 320 | 80 | 87.70% (4.10%) | 97.35% (1.95%) | **-9.65%** [$-14.92\%$, $-4.38\%$] | $t = -3.7808$ | $p = 0.000915$ | **Survives (Reversed: Classical Superior)** |
| **Parkinson's Phonation** (Daga et al.) | Specificity | 25 | 159 | 36 | 43.33% (34.16%) | 43.00% (39.70%) | **+0.33%** [$-24.97\%$, $+25.64\%$] | $t = +0.0272$ | $p = 0.978533$ | Not Surviving ($p > 0.05$) |
| **Pima Diabetes** (Daga et al.) | Sensitivity | 25 | 615 | 153 | 69.34% (6.86%) | 74.64% (5.05%) | **-5.30%** [$-11.00\%$, $+0.39\%$] | $t = -1.9216$ | $p = 0.066610$ | Not Surviving ($p > 0.05$) |
| **Schizophrenia EEG** (Aksoy & Cattan) | Accuracy | 25 | 68 | 16 | 59.10% (11.06%) | 66.71% (7.83%) | **-7.60%** [$-23.01\%$, $+7.81\%$] | $t = -1.0184$ | $p = 0.318652$ | Not Surviving ($p > 0.05$) |
| **Glioma Tumor Grading** (Akpinar & Oduncuoglu) | Accuracy | 25 | 672 | 167 | 85.60% (2.44%) | 85.41% (2.88%) | **+0.19%** [$-1.22\%$, $+1.60\%$] | $t = +0.2793$ | $p = 0.782425$ | Not Surviving ($p > 0.05$) |
| **Heart Disease Low-Data** (Ghisoni et al.) | Accuracy | 25 | 74 | 184 | 81.50% (3.80%) | 82.60% (3.20%) | **-1.10%** [$-3.20\%$, $+1.00\%$] (paired $t$)<br>[$-18.45\%$, $+16.25\%$] (NB) | Paired $t = -1.0400$;<br>Nadeau–Bengio $t = -0.1309$ | $p = 0.3100$ (paired $t$);<br>$p = 0.8970$ (NB) | Not Surviving ($p > 0.05$) |


---

## Supplementary Note 9: Confirmatory Master Results Ledger (Analyses C1, C1b, C2, and M2)

All analyses scored on the 56,534 held-out CheXpert patients (`C1_y_u0_probs.npz`, `v6_confirmatory.json`, `v6_mechanism.json`):

| Analysis Tag | Model Architecture | Train / Cal Partitions | Test Sample Size ($n$) | Metric Evaluated | Point Estimate | Uncertainty Interval | Statistical Significance | Confirmatory Pre-registered Reading |
|---|---|---|:---:|---|:---:|:---:|:---:|---|
| **C1.classical** | Classical Product-of-Experts | 6,400 / 1,600 | 56,534 | Test NLL | 1.1157 | — | — | Reference Baseline |
| **C1.logistic** | Multinomial Logistic Regression | 6,400 / 1,600 | 56,534 | Test NLL | 1.1152 | 95% CI: $[1.109, 1.121]$ | $p = 0.2767$ vs PoE | Practically Equivalent |
| **C1.learnable**| Attention-Gated Nonlinear Head | 6,400 / 1,600 | 56,534 | Test NLL | 1.1165 | 95% CI: $[1.110, 1.123]$ | $p = 4.94 \times 10^{-7}$ vs PoE | Practically Equivalent |
| **C1.quantum** | Variational Quantum Circuit | 6,400 / 1,600 | 56,534 | Test NLL | 1.1362 | 95% CI: $[1.130, 1.142]$ | $p = 7.65 \times 10^{-98}$ vs PoE | Quantum Materially Worse |
| **C1.primary** | VQC vs PoE Paired Difference | 6,400 / 1,600 | 56,534 | Paired $\Delta\text{NLL}$ | **+0.02049** | 90% CI: $[0.01889, 0.02210]$<br>95% CI: $[0.01858, 0.02240]$ | $t = 21.034, p = 7.65 \times 10^{-98}$<br>$p_{\text{TOST}} = 1.000$ | Quantum Materially Worse |
| **C1b.seed** | Entangled vs Product Circuit | 6,400 / 1,600 | 56,534 | $\Delta\text{NLL}$ (10 seeds) | **+0.01037** | 95% boot CI: $[0.00934, 0.01138]$ | $t = 4.88, p = 0.00127$ (seeds)<br>9/10 seeds favor unentangled | Entanglement Penalty Replicates |
| **C2.quantum** | Best Quantum (4L, $s=0.5$) | 6,400 / 1,600 | 56,534 | Test NLL | 1.11438 | — | — | Selected on Dev Cal NLL |
| **C2.classical**| Best Classical PoE | 6,400 / 1,600 | 56,534 | Test NLL | 1.11698 | — | — | Selected on Dev Cal NLL |
| **C2.primary** | Best Quantum vs Best PoE | 6,400 / 1,600 | 56,534 | Paired $\Delta\text{NLL}$ | **-0.00260** | 90% CI: $[-0.00343, -0.00176]$<br>95% CI: $[-0.00359, -0.00160]$ | $t = -5.10, p = 3.07 \times 10^{-7}$<br>$p_{\text{TOST}} < 10^{-6}$ | Practically Equivalent ($\Delta < 0.010$) |
| **C2.learnable**| Best Quantum vs Best Gated | 6,400 / 1,600 | 56,534 | Paired $\Delta\text{NLL}$ | **-0.00116** | 90% CI: $[-0.00180, -0.00053]$<br>95% CI: $[-0.00192, -0.00041]$ | $t = -3.05, p = 0.00247$<br>$p_{\text{TOST}} < 10^{-6}$ | Practically Equivalent ($\Delta < 0.010$) |
| **M2.order1** | Classical Fourier Surrogate (K=1)| 6,400 / 1,600 | 56,534 | Test NLL (14 feat) | 1.11233 | 95% CI: $[1.106, 1.118]$ | Beats Quantum by $+0.0238$ | Surrogate Outperforms Quantum |
| **M2.order2** | Classical Fourier Surrogate (K=2)| 6,400 / 1,600 | 56,534 | Test NLL (112 feat)| 1.10950 | 95% CI: $[1.103, 1.116]$ | Beats Quantum by $+0.0267$ | Surrogate Outperforms Quantum |

---

## Supplementary Note 10: TRIPOD+AI Reporting Alignment Checklist

| Item | TRIPOD+AI Item Description | Section in Manuscript / Supplement | Compliance Evidence |
|:---:|---|---|---|
| **1** | Title identifying artificial intelligence and study design | Title, Page 1 | Title specifies symmetric evaluation and model class. |
| **2** | Abstract structured summary | Abstract, Page 1 | Reports objectives, data source, sample size ($n=56,534$), primary contrast, and findings. |
| **3** | Background and clinical rationale | Introduction, Pages 1–2 | Delineates asymmetric evaluation risks and digital health QML context. |
| **4** | Study objectives and pre-registration | Introduction, Page 2 | States protocol rules and pre-registration status (`61e777b`). |
| **5** | Source of data and cohort assembly | Methods, Page 6 | MIMIC-CXR and CheXpert-v1.0-Small selection rules and licenses detailed. |
| **6** | Eligibility and exclusion criteria | Methods, Page 6; Supp Note 2 | Highest-numbered frontal view per patient; single image per patient. |
| **7** | Target outcome definitions | Methods, Page 6; Supp Note 2 | Explicit weighted acuity mapping across 8 DenseNet findings. |
| **8** | Predictors and feature representations | Methods, Page 6 | DenseNet-121 finding probabilities with per-finding Platt calibration. |
| **9** | Sample size determination | Results, Page 4 | 56,534 held-out patients providing statistical power to resolve $\pm 0.010$ nats. |
| **10** | Missing data handling | Methods, Page 6 | Zero-imputation / absence encoding for unobserved findings. |
| **11** | Model development and tuning | Methods, Page 6; Supp Note 7 | 36-configuration search space, early stopping patience 20, learning-rate grid. |
| **12** | Calibration procedures | Methods, Page 6; Supp Note 7 | Independent per-arm temperature scaling on dedicated calibration split. |
| **13** | Model validation and test isolation | Methods, Page 6; Supp Note 5 | Prospective evaluation on 56,534 patient-disjoint held-out patients. |
| **14** | Performance measures and discrimination | Results, Table 3 | Reports multi-class NLL, accuracy, ECE, and macro-AUROC. |
| **15** | Statistical uncertainty and comparisons | Methods, Page 6; Supp Note 8 | Paired $t$-tests, Wilcoxon signed-rank, DeLong, McNemar, TOST, and bootstrap CIs. |
| **16** | Code and model availability | Code Availability, Page 7 | Complete reproduction command and open GitHub repository. |
