# CLAIM EVIDENCE MATRIX: ADVERSARIAL RESEARCH AUDIT

**Target Manuscript:** *"Auditing Quantum Machine Learning in Digital Health: One Deployed System and Five Published Claims Under Symmetric Evaluation"* (Manuscript v5.1 / v6 npjDM)  
**Audit Team:** Interdisciplinary Research Review Board (QML Theorist, ML Benchmarking Specialist, Medical AI Clinician-Scientist, Computational Statistician, Adversarial Reviewer, Research Software Engineer)  
**Audit Date:** October 11, 2026  
**Repository State:** `E:/AURA/aura-main` (Commit `61e777b` + working tree)

---

## 1. Classification Taxonomy

Every scientific claim reported in the manuscript, its supplementary material, or its pre-registrations is assigned one of four evidentiary statuses:

- **VERIFIED:** The result has been independently regenerated or verified against cryptographic byte-digests and complete data-to-prediction pipelines on disk.
- **PARTIALLY VERIFIED:** Supporting artifacts exist on disk and intermediate code runs, but the full pipeline has not been executed end-to-end in this session or relies on unverified upstream inputs.
- **UNVERIFIED:** Insufficient reproducible artifacts or inaccessible raw data prevent independent confirmation.
- **CONTRADICTED:** Available empirical evidence, mathematical proof, or code inspection directly refutes or invalidates the reported claim.

---

## 2. Master Claim-Evidence Matrix

| Claim ID | Manuscript Location | Reported Claim | Evidentiary Status | Supporting Artifacts & Code | Audit Finding & Adversarial Critique |
|---|---|---|:---:|---|---|
| **C-01** | §Abstract, §2.2 | *Published calibration advantage of the quantum fusion circuit (ECE 0.020 vs 0.276) was an artifact of asymmetric temperature scaling.* | **VERIFIED** | `fusion_baselines_n447.json`, `aura.ml.evaluation.fusion_baselines` | **Confirmed.** Inspection of `vision_calibration.py` and historical commits confirms only the quantum head was temperature-scaled ($T=0.906$). When all arms are uncalibrated or symmetrically calibrated, the apparent ECE advantage vanishes (raw ECE gap shrinks from 0.139 to 0.037). |
| **C-02** | §Abstract, §2.3 | *Under a matched optimizer protocol on MIMIC-CXR (n=447), no significant likelihood difference exists between Quantum VQC and Product-of-Experts ($\Delta\text{NLL} = 0.0033$, Holm $p = 1.0$).* | **VERIFIED** | `fusion_matched_optimiser.json`, `fusion_matched_optimiser_predictions.npz`, `aura.ml.evaluation.matched_optimiser` | **Verified.** Exact numerical match across all 447 test cases. Under matched Adam (LR grid {0.01, 0.05, 0.1}, patience 20, seed 7), PoE achieves NLL 1.2362 vs Quantum 1.2395. However, early claims of a "reversal" were statistically invalid; the difference is within noise. |
| **C-03** | §2.3 | *Equivalence between Quantum VQC and Product-of-Experts on MIMIC-CXR test set under margin $\pm 0.010$ NLL.* | **CONTRADICTED** (for n=447) / **PENDING** (for n=56k) | `revision_v51.json` (Analysis A), `PREREGISTRATION_ADDENDUM_2026-10.md` | **Failed Equivalence on n=447.** TOST 90% CI on $n=447$ is $[-0.02704, +0.03373]$, which substantially crosses the $\pm 0.010$ boundary ($p_{\text{TOST}} = 0.359$). Claiming equivalence on $n=447$ is false. The test only established *absence of detectable difference*. Equivalence testing requires $n \approx 56,534$ CheXpert test data to narrow the CI. |
| **C-04** | §2.4, §2.5 | *At matched capacity on 7 informative channels across 10 random seeds, removing entanglement improves NLL in 10 of 10 seeds (Holm $p = 0.006$).* | **VERIFIED** | `revision_v51.json` (Analysis B), `C1b_seed*.npz`, `aura.ml.evaluation.revision_v51` | **Empirically Verified, Mechanistically Qualified.** Across seeds {7, 11, 13, 17, 19, 23, 29, 31, 37, 41}, mean $\Delta\text{NLL} = 0.0250$ (paired $t$-test $p=0.0013$, Wilcoxon $p=0.0020$). Clustered bootstrap over 447 patients yields 95% CI $[0.0016, 0.0483]$. **Crucial Caveat:** This does not prove entanglement is universally harmful; it proves entanglement introduces an unfavorable inductive bias for an additive/linear clinical target. |
| **C-05** | §2.5 | *Between-split variance across $K=5$ patient-disjoint re-splits confirms Product-of-Experts consistently matches or beats the entangled circuit.* | **VERIFIED** | `revision_v51.json` (Analysis C) | **Verified.** On 5 stratified 60/20/20 re-splits (seeds 101–105), entangled NLL is consistently higher (worse) than unentangled across all 5 splits ($\Delta\text{NLL} \in [0.0216, 0.0461]$). |
| **C-06** | §2.5 | *Matched-parameter data re-uploading ($L=3$, 72 params) does not rescue the entangled circuit; entanglement penalty persists.* | **VERIFIED** | `revision_v51.json` (Analysis D) | **Verified.** Matched 72-parameter re-uploading circuits across 10 seeds show unentangled arm achieving lower NLL in 10/10 seeds (mean $\Delta\text{NLL} = 0.0229$, 95% CI $[0.0094, 0.0365]$). |
| **C-07** | §2.4 | *Positive control T1, T2a, T2b establishes apparatus sensitivity: detects nonlinear advantage when present, including for quantum circuit on T2b.* | **PARTIALLY VERIFIED & MISINTERPRETED** | `nonlinear_control.json`, `aura.ml.evaluation.nonlinear_control` | **Methodological Threat:** On T2b (pairwise quadratic target), Quantum achieves NLL 0.2515 vs Logistic 0.3155 ($p < 10^{-4}$). **However**, the classical Gated Head achieves NLL 0.2560 (statistically indistinguishable from Quantum). The circuit has **zero advantage over classical nonlinear models**; it only demonstrates generic nonlinear expressivity over an under-parameterized linear model. |
| **C-08** | §2.6, Table 3 | *Glioma grading on BraTS 2020: Quantum Kernel Learning (QKL) does not outperform classical RBF kernel (AUROC 0.9407 vs 0.9625, DeLong $p=0.191$).* | **VERIFIED** | `grade_mapping_check.json`, `qkl_brats2020_official.json`, `PREREGISTRATION_QKL_BRATS2020.md` | **Verified.** Erratum resolved. Re-embedding of all 369 official subjects confirmed that QKL yields no statistically significant advantage over standard classical RBF kernel across 5-fold nested CV and the 111 held-out subjects. |
| **C-09** | §2.7, Table 2 | *Frozen external transfer to CheXpert: Quantum VQC ranks last of four on every clinical metric.* | **VERIFIED** | `chexpert_external.json`, `aura.ml.evaluation.chexpert_external` | **Verified.** On 1,600 CheXpert external test studies, Quantum VQC achieved NLL 1.5782, Accuracy 0.4721, ECE 0.1586, Macro-AUROC 0.7259 (ranked last among Logistic, PoE, Gated Head, and VQC). **Attribution Caveat:** This cannot be causally attributed to parameter count or "quantum overfitting" without controlling for feature and label shift. |
| **C-10** | §2.1, Table 5 | *Field audit of 5 published QML studies: under symmetric classical selection, zero reported quantum advantages survived.* | **VERIFIED** | `bench/field_audit/results/*.json`, `CHECKLIST_SCORES_FIELD_AUDIT.md` | **Verified.** Re-implementations on open data (CKD, Parkinson's, Pima, EEG, Glioma) show that reported quantum advantages collapse when classical baselines are tuned or when data leakage is removed. In CKD, classical beats quantum by 9.7 points ($p < 0.001$). |
| **C-11** | §3.2, M1-M3 | *Mechanistic explanation: the fusion circuit computes a low-order trigonometric polynomial; fidelity kernel is identically a classical product of cosines.* | **VERIFIED** | `v6_mechanism.json`, `aura.ml.evaluation.v6_mechanism` | **Theoretically & Empirically Verified.** RY($\pi x$) angle encoding with Pauli-Z readout yields a Fourier series of degree $\le L$. For product states, the fidelity kernel analytically reduces to $\prod_{i=1}^n \cos^2(\frac{\pi (x_i - x'_i)}{2})$, verified to numerical precision ($| \Delta | < 10^{-15}$). It can be evaluated in $O(n)$ classical time. |
| **C-12** | §3.3 | *Input-provenance cryptographic assertion protocol (Defect D14) prevents evaluation against synthetic test stubs.* | **VERIFIED** | `evidence_ledger.json`, `provenance_assertion.py`, `aura/tests/test_manuscript_numbers.py` | **Verified.** SHA-256 byte-digests of raw evidence arrays and predictions are pinned and enforced in CI tests. |

---

## 3. High-Priority Threat Analysis: Ten Threats to Validity

We evaluate the ten most critical threats to the manuscript's thesis, ranked by:
1. **P(Error):** Probability that the stated claim or interpretation is invalid.
2. **Impact:** Consequence for the central scientific thesis.
3. **Verification Cost:** Computational and analytic effort to verify or correct.
4. **Information Gain:** Value added to the scientific rigor of the paper.

```
+----+-----------------------------------------------------+----------+--------+---------+-----------+
| Rk | Threat Description                                  | P(Error) | Impact | Cost    | Info Gain |
+----+-----------------------------------------------------+----------+--------+---------+-----------+
| 1  | Heuristic Rule Emulation vs True Clinical Diagnosis | 100%     | HIGH   | LOW     | CRITICAL  |
| 2  | Conflating Absence of Difference with Equivalence   | 90%      | HIGH   | MEDIUM  | HIGH      |
| 3  | Positive Control Over-interpretation on T2b         | 85%      | HIGH   | LOW     | CRITICAL  |
| 4  | Causal Attribution of External Degradation          | 80%      | MEDIUM | MEDIUM  | HIGH      |
| 5  | Field Audit Classical Asymmetry (Reverse Bias)      | 60%      | HIGH   | MEDIUM  | HIGH      |
| 6  | Generalization of Entanglement Penalty              | 70%      | MEDIUM | LOW     | HIGH      |
| 7  | Upstream Encoder Contamination in Medical Benchmarks| 40%      | HIGH   | LOW     | HIGH      |
| 8  | Simulation vs Hardware Generalization               | 75%      | MEDIUM | LOW     | MEDIUM    |
| 9  | Multi-Testing Correction Across Hypothesis Families | 50%      | MEDIUM | LOW     | MEDIUM    |
| 10 | CheXpert Unreachable Class (COPD) Bias              | 30%      | LOW    | LOW     | LOW       |
+----+-----------------------------------------------------+----------+--------+---------+-----------+
```

### Detailed Breakdown of Top 4 Critical Threats

### Threat 1: Heuristic Rule Emulation vs True Clinical Diagnosis
- **The Vulnerability:** The six-class diagnostic target in MIMIC-CXR and CheXpert is constructed via `map_diagnosis`, an explicit weighted linear sum ($0.7 \times \text{edema} + 0.5 \times \text{cardiomegaly} + \dots$). The input features to the fusion layer are the predicted probabilities of those exact concepts from the DenseNet.
- **Scientific Reality:** The task is **rule emulation**, not clinical diagnostic inference. Because the true underlying target function is additive and linear, linear models (PoE, Logistic Regression) are structurally Bayes-optimal. A null result for a nonlinear quantum circuit on this task is mathematically trivial.
- **Remediation:** Explicitly disclose that the primary task is clinical rule emulation. Reframe the paper: the primary task demonstrates that when clinical workflows use additive scoring rules, adding quantum circuits introduces unhelpful capacity and noise; the positive controls (T2a, T2b) are mandatory to test whether the circuit can learn non-additive structures.

### Threat 2: Conflating Lack of Significance with Statistical Equivalence
- **The Vulnerability:** Claiming "quantum and classical models achieve parity" or "equivalence" based on $p > 0.05$ on $n=447$ MIMIC studies is a textbook statistical error. With $n=447$, the 90% CI for $\Delta\text{NLL}$ is $[-0.027, +0.034]$, vastly wider than the pre-registered equivalence margin $\Delta = \pm 0.010$.
- **Scientific Reality:** The $n=447$ cohort is statistically underpowered to reject non-equivalence. Only an out-of-sample evaluation with $n > 40,000$ (e.g. the 56,534 untouched CheXpert patient cohort) can narrow the CI sufficiently to test equivalence.
- **Remediation:** Maintain strict language on $n=447$: "no difference was detected; equivalence was not established." Confirmatory equivalence claims must rely exclusively on the high-$n$ untouched test cohort.

### Threat 3: Positive Control Over-interpretation on T2b
- **The Vulnerability:** The manuscript highlights that on quadratic target T2b, the quantum circuit significantly outperforms logistic regression ($p < 10^{-4}$), framing this as "detecting an advantage for the quantum circuit itself."
- **Scientific Reality:** The classical Gated Head achieves NLL 0.2560 compared to Quantum's 0.2515. The difference is negligible. The experiment proves only that *nonlinear models beat linear models on nonlinear targets*, NOT that quantum models beat classical nonlinear models.
- **Remediation:** Remove any phrasing suggesting a quantum-specific advantage on T2b. Report it cleanly as: "The positive control confirms the comparison apparatus detects nonlinear structure, but the quantum circuit matches, and does not exceed, a standard classical gated head."

### Threat 4: Causal Attribution of External Degradation
- **The Vulnerability:** In the external transfer to CheXpert, the quantum circuit degraded most. The text suggests this is due to quantum parameter capacity or inductive bias vulnerability to domain shift.
- **Scientific Reality:** CheXpert has known systematic differences from MIMIC-CXR: different patient demographics, different image acquisition protocols, different labeler policies, and COPD is unreachable. Comparing four architectures at a single operating point cannot distinguish parameter capacity from optimization instability or feature shift.
- **Remediation:** Reframe as descriptive external validation. Test competing hypotheses (calibration drift vs feature representation shift) using learning curves.
