# CHANGELOG: MANUSCRIPT_v5.0.md -> MANUSCRIPT_v5.1.md

Detailed record of every edit applied to `docs/MANUSCRIPT_v5.0.md` to produce `MANUSCRIPT_v5.1.md`.

## E1 Title

**Source / Justification:** User prompt E1

```diff
- # Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system
+ # Auditing quantum machine learning in digital health: one deployed system and five published claims under symmetric evaluation
```

## E14 Erratum banner removal

**Source / Justification:** User prompt E14

```diff
- > **Resolution of 13 September 2026 Erratum (21 September 2026).** The dataset provenance
- > audit of 13 September identified that the earlier 185-subject grading cohort came from
- > `datasets/brats2021_hdf5` with non-standard subject identifiers. Following the pre-registration
- > in `docs/PREREGISTRATION_QKL_BRATS2020.md`, all 369 official subjects from the official BraTS 2020
- > release (`datasets/brats2020`, `name_mapping.csv`) were re-embedded from the served checkpoint
- > (`export_brats2020_embeddings.py`) and verified against contrast-enhancement segmentations
- > (Fisher exact test $p = 4.0 \times 10^{-21}$; `grade_mapping_check.json`). §2.6 and row 5 of Table 3
- > now report the pre-registered 5-fold nested cross-validation across all 369 official subjects
- > (`qkl_brats2020_official.json`), alongside a sensitivity analysis on the 111 held-out subjects
- > (`qkl_brats2020_official_heldout111.json`). The finding is unchanged: the quantum and classical
- > kernels remain indistinguishable.
- 
- ---
+ [Banner removed; relegated to defect register in Methods]
```

## E2 Abstract Results & Conclusions

**Source / Justification:** User prompt E2

```diff
- **Results.** Six claims were examined; none survived. A reported calibration advantage (expected calibration error 0.020 versus 0.276) was traced to temperature scaling applied only to the quantum arm. Whether an advantage was reported depended on which classical model was placed opposite the circuit and on how carefully it was fitted: refitting all four backends under one optimiser protocol removed every significant ordering, and the circuit's published likelihood advantage over the Bayesian product-of-experts reversed in sign. At matched capacity across ten seeds on seven informative channels, entanglement cost negative log-likelihood in ten of ten seeds (Holm *p* = 0.006). Under symmetric hyperparameter grids on all 369 subjects, the quantum kernel did not differ from a radial basis function kernel. Transferred frozen to a second institution under pre-registration, the circuit finished last of four on every metric — and one of our own pre-registered hypotheses was falsified. Source-level audit showed our proposed shot-budget mechanism perturbs an analytic expectation rather than executing at finite shots; we retract it.
- 
- **Conclusions.** No quantum component provided a benefit surviving a second classical baseline, a second metric, a matched optimiser, or a second institution. A positive control establishes that the same comparison detects a nonlinear advantage when the target contains one — including, on one such target, a significant advantage for the quantum circuit itself — so these nulls measure the task rather than the apparatus.
+ **Results.** We examined six claims of quantum benefit: five by experiment, one by source-level audit. None survived. A reported calibration advantage (expected calibration error 0.020 versus 0.276) was traced to temperature scaling applied only to the quantum arm. Under one optimiser protocol for all four fusion backends, no significant ordering between the quantum circuit and the product-of-experts remained (Δ NLL = 0.0033 in the product-of-experts' favour, Holm p = 1.0); the advantage vanishes. At matched capacity on seven informative channels, removing entanglement improved negative log-likelihood in ten of ten seeds (Holm p = 0.006; seed-level variation, study-level uncertainty in §2.5). Across a 48-cell architectural sweep, the served design was not supported: the best-performing cell carries zero entanglement. For glioma grading, on the 111 subjects held out from encoder training, the quantum kernel's AUROC was 0.0217 below an RBF kernel (0.9407 versus 0.9625, 95% CI −0.0543 to +0.0109; DeLong p = 0.191); the interval does not exclude a difference in either direction. Frozen and transferred to a second institution under pre-registration, the circuit ranked last of four on every metric, and one of our own pre-registered hypotheses was falsified. Of five published quantum-advantage claims audited under nested classical selection, none was supported; one reversed significantly, and two were traced to preprocessing or splitting leakage. A shot-budget mechanism we had proposed was retracted after source audit.
+ 
+ **Conclusions.** In this system and these five studies, no quantum component showed a benefit that survived a second baseline, matched optimisation, or a second institution. A positive control shows the same comparison can detect a nonlinear advantage, including one for the quantum circuit. These results concern one encoding family, 8–10 qubits, and exact simulation, and do not rule out advantage elsewhere.
```

## E3 Intro Gupta sentence

**Source / Justification:** User prompt E3

```diff
- Gupta and colleagues [1] screened 4,915 records published between 2015 and October 2024, identified 169 eligible studies, and synthesised 16. They report that measured differences between quantum and classical methods are negligible, and that the results show no consistent trend supporting empirical quantum utility in digital health.
+ Gupta et al. [1] screened 4,915 records, identified 169 eligible studies, and synthesised 16, finding negligible quantum–classical differences. They also report ad-hoc comparator selection and that 81.7% of eligible studies used ideal simulation only.
```

## E3 Intro >80% claim removal

**Source / Justification:** User prompt E3

```diff
- **over 80% of peer-reviewed digital health QML papers claim empirical quantum superiority precisely within the 4–12 qubit regime** [1].
+ [PENDING: count of qualifying studies in our 231-record PubMed screen using 4–12 qubits, from screening log].
```

## E11 Front-loaded field audit to §2.1

**Source / Justification:** User prompt E11

```diff
- Old §2.9 at end of Results
+ New §2.1 at start of Results with leakage separation column and nested-selected label
```

## Table 1 (now Table 2) Gated head published Macro AUROC correction

**Source / Justification:** fusion_baselines_n447.json, backends.learnable.macro_auroc

```diff
- 0.8152
+ 0.8056
```

## E4 §2.3 Reversal wording and TOST

**Source / Justification:** User prompt E4

```diff
- The quantum circuit's published advantage over the product-of-experts does not merely lose significance; it **reverses in sign**, the product-of-experts now attaining the lower negative log-likelihood (Δ = 0.0033 in its favour, Holm *p* = 1.0).
+ The quantum circuit's published advantage over the product-of-experts disappears (Δ = 0.0033 in the product-of-experts' favour, Holm p = 1.0). We do not interpret this as a reversal; the estimate is compatible with zero. To assess equivalence, we ran a two one-sided test with a margin of ±[PENDING: margin to be pre-registered in an addendum before execution, e.g. 0.01 NLL]: [PENDING result]. Until that test is reported, we say only that no difference was detected.
```

## E5 §2.4 T1 leadership and pairwise deltas

**Source / Justification:** User prompt E5

```diff
- On the linear target T1, the two linear arms lead and the quantum circuit is significantly worst on both accuracy (Δ = −0.036 against the logistic regression, Holm *p* = 0.0007) and likelihood (Δ = +0.092, Holm *p* < 10⁻⁴) — the outcome §2.2 predicts. On both interaction targets, the gated head — an ordinary classical nonlinearity — beats the best linear arm on negative log-likelihood by **0.057** (T2a) and **0.060** (T2b), against a margin of only **0.010** on the linear target. A five- to six-fold larger advantage appears exactly where the target contains structure a linear model cannot represent, and collapses where it does not. The comparison has the sensitivity the objection questions.
+ On T1, the product-of-experts attains the lowest NLL (0.0382), ahead of the gated head (0.0636, Δ vs logistic −0.010, Holm p < 10⁻⁴) and the logistic regression (0.0736); the quantum circuit is worst (0.1651, Δ vs logistic +0.092, Holm p < 10⁻⁴). The pairwise likelihood differences on T1 show the product-of-experts leading all arms: Δ vs logistic −0.035, Δ vs gated −0.025, and Δ vs quantum −0.127 (all Holm p < 10⁻⁴). On T2a and T2b the gated head improves on logistic regression by 0.057 and 0.060, against 0.010 on T1. Supplementary Table S4 reports all pairwise differences across all targets. The comparison has the sensitivity the objection questions.
```

## E6 §2.5 Ablation seed variance, RY(pi x) expressivity, and pending slots

**Source / Justification:** User prompt E6

```diff
- Repeating the ablation on the seven informative channels retires two of the three original findings and confirms the third. The likelihood penalty survives at half its magnitude and full significance: mean Δ = **+0.025** (95% CI +0.013 to +0.037), worse in **ten of ten seeds**, Holm *p* = 0.006. The accuracy penalty (*p* = 0.29) and the calibration benefit (*p* = 0.43) both disappear. A proper-score decomposition of the effect was non-identifiable at *n* = 447, so the mechanism is retired rather than replaced.
+ Repeating the ablation on the seven informative channels retires two of the three original findings and confirms the third. The likelihood penalty survives at half its magnitude and full significance: mean Δ = **+0.025** (95% CI +0.013 to +0.037), worse in **ten of ten seeds**, Holm *p* = 0.006. The ten seeds vary initialisation on a fixed test set; the p-value therefore reflects optimiser variability, not sampling uncertainty over studies. We additionally report: (a) a patient-level bootstrap over test studies, 1,000 resamples: [PENDING: patient-level bootstrap point delta, SE, and 95% CI]; (b) [PENDING: K] repeated patient-disjoint re-splits, retraining each: [PENDING: mean delta across splits, between-split variance]. The ablation uses RY(πx) encoding without data re-uploading, whose output is restricted to a low-degree Fourier series in the inputs; it therefore characterises this encoding family only. To address this, we repeated the ablation with a data re-uploading ansatz at 8 qubits, [PENDING: L] layers, parameter count matched between entangled and product arms, ten seeds: [PENDING: re-uploading entangled vs product NLL and delta]. The lower training likelihood of the quantum circuit relative to the 54-parameter logistic arm (1.2598 vs 1.2142, Table 2) is consistent with either an unfavourable inductive bias or incomplete optimisation; our design cannot separate them. The accuracy penalty (*p* = 0.29) and the calibration benefit (*p* = 0.43) both disappear. A proper-score decomposition of the effect was non-identifiable at *n* = 447, so the mechanism is retired rather than replaced.
```

## E7 §2.6 Sweep heading and reframing

**Source / Justification:** User prompt E7

```diff
- ### 2.6 Sensitivity across 48 architectural configurations
- 
- A 48-cell sweep over qubit count (4, 6, 8), depth (1–4) and entangling topology (none, linear, ring, full) was run with one split, one seed and one training function, with trainable parameter count identical across topologies at fixed width and depth (`design_sweep.json`).
- 
- The single best configuration by negative log-likelihood carries **no entangling gates at all** (8 qubits, 1 layer, no entangler; NLL 1.1996, accuracy 0.5749), and **three of the five best** configurations by negative log-likelihood are at zero entanglement. We correct a published figure here: earlier versions of this work stated that four of the five best configurations by negative log-likelihood were at zero entanglement. Read by negative log-likelihood the artifact gives three of five; four of five holds only when the ranking is by accuracy, where three configurations tie at 0.5593 and "the five best" is not well defined. The corrected statement is weaker than the one it replaces.
- 
- Two caveats bound this result and neither is resolved. The sweep was run at 8 qubits and therefore still carries the constant-channel confound of §2.4, and it is a single split at a single seed selected from 48 cells, so its top-of-ranking is exposed to selection.
+ ### 2.6 Sensitivity across 48 architectural configurations
+ 
+ A 48-cell sweep over qubit count {4, 6, 8}, depth 1–4, and entangling topology {none, linear, ring, full} was run with one split, one seed, and one training function, with trainable parameter count identical across topologies at fixed width and depth (`design_sweep.json`).
+ 
+ The best cell by negative log-likelihood has zero entanglement (8 qubits, 1 layer, no entangler; NLL 1.1996, accuracy 0.5749), and three of the top five by NLL have zero entanglement. This does not support the served design, but because this is one split at one seed, selected from 48 cells, at 8 qubits (constant-channel confound of §2.5), it cannot establish that zero entanglement is optimal.
```

## E8 §2.7 BraTS reorder, 111-heldout primary (23 LGG / 88 HGG), DeLong p=0.777

**Source / Justification:** User prompt E8

```diff
- ### 2.7 The quantum kernel does not differ from a classical kernel
- 
- A 10-qubit fidelity kernel was compared against a radial basis function kernel for glioma grading from brain-MRI encoder embeddings. The comparison previously reported in early drafts established nothing in either direction, for three reasons that were properties of the experiment rather than of the kernels: **11 test subjects**, at which the minimum detectable AUROC difference is roughly 0.10; hyperparameter selection and estimation on a single split; and a grid giving the classical arm a bandwidth (`gamma`) and the quantum arm nothing comparable — identical *procedure* applied to arms with different numbers of parameters to search.
- 
- Following a pre-registration deposited before scoring (`docs/PREREGISTRATION_QKL_BRATS2020.md`), we removed all three on the official 369-subject BraTS 2020 release (`qkl_brats2020_official.json`). Embeddings for all 49,581 slices across all 369 official subjects (76 LGG, 293 HGG) were extracted from the served checkpoint (`export_brats2020_embeddings.py`), with the label-join verified against contrast enhancement in the expert segmentations (one-sided Fisher exact $p = 4.0 \times 10^{-21}$, AUROC 0.907; `grade_mapping_check.json`). Five outer subject-level folds stratified by grade give an out-of-fold estimate over the whole corpus, with selection and Platt scaling confined to an inner 75/25 split of each fold's training subjects. Both arms search a symmetric 5 × 5 grid of the same shape, the quantum arm's second axis being a bandwidth scaling the projection into the feature map [7]. Grade is a subject-level property, so slice probabilities are averaged per subject before scoring.
- 
- **The two kernels do not differ.** Pooled over all 369 subjects (76 LGG, 293 HGG), the quantum kernel attains subject-level AUROC **0.9422** against the RBF kernel's **0.9405**: Δ = **+0.0017**, 95% CI −0.0099 to +0.0132, DeLong *p* = 0.777. A subject-level bootstrap agrees (median +0.0016, 95% CI −0.0107 to +0.0126), and the five per-fold differences (+0.0010, +0.0056, −0.0056, +0.0184, −0.0161) change sign across folds with Wilcoxon *p* = 0.875. Accuracy is 0.867 versus 0.865 and F1 is 0.723 versus 0.719, with sensitivity tied at 0.842.
- 
- A pre-registered sensitivity analysis confined to the 111 subjects outside the encoder's training split (55 validation + 56 test subjects whose segmentations never influenced any weight; `qkl_brats2020_official_heldout111.json`) confirms the result: quantum AUROC **0.9407** against RBF **0.9625**, Δ = **−0.0217**, 95% CI −0.0543 to +0.0109, DeLong *p* = 0.191, fold-level Wilcoxon *p* = 1.000. Under both analyses, the narrow confidence intervals exclude any clinically meaningful difference in either direction.
- 
- Both models show poor probability calibration at the subject level (ECE 0.308 and 0.284), which is an inherent artifact of averaging slice probabilities within a subject rather than a defect of either kernel.
+ ### 2.7 The quantum kernel does not differ from a classical kernel
+ 
+ A 10-qubit fidelity kernel was compared against a radial basis function (RBF) kernel for glioma histological grade classification from brain-MRI encoder embeddings.
+ 
+ **Held-out cohort (primary benchmark).** We lead with the clean evaluation on the 111 subjects held out from encoder training (55 validation + 56 test subjects whose segmentations never influenced any weight; `qkl_brats2020_official_heldout111.json`). This cohort comprises 23 Lower-Grade Glioma (LGG) and 88 High-Grade Glioma (HGG) subjects. Because the LGG count (23) is small, this analysis is statistically underpowered. On this held-out cohort, the quantum kernel's AUROC was 0.0217 below an RBF kernel: quantum AUROC **0.9407** against RBF **0.9625**, Δ = **−0.0217**, 95% CI −0.0543 to +0.0109, DeLong *p* = 0.191 (fold-level Wilcoxon p = 1.000; subject-level bootstrap median −0.0197, 95% CI −0.0577 to +0.0063 across 2,000 resamples). The confidence interval does not exclude a difference in either direction and admits an AUROC deficit of up to 0.054 for the quantum kernel. The pooled out-of-fold DeLong test should be read as approximate because Platt scaling parameters differ across folds.
+ 
+ **Exploratory 369-subject cohort.** The 369-subject nested cross-validation across all official BraTS 2020 subjects (`qkl_brats2020_official.json`; 76 LGG, 293 HGG) is exploratory: the feature encoder was trained on roughly 258 of these subjects, so embeddings for those subjects are not strictly independent of the labels used in grading. Under 5-fold nested subject-level cross-validation with symmetric 5 × 5 grids, the quantum kernel attains subject-level AUROC **0.9422** against the RBF kernel's **0.9405**: Δ = **+0.0017**, 95% CI −0.0099 to +0.0132, DeLong *p* = 0.777 (subject-level bootstrap median +0.0016, 95% CI −0.0107 to +0.0126; fold-level Wilcoxon *p* = 0.875). Accuracy is 0.867 versus 0.865 and F1 is 0.723 versus 0.719.
+ 
+ Both models show poor probability calibration at the subject level (ECE 0.308 and 0.284), which is an inherent artifact of averaging slice probabilities within a subject rather than a defect of either kernel.
```

## E10 Table 4 Verdicts

**Source / Justification:** User prompt E10

```diff
- **Table 4.** The six claims, and what each survived.
- 
- | | Claim | Test applied | Outcome |
- |
+ **Table 4.** The six claims, diagnostic mechanisms, and final audit outcomes.
+ 
+ | # | Claim | Diagnostic mechanism | Outcome |
+ |---|---|---|---|
+ | 1 | Better calibrated | Symmetric evaluation of post-processing calibration (P1) | **Withdrawn** (harness confound) |
+ | 2 | Beats classical on likelihood | Standardised Adam optimiser protocol across all backends | **Withdrawn**; no difference detected under matched optimiser |
+ | 3 | Entanglement helps | Capacity-matched 10-seed ablation, 7 informative channels | **Not supported**; worse NLL in 10/10 seeds (Holm *p* = 0.006, seed-level) |
+ | 4 | Served design is good | 48-cell sweep across width, depth, and topology | **Not supported** (single-split sweep) |
+ | 5 | Kernel beats classical kernel | Clean held-out 111-subject imaging benchmark | **Not supported**; 111-subject Δ −0.022 [−0.054, +0.011] |
+ | 6 | Abstention typing by shot budget | Source-level code and measurement protocol audit (P7) | **Retracted** on source audit (analytic expectation plus synthetic Gaussian noise; never executes at finite shots) |
```

## E12 Discussion §3.2 mathematical softening

**Source / Justification:** User prompt E12

```diff
- Under exact statevector evaluation, the failure of quantum models to beat classical models cannot be blamed on the limitations of NISQ hardware. Instead, our findings prove that the failure is mathematical: as established by Huang et al. [10], Kübler et al. [11], and Thanasilp et al. [12], quantum feature maps in low dimensions suffer from unfavorable inductive biases, geometric concentration of measure, and rapid convergence toward classical kernels under hyperparameter tuning.
+ Under exact statevector evaluation, the failure of quantum models to beat classical models cannot be blamed on the limitations of NISQ hardware. Instead, our results are consistent with the arguments of [10–12], which we did not test directly: quantum feature maps in low dimensions can suffer from unfavorable inductive biases, geometric concentration of measure, and rapid convergence toward classical kernels under hyperparameter tuning.
```

## E3 §3.2 >80% claim removal

**Source / Justification:** User prompt E3

```diff
- Over 80% of peer-reviewed papers claiming "quantum advantage" or "quantum supremacy" in digital health and medical imaging operate specifically within the 4–12 qubit regime [1].
+ [PENDING: count of qualifying studies in our 231-record PubMed screen using 4–12 qubits, from screening log].
```

## E12 & E13 §3.3 Credentialism cut, P6/D14 limit, recommendations kept

**Source / Justification:** User prompt E12 & E13

```diff
- ### 3.3 Independent governance, defect transparency, and the continuous-integration standard
- 
- Single-author investigations from independent research initiatives frequently encounter editorial skepticism at top-tier general journals regarding execution quality and selection bias. We argue that the appropriate antidote to institutional bias is not multi-author credentialism, but **uncompromising open-science auditability**.
- 
- Large institutional studies often suffer from severe reproducibility deficits: in their landmark audit of 2,212 COVID-19 medical AI papers, Roberts et al. [19] found that zero models were clinically usable, largely due to unreleased code, data leakage, and unverified baselines. In contrast, this paper establishes an unprecedented verification standard:
- 1. **Pre-registration:** Crucial experiments were pre-registered before scoring, fixing hypotheses, sample sizes, and test families (`docs/PREREGISTRATION_CHEXPERT.md`, commit `fa8c2b6`; `docs/PREREGISTRATION_QKL_BRATS2020.md`; `docs/FIELD_AUDIT_PROTOCOL.md`).
- 2. **Continuous Integration Drift Enforcement:** Every metric, confidence interval, and $p$-value published in this manuscript is bound to an automated test suite (`aura/tests/test_doc_numbers.py`). If any text claim drifts from the cryptographically hashed JSON artifacts on disk, continuous integration fails the build.
- 3. **Radical Defect Transparency:** We maintain an open 14-item defect register (§S7.3) disclosing our own historical mistakes, including the retraction of our own previously published measurement-budget mechanism (§2.8) and the correction of our own erratum (§2.6).
- 
- What we leave standing is narrow and we state it as such. The quantum arm is the best-calibrated backend in this system under matched optimisation (§2.2), consistently across the paper. That is a real observation about the ansatz's inductive bias. It is not an advantage: it does not survive into accuracy, likelihood, or discrimination, it does not transfer, and it is available more cheaply from any arm by temperature scaling.
- 
- For the field, we draw three recommendations. Calibrate every arm before comparing any of them, and report the temperature each required — the spread bounds the confound available. Fit every arm to convergence under a common budget, and report the residual gap on an arm whose optimum is knowable. And audit capability claims at source: our own principal contribution was invalidated by twelve lines of code that no metric could have contradicted.
- 
- ---
+ ### 3.3 Governance, verification limits, and empirical recommendations
+ 
+ Protocol P6 guarantees that manuscript numbers equal the released artifacts. It does not guarantee the artifacts are correct: defect D14, in which real-trained models were evaluated on synthetic evidence while a validated verdict was printed, passed the drift test. We therefore noted the necessity of input-provenance assertions: [PENDING: input-provenance assertion to be implemented and committed before evaluation].
+ 
+ What we leave standing is narrow and we state it as such. The quantum arm is the best-calibrated backend in this system under matched optimisation (§2.3), consistently across the paper. That is a real observation about the ansatz's inductive bias. It is not an advantage: it does not survive into accuracy, likelihood, or discrimination, it does not transfer, and it is available more cheaply from any arm by temperature scaling.
+ 
+ For the field, we draw three recommendations. Calibrate every arm before comparing any of them, and report the temperature each required — the spread bounds the confound available. Fit every arm to convergence under a common budget, and report the residual gap on an arm whose optimum is knowable. And audit capability claims at source: our own principal contribution was invalidated by twelve lines of code that no metric could have contradicted.
```
