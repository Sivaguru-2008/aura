# Additional file 3 — Supplementary Information

**Auditing quantum machine learning in digital health: one deployed system and four published studies under symmetric evaluation**

Sivaguru R. M. (Independent researcher) — Submission version 5.1, October 2026

This document accompanies the main manuscript (`MANUSCRIPT_v5.1.md`). Section numbers prefixed **S** are Supplementary; unprefixed numbers refer to the main text.

---

## Contents

| | |
|---|---|
| S1 | System under evaluation |
| S2 | Prediction target and its derivation |
| S3 | Comparison protocol and its provenance |
| S4 | Extended results |
| S5 | Positive controls |
| S6 | Selective prediction, and a withdrawn claim |
| S7 | Threats to validity, limitations, and a defect register |
| S8 | Analyses identified as necessary and not performed |
| S9 | Map from the full-length preprint (v4.3) to this submission |
| S10 | Full reference list |

---

## S1. System under evaluation

AURA is an offline clinical decision-support system. Nothing leaves the host; there is no external inference call at any point in the pipeline.

**Chest radiography path.** A DenseNet-121 at 224 × 224 produces seven per-finding probabilities (opacity, consolidation, effusion, cardiomegaly, nodule, hyperinflation, pneumothorax), each Platt-scaled per finding on a held-out calibration split. Grad-CAM++ supplies localisation, from raw activations, with calibrated probability used only to modulate overlay alpha.

**Brain MRI path.** A 2-D residual U-Net with five heads (segmentation, presence, size, quality, embedding) trained on BraTS-2020, serving per-slice segmentation and a Platt-calibrated presence probability. A 128-dimensional encoder bottleneck embedding is stored per slice with the checkpoint that produced it.

**Evidence fusion.** An encoder assembles the chest findings into an eight-channel evidence vector. Five backends are interchangeable behind one interface:

| Backend | Form | Trainable parameters |
|---|---|---:|
| Product-of-experts | `softmax(Wx + b)` | 54 |
| Quantum VQC | `RY(π·xᵢ)` encoding, 3 layers of trainable RY/RZ plus a CNOT ring, `⟨Zᵢ⟩` readout, linear head | 102 |
| Gated head | `(σ(Ax + c) ⊙ x)Wᵀ + b` | 126 |
| Logistic regression | `softmax(Wx + b)`, L-BFGS to optimality | 54 |
| Deep ensemble | 5 bootstrap members of the product-of-experts | 270 |

Parameter counts are read from the served artifacts (`fusion_{classical,quantum,learnable}.npz`). The gated head was published as 54 parameters through four revisions of this work; it has 126 (A 8×8, c 8, W 6×8, b 6), which makes it the *largest* arm rather than a smaller one. No artifact recorded a parameter count for any arm, so nothing could have caught the error; all four are now recorded in `fusion_baselines_n447.json`.

**Quantum kernel.** A separate 10-qubit IQP-style fidelity kernel over a PCA projection of the brain encoder embedding, classifying glioma grade (HGG/LGG). The three-class subtype surface it was originally sketched against cannot be honestly trained — the repository holds no meningioma or metastasis imaging — and returns a uniform prior with `abstained = True`.

**Safety layer.** Temperature scaling, split-conformal prediction with class-conditional variants and adaptive conformal inference, deep-ensemble and perturbation-based epistemic uncertainty, energy-score out-of-distribution detection, and a four-criterion abstention policy (§S6).

**Simulation and hardware.** PennyLane `default.qubit` for all reported experiments except one fusion circuit executed on IBM `ibm_marrakesh` (156 qubits, job `d9js49rjf64c739haeg0`); transpilation and noise modelling used Qiskit against `FakeMarrakesh`.

---

## S2. Prediction target and its derivation

The six-way target is derived from the free-text report by rule (`aura/mimic/labeling.py`, `_map_diagnosis`). Concepts are extracted with CheXpert-style negation and uncertainty scoping and mapped by a fixed weighted score:

| Category | Score contributions |
|---|---|
| Pneumothorax | pneumothorax 1.0 |
| Malignancy | nodule 1.0 |
| Heart failure | oedema 0.7 + cardiomegaly 0.5 + effusion 0.2 |
| Pneumonia | pneumonia 0.9 + consolidation 0.6 + opacity 0.3 |
| COPD | hyperinflation 0.9 |
| Normal | assigned when no pathological concept is positive |

The highest-scoring category is taken, ties broken by a fixed clinical-acuity priority (pneumothorax > malignancy > heart failure > pneumonia > COPD > normal). Patient-level labels aggregate across studies and take the highest-acuity category observed.

Three consequences bound every result in the main text.

**These are radiographic-pattern proxies, not clinical diagnoses.** "Malignancy" is assigned on a nodule mention; no histology, follow-up or outcome data enter. "Heart failure" is a radiographic constellation; no echocardiography, natriuretic peptide or clinical assessment enters.

**The target is a deterministic function of the same concepts that generate the model's input features.** The fusion task is recovery of a known weighted-sum-then-argmax map from noisy estimates of its arguments. The only thing standing between input and target is the vision model's error.

**The Bayes-optimal decision rule is therefore linear in the evidence vector, up to the tie-break.** Previous versions of this work asserted this from the labeller's source. §2.4 measures it: a multinomial logistic regression fitted against the noise-free target attains accuracy **0.9991** against an interaction model's 1.0000. The residual 0.0009 is the tie-break and the constant "normal" floor, neither of which is linear. This is a task with no inter-channel nonlinearity for an entangling ansatz to exploit, and the null of §2.3 is a predicted outcome for it.

**What is and is not validated.** The seven *finding* labels feeding the evidence vector are validated against a hand-read reference standard (§S5.2). The six-way *category map above is not validated*: its weights are transparent clinical heuristics chosen by the author, no clinician reviewed them, and no agreement study was performed against clinician-assigned categories.

**Class support** on the 447-study test split: heart failure 106, normal 102, pneumonia 98, COPD 67, malignancy 59, pneumothorax 15. Majority-class accuracy 0.2371.

---

## S3. Comparison protocol and its provenance

**P1. Per-arm calibration.** Each backend fits its own temperature on its own calibration split before any metric is computed.

**P2. Matched capacity within an ablation.** Compared arms have identical trainable parameter counts and are trained by the same function, on the same data, with the same seed. Parameter count is a verifiable but weak proxy for capacity in quantum models; effective-dimension matching was not performed.

**P3. Disjoint splits at the correct unit.** Patient-disjoint for fusion, volume-disjoint for segmentation, subject-disjoint for grading, with resampling at the same unit.

**P4. Significance before ranking.** Every ordering claim carries a test, an interval and a multiplicity policy.

**P5. Publication of negative outcomes.** Every comparison run is reported regardless of direction.

**P6. Artifact-derived reporting.** Every published figure is read from the artifact that produced it, enforced by `aura/tests/test_doc_numbers.py`.

**P7. Audit of mechanisms, not only metrics.** A claim about what a system can express is a claim about its implementation and is verified at source. Metrics cannot falsify a capability that holds by construction.

**P8. Optimiser symmetry** *(added at this revision)*. Where arms of different model classes are compared, they are fitted under one optimiser protocol with the same budget and the same per-arm hyperparameter search, and the residual optimisation gap is measured on an arm whose optimum is knowable. §2.3 is the demonstration that this is not a hypothetical concern: converging one arm moved it further than the quantum effect the project opened with.

**Provenance.** We state what can be evidenced and no more. P1 was written in response to the audit reported in §2.2 and was in force for every subsequent comparison. P2 to P6 were formulated during the same revision period as the experiments they govern; **no timestamped pre-registration is held for them**, and artifact timestamps establish only when experiments ran, not when the rules were written. P7 was written as a consequence of the finding in §2.9 and was not in force when the mechanism it invalidated was proposed. P8 was written as a consequence of §2.3. Only §2.8 is governed by a pre-registration deposited before its run (`docs/PREREGISTRATION_CHEXPERT.md`, commit `fa8c2b6`), which is the only form in which the pre-specification claim is verifiable rather than asserted.

---

## S4. Extended results

### S4.1 The calibration confound, measured

**Table S1.** Per-backend calibration on the 446-study calibration split. Source: `backend_calibration.json`.

| Backend | Temperature | Conformal q̂ | Empirical coverage | Mean set size | Test accuracy | Test NLL | Test ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Product-of-experts | 0.479 | 0.9098 | 0.9038 | 3.539 | 0.5414 | 1.3026 | 0.1266 |
| Quantum VQC | 0.906 | 0.9144 | 0.9150 | 3.537 | 0.5324 | 1.2853 | 0.0896 |
| Gated head | 0.887 | 0.9191 | 0.9083 | 3.421 | 0.5526 | 1.2663 | 0.0902 |

**Temperature spread: 1.89.** This is the factor by which the three served backends' calibration requirements differ, and therefore the headroom available to any comparison that calibrates one arm and not another.

### S4.2 Fusion comparison: all pairwise tests

**Table S2.** Pairwise comparisons on the 447-study split, as published and under the matched optimiser protocol. Holm within each metric family of six. Sources: `fusion_baselines_n447.json`, `fusion_matched_optimiser.json`.

| Comparison | Δ acc. (pub.) | McNemar Holm (pub.) | Δ NLL (pub.) | Wilcoxon Holm (pub.) | Δ acc. (matched) | McNemar Holm (matched) | Δ NLL (matched) | Wilcoxon Holm (matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PoE vs. quantum | +0.0090 | 1.000 | +0.0173 | 0.006 | +0.0157 | 1.000 | −0.0033 | 1.000 |
| PoE vs. gated | −0.0112 | 1.000 | +0.0363 | <10⁻⁵ | −0.0089 | 1.000 | +0.0071 | **0.044** |
| PoE vs. logistic | −0.0156 | 1.000 | +0.0716 | 1.7×10⁻⁷ | +0.0023 | 1.000 | +0.0052 | 0.643 |
| Quantum vs. gated | −0.0202 | 1.000 | +0.0190 | 0.052 | −0.0246 | 0.914 | +0.0104 | 0.896 |
| Quantum vs. logistic | −0.0246 | 1.000 | +0.0543 | **0.036** | −0.0134 | 1.000 | +0.0085 | 1.000 |
| Gated vs. logistic | −0.0044 | 1.000 | +0.0353 | 0.058 | +0.0112 | 1.000 | −0.0019 | 0.406 |

A positive Δ favours the second-named arm on NLL (lower is better) and the first-named arm on accuracy.

**Table S2b.** Matched-protocol fitting diagnostics. All three trained arms converged by early stopping. Source: `fusion_matched_optimiser.json`.

| Arm | Selected lr | Best epoch | Converged | Final training NLL |
|---|---:|---:|---|---:|
| Product-of-experts | 0.05 | 32 | yes | 1.2142 |
| Gated head | 0.10 | 27 | yes | 1.1961 |
| Quantum VQC | 0.10 | 76 | yes | 1.2598 |
| Logistic (L-BFGS anchor) | — | 356 iterations | yes | — |

The quantum circuit's *training* NLL is the worst of the three at 102 parameters against 54 and 126. It underfits this task relative to a linear model with half its parameters, which is a statement about the ansatz's inductive bias on this target rather than about generalisation.

**Budget adequacy.** Adam-fitted linear arm 1.2362 versus L-BFGS optimum 1.2310 — residual gap 0.0052 NLL, 0.0023 accuracy. This gap is of the same order as the single surviving effect (0.0071), so the matched block establishes that no ordering is resolvable, not that the gated head's margin is established.

### S4.3 Entanglement ablation

**Table S3.** Ten-seed capacity-matched ablation. Positive Δ NLL means the entangled ring is worse. Sources: `entanglement_multiseed.json` (8 qubits, confounded), `entanglement_multiseed_7q.json` (7 informative channels, controlled).

| Metric | 8 qubits (confounded) | | 7 qubits (controlled) | |
|---|---:|---|---:|---|
| | Mean Δ | Seeds favouring ring | Mean Δ | Seeds favouring ring |
| Negative log-likelihood | +0.049 | 0 of 10 | **+0.025** (95% CI +0.013 to +0.037) | **0 of 10** |
| Accuracy | −0.013 | 1 of 10 | −0.006 (CI −0.015 to +0.004) | 3 of 10 |
| Expected calibration error | −0.018 | 8 of 10 | −0.006 (CI −0.017 to +0.006) | 5 of 10 |

Controlled significance: NLL Wilcoxon *p* = 0.002, Holm *p* = 0.0059, *t*-test *p* = 0.0013. Accuracy *p* = 0.291 (Holm 0.582). ECE *p* = 0.432 (Holm 0.432). Per-seed Δ NLL: +0.0039, +0.0351, +0.0347, +0.0264, +0.0630, +0.0041, +0.0202, +0.0154, +0.0220, +0.0252.

**Table S3b.** What the seven-qubit control changed.

| Finding at 8 qubits | Status at 7 qubits |
|---|---|
| Entanglement costs likelihood (10/10 seeds) | **Confirmed**, at half the magnitude, fully significant |
| Entanglement costs accuracy (9/10 seeds) | **Retired** — *p* = 0.29 |
| Entanglement improves calibration (8/10 seeds) | **Retired** — *p* = 0.43 |

A Murphy decomposition of the Brier score across ten seeds, two widths and three group counts was run to measure the calibration–refinement trade the eight-qubit result had inferred from the sign of ECE. It is **non-identifiable at *n* = 447**: the reconstruction residual exceeds the effect. The mechanism is retired, not replaced (`murphy_decomposition.json`).

### S4.4 Design sweep

48 cells: qubits (4, 6, 8) × layers (1–4) × entangler (none, linear, ring, full), one split, one seed, one training function, 20 epochs.

**Table S4.** Five best cells by negative log-likelihood. Source: `design_sweep.json`.

| Rank | NLL | Accuracy | Qubits | Layers | Entangler | Two-qubit gates |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 1.1996 | 0.5749 | 8 | 1 | **none** | 0 |
| 2 | 1.2052 | 0.5593 | 8 | 2 | **none** | 0 |
| 3 | 1.2078 | 0.5638 | 8 | 1 | full | 28 |
| 4 | 1.2237 | 0.5593 | 8 | 4 | **none** | 0 |
| 5 | 1.2282 | 0.5548 | 8 | 2 | full | 56 |

**Three of five**, and the single best cell, are at zero entanglement. Earlier versions of this work published "four of the five best configurations by negative log-likelihood"; read by negative log-likelihood the artifact gives three of five. Four of five holds only under an accuracy ranking, where ranks 4, 5 and 6 tie at 0.5593 and "the five best" is not well defined. The corrected statement is weaker than the one it replaces (defect D12, §S7.3).

The served configuration is 8 qubits, 3 layers, ring — which ranks outside the top five on both metrics.

**Caveats.** The sweep was run at 8 qubits and carries the constant-channel confound of §2.5; it has not been repeated at 7. It is one split at one seed with the top of the ranking selected from 48 cells, and is therefore exposed to selection. No interval is attached to any cell.

### S4.5 Quantum kernel, full coverage

> **Withdrawn (13 September 2026).** The grade labels behind this analysis do not trace to
> BraTS 2020; see the erratum at the head of the main text.

Design: five outer subject-level folds stratified by grade over all 185 subjects; inner 75/25 split of each fold's training subjects for hyperparameter selection and Platt scaling; 1,200 balanced fit slices and 800 calibration slices per fold; at most 30 slices per test subject so no large subject dominates; subject-level aggregation by mean predicted probability. Source: `qkl_full_coverage.json`.

**Table S5.** Per-fold results and selections. Both arms search a 5 × 5 grid: `C` ∈ {0.1, 1, 10, 100, 1000} and a bandwidth — `gamma` ∈ {0.001, 0.01, 0.1, 1, 10} for RBF, projection scaling ∈ {0.25, 0.5, 1, 2, 4} for the quantum kernel.

| Fold | Test subjects | LGG | Quantum AUROC | (C, bandwidth) | RBF AUROC | (C, γ) | Δ |
|---:|---:|---:|---:|---|---:|---|---:|
| 0 | 38 | 10 | 0.7286 | (1.0, 0.25) | 0.7143 | (10, 0.001) | +0.0143 |
| 1 | 38 | 10 | 0.8679 | (0.1, 0.5) | 0.8250 | (1, 0.01) | +0.0429 |
| 2 | 37 | 9 | 0.7857 | (1000, 0.25) | 0.7063 | (1000, 0.01) | +0.0794 |
| 3 | 36 | 9 | 0.6708 | (1.0, 0.5) | 0.7572 | (10, 0.001) | −0.0864 |
| 4 | 36 | 9 | 0.7778 | (10, 1.0) | 0.8313 | (100, 0.01) | −0.0535 |

**Table S5b.** Pooled out-of-fold, subject level, *n* = 185 (47 LGG / 138 HGG).

| | Quantum kernel | RBF kernel |
|---|---:|---:|
| AUROC | 0.7723 | 0.7749 |
| AUPRC | 0.4929 | 0.4756 |
| Accuracy | 0.6865 | 0.7405 |
| Sensitivity | 0.7447 | 0.7872 |
| Specificity | 0.6667 | 0.7246 |
| Precision | 0.4321 | 0.4933 |
| F1 | 0.5469 | 0.6066 |
| ECE | 0.2378 | 0.2389 |
| Brier | 0.2349 | 0.2358 |

**ΔAUROC = −0.0026**, 95% CI −0.0443 to +0.0391, **DeLong *p* = 0.902**. Subject-level bootstrap (2,000 resamples): median −0.0025, CI −0.0446 to +0.0372. Fold-level Wilcoxon *p* = 1.0; three of five fold differences favour the quantum kernel and two the classical, with the sign changing across folds.

**Interpretation.** This is an informative null. The interval excludes any difference larger than about 0.045 AUROC in either direction, against the superseded comparison's interval of roughly ±0.10 on 11 subjects, which could exclude nothing. Two observations qualify it. The quantum arm's selected bandwidth is at or below 1.0 in every fold and at the bottom of the grid in two, which is the compression regime; a grid extending further down might move it. And the arms differ at the operating point even though they do not differ in ranking — the classical arm attains accuracy 0.741 against 0.686 and F1 0.607 against 0.547 — because equal AUROC converted through Platt scaling yields different thresholds. Both are poorly calibrated at subject level (ECE ≈ 0.24), which is a property of averaging slice probabilities within a subject rather than of either kernel, and which no result here depends on.

**Superseded comparison.** `brain/qkl_training_report.json` (11 test subjects, single split, asymmetric grid) is retained for provenance. Its reported ΔAUROC of +0.0446 at *p* = 0.55 should not be cited; nor should the earlier 669-slice figure of −0.016 at *p* = 0.002 (defect D1, §S7.3).

### S4.6 External validation: all pairwise tests

**Table S6.** Pairwise comparisons on 4,800 CheXpert test studies, Holm within each metric family of six. Source: `chexpert_external.json`.

| Comparison | Δ accuracy | Discordant (a/b) | McNemar Holm | Δ NLL | Wilcoxon *p* | Holm |
|---|---:|---|---:|---:|---:|---:|
| PoE vs. quantum | +0.0552 | 426 / 161 | **<10⁻¹⁰** | −0.1589 | 2.6×10⁻¹⁵³ | **<10⁻¹⁰** |
| PoE vs. gated | +0.0444 | 414 / 201 | **<10⁻¹⁰** | −0.0923 | 4.2×10⁻⁸⁰ | **<10⁻¹⁰** |
| PoE vs. logistic | +0.0063 | 304 / 274 | 0.228 | +0.0378 | 3.4×10⁻³⁵ | **<10⁻¹⁰** |
| Quantum vs. gated | −0.0108 | 191 / 243 | **0.0285** | +0.0666 | 3.2×10⁻⁹⁵ | **<10⁻¹⁰** |
| Quantum vs. logistic | −0.0489 | 218 / 453 | **<10⁻¹⁰** | +0.1967 | 2.0×10⁻¹⁹⁷ | **<10⁻¹⁰** |
| Gated vs. logistic | −0.0381 | 124 / 307 | **<10⁻¹⁰** | +0.1301 | 1.9×10⁻¹⁹⁶ | **<10⁻¹⁰** |

**Sensitivity.** Mapping CheXpert's uncertain labels to positive rather than negative moves the floor to 0.3227 and every accuracy by 1 to 2 points. No conclusion changes: the circuit remains last on accuracy, NLL and calibration; all five significant accuracy comparisons remain significant; PoE versus logistic remains the one that is not (Holm *p* = 0.967).

**Deviations from the pre-registration.** None.

### S4.7 Nonlinear positive control

The control varies only the label-generating rule. Evidence vectors, splits, arms, optimiser protocol, calibration and tests are identical to §2.3's matched-optimiser experiment. Source: `nonlinear_control.json`.

**Target definitions.** **T1** reproduces the clinical target's generating rule of §S2 as a function of the evidence channels. **T2a** assigns categories by conjunction and negated conjunction of channels binarised at their corpus median — pneumothorax alone; nodule **and not** consolidation; cardiomegaly **and** effusion; consolidation **and** opacity **and not** effusion; hyperinflation **and not** opacity; normal otherwise. **T2b** is `argmax_k (a_k·x + xᵀM_kx)` with `a_k` and symmetric zero-diagonal `M_k` drawn once at seed 11.

**Ceilings** are fitted against the noise-free target over the whole corpus, because the question is representability rather than generalisation: a linear ceiling below the interaction ceiling is structure a linear model provably cannot express.

**Table S6b.** Measured structure of each target.

| Target | Linear ceiling | Interaction ceiling | Gap | Majority class | Classes absent from a split |
|---|---:|---:|---:|---:|---|
| T1 clinical (linear) | 0.9991 | 1.0000 | +0.0009 | 0.4070 | none |
| T2a conjunctive | 0.8740 | 0.9301 | +0.0560 | 0.5020 | none |
| T2b pairwise | 0.9081 | 0.9821 | +0.0740 | 0.4594 | COPD |

T1's linear ceiling of 0.9991 is the empirical confirmation of §S2's claim. The residual 0.0009 is the acuity tie-break and the constant "normal" floor, neither of which is linear.

**Table S6c.** All pairwise tests against the converged logistic regression. Negative Δ NLL means the arm beats the best linear model. Holm within each metric family per target.

| Target | Arm | Δ accuracy | McNemar Holm | Δ NLL | Wilcoxon Holm |
|---|---|---:|---:|---:|---:|
| T1 | Product-of-experts | +0.0044 | 1.000 | −0.0354 | <10⁻⁴ |
| | Gated head | +0.0022 | 1.000 | −0.0100 | <10⁻⁴ |
| | Quantum VQC | −0.0358 | **0.0007** | **+0.0915** | <10⁻⁴ |
| T2a | Product-of-experts | −0.0156 | 0.355 | +0.0265 | <10⁻⁴ |
| | Gated head | −0.0156 | 0.355 | **−0.0571** | **7.5×10⁻⁴** |
| | Quantum VQC | −0.0962 | **<10⁻⁴** | **+0.2334** | <10⁻⁴ |
| T2b | Product-of-experts | 0.0000 | 1.000 | +0.0083 | 0.220 |
| | Gated head | +0.0224 | 0.453 | **−0.0595** | <10⁻⁴ |
| | Quantum VQC | 0.0000 | 1.000 | **−0.0640** | **2.7×10⁻⁴** |

**Reading the control.** The gated head's likelihood advantage over the best linear arm is 0.010 on the linear target and 0.057 and 0.060 on the two interaction targets — five to six times larger where structure exists. The comparison therefore has the sensitivity required for §2.3's null to be informative.

**The quantum circuit's single win in this paper is T2b** (Δ NLL −0.0640, Holm *p* = 2.7 × 10⁻⁴), having lost to the same arm by +0.0915 on T1 and +0.2334 on T2a. The entangling ansatz captures generic pairwise interaction; it does not capture conjunctive structure, where it finishes worst of four on both metrics; and it has nothing to capture on the clinical target.

**Accuracy resolves almost none of this**, which is why the claim is made on negative log-likelihood. At *n* = 447 the minimum detectable accuracy difference is 9.3 points (§S7.1); only the quantum arm's deficits on T1 and T2a exceed it. A reader who prefers accuracy as the primary metric should read this control as establishing nothing, and should read §2.3's accuracy nulls the same way.

### S4.8 Field-wide literature audit: study specifications and extended metrics

Following the pre-registered protocol (`docs/FIELD_AUDIT_PROTOCOL.md`), five peer-reviewed studies claiming quantum advantage on biomedical data were audited. Each study was evaluated across 25 outer folds (5 repeats of stratified 5-fold cross-validation, seed 7) with 3-fold inner cross-validation for hyperparameter tuning. The Nadeau–Bengio corrected resampled *t*-test (25 folds, test/train ratio 0.25) evaluated the primary metric difference.

**Table S6d.** Complete evaluation metrics across all 25 outer folds for the five audited studies (`bench/field_audit/results/`). Values are mean ± standard deviation across folds.

| Study | Model Arm | Primary Metric | Balanced Acc. | F1 Score | Sensitivity | Specificity | AUROC | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **CKD** | Quantum VQC | 0.877 ± 0.041 | 0.848 ± 0.056 | 0.908 ± 0.028 | 0.963 ± 0.025 | 0.733 ± 0.119 | 0.971 ± 0.018 | 0.459 ± 0.022 |
| (*PeerJ* 2025) | Classical (Nested) | **0.974 ± 0.019** | **0.974 ± 0.019** | **0.979 ± 0.016** | 0.973 ± 0.024 | **0.975 ± 0.024** | **0.997 ± 0.003** | **0.092 ± 0.070** |
| | Named Classical | 0.969 ± 0.018 | 0.966 ± 0.021 | 0.975 ± 0.015 | 0.975 ± 0.021 | 0.957 ± 0.037 | 0.995 ± 0.005 | 0.086 ± 0.053 |
| **Glioma** | Quantum VQC | 0.856 ± 0.024 | 0.863 ± 0.024 | 0.841 ± 0.025 | 0.903 ± 0.040 | 0.822 ± 0.041 | 0.901 ± 0.025 | 0.539 ± 0.025 |
| (*Sci Rep* 2025) | Classical (Nested) | 0.854 ± 0.029 | 0.860 ± 0.029 | 0.838 ± 0.031 | 0.897 ± 0.046 | 0.823 ± 0.042 | 0.906 ± 0.027 | 0.368 ± 0.054 |
| | Named Classical | 0.849 ± 0.027 | 0.853 ± 0.029 | 0.830 ± 0.031 | 0.883 ± 0.057 | 0.824 ± 0.042 | 0.905 ± 0.025 | 0.371 ± 0.050 |
| **Parkinson's** | Quantum Stack | 0.723 ± 0.077 | 0.621 ± 0.144 | 0.811 ± 0.061 | 0.808 ± 0.115 | 0.433 ± 0.342 | 0.675 ± 0.236 | 0.609 ± 0.214 |
| (*Sci Rep* 2026) | Classical (Nested) | 0.727 ± 0.120 | 0.622 ± 0.175 | 0.810 ± 0.103 | 0.813 ± 0.158 | 0.430 ± 0.397 | 0.768 ± 0.173 | 0.973 ± 0.736 |
| | Named Classical | 0.752 ± 0.080 | 0.587 ± 0.127 | 0.843 ± 0.058 | 0.900 ± 0.101 | 0.273 ± 0.296 | 0.673 ± 0.237 | 0.932 ± 0.804 |
| **Diabetes** | Quantum Stack | 0.748 ± 0.036 | 0.735 ± 0.039 | 0.657 ± 0.051 | 0.693 ± 0.069 | 0.778 ± 0.042 | 0.824 ± 0.036 | 0.525 ± 0.043 |
| (*Sci Rep* 2026) | Classical (Nested) | 0.752 ± 0.028 | 0.751 ± 0.029 | 0.678 ± 0.035 | **0.746 ± 0.051** | 0.756 ± 0.036 | 0.830 ± 0.030 | 0.508 ± 0.035 |
| | Named Classical | 0.752 ± 0.031 | 0.717 ± 0.032 | 0.627 ± 0.043 | 0.599 ± 0.055 | 0.834 ± 0.040 | 0.824 ± 0.039 | 0.487 ± 0.044 |
| **Schizophrenia** | Quantum QSVM | 0.591 ± 0.111 | 0.583 ± 0.109 | 0.644 ± 0.106 | 0.707 ± 0.163 | 0.459 ± 0.160 | 0.641 ± 0.142 | 0.672 ± 0.080 |
| (*J Med Syst* 2024)| Classical (Nested) | 0.667 ± 0.078 | 0.664 ± 0.080 | 0.698 ± 0.073 | 0.724 ± 0.122 | 0.603 ± 0.174 | 0.735 ± 0.095 | 0.881 ± 0.415 |
| | Named Classical | 0.651 ± 0.082 | 0.643 ± 0.086 | 0.703 ± 0.068 | 0.778 ± 0.130 | 0.508 ± 0.205 | 0.700 ± 0.097 | 0.648 ± 0.050 |

---

## S5. Positive controls

These establish that the apparatus functions. No novelty is claimed for any component.

### S5.1 Chest finding extraction

**Table S7.** Per-finding performance, DenseNet-121, *n* = 2,099 held-out images. PPV and F1 at the served operating point.

| Finding | AUROC | AUPRC | AUPRC baseline | Sensitivity | Specificity | PPV | NPV | F1 | Positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hyperinflation | 0.912 | 0.430 | 0.057 | 0.808 | 0.865 | 0.267 | 0.987 | 0.401 | 120 |
| Pleural effusion | 0.900 | 0.703 | 0.250 | 0.884 | 0.769 | 0.560 | 0.952 | 0.685 | 524 |
| Cardiomegaly | 0.862 | 0.476 | 0.135 | 0.852 | 0.710 | 0.315 | 0.968 | 0.460 | 284 |
| Pneumothorax | 0.825 | 0.114 | 0.030 | 0.460 | 0.866 | 0.096 | 0.981 | 0.159 | 63 |
| Consolidation | 0.807 | 0.200 | 0.057 | 0.808 | 0.660 | 0.126 | 0.983 | 0.218 | 120 |
| Nodule | 0.729 | 0.165 | 0.045 | 0.358 | 0.874 | 0.119 | 0.966 | 0.178 | 95 |
| Opacity | 0.712 | 0.416 | 0.236 | 0.677 | 0.635 | 0.365 | 0.864 | 0.474 | 496 |
| **Macro** | **0.821** | **0.358** | 0.116 | **0.693** | **0.768** | **0.264** | **0.957** | **0.368** | — |

**Macro PPV is 0.264 and macro F1 0.368** — substantially less favourable than the discrimination figures and the clinically decisive quantities at these prevalences. Pneumothorax has PPV 0.096 at the served operating point; roughly nine of ten positive calls are false. High macro NPV (0.957) makes the system more defensible as a rule-out than a rule-in aid, and its deployment framing reflects that. Earlier versions of this table reported discrimination and operating-point sensitivity and specificity without PPV; that omission is a specific instance of the metric-selection practice this paper documents in others.

Per-finding Platt scaling reduced mean ECE from 0.181 to 0.026. Throughput 252 images/s; latency 29.1 ms GPU, 83.2 ms CPU. A third training run reached macro AUROC 0.8365 against 0.8372 for its predecessor — no change, recorded because it is decision-relevant for anyone considering the same computation.

### S5.2 Label provenance and circularity

Chest findings are derived from reports by rule, raising the question of whether the reported AUROC measures anything beyond the model reproducing its own label noise. Two independent axes address this. **Scope: this validates the seven finding labels. It does not validate the six-way category map of §S2.**

*Axis 1 — the labeller.* Sixty-six MIMIC-CXR reports read by hand (45 uniformly at random, seed 20260722; 21 keyword-screened for rare findings, stratum recorded). Macro Cohen's κ 0.861, macro F1 0.894, micro cell accuracy 0.957.

Stated ceiling and stated defect: a single annotator applying CheXpert conventions, not a board-certified radiologist; no second-reader statistic on the reference standard; no image review; rare-finding cells contain six or fewer positives. **The pooled κ of 0.861 mixes a uniform-random stratum with a keyword-enriched stratum without reweighting**, so it is not an unbiased corpus-level estimate; the random stratum alone is appropriate and is under-powered for rare findings. Radiologist over-reading with a majority-vote design is the appropriate next step, and until performed, 0.861 should not be treated as adjudicated ground truth.

*Axis 2 — an independently trained image model.* 700 images scored by both AURA and public `torchxrayvision densenet121-res224-mimic_ch`, prevalence-matched.

**Table S8.** Cross-model agreement, *n* = 700.

| Finding | Spearman ρ | AUROC vs. independent label | κ | Interpretation |
|---|---:|---:|---:|---|
| Pleural effusion | 0.745 | 0.870 | 0.513 | Moderate |
| Consolidation | 0.738 | 0.898 | 0.315 | Fair |
| Opacity | 0.679 | 0.806 | 0.355 | Fair |
| Cardiomegaly | 0.661 | 0.763 | 0.126 | Slight |
| Pneumothorax | 0.535 | 0.879 | 0.060 | Slight |
| Nodule | 0.097 | 0.439 | −0.031 | **No agreement** |

Six of seven corroborate. Nodule does not, on any axis: rank correlation near zero, AUROC below chance. Disclosed on the model card, in the interface, and in the generated report.

Running both axes **localises** the failure. Nodule is the labeller's strongest finding (κ 0.933, axis 1) and the image model's weakest (κ −0.031, axis 2). Text extraction is adequate; image discrimination is not. A single-axis validation would have attributed the failure to the wrong component. The consequence propagates: the malignancy category is defined by the nodule concept, so the diagnosis-level malignancy failure in §2.3 *is* this image-model failure, and no choice of fusion backend could repair it.

Independence is bounded: the comparison model was also trained on MIMIC-CXR and patient-level overlap with our evaluation images was not excluded.

### S5.3 Brain segmentation, and metric comparability

On 7,531 held-out BraTS-2020 slices with volume-disjoint splitting: pooled 2-D Dice 0.915 (whole tumour), 0.846 (tumour core), 0.835 (enhancing tumour), composite mean 0.865; 95th-percentile Hausdorff 7.08, 6.15, 4.53 px. Presence head AUROC 0.987; Platt calibration reduced ECE from 0.095 to 0.018.

**These figures must not be compared with BraTS leaderboard values.** Ours are pooled over 2-D axial slices; the literature reports per-case 3-D Dice averaged over volumes. Pooled-2-D scoring is systematically more generous — slices with no tumour are easy and numerous — and never penalises through-plane inconsistency. Treat ours as an internal regression metric, not a leaderboard position. A comparable number requires per-case 3-D evaluation on the official validation set (§S8, item 8).

`docs/benchmark_report.md` contains nnU-Net, SwinUNETR and MONAI rows that are **published literature values, not measurements on these data**. They are labelled as such in the table, in a warning above it, and by two build tests; they were once bare literals under prose describing a head-to-head that never ran (defect D2, §S7.3).

### S5.4 Circuit execution, compilation, and noise

One fusion circuit was executed on IBM `ibm_marrakesh` (156 qubits, job `d9js49rjf64c739haeg0`); the top-1 diagnosis survived device noise. Raising the transpiler optimisation level from 1 to 3 removed 20% of two-qubit gates at no cost, and the served path now does so.

A three-rung noise attribution on one evidence vector gave mean absolute error in ⟨Z⟩ of 0.0074 (sampling), 0.0544 (device model) and 0.1858 (hardware), attributing 88.1% of simulated error to decoherence and 11.9% to sampling — identifying circuit depth rather than shot count as the binding constraint.

**Both rest on a single case.** They are an existence proof and a descriptive share respectively, and neither carries an interval. `quantum_study.json` records `"hardware": "simulated; no quantum hardware was used"` in its honesty block — true when written, contradicted once this run existed. The field has been corrected in place to state that the results *in that artifact* are simulated while naming the hardware run (defect D4, §S7.3).

---

## S6. Selective prediction, and a withdrawn claim

AURA abstains when any of four criteria holds, evaluated in this order by `SafetyEngine.assess()`: OOD energy *z* > 1.5; top calibrated probability < 0.30; conformal set size > 4; epistemic uncertainty > 0.20.

This project's remaining positive claim was that the four-criterion policy is better than a confidence threshold. A risk–coverage analysis on the primary 447-study split under P1 **withdraws it**: the policy is not distinguishable from thresholding the maximum calibrated probability at matched coverage, in any of the four backends (`risk_coverage_n447.json`). Selective prediction is therefore reported here as a positive control, not as a contribution.

**A published safety envelope that was wrong.** Through four revisions, this project's documentation, this manuscript's predecessors, and the build test that enforces P6 all published the abstention thresholds as epistemic 0.15 and OOD *z* 2.5. Those are the **CDRE controller's** gate-check thresholds (`SafetyPolicyThresholds`), read by a different subsystem. The abstention path reads `Settings.ood_energy_threshold` = 1.5, and its epistemic criterion was a literal `0.20` hardcoded in the engine and unreachable from configuration. The build test compared the document against settings the serving path never read, so it passed while the published envelope was wrong on two of its four criteria.

This is the failure mode of P7 occurring inside the test that enforces P6: **a test that verifies a metric rather than a mechanism**. It is now fixed at the mechanism. The epistemic threshold is a named setting; the abstention branch reads all four criteria from configuration; and a second test parses the source of that branch and fails if any criterion is a numeric literal or is not read from `self.settings` (defect D10, §S7.3).

---

## S7. Threats to validity, limitations, and a defect register

### S7.1 Threats to validity

**Construct validity.** Quantum advantage is operationalised as per-metric superiority under matched capacity, per-arm calibration and a matched optimiser. A reader preferring a different construct — expressivity, sample efficiency, robustness under shift — would require different experiments; none is tested. Parameter-count matching is a weak operationalisation of capacity for quantum models, and effective-dimension matching was not performed.

**The prediction target.** The six-way target is a rule-derived radiographic-pattern proxy whose derivation map has not been validated against clinician judgement, and whose Bayes-optimal rule is linear (§S2). Results in §2.3 to §2.6 are scoped to targets of this kind. §2.4 establishes that the *comparison* detects nonlinear advantage where a target contains it, which addresses circularity but does not extend the clinical result to structured targets.

**Internal validity.** The ablation in §2.5 is replicated across ten seeds with data held fixed, so its effect is not an initialisation artifact. Its seed intervals condition on a single test split; the patient bootstrap and five patient-disjoint re-splits added in revision (§2.5(b)) address this, and over patients the effect is marginal. The sweep in §2.6 remains single-seed and single-split with its top-of-ranking selected from 48 cells.

**Statistical power and test-set reuse.** The primary 447-study comparison has a minimum detectable accuracy difference of 9.3 points at 80% power; accuracy nulls there are statements about undetectability, not equality. §2.8 converts this from a hedge into a measured fact: at *n* = 4,800 (MDE ≈ 3.6 points) five of six pairwise accuracy comparisons are significant and the true differences are 4 to 6 points — inside the internal experiment's blind spot by construction. **The sweep in §2.6 has not been re-run at adequate power and has no external check**, so it remains an undetectability statement. The §2.5 ablation was repeated on out-of-sample CheXpert evidence, where the penalty was not detected (§2.5(d)). Separately, the same 447-study split is used for §2.2 to §2.6 and §2.9 across more than sixty model evaluations including a 48-cell sweep from which the top five were selected. No held-out split remains to bound adaptive overfitting, and its magnitude is not estimated.

**Comparator and hyperparameter symmetry.** §2.3's matched protocol removes the optimiser confound that Table S2's published block carries, but leaves a residual gap of 0.0052 NLL on the anchor arm, of the same order as the one surviving effect. §2.7's kernel comparison gives both arms grids of identical shape, which the superseded comparison did not.

**A constant evidence channel.** `prior_risk` is identically zero in all 2,231 MIMIC and all 8,000 CheXpert studies. §2.5's controlled re-run resolves this for the ablation. **The 48-cell sweep (§2.6) and the 20-epoch robustness check were run at eight qubits and have not been repeated at seven**, so their magnitudes are not controlled estimates.

**Single-instance results.** The hardware execution and the noise attribution each rest on one case and carry no interval.

**Generalisation.** Development used MIMIC-CXR, a single United States tertiary academic centre; BraTS-2020 comprises preoperative glioma only. §2.8 supplies external validity **across sites and only for §2.3**, and its result is a caution rather than a reassurance: every arm degraded, higher-capacity arms degraded more, and one internal ordering reversed. It is not external validity across populations, label regimes or care settings — both corpora are large US academic centres labelling by rule from free text. There is no prospective cohort, no reader study, and no demographic subgroup analysis despite sex, age and race being available in the MIMIC-CXR metadata. The brain model is not validated for postoperative monitoring, radiation-necrosis differentiation, or non-glial tumours, and no external brain cohort was used. The modality router cannot distinguish axial head CT from axial brain MRI on pixel data alone once DICOM metadata is removed.

### S7.2 A note on an error caught before publication

The first execution of the full-coverage kernel comparison (§2.7) produced subject-level AUROC of 0.228 for the quantum arm and 0.225 for the RBF arm — both far below chance, and nearly equal. The cause was a sign convention: `fit_platt` returns coefficients for `p = 1/(1 + exp(a·f + b))`, Platt's original parameterisation, and the new script applied the more familiar `σ(a·f + b)`, inverting every probability. Probabilities stayed in [0, 1], calibration looked plausible, and only the AUROC betrayed it.

We record this because it is the same class of error as the defects below, caught one step earlier, and because the fix generalises: the script now derives both arms' probabilities from one helper, and refuses to write an artifact when both arms score below chance — an outcome that is an orientation bug rather than a result. An inverted run produces numbers that look publishable.

### S7.3 Defect register

Applying P6 to our own repository identifies fourteen defects. We report them rather than correct them silently. **Closed at this revision: D1, D2, D3, D4, D6, D8, D10, D12, D13, D14** — in addition to D5, D7, D9 and D11, closed previously. Three of the fourteen (D12, D13, D14) were found while preparing this revision, and D14 is the most serious defect this project has identified in itself.

| | Defect | Status |
|---|---|---|
| D1 | README and grading documentation quote the superseded kernel result ΔAUROC = −0.016, *p* = 0.002 from the 669-slice run, superseded twice since | **Closed** — updated to the §2.7 full-coverage result |
| D2 | `docs/benchmark_report.md` published nnU-Net / SwinUNETR / MONAI rows as if measured on these data | **Closed** — rows attributed to their publications, marked "not measured", warned above the table, and held by two build tests |
| D3 | `quantum_study.json` records the retracted measurement-budget interpretation ("No classical fusion model can make this distinction") | **Closed** — annotated in place with the retraction; original text retained as `note_withdrawn` for provenance |
| D4 | The same artifact's honesty block records "simulated; no quantum hardware was used", contradicted by §S5.4 | **Closed** — corrected to scope the statement to that artifact and name the hardware run |
| D5 | `fusion_headline_n447.json` recorded a Wilcoxon *p* as `0.0` rather than at full precision | Closed previously — regenerated as 7.296×10⁻⁶ |
| D6 | `fusion_ablation.json` records neither sample size nor split identifier; `fusion_headline_n447.json` did not record the training length of the models it evaluates | **Closed** — ablation artifact marked superseded with its unknowns stated; training lengths recorded with their provenance |
| D7 | `brain/qkl_training_report.json` reported tuned results without the hyperparameter grid | Closed previously |
| D8 | Brain embedding dumps inconsistent in coverage: `epoch_001.npz` holds 25,268 slices from 185 subjects, later checkpoints 7,407 from 55, with nothing recording the discrepancy | **Closed** — the discrepancy is documented, and §2.7 now uses the full-coverage dump at the same checkpoint, which removes the power loss it caused |
| D9 | Table 2 published the gated head at 54 trainable parameters through four revisions; it has 126 | Closed previously — all four counts now recorded in the artifact |
| D10 | The published abstention envelope was wrong on two of four criteria, and the build test enforcing it compared documentation against settings the serving path never read | **Closed** — see §S6; fixed at the mechanism, with a source-level test |
| D11 | `murphy_decomp.py` was reported as a completed experiment that had changed a conclusion; it had never been executed, and the drift test skipped silently when the artifact was absent | Closed previously — executed, reported, and the test now fails on a missing artifact |
| D12 | **New.** The abstract and §8.4 of previous versions state that four of the five best sweep configurations by negative log-likelihood lie at zero entanglement. The artifact gives **three of five** by that metric; four of five holds only under an accuracy ranking with an ill-defined cut-off | **Closed** — corrected in §2.6 and §S4.4; the corrected statement is weaker |
| D13 | **New.** DeLong's test, which §5.3 of previous versions named as the method for AUROC comparisons, existed only inside `audit_all.py` at the repository root and in no packaged evaluation module. Consolidating it revealed the risk directly: two implementations of one statistic can drift | **Closed** — single implementation in `aura/ml/evaluation/delong.py`, validated against `sklearn.metrics.roc_auc_score`, with `audit_all.py` delegating to it and a test forbidding a second copy |
| D14 | **New, and the most serious.** `audit_all.py` — the harness this paper's Code Availability section directs a reviewer to run — drew its held-out split from the **synthetic** evidence world. That was correct while the fusion heads were themselves fitted on synthetic evidence; audit finding F1 then refitted them on real MIMIC-CXR evidence (2026-07-21) and this file was never updated. Every run from that date evaluated real-trained models on out-of-distribution vectors and reported **below-chance accuracy — 0.010 against a 1/6 floor** — while still printing `CLAIM [VALIDATED]` for one of the three claims it checks. The failure is silent because both data paths return well-formed arrays | **Closed** — the harness now reads the same cached real-evidence split the manuscript's experiments use, records the data world in its manifest, falls back to synthetic only with a loud warning, and **aborts** rather than reporting metrics when a served head scores below chance. It now reproduces Table 1's published block exactly (quantum accuracy 0.5324, classical 0.5414, ECE 0.0896 / 0.1266, temperatures 0.9057 / 0.4790) |

**Consequence of D14 for §2.2.** The ECE figures of 0.020 and 0.276 that §2.2 audits were produced by this harness on the synthetic split, before the F1 refit. §2.2 now reports the real-data replication alongside them: 0.0896 quantum, 0.1266 classical-fair, 0.2289 classical-raw, with the confound accounting for 0.139 of a 0.176 difference. The diagnosis is unchanged and the magnitude is smaller. `audit_artifacts/run_20260719T175647Z` is retained as the record of the original audit and is labelled with its data world; `run_20260821T173709Z` is the real-data replication.

The drift test invoked by P6 covers this manuscript's Tables 1, 2, S1, S2, S3, S4 and S6 against their artifacts. It did not do so when D5 and D6 reached publication, and it does not cover `audit_all.py`'s outputs at all — which is why D14 survived four months and thirteen prior defect entries.

---

## S8. Analyses identified as necessary and not performed

Listed explicitly rather than as an omission, in order of the strength they would add.

1. **External validation of §2.6 and §2.7.** §2.8 retires this for §2.3, and the out-of-sample CheXpert analysis (§2.5(d)) for the entanglement ablation. The design sweep has not been repeated on CheXpert-derived evidence, and no external brain cohort has been used for the kernel comparison.
2. **Seven-qubit repetition of the 48-cell design sweep** and of the 20-epoch robustness check, both of which still carry the constant-channel confound.
3. **Re-fitting the §2.3 arms under a common budget at convergence**, closing the residual 0.0052 NLL gap on the anchor arm, so that the one surviving effect can be attributed to model class rather than to residual optimisation.
4. **Nested bootstrap** over seeds and studies jointly for the re-uploading ablation and the out-of-sample replication. (For the seven-channel ablation, the patient bootstrap including initialisation variance is reported in §2.5(b).)
5. **Noise attribution over ≥100 evidence vectors** with a squared-error decomposition, and hardware execution over ≥50 cases with a top-1 agreement interval, to give §S5.4 a sample size.
6. **Subgroup analysis** on sex, age band and race for the vision model and the fusion backends.
7. **Per-case three-dimensional segmentation re-scoring**, to make a comparison with BraTS values valid.
8. **Radiologist over-read** of the 66-report reference standard with a majority-vote design, and a stratum-weighted κ estimate.
9. **Extension of the drift test** to the abstract's figures and to prose constants.
10. **Execution of the shot-allocation loop on hardware**, and the margin-versus-shots rank correlation, both bearing on §2.9.
11. **Extraction of reported advantage margins** from the studies synthesised in [1], to support or retire the field-level comparison.
12. **A rejector ablation** for §S6, isolating each of the four abstention criteria, now that the policy as a whole is known not to beat its control.
13. **A nonlinear-target replication on a second corpus**, so that §2.4's positive control is not itself single-split.

---

## S9. Map from the full-length preprint (v4.3) to this submission

| v4.3 section | Location here |
|---|---|
| §1 Introduction, §2 Related work, §3 Research gap | Main §1 (condensed); full related-work treatment and the G1–G9 gap analysis in the v4.3 preprint |
| §4 System | §S1 |
| §5.1 Data, §5.1.2 target, §5.1.3 constant channel | Main Methods; §S2 |
| §5.2 Environment, §5.3 Statistics | Main Methods |
| §6 Protocol | §S3 (with P8 added) |
| §7 Size of the confound | Main §2.2; §S4.1 |
| §8.1 Calibration artifact | Main §2.2 |
| §8.2 Comparator choice | Main §2.3; §S4.2 — **extended with the matched-optimiser experiment** |
| §8.3, §8.3.1, §8.3.2 Entanglement | Main §2.5; §S4.3 |
| §8.4 Design sweep | Main §2.6; §S4.4 — **corrected (D12)** |
| §8.5 Quantum kernel | Main §2.7; §S4.5 — **superseded by the full-coverage nested-CV comparison** |
| §8.6 External validation | Main §2.8; §S4.6 |
| §9.1 Selective prediction | §S6 |
| §9.2–§9.5 Positive controls | §S5 |
| §10 Discussion, §10.6 Retraction | Main §3, §2.9 |
| §11 Threats and limitations, §11.7 defects | §S7 |
| §11.9 Analyses not performed | §S8 |
| — | Main §2.4 — **new: nonlinear positive control** |

### S9.1 Claim-to-artifact map

**Table S9.** Every reported claim and the artifact it is read from. Paths are relative to the repository root. Regenerate the benchmark with `py -m aura.aura_cli bench` and the significance tests with `python audit_all.py`.

| Section | Claim | Artifact |
|---|---|---|
| Methods, §S2 | Prediction-target derivation | `aura/mimic/labeling.py`, `_map_diagnosis` |
| Methods, §2.5 | Constant `prior_risk` channel | `aura/artifacts/quantum_study_evidence.npz` |
| §2.2, §S4.1 | Calibration confound and the 1.89 temperature spread | `aura/artifacts/backend_calibration.json` |
| §2.2 | Audit of previously published fusion claims | `audit_artifacts/run_20260719T175647Z/metrics/claim_verdicts.json` |
| §2.3 upper, §S4.2 | Published fusion comparison, *n* = 447 | `aura/artifacts/fusion_headline_n447.json`, `fusion_baselines_n447.json` |
| §2.3 lower, §S4.2 | **Matched optimiser comparison** | `aura/artifacts/fusion_matched_optimiser.json` |
| §2.3, §2.5(b), §2.5(c) | Equivalence test (A), patient bootstrap (B), re-splits (C) and re-uploading ablation (D), added in revision | `aura/artifacts/revision_v51.json` |
| §2.3, §S1 | Trainable parameter counts of every arm | `aura/artifacts/fusion_{classical,quantum,learnable}.npz` |
| §2.4, Fig. 1B | **Nonlinear positive control** | `aura/artifacts/nonlinear_control.json` |
| §2.5, §S4.3 | Entanglement ablation, 10 seeds, 8 qubits (confounded) | `aura/artifacts/entanglement_multiseed.json` |
| §2.5, §S4.3 | Entanglement ablation, 10 seeds, 7 channels (controlled) | `aura/artifacts/entanglement_multiseed_7q.json` |
| §S4.3 | Murphy decomposition (non-identifiable) | `aura/artifacts/murphy_decomposition.json` |
| §2.6, §S4.4 | 48-cell design sweep | `aura/artifacts/design_sweep.json` |
| §2.7, §S4.5 | **Quantum kernel, official 369 subjects, nested CV** | `aura/artifacts/brain/qkl_brats2020_official.json`, `grade_mapping_check.json` |
| §2.7 | Pre-registered sensitivity analysis, 111 held-out subjects | `aura/artifacts/brain/qkl_brats2020_official_heldout111.json` |
| §2.1, §S4.8 | **Field-wide literature audit, 5 medical QML studies** | `bench/field_audit/results/`, `docs/FIELD_AUDIT_PROTOCOL.md` |
| §S4.5 | Superseded kernel comparison, retained for provenance | `aura/artifacts/brain/qkl_training_report.json` |
| §2.8, §S4.6 | External validation on CheXpert | `aura/artifacts/chexpert_external.json` |
| §2.8 | CheXpert evidence tensor, 8,000 studies | `aura/artifacts/chexpert_evidence.npz` |
| §2.8, §2.5(d) | Out-of-sample replication within CheXpert evidence | `aura/artifacts/chexpert_internal.json`, `docs/PREREGISTRATION_ADDENDUM_2026-10b_CHEXPERT_INTERNAL.md` |
| §2.8, §S3 | Pre-registration, deposited before the run | `docs/PREREGISTRATION_CHEXPERT.md` (commit `fa8c2b6`) |
| §2.9 | Measurement-budget statistics, with the retraction annotated | `aura/artifacts/quantum_study.json`, key `q3_measurement_budget` |
| §2.9 | The estimation step under audit | `aura/services/fusion/qmba.py`, lines 377–387 |
| §S5.1 | Per-finding chest metrics | `aura/artifacts/evaluation_retrain_v2/metrics.json` |
| §S5.1 | Per-finding Platt calibration | `aura/artifacts/vision_serving_calibration.json` |
| §S5.2 | Labeller reference standard, 66 reports | `aura/artifacts/labeler_validation.json` |
| §S5.2 | Cross-model agreement, 700 images | `aura/artifacts/kappa_crosscheck.json` |
| §S5.3 | Brain segmentation | `aura/artifacts/brain/reports/test_report.json` |
| §S5.3 | Presence calibration and sequence ablation | `aura/artifacts/brain/presence_calibration.json` |
| §S5.4 | Hardware execution | `aura/artifacts/ibm_hardware_run.json` |
| §S5.4 | Transpilation study | `aura/artifacts/transpile_study.json` |
| §S5.4 | Noise attribution | `aura/artifacts/noise_rung.json` |
| §S6 | Risk–coverage and the matched-coverage control | `aura/artifacts/risk_coverage_n447.json` |
| §S6 | Superseded abstention table, provenance only | `aura/artifacts/fusion_ablation.json` |
| Methods | Environment and artifact checksums | `audit_artifacts/run_20260719T175647Z/environment.json` |

---

## S10. Full reference list

References 1–18 are as numbered in the main text. The full list of 82 works cited by the full-length preprint — covering the qualifying QML literature, barren plateaus and trainability, dequantization, calibration and conformal prediction, selective prediction and learning-to-defer, imaging baselines and label provenance, and the proper-scoring-rule decompositions — is retained in `docs/RESEARCH_PAPER_v4.md` §References and is not reproduced here.

Additional works cited in this Supplement and not in the main text:

19. Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: *Advances in Large Margin Classifiers*. MIT Press; 1999:61–74.
20. Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*. 2014;21(11):1389–1393.
21. Murphy AH. A new vector partition of the probability score. *Journal of Applied Meteorology*. 1973;12(4):595–600.
22. Geifman Y, Uziel G, El-Yaniv R. Bias-reduced uncertainty estimation for deep neural classifiers. In: *ICLR*. 2019.
23. Cohen JP, Viviano JD, Bertin P, et al. TorchXRayVision: a library of chest X-ray datasets and models. In: *MIDL*. 2022.
