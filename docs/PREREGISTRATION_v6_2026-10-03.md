# Pre-registration v6 — confirmatory test, symmetric tuning, mechanism, and audit-factor separation

**Deposited:** 3 October 2026. Committed before any analysis below was run and before any
image of the confirmatory cohort was scored.
**Executes in:** `aura/ml/evaluation/v6_confirmatory.py`, `aura/ml/evaluation/v6_mechanism.py`,
`bench/field_audit/factors.py`, and `python -m bench.field_audit.studies heart_qnn`.
Results go to `aura/artifacts/v6/` and `bench/field_audit/results/`.

This addendum follows the earlier ones (`PREREGISTRATION_CHEXPERT.md`,
`PREREGISTRATION_ADDENDUM_2026-10.md`, `PREREGISTRATION_ADDENDUM_2026-10b_CHEXPERT_INTERNAL.md`).
Nothing in them is changed. Every result below is reported whatever it shows. If a result
changes a conclusion, the manuscript changes.

## 0. What is known when this is written

Known: every result in manuscript v5.1, including the CheXpert-internal run
(`chexpert_internal.json`: quantum − PoE ΔNLL +0.0076 on 1,600 test studies, 90% CI −0.001 to
+0.016, equivalence not established). Unknown: no image of any CheXpert patient beyond the
first 8,000 (selection order below) has been scored by any model. None of the analyses in
§§2–6 has been run, and none of their code has been written.

## 1. The manuscript's primary claim, fixed before the confirmatory test

**Primary claim:** under symmetric evaluation, no quantum component examined (in AURA or in
the audited studies) improves on the best classical comparator. Analysis C1 is the confirmatory
test of the central instance of that claim, which is the fusion circuit against the
product-of-experts on likelihood. Every other analysis in this document is secondary or
explanatory and is labelled that way in the manuscript.

## 2. C1 — confirmatory test on untouched patients (primary)

**Cohort.** CheXpert-v1.0-small `train.csv`, using the selection rule of
`chexpert_external.select_studies`: frontal views only; patients in order of first appearance in
the file; one study per patient (the highest-numbered study with a frontal view). The first
8,000 patients in that order form the existing evidence set (`chexpert_evidence.npz`) and are
the **development set**. **Every** later patient forms the **confirmatory test set**. There is
no sampling and no choice of n. The extractor (DenseNet-121 with per-finding Platt) has never
seen a CheXpert image, so the test evidence is out-of-sample for the extractor and for every
fusion head.

**Labels.** As pre-registered for CheXpert: the weighted-score map with acuity tie-break, with
uncertain treated as negative (`y_u0`) as the primary policy and uncertain treated as positive
(`y_u1`) as a sensitivity analysis. Five reachable categories; COPD is unreachable.

**Fitting.** Development set split once with `numpy.random.default_rng(7)`: 80% train
(6,400), 20% calibration (1,600). The four arms are fitted under the published
matched-optimiser protocol (`matched_optimiser.py`: Adam, LR grid {0.01, 0.05, 0.1} selected per
arm on calibration cross-entropy, batch 50, ≤150 epochs, patience 20, seed 7, weight decay 0,
one temperature per arm fitted on calibration; L-BFGS logistic anchor). The arms are then
frozen and scored once on the test set.

**Simulator.** For speed the circuit may be evaluated by a batched torch statevector
implementation of the identical circuit (RY(πx) encoding; per layer RY, RZ per qubit then CNOT
ring; ⟨Z⟩ readout). It is used only if, before any training, it matches the PennyLane qnode
(`make_qnode`) to max |Δ| < 1e-8 on 1,000 random (x, θ) pairs at 7 and 8 qubits, for the
ring and no-entangler variants. Otherwise PennyLane is used. The check is saved with the
results.

**Primary endpoint.** Paired per-study difference d = NLL_quantum − NLL_PoE on the test set
(positive means the quantum arm is worse). Equivalence margin ±0.010 nats, the margin of the
earlier addenda. Inference: the TOST on the paired mean (`revision_v51.tost`, α = 0.05,
reporting the 90% CI) and the two-sided 95% t-interval. With roughly 55,000 test studies and
the CheXpert-internal per-study SD (about 0.21), the 90% half-width is expected to be about
0.0015 nats.

**Reading (fixed now):**

| Result | Reported as |
|---|---|
| 90% CI inside ±0.010 and 95% CI contains 0 | equivalent within ±0.01 nats |
| 90% CI inside ±0.010 and 95% CI excludes 0 | a detectable difference in the stated direction, smaller than the margin; practically equivalent |
| 95% CI entirely above +0.010 | quantum materially worse |
| 95% CI excludes 0 and crosses +0.010 | quantum worse; size relative to margin unresolved |
| 95% CI entirely below −0.010, or below 0 and crossing −0.010 | **a quantum benefit survived**; the primary claim of §1 is withdrawn for this instance and the manuscript is rewritten accordingly |
| none of the above | inconclusive |

**Secondary (C1).** All six pairwise accuracy (McNemar) and NLL (Wilcoxon) contrasts with Holm
correction; ECE and macro-AUROC per arm, with 2,000-resample bootstrap CIs; the same endpoint
under `y_u1`.

**C1b — entanglement ablation on untouched patients (secondary).** The addendum (a) analysis-B
protocol unchanged (7 informative channels, 3 layers, `train_quantum` Adam lr 0.05, batch 50,
30 epochs; seeds {7, 11, 13, 17, 19, 23, 29, 31, 37, 41}; ring vs no entangler; per-arm
temperature on calibration), trained on the development train split and scored on the test set.
Primary statistic: the patient bootstrap of the seed-averaged per-patient ΔNLL (10,000
resamples). Reading as in addendum (b): CI above 0 means the penalty replicates on untouched
patients; CI containing 0 means not replicated; CI below 0 means a reversal. Equivalence within
±0.01 by TOST is also reported.

## 3. C2 — symmetric hyperparameter search beyond the learning rate (secondary)

Addresses the objection that the quantum arm was tuned only on learning rate.

Each arm receives **the same budget: 36 configurations**, drawn uniformly without replacement
(`numpy.random.default_rng(7)`) from its own declared space. Each configuration is trained under
the shared trainer of `matched_optimiser.py` (Adam, batch 50, ≤150 epochs, patience 20, seed 7).
The configuration with the lowest calibration cross-entropy is selected, temperature-scaled on
calibration, and scored once on the C1 test set. Data and splits are those of C1.

| Arm | Space |
|---|---|
| Quantum (CNOT ring, 8 qubits) | layers {1, 2, 3, 4, 6} × encoding scale s ∈ {0.5, 1, 2} (RY(s·π·x)) × θ init SD {0.01, 0.1, 1.0} × LR {0.01, 0.05, 0.1}: 135 points |
| Product-of-experts (linear) | LR {0.003, 0.01, 0.05, 0.1, 0.3} × weight decay {0, 1e-5, 1e-4, 1e-3} × init SD {0.01, 0.1, 1.0}: 60 points |
| Gated head | the same 60-point space as the linear arm |

The quantum space includes capacity (depth) and the classical spaces do not. This
deliberately favours the quantum arm, so a null under C2 is conservative. Parameter counts of
the selected configurations are reported. If the quantum selection lies on an edge of its space
(layers = 6, s ∈ {0.5, 2}, or init SD 1.0), that is reported as "search boundary reached".

**Endpoint and reading:** d = NLL_quantum(C2-selected) − NLL_PoE(C2-selected), with the reading
table of §2.

## 4. M — why: mechanism measurements (explanatory, no hypothesis tests)

Pre-specified quantities. No result here changes the primary claim. They are reported as an
explanation of the observed nulls.

**M1 — interaction-order spectrum of the fusion circuit.** With one RY(πxᵢ) encoding per
qubit, ρ(x) = ⊗ᵢ (I + cos(πxᵢ)Z + sin(πxᵢ)X)/2. So each readout ⟨Z_j⟩ = Σ_P c_P ∏ᵢ f_{Pᵢ}(xᵢ)
over P ∈ {I, X, Z}^n, with c_P = Tr(U†Z_jU · P)/2ⁿ computed exactly. The interaction order
of P is its number of non-identity factors. For the trained circuits of C1 (published 8-qubit
ring, and for C1b's 7-qubit ring and product arms at every seed) we report: (i) the share of
Σc_P² at each order; (ii) the share of the test-set variance of each logit carried by terms of
order ≥ 2. Sanity check fixed in advance: the product circuit must have zero weight above
order 1.

**M2 — matched classical surrogate.** L2 logistic regression on the same trigonometric basis
truncated at order ≤ 1 (2n features) and order ≤ 2. C is selected from {0.01, 0.1, 1, 10, 100}
on calibration NLL, with temperature scaling, and scored on the C1 test set against the C1
quantum arm. Reported descriptively (ΔNLL with 95% CI). This tests whether the circuit's
function class, fitted classically, is matched or exceeded.

**M3 — kernel diagnostics.** For (a) the fusion encoding's fidelity kernel on the C1
development evidence (a 2,000-study subsample, seed 7), and (b) the served 10-qubit glioma QKL
feature map on the 111 held-out BraTS subjects' embeddings, at the bandwidths selected in the
published runs, we report:

- the mean and variance of off-diagonal kernel values (concentration);
- the effective dimension (Σλ)²/Σλ² of the trace-normalised kernel;
- the kernel–target alignment;
- the geometric difference g(K_C‖K_Q) = √‖√K_Q K_C⁻¹ √K_Q‖_∞ (Huang et al., 2021), with
  K_C the RBF kernel at the selected γ, trace-normalised, regulariser 1e-3·I.

For (a) we also check numerically that the fidelity kernel equals the closed form
∏ᵢ cos²(π(xᵢ − x′ᵢ)/2), max |Δ| < 1e-10.

## 5. F — separating confounded audit factors (secondary)

**F1 — Parkinson's (PMID 42286086, specificity).** 2 × 2 × 2 factorial:

- PCA scope: fitted on the full data, or inside training folds.
- CV unit: stratified by recording, or stratified-grouped by subject (32 subjects).
- Comparator: untuned RBF-SVM (sklearn defaults), or the nested-selected classical arm of
  `protocol.classical_arm`.

The quantum model is the paper-configured `KernelStack` in every cell, untuned. The outer
scheme is 10-fold, repeated with seeds 0–4 (50 folds). Per cell we report
Δ = specificity(stack) − specificity(comparator), mean over folds, with a Nadeau–Bengio
corrected t CI.

**F2 — Pima (same paper, sensitivity).** 2 × 2: PCA scope × comparator. No grouping exists.

**Reading:** the advantage "reproduces" in a cell if Δ > 0 with corrected p < 0.05. We report:

- the main effect of each factor (mean Δ across cells at one level minus the other);
- which single correction, applied to the paper cell (global PCA, recording-level, untuned),
  removes the advantage (p ≥ 0.05, or Δ ≤ 0).

If no single correction removes it, we report the smallest combination that does.

**F3 — EEG schizophrenia (PMID 38441727).** The raw recordings (16 channels × 7,680 samples
at 128 Hz) are split into non-overlapping epochs: 5 s primary, giving 12 per subject; 1 s and
2 s are sensitivity analyses. Features are computed as in `datasets.eeg_schizophrenia` on Cz.
The models are those of the reproduction arm: QSVM Pauli-Z with k = 5 and k = 15, LR, RF, and
SVM.

Two split schemes, 200 seeds each, 70/30:

- (a) random at epoch level, so a subject's epochs cross the split;
- (b) grouped by subject.

We report mean accuracy and the share of splits reaching 100%.

**Reading:** if under (a) any model's mean accuracy is ≥ 0.95 while under (b) it is ≤ 0.75,
we report that epoch-level splitting is sufficient to produce near-perfect accuracy on these
data. This does not show that the published study did it.

## 6. H — finish the heart-disease QNN (PMID 41513819)

Run `python -m bench.field_audit.studies heart_qnn` unchanged, under
`docs/FIELD_AUDIT_PROTOCOL.md` (training fractions 10/30/50%). The verdict rule is the
protocol's. This brings the run studies from four to five. The 3-million-parameter hybrid
(PMID 40865268, no released code) is reported as "eligible, not reproducible: no code" and is
not counted as an audited claim. PROSTATEx and INCOV remain "eligible, not run".

## 7. K — protocol checklist applied to the eight eligible studies (descriptive)

The seven protocol rules (P1–P7) are restated as yes/no/unclear items with decision rules,
written in `docs/SYMMETRIC_EVALUATION_CHECKLIST.md` before any study is scored. Each of the
eight eligible studies is scored from its full text and any released code. This is
single-rater scoring and is disclosed as such. A second rater is sought.

## 8. Not in scope

The following are not part of this addendum: retraining the MIMIC extractor; new qubit
counts; hardware runs; a second glioma cohort; PROSTATEx and INCOV. Clinician re-labelling
follows `docs/RADIOLOGIST_ANNOTATION_PROTOCOL.md` when a credentialed reader is available. It
is not part of this addendum.

## 9. Deviations

Any deviation from this document is recorded in `aura/artifacts/v6/DEVIATIONS.md` with its
reason and the time it was made relative to seeing results, and is reported in the
manuscript.
