# Symmetric Evaluation Checklist for claims of a novel model component

A reusable form of the seven-rule protocol (P1–P7) from the manuscript. Use it when a paper
claims that a new component (a quantum layer, a new architecture block, a new loss) beats a
classical or established comparator. It does not certify a result. It asks whether the
comparison could have shown the opposite.

Each item is answered **Yes**, **No** or **Unclear**:

- **Yes** means the paper text or released code shows the condition is met.
- **No** means the paper or code shows it is not met.
- **Unclear** means the material does not say.

Score only what the paper and code state. Do not infer intent. Quote the passage, or give the
file and line, that decided each answer.

Written 3 October 2026, before any study was scored with it (see
`PREREGISTRATION_v6_2026-10-03.md` §7).

---

## C1. Calibration and post-processing symmetry (P1)

**C1.1** If any arm receives post-hoc calibration, thresholding, ensembling or test-time
augmentation, does every compared arm receive the same, each fitted on its own data?
- *Yes* if the paper applies the same post-processing to all arms, or to none.
- *No* if only the proposed arm gets it.
- *Unclear* if post-processing is not described for the comparators.

**C1.2** Are probability-based metrics (log-loss, ECE, Brier) computed on arms calibrated the
same way?
- *Yes*, *No*, or *N/A* if no probability metric is reported.

## C2. Tuning and capacity symmetry (P2)

**C2.1** Do comparator arms receive a hyperparameter search of comparable size and on the same
selection data as the proposed arm?
- *No* if comparators use library defaults or values from another paper while the proposed arm
  is tuned.

**C2.2** Is the comparator set justified, and does it include at least one strong, tuned model
of the same input?
- *No* if only weak baselines are used (e.g. untuned SVM, perceptron, single tree) while
  stronger standard models are omitted.

**C2.3** Is capacity (parameters, or a stated capacity proxy) reported for each arm, and are
ablations capacity-matched?
- *N/A* if there is no ablation.

**C2.4** Are all arms trained by the same optimiser protocol (or each to convergence), with
the same stopping rule and the same data?

## C3. Split integrity (P3)

**C3.1** Is the split made at the correct unit — patient or subject, not image, recording,
epoch or row — whenever a unit contributes more than one row?
- *No* if repeated units can cross the split.

**C3.2** Are all data-dependent preprocessing steps (scaling, PCA, feature selection,
imputation, resampling such as SMOTE) fitted inside the training fold only?

**C3.3** Is model, epoch or hyperparameter selection made without the test data?
- *No* if early stopping or selection uses the test set.

## C4. Inference before ranking (P4)

**C4.1** Is every claimed superiority supported by a paired test or a confidence interval on
the difference, not only by point estimates?

**C4.2** If several comparisons, metrics or datasets are reported, is a multiplicity policy
stated?

**C4.3** Is the resampling scheme's dependence handled (e.g. the Nadeau–Bengio correction for
repeated CV), or is a held-out test set used?

## C5. Negative outcomes (P5)

**C5.1** Are metrics, datasets or settings where the proposed arm did not win reported
alongside the wins?
- *Unclear* if only winning metrics appear and no negative result is mentioned.

## C6. Artifact-derived reporting (P6)

**C6.1** Are code and the exact data split (or a recipe that regenerates it) released?

**C6.2** Can the headline numbers be regenerated from released artifacts without manual
transcription?
- *Unclear* if code is released but no outputs are.

## C7. Mechanism audit (P7)

**C7.1** If the paper attributes the result to a specific mechanism (entanglement, a quantum
feature space, a hardware effect), is there an ablation or source-level check showing that the
mechanism is present and responsible?
- *No* if the mechanism is asserted only.
- *N/A* if no mechanism is claimed.

**C7.2** If hardware or shot-based execution is claimed, does the implementation actually
execute at finite shots or on the stated device?
- *N/A* if the paper does not claim this.

---

## Scoring sheet

| Item | Answer | Evidence (quote or file:line) |
|---|---|---|
| C1.1 | | |
| C1.2 | | |
| C2.1 | | |
| C2.2 | | |
| C2.3 | | |
| C2.4 | | |
| C3.1 | | |
| C3.2 | | |
| C3.3 | | |
| C4.1 | | |
| C4.2 | | |
| C4.3 | | |
| C5.1 | | |
| C6.1 | | |
| C6.2 | | |
| C7.1 | | |
| C7.2 | | |

**Summary rule.** A claim is *at risk of asymmetric evaluation* if any of C1.1, C2.1, C3.1,
C3.2 or C3.3 is **No**. It is *unsupported by inference* if C4.1 is **No**. These two labels are
descriptive. They do not mean the claim is false. That needs a re-test such as the field audit.
