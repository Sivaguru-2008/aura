# Pre-registration addendum (b) — fusion comparisons on out-of-sample evidence

**Deposited:** 2 October 2026, committed before any analysis below was run.
**Executes in:** `aura/ml/evaluation/chexpert_internal.py` → `aura/artifacts/chexpert_internal.json`.

## Why

Every internal fusion result in the manuscript uses evidence vectors that the DenseNet-121 finding
extractor produced for images in its own training split (`mimic_cxr_aug_train.csv`). All arms see
the same vectors, so the comparisons are symmetric, but the evidence is in-sample for the image model.
The MIMIC validation split has only 500 patients, too few to replace the 2,231-study set. CheXpert
images were never seen by the extractor, and 8,000 patient-disjoint CheXpert evidence vectors
already exist (`chexpert_evidence.npz`). Here the fusion arms are **trained and tested within**
CheXpert. That is unlike §2.8, where they were trained on MIMIC and only tested on CheXpert.

## What is and is not blind

Nothing in this addendum has been run. The §2.8 external results (MIMIC-trained arms scored on
CheXpert) are known. No arm has been trained on CheXpert evidence.

## Data

* `chexpert_evidence.npz`: 8,000 frontal studies, one per patient (selection rule in
  `chexpert_external.select_studies`). Labels: `y_u0` (uncertain → negative), the pre-registered
  primary policy of `PREREGISTRATION_CHEXPERT.md`. Five reachable categories (no COPD).
* Split: one permutation with `numpy.random.default_rng(7)`, 60/20/20 → 4,800 train / 1,600
  calibration / 1,600 test. One study per patient, so the split is patient-disjoint.
* Channel 8 (`prior_risk`) is identically zero, as on MIMIC.

## E1 — matched-optimiser comparison (replicates §2.3)

`matched_optimiser.main` run **unchanged** (same arms, LR grid, early stopping, seed, per-arm
temperature, McNemar and Wilcoxon with Holm over six pairs), with only the data source replaced.
Additionally: TOST for quantum vs product-of-experts on per-study NLL, margin ±0.010 (the same margin
as addendum (a)).

**Reading:** a significant quantum-vs-PoE NLL or accuracy difference (Holm p < 0.05) in either
direction is reported as a change from §2.3. Otherwise §2.3's "no difference detected" stands.
Equivalence is claimed only if the TOST passes.

## E2 — 7-qubit entanglement ablation (replicates §2.5 analysis B)

The analysis-B protocol from addendum (a), unchanged: 7 informative channels, 3 layers, RY(πx),
`train_quantum` (Adam lr 0.05, batch 50, 30 epochs), seeds {7, 11, 13, 17, 19, 23, 29, 31, 37, 41},
ring vs no entangler, per-arm temperature on calibration, per-study test predictions saved.
Primary: patient bootstrap of the seed-averaged per-patient ΔNLL (10,000 resamples). Secondary:
hierarchical seed × patient bootstrap.

**Reading:** primary 95% CI above 0 → "the entanglement penalty replicates on out-of-sample
evidence". CI containing 0 → "not replicated on out-of-sample evidence". CI below 0 → reversal.

## Not in scope

The re-uploading ablation, the design sweep and the positive control are not repeated here.
Labels differ from MIMIC (five categories, different label source), so these results are a
replication on a second corpus with out-of-sample evidence, not a substitute for the MIMIC numbers.
