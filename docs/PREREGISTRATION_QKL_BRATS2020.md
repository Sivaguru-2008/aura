# Pre-specification: repeating the glioma-grade kernel comparison on official BraTS 2020 labels

**Written** 21 September 2026, while the embedding export was running and before any kernel was fitted to it.
**Author:** Sivaguru R. M.

## Why this exists

The manuscript's §2.6 (v5) reported that a 10-qubit quantum fidelity kernel did not differ from an RBF kernel for glioma grading on 185 subjects. On 13 September a dataset provenance audit found that those 185 subjects came from `datasets/brats2021_hdf5`, whose subject identifiers do not exist in BraTS 2020 and whose grades match the official mapping in only 32 of 47 colliding names. v5 carries an erratum withdrawing §2.6. This document fixes, before any result exists, how the comparison is repeated.

## What is not changed

The analysis is `aura/ml/evaluation/qkl_full_coverage.py` as committed in `3d9aea8` (12 September 2026, before the audit): five outer subject-level folds stratified by grade, a 75/25 inner fit/calibration split, symmetric 5 × 5 grids (C × γ for RBF; C × projection bandwidth for the quantum kernel), selection by inner-calibration AUROC, Platt scaling on the inner-calibration split, subject-level aggregation by mean slice probability, DeLong test on pooled out-of-fold subject predictions, bootstrap CI, seed 7. The working-tree diff adds only `--output`, `--exclude-encoder-train`, and data-driven description strings. No grid, fold count, sample size, seed, metric, or test was altered.

## Data

- **Images.** `E:\AURA\datasets\brats2020\BraTS2020_training_data\content\data`: the BraTS 2020 HDF5 corpus (369 subjects) that the served brain model `best_brain_model.pt` was trained on.
- **Labels.** `name_mapping.csv` shipped with that corpus: 293 HGG / 76 LGG, joined by `volume_N → BraTS20_Training_{N:03d}`.
- **Features.** 128-d encoder embeddings from the served checkpoint, every retained slice (`export_brats2020_embeddings.py`), no augmentation, no synthetic degradation.

## Label-join check (gate)

Before the kernel result is interpreted, `verify_brats2020_grade_mapping.py` tests the join against the images: under a correct join, the proportion of subjects with no enhancing-tumour voxels is higher among LGG than HGG (one-sided Fisher exact, α = 0.01). **If the official corpus fails this test, the kernel comparison is not reported as a grading result**, and §2.6 is removed from the manuscript. The same test is run on the withdrawn cohort and reported whatever it shows.

## Analyses

- **Primary.** All 369 subjects. Quantity: pooled out-of-fold subject-level ΔAUROC (quantum − RBF), DeLong p, bootstrap 95% CI.
- **Sensitivity (one, declared here).** The 111 subjects outside the encoder's training split (55 validation + 56 test), `--exclude-encoder-train`. Reason: the encoder was fitted to the other 258 subjects' segmentations, so their embeddings may be unrepresentatively clean. Grade was never a training target for any subject.

## Reporting commitments

- Whichever arm wins, and whether or not the difference is significant, the primary result replaces §2.6 in the next manuscript version, with the sensitivity result beside it.
- If both arms are near chance (pooled AUROC < 0.60), that is reported as the embedding not carrying grade, which the model card's own k-NN probe already suggests (test split: 0.780 against a 0.789 majority baseline). It is then not described as evidence about kernels.
- The withdrawn artifact `qkl_full_coverage.json` is kept, not overwritten, so the audit trail stays intact. New outputs go to `qkl_brats2020_official.json` and `qkl_brats2020_official_heldout111.json`.

## Limitation of this document

It is not externally timestamped. What gives it weight is that the analysis it governs was committed on 12 September, before the data problem was known, and is used unchanged.
