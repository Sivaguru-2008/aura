# AURA vs. Published Literature and Market Products — Accuracy Comparison

**Date:** 2026-07-30
**Rule followed:** every AURA number below is quoted from a JSON/CSV artifact in this repo, never from prose in a `.md`. Artifact path is given for each. Every external number is a citation to a published source.

---

## 0. Executive verdict

| Track | AURA (measured) | Nearest published academic | Market product | Verdict |
|---|---|---|---|---|
| Chest X-ray, 7 findings | macro AUROC **0.821** (n=2099) | DenseNet-121 on MIMIC-CXR, mean per-class AUC 0.8335 (13 findings) | Annalise CXR macro AUC **0.957** (124 findings) | **At academic-baseline parity. Far below market.** |
| Brain MRI tumour segmentation | Dice WT **0.915** / TC **0.846** / ET **0.835** (n=56 subjects) | nnU-Net BraTS2020 winner **0.8895 / 0.8506 / 0.8203** | Brainlab / icobrain (regulatory-cleared) | **Not comparable — different metric. See §2.3. The apparent win is a metric artifact.** |
| Cross-modal fusion diagnosis | accuracy **0.6957** classical / **0.6377** quantum (n=69) | — | — | **Barely above the majority-class baseline (0.6377). No external comparator exists at this n.** |

**The single most important finding of this exercise is not an accuracy number.** It is that the repo's existing `docs/benchmark_report.md` compares AURA against nnUNet, SwinUNETR and MONAI using numbers that were **never measured** — they are hardcoded string literals in the generator script. See §4. That table must not be shown to a judge, reviewer, or customer in its current form.

---

## 1. Track A — Chest radiograph findings

### 1.1 What AURA actually scores

Source: `aura/artifacts/evaluation_retrain_v2/metrics.json` (served checkpoint, `retrain_v2` epoch 7, n=2099 MIMIC-CXR images, 224 px).

| Metric | Value | 95% CI |
|---|---|---|
| Macro AUROC | **0.821** | [0.8083, 0.8332] |
| Macro AUPRC | 0.3576 | [0.3402, 0.3847] |
| Macro F1 | 0.3679 | — |
| Macro sensitivity | 0.6926 | — |
| Macro specificity | 0.7684 | — |
| **Macro precision** | **0.2638** | — |
| Macro ECE | 0.1999 | — |
| Throughput | 252 img/s (GPU batch), 47.9 ms/img CPU single | `aura/artifacts/performance/benchmark.json` |

Per finding (AUROC, with 1000-bootstrap CI):

| Finding | AUROC | CI95 | Sens | Spec | Precision | Support |
|---|---|---|---|---|---|---|
| hyperinflation | 0.912 | [0.884, 0.936] | 0.808 | 0.865 | 0.267 | 120 |
| pleural_effusion | 0.900 | [0.888, 0.913] | 0.884 | 0.769 | 0.560 | 524 |
| cardiomegaly | 0.862 | [0.841, 0.881] | 0.852 | 0.710 | 0.315 | 284 |
| pneumothorax | 0.825 | [0.778, 0.863] | 0.460 | 0.866 | 0.096 | 63 |
| consolidation | 0.807 | [0.771, 0.839] | 0.808 | 0.660 | 0.126 | 120 |
| nodule | 0.729 | [0.679, 0.776] | 0.358 | 0.874 | 0.119 | 95 |
| opacity | 0.712 | [0.687, 0.737] | 0.677 | 0.635 | 0.365 | 496 |

### 1.2 Against published academic work

| System | Task | Reported | AURA equivalent | Gap |
|---|---|---|---|---|
| DenseNet-121 on MIMIC-CXR, 13 findings | multi-label | mean per-class AUC **0.8335** | 0.821 (7 findings) | **−0.013 — parity** |
| DenseNet-121 on MIMIC-CXR, 5 findings | multi-label | mean per-class AUC **0.8812** | 0.856 (AURA's effusion/cardiomegaly/consolidation mean) | **−0.025** |
| CheXzero (zero-shot, external NIH set) | cardiomegaly | 0.825 | 0.862 | +0.037 (but in-domain vs external — not a fair win) |
| CheXzero | pleural effusion | 0.836 | 0.900 | +0.064 (same caveat) |
| CheXzero | pneumothorax | 0.764 | 0.825 | +0.061 (same caveat) |

**Read:** AURA's chest backbone sits **at the level of a competently-trained standard DenseNet-121 baseline on MIMIC-CXR**. It is not below the literature and it is not above it. The apparent wins over CheXzero are not real wins — CheXzero's numbers are *external-domain zero-shot*, AURA's are *in-domain on the same corpus it trained on*, which is a strictly easier setting.

### 1.3 Against market products

| Product | Scope | Published performance | AURA |
|---|---|---|---|
| **Annalise CXR** (harrison.ai) | 124 findings | macro AUC **0.957**; beat unassisted radiologists (0.713 macro AUC) on 117/124 findings — Seah et al., *Lancet Digital Health* 2021 | 0.821 on 7 findings |
| **Lunit INSIGHT CXR** | 10 findings | **AUC 0.93 for lung nodule** in a head-to-head multi-vendor study; human reader mean 0.81 | **nodule AUROC 0.729** |
| **Lunit INSIGHT CXR** | abnormality | sensitivity 0.89 at specificity comparable to readers (~0.80) | macro sens 0.693 at spec 0.768 |

**Read:** the gap to commercial systems is **large and not closeable by tuning**. Two structural reasons:

1. **Label ceiling.** AURA's training labels are NLP-derived from MIMIC free-text reports, not expert pixel/finding annotations. That regex labeller was validated (`aura/artifacts/kappa_crosscheck.json` + the 66-report hand-read gold: kappa 0.86, F1 0.89), so the labels are *honest* — but a report-derived label is a noisier target than the radiologist-consensus ground truth Annalise and Lunit train against. You cannot exceed your label quality.
2. **Nodule is the weakest finding and the one that failed independent corroboration.** `kappa_crosscheck.json` records that nodule does not corroborate against `torchxrayvision densenet121-res224-mimic_ch`, and it is also AURA's worst AUROC (0.729) against Lunit's flagship 0.93. This is the clearest single accuracy deficit in the system.

### 1.4 The metric nobody quotes — precision

This is where AURA is furthest from clinical usability, and it is invisible in an AUROC-only comparison.

At the served operating point, **macro precision is 0.2638**. Concretely, from the confusion matrices in `evaluation_retrain_v2/metrics.json`:

- **consolidation**: 97 true positives against **672 false positives** (precision 0.126)
- **nodule**: 34 true positives against 253 false positives (precision 0.119)
- **pneumothorax**: 29 true positives against 273 false positives (precision 0.096) — and it misses 54% of real pneumothoraces (sens 0.460)

A commercial triage device flagging ~10 false alarms per true consolidation would be switched off within a week. The operating point is tuned hard toward sensitivity; that is a defensible research choice, but it must be stated whenever the 0.821 AUROC is quoted, because AUROC is threshold-free and hides it entirely.

Calibration is also weak at the finding level: **macro ECE 0.1999**. (Note this is *separate* from the diagnosis-presence head, which is properly Platt-scaled — ECE 0.095 → 0.018, `aura/artifacts/brain/presence_calibration.json` and the vision serving calibration.)

---

## 2. Track B — Brain MRI tumour segmentation

### 2.1 What AURA scores

Source: `aura/artifacts/brain/reports/test_report.json` — 56 held-out subjects, 7531 slices, split by subject stratified by tumour grade (patient-disjoint, per `reports/model_card.json`).

| Region | Dice (pooled) | Dice (per-slice mean) | HD95 (px) |
|---|---|---|---|
| Whole tumour | **0.91498** | 0.87994 | 7.08 |
| Tumour core | **0.84561** | 0.88614 | 6.15 |
| Enhancing tumour | **0.83486** | 0.89666 | 4.53 |
| **Composite mean** | **0.86515** | — | — |

### 2.2 Published state of the art

| System | BraTS2020 validation, per-case 3D Dice | | |
|---|---|---|---|
| | WT | TC | ET |
| **nnU-Net (Isensee et al., BraTS2020 winner)** | **0.8895** | **0.8506** | **0.8203** |
| AURA (as reported by `test_report.json`) | 0.91498 | 0.84561 | 0.83486 |

### 2.3 Why you cannot claim a win here

**The two rows above are not the same measurement.** Three mismatches, all favouring AURA:

1. **Pooled voxel Dice vs per-case Dice.** AURA's headline is pooled over every voxel in the split — `test_report.json` says so explicitly: *"dice is pooled over all voxels of the split"*. BraTS computes Dice **per patient, then averages**. Pooled Dice is a micro-average dominated by large tumours; a patient whose tumour is missed entirely contributes a handful of voxels to a pooled score but contributes a **0.0** to a per-case average. This is the single largest source of inflation.
2. **2D slices vs 3D volumes.** AURA is a 2D slice model evaluated slice-wise. Standard guidance for volumetric segmentation is to compute Dice over the full volume rather than averaging slices, because slice-wise evaluation biases toward slice-rich regions and toward slices where the tumour is large and easy.
3. **Different evaluation set.** nnU-Net's numbers are on the *official BraTS2020 validation server*. AURA's are on a self-made 56-subject split of the BraTS2020 **training** archive. Not the same cases, not the same difficulty, no server-side arbitration.

**Correct statement:** *"AURA's 2D segmentation achieves pooled voxel Dice 0.915/0.846/0.835 on a held-out 56-subject split of BraTS2020 training data. This is not directly comparable to the BraTS2020 leaderboard, which reports per-case 3D Dice on the official validation set."* Anything stronger than that is unsupported.

### 2.4 A hard operational limit the market comparison must include

From `aura/artifacts/brain/presence_calibration.json`:

> measured single-sequence whole-tumour Dice is **0.52 FLAIR, 0.28 T2, 0.02 T1, 0.00 T1ce** against 0.58 with all four.

The model **requires all four MRI sequences**. Commercial neuro tools are generally validated for degraded/partial protocols. AURA's performance collapses to near-zero on a T1-only study. That is a deployment blocker, not a caveat.

---

## 3. Track C — Cross-modal fusion diagnosis

Source: `aura/artifacts/benchmark.json`, n=69 MIMIC-CXR studies, 6 diagnosis classes.

| Backend | Accuracy | NLL | ECE | Brier | Macro AUROC | Macro F1 |
|---|---|---|---|---|---|---|
| Quantum (served) | 0.6377 | 1.2123 | 0.2381 | 0.5699 | 0.7696 | 0.3321 |
| Classical | **0.6957** | **1.0577** | 0.2194 | 0.4857 | **0.7875** | **0.4241** |
| Ensemble | 0.6957 | 1.0213 | 0.2267 | 0.4656 | 0.7798 | 0.4241 |
| Learnable | 0.6232 | 1.1306 | **0.1879** | 0.5273 | 0.7306 | 0.3803 |

Three observations that any external reviewer will make immediately:

1. **The majority-class baseline is 44/69 = 0.6377.** The served quantum backend's accuracy is *numerically identical to always predicting "normal"* (44 correct out of 69). It is not literally doing that — its `normal` sensitivity is 0.7955, not 1.0 — but the headline accuracy carries no information above the trivial baseline. Classical clears it by 4 studies.
2. **Two of six classes have no discrimination at all.** `malignancy` AUROC 0.577 (quantum) / 0.627 (classical), `pneumothorax_dx` 0.493 / 0.485 — i.e. chance — with sensitivity **0.0** for both, in both backends. Supports are 4 and 2 studies.
3. **n=69 is too small for any comparison to the market.** With 2–8 studies per minority class, no confidence interval here excludes anything.

### 3.1 The quantum claim, as adjudicated by your own audit

`audit_artifacts/run_20260719T175647Z/metrics/claim_verdicts.json`:

| Claim | Verdict |
|---|---|
| "Quantum fusion accuracy exceeds classical" | **NOT VALIDATED** (McNemar p=0.25, n=100) |
| "Quantum is far better calibrated (ECE 0.020 vs 0.276)" | **CONFOUNDED** — 0.276 was the *uncalibrated* classical; given its own temperature the gap shrinks from 0.256 to 0.007 |
| "Quantum ranking (AUROC) beats classical" | **NOT VALIDATED** — 0 of 6 classes significant by DeLong |

On the served benchmark the classical backend is **ahead on every accuracy metric** (accuracy +0.058, NLL −0.155, macro AUROC +0.018, macro F1 +0.092). The honest positioning of the quantum layer is the measurement-efficiency and typed-abstention result, not accuracy.

---

## 4. Red flags in the repo's existing comparison table

`docs/benchmark_report.md` currently publishes this:

| Architecture | Mean Composite Dice | CPU latency |
|---|---|---|
| nnUNet | 0.871 | 185.0 s |
| SwinUNETR | 0.858 | 240.0 s |
| MONAI Baseline | 0.835 | 12.5 s |

**None of these were measured.** They are hardcoded literals:

- `aura/ml/evaluation/run_pipeline.py:273-275` — the markdown table rows
- `aura/ml/evaluation/run_pipeline.py:310-312` — the same values written into `comparison_table.csv`
- `aura/ml/evaluation/run_pipeline.py:287` — "Classical Chest Baseline 0.7650 / 0.2850 / 0.2850", also invented

A duplicate of the same generator exists at `aura/change/ml/ml/evaluation/run_pipeline.py` with the identical literals at the identical lines (266, 273, 310) — fix both or the table regenerates.

No nnUNet, SwinUNETR or MONAI model was ever run in this repo; no citation is attached; the latency figures (185 s, 240 s, 8.52 s GPU) have no source. The AURA columns in that table are real (they interpolate from `evaluation.json`), which makes the fabricated comparator columns *more* dangerous, not less — the table looks internally consistent.

Two further issues in the same file:

- The chest row quotes **AUROC 0.8095 / F1 0.3330 / ECE 0.2087** from `evaluation.json`, which is a **602-image** run. The full validated run is **2099 images, AUROC 0.821** (`evaluation_retrain_v2/metrics.json`). Two different headline numbers for the same model are in circulation.
- The nnUNet comparison is stated against AURA's pooled 2D Dice, compounding §2.3.

**Recommended action:** delete rows 273–275, 287 and 310–312 from `run_pipeline.py`, regenerate, and replace with a cited literature table clearly labelled *"published figures, different evaluation protocol — not a head-to-head run."*

---

## 5. Bottom line

- **Chest CXR:** genuinely at the level of a solid published DenseNet-121 MIMIC baseline (0.821 vs 0.8335). The honest gap to market is precision (0.264) and nodule detection (0.729 vs 0.93), not AUROC.
- **Brain MRI:** the numbers look competitive with the BraTS2020 winner only because the metric is different. On a like-for-like per-case 3D protocol, expect materially lower. The four-sequence requirement is a harder limit than any accuracy number.
- **Fusion:** at n=69 with a 0.6377 majority baseline, this is a demo, not a benchmark. The quantum-beats-classical claim is refuted by the repo's own audit; classical currently wins.
- **What is actually differentiating** and does not appear in any competitor's spec sheet: conformal coverage guarantees (measured 91.3–92.8% at a 90% target), typed abstention, the OOD/non-radiograph gate, and the label-provenance chain (66-report hand-read gold, independent torchxrayvision corroboration with a disclosed nodule failure). Lead with those; do not lead with accuracy.

### Sources

- [nnU-Net for Brain Tumor Segmentation (Isensee et al., BraTS2020 winner)](https://arxiv.org/abs/2011.00848)
- [Seah et al., *Lancet Digital Health* 2021 — comprehensive deep-learning CXR model, MRMC study](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00106-0/fulltext)
- [Seah et al., real-world observational study of the same model](https://pmc.ncbi.nlm.nih.gov/articles/PMC8689166/)
- [Lunit INSIGHT CXR — head-to-head lung nodule study published in *Radiology*](https://www.lunit.io/en/company/news/lunit-insight-cxr-excels-in-lung-nodule-detection---exceptional-performance-in-head-to-head-study-published-in-radiology)
- [Lunit ECR 2024 validation results](https://www.lunit.io/en/media-hub/lunit-presents-seven-study-results-at-ecr-2024-showcasing-ais-robust-performance-in-diverse-clinical-settings/)
- [Open-source deep neural networks for comprehensive chest x-ray reading, *Lancet Digital Health* 2023](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00218-2/fulltext)
- [CheXpert dataset and leaderboard](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [MS-CXR — DenseNet-121 MIMIC-CXR multi-label baselines](https://link.springer.com/chapter/10.1007/978-981-95-3355-8_11)
- [Image Segmentation Evaluation With the Dice Index: Methodological Issues](https://onlinelibrary.wiley.com/doi/10.1002/ima.23203)
- [Official BraTS segmentation performance metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics)
