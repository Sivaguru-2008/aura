# Reporting checklist — TRIPOD+AI and CLAIM

**Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system** (submission version 5.0)

This document maps the manuscript against TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378) for the prediction-model components and CLAIM (Mongan et al., *Radiol Artif Intell* 2020;2:e200029) for the imaging components. Item descriptions below are **paraphrases**, not the checklists' wording; consult the source publications for the authoritative text.

**Two notes on scope, stated up front rather than buried in the table.**

First, this is a **methodological comparison study**, not a clinical prediction-model report. Its primary claims concern how quantum and classical components compare under a symmetric protocol. The prediction models exist to make that comparison possible. Several TRIPOD+AI items therefore concern reporting we deliberately do not make — the manuscript makes no clinical performance claim and recommends no clinical use.

Second, **items marked "Not reported" are genuine gaps, not editorial choices.** They are listed again in Supplementary §S8 as analyses identified as necessary and not performed. The most consequential is item 13c: no demographic subgroup analysis was conducted despite the metadata being available.

Legend: **✔** reported · **◐** partly reported, with the limit stated in the manuscript · **✘** not reported · **n/a** not applicable to this study design.

---

## TRIPOD+AI

### Title and abstract

| # | Item | Status | Location |
|---|---|:--:|---|
| 1 | Title identifies the study as developing/validating a prediction model, the target population, and the outcome | ◐ | Title. Identifies the comparison and the outcome class; frames the work as a methodological study rather than a prediction-model report, which is what it is |
| 2 | Structured abstract covering objectives, methods, results, conclusions | ✔ | Abstract (372 words, structured) |

### Introduction

| # | Item | Status | Location |
|---|---|:--:|---|
| 3a | Background: rationale, existing models, and why this work is needed | ✔ | §1 |
| 3b | Objectives, stated with the target population and outcome | ✔ | §1, final three paragraphs |

### Methods — data

| # | Item | Status | Location |
|---|---|:--:|---|
| 4a | Data sources, separately for development and evaluation | ✔ | Methods "Data"; §S1 |
| 4b | Dates of data collection and follow-up | ◐ | Corpus release versions are named (MIMIC-CXR, CheXpert, BraTS-2020); acquisition date ranges are properties of the public corpora and are not restated |
| 5a | Participant eligibility, setting, and inclusion/exclusion | ✔ | Methods "Data"; pre-registration §4 for the external cohort (frontal views, one study per patient) |
| 5b | Details of treatments received, where relevant | n/a | No treatment data; imaging interpretation only |
| 6a | Outcome definition and how/when it was determined | ✔ | Methods "Prediction target"; §S2 — full weighted-score derivation |
| 6b | Whether outcome assessors were blinded to predictors | ✘ | The outcome is rule-derived from the report by an automated labeller, so blinding does not apply in the usual sense. **The 66-report reference standard was read by the author, who was not blinded to the labeller's output** — stated in §S5.2 |
| 7 | Predictors: definition, timing, and assessment | ✔ | Methods "System"; §S1. Eight-channel evidence vector; the eighth channel is identically zero and this is disclosed (Methods; §2.4) |
| 8 | Sample size rationale | ◐ | Minimum detectable effects are reported rather than a prospective size calculation: 9.3 accuracy points internally, ≈3.6 externally (§S7.1). The external cohort's target size was fixed in the pre-registration before the run |
| 9 | Missing data handling | ✔ | Methods; CheXpert uncertainty policy pre-specified (U-Zeros primary, U-Ones sensitivity), §2.7 |

### Methods — model development and evaluation

| # | Item | Status | Location |
|---|---|:--:|---|
| 10a | How predictors were handled in the analysis | ✔ | Methods; §S1 |
| 10b | Model type, building procedure, and internal validation | ✔ | Methods "Matched optimiser protocol"; §S1, §S4.2 |
| 10c | Approach to class imbalance | ✔ | Per-class capping in the fusion training split; majority-class floors reported for every comparison; class support tabulated (§S2) |
| 11 | Model output: predicted probability, class, or score | ✔ | Six-way calibrated posterior; Methods |
| 12 | Performance measures and their rationale | ✔ | Methods "Statistics". Accuracy, negative log-likelihood, ECE and macro AUROC; NLL is the primary proper scoring rule |
| 13a | Model updating for the evaluation cohort | ✔ | §2.7 — all weights frozen, one temperature per arm refit, boundary pre-registered |
| 13b | Fairness: how it was assessed | ✘ | **Not assessed.** No demographic subgroup analysis, despite sex, age and race being available in MIMIC-CXR metadata. §S7.1; §S8 item 6 |
| 14 | Model interpretability/explainability methods | ✔ | Grad-CAM++ localisation from raw activations, with calibrated probability modulating overlay alpha only; §S1 |
| 15 | Risk of bias or methodological quality assessment | ✔ | §S7 in full, including a thirteen-item register of this project's own defects (§S7.3) |

### Results

| # | Item | Status | Location |
|---|---|:--:|---|
| 16 | Flow of participants, with a diagram where useful | ◐ | Split sizes and disjointness are reported at every stage (Methods; §S2); no flow diagram is provided |
| 17 | Participant characteristics, including outcome distribution | ◐ | Outcome distribution reported for every split (§S2, §S4.6). **Demographic characteristics are not reported** — the same gap as 13b |
| 18 | Comparison of development and evaluation data | ✔ | §2.7 — corpus differences in prevalence, site, scanner mix and reachable categories stated, with the explicit warning that internal and external accuracy are not comparable |
| 19 | Model specification sufficient for use | ✔ | §S1; all weights and artifacts released |
| 20 | Model performance with uncertainty | ✔ | Tables 1, 2, S1–S7; Jeffreys intervals, bootstrap intervals, Holm-adjusted *p*-values |
| 21 | Model updating results | ✔ | §2.7; refit temperatures reported per arm |

### Discussion

| # | Item | Status | Location |
|---|---|:--:|---|
| 22 | Interpretation, in the context of objectives and evidence | ✔ | §3 |
| 23 | Limitations | ✔ | §3; §S7 in full |
| 24 | Usability in practice and implications | ✔ | §3; Disclaimer. **The manuscript makes no clinical claim and recommends no clinical use** |

### Other

| # | Item | Status | Location |
|---|---|:--:|---|
| 25 | Funding and role of funders | ✔ | Declarations — none |
| 26 | Conflicts of interest | ✔ | Declarations — none |
| 27a | Protocol and registration | ◐ | §2.7 is governed by a pre-registration deposited before the run (`docs/PREREGISTRATION_CHEXPERT.md`, commit `fa8c2b6`). **§2.1–§2.6 are not pre-registered**, and §S3 states this rather than implying otherwise |
| 27b | Data availability | ✔ | Declarations |
| 27c | Code availability | ✔ | Declarations; Supplementary Table S8 maps each claim to its artifact |

---

## CLAIM — imaging-specific items

Applies to the chest-radiograph model (§S5.1) and the brain-MRI model (§S5.3).

| Topic | Status | Location |
|---|:--:|---|
| Imaging protocol, acquisition parameters | ◐ | Properties of the public corpora; the standardisation applied (224 × 224, per-finding preprocessing; per-slice z-score for MRI) is stated in §S1 |
| Data de-identification | ✔ | Declarations — all three corpora are released de-identified |
| Definition of ground truth and how it was established | ✔ | §S2; §S5.2, including the explicit statement that the category map is **not** validated |
| Ground-truth rater qualifications and adjudication | ◐ | Stated and stated as a limitation: a single non-radiologist annotator, no second reader, no adjudication (§S5.2) |
| Train/validation/test partition and independence | ✔ | Methods; patient-, volume- and subject-disjoint at the correct unit (P3) |
| Model architecture, initialisation, training | ✔ | §S1; Methods |
| Performance metrics, including at the operating point | ✔ | §S5.1 — AUROC, AUPRC with prevalence baseline, sensitivity, specificity, **PPV**, NPV and F1. Macro PPV 0.264 is reported prominently as the clinically decisive quantity |
| Calibration | ✔ | Per-finding Platt scaling, ECE 0.181 → 0.026 (§S5.1); presence head 0.095 → 0.018 (§S5.3) |
| Failure analysis | ✔ | §S5.2 — the nodule failure is disclosed, localised to the image model rather than the labeller, and traced through to the malignancy category |
| Comparison with other methods | ◐ | §S5.3 states explicitly that the cited nnU-Net/SwinUNETR/MONAI figures are **published literature values, not measurements on these data**, and that pooled-2-D and per-case-3-D Dice are not comparable |
| External validation | ◐ | Present for the fusion comparison (§2.7). **Absent for both imaging models**; §S8 item 1 |

---

## Summary of gaps

Four items are genuine and material gaps, and all four appear in Supplementary §S8:

1. **No fairness or subgroup analysis** (TRIPOD+AI 13b, 17) — the largest gap for a clinical venue, and the metadata to close it is available.
2. **No external validation of the imaging models** (CLAIM) — external validity is established for the fusion comparison only.
3. **No blinded, adjudicated reference standard** (TRIPOD+AI 6b; CLAIM) — a single non-radiologist reader, no second reader.
4. **Pre-registration covers one of seven analyses** (TRIPOD+AI 27a) — §2.7 only.

Two further items are limitations of study design rather than of reporting: there is no participant flow diagram, and no prospective sample-size calculation (minimum detectable effects are reported instead).
