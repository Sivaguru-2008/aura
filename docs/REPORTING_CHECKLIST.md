# Reporting checklist — TRIPOD+AI and CLAIM

**Auditing quantum machine learning in digital health: one deployed system and five published claims under symmetric evaluation** (manuscript v5.1)

This document maps the manuscript against TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378): 27 items, 52 sub-items. It also maps the imaging components against CLAIM (Mongan et al., *Radiol Artif Intell* 2020;2:e200029).

> **Before submission:** the item descriptions below are paraphrases. The numbering and sub-item structure follow the published statement: 27 items and 52 sub-items, with open science at item 18 and patient and public involvement at item 19. That structure is cross-checked against the statement's own text (PMID 38626948). The paraphrases of 12d–12g, however, were reconstructed without the official form. Transfer this mapping onto the official template (the statement's supplementary table 2) and check every item's wording there. **The previous version of this file used TRIPOD 2015 numbering under a TRIPOD+AI heading. It is replaced, not amended.**

**Scope.** This is a methodological comparison study, not a clinical prediction-model report. The models exist so that the comparison can be made. Several items concern claims the manuscript deliberately does not make: it makes no clinical performance claim and recommends no clinical use.

Legend: **✔** reported · **◐** partly reported, with the limit stated · **✘** not reported (a genuine gap) · **n/a** not applicable to this design. Section numbers refer to v5.1. §S refers to the Supplementary.

---

## TRIPOD+AI

### Title and abstract

| # | Item (paraphrase) | Status | Location / note |
|---|---|:--:|---|
| 1 | Title identifies a study developing or evaluating a prediction model, the population and the outcome | ◐ | Title names the study as an audit of comparisons, not a prediction-model report |
| 2 | Abstract per TRIPOD+AI for Abstracts | ◐ | Structured abstract. Not separately checked against the 13-item abstract checklist |

### Introduction

| # | Item | Status | Location / note |
|---|---|:--:|---|
| 3a | Healthcare context and rationale | ✔ | §1 |
| 3b | Target population and intended purpose in the care pathway | ◐ | §1 and Disclaimer: no intended clinical use. The system is a test bed |
| 3c | Known health inequalities between groups | ✘ | Not discussed |
| 4 | Objectives, including development vs evaluation | ✔ | §1, final paragraphs |

### Methods

| # | Item | Status | Location / note |
|---|---|:--:|---|
| 5a | Data sources and their representativeness | ◐ | Methods "Data"; §S1. Representativeness of MIMIC-CXR/CheXpert/BraTS populations not discussed |
| 5b | Dates of data collection | ◐ | Corpus versions named; date ranges are corpus properties, not restated |
| 6a | Setting and centres | ✔ | Methods "Data" (two chest centres, BraTS multi-institution) |
| 6b | Eligibility criteria | ✔ | Methods "Data"; one study per patient (most recent) for fusion; frontal views for CheXpert |
| 6c | Treatments received | n/a | Imaging interpretation only |
| 7 | Data preparation and quality checks, incl. across groups | ◐ | §S1 (standardisation); `grade_mapping_check.json` label-integrity check (§3.1). Not by demographic group |
| 8a | Outcome definition, time horizon, and why | ✔ | Methods "Prediction target"; §S2. **The chest target is rule-derived from reports, not a clinical reference standard** (§2.4, §3.1) |
| 8b | Qualifications and demographics of outcome assessors, if subjective | ◐ | The rule labeller is automatic. The 66-report reference read was done by the author, a single non-radiologist (§S5.2) |
| 8c | Blinding of outcome assessment | ✘ | The 66-report reader was not blinded to the labeller's output (§S5.2) |
| 9a | Choice of initial predictors and pre-selection | ✔ | Methods "System"; §S1. The eight-channel evidence vector has one identically zero channel, which is disclosed (§2.5) |
| 9b | Predictor definitions and measurement | ✔ | §S1 |
| 9c | Qualifications and demographics of predictor assessors | n/a | Predictors are model outputs, not human assessments |
| 10 | Sample size and its justification | ◐ | No prospective calculation. Minimum detectable effects are reported instead (§S7.1). The external cohort size was fixed in its pre-registration. The TOST interval (§2.3) shows that n = 447 cannot resolve differences below about 0.03 nats |
| 11 | Missing data handling | ✔ | CheXpert uncertainty policy pre-specified (§2.8) |
| 12a | How data were used (development / tuning / evaluation) | ✔ | Methods; 1,338/446/447 train/calibration/test; per-arm temperature on calibration only (P1) |
| 12b | Predictor handling in the analysis | ✔ | §S1 |
| 12c | Model type, building, hyperparameter tuning, internal validation | ✔ | Methods "Matched optimiser protocol" (§2.3); §S4.2 |
| 12d | Heterogeneity across clusters (e.g. centres) | ◐ | Two chest centres are analysed separately (§2.8). There is no within-corpus cluster analysis |
| 12e | Performance measures and rationale, incl. in subgroups | ◐ | Methods "Statistics": NLL as the primary proper scoring rule, plus accuracy, ECE and macro AUROC. **No subgroup measures** |
| 12f | Model updating and recalibration | ✔ | Frozen weights and one temperature refit per arm (§2.8). Boundary pre-registered |
| 12g | For evaluation, how predictions were computed | ✔ | Methods; all prediction arrays released (`*_predictions.npz`) |
| 13 | Class imbalance and its handling | ✔ | Per-class capping; majority-class floors reported for every comparison |
| 14 | Fairness approaches and rationale | ✘ | **Not assessed.** Sex, age and race are available in MIMIC-CXR metadata and were not used (§S8) |
| 15 | Model output and how it was derived | ✔ | Six-way calibrated posterior |
| 16 | Differences between development and evaluation data | ✔ | §2.8: prevalence, site, reachable categories. The text warns that internal and external accuracy are not comparable |
| 17 | Ethical approval | ✔ | Declarations "Ethics" (public de-identified data; PhysioNet credentialed DUA) |

### Open science

| # | Item | Status | Location / note |
|---|---|:--:|---|
| 18a | Funding | ✔ | Declarations: none |
| 18b | Conflicts of interest | ✔ | Declarations: none |
| 18c | Protocol availability | ✔ | `PREREGISTRATION_CHEXPERT.md`, `FIELD_AUDIT_PROTOCOL.md`, `PREREGISTRATION_ADDENDUM_2026-10.md`, all in the repository |
| 18d | Study registration | ◐ | Each document is registered by its git commit, not on an external registry (OSF/ClinicalTrials). **Only §2.8 and the §2.5 re-split and re-uploading analyses were blind at registration.** The addendum states that §2.3's TOST and §2.5's bootstrap were not blind |
| 18e | Data availability | ✔ | Declarations; credentialed corpora are not redistributed |
| 18f | Code availability | ✔ | Declarations; claim-to-artifact map in §S9.1 |
| 19 | Patient and public involvement | ✘ | None. State this explicitly in Declarations |

### Results

| # | Item | Status | Location / note |
|---|---|:--:|---|
| 20a | Participant flow, with diagram | ◐ | Split sizes at every stage. **No flow diagram.** For the field audit, the screening funnel 231 → 129 → 41 → 8 → 4 studies / 5 claims is in §2.1 |
| 20b | Participant characteristics, incl. demographics | ◐ | Outcome distribution per split (§S2). **No demographics** |
| 20c | Comparison of development and evaluation data | ✔ | §2.8 |
| 21 | Number of participants and outcome events per analysis | ✔ | Per-class support for every split (§S2, §S4.6); 23 LGG in the 111-subject glioma cohort (§2.7) |
| 22 | Full model specification | ✔ | §S1; all weights released |
| 23a | Performance with CIs, incl. key subgroups | ◐ | Jeffreys, bootstrap and t intervals; Holm-adjusted *p*. **No subgroups** |
| 23b | Heterogeneity across clusters | ◐ | Internal vs external only (§2.8) |
| 24 | Model updating results | ✔ | Refit temperatures per arm (§2.8) |

### Discussion

| # | Item | Status | Location / note |
|---|---|:--:|---|
| 25 | Interpretation, incl. fairness | ◐ | §3. Fairness is not discussed (follows from 14) |
| 26 | Limitations, incl. non-representative data and bias | ✔ | §3; §S7, including the defect register |
| 27a | Handling of poor-quality or unavailable input data at deployment | ◐ | §S1 (intake QC, modality gate); not discussed in the manuscript body |
| 27b | Whether users need to interact with or interpret inputs/outputs, and the expertise required | n/a | No clinical use is recommended |
| 27c | Future research and open questions | ✔ | §S8 |

---

## Summary of TRIPOD+AI gaps

| Gap | Items | Fixable before submission? |
|---|---|---|
| No fairness / subgroup analysis | 3c, 12e, 14, 20b, 23a, 25 | **Not with the data on disk.** The MIMIC-CXR derivative used (`mimic_cxr_aug_*.csv`) carries no age, sex or race, and the evidence cache stores no patient IDs, so test studies cannot be linked to demographics. Doing so needs the MIMIC-IV `patients` table. This needs a re-download and a rebuild with IDs retained. Even then, the 447-study test set gives wide subgroup intervals, which should be reported as descriptive |
| No patient and public involvement | 19 | State "none" explicitly in Declarations |
| Unblinded single-reader reference read | 8b, 8c | No; disclose (already done) |
| No flow diagram | 20a | Yes; numbers already exist |
| Registration is by git commit, not an external registry | 18d | Partly: deposit the three protocol files on OSF with their commit hashes |

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
| External validation | ◐ | Present for the fusion comparison (§2.8). **Absent for both imaging models**; §S8 item 1 |
