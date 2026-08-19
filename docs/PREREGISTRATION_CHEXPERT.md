# Pre-registration: external validation of the AURA fusion comparison on CheXpert

**Deposited** 19 August 2026, before any evidence extraction or model evaluation on CheXpert.
**Repository state at deposit:** branch `chore/review-remediation-and-quantum-evidence`, parent commit `282f8d0`.
**Author:** Sivaguru R. M.

This document exists because §6 of the manuscript states that the protocol rules P1–P6 were formulated during the same revision period as the experiments they govern, that no timestamped pre-registration is held, and that "for the external-validation work specified in §11.9 we will deposit a timestamped pre-registration before any run, which is the only way this claim can be made verifiable rather than asserted." This is that deposit. Its timestamp is the commit that introduces it; the run it governs is performed in a later commit.

---

## 1. Objective

To test whether the conclusions of §8.2 — that the four fusion backends are indistinguishable on accuracy, and that the quantum variational circuit does not outperform a converged multinomial logistic regression on negative log-likelihood — survive transfer to a cohort from a different institution.

MIMIC-CXR (Beth Israel Deaconess, Boston) and CheXpert (Stanford) differ in scanner mix, acquisition protocol, case mix, and label derivation. §11.9 identifies this as the single highest-value outstanding analysis.

## 2. Hypotheses, stated before the data are seen

**H1 (primary).** On the CheXpert test split, the four backends are not distinguishable on accuracy: no pairwise McNemar comparison reaches significance after Holm adjustment within the family.

**H2 (primary).** On the CheXpert test split, the quantum variational circuit does not beat the converged multinomial logistic regression on negative log-likelihood. Internally the ordering favours the logistic regression at Holm *p* = 0.036; we predict the same direction externally.

**H3 (secondary).** Every backend's accuracy falls relative to its MIMIC value, because all models are frozen and the cohort is out of distribution. We do not predict the magnitude.

**H4 (secondary).** The `prior_risk` channel is identically zero on CheXpert as it is on MIMIC, because it is populated from structured priors that neither corpus supplies. Confirming this establishes that the constant-channel confound of §5.1.3 is a property of the encoder rather than of MIMIC.

## 3. What is frozen and what is refitted

**Frozen — no refitting of any kind:**
- the DenseNet-121 chest model and its per-finding Platt calibration;
- the evidence encoder;
- all four fusion backends, including the logistic regression, which was fitted on the MIMIC training split only.

**Refitted on the CheXpert calibration split, and nothing else:**
- one temperature per backend, as required by P1.

Refitting temperature is required by P1 and is the minimum that makes a probabilistic comparison meaningful under shift. No backend's weights are updated. Should we later decide to refit weights, that is a different experiment and will be reported separately as such.

## 4. Data and sampling

- **Corpus.** CheXpert-v1.0-small, `train.csv`, 223,414 rows over 64,540 patients.
- **Inclusion.** Frontal views only. One study per patient — the highest-numbered study directory carrying a frontal view — to mirror the MIMIC protocol's one-study-per-patient rule and keep the cohort patient-disjoint.
- **Target size.** 8,000 patients scanned in file order from the start of `train.csv`, yielding up to 8,000 studies. Order is fixed by the file, not sampled, so there is no seed to choose and no opportunity to reselect.
- **Splits.** 40% calibration / 60% test, split by a fixed permutation at seed 7. Patient-disjoint by construction. The training split is empty: nothing is trained.
- **Expected power.** At n_test ≈ 4,800 the minimum detectable accuracy difference at 80% power is approximately 3.6 points, against 9.3 internally.

## 5. Label derivation, and one category that cannot be derived

The six-way target is reproduced from CheXpert's label columns using the identical weighted-score map and tie-break of `aura/mimic/labeling.py` (§5.1.2 of the manuscript):

| Concept in the map | CheXpert column |
|---|---|
| pneumothorax | Pneumothorax |
| nodule | Lung Lesion |
| edema | Edema |
| cardiomegaly | Cardiomegaly |
| effusion | Pleural Effusion |
| pneumonia | Pneumonia |
| consolidation | Consolidation |
| opacity | Lung Opacity |
| normal cue | No Finding |
| **hyperinflation** | **no equivalent column** |

**COPD is therefore not derivable on CheXpert**, because the map reaches it only through hyperinflation. The external cohort has five reachable categories, not six. The backends retain their six-way output head, so a COPD prediction on this cohort can only ever be an error; we will report how often each backend emits one, since that is itself a measure of transfer behaviour. Accuracy on the external cohort is therefore not directly comparable with the internal figure, and we will not compare them as if it were.

**Uncertainty policy.** CheXpert encodes 1.0 positive, 0.0 negative, −1.0 uncertain, blank not mentioned. The primary analysis treats uncertain as negative (U-Zeros). A U-Ones sensitivity analysis is pre-specified and will be reported whatever it shows.

## 6. Analysis plan

Fixed before the run:

- **Metrics.** Accuracy with Jeffreys intervals, negative log-likelihood, expected calibration error at 10 equal-width bins, macro one-vs-rest AUROC, and the majority-class floor.
- **Tests.** McNemar's exact test on paired accuracy; Wilcoxon signed-rank on paired per-study negative log-likelihood. Six pairwise comparisons among four backends.
- **Multiplicity.** Holm within each metric family of six, matching Table 7.
- **Resampling.** Study-level, which is the unit of independence here (one study per patient).
- **Secondary cohort.** The same analysis on a class-capped subsample mirroring the internal protocol's `per_class_cap`, reported alongside the natural-prevalence primary, so that a change in ordering can be attributed to prevalence rather than to site.

## 7. What would falsify the manuscript's conclusions

- **H1 falsified** if any pairwise accuracy comparison survives Holm on the external cohort. §8.2's indistinguishability claim would then be shown to be an artifact of internal power.
- **H2 falsified** if the quantum circuit beats the logistic regression on negative log-likelihood at Holm *p* < 0.05. This would directly contradict §8.2 and we would report it as such in the abstract, not as a subgroup finding.

We commit to reporting the outcome in the manuscript regardless of direction (P5), including the case where the external cohort favours the quantum arm.

## 8. Deviations

Any departure from this document will be recorded in the results section with its reason, and this file will not be edited after its deposit commit except to add a dated deviation note at the end.

### Deviations recorded after deposit

*(none at deposit)*
