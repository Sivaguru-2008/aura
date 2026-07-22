# AURA Clinical Report — sample
_Generated 2026-07-22T12:38:45+00:00_

## Patient Summary
- **Study:** sample (CXR)
- **Demographics:** age unknown, sex unknown

## Vision Findings
| Finding | Probability | Present |
|---|---|---|
| Pleural effusion | 0.787 | ✓ |
| Cardiomegaly | 0.440 | ✓ |
| Airspace opacity | 0.264 |  |
| Consolidation | 0.114 |  |
| Hyperinflation | 0.015 |  |
| Pneumothorax | 0.014 |  |
| Pulmonary nodule | 0.013 |  |

## Confidence
- **Top diagnosis:** Congestive heart failure — 52% (95% CI 0.33–0.71)
- **Uncertainty:** epistemic 0.067, aleatoric 0.785 (deep_ensemble)

## Calibration
- **90% mondrian conformal set:** Pneumonia, Congestive heart failure, Suspicious pulmonary malignancy, Pneumothorax
- **Temperature:** 0.9386 · **reported ECE:** 0.0596
- **Out-of-distribution:** no (energy z=-0.3005)

## Differential Diagnosis
- **Congestive heart failure** — 52%; supported by pleural effusion
- **Suspicious pulmonary malignancy** — 19%
- **Pneumonia** — 15%
- **Pneumothorax** — 11%

## Evidence Used
Pleural effusion (p=0.79), Cardiomegaly (p=0.44)

## Evidence Missing
- opacity indeterminate (p=0.26)
- consolidation indeterminate (p=0.11)
- cardiomegaly indeterminate (p=0.44)
- 4 diagnoses remain in the 90% confidence set — discriminating evidence would narrow it

## Recommended Tests
- **Retrieve & compare prior films** (cost low, risk none, utility 0.0111) — Temporal stability changes malignancy likelihood markedly. Expected value of information 0.011 (loss units); reduces diagnostic uncertainty ~10%.
- **Order CT chest (low-dose)** (cost high, risk low, utility 0.0019) — CT characterizes nodules and consolidation the frontal film leaves ambiguous. Expected value of information 0.011 (loss units); reduces diagnostic uncertainty ~8%.
- **Acquire lateral chest view** (cost low, risk none, utility 0.0) — A lateral projection disambiguates airspace opacity from pleural fluid. Expected value of information 0.000 (loss units); reduces diagnostic uncertainty ~17%.
- **Order BNP + bedside echo** (cost medium, risk none, utility 0.0) — Confirms or excludes cardiac cause of an enlarged silhouette. Expected value of information 0.000 (loss units); reduces diagnostic uncertainty ~12%.

## Risk Level: **MODERATE**
Congestive heart failure at 52% calibrated confidence

## Clinical Impression
Impression: findings most consistent with congestive heart failure (calibrated probability 52%). 90% mondrian confidence set: Pneumonia, Congestive heart failure, Suspicious pulmonary malignancy, Pneumothorax.

## Limitations
- Automated decision support only — not a substitute for a radiologist's read.
- DenseNet-121 trained on MIMIC-CXR; performance may degrade on out-of-distribution equipment, projections, or populations.
- Ground-truth labels come from a rule-based report labeler (not the official CheXpert labeler). Validated against 66 hand-read reports: macro F1 0.89, Cohen's kappa 0.86 vs expert-convention reading.
- Independently cross-checked against torchxrayvision (separate labels) on 700 images: mean cross-model AUROC 0.78; nodule agreement is below chance and is flagged unreliable.
- Findings localize regions of interest; they are observations, not tissue diagnoses.

## Provenance
- **vision:** vision-cxr-densenet121-production
- **fusion:** quantum:fusion-vqc-v1
- **safety:** safety-v2
- **reasoning:** reasoning-v1
- **report:** structured+reasoning
- **Inference time:** 5832 ms