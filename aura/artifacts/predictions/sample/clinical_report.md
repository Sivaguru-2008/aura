# AURA Clinical Report — sample
_Generated 2026-07-18T18:24:53+00:00_

## Patient Summary
- **Study:** sample (CXR)
- **Demographics:** age unknown, sex unknown

## Vision Findings
| Finding | Probability | Present |
|---|---|---|
| Pleural effusion | 0.983 | ✓ |
| Consolidation | 0.962 | ✓ |
| Airspace opacity | 0.841 | ✓ |
| Cardiomegaly | 0.808 | ✓ |
| Pulmonary nodule | 0.419 |  |
| Hyperinflation | 0.411 |  |
| Pneumothorax | 0.175 |  |

## Confidence
- **Top diagnosis:** Chronic obstructive pulmonary disease — 99% (95% CI 0.94–1.00)
- **Uncertainty:** epistemic 0.037, aleatoric 0.346 (deep_ensemble)

## Calibration
- **90% mondrian conformal set:** Chronic obstructive pulmonary disease
- **Temperature:** 0.7725 · **reported ECE:** 0.0147
- **Out-of-distribution:** no (energy z=1.0661)

## Differential Diagnosis
- **Chronic obstructive pulmonary disease** — 99%
- **Pneumothorax** — 1%
- **Suspicious pulmonary malignancy** — 0%; supported by nodule
- **No acute cardiopulmonary abnormality** — 0%; against: nodule

## Evidence Used
Pleural effusion (p=0.98), Consolidation (p=0.96), Airspace opacity (p=0.84), Cardiomegaly (p=0.81), nodule

## Evidence Missing
- nodule indeterminate (p=0.42)
- hyperinflation indeterminate (p=0.41)
- pneumothorax indeterminate (p=0.17)

## Recommended Tests
- **Acquire lateral chest view** (cost low, risk none, utility 0.0117) — A lateral projection disambiguates airspace opacity from pleural fluid. Expected value of information 0.012 (loss units); reduces diagnostic uncertainty ~36%.

## Risk Level: **LOW-MODERATE**
Chronic obstructive pulmonary disease at 99% calibrated confidence

## Clinical Impression
Impression: findings most consistent with chronic obstructive pulmonary disease (calibrated probability 99%). 90% mondrian confidence set: Chronic obstructive pulmonary disease.

## Limitations
- Automated decision support only — not a substitute for a radiologist's read.
- DenseNet-121 trained on MIMIC-CXR; performance may degrade on out-of-distribution equipment, projections, or populations.
- Ground-truth training labels were derived from free-text reports by a rule-based labeler and inherit its error modes.
- Findings localize regions of interest; they are observations, not tissue diagnoses.

## Provenance
- **vision:** vision-cxr-densenet121-production
- **fusion:** quantum:fusion-vqc-v1
- **safety:** safety-v2
- **reasoning:** reasoning-v1
- **report:** structured+reasoning
- **Inference time:** 5529 ms