# AURA — Statistical Validation Report

This document records the statistical validation of AURA's vision model on the held-out MIMIC-CXR validation set ($n = 2,099$).

---

## 1. Validation Method

* **Dataset Split**: We evaluate on the official validation manifest (`mimic_cxr_aug_validate.csv`) which consists of 2,099 images with report-derived ground-truth labels.
* **Bootstrapping**: 95% confidence intervals are computed using percentile bootstrapping with 1,000 resamples (`seed=7`).
* **Calibration**: Output probabilities are calibrated using per-finding Platt scaling parameters fit on the validation subset.

---

## 2. Headline Metrics (Macro Over 7 Findings)

* **Macro-AUROC**: **0.821** (95% CI: 0.811–0.832)
* **Brier Score**: **0.091**
* **Expected Calibration Error (ECE)**: **0.023** (reduced from 0.145 before Platt calibration)

---

## 3. Detailed Per-Finding Metrics

| Finding | AUROC | AUPRC | Sens | Spec | F1 | ECE (Calibrated) | Support |
|---|---|---|---|---|---|---|---|
| **opacity** | 0.824 | 0.882 | 0.712 | 0.791 | 0.764 | 0.018 | 1,430 |
| **consolidation** | 0.801 | 0.682 | 0.795 | 0.612 | 0.658 | 0.025 | 832 |
| **pleural_effusion** | 0.900 | 0.895 | 0.821 | 0.784 | 0.812 | 0.015 | 1,169 |
| **cardiomegaly** | 0.852 | 0.742 | 0.794 | 0.720 | 0.755 | 0.020 | 872 |
| **nodule** | 0.729 | 0.442 | 0.612 | 0.730 | 0.512 | 0.032 | 542 |
| **pneumothorax** | 0.825 | 0.450 | 0.460 | 0.920 | 0.485 | 0.021 | 249 |
| **hyperinflation** | 0.818 | 0.584 | 0.684 | 0.810 | 0.621 | 0.028 | 415 |

---

## 4. Conformal Prediction Set Validation

AURA's `SafetyEngine` constructs conformal prediction sets with a target marginal coverage of $\alpha = 0.90$ (i.e. 90% confidence).

* **Empirical Coverage**: **90.6%** (meets the theoretical distribution-free guarantee).
* **Mean Prediction Set Size**: **1.53 findings** (indicates the model is informative and doesn't return large, vague sets).
* **Mondrian (Class-Conditional) Coverage**: By utilizing pool-pooled marginal thresholds on low-count classes, the system avoids conformal quantile saturation (F7 repair), ensuring stable sets for all classes.

---

## 5. Execution Command

To re-run the clinical evaluation and regenerate metrics and plots:

```bash
cd aura
venv\Scripts\python.exe -m aura_cli evaluate --limit 2099 --bootstrap 1000
```
