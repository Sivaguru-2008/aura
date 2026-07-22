# AURA — Model Card & Pipeline Specifications

This document describes the specifications and quantitative results of the primary convolutional neural network (CNN) and Platt calibration layers serving AURA's vision findings.

---

## 1. Vision Model Details

* **Name / Version**: `vision-cxr-densenet121-production`
* **Architecture**: DenseNet-121 (torchvision, pretrained on ImageNet).
  - The first convolution layer is adapted to **1-channel grayscale** by summing ImageNet channel weights.
  - The classifier head is replaced with a **7-way multi-label sigmoid classifier** mapping to clinical findings.
* **Weights**: Promoted to `artifacts/best_model.pt` (trained to Epoch 7, achieving **0.821 macro-AUROC** on the validation set).
* **Input Resolution**: $224 \times 224$ grayscale pixels.
* **Normalization**: Grayscale values normalized to ImageNet statistics (mean 0.449, std 0.226).
* **Output**: Independent sigmoid probabilities for 7 findings:
  1. Airspace opacity
  2. Consolidation
  3. Pleural effusion
  4. Cardiomegaly
  5. Nodule
  6. Pneumothorax
  7. Hyperinflation

---

## 2. Served Operating Points & Platt Calibration

To correct multi-label overconfidence, the vision model applies per-finding **Platt calibration** parameters fit on a held-out dataset ($n=2,099$). The calibrated probability $p$ is calculated from the raw logit $z$ as:

$$p = \sigma(a \cdot z + b)$$

Where $a$ and $b$ are fitted scaling parameters. These parameters are stored in `vision_serving_calibration.json` and are loaded at startup:

| Finding | Platt $a$ | Platt $b$ | Calibrated Present Threshold |
|---|---|---|---|
| **opacity** | 0.941 | -0.124 | 0.285 |
| **consolidation** | 0.812 | -0.405 | 0.221 |
| **pleural_effusion** | 1.102 | -0.198 | 0.290 |
| **cardiomegaly** | 0.954 | -0.312 | 0.250 |
| **nodule** | 0.652 | -1.104 | 0.132 |
| **pneumothorax** | 0.551 | -2.684 | 0.169 |
| **hyperinflation** | 0.884 | -0.554 | 0.204 |

*The `finding_present_threshold` indicates the probability cutoff above which a finding is considered clinically present in the report and dashboard.*

---

## 3. Quantitative Evaluation (Held-out Set, $n = 2,099$)

Evaluating the model on the official held-out validation split (`mimic_cxr_aug_validate.csv`) yields a **macro-AUROC of 0.821** (95% CI 0.811–0.832) and a **Brier score of 0.091**. Platt calibration reduces the mean Expected Calibration Error (ECE) from 0.145 to **0.023**.

| Finding | AUROC | AUPRC | Sens | Spec | F1 | ECE (Calibrated) | Support |
|---|---|---|---|---|---|---|---|
| **opacity** | 0.824 | 0.882 | 0.712 | 0.791 | 0.764 | 0.018 | 1430 |
| **consolidation** | 0.801 | 0.682 | 0.795 | 0.612 | 0.658 | 0.025 | 832 |
| **pleural_effusion** | 0.900 | 0.895 | 0.821 | 0.784 | 0.812 | 0.015 | 1169 |
| **cardiomegaly** | 0.852 | 0.742 | 0.794 | 0.720 | 0.755 | 0.020 | 872 |
| **nodule** | 0.729 | 0.442 | 0.612 | 0.730 | 0.512 | 0.032 | 542 |
| **pneumothorax** | 0.825 | 0.450 | 0.460 | 0.920 | 0.485 | 0.021 | 249 |
| **hyperinflation** | 0.818 | 0.584 | 0.684 | 0.810 | 0.621 | 0.028 | 415 |

---

## 4. Hardware Performance & Latency

Latency and throughput metrics were benchmarked on a standard test platform:
* **GPU Latency (RTX 5050 Mobile)**: 29 ms (single image).
* **CPU Latency (Intel i7, 8-cores)**: 83 ms (single image).
* **Throughput (GPU Batch 32)**: 618 images / second.
* **Peak GPU VRAM Utilization**: 693 MB.
