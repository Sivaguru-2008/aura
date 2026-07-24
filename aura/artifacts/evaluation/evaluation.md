# AURA Clinical Model Validation Report

This report summarizes the measured clinical diagnostic performance and computational efficiency of AURA's vision backbones, evidence fusion networks, and safety engines.

- **Generated at:** 2026-07-24T04:59:52Z
- **Environment:** Windows-11-10.0.26200-SP0
- **Framework:** PyTorch 2.11.0+cu128 (CUDA: True)

---

## 1. Chest Radiograph Model (DenseNet-121)

Evaluated on **602** validation images from the MIMIC-CXR dataset.

### Clinical Metrics
| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **AUROC** | 0.8095 | Overall diagnostic discrimination |
| **AUPRC** | 0.3188 | Precision-recall area (handling label imbalance) |
| **F1 Score** | 0.3330 | Harmonic mean of precision and recall |
| **Sensitivity (Recall)** | 0.7243 | True positive rate (clinical safety floor) |
| **Specificity** | 0.7481 | True negative rate (avoiding alarm fatigue) |
| **Precision** | 0.2391 | Positive predictive value |
| **ECE** | 0.2087 | Expected Calibration Error (calibration honesty) |
| **Brier Score** | 0.1556 | Overall posterior forecast quality |

### Computational Performance (CPU)
- **Model Size:** 27.12 MB
- **Inference Latency:** 47.88 ms
- **Throughput:** 20.88 images/sec

---

## 2. Brain MRI Model (Multi-Task ResU-Net)

Evaluated on **7531** volumetric slices from the BraTS2020 dataset.

### Clinical Segmentation Metrics
| Region | Dice Similarity Coefficient | Hausdorff95 Distance (px) |
| :--- | :---: | :---: |
| **Whole Tumor (WT)** | 0.91498 | 7.08 |
| **Tumor Core (TC)** | 0.84561 | 6.15 |
| **Enhancing Tumor (ET)** | 0.83486 | 4.53 |
| **Composite Mean** | 0.86515 | — |

- **Tumor Presence AUROC:** 0.97743 (tumor presence classification head)

### Computational Performance (CPU)
- **Model Size:** 86.25 MB
- **Throughput:** 12.16 slices/sec
- **Study-Level Latency (155 slices):** 12747.84 ms (~12.75 sec)

---

## 3. Evidence Fusion & Conformal Safety

### Conformal Prediction Sets
Conformal sets guarantee that the true clinical label is included in the output set with a user-specified probability (90% target coverage).
- **Quantum Fusion Coverage:** 92.8% (Average set size: 3.46 diagnoses)
- **Classical Fusion Coverage:** 91.3% (Average set size: 3.26 diagnoses)

### Out-of-Distribution (OOD) Detection
- **Method:** Energy-based anomaly score (Z-Score on logits)
- **FPR at 95% TPR:** 4.5% (effectively flags non-chest films and corrupt studies while preserving clean diagnostic intake)
