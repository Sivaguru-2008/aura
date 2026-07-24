# Model Card: Chest Radiograph Vision Model (DenseNet-121)

This model card describes AURA's vision backbone for chest radiograph analysis.

---

## 1. Model Details
- **Architecture:** DenseNet-121 backbone, custom single-channel input head, multi-label classification output head.
- **Model Version:** `vision-cxr-region-v1`
- **Output Labels:** Opacity, Consolidation, Pleural Effusion, Cardiomegaly, Nodule, Pneumothorax, Hyperinflation.
- **Size on Disk:** 27.12 MB (28,436,776 bytes)
- **Framework:** PyTorch (reloads weights from `best_model.pt`)
- **Developer:** AURA AI Team

---

## 2. Intended Use
- **Intended User:** Clinical radiologists and emergency department physicians.
- **Primary Application:** Fast detection and triage of acute thoracic findings in bedside chest X-rays.
- **Out of Scope:** Mammograms, CT scout views, pediatric radiography, and skeletal pathology scoring.

---

## 3. Training & Validation Data
- **Dataset:** MIMIC-CXR validation split.
- **Image Input Format:** Single film (2D grayscale, resized to 224x224, normalized to range [0, 1]).
- **Validation Dataset Size:** 602 images (fully labeled by clinical report rules).

---

## 4. Measured Performance
These metrics reflect real validation splits and local CPU benchmark measurements:

| Metric | Macro Average | Opacity | Effusion | Cardiomegaly |
| :--- | :---: | :---: | :---: | :---: |
| **AUROC** | 0.8095 | 0.7282 | 0.8797 | 0.8250 |
| **AUPRC** | 0.3188 | 0.4774 | 0.7303 | 0.5512 |
| **F1 Score** | 0.3330 | 0.5544 | 0.6572 | 0.4632 |
| **Sensitivity** | 0.7243 | 0.7429 | 0.8797 | 0.7895 |
| **Specificity** | 0.7481 | 0.6159 | 0.7162 | 0.7318 |

- **Inference Latency (CPU):** ~47.88 ms
- **Throughput (CPU):** ~20.88 images/sec
- **Throughput (GPU Batch=64):** ~593.31 images/sec

---

## 5. Calibration & Safety
- **Calibration Method:** Platt temperature scaling (temperature = 0.8784).
- **ECE (Expected Calibration Error):** 0.2087
- **Brier Score:** 0.1556
- **OOD Detection:** Energy score threshold (FPR: 4.5% at 95% TPR).
