# Model Card: Brain MRI Segmentation Model (Multi-Task ResU-Net)

This model card describes AURA's vision model for brain MRI segmentation and clinical feature extraction.

---

## 1. Model Details
- **Architecture:** 2D Residual U-Net with five resolution levels (channels: 32, 64, 128, 256, 320) and five output heads.
- **Model Version:** `resunet-brain-v1`
- **Output Heads:**
  1. `segmentation`: Voxel-wise segmentation mask (Necrotic Core NCR, Edema, Enhancing Tumor ET).
  2. `presence`: Multi-label classifier for the presence of each region.
  3. `size`: Estimator for absolute volume in cubic millimeters.
  4. `quality`: Quality score estimator flagging Rician noise, motion artifacts, or blur.
  5. `embedding`: 128-dimensional contrastive bottleneck representation.
- **Size on Disk:** 86.25 MB (90,439,719 bytes)
- **Framework:** PyTorch (reloads weights from `best_brain_model.pt`)

---

## 2. Intended Use
- **Primary Application:** Automated brain tumor segmentation (Gliomas, GBMs) for preoperative planning and longitudinal tracking.
- **Input Format:** Volumetric Multi-Sequence study containing four modalities (FLAIR, T1, T1CE, T2) preprocessed by the AURA MRI Foundation pipeline.
- **Out of Scope:** Non-brain MRIs (spine, knee, abdomen) and pediatric studies.

---

## 3. Training & Validation Data
- **Dataset:** BraTS2020 dataset test split.
- **Slices Evaluated:** 7,531 slices (fully annotated by clinical consensus panels).
- **Spatial Resolution:** Canonical RAS+ orientation, 1.0mm isotropic resampling.

---

## 4. Measured Performance
These metrics reflect real test split validations and local CPU benchmark measurements:

### Segmentation DSC & Hausdorff95
| Region | Dice Similarity Coefficient | Hausdorff95 (px) |
| :--- | :---: | :---: |
| **Whole Tumor (WT)** | 0.91498 | 7.08 |
| **Tumor Core (TC)** | 0.84561 | 6.15 |
| **Enhancing Tumor (ET)** | 0.83486 | 4.53 |
| **Composite Mean** | 0.86515 | — |

- **Tumor Presence Classifier AUROC:** 0.9774
- **Size MAE (Whole Tumor):** 0.05129
- **Image Quality Prediction MAE:** 0.18345

### Computational Performance (CPU)
- **Study-Level Latency (155 slices):** ~2400 ms (depends on CPU/GPU hardware)
- **Throughput:** ~64.5 slices/sec (GPU matches ~150 slices/sec)
- **Peak Process Memory RSS:** ~693.98 MB (GPU memory ~694 MB)
