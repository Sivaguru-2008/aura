# AURA — Datasets & Ingestion Specifications

This document describes the structure, labeling, splits, and loading policies of AURA's primary training and validation dataset: **MIMIC-CXR**.

---

## 1. MIMIC-CXR Dataset Structure

AURA is trained and validated on a subset of the **MIMIC-CXR** (Medical Information Mart for Intensive Care, Chest X-Ray) database. The local dataset is stored at:
`E:\AURA\datasets`

### File Schema & Splits
The dataset is cataloged into train and validation manifest CSVs (`mimic_cxr_aug_train.csv` and `mimic_cxr_aug_validate.csv`).

| Parameter | Train Split | Validation Split |
|---|---|---|
| **File Size** | 235.8 MB | 1.9 MB |
| **Unique Patients (`subject_id`)** | 64,586 | 500 |
| **Referenced Images** | 368,960 | 2,991 |
| **Actually Present on Disk** | 256,427 (69.5%) | 2,099 (70.2%) |
| **Cross-Split Patient Leakage** | 0 | 0 |

### CSV Columns (10)
1. `subject_id`: Primary key representing the patient.
2. `image`: A list of file paths to the patient's radiographs.
3. `view`: Associated projection types (e.g. AP, PA, Lateral).
4. `AP`, `PA`, `Lateral`: Lists of files matching each projection.
5. `text`: The raw text of the clinician's radiology report.
6. `text_augment`: Paraphrased reports used for training augmentation.
7. `Unnamed: 0.1`, `Unnamed: 0`: Legacy index columns (ignored during loading).

---

## 2. Ingestion & Data Quality Policies

* **Path Verification**: Because only ~70% of the referenced images are present in the local database slice, loaders are configured to verify image paths before executing training or validation passes, dropping empty patients/studies.
* **Greyscale Conversion**: All images are decoded using OpenCV or PyDicom, down-scaled, and area-averaged to $224 \times 224$ pixels.
* **No EHR Multimodal Data**: The local MIMIC-CXR database does not contain patient admissions or laboratory tables (MIMIC-IV). Thus, multimodal labs/symptoms are generated synthetically or simulated in clinical console demos.

---

## 3. Labeling Rules (CheXpert Parser)

Training labels are extracted from the free-text radiology reports (`text` column) using a negation-aware, rule-based CheXpert-style labeler (`mimic/labeling.py`).
* **Findings (7)**: Cardiomegaly, pleural effusion, opacity, consolidation, nodule, pneumothorax, hyperinflation.
* **Diagnoses (6)**: Pneumonia, heart failure, COPD, pneumothorax, malignancy, normal.
* **Labeling Rule Logic**: Negations (e.g. "no evidence of pneumothorax") and uncertain mentions are filtered using regex boundaries to assign binary target values (`0` or `1`).
