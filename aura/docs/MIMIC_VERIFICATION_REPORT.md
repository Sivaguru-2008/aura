# MIMIC-CXR Dataset Verification Report

_Generated: 2026-07-18T12:46:58+00:00 • Root: `E:\AURA\datasets\simhadrisadaram\mimic-cxr-dataset\versions\2`_

## Path existence

| Path | Present |
|---|---|
| `root` | ✅ |
| `train_csv` | ✅ |
| `validate_csv` | ✅ |
| `images_root` | ✅ |

## `mimic_cxr_aug_train.csv`

- **Size:** 235.8 MB
- **Rows:** 64,586
- **Columns (10):** Unnamed: 0.1, Unnamed: 0, subject_id, image, view, AP, PA, Lateral, text, text_augment
- **Primary key:** `subject_id` → 64,586 unique (✅ unique=True, duplicates=0)
- **Foreign key (image → file):** 368,960 paths referenced; sampled 3,000, found 2,084 (69.5% ❌)

| Column | dtype-role | non-null | null | empty-list | items (total / max) |
|---|---|---|---|---|---|
| `Unnamed: 0.1` | scalar | 64,586 | 0 | — | — |
| `Unnamed: 0` | scalar | 64,586 | 0 | — | — |
| `subject_id` | scalar | 64,586 | 0 | — | — |
| `image` | list | 64,586 | 0 | 0 | 368,960 / 174 |
| `view` | list | 64,586 | 0 | 0 | 146,593 / 6 |
| `AP` | list | 64,586 | 0 | 31,612 | 143,556 / 167 |
| `PA` | list | 64,586 | 0 | 19,504 | 94,416 / 69 |
| `Lateral` | list | 64,586 | 0 | 20,060 | 81,319 / 71 |
| `text` | list | 64,586 | 0 | 0 | 222,758 / 158 |
| `text_augment` | list | 64,586 | 0 | 0 | 222,758 / 158 |

## `mimic_cxr_aug_validate.csv`

- **Size:** 1.9 MB
- **Rows:** 500
- **Columns (10):** Unnamed: 0.1, Unnamed: 0, subject_id, image, view, AP, PA, Lateral, text, text_augment
- **Primary key:** `subject_id` → 500 unique (✅ unique=True, duplicates=0)
- **Foreign key (image → file):** 2,991 paths referenced; sampled 2,991, found 2,099 (70.2% ❌)

| Column | dtype-role | non-null | null | empty-list | items (total / max) |
|---|---|---|---|---|---|
| `Unnamed: 0.1` | scalar | 500 | 0 | — | — |
| `Unnamed: 0` | scalar | 500 | 0 | — | — |
| `subject_id` | scalar | 500 | 0 | — | — |
| `image` | list | 500 | 0 | 0 | 2,991 / 114 |
| `view` | list | 500 | 0 | 0 | 1,134 / 5 |
| `AP` | list | 500 | 0 | 242 | 1,212 / 95 |
| `PA` | list | 500 | 0 | 156 | 747 / 18 |
| `Lateral` | list | 500 | 0 | 156 | 633 / 24 |
| `text` | list | 500 | 0 | 0 | 1,808 / 95 |
| `text_augment` | list | 500 | 0 | 0 | 1,808 / 95 |

## Cross-split leakage check

- **train ∩ validate patients:** 0 ✅

## Findings & implications

1. **Schema is clean & consistent** — both splits share the same 10 columns, `subject_id` is a unique primary key with **zero duplicates and zero nulls**.
2. **Partial image subset (data-quality risk).** 371,951 image paths are referenced but only ~69%/70% resolve to files on disk. Loaders MUST filter to existing images and drop studies/patients left with none.
3. **Two index-junk columns** (`Unnamed: 0.1`, `Unnamed: 0`) are pandas `to_csv` artifacts and must be dropped in cleaning (Step 3).
4. **`text` and `text_augment` are parallel** (identical item counts): `text` is the real radiology report, `text_augment` a paraphrase for augmentation.
5. **No test split exists** — only train + validate. Step 7 must carve a patient-disjoint test set from train (validate already shares 0 patients with train).
6. **No tabular EHR tables** (admissions/labevents/icustays/…): this is MIMIC-CXR, not MIMIC-IV. Loaders/features target images + reports; outcome-based ML tasks (mortality/sepsis/LOS) are not backed by this corpus.
