# Data licences and what they mean for commercial use

**Status:** 2026-09-12. This is an engineering inventory, **not legal advice**. Before AURA is
sold, licensed, or used to deliver a paid service, a lawyer has to read the actual agreement
for every dataset below and confirm the conclusions here.

## The short version

Every model AURA serves was trained on a dataset released for **research**. None of these
releases grants a commercial licence on its face. Until that changes, the served weights are
research artifacts. There are two ways to change it:

1. **Get written permission.** Ask each data owner for commercial-use terms covering the
   trained weights.
2. **Retrain on data you are licensed to use commercially.** Usually that means a partner
   hospital's own images under a data-use agreement that names commercial product
   development, or a dataset sold with a commercial licence.

The on-premises product design in `docs/HOSPITAL_IT_INSTALL.md` supports option 2: the
software runs inside the hospital, on the hospital's data, and nothing leaves the building.

## Dataset by dataset

| Dataset | Governing terms | Commercial use | What in this repo derives from it |
|---|---|---|---|
| **MIMIC-CXR** (Beth Israel Deaconess, via PhysioNet) | PhysioNet Credentialed Health Data License 1.5.0, plus the matching data use agreement. Requires credentialing and CITI training | The licence limits use to "the sole purpose of lawful use in scientific research and no other". It does not address models trained on the data. **Get legal review and ask MIT-LCP before any commercial use** | **Served:** `best_model.pt` (chest DenseNet-121), `vision_serving_calibration.json`, `fusion_*.npz`, `safety*.npz`, `conformal_mondrian.npy` |
| **BraTS 2020** (UPenn CBICA; source scans include TCIA collections) | The challenge data page grants use "in your own research" and requires citing three core papers. It gives no commercial terms, and calls MLPerf benchmark submissions non-commercial use | Not granted. **Ask the organisers, and check the terms of the underlying TCIA collections** | **Served:** `brain/checkpoints/best_brain_model.pt`, `brain/presence_calibration.json`, `brain/qkl_classifier.npz`, `brain/embeddings/latest.npz` |
| **CheXpert** (Stanford) | Stanford University Dataset Research Use Agreement. Secondary sources describe it as non-commercial research only, with no redistribution | **Excluded** by the agreement as described. Treat CheXpert-trained weights as unsellable | **Not served.** `vision_cnn_chexpert.pt` and `vision_cnn_chexpert_last.pt` (trained weights), `chexpert_evidence.npz` (derived evaluation data). All three are excluded from Docker images by `.dockerignore`. `chexpert_external.json` (aggregate metrics only) stays |
| **torchxrayvision `densenet121-res224-mimic_ch`** | Model weights trained on MIMIC-CXR; the MIMIC terms flow through | Same as MIMIC-CXR | Used only for the label cross-check (`artifacts/kappa_crosscheck.json`), never served |

## What this changes in practice

- **External validation on CheXpert is fine.** It is research use. Publishing aggregate
  metrics from it is what the dataset exists for.
- **Shipping CheXpert-trained weights is not.** They are excluded from every Docker image,
  and no serving code loads them (checked 2026-09-12: no `.py` references
  `vision_cnn_chexpert`).
- **Selling the current chest or brain model requires one of the two paths above.** A pilot
  under a research agreement with a partner hospital (IRB/ethics approval, research
  purpose, no fee) is the normal way to produce evidence while that is resolved.
- **Record provenance for every new training set**: source, agreement, and whether it
  permits commercial use. Put it in the model card, so a future buyer's diligence can
  read it in one place.

## Dataset provenance audit (2026-09-13)

What is on disk under `E:\AURA`, checked against what each served model was trained on.

| Folder | On disk | Used by a served model | How much |
|---|---|---|---|
| MIMIC-CXR (`datasets\simhadrisadaram\mimic-cxr-dataset\versions\2`, the default in `aura/mimic/config.py`) | **Missing.** The folder no longer exists and was not found elsewhere. Its path layout is a Kaggle download, i.e. a third-party re-upload rather than the PhysioNet release | Chest model, fusion heads, calibration | Chest DenseNet-121: about 259,000 training images, everything present on 2026-07-20 (the index listed about 369,000), plus 2,099 validation images. The 2,099-image set selected the best epoch (checkpoint `best_metric` = the headline 0.821) and is what the vision calibration reads, so the chest model has no untouched test set. Fusion: at most 540 studies. Safety calibration: 446 studies. |
| BraTS 2020 (`datasets\brats2020\...\content\data`) | 369 subjects, 57,195 slices | Brain model | All 369 subjects: 258 train, 55 validation, 56 test (`artifacts/brain/reports/model_card.json`). The served QKL grade head used only the 55 validation subjects. |
| `datasets\brats2021_hdf5` | 185 volumes | None | **Not what its name says, and its labels are untraceable.** Its "BraTS20_Training_…" IDs are volume numbers running to 1161, but BraTS 2020 stops at 369. Where a name happens to exist in BraTS 2020 (47 cases), the grade agrees 32 times, about chance. It feeds the current brain cache (`artifacts/brain/cache/manifest.json`) and the QKL "full coverage" study, whose result is therefore **withdrawn**. |
| BraTS 2021 raw (`E:\AURA\BraTS2021`) | 454 of the 1,251 released subjects | None | No code reads it; two subjects became demo uploads. |
| CheXpert-v1.0-small (stored twice, byte-identical) | 223,414 images, 64,540 patients | None | 8,000 patients for the preregistered external validation; one unserved experiment. |
| MIMIC-IV (`datasets\mimiciv`) | Empty folders | None | — |

Before any retraining or new published claim: restore MIMIC-CXR (or replace it with data
licensed for the purpose), set aside a chest test set that nothing selects on, and either
trace the `brats2021_hdf5` labels to an official source or rebuild the brain cache from
`datasets\brats2020`.

## Sources

- MIMIC-CXR 2.1.0 on PhysioNet: <https://physionet.org/content/mimic-cxr/2.1.0/>
- PhysioNet Credentialed Health Data License 1.5.0:
  <https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/>
- BraTS 2020 data page (UPenn CBICA): <https://www.med.upenn.edu/cbica/brats2020/data.html>
- CheXpert terms as described in the CheXpert Plus paper (arXiv:2405.19538) and the CheXpert
  datasheet (arXiv:2105.03020). The agreement itself is presented at download on the Stanford
  AIMI site; read that text, not these summaries, before relying on it.
