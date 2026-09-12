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

## Sources

- MIMIC-CXR 2.1.0 on PhysioNet: <https://physionet.org/content/mimic-cxr/2.1.0/>
- PhysioNet Credentialed Health Data License 1.5.0:
  <https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/>
- BraTS 2020 data page (UPenn CBICA): <https://www.med.upenn.edu/cbica/brats2020/data.html>
- CheXpert terms as described in the CheXpert Plus paper (arXiv:2405.19538) and the CheXpert
  datasheet (arXiv:2105.03020). The agreement itself is presented at download on the Stanford
  AIMI site; read that text, not these summaries, before relying on it.
