# Data provenance registry

**Status:** 2026-09-14. An engineering record, **not legal advice**.

Three places hold the provenance story, each with one job:

| Where | What it holds |
|---|---|
| `aura/common/data_governance.py` (`KNOWN_PROVENANCE`) | The machine-readable record: each trained or fitted artifact, its source and its licence status. The commercial gate reads it. |
| [`DATA_LICENSES.md`](DATA_LICENSES.md) | The dataset-by-dataset licence review, and the audit of what is on disk. |
| This file | How the gate works, where things stand, and how to retrain for commercial use. |

## The commercial gate

`AURA_COMMERCIAL_MODE=1` declares a commercial deployment. In that mode the gateway (at
startup) and `deploy/preflight.py` (step 2) refuse to start while any trained or fitted
file under `aura/artifacts/` lacks a `COMMERCIAL_AUTHORIZED` record. The gate fails
closed: a weight file with no record at all blocks too, so a new checkpoint cannot slip
through by being unknown.

Every artifact is `RESEARCH_ONLY` today, so commercial mode cannot start. That is the
correct answer until the licence question is resolved. Research and pilot deployments run
with `AURA_COMMERCIAL_MODE=0`, the default, and preflight states on every boot that the
weights are research-licensed.

## Where things stand

| Artifact | Trained on | Terms | Commercial use |
|---|---|---|---|
| `best_model.pt` (served chest DenseNet-121) | MIMIC-CXR, from a third-party Kaggle re-upload; no longer on disk | PhysioNet Credentialed Health Data License 1.5.0 | Not granted |
| `vision_serving_calibration.json`, `fusion_*.npz`, `safety*.npz` (served) | MIMIC-CXR: 2,099 validation images, ≤540 fusion studies, 446 calibration studies | Same | Not granted |
| `brain/checkpoints/best_brain_model.pt` and the brain calibration (served) | BraTS 2020, 369 subjects | BraTS 2020 data terms (research) | Not granted |
| `vision_cnn_chexpert*.pt`, `chexpert_evidence.npz` (not served) | CheXpert | Stanford research use agreement | Excluded, and kept out of Docker images |

The chest model also starts from ImageNet-pretrained weights. ImageNet's terms of access
limit the images to non-commercial research. Whether that reaches pretrained weights is
unsettled, so include it in the legal review.

## Retraining for commercial use

1. **Data under an agreement that names commercial product development.** The draft in
   [`COMMERCIAL_DATA_USE_AGREEMENT.md`](COMMERCIAL_DATA_USE_AGREEMENT.md) is a starting point
   for counsel, not a signable contract.
2. **De-identification** certified by the provider (Safe Harbor or Expert Determination),
   including text burned into the pixels.
3. **Radiologist labels** per [`RADIOLOGIST_ANNOTATION_PROTOCOL.md`](RADIOLOGIST_ANNOTATION_PROTOCOL.md),
   with a held-out test set that no training, threshold choice or calibration step reads.
4. **No initialisation from MIMIC-CXR- or BraTS-trained weights.** Use random
   initialisation, or a backbone whose weights carry terms counsel has cleared.
5. **A manifest for every training run:** agreement id, patient and study counts, scanner
   vendors, de-identification certificate, split seeds, and results on the held-out set.
6. **Record the licence last.** Add the artifact to `KNOWN_PROVENANCE` as
   `COMMERCIAL_AUTHORIZED` only after counsel confirms the terms. Until then the gate keeps
   blocking.
