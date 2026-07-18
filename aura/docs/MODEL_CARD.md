# Model Card — AURA Vision (DenseNet-121 CXR)

## Model details
- **Name / version:** `vision-cxr-densenet121-production`
- **Architecture:** DenseNet-121 (torchvision, ImageNet-pretrained), first conv
  adapted to **1-channel grayscale** (RGB weights summed), classifier replaced with
  a **7-way multi-label** head (`ml/vision_cxr/model.py`).
- **Weights:** `artifacts/best_model.pt` (`epoch=1`, `arch=densenet121`,
  `best_metric≈0.696` macro-AUROC — consistent with the validation numbers below).
- **Input:** grayscale CXR resized to 224×224, channel-averaged ImageNet
  normalization (mean 0.449, std 0.226).
- **Output:** independent sigmoid probability per finding.
- **Findings (7):** opacity, consolidation, pleural_effusion, cardiomegaly, nodule,
  pneumothorax, hyperinflation.
- **Loads automatically:** `VisionEngine.load()` picks up `best_model.pt` and serves
  it behind the `score_findings` / `embedding` contract; the 1024-d penultimate
  feature is the evidence embedding.

## Intended use
- **Use:** decision-support triage/prioritization and region localization for adult
  chest radiographs; research and educational evaluation of an epistemic AI pipeline.
- **Not for:** autonomous diagnosis, pediatric imaging, non-CXR modalities, or any
  unsupervised clinical action. Downstream diagnosis fusion is a **prototype** (see
  Limitations).

## Training data
- **Corpus:** MIMIC-CXR (`mimic_cxr_aug_*` CSVs + `official_data_iccv_final/files`).
- **Labels:** derived from free-text radiology reports by a transparent, negation-
  aware, CheXpert-style rule labeler (`mimic/labeling.py`). Labels are patient-level
  and applied to that patient's images (`ml/vision_cxr/dataset.load_mimic_samples`).

## Evaluation data
- **Split:** the held-out MIMIC validation CSV (`mimic_cxr_aug_validate.csv`),
  **2,099 images** decoded from disk, unseen at training. Labels via the same
  report labeler (matching the training convention).

## Quantitative results (full validation set, n = 2,099)

**Macro (7 findings):** AUROC **0.702** (95% CI 0.688–0.716, 1000 bootstraps),
AUPRC 0.558, F1 0.547, sensitivity 0.584, specificity 0.719, Brier 0.202,
ECE 0.147 (→ **0.130** after per-finding temperature scaling).
**Micro:** F1 0.623, accuracy 0.704.

| finding | AUROC | AUPRC | sens | spec | F1 | ECE | support |
|---|---|---|---|---|---|---|---|
| opacity | 0.776 | 0.878 | 0.656 | 0.782 | 0.746 | 0.162 | 1430 |
| consolidation | 0.723 | 0.605 | 0.785 | 0.529 | 0.627 | 0.178 | 832 |
| pleural_effusion | 0.776 | 0.825 | 0.746 | 0.657 | 0.739 | 0.074 | 1169 |
| cardiomegaly | 0.743 | 0.632 | 0.771 | 0.620 | 0.669 | 0.127 | 872 |
| nodule | 0.485 | 0.259 | 0.292 | 0.713 | 0.276 | 0.206 | 542 |
| pneumothorax | 0.638 | 0.216 | 0.269 | 0.904 | 0.272 | 0.113 | 249 |
| hyperinflation | 0.773 | 0.492 | 0.571 | 0.826 | 0.502 | 0.166 | 415 |

Plots (ROC, PR, calibration, confidence histograms, confusion matrices) are in
`artifacts/evaluation/plots/`. Reproduce with `python aura_cli.py evaluate`.

## Calibration & uncertainty
- **Temperature scaling** (per finding) reduces mean ECE 0.145 → 0.130. Fitted T
  ranges 0.80–2.74 (consolidation/nodule most over-confident).
- **Conformal prediction:** empirical coverage **0.906** at target 0.90 (mean set
  size 1.53) — the coverage guarantee holds.
- **Monte-Carlo dropout:** the DenseNet-121 head has **no dropout layers**, so MC-
  dropout is degenerate; **test-time augmentation** is used as the epistemic proxy
  (mean per-finding std ≈ 0.040).
- **Deep ensembles:** optional; not trained (single checkpoint).

## Performance (RTX 5050 Laptop, torch 2.11 + CUDA 12.8)
Single-image GPU latency 29 ms (34 img/s); CPU 83 ms; peak batch throughput
618 img/s (batch 32); peak GPU memory 693 MB. See `PERFORMANCE_REPORT.md`.

## Limitations & ethical considerations
- **1-epoch checkpoint:** macro AUROC ≈ 0.70 — useful signal for the strong findings
  (opacity/effusion/cardiomegaly/hyperinflation, AUROC ≈ 0.72–0.78) but **weak on
  nodule (0.48, near chance) and pneumothorax (0.64)**, both low-prevalence. Do not
  rely on the model to exclude nodules or pneumothorax.
- **Label noise:** rule-based report labels are imperfect and patient-level, which
  caps achievable AUROC and inflates apparent errors on rare findings.
- **Downstream diagnosis fusion** (`services/fusion`) was trained on *synthetic*
  evidence and does not generalize to real multi-finding images (it can emit a
  confident but wrong diagnosis). The **vision findings**, not the fused diagnosis,
  are the validated output. Retraining fusion on real evidence is future work.
- **Distribution shift:** trained on MIMIC (Beth Israel); expect degradation on other
  scanners, projections (AP vs PA vs lateral), and populations. The safety engine
  flags out-of-distribution inputs by energy score, but this is not a guarantee.
- **Fairness:** no subgroup (age/sex/device) performance audit has been run; a
  bias evaluation is required before any clinical use.

## Maintenance
- Retrain / longer training via `python aura_cli.py train-cnn densenet121` with
  `AURA_CNN_MANIFEST`. Re-validate with `python aura_cli.py evaluate` and refresh
  this card from `artifacts/evaluation/metrics.json`.
