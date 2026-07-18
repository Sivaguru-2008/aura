# AURA — Evaluation Report

**Model:** `vision-cxr-densenet121-production` (`artifacts/best_model.pt`)
**Data:** MIMIC-CXR held-out validation split — **2,099 images**, unseen at training.
**Command:** `python aura_cli.py evaluate` (regenerates everything below).
**Artifacts:** `artifacts/evaluation/` — `metrics.json`, `EVALUATION_SUMMARY.md`,
`plots/{roc_curves,pr_curves,calibration,confidence_histograms,confusion_matrices}.png`.

---

## 1. Method

- Labels come from the free-text report labeler (`mimic/labeling.py`) via
  `ml/vision_cxr/dataset.load_mimic_samples` — the **same convention used at
  training**, so metrics measure the model, not a label mismatch.
- The DenseNet is run over the whole split in GPU batches (`ml/evaluation/clinical_eval.py`).
- The finding task is **multi-label**; every metric is computed per finding
  (one-vs-rest) and aggregated macro / micro.
- 95% confidence intervals are percentile bootstraps (1,000 resamples).

## 2. Headline metrics (macro over 7 findings)

| AUROC | AUPRC | Accuracy | Precision | Recall/Sens | Specificity | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| **0.702** | 0.558 | 0.704 | 0.528 | 0.584 | 0.719 | 0.547 | 0.202 | 0.147 |

- **Macro AUROC 95% CI:** [0.688, 0.716] (1000 bootstraps)
- **Macro AUPRC 95% CI:** [0.540, 0.577]
- **Micro:** F1 0.623 · accuracy 0.704 · Brier 0.202

This matches the checkpoint's stored `best_metric ≈ 0.696` — the model is a genuine
1-epoch DenseNet with real but modest skill.

## 3. Per-finding metrics

| finding | AUROC | AUPRC | sens | spec | precision | F1 | Brier | ECE | support |
|---|---|---|---|---|---|---|---|---|---|
| opacity | 0.776 | 0.878 | 0.656 | 0.782 | — | 0.746 | — | 0.162 | 1430 |
| consolidation | 0.723 | 0.605 | 0.785 | 0.529 | — | 0.627 | — | 0.178 | 832 |
| pleural_effusion | 0.776 | 0.825 | 0.746 | 0.657 | — | 0.739 | — | 0.074 | 1169 |
| cardiomegaly | 0.743 | 0.632 | 0.771 | 0.620 | — | 0.669 | — | 0.127 | 872 |
| nodule | 0.485 | 0.259 | 0.292 | 0.713 | — | 0.276 | 0.206 | 0.206 | 542 |
| pneumothorax | 0.638 | 0.216 | 0.269 | 0.904 | — | 0.272 | — | 0.113 | 249 |
| hyperinflation | 0.773 | 0.492 | 0.571 | 0.826 | — | 0.502 | — | 0.166 | 415 |

(Full precision/NPV/confusion numbers are in `metrics.json` per finding.)

## 4. Reading the results

**Strong findings** (AUROC 0.72–0.78): opacity, pleural effusion, cardiomegaly,
consolidation, hyperinflation — these are common and visually salient, and the model
ranks them well. `opacity` and `pleural_effusion` also have high AUPRC (0.88 / 0.83).

**Weak findings:**
- **nodule — AUROC 0.485 (≈ chance).** Small, low-contrast, low-prevalence (26%);
  the 1-epoch model has not learned it. **Do not use the model to rule nodules in/out.**
- **pneumothorax — AUROC 0.638, sensitivity 0.27.** High specificity (0.90) but it
  misses most positives; only 249 positives in the split.

**Calibration:** overall ECE 0.147; effusion is best-calibrated (0.074), nodule worst
(0.206). Per-finding temperature scaling lowers mean ECE to 0.130 (see
`CALIBRATION_REPORT` section / `artifacts/calibration/`).

## 5. Confusion matrices

Per-finding 2×2 confusion matrices at threshold 0.5 are rendered in
`artifacts/evaluation/plots/confusion_matrices.png` and stored numerically under
`per_label[*].confusion_matrix` in `metrics.json`.

## 6. Plots

| file | content |
|---|---|
| `plots/roc_curves.png` | ROC per finding with AUROC in the legend |
| `plots/pr_curves.png` | Precision–Recall per finding with AUPRC |
| `plots/calibration.png` | reliability curve per finding |
| `plots/confidence_histograms.png` | predicted-probability histograms |
| `plots/confusion_matrices.png` | 2×2 confusion per finding @0.5 |

## 7. Reproducibility

```bash
python aura_cli.py evaluate                 # full split, 1000-bootstrap CIs, all plots
python aura_cli.py evaluate --limit 100     # quick subset
python aura_cli.py evaluate --calibrate     # also run the calibration suite
```

Deterministic given the fixed checkpoint and the frozen validation CSV; bootstrap
CIs use `seed=7`.

## 8. Caveats

- Metrics are bounded above by **report-derived label quality** (rule labeler,
  patient-level assignment).
- These numbers characterize the **vision findings only**. The fused *diagnosis*
  posterior (`services/fusion`) is a synthetic-trained prototype and is evaluated
  separately by `python aura_cli.py bench` (synthetic data); it should not be read as
  a real-data diagnostic metric.
