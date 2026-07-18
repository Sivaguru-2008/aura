# MIMIC-CXR Validation — Inference Metrics

- **Model:** `E:\AURA\aura\aura\artifacts\best_model.pt`
- **Images:** 2099
- **Inference:** 9.476s (221.51 img/s)

## Headline (macro over 7 findings)

| AUROC | AUPRC | F1 | Sens | Spec | Prec | Brier | ECE |
|---|---|---|---|---|---|---|---|
| 0.7019 | 0.5582 | 0.5472 | 0.5841 | 0.7187 | 0.5276 | 0.2019 | 0.1465 |

- **Macro AUROC 95% CI:** [0.6883, 0.7161] (1000 bootstraps)
- **Macro AUPRC 95% CI:** [0.5436, 0.5767]

## Per-finding

| finding | AUROC | AUPRC | sens | spec | F1 | Brier | ECE | support |
|---|---|---|---|---|---|---|---|---|
| opacity | 0.776 | 0.8781 | 0.6559 | 0.7818 | 0.7462 | 0.2006 | 0.162 | 1430 |
| consolidation | 0.7226 | 0.605 | 0.7849 | 0.5288 | 0.6273 | 0.247 | 0.1783 | 832 |
| pleural_effusion | 0.7762 | 0.825 | 0.7459 | 0.657 | 0.739 | 0.1982 | 0.0743 | 1169 |
| cardiomegaly | 0.7432 | 0.632 | 0.7706 | 0.6202 | 0.6687 | 0.2167 | 0.1268 | 872 |
| nodule | 0.4848 | 0.2587 | 0.2915 | 0.7129 | 0.2755 | 0.2535 | 0.2058 | 542 |
| pneumothorax | 0.6375 | 0.2162 | 0.2691 | 0.9043 | 0.2718 | 0.1354 | 0.1126 | 249 |
| hyperinflation | 0.7727 | 0.4924 | 0.5711 | 0.826 | 0.5016 | 0.162 | 0.1657 | 415 |

_Micro F1 0.6228 · micro accuracy 0.7035._