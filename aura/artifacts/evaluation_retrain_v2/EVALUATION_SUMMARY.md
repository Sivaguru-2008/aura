# MIMIC-CXR Validation — Inference Metrics

- **Model:** `E:\AURA\aura-main\aura\artifacts\retrain_v2\best_model.pt`
- **Images:** 2099
- **Inference:** 8.317s (252.37 img/s)

## Headline (macro over 7 findings)

| AUROC | AUPRC | F1 | Sens | Spec | Prec | Brier | ECE |
|---|---|---|---|---|---|---|---|
| 0.821 | 0.3576 | 0.3679 | 0.6926 | 0.7684 | 0.2638 | 0.1484 | 0.1999 |

- **Macro AUROC 95% CI:** [0.8083, 0.8332] (1000 bootstraps)
- **Macro AUPRC 95% CI:** [0.3402, 0.3847]

## Per-finding

| finding | AUROC | AUPRC | sens | spec | F1 | Brier | ECE | support |
|---|---|---|---|---|---|---|---|---|
| opacity | 0.7117 | 0.4159 | 0.6774 | 0.6351 | 0.4742 | 0.215 | 0.2125 | 496 |
| consolidation | 0.8069 | 0.2003 | 0.8083 | 0.6604 | 0.2182 | 0.2062 | 0.3085 | 120 |
| pleural_effusion | 0.9002 | 0.7025 | 0.8836 | 0.7689 | 0.6854 | 0.1383 | 0.1328 | 524 |
| cardiomegaly | 0.8617 | 0.4756 | 0.8521 | 0.7096 | 0.4596 | 0.168 | 0.2213 | 284 |
| nodule | 0.729 | 0.1648 | 0.3579 | 0.8738 | 0.178 | 0.1119 | 0.2076 | 95 |
| pneumothorax | 0.8254 | 0.1139 | 0.4603 | 0.8659 | 0.1589 | 0.0996 | 0.1492 | 63 |
| hyperinflation | 0.9118 | 0.43 | 0.8083 | 0.8651 | 0.4008 | 0.0998 | 0.1677 | 120 |

_Micro F1 0.437 · micro accuracy 0.7723._