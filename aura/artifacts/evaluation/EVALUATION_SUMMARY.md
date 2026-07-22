# MIMIC-CXR Validation — Inference Metrics

- **Model:** `E:\AURA\aura-main\aura\artifacts\best_model.pt`
- **Images:** 602
- **Inference:** 3.013s (199.82 img/s)

## Headline (macro over 7 findings)

| AUROC | AUPRC | F1 | Sens | Spec | Prec | Brier | ECE |
|---|---|---|---|---|---|---|---|
| 0.8095 | 0.3188 | 0.333 | 0.7243 | 0.7481 | 0.2391 | 0.1556 | 0.2087 |

- **Macro AUROC 95% CI:** [0.7865, 0.8367] (50 bootstraps)
- **Macro AUPRC 95% CI:** [0.3002, 0.3675]

## Per-finding

| finding | AUROC | AUPRC | sens | spec | F1 | Brier | ECE | support |
|---|---|---|---|---|---|---|---|---|
| opacity | 0.7282 | 0.4774 | 0.7429 | 0.6159 | 0.5544 | 0.2175 | 0.1874 | 175 |
| consolidation | 0.7765 | 0.1929 | 0.8 | 0.6373 | 0.2544 | 0.2298 | 0.325 | 45 |
| pleural_effusion | 0.8856 | 0.6976 | 0.8797 | 0.7162 | 0.6572 | 0.1524 | 0.1425 | 158 |
| cardiomegaly | 0.8732 | 0.5429 | 0.8737 | 0.7022 | 0.5046 | 0.172 | 0.2185 | 95 |
| nodule | 0.5899 | 0.0354 | 0.0556 | 0.8647 | 0.0204 | 0.1119 | 0.2212 | 18 |
| pneumothorax | 0.9142 | 0.1416 | 0.8182 | 0.8257 | 0.1463 | 0.12 | 0.1933 | 11 |
| hyperinflation | 0.8986 | 0.1435 | 0.9 | 0.875 | 0.1935 | 0.0859 | 0.1733 | 10 |

_Micro F1 0.4477 · micro accuracy 0.7617._