# MIMIC-CXR Validation — Inference Metrics

- **Model:** `E:\AURA\aura-main\aura\artifacts\best_model.pt`
- **Images:** 2099
- **Inference:** 9.763s (214.99 img/s)

## Headline (macro over 7 findings)

| AUROC | AUPRC | F1 | Sens | Spec | Prec | Brier | ECE |
|---|---|---|---|---|---|---|---|
| 0.6665 | 0.214 | 0.2549 | 0.606 | 0.621 | 0.1701 | 0.2445 | 0.337 |

- **Macro AUROC 95% CI:** [0.6496, 0.6828] (1000 bootstraps)
- **Macro AUPRC 95% CI:** [0.2033, 0.2336]

## Per-finding

| finding | AUROC | AUPRC | sens | spec | F1 | Brier | ECE | support |
|---|---|---|---|---|---|---|---|---|
| opacity | 0.624 | 0.323 | 0.6532 | 0.5259 | 0.4101 | 0.2766 | 0.283 | 496 |
| consolidation | 0.7101 | 0.126 | 0.8583 | 0.4204 | 0.1504 | 0.3793 | 0.5165 | 120 |
| pleural_effusion | 0.7586 | 0.4888 | 0.855 | 0.5283 | 0.5224 | 0.2827 | 0.3201 | 524 |
| cardiomegaly | 0.744 | 0.2469 | 0.8592 | 0.5074 | 0.3432 | 0.2866 | 0.4006 | 284 |
| nodule | 0.4436 | 0.039 | 0.2316 | 0.7091 | 0.0629 | 0.2092 | 0.3485 | 95 |
| pneumothorax | 0.6106 | 0.047 | 0.1429 | 0.8846 | 0.0586 | 0.1077 | 0.1842 | 63 |
| hyperinflation | 0.7747 | 0.2275 | 0.6417 | 0.7711 | 0.2369 | 0.1693 | 0.3062 | 120 |

_Micro F1 0.3169 · micro accuracy 0.64._