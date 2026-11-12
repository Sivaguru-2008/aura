# Vision Calibration & Uncertainty

- **Images:** 2099
- **Mean ECE:** 0.1808 → **0.0256** after per-finding temperature scaling
- **Conformal coverage (target 0.9):** 0.9098 · mean set size 1.2631
- **MC-dropout layers:** 0 · TTA mean epistemic std 0.02052

## Per-finding temperature / ECE

| finding | T | ECE before | ECE after | conformal cov | set size |
|---|---|---|---|---|---|
| opacity | 0.8878 | 0.1461 | 0.0553 | 0.9086 | 1.5162 |
| consolidation | 1.4941 | 0.3164 | 0.0214 | 0.9095 | 1.4552 |
| pleural_effusion | 1.3662 | 0.1266 | 0.0295 | 0.9076 | 1.2362 |
| cardiomegaly | 1.1438 | 0.1454 | 0.0313 | 0.9067 | 1.2924 |
| nodule | 0.6058 | 0.2115 | 0.0158 | 0.9114 | 1.1381 |
| pneumothorax | 1.1345 | 0.1553 | 0.0096 | 0.9133 | 1.101 |
| hyperinflation | 1.0732 | 0.1643 | 0.0162 | 0.9114 | 1.1029 |

## Operating point (threshold) policy

Thresholds are **not** plain argmax-F1. Findings whose miss is clinically
severe carry a minimum-sensitivity floor, and the highest-precision cutoff
meeting that floor is served. Floors are assigned on severity of a miss, not
by tuning against held-out scores. Argmax-F1 is additionally unstable on rare
findings — fitted on ~1000 studies, pneumothorax carries ~30 positives.

| finding | threshold | policy |
|---|---|---|
| opacity | 0.2743 | f1_optimal |
| consolidation | 0.0596 | sensitivity_floor>=0.80 |
| pleural_effusion | 0.3847 | sensitivity_floor>=0.70 |
| cardiomegaly | 0.3057 | sensitivity_floor>=0.70 |
| nodule | 0.0342 | sensitivity_floor>=0.70 |
| pneumothorax | 0.0334 | sensitivity_floor>=0.80 |
| hyperinflation | 0.1351 | f1_optimal |

**Disclosed trade:** the `nodule` floor buys sensitivity at a large precision cost, and nodule is independently the weakest head (it does not corroborate against torchxrayvision — see `kappa_crosscheck.json`). It is floored anyway because a nodule head that catches one in five is not a safety net; treat positive nodule calls as prompts to look, not as findings.