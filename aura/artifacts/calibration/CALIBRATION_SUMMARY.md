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