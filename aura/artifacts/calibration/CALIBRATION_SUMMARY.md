# Vision Calibration & Uncertainty

- **Images:** 2099
- **Mean ECE:** 0.1445 → **0.1301** after per-finding temperature scaling
- **Conformal coverage (target 0.9):** 0.9057 · mean set size 1.5268
- **MC-dropout layers:** 0 · TTA mean epistemic std 0.03951

## Per-finding temperature / ECE

| finding | T | ECE before | ECE after | conformal cov | set size |
|---|---|---|---|---|---|
| opacity | 1.0245 | 0.1836 | 0.1839 | 0.899 | 1.519 |
| consolidation | 2.7392 | 0.1674 | 0.125 | 0.9152 | 1.7057 |
| pleural_effusion | 1.6357 | 0.0656 | 0.0489 | 0.9048 | 1.4905 |
| cardiomegaly | 1.4032 | 0.1224 | 0.1112 | 0.9095 | 1.6448 |
| nodule | 2.3862 | 0.2012 | 0.177 | 0.899 | 1.7248 |
| pneumothorax | 1.2318 | 0.1121 | 0.1252 | 0.9143 | 1.2524 |
| hyperinflation | 0.7982 | 0.159 | 0.1392 | 0.8981 | 1.3505 |