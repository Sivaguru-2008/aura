# Vision Calibration & Uncertainty

- **Images:** 2099
- **Mean ECE:** 0.1986 → **0.0226** after per-finding temperature scaling
- **Conformal coverage (target 0.9):** 0.9076 · mean set size 1.281
- **MC-dropout layers:** 0 · TTA mean epistemic std 0.02052

## Per-finding temperature / ECE

| finding | T | ECE before | ECE after | conformal cov | set size |
|---|---|---|---|---|---|
| opacity | 1.0877 | 0.1918 | 0.0471 | 0.9029 | 1.5676 |
| consolidation | 1.4941 | 0.3145 | 0.0233 | 0.9086 | 1.4552 |
| pleural_effusion | 1.3593 | 0.1257 | 0.0298 | 0.9086 | 1.2448 |
| cardiomegaly | 1.3521 | 0.2216 | 0.0181 | 0.8914 | 1.3371 |
| nodule | 0.6189 | 0.2143 | 0.0145 | 0.9095 | 1.139 |
| pneumothorax | 1.1411 | 0.1581 | 0.0088 | 0.921 | 1.12 |
| hyperinflation | 1.0732 | 0.1643 | 0.0162 | 0.9114 | 1.1029 |