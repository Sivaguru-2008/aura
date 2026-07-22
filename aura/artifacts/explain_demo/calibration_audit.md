# AURA Vision Calibration Audit (measured on full val, n=2099)

**Macro ECE:** raw 0.184 → served 0.095 (Platt, fit n=16) → **full-val 0.018** (Platt, fit n=2099)

| finding | n_pos | ECE raw | ECE served | ECE full | served thr | served fires | full thr | full fires |
|---|---|---|---|---|---|---|---|---|
| opacity | 594 | 0.165 | 0.130 | 0.031 | 0.14 | 887 | 0.27 | 887 |
| consolidation | 118 | 0.309 | 0.047 | 0.016 | 0.15 | 161 | 0.20 | 136 |
| pleural_effusion | 524 | 0.133 | 0.264 | 0.028 | 0.00 | 2099 | 0.29 | 751 |
| cardiomegaly | 415 | 0.159 | 0.093 | 0.022 | 0.26 | 423 | 0.22 | 688 |
| nodule | 98 | 0.206 | 0.045 | 0.009 | 0.50 | 3 | 0.15 | 114 |
| pneumothorax | 65 | 0.148 | 0.031 | 0.005 | 0.50 | 0 | 0.17 | 68 |
| hyperinflation | 120 | 0.168 | 0.057 | 0.016 | 0.50 | 0 | 0.14 | 243 |

BEFORE/AFTER of fixing the calibration-clobber bug. served = the OLD degenerate fit (n=16, written by run_calibration(limit=3) in tests, backup vision_serving_calibration.n16.bak.json); full = the CURRENT served calibration (validated full-val fit, n=2099). Root cause fixed in ml/evaluation/vision_calibration.py: production serving write is now gated to canonical runs.