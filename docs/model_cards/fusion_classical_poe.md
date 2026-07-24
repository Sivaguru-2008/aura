# Model Card: Classical Evidence Fusion Model (PoE)

This model card describes AURA's Classical Evidence Fusion model.

---

## 1. Model Details
- **Architecture:** Classical Product of Experts (PoE) log-linear fusion model.
- **Model Version:** `fusion-poe-classical-v2`
- **Framework:** NumPy / PyTorch (reloaded from `fusion_classical.npz`).
- **Input:** 7-dimensional chest radiograph probability vector plus demographic/symptomatic prior variables.

---

## 2. Intended Use
- **Primary Application:** Multi-class diagnostic evidence fusion, integrating clinical priors and imaging features under classical Bayesian assumptions.

---

## 3. Measured Performance
Evaluated on a held-out evidence calibration split. Temperature-scaled to prevent ECE inflation.

- **Diagnostic Accuracy:** 0.6957
- **Negative Log-Likelihood (NLL):** 1.0577
- **Expected Calibration Error (ECE):** 0.2194
- **Brier Score:** 0.4857
- **Conformal Coverage (90% target):** 91.30%
- **Average Conformal Set Size:** 3.26 diagnoses

---

## 4. Calibration & Temperature Scaling
- **Optimal Temperature (T_c):** 0.3579
- **Conformal q-hat:** 0.8654
