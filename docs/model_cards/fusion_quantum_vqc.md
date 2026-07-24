# Model Card: Quantum Evidence Fusion Model (VQC)

This model card describes AURA's Quantum Evidence Fusion model, which fuses diagnostic imaging features with clinical priors.

---

## 1. Model Details
- **Architecture:** Variational Quantum Classifier (VQC) using a parameterized quantum circuit with ring entanglement.
- **Model Version:** `fusion-vqc-quantum-v2`
- **Qubits:** 6-qubit system.
- **Framework:** Pennylane / PyTorch (reloaded from `fusion_quantum.npz`).
- **Input:** 7-dimensional chest radiograph probability vector plus demographic/symptomatic prior variables.

---

## 2. Intended Use
- **Primary Application:** Multi-class diagnostic evidence fusion. Modulates raw chest X-ray findings using patient history, laboratory values, and symptoms.
- **Benefits:** Exploits quantum entanglement to capture non-linear correlation structures between findings and priors.

---

## 3. Measured Performance
Evaluated on a held-out evidence calibration split. Temperature-scaled to prevent ECE inflation.

- **Diagnostic Accuracy:** 0.6377
- **Negative Log-Likelihood (NLL):** 1.2123
- **Expected Calibration Error (ECE):** 0.2381
- **Brier Score:** 0.5699
- **Conformal Coverage (90% target):** 92.75%
- **Average Conformal Set Size:** 3.46 diagnoses

---

## 4. Calibration & Temperature Scaling
- **Optimal Temperature (T_q):** 0.8784
- **Conformal q-hat:** 0.8925
