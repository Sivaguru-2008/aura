# AURA — Training & Recalibration Guide

This document describes how to execute model training, calibrate output probabilities, and tune decision thresholds within the AURA clinical intelligence platform.

---

## 1. Fine-Tuning the CNN Vision Backbone

AURA's vision model is a 7-label multi-label DenseNet-121 classifier. Fine-tuning runs via PyTorch and supports synthetic generator runs or real MIMIC-CXR training:

```bash
# Run from the aura directory:
cd aura

# Option A: Fine-tune on synthetic radiographs (dev check)
venv\Scripts\python.exe -m aura_cli train-cnn densenet121

# Option B: Fine-tune on real chest radiographs
# Provide path to the manifest CSV linking image files to CheXpert-style labels
set AURA_CNN_MANIFEST=..\datasets\mimiciv\mimic_cxr_aug_train.csv
venv\Scripts\python.exe -m aura_cli train-cnn densenet121
```

*Supported architectures for `--arch` are `densenet121`, `efficientnetv2`, `convnext`, and `swin`.*

---

## 2. Fitting Platt Calibration & Conformal Quantiles

After training the CNN backbone, run the calibration script to fit per-finding Platt scaling parameters and conformal prediction quantiles. This generates `vision_serving_calibration.json`:

```bash
# Run validation calibration on the held-out split (e.g. n=2099)
venv\Scripts\python.exe -m aura_cli calibrate --limit 2099
```

This updates the following files in `artifacts/`:
* `vision_serving_calibration.json`: Sigmoid parameters $a, b$ and operating threshold cutoffs.
* `conformal_mondrian.npy`: Class-conditional conformal quantiles $\hat{q}_c$.

---

## 3. Training the Fusion Backend

The fusion models (Variational Quantum Circuit, Bayesian Product-of-Experts, and Learnable linear heads) map 8-channel evidence vectors to 6 final diagnoses. 

To train the fusion layer on real MIMIC-CXR evidence distributions:

```bash
# Train the classical and quantum fusion models on n=700 patient splits
venv\Scripts\python.exe -m aura_cli train 700
```

This command:
1. Loads the DenseNet-121 model.
2. Infers evidence vectors on real patient splits.
3. Optimizes parameters for `QuantumFusion` (VQC) and `LearnableFusion` via PyTorch and PennyLane.
4. Generates the corresponding model artifacts: `fusion_quantum.npz`, `fusion_classical.npz`, and `fusion_learnable.npz` in the `artifacts/` folder.
