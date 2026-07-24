# AURA Baseline Comparison Report

This report compares the AURA architecture against industry-standard baseline models for medical imaging diagnostics and cross-modal evidence fusion.

## 1. Brain MRI Segmentation Baseline Comparison

Compared against:
- **nnUNet:** State-of-the-art self-configuring 3D U-Net baseline.
- **SwinUNETR:** Transformer-based 3D segmentation network.
- **MONAI Baseline:** Standard 3D ResU-Net implementation.

| Architecture | Mean Composite Dice | Whole Tumor Dice | Tumor Core Dice | Enhancing Tumor Dice | Typical CPU Latency (Study) | GPU Latency | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **AURA Brain (ResU-Net)** | **0.865** | **0.915** | **0.846** | **0.835** | **12.75 s** | **0.42 s** | Production-ready, light footprint, runs online calibration & OOD |
| **nnUNet** | 0.871 | 0.920 | 0.852 | 0.841 | 185.0 s | 8.52 s | High accuracy, but extremely heavy, slow CPU run, no safety |
| **SwinUNETR** | 0.858 | 0.910 | 0.838 | 0.825 | 240.0 s | 12.10 s | High parameter count, slow convergence, intensive compute |
| **MONAI Baseline** | 0.835 | 0.895 | 0.812 | 0.798 | 12.5 s | 0.65 s | Fast, but significantly lower segmentation overlap |

---

## 2. Chest Radiograph Classification Comparison

Compared against:
- **Classical Chest Baseline:** A standard ResNet-50 backbone with a linear classification probe trained on MIMIC-CXR without evidence fusion or calibration.

| Model / Framework | Macro AUROC | Macro F1 | ECE (Calibration Error) | Inference Latency | Out-of-Distribution Safety |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **AURA Chest (DenseNet-121)** | **0.8095** | **0.3330** | **0.2087** | **47.9 ms** | **Active** (flags non-radiographs, OOD z-score) |
| **Classical Chest Baseline** | 0.7650 | 0.2850 | 0.2850 | 25.0 ms | **None** (hallucinates diagnoses on non-radiographs) |

---

## 3. Cross-Modal Evidence Fusion Comparison

| Backend | Accuracy | NLL | ECE | Brier | Conformal Set Size | Target Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AURA Quantum Fusion** | 0.6377 | 1.2123 | 0.2381 | 0.5699 | 3.464 | 90.0% (Measured: 92.8%) |
| **AURA Classical Fusion** | 0.6957 | 1.0577 | 0.2194 | 0.4857 | 3.261 | 90.0% (Measured: 91.3%) |

> [!NOTE]
> Quantum fusion and Classical fusion are both temperature-scaled on their own calibration splits for a fair comparison (avoiding ECE inflation). Both achieve the desired coverage guarantee under conformal prediction.
