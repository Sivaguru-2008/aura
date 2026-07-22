# AURA — Deployment & Installation Guide

This document describes the prerequisites, installation steps, configuration options, and production deployment guidelines for the AURA clinical intelligence platform.

---

## 1. Prerequisites & System Requirements

### Hardware Requirements
* **GPU (Recommended for serving & fine-tuning)**: NVIDIA GPU with CUDA compatibility (e.g., RTX 3060/4060, T4, L4, or newer). Peak VRAM usage is under 1 GB for single instances.
* **CPU**: Multicore CPU (8-cores recommended) for running the PennyLane quantum simulators.
* **Storage**: At least 2 GB of storage for model checkpoints, plus additional storage matching the size of the ingested MIMIC-CXR dataset.

### Software Prerequisites
* **Operating System**: Windows, Linux, or macOS.
* **Python Version**: Python 3.10 to 3.12. (Note: standard execution is tested using Python 3.12).
* **CUDA & cuDNN**: Matches your PyTorch CUDA build (e.g., CUDA 11.8 or 12.1+).

---

## 2. Installation Steps

AURA is configured as a Python-native application. Install standard dependencies from the `aura-main` directory:

```bash
# Clone the repository
git clone https://github.com/Sivaguru-2008/aura
cd aura/aura-main/aura

# Create a virtual environment
python -m venv venv
venv\Scripts\activate     # On Linux/macOS: source venv/bin/activate

# Install core dependencies (numpy, scipy, scikit-learn, fastapi, uvicorn, pennylane)
pip install -r requirements.txt

# Install PyTorch and Vision packages (CUDA GPU support)
pip install torch torchvision torchxrayvision timm albumentations opencv-python pydicom
```

*Note: The model weights `artifacts/best_model.pt` are pre-shipped in the repository.*

---

## 3. Running the Service

```bash
# Start the FastAPI gateway and doctor dashboard on port 8000:
venv\Scripts\python.exe -m aura_cli serve 8000
```

Open **http://127.0.0.1:8000** in your browser to access the static clinical dashboard.

---

## 4. Command-Line Interface (CLI) Utilities

AURA provides a unified entry point CLI (`aura_cli.py`) for offline operations:

* **serve [port]**: Starts the FastAPI gateway.
* **predict --image [path]**: Runs the full 9-stage pipeline on a radiograph, generating reports and saliency overlays in `artifacts/predictions/`.
* **explain --image [path]**: Generates saliency heatmaps using Grad-CAM++ and occlusion.
* **evaluate**: Evaluates DenseNet-121 classification performance and outputs plots and metrics.
* **calibrate**: Refits Platt temperature scaling parameters.
* **benchmark**: Evaluates image throughput, batch latency, and VRAM utilization.
* **agent --image [path]**: Executes sequential diagnostic rollouts for active diagnosis.

---

## 5. Configuration Knobs

AURA is configured via `common/config.py` (which parses `pyproject.toml` and environment variables prefixed with `AURA_`):

| Environment Variable | Default Value | Purpose |
|---|---|---|
| `AURA_DEVICE` | `cpu` or `cuda` | Forces CPU or GPU execution for the vision model. |
| `AURA_FUSION_BACKEND` | `quantum` | Active backend for fusion (`quantum` \| `classical` \| `learnable`). |
| `AURA_MIMIC_ROOT` | `../datasets` | Paths to dataset files. |
| `AURA_DATA_SOURCE` | `mimic` | Seed source for cases (`mimic` \| `synthetic`). |
| `AURA_OOD_ENERGY_THRESHOLD` | `1.5` | Threshold for energy-score OOD rejection. |
| `AURA_LOW_CONFIDENCE_THRESHOLD` | `0.45` | Probability cutoff for low-confidence abstention. |
| `AURA_SEC_AUTH_ENABLED` | `false` | Enables bearer token authentication gate. |
