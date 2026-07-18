# AURA — Deployment Guide

AURA is a clinical-intelligence copilot for chest radiography: a trained
DenseNet-121 vision model behind a fusion → safety → reasoning → report pipeline,
served by a FastAPI gateway with a static dashboard, plus a command-line interface
for offline prediction, evaluation, explainability, calibration, and benchmarking.

---

## 1. Requirements

| | |
|---|---|
| Python | 3.10+ (tested on 3.12) |
| Core deps | `numpy scipy scikit-learn pennylane fastapi uvicorn pydantic pillow SQLAlchemy aiosqlite httpx pytest matplotlib` (see `requirements.txt`) |
| Vision (GPU) stack | `torch torchvision torchxrayvision timm albumentations opencv-python pydicom` — a **CUDA build of torch** for GPU (tested: torch 2.11 + CUDA 12.8) |
| Model weights | `artifacts/best_model.pt` (DenseNet-121, 7-finding multilabel head) — already present |
| Dataset (evaluation only) | MIMIC-CXR under `datasets/…/versions/2/` (`mimic_cxr_aug_validate.csv` + `official_data_iccv_final/files/…`) |

The vision engine loads `artifacts/best_model.pt` **automatically** on startup; no
configuration is required. When torch/weights are unavailable it degrades to a
numpy feature model so the offline demo still runs.

## 2. Install

```bash
cd aura/aura
python -m venv venv && . venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# GPU vision stack (into a CUDA torch env):
pip install torch torchvision torchxrayvision timm albumentations opencv-python pydicom
```

## 3. Run the service

```bash
python -m aura_cli serve 8000          # gateway + dashboard on http://127.0.0.1:8000
```

- Dashboard: `/` and `/app`
- API: `/v1/health`, `/v1/cases`, `/v1/cases/{id}`, `/v1/studies/simulate`,
  `/v1/models`, `/v1/admin/safety` (unchanged — no routes or schemas were modified).
- Data source: real MIMIC-CXR by default; `AURA_DATA_SOURCE=synthetic` for the
  offline seeder.

## 4. Command-line interface

```bash
python aura_cli.py predict  --image sample.jpg      # full inference + report + heatmaps
python aura_cli.py evaluate                          # MIMIC validation metrics + plots
python aura_cli.py evaluate --calibrate              # + temperature/conformal/MC-dropout
python aura_cli.py explain  --image sample.jpg       # saliency overlays (Grad-CAM++/Score-CAM/IG)
python aura_cli.py benchmark                          # latency / throughput / memory
python aura_cli.py calibrate                          # calibration suite only
# Existing commands are unchanged:
python aura_cli.py train | train-cnn | bench | serve | demo
```

`predict` / `explain` write to `artifacts/predictions/<study>/`; `evaluate` to
`artifacts/evaluation/`; `calibrate` to `artifacts/calibration/`; `benchmark` to
`artifacts/performance/`.

## 5. Configuration knobs

Central config is `common/config.py` (`[tool.aura]` in `pyproject.toml`, overridable
by `AURA_*` env vars). Relevant to deployment:

| Env var | Purpose |
|---|---|
| `AURA_DEVICE=cuda` | force GPU |
| `AURA_DATA_SOURCE=mimic\|synthetic` | worklist seed source |
| `AURA_MIMIC_ROOT` / `AURA_MIMIC_IMAGES` / `AURA_MIMIC_VALIDATE` | relocate the corpus |
| `AURA_FUSION_BACKEND=quantum\|classical` | evidence-fusion backend |
| `AURA_LOW_CONFIDENCE_THRESHOLD`, `AURA_OOD_ENERGY_THRESHOLD` | abstention thresholds |

## 6. Container / production notes

- **Stateless serving**: the gateway holds an in-memory pipeline + a SQLite store
  (`artifacts/aura.db`); mount `artifacts/` on a persistent volume.
- **GPU**: one DenseNet-121 fits in <1 GB VRAM (see `PERFORMANCE_REPORT.md`); batch
  inference reaches hundreds of images/s. Set `AURA_DEVICE=cuda`.
- **Auth**: the gateway ships a header-based principal stub (`x-aura-user`); the
  RBAC/OIDC seam is marked in `gateway/app.py` for production auth.
- **Health check**: `GET /v1/health` returns backend + trained + case count.
- **Model registry**: `GET /v1/models` and `artifacts/registry.json`.

## 7. Regression gate (run before any deploy)

```bash
python -m compileall .            # syntax/import-time validation
python -m pytest -q               # full test suite (117 tests)
python aura_cli.py predict --image sample.jpg --brief   # end-to-end smoke
```

Any failure should block the deploy. `ruff` / `mypy` can be added to CI; they are not
in the pinned offline environment.
