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
cd aura/aura-main

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
venv\Scripts\python.exe -m aura.aura_cli serve 8000
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
| `AURA_FUSION_BACKEND` | `classical` | Active backend for fusion (`classical` \| `quantum` \| `learnable`). Classical is served since 2026-09-12 — see `docs/MARKET_COMPARISON.md` §3.2. |
| `AURA_MIMIC_ROOT` | `../datasets` | Paths to dataset files. |
| `AURA_DATA_SOURCE` | `mimic` | Seed source for cases (`mimic` \| `synthetic`). |
| `AURA_OOD_ENERGY_THRESHOLD` | `1.5` | Threshold for energy-score OOD rejection. |
| `AURA_LOW_CONFIDENCE_THRESHOLD` | `0.3` | Probability cutoff for low-confidence abstention. |
| `AURA_AUTH_TOKEN` | *(empty)* | **Shared API token.** Empty = the API answers direct requests from this machine only; set = every `/v1` data route needs it. See below. |
| `AURA_AUTH_HEADER` | `x-aura-token` | Header the token is read from (`Authorization: Bearer …` also accepted). |
| `AURA_ALLOW_OPEN_ACCESS` | `0` | `1` = with no token, serve network clients anyway. Isolated demo networks only. |
| `AURA_SESSION_TTL_HOURS` | `12` | Lifetime of a browser sign-in. |
| `AURA_RATE_LIMIT_RPM` | `0` | Per-principal requests/minute; `0` disables the cap. |
| `AURA_DATABASE_URL` | *(empty)* | SQLAlchemy URL for the case store; empty = SQLite at `AURA_DB_PATH`. PostgreSQL: `postgresql+psycopg://user:pass@host:5432/aura`. |
| `AURA_SITE_ID` | *(empty)* | The hospital/site this server belongs to; binds the database on first start. |

A present-but-empty `AURA_*` variable is treated as unset, so writing `KEY=` in a
`.env` falls back to the default rather than being parsed as a value.

### Access control

AURA is **secure by default** (since 2026-09-12). With no `AURA_AUTH_TOKEN` the API
answers only direct, un-proxied requests from the machine it runs on: the offline
single-box demo keeps working and nothing else gets in. Network clients receive a 403
telling them to set a token. That protects exactly the reads that matter:
`GET /v1/cases` enumerates case ids, `GET /v1/cases/{id}` returns the full bundle, and
`/export/fhir` and `/export/hl7` emit standards-conformant clinical records built for
another system to ingest. A request relayed by a proxy counts as remote even when the
proxy runs on the same host. `AURA_ALLOW_OPEN_ACCESS=1` restores the old open behaviour
for an isolated demo network; preflight warns on every boot while it is set.

Set a token before a second machine needs to reach AURA:

```bash
export AURA_AUTH_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
```

Clients then send `x-aura-token: <value>` **and** `x-aura-user: <who>` — an
authenticated call without a named principal is rejected 403, which is what keeps
every call attributable in the audit log. `/v1/health` and the dashboard shell stay
public so container probes and the UI still load; every `/v1` data route is gated.
`deploy/preflight.py` prints a warning on each boot while the token is empty.

> [!NOTE]
> **Rate limiting is per-process.** The limiter is an in-memory token bucket with no
> shared backend, so N replicas behind a load balancer give an effective ceiling of
> `N × AURA_RATE_LIMIT_RPM`, and a restart clears every bucket. It is a courtesy
> throttle against runaway clients, not a defence against a determined one — put a
> real limiter in the ingress if you need that guarantee.
