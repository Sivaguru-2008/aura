# AURA — Container Deployment & CI/CD

Building, running, and shipping AURA as a container. For bare-metal install and
hardware requirements see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 1. Quick start

```bash
docker compose up --build
```

Then open <http://localhost:8000>. The API is under `/v1`, the console at
`/app`, the report portal at `/history`.

First build takes 8–15 minutes (the torch CPU wheels dominate). Later builds
reuse the dependency layer and take under a minute unless
`requirements-docker.txt` changed.

Without Compose:

```bash
docker build -t aura:latest .
docker run --rm -p 8000:8000 -v aura-state:/var/lib/aura \
  -e AURA_DB_PATH=/var/lib/aura/aura.db aura:latest
```

---

## 2. What ships in the image

| Path | Size | Source |
|---|---:|---|
| `aura/artifacts/best_model.pt` | 28 MB | git-tracked — served DenseNet-121 chest CXR |
| `aura/artifacts/fusion_*.npz`, `safety.npz`, `*calibration*.json` | < 1 MB | git-tracked |
| `aura/artifacts/brain/checkpoints/best_brain_model.pt` | 90 MB | **not git-tracked** — see §3 |
| Python runtime + torch CPU | ~1.6 GB | built in stage 1 |

The chest-X-ray path is fully self-contained: `git clone && docker build`
produces a working chest image with no extra steps.

### The build context

`.dockerignore` is load-bearing, not hygiene. The working tree is ~16 GB
(`.venv` 1.4 GB, `.git` 1.2 GB, `aura/artifacts` 13 GB — of which 11 GB is the
regenerable brain voxel cache). Without the exclusions `docker build` spends
minutes streaming the context to the daemon before running a single instruction,
and typically dies. With them the context measures **196 MB** — 193 MB of served
weights and calibration, 3 MB of source and web UI.

If you add a new artifact directory, decide whether it belongs in the image.
Rule of thumb: **if `aura_cli train` or `cli ingest` can regenerate it, exclude it.**

One subtlety worth keeping in mind when editing `.dockerignore`: a bare `*` does
not cross a `/`, so `*.bak` matches only the context root. Nested matches need
`**/*.bak`.

---

## 3. Brain-MRI weights

`aura/.gitignore` excludes `aura/artifacts/brain/checkpoints/`, so a fresh clone
has no BraTS model. Three ways to build:

**a. From a working tree that already has the checkpoint** (the dev machine) —
nothing to do; `docker build` picks it up.

**b. Chest-only image:**

```bash
docker build --build-arg AURA_SKIP_BRAIN=1 -t aura:chest .
```

Brain studies are unavailable; preflight downgrades the missing weight from an
error to a warning.

**c. Fetch from a published release:**

```bash
docker build --build-arg AURA_MODELS_URL=https://github.com/<owner>/aura/releases/download/models-v1 -t aura:latest .
```

To publish that release, upload the contents of `model_export/` (157 MB, already
staged with `SHA256SUMS.txt`) as release assets, then set the repository
variable `AURA_MODELS_URL` so CI builds full images too.

Locally:

```bash
python deploy/fetch_models.py --from-local ../model_export
python deploy/fetch_models.py --verify-only     # report what is installed
```

Every file is verified against the SHA-256 recorded in `deploy/fetch_models.py`
(mirrored from `model_export/SHA256SUMS.txt`) *before* it is moved into place, so
a truncated download fails loudly instead of installing a model that loads and
predicts nonsense.

---

## 4. Preflight — why "healthy" was not proof of anything

Three paths in the codebase turn a broken deployment into a healthy-looking one:

1. **`services/vision/engine.py`** falls back to a numpy feature model when
   `AURA_ALLOW_FALLBACK_VISION=1`. The comment above it says *dev/test only* —
   the previous Dockerfile set it to `1` unconditionally. A container whose
   DenseNet weights failed to load would serve fabricated chest findings, and
   `/v1/health` would still return `{"status": "ok"}`.
2. **`gateway/app.py`** wraps `install_router()` in `try/except` and only prints
   a warning. A router failure means every `/v1/studies/analyze` and brain-MRI
   route 404s while the process stays up and reports healthy.
3. **`pydicom` is imported lazily** inside functions, so a missing wheel surfaces
   as a 500 on the first real DICOM upload rather than at startup. It was absent
   from both `aura/requirements.txt` and the old Dockerfile.

`deploy/preflight.py` converts all three into a boot-time failure with a
specific message. It runs twice: **at build time** (so a broken image can never
be pushed) and **at container start**, from the entrypoint.

```bash
python deploy/preflight.py               # exit 1 on any failure
docker compose run --rm aura preflight   # diagnose a bad deploy
```

`AURA_ALLOW_FALLBACK_VISION` is deliberately **not** set in the Dockerfile or the
compose file. Preflight forces the strict path regardless when it tests the
vision engine, and warns loudly if it finds the flag enabled.

| `AURA_PREFLIGHT` | Behaviour |
|---|---|
| `strict` (default) | a failed check stops the container |
| `warn` | report problems, start anyway |
| `skip` | do not check |

---

## 5. Smoke test

Preflight proves the process *can* load its models. `deploy/smoke_test.py`
proves it actually serves a diagnosis — health, modality router, upload, case
detail, model card — with 16 assertions, including that the returned posterior
is normalised and **not uniform** (a uniform posterior is what an untrained or
fallback fusion head emits).

```bash
python deploy/smoke_test.py                              # in-process, temp DB
python deploy/smoke_test.py --url http://localhost:8000  # against a container
python deploy/smoke_test.py --expect-brain               # also require neuromind
```

In-process mode writes to a throwaway SQLite file so it does not append test
cases to your real worklist.

---

## 6. Persistence

`common/config.py` derives `ARTIFACTS` from `__file__`, so the case store lands
at `aura/artifacts/aura.db` — inside the image layer, discarded on every restart.

The previous compose file set `AURA_DATA_DIR`, which **nothing in the codebase
reads**. `AURA_DB_PATH` was added to `common/config.py` to make the store
relocatable; unset, behaviour is unchanged.

```yaml
environment:
  AURA_DB_PATH: /var/lib/aura/aura.db
volumes:
  - aura-state:/var/lib/aura
```

A named volume rather than a bind mount onto `aura/artifacts`: mounting over that
directory would shadow the baked-in weights. `/var/lib/aura` is created in the
image owned by uid 10001, because Docker seeds a fresh named volume from the
image directory — a root-owned mount point leaves the non-root process unable to
write, and the container dies on its first upload.

---

## 7. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `8000` / `0.0.0.0` | bind address |
| `AURA_DB_PATH` | `aura/artifacts/aura.db` | SQLite case store |
| `AURA_PREFLIGHT` | `strict` | `strict` / `warn` / `skip` |
| `AURA_SKIP_BRAIN` | `0` | treat missing brain weights as a warning |
| `AURA_ALLOW_FALLBACK_VISION` | `0` | **leave at 0** outside dev/tests |
| `AURA_FUSION_BACKEND` | from `pyproject.toml` | `classical` / `quantum` |
| `AURA_LOG_LEVEL` | `info` | uvicorn log level |
| `OMP_NUM_THREADS` | `4` | torch / scikit-learn CPU threads |

`AURA_FUSION_BACKEND` is intentionally absent from `docker-compose.yml`.
`common/config.py` accepts an env override on `is not None`, so a compose default
of `""` would be read as a real backend name and break fusion. Left unset, the
container serves exactly what `[tool.aura]` selects.

> `aura/pyproject.toml` sets `fusion_backend = "quantum"` while
> `artifacts/registry.json` marks classical `role: served` and quantum
> `status: research`. That disagreement predates this deployment work and is not
> resolved here — decide which is authoritative before a judged run.

### Workers

`--workers 1`, deliberately. The gateway keeps per-process state in a
module-level dict and binds the mock PACS listener to `:11112` during lifespan;
a second worker races on both and fails to bind. Scale with replicas behind a
load balancer, not with in-process workers, and give each replica its own
`AURA_DB_PATH`.

---

## 8. CI/CD

**`.github/workflows/ci.yml`** — every push and PR: install CPU torch, install
deps, `preflight --skip-brain`, the full `-m "not slow"` pytest suite, then the
in-process smoke test. `AURA_ALLOW_FALLBACK_VISION=0` is set at job level, so a
green run means the real DenseNet-121 loaded — not that the numpy stand-in
covered for it.

**`.github/workflows/docker.yml`** — builds the image, loads it into the local
daemon, starts it, and asserts:

- the smoke test passes over HTTP against the running container
- the container runs as a non-root uid
- Docker's own `HEALTHCHECK` reaches `healthy`

Only then does it push to GHCR (`main`/`master`, `v*` tags, or a manual
dispatch). PRs build and test but never push.

Set the repository variable `AURA_MODELS_URL` to have CI build full images with
brain support; without it CI builds chest-only.

---

## 9. Failure modes found and closed

| Symptom | Cause | Fix |
|---|---|---|
| `COPY failed: requirements.txt not found` | the file is at `aura/requirements.txt`, not the context root | corrected path |
| `E: Unable to locate package libgl1-mesa-glx` | that package does not exist on bookworm, which `python:3.11-slim` is based on | `libgomp1` + `libglib2.0-0` |
| Container exits 0 immediately | old `CMD ["python", "aura/gateway/app.py"]` — `app.py` has no `__main__` block, so it imported and exited | entrypoint runs uvicorn |
| Server unreachable from the host | `aura_cli serve` binds `127.0.0.1`, which inside a container is not the host | entrypoint binds `0.0.0.0` |
| `exec /app/deploy/entrypoint.sh: no such file or directory` | CRLF on the shebang, or no exec bit, from a Windows working tree | `.gitattributes` forces LF; the Dockerfile strips CR and chmods |
| Build hangs on "transferring context" | 16 GB build context | `.dockerignore` |
| Worklist empty after every restart | `AURA_DATA_DIR` is read by nothing | `AURA_DB_PATH` + named volume |
| DICOM upload 500s in the container | `pydicom` missing; the lazy import hid it until upload time | added to `requirements-docker.txt`; preflight imports it eagerly |
| `mock DICOM listener failed to start` | `pynetdicom` missing | added to `requirements-docker.txt` |
| First brain study OOM-kills the container | CPU torch with a 3D BraTS model | memory limit ≥ 4 GB (compose defaults to 6 GB) |

---

## 10. Layout

```
Dockerfile                  multi-stage build; runs preflight as its last step
.dockerignore               16 GB -> 196 MB build context
.gitattributes              LF for .sh/Dockerfile/yml — CRLF breaks the entrypoint
docker-compose.yml          single-host deployment
requirements-docker.txt     container deps (adds pydicom, pynetdicom, pandas, cv2)
deploy/
  entrypoint.sh             preflight, then uvicorn on 0.0.0.0
  preflight.py              deps + artifacts + real model load + router + ASGI app
  smoke_test.py             end-to-end, in-process or over HTTP
  fetch_models.py           SHA-256-verified weight delivery
.github/workflows/
  ci.yml                    tests + smoke on every push/PR
  docker.yml                build, run, verify, publish to GHCR
```
