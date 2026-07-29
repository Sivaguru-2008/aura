# syntax=docker/dockerfile:1.7
#
# AURA — Clinical Intelligence Copilot
#
# Build (dev machine, brain weights present in the working tree):
#   docker build -t aura:latest .
#
# Build (CI / fresh clone, no 86 MB brain checkpoint):
#   docker build --build-arg AURA_SKIP_BRAIN=1 -t aura:latest .
#   docker build --build-arg AURA_MODELS_URL=https://.../models-v1 -t aura:latest .
#
# The build runs deploy/preflight.py as its last step, so an image that cannot
# load its own served weights fails at `docker build` rather than at 3 a.m. in
# front of a judge.

# =========================================================================== #
# Stage 1 — builder: resolve the dependency tree into a self-contained venv
# =========================================================================== #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed only to compile any sdist-only wheels; it stays in
# this stage and never reaches the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# A real venv rather than copying site-packages between stages: this carries the
# console scripts (uvicorn) and the correct sysconfig paths with it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# torch first, from the CPU index. The default PyPI wheel drags in ~2.5 GB of
# nvidia-* CUDA packages that are dead weight on a CPU host; installing it up
# front also stops a later resolver pass from pulling the CUDA build back in.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3" "torchvision>=0.17,<1"

# Dependency layer, cached independently of the source tree.
COPY aura/requirements.txt ./aura/requirements.txt
COPY requirements-docker.txt ./requirements-docker.txt
RUN pip install -r requirements-docker.txt

# =========================================================================== #
# Stage 2 — runtime
# =========================================================================== #
FROM python:3.11-slim AS runtime

# libgomp1  OpenMP runtime for torch / scikit-learn
# libglib2.0-0  required by opencv-python-headless
# (the previous Dockerfile asked for libgl1-mesa-glx, which does not exist on
#  bookworm — python:3.11-slim's base — and failed the apt step outright.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4 \
    AURA_DEVICE=cpu

WORKDIR /app

# Run as a non-root user. Created before COPY so the chown lands in one layer.
RUN useradd --create-home --uid 10001 aura

COPY --chown=aura:aura . /app

# The repo is authored on Windows, where the executable bit does not survive and
# git may check .sh files out with CRLF. Both make the entrypoint fail with a
# bare "no such file or directory"; normalise here so the image builds the same
# way from a Windows working tree, a Linux clone, or a CI checkout.
RUN sed -i 's/\r$//' /app/deploy/entrypoint.sh && chmod +x /app/deploy/entrypoint.sh

# --- served weights -------------------------------------------------------- #
# The chest path is entirely git-tracked and arrives with the COPY above.
# The brain checkpoint is gitignored, so it is present only when building from a
# working tree that has it. Otherwise fetch it, or explicitly opt out.
ARG AURA_SKIP_BRAIN=0
ARG AURA_MODELS_URL=""
RUN if [ -n "$AURA_MODELS_URL" ]; then \
        python deploy/fetch_models.py --url "$AURA_MODELS_URL"; \
    fi

# --- build-time preflight -------------------------------------------------- #
# Imports every dependency, loads the real DenseNet-121 through torch, checks
# each served artifact against the list, and imports the ASGI app. A broken
# image can never be pushed.
# preflight reads AURA_SKIP_BRAIN itself, so the ENV is all it needs — and the
# ENV also carries the setting through to the runtime entrypoint.
ENV AURA_SKIP_BRAIN=${AURA_SKIP_BRAIN}
RUN NO_COLOR=1 python -m deploy.preflight

# Writable state. /var/lib/aura is the mount point for the compose named volume
# holding the SQLite case store (see AURA_DB_PATH). It must exist in the image
# with the right ownership: Docker seeds an empty named volume from the image
# directory, so a missing or root-owned mount point leaves the non-root process
# unable to write and the container dies on its first upload.
#
# /app/aura/artifacts is deliberately NOT in this chown: the COPY above already
# set its ownership, and a second `chown -R` over it would rewrite all 193 MB of
# weights into a new layer, roughly doubling the image.
RUN mkdir -p /var/lib/aura /app/data && chown aura:aura /var/lib/aura /app/data

USER aura

# 8000 gateway + dashboard, 11112 mock PACS DICOM listener
EXPOSE 8000 11112

# curl is not in the slim image; use the interpreter that is guaranteed present.
# start-period covers the cold start, which loads DenseNet-121 (and the BraTS
# checkpoint) through torch on CPU — a shorter window restart-loops the
# container as "unhealthy" before it has finished booting.
# PORT is read inside python so the check follows a remapped port.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/v1/health', timeout=5).status==200 else 1)"

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
CMD ["serve"]
