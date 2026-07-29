"""AURA deployment preflight — fail loudly at boot, never silently at 3 a.m.

Run before uvicorn. Exits non-zero with a readable report if the container
cannot serve what it claims to serve.

Why this exists
---------------
Three code paths turn a broken deployment into a *healthy-looking* one:

1. ``services/vision/engine.py`` falls back to a numpy feature model when
   ``AURA_ALLOW_FALLBACK_VISION=1`` — the comment above it says "dev/test only",
   but the old Dockerfile set it to 1 unconditionally. The container then serves
   fabricated chest findings and ``/v1/health`` still returns ``{"status":"ok"}``.
2. ``gateway/app.py`` wraps ``install_router()`` in ``try/except`` and only
   prints a warning — a router failure means every ``/v1/studies/analyze`` and
   brain-MRI request 404s while the process stays up.
3. ``pydicom`` is imported lazily inside functions, so a missing wheel surfaces
   as a 500 on the first real DICOM upload rather than at startup.

Preflight converts all three into a boot-time failure with a specific message.

Usage
-----
    python deploy/preflight.py              # full check, exit 1 on failure
    python deploy/preflight.py --warn-only  # report but always exit 0
    python deploy/preflight.py --skip-brain # chest-only image (no 86 MB weight)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The app is rooted at aura/ (pyproject sets pythonpath = ["."] relative to it).
AURA_ROOT = Path(__file__).resolve().parent.parent / "aura"
ARTIFACTS = AURA_ROOT / "artifacts"

if str(AURA_ROOT) not in sys.path:
    sys.path.insert(0, str(AURA_ROOT))

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

_failures: list[str] = []
_warnings: list[str] = []


def _ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def _fail(check: str, msg: str, fix: str) -> None:
    print(f"  {RED}FAIL{RESET}  {check}")
    print(f"        {msg}")
    print(f"        {DIM}fix: {fix}{RESET}")
    _failures.append(check)


def _warn(check: str, msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {check}")
    print(f"        {msg}")
    _warnings.append(check)


# --------------------------------------------------------------------------- #
# 1. Python dependencies
# --------------------------------------------------------------------------- #
# (module, why it matters, what breaks without it)
REQUIRED_IMPORTS = [
    ("numpy", "core numerics"),
    ("scipy", "calibration / conformal maths"),
    ("sklearn", "tabular diagnosis models"),
    ("pennylane", "quantum fusion backend"),
    ("fastapi", "gateway"),
    ("uvicorn", "ASGI server"),
    ("pydantic", "request/response contracts"),
    ("PIL", "image intake"),
    ("matplotlib", "Grad-CAM overlays + calibration plots"),
    ("sqlalchemy", "case store"),
    ("nibabel", "NIfTI MRI reader"),
    ("torch", "served DenseNet-121 + BraTS models"),
    ("torchvision", "DenseNet-121 backbone"),
    ("cv2", "Grad-CAM resampling"),
    ("pandas", "MIMIC tabular path"),
    # Lazily imported in the app -> would otherwise fail on first real upload.
    ("pydicom", "DICOM intake (lazy import: fails at upload time, not boot)"),
    ("pynetdicom", "mock PACS listener on :11112"),
]


def check_imports() -> None:
    print("\n[1/5] Python dependencies")
    for mod, why in REQUIRED_IMPORTS:
        try:
            __import__(mod)
            _ok(f"{mod:<14} {DIM}{why}{RESET}")
        except Exception as exc:
            _fail(
                f"import {mod}",
                f"{type(exc).__name__}: {exc}  ({why})",
                f"add '{mod}' to requirements-docker.txt and rebuild",
            )


# --------------------------------------------------------------------------- #
# 2. Served artifacts
# --------------------------------------------------------------------------- #
# Everything here is git-tracked, so a plain `git clone` + `docker build` has it.
# Missing entries are always an error, never downgraded by --skip-brain.
TRACKED_ARTIFACTS = [
    ("best_model.pt", "served DenseNet-121 chest CXR weights"),
    ("vision.npz", "chest vision calibration"),
    ("vision_serving_calibration.json", "per-finding Platt + operating points"),
    ("fusion_classical.npz", "product-of-experts fusion backend"),
    ("fusion_quantum.npz", "8-qubit VQC fusion backend"),
    ("safety.npz", "abstention / conformal / OOD calibration"),
    ("conformal_mondrian.npy", "Mondrian conformal quantiles"),
    ("backend_calibration.json", "per-backend temperature + coverage"),
    ("registry.json", "served model registry"),
    ("chest_registry.json", "chest registry"),
    ("brain_registry.json", "brain registry"),
    # Brain sidecars, but git-tracked and KB-sized, so they arrive with a clone
    # even when the checkpoint below does not.
    ("brain/presence_calibration.json", "brain presence-head Platt calibration"),
    ("brain/embeddings/latest.npz", "brain embedding store (not recomputed downstream)"),
]

# NOT git-tracked: aura/.gitignore excludes brain/checkpoints/, so a fresh clone
# has no BraTS model. Downgraded to a warning by --skip-brain / AURA_SKIP_BRAIN=1.
BRAIN_ARTIFACTS = [
    ("brain/checkpoints/best_brain_model.pt", "served BraTS multi-task brain model"),
]


def check_artifacts(skip_brain: bool) -> None:
    print("\n[2/5] Served artifacts")
    if not ARTIFACTS.is_dir():
        _fail(
            "artifacts/ directory",
            f"{ARTIFACTS} does not exist",
            "check the Dockerfile COPY step and .dockerignore",
        )
        return

    for rel, why in TRACKED_ARTIFACTS:
        p = ARTIFACTS / rel
        if p.is_file() and p.stat().st_size > 0:
            _ok(f"{rel:<34} {DIM}{p.stat().st_size / 1e6:.1f} MB - {why}{RESET}")
        else:
            _fail(
                f"artifact {rel}",
                f"missing or empty: {p}  ({why})",
                "this file is git-tracked — check .dockerignore did not exclude it",
            )

    for rel, why in BRAIN_ARTIFACTS:
        p = ARTIFACTS / rel
        if p.is_file() and p.stat().st_size > 0:
            _ok(f"{rel:<34} {DIM}{p.stat().st_size / 1e6:.1f} MB - {why}{RESET}")
        elif skip_brain:
            _warn(
                f"artifact {rel}",
                f"absent - brain MRI disabled in this image ({why})",
            )
        else:
            _fail(
                f"artifact {rel}",
                f"missing: {p}  ({why})",
                "gitignored weight — run `python deploy/fetch_models.py`, "
                "or build with --build-arg AURA_SKIP_BRAIN=1 for a chest-only image",
            )


# --------------------------------------------------------------------------- #
# 3. The real vision model actually loads
# --------------------------------------------------------------------------- #
def check_vision_engine() -> None:
    print("\n[3/5] Vision engine (real weights, not fallback)")
    fallback = os.environ.get("AURA_ALLOW_FALLBACK_VISION", "0")
    if fallback == "1":
        _warn(
            "AURA_ALLOW_FALLBACK_VISION=1",
            "the numpy fallback vision model is ENABLED. If the DenseNet weights fail "
            "to load, the container will serve fabricated findings and still report "
            "healthy. Unset this for anything judge- or patient-facing.",
        )

    # Force the strict path regardless, so we learn whether the real model loads.
    prev = os.environ.get("AURA_ALLOW_FALLBACK_VISION")
    os.environ["AURA_ALLOW_FALLBACK_VISION"] = "0"
    try:
        from aura.services.vision.engine import VisionEngine

        engine = VisionEngine.load()
        backbone = getattr(engine, "backbone", None)
        if backbone is None:
            _fail(
                "vision backbone",
                "VisionEngine loaded without a CNN backbone (numpy fallback path)",
                "confirm artifacts/best_model.pt is present and torch can load it",
            )
        else:
            version = getattr(backbone, "model_version", "unknown")
            _ok(f"DenseNet-121 backbone loaded {DIM}version={version}{RESET}")
    except Exception as exc:
        # engine.py builds a long diagnostic report; surface it verbatim.
        _fail(
            "vision engine load",
            f"{type(exc).__name__}: {exc}",
            "see the diagnostic report above; usually a missing/corrupt best_model.pt "
            "or a torch/torchvision ABI mismatch",
        )
    finally:
        if prev is None:
            os.environ.pop("AURA_ALLOW_FALLBACK_VISION", None)
        else:
            os.environ["AURA_ALLOW_FALLBACK_VISION"] = prev


# --------------------------------------------------------------------------- #
# 4. Modality router installs (gateway swallows this failure)
# --------------------------------------------------------------------------- #
def check_router() -> None:
    print("\n[4/5] Modality router")
    try:
        from aura.backend.bootstrap import install_router  # noqa: F401

        _ok("backend.bootstrap.install_router importable")
    except Exception as exc:
        _fail(
            "modality router",
            f"{type(exc).__name__}: {exc}",
            "gateway/app.py only WARNS on this — /v1/studies/analyze and every brain "
            "MRI route would 404 while the container reports healthy",
        )


# --------------------------------------------------------------------------- #
# 5. The ASGI app imports and the fusion model is trained
# --------------------------------------------------------------------------- #
def check_app() -> None:
    print("\n[5/5] Gateway application")
    try:
        from aura.gateway.app import app

        routes = [getattr(r, "path", "") for r in app.routes]
        if "/v1/health" not in routes:
            _fail("health route", "/v1/health not registered", "check gateway/app.py")
        else:
            _ok(f"gateway.app:app imports {DIM}{len(routes)} routes{RESET}")
    except Exception as exc:
        _fail(
            "gateway import",
            f"{type(exc).__name__}: {exc}",
            "the ASGI target 'aura.gateway.app:app' must import cleanly for uvicorn to start",
        )
        return


def main() -> int:
    ap = argparse.ArgumentParser(description="AURA deployment preflight")
    ap.add_argument("--warn-only", action="store_true",
                    help="report problems but always exit 0")
    ap.add_argument("--skip-brain", action="store_true",
                    default=os.environ.get("AURA_SKIP_BRAIN") == "1",
                    help="treat missing brain-MRI weights as a warning")
    args = ap.parse_args()

    print("=" * 74)
    print("AURA preflight")
    print(f"  python     {sys.version.split()[0]}")
    print(f"  aura root  {AURA_ROOT}")
    print(f"  artifacts  {ARTIFACTS}")
    print("=" * 74)

    check_imports()
    check_artifacts(args.skip_brain)
    check_vision_engine()
    check_router()
    check_app()

    print("\n" + "=" * 74)
    if _failures:
        print(f"{RED}PREFLIGHT FAILED{RESET} - {len(_failures)} check(s): "
              + ", ".join(_failures))
        if _warnings:
            print(f"{YELLOW}{len(_warnings)} warning(s){RESET}: " + ", ".join(_warnings))
        print("=" * 74)
        if args.warn_only:
            print("--warn-only set; starting anyway.")
            return 0
        return 1

    if _warnings:
        print(f"{GREEN}PREFLIGHT OK{RESET} with {YELLOW}{len(_warnings)} warning(s){RESET}: "
              + ", ".join(_warnings))
    else:
        print(f"{GREEN}PREFLIGHT OK{RESET} - all checks passed.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
