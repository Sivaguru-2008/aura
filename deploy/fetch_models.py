"""Fetch the served model weights that are too large to live in git.

The chest-X-ray path is fully self-contained: every artifact it needs
(``best_model.pt``, the fusion ``.npz`` backends, the calibration JSON) is
git-tracked, so ``git clone && docker build`` produces a working chest image.

The brain-MRI path is not. ``aura/artifacts/brain/checkpoints/`` is excluded by
.gitignore (the directory is ~300 MB of training state), which means a CI-built
image has no BraTS model and every brain study 404s or falls through. This
script closes that gap.

Two sources, tried in order:

1. **Local stage** — ``model_export/`` as produced alongside the repo. This is
   the fast path on the dev machine; nothing is downloaded.
2. **Release URL** — ``AURA_MODELS_URL`` (or ``--url``) pointing at a base URL
   that serves the ``model_export/`` file names, e.g. a GitHub Release:
   ``https://github.com/<owner>/aura/releases/download/models-v1``

Every file is checked against the SHA-256 in ``KNOWN_SHA256`` (mirrored from
``model_export/SHA256SUMS.txt``) before it is moved into place, so a truncated
download or a swapped asset fails loudly instead of producing a model that
loads but predicts nonsense.

Usage
-----
    python deploy/fetch_models.py                          # local stage, then URL
    python deploy/fetch_models.py --from-local ../model_export
    python deploy/fetch_models.py --url https://.../models-v1
    python deploy/fetch_models.py --verify-only            # check what is present
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "aura" / "artifacts"

# Default local stage: model_export/ sits next to the repo checkout.
DEFAULT_LOCAL = REPO_ROOT.parent / "model_export"

# export file name -> (destination under aura/artifacts/, sha256, why it is needed)
ASSETS: dict[str, tuple[str, str, str]] = {
    "brain_mri_multitask.pt": (
        "brain/checkpoints/best_brain_model.pt",
        "732ff9b5afdd09f7db58a6f04b1981087560608fe33b2b5d94f9eff1b2cfc958",
        "served BraTS multi-task brain model (5 heads, FLAIR-driven)",
    ),
    "brain_mri_encoder.pt": (
        "brain/checkpoints/brain_encoder.pt",
        "80659cae150ebadff8792040f487dcb585e669f7392e0420cda7b471b5488479",
        "brain encoder component",
    ),
    "brain_mri_decoder.pt": (
        "brain/checkpoints/brain_decoder.pt",
        "ec01e35bd388cfa591f3e0dd79f60ae256b3d6f048321912370adb9ba8f9c988",
        "brain decoder / segmentation head",
    ),
    "brain_mri_embedding_head.pt": (
        "brain/checkpoints/brain_embedding_head.pt",
        "ee65d7abf4b98d01acd1df204cfaf4afdb9196b7e659d8f1b19c4899fc3b8278",
        "brain embedding head",
    ),
}

# Present in model_export/ and git-tracked in the repo already. Listed so
# --verify-only can confirm the shipped copy has not been corrupted or swapped.
TRACKED_CHECKS: dict[str, str] = {
    "best_model.pt": "6af40ad4eefbe4ef2fdf7399b911c4d8983ec6177eea695f81cb283189a5b4aa",
}

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _human(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _place(src: Path, dest: Path, expected: str, label: str) -> bool:
    """Verify src against `expected`, then move it to dest atomically."""
    actual = sha256_of(src)
    if actual != expected:
        print(f"  ERROR {label}: SHA-256 mismatch")
        print(f"        expected {expected}")
        print(f"        actual   {actual}")
        print("        refusing to install a weight file that does not match the "
              "recorded hash")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    print(f"  OK    {label} -> {dest.relative_to(REPO_ROOT)} ({_human(dest.stat().st_size)})")
    return True


def _download(url: str, into: Path) -> Path:
    print(f"  ..    downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            with into.open("wb") as fh:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    got += len(block)
                    if total:
                        pct = 100 * got / total
                        print(f"\r        {pct:5.1f}%  {_human(got)} / {_human(total)}",
                              end="", flush=True)
            if total:
                print()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    return into


def verify_present() -> int:
    """Report the state of every served weight without changing anything."""
    print("Verifying installed weights\n")
    missing = 0

    for name, expected in TRACKED_CHECKS.items():
        p = ARTIFACTS / name
        if not p.is_file():
            print(f"  MISSING  {name} (git-tracked - working tree is incomplete)")
            missing += 1
            continue
        actual = sha256_of(p)
        status = "OK     " if actual == expected else "CORRUPT"
        print(f"  {status}  {name} ({_human(p.stat().st_size)})")
        if actual != expected:
            print(f"           expected {expected}\n           actual   {actual}")
            missing += 1

    for name, (rel, expected, why) in ASSETS.items():
        p = ARTIFACTS / rel
        if not p.is_file():
            print(f"  MISSING  {rel}  - {why}")
            missing += 1
            continue
        actual = sha256_of(p)
        status = "OK     " if actual == expected else "CORRUPT"
        print(f"  {status}  {rel} ({_human(p.stat().st_size)})")
        if actual != expected:
            print(f"           expected {expected}\n           actual   {actual}")
            missing += 1

    print()
    if missing:
        print(f"{missing} weight(s) missing or corrupt.")
    else:
        print("All served weights present and matching their recorded hashes.")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch AURA served model weights")
    ap.add_argument("--from-local", type=Path, default=None,
                    help=f"directory holding the model_export files (default: {DEFAULT_LOCAL})")
    ap.add_argument("--url", default=os.environ.get("AURA_MODELS_URL"),
                    help="base URL serving the model_export file names "
                         "(env: AURA_MODELS_URL)")
    ap.add_argument("--verify-only", action="store_true",
                    help="report what is installed; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-install even if the file is already present and valid")
    args = ap.parse_args()

    if args.verify_only:
        return verify_present()

    local = args.from_local or DEFAULT_LOCAL
    print(f"artifacts   {ARTIFACTS}")
    print(f"local stage {local}  {'(found)' if local.is_dir() else '(absent)'}")
    print(f"release url {args.url or '(not set)'}\n")

    failed: list[str] = []
    for name, (rel, expected, why) in ASSETS.items():
        dest = ARTIFACTS / rel
        label = f"{name} [{why}]"

        if dest.is_file() and not args.force:
            if sha256_of(dest) == expected:
                print(f"  SKIP  {rel} already present and valid")
                continue
            print(f"  ..    {rel} present but hash mismatch; re-fetching")

        src = local / name
        if src.is_file():
            if not _place(src, dest, expected, label):
                failed.append(name)
            continue

        if args.url:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / name
                try:
                    _download(f"{args.url.rstrip('/')}/{name}", tmp)
                except RuntimeError as exc:
                    print(f"  ERROR {label}: {exc}")
                    failed.append(name)
                    continue
                if not _place(tmp, dest, expected, label):
                    failed.append(name)
            continue

        print(f"  ERROR {label}: not in the local stage and no --url/AURA_MODELS_URL set")
        failed.append(name)

    print()
    if failed:
        print(f"FAILED to install {len(failed)} weight(s): {', '.join(failed)}")
        print()
        print("Options:")
        print(f"  - point --from-local at a directory containing {', '.join(ASSETS)}")
        print("  - publish model_export/ as a GitHub Release and set AURA_MODELS_URL")
        print("  - build a chest-only image: docker build --build-arg AURA_SKIP_BRAIN=1 .")
        return 1

    print("All served weights installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
