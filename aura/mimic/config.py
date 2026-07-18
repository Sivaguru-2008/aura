"""Dataset paths and schema constants for the real MIMIC-CXR corpus.

The dataset physically present on this machine is **MIMIC-CXR** (chest
radiographs + radiology reports), stored per-patient under::

    E:\\AURA\\datasets\\simhadrisadaram\\mimic-cxr-dataset\\versions\\2\\
        mimic_cxr_aug_train.csv        # one row per subject_id (patient)
        mimic_cxr_aug_validate.csv
        official_data_iccv_final\\files\\p10\\p10000032\\s5041.../<dicom>.jpg

Every path is overridable by environment variable so the same code runs against
a different mount (CI, another workstation, a future MIMIC-IV drop) without edits.
Nothing here imports heavy deps, so it is safe to import from anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Default location of the real corpus on this machine (discovered on disk).
# Override any of these with AURA_MIMIC_ROOT / AURA_MIMIC_IMAGES / AURA_MIMIC_*.
# --------------------------------------------------------------------------- #
_DEFAULT_ROOT = Path(
    os.environ.get(
        "AURA_MIMIC_ROOT",
        r"E:\AURA\datasets\simhadrisadaram\mimic-cxr-dataset\versions\2",
    )
)


@dataclass(frozen=True)
class MimicPaths:
    """Resolved, validated filesystem locations for the MIMIC-CXR corpus."""

    root: Path
    train_csv: Path
    validate_csv: Path
    images_root: Path                       # prefix that image paths are relative to
    cache_dir: Path                         # parquet / memmap cache (Step 14)

    # Columns present in the aug CSVs, in file order.
    columns: tuple[str, ...] = (
        "Unnamed: 0.1",
        "Unnamed: 0",
        "subject_id",
        "image",
        "view",
        "AP",
        "PA",
        "Lateral",
        "text",
        "text_augment",
    )
    # Columns whose values are Python-list literals (stringified lists).
    list_columns: tuple[str, ...] = ("image", "view", "AP", "PA", "Lateral", "text", "text_augment")
    primary_key: str = "subject_id"
    # Columns that are noise from a prior ``to_csv`` (pandas index dumps).
    index_junk_columns: tuple[str, ...] = ("Unnamed: 0.1", "Unnamed: 0")

    def exists_report(self) -> dict[str, bool]:
        """Cheap existence probe for every required path (used by verify.py)."""
        return {
            "root": self.root.is_dir(),
            "train_csv": self.train_csv.is_file(),
            "validate_csv": self.validate_csv.is_file(),
            "images_root": self.images_root.is_dir(),
        }

    def resolve_image(self, rel: str) -> Path:
        """Map a CSV image path (``files/p10/...jpg``) to an absolute path on disk."""
        return self.images_root / rel


def get_mimic_paths(root: Path | str | None = None) -> MimicPaths:
    """Build a :class:`MimicPaths` from the default (or an explicit) root.

    Raises nothing — existence is reported via :meth:`MimicPaths.exists_report`
    so callers (verification) can surface missing files as findings, not crashes.
    """
    base = Path(root) if root is not None else _DEFAULT_ROOT
    images_root = Path(
        os.environ.get("AURA_MIMIC_IMAGES", str(base / "official_data_iccv_final"))
    )
    cache_dir = Path(os.environ.get("AURA_MIMIC_CACHE", str(base / "_aura_cache")))
    return MimicPaths(
        root=base,
        train_csv=Path(os.environ.get("AURA_MIMIC_TRAIN", str(base / "mimic_cxr_aug_train.csv"))),
        validate_csv=Path(
            os.environ.get("AURA_MIMIC_VALIDATE", str(base / "mimic_cxr_aug_validate.csv"))
        ),
        images_root=images_root,
        cache_dir=cache_dir,
    )
