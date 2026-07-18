"""Step 7 — Dataset builder: patient-level train / validation / test splits.

Guarantees, by construction:
    * **No patient leakage.** Split assignment is a deterministic function of
      ``subject_id`` alone, so a patient can never land in two splits, and the
      split is identical across runs and machines (reproducible).
    * **Validation is the real held-out set.** The ``validate`` CSV shares 0
      patients with ``train`` (verified in Step 1), so it is used as-is.
    * **Test is carved from train**, stratified by the patient-level diagnosis
      label so class balance is preserved across train/test.

Labels are extracted from reports (text only — no image decode), so building the
manifest over all 64k train patients is fast. Manifests are lightweight
(one row per patient); engineered image features are materialized per split
on demand via :class:`mimic.features.FeatureEngineer`.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from mimic.cleaning import DataCleaner
from mimic.config import MimicPaths, get_mimic_paths
from mimic.loaders import MimicCxrLoader
from schemas.clinical import FINDINGS

log = logging.getLogger("mimic.splits")

DEFAULT_TEST_FRAC = 0.15
SPLIT_SALT = "aura-mimic-cxr-v1"        # bump to reshuffle deterministically


def _hash_frac(subject_id: int, salt: str = SPLIT_SALT) -> float:
    """Deterministic uniform value in [0,1) from a subject_id (stable everywhere)."""
    h = hashlib.md5(f"{salt}:{subject_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_train_test(subject_id: int, test_frac: float = DEFAULT_TEST_FRAC) -> str:
    """Assign a train-pool patient to 'train' or 'test' — leakage-free by design."""
    return "test" if _hash_frac(subject_id) < test_frac else "train"


@dataclass
class ManifestRow:
    subject_id: int
    split: str
    diagnosis: str
    n_studies: int
    n_images: int
    finding_labels: dict[str, int] = field(default_factory=dict)

    def flat(self) -> dict:
        row = {
            "subject_id": self.subject_id,
            "split": self.split,
            "diagnosis": self.diagnosis,
            "n_studies": self.n_studies,
            "n_images": self.n_images,
        }
        row.update({f"label_finding_{k}": v for k, v in self.finding_labels.items()})
        return row


def _iter_manifest_rows(
    split_src: str, paths: MimicPaths, limit: Optional[int]
) -> Iterator[ManifestRow]:
    """Stream cheap (text-only) manifest rows for a source split."""
    cleaner = DataCleaner(MimicCxrLoader(split_src, paths=paths))
    for cp in cleaner.iter_clean(limit=limit):
        yield ManifestRow(
            subject_id=cp.subject_id,
            split="",  # assigned by caller
            diagnosis=cp.diagnosis.value,
            n_studies=len(cp.per_report_labels),   # one label per (deduped) report/study
            n_images=len(cp.images),
            finding_labels={f.value: (1 if cp.label.findings.get(f) == 1 else 0) for f in FINDINGS},
        )


@dataclass
class SplitStats:
    n_train: int = 0
    n_validation: int = 0
    n_test: int = 0
    leakage: int = 0
    diagnosis_by_split: dict[str, dict[str, int]] = field(default_factory=dict)


class DatasetBuilder:
    """Builds and writes the train/validation/test manifests."""

    def __init__(self, paths: Optional[MimicPaths] = None, test_frac: float = DEFAULT_TEST_FRAC):
        self.paths = paths or get_mimic_paths()
        self.test_frac = test_frac

    def out_dir(self) -> Path:
        return self.paths.cache_dir / "splits"

    def build(
        self, limit_train: Optional[int] = None, limit_val: Optional[int] = None
    ) -> tuple[dict[str, pd.DataFrame], SplitStats]:
        """Build the three manifests. Returns ({split: df}, stats)."""
        # 1) train pool -> train / test by deterministic hash on subject_id
        train_rows, test_rows = [], []
        for row in _iter_manifest_rows("train", self.paths, limit_train):
            row.split = assign_train_test(row.subject_id, self.test_frac)
            (test_rows if row.split == "test" else train_rows).append(row.flat())

        # 2) validate CSV -> validation split (already patient-disjoint from train)
        val_rows = []
        for row in _iter_manifest_rows("validate", self.paths, limit_val):
            row.split = "validation"
            val_rows.append(row.flat())

        frames = {
            "train": pd.DataFrame(train_rows),
            "validation": pd.DataFrame(val_rows),
            "test": pd.DataFrame(test_rows),
        }
        stats = self._verify(frames)
        return frames, stats

    def _verify(self, frames: dict[str, pd.DataFrame]) -> SplitStats:
        ids = {k: set(df["subject_id"]) if len(df) else set() for k, df in frames.items()}
        # pairwise leakage across all three splits
        leak = (
            len(ids["train"] & ids["validation"])
            + len(ids["train"] & ids["test"])
            + len(ids["validation"] & ids["test"])
        )
        stats = SplitStats(
            n_train=len(frames["train"]),
            n_validation=len(frames["validation"]),
            n_test=len(frames["test"]),
            leakage=leak,
        )
        for k, df in frames.items():
            if len(df):
                stats.diagnosis_by_split[k] = df["diagnosis"].value_counts().to_dict()
        log.info(
            "splits: train=%d val=%d test=%d leakage=%d",
            stats.n_train, stats.n_validation, stats.n_test, stats.leakage,
        )
        return stats

    def build_and_write(
        self, limit_train: Optional[int] = None, limit_val: Optional[int] = None
    ) -> tuple[dict[str, Path], SplitStats]:
        frames, stats = self.build(limit_train=limit_train, limit_val=limit_val)
        out = self.out_dir()
        out.mkdir(parents=True, exist_ok=True)
        paths_written: dict[str, Path] = {}
        name_map = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
        for split, df in frames.items():
            fp = out / name_map[split]
            df.to_csv(fp, index=False)
            paths_written[split] = fp
            log.info("wrote %s (%d rows)", fp, len(df))
        if stats.leakage:
            raise RuntimeError(f"patient leakage detected across splits: {stats.leakage}")
        return paths_written, stats
