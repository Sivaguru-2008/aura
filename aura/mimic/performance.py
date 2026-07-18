"""Step 14 — Performance utilities for the MIMIC-CXR data path.

Adds the performance features the plan asks for, on top of what earlier steps
already provide:

    parquet cache        — already in loaders/features (``*_cached`` helpers)
    chunk processing     — already in the loaders (pandas ``chunksize``)
    memory mapping       — ``MemmapCache``: feature matrices persisted as ``.npy``
                           and reopened with ``mmap_mode='r'`` (no full RAM load)
    parallel dataloading — ``ParallelFeatureEngineer``: image-feature extraction
                           fanned out across threads (the JPG-decode bottleneck)
    GPU preprocessing    — ``gpu_standardize``: device-aware batch standardization

Parallelism is thread-based because the cost is JPG decode + numpy (both release
the GIL), so threads give real speedup without process-spawn overhead on Windows.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mimic.config import MimicPaths, get_mimic_paths
from mimic.features import FeatureRow, feature_names, patient_feature_row
from mimic.patient import Patient, iter_patients

log = logging.getLogger("mimic.performance")


# --------------------------------------------------------------------------- #
# Memory-mapped feature-matrix cache
# --------------------------------------------------------------------------- #
class MemmapCache:
    """Persist a float32 feature matrix and reopen it memory-mapped.

    A memmapped array is paged from disk on access, so a multi-GB feature matrix
    can be sampled/batched without ever being fully resident in RAM.
    """

    def __init__(self, paths: Optional[MimicPaths] = None) -> None:
        self.dir = (paths or get_mimic_paths()).cache_dir / "memmap"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _p(self, name: str) -> Path:
        return self.dir / f"{name}.npy"

    def save_matrix(self, name: str, X: np.ndarray) -> Path:
        arr = np.asarray(X, dtype=np.float32)
        np.save(self._p(name), arr)
        log.info("memmap cache wrote %s %s", name, arr.shape)
        return self._p(name)

    def load_matrix(self, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self._p(name), mmap_mode="r" if mmap else None)

    def exists(self, name: str) -> bool:
        return self._p(name).is_file()


# --------------------------------------------------------------------------- #
# Parallel feature materialization
# --------------------------------------------------------------------------- #
@dataclass
class BuildTiming:
    n: int
    seconds: float
    workers: int

    @property
    def rows_per_sec(self) -> float:
        return self.n / self.seconds if self.seconds else 0.0


class ParallelFeatureEngineer:
    """Materialize the feature frame with threaded image-feature extraction."""

    def __init__(self, paths: Optional[MimicPaths] = None) -> None:
        self.paths = paths or get_mimic_paths()

    def _patients(
        self, split: str, limit: Optional[int], subject_ids: Optional[set[int]]
    ) -> list[Patient]:
        # build_patient does NOT decode images (only feature extraction does),
        # so collecting Patient objects up front is cheap and lets us fan out.
        pts = []
        for p in iter_patients(split, paths=self.paths, limit=limit):
            if subject_ids is not None and p.subject_id not in subject_ids:
                continue
            pts.append(p)
        return pts

    def build_frame(
        self,
        split: str = "validate",
        limit: Optional[int] = None,
        subject_ids: Optional[set[int]] = None,
        workers: int = 8,
    ) -> tuple[pd.DataFrame, BuildTiming]:
        patients = self._patients(split, limit, subject_ids)
        t0 = time.perf_counter()
        rows: list[FeatureRow] = []
        if workers <= 1:
            for p in patients:
                r = patient_feature_row(p)
                if r is not None:
                    rows.append(r)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for r in ex.map(patient_feature_row, patients):
                    if r is not None:
                        rows.append(r)
        dt = time.perf_counter() - t0
        df = pd.DataFrame([r.flat() for r in rows])
        log.info("parallel feature build: %d rows in %.1fs (%d workers)", len(df), dt, workers)
        return df, BuildTiming(len(df), dt, workers)


# --------------------------------------------------------------------------- #
# GPU preprocessing
# --------------------------------------------------------------------------- #
def gpu_standardize(X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Standardize a feature matrix on GPU when available (falls back to CPU).

    Trivial arithmetic, but this is the seam where heavier GPU preprocessing
    (e.g. batched image transforms) would live; it is device-aware today.
    """
    try:
        import torch
    except ImportError:
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
        return (X - mu) / sd
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
    out = (t - t.mean(0, keepdim=True)) / (t.std(0, keepdim=True) + 1e-6)
    return out.cpu().numpy()
