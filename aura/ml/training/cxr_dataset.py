"""CXR datasets + GPU dataloaders for training the CNN vision backbone.

Provides a real-image dataset driven by a CSV/JSONL manifest (path + multi-label
targets over AURA's findings) with Albumentations augmentation, an LRU decode cache,
and class-balanced weighted sampling — and a synthetic dataset that draws from the
existing world generator so the whole training harness is runnable end-to-end with
no downloads. Both yield ``(image_tensor[1,H,W], target[len(FINDINGS)])``.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from schemas.clinical import FINDINGS, Finding

IMG_SIZE = 224
_MEAN, _STD = 0.5, 0.5


def build_transforms(train: bool):
    """Albumentations pipeline. Photometric+geometric aug for train, resize-only for val."""
    import albumentations as A

    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.GaussNoise(p=0.2),
        ])
    return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])


def _to_tensor(img01: np.ndarray) -> torch.Tensor:
    """HxW float [0,1] -> (1,H,W) normalized tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img01, dtype=np.float32))[None]
    return (t - _MEAN) / _STD


class CXRManifestDataset(Dataset):
    """Real radiographs from a manifest of (path, label_vector).

    ``rows``: list of dicts {"path": str, "labels": [0/1]*len(FINDINGS)}.
    ``cache``: decode-cache size (0 disables) — decoding DICOM dominates I/O.
    """

    def __init__(self, rows: list[dict], train: bool = True, cache: int = 512):
        self.rows = rows
        self.tf = build_transforms(train)
        self.labels = np.array([r["labels"] for r in rows], dtype=np.float32)
        if cache:
            self._load = functools.lru_cache(maxsize=cache)(self._load_raw)
        else:
            self._load = self._load_raw

    def _load_raw(self, path: str) -> np.ndarray:
        from services.vision.io import load_cxr
        return load_cxr(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = self._load(self.rows[i]["path"])
        aug = self.tf(image=(img * 255).astype(np.uint8))["image"].astype(np.float32) / 255.0
        return _to_tensor(aug), torch.from_numpy(self.labels[i])


class SyntheticCXRDataset(Dataset):
    """Synthetic world as a multi-label finding dataset (harness smoke test)."""

    def __init__(self, n: int, seed: int = 7, train: bool = True):
        from ml.data import make_dataset
        self.samples = make_dataset(n, seed=seed)
        self.tf = build_transforms(train)
        self.labels = np.array(
            [[1.0 if s.findings[f] >= 0.5 else 0.0 for f in FINDINGS] for s in self.samples],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = self.samples[i].image
        aug = self.tf(image=(img * 255).astype(np.uint8))["image"].astype(np.float32) / 255.0
        return _to_tensor(aug), torch.from_numpy(self.labels[i])


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Balance by inverse finding-prevalence so rare findings are seen enough.

    Each sample's weight is the max inverse-frequency over its positive findings
    (a positive-heavy heuristic that works for multi-label CXR).
    """
    labels = np.asarray(labels, dtype=np.float32)
    pos = labels.sum(axis=0)
    inv = 1.0 / np.clip(pos, 1.0, None)
    w = np.array([max(inv[labels[i] > 0]) if labels[i].any() else inv.min()
                  for i in range(len(labels))], dtype=np.float64)
    return WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)


def make_loader(ds, batch: int, train: bool, device: str, balanced: bool = True) -> DataLoader:
    sampler = None
    if train and balanced:
        sampler = make_weighted_sampler(ds.labels)
    return DataLoader(
        ds, batch_size=batch, sampler=sampler, shuffle=(train and sampler is None),
        num_workers=0, pin_memory=(device == "cuda"), drop_last=False,
    )
