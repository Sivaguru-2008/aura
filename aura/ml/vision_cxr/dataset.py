import os
import re
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from schemas.clinical import FINDINGS, Finding
from mimic.parsing import safe_str_list
from mimic.labeling import label_report

# MIMIC image paths embed the study id: files/pXX/pXXXXXXX/sYYYYYYYY/<dicom>.jpg
_STUDY_RE = re.compile(r"[\\/](s\d+)[\\/]")


def _study_of(rel_path) -> str | None:
    """Extract the MIMIC study id (sYYYYYYYY) from an image path, or None."""
    m = _STUDY_RE.search(str(rel_path).replace("\\", "/"))
    return m.group(1) if m else None


# Labeler selection: default v1 (mimic.labeling); AURA_LABELER=v2 uses the improved,
# validated labeler (mimic.labeling_v2 — fixes cardiomegaly/opacity/temporal-negation).
_USE_V2_LABELER = os.environ.get("AURA_LABELER", "v1").strip().lower() == "v2"
if _USE_V2_LABELER:
    from mimic.labeling_v2 import label_v2


def _label_vec(text: str) -> list[float]:
    """Binary AURA finding vector from one report (present=1, uncertain/absent=0)."""
    if _USE_V2_LABELER:
        d = label_v2(text)
        return [d[f.value] for f in FINDINGS]
    rl = label_report(text)
    return [1.0 if rl.findings.get(f) == 1 else 0.0 for f in FINDINGS]

def get_transforms(train: bool):
    """Albumentations pipeline: resizing + intensity/mild-spatial augmentation for
    training, resize only for validation.

    HorizontalFlip is deliberately **omitted**: a chest radiograph has a fixed situs
    (heart and aortic arch on the left, three-lobed right lung, gastric bubble on the
    left). Mirroring it teaches the model an anatomically impossible layout and is a
    known CXR anti-pattern that directly harms laterality-dependent findings such as
    cardiomegaly and effusion side (audit §5.6 / §12.2). Rotation is kept but tightened
    to ±8° — films are acquired in a standardised upright position, so large rotations
    are unrealistic.
    """
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=8, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.2),
        ])
    return A.Compose([A.Resize(224, 224)])

class ChestXrayDataset(Dataset):
    """Dataset class for MIMIC-CXR images and their corresponding pathology labels."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # ImageNet channel-averaged mean and std (since our model is grayscale)
        self.mean = 0.449
        self.std = 0.226

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for unreadable or missing images
            img = np.zeros((224, 224), dtype=np.uint8)
            
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
            
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor, add channel dimension: (1, 224, 224)
        t = torch.from_numpy(img).unsqueeze(0)
        
        # Standard normalization: (t - mean) / std
        t = (t - self.mean) / self.std
        
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return t, y

def load_mimic_samples(csv_path: Path, images_root: Path, limit: int = None,
                       per_study: bool = True):
    """Load MIMIC images paired with report-derived finding labels.

    Each CSV row is one *subject* with parallel lists of images and reports (one
    report per study). ``per_study=True`` (default) labels every image with the
    report of **its own study** — the study id is parsed from the image path
    (``.../sYYYYYYYY/...``) and studies align 1:1 to the report list (verified: 100%
    of validate rows). ``per_study=False`` reproduces the legacy behaviour, which
    concatenated *all* of a subject's reports and applied one smeared label vector to
    every image — cross-study label contamination that inflated positives ~3–13× and
    leaked patient-level signal into the per-image task (see ``vision_audit.md`` F3).
    """
    df = pd.read_csv(csv_path, dtype=str)
    if limit:
        df = df.head(limit)

    image_paths: list[Path] = []
    labels: list[list[float]] = []
    n_fallback = 0

    for _, row in df.iterrows():
        # Reports: one entry per study (each entry may itself be a stringified
        # sentence list). Preserve row order — it matches study order in `image`.
        raw_reports = safe_str_list(row.get("text", ""))
        reports = [
            " ".join(safe_str_list(r)) if str(r).startswith("[") else str(r)
            for r in raw_reports
        ]
        img_list = safe_str_list(row.get("image", ""))

        # Legacy smeared vector (all reports joined) — fallback + per_study=False.
        smear_vec = _label_vec(" ".join(reports))

        study_order = list(dict.fromkeys(s for s in map(_study_of, img_list) if s))
        study_label: dict[str, list[float]] | None = None
        if per_study and study_order and len(study_order) == len(reports):
            study_label = {s: _label_vec(reports[i]) for i, s in enumerate(study_order)}
        elif per_study and study_order:
            n_fallback += 1  # counts/ids didn't line up — fall back to smeared

        for img_rel_path in img_list:
            img_abs_path = images_root / img_rel_path
            if img_abs_path.is_file():
                if study_label is not None:
                    y_vec = study_label.get(_study_of(img_rel_path), smear_vec)
                else:
                    y_vec = smear_vec
                image_paths.append(img_abs_path)
                labels.append(y_vec)

    if per_study and n_fallback:
        print(f"[Dataset] per-study labels: {n_fallback} rows fell back to combined "
              f"labels (study/report count mismatch)")
    return image_paths, np.array(labels, dtype=np.float32)

def build_loaders(config, limit=None):
    """Builds and returns training and validation dataloaders."""
    paths = config.mimic_paths
    
    # Load training samples
    print(f"[Dataset] Loading train samples from {paths.train_csv.name}...")
    train_paths, train_labels = load_mimic_samples(
        paths.train_csv, paths.images_root, limit=limit
    )
    
    # Load validation samples
    print(f"[Dataset] Loading validation samples from {paths.validate_csv.name}...")
    val_paths, val_labels = load_mimic_samples(
        paths.validate_csv, paths.images_root, limit=limit
    )
    
    print(f"[Dataset] Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    train_ds = ChestXrayDataset(train_paths, train_labels, transform=get_transforms(train=True))
    val_ds = ChestXrayDataset(val_paths, val_labels, transform=get_transforms(train=False))
    
    # Class imbalance is corrected in EXACTLY ONE place. By default that is
    # `pos_weight` in the BCE loss (see train.py). The WeightedRandomSampler below
    # is opt-in (`config.use_sampler=True`) because stacking both over-corrects:
    # it inflates recall and destroys calibration on rare findings (audit F5).
    sampler = None
    if getattr(config, "use_sampler", False) and len(train_labels) > 0:
        pos = train_labels.sum(axis=0)
        inv = 1.0 / np.clip(pos, 1.0, None)
        weights = np.array([
            max(inv[train_labels[i] > 0]) if train_labels[i].any() else inv.min()
            for i in range(len(train_labels))
        ], dtype=np.float64)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    print(f"[Dataset] weighted sampler: {'ON' if sampler is not None else 'OFF (pos_weight only)'}")

    # JPEG decode dominates step time; workers overlap it with GPU compute.
    nw = int(getattr(config, "num_workers", 0) or 0)
    extra = {"persistent_workers": True, "prefetch_factor": 4} if nw > 0 else {}

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=nw,
        pin_memory=(config.device == "cuda"),
        drop_last=False,
        **extra,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=(config.device == "cuda"),
        drop_last=False,
        **extra,
    )

    return train_loader, val_loader, train_labels
