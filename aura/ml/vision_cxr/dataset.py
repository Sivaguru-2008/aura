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

def get_transforms(train: bool):
    """Albumentations pipeline: resizing, spatial and intensity augmentation for training, resize only for val."""
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, border_mode=0, p=0.5),
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

def load_mimic_samples(csv_path: Path, images_root: Path, limit: int = None):
    """Load samples from MIMIC CSV and pair with labels extracted from reports."""
    df = pd.read_csv(csv_path, dtype=str)
    if limit:
        df = df.head(limit)
        
    image_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        # 1. Parse report text and extract labels
        text_col = row.get("text", "")
        # text could be string representation of list of sentences, combine them
        sentences = safe_str_list(text_col)
        report_text = " ".join(sentences)
        
        # Extract findings using existing safety-aligned labeling engine
        rl = label_report(report_text)
        
        # Binary target vector for AURA findings (1.0 = present, 0.0 = absent/uncertain)
        y_vec = [1.0 if rl.findings.get(f) == 1 else 0.0 for f in FINDINGS]
        
        # 2. Extract image list and verify existence on disk
        img_col = row.get("image", "")
        img_list = safe_str_list(img_col)
        
        for img_rel_path in img_list:
            img_abs_path = images_root / img_rel_path
            if img_abs_path.is_file():
                image_paths.append(img_abs_path)
                labels.append(y_vec)
                
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
    
    # Weighted sampler to balance positive prevalence in training
    sampler = None
    if len(train_labels) > 0:
        pos = train_labels.sum(axis=0)
        inv = 1.0 / np.clip(pos, 1.0, None)
        weights = np.array([
            max(inv[train_labels[i] > 0]) if train_labels[i].any() else inv.min()
            for i in range(len(train_labels))
        ], dtype=np.float64)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=(config.device == "cuda"),
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.device == "cuda"),
        drop_last=False
    )
    
    return train_loader, val_loader, train_labels
