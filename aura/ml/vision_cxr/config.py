import os
from pathlib import Path
from common.config import ARTIFACTS
from mimic.config import get_mimic_paths

class TrainConfig:
    def __init__(self, **kwargs):
        # Retrieve MIMIC-CXR paths dynamically (respects environment overrides)
        self.mimic_paths = get_mimic_paths()
        
        # Training parameters
        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 16)
        self.lr = kwargs.get("lr", 3e-4)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.patience = kwargs.get("patience", 4)
        self.seed = kwargs.get("seed", 7)
        self.amp = kwargs.get("amp", True)
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        
        # Device
        self.device = kwargs.get("device", "cuda" if os.environ.get("AURA_DEVICE") == "cuda" else None)
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Limits (for debugging / quick smoke testing)
        self.limit = kwargs.get("limit", None)
        
        # Output directory
        self.out_dir = Path(kwargs.get("out_dir", ARTIFACTS))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Output filenames
        self.best_model_name = "best_model.pt"
        self.last_checkpoint_name = "last_checkpoint.pt"
        self.history_csv_name = "history.csv"
        
        # Tensorboard and plots directories
        self.tb_dir = self.out_dir / "tensorboard"
        self.plots_dir = self.out_dir / "plots"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_model_path(self):
        return self.out_dir / self.best_model_name

    @property
    def last_checkpoint_path(self):
        return self.out_dir / self.last_checkpoint_name

    @property
    def history_csv_path(self):
        return self.out_dir / self.history_csv_name
