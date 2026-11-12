import os
from pathlib import Path
from aura.common.config import ARTIFACTS
from aura.mimic.config import get_mimic_paths

#: Input resolution the backbone is trained and served at.
#:
#: 224 is the DenseNet-121 native size and what every shipped checkpoint was trained
#: at. It is also the binding constraint on small-lesion findings: an 8 mm nodule on a
#: ~2500 px film survives as roughly one pixel at 224, which is the mechanical reason
#: the nodule head sits near chance. Raising this is the single largest expected AUROC
#: gain available — but it REQUIRES a full retrain, because a checkpoint trained at one
#: resolution and fed another produces garbage.
#:
#: The value is therefore written into every checkpoint (``checkpoint.py``) and read
#: back by the serving wrapper (``inference.py``), so training and serving cannot
#: silently disagree. Override for a new run with AURA_CXR_IMG_SIZE=384.
DEFAULT_IMG_SIZE = 224


def configured_img_size() -> int:
    raw = os.environ.get("AURA_CXR_IMG_SIZE", "").strip()
    if not raw:
        return DEFAULT_IMG_SIZE
    try:
        v = int(raw)
    except ValueError:
        return DEFAULT_IMG_SIZE
    # DenseNet downsamples by 32; a non-multiple silently changes the feature-map
    # geometry Grad-CAM++ and the TV regulariser assume.
    return v if v >= 64 and v % 32 == 0 else DEFAULT_IMG_SIZE


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
        self.img_size = int(kwargs.get("img_size", configured_img_size()))
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        # Dataloader workers (0 = main thread; >0 overlaps JPEG decode with GPU).
        self.num_workers = kwargs.get("num_workers", 0)
        # Imbalance is corrected once, via pos_weight. Opt in to also oversample
        # rare positives (not recommended — see audit F5).
        self.use_sampler = kwargs.get("use_sampler", False)
        
        # Device
        self.device = kwargs.get("device", os.environ.get("AURA_DEVICE"))
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
