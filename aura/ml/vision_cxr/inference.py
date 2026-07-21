import torch
import torch.nn.functional as F
import cv2
import numpy as np
from ml.vision_cxr.model import DenseNet121CXR
from schemas.clinical import FINDINGS, Finding

class VisionModel:
    """Wrapper for the trained production DenseNet121 chest X-ray model.
    
    Conforms to the CXRBackbone contract from services/vision/cnn.py.
    """
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenseNet121CXR(num_classes=len(FINDINGS))
        
        # Load weights. weights_only=True runs torch's safe unpickler so a tampered
        # checkpoint on the serving path cannot execute arbitrary code (audit
        # §11.5); the checkpoint is a plain dict of tensors + scalars, so this is
        # loss-free.
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.pathologies = [f.value for f in FINDINGS]
        self.model_version = "vision-cxr-densenet121-production"
        self._finding_index = {f: idx for idx, f in enumerate(FINDINGS)}

        # Target layer for Grad-CAM++ (last conv layer of DenseNet features block)
        self.cam_layer = self.model.densenet.features.norm5

        # Per-finding calibration (audit: computed but not applied). Loaded from the
        # reproducible artifact written by ml.evaluation.vision_calibration. Prefers
        # Platt scaling sigmoid(a·logit + b) — which subsumes temperature — and
        # falls back to per-finding temperature, then to raw sigmoid when absent.
        self._platt_a, self._platt_b, self._finding_temps = self._load_calibration()

    @staticmethod
    def _load_calibration():
        """Return (platt_a, platt_b, temps) vectors aligned to FINDINGS, any of which
        may be None when that calibration form is unavailable."""
        import json
        from pathlib import Path
        from common.config import ARTIFACTS

        path = ARTIFACTS / "vision_serving_calibration.json"
        if not Path(path).exists():
            return None, None, None
        try:
            d = json.loads(Path(path).read_text())
            platt = d.get("per_finding_platt")
            a = b = temps = None
            if platt:
                a = np.array([float(platt.get(f.value, {}).get("a", 1.0)) for f in FINDINGS],
                             dtype=np.float32)
                b = np.array([float(platt.get(f.value, {}).get("b", 0.0)) for f in FINDINGS],
                             dtype=np.float32)
            t = d.get("per_finding_temperature")
            if t:
                temps = np.array([float(t.get(f.value, 1.0)) for f in FINDINGS], dtype=np.float32)
                temps[temps <= 0] = 1.0
            return a, b, temps
        except Exception as e:
            print(f"[VisionModel] failed to load serving calibration: {e}")
            return None, None, None

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Preprocesses image to tensor (grayscale, 224x224, ImageNet normalization)."""
        import torch
        import cv2
        
        # Resize to 224x224
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            
        # Convert to grayscale if it is RGB/BGR
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
            
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Standard channel-averaged ImageNet normalization
        mean = 0.449
        std = 0.226
        t = (t - mean) / std
        return t

    def _pathology_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model to obtain raw logits."""
        return self.model(x)

    def score_findings(self, img: np.ndarray) -> dict[Finding, float]:
        """Calibrated probabilities for all AURA findings.

        Applies per-finding temperature scaling when a serving calibration artifact
        is present — sigmoid(logit / T_f) — so the finding probabilities are
        calibrated (audit: this calibration was previously computed but never
        applied, leaving vision ECE ~0.34). Falls back to the raw sigmoid when no
        calibration has been fitted, preserving the original behaviour.
        """
        with torch.no_grad():
            x = self._to_tensor(img)
            logits = self._pathology_logits(x)[0].cpu().numpy()
        if self._platt_a is not None and self._platt_b is not None:
            logits = self._platt_a * logits + self._platt_b       # Platt: sigmoid(a·z + b)
        elif self._finding_temps is not None:
            logits = logits / self._finding_temps                  # temperature fallback
        probs = 1.0 / (1.0 + np.exp(-logits))
        return {f: float(probs[idx]) for idx, f in enumerate(FINDINGS)}

    def embedding(self, img: np.ndarray) -> np.ndarray:
        """Extracts the 1024-dimensional feature embedding from the DenseNet backbone."""
        with torch.no_grad():
            x = self._to_tensor(img)
            # Forward pass up to features block
            features = self.model.densenet.features(x)
            features = F.relu(features)
            # Global Average Pooling
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            return pooled[0].cpu().numpy().astype(float)
