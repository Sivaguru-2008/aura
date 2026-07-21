"""Production CNN vision backbone (GPU-aware).

Why this exists
---------------
The shipped vision model is per-finding *logistic regression over hand-crafted
anatomical features* (see ``features.py``). That is a toy: it cannot see texture,
context, or anything the ten hand features don't encode, and it has never seen a
real radiograph. This module is the production replacement — a real convolutional
network — dropped in behind the *same* ``score_findings`` / ``embedding`` contract
so nothing downstream changes.

Two backbones, both device-aware (CUDA when available, else CPU):

  * ``densenet_mimic`` — a genuine **DenseNet-121 trained on MIMIC-CXR**
    (torchxrayvision weights ``densenet121-res224-mimic_ch``). Real weights, no
    fabricated training run; multi-label over 18 pathologies, 1024-d deep
    embedding, and a conv feature map that Grad-CAM differentiates through.
  * ``timm`` — DenseNet-121 / EfficientNetV2 / ConvNeXt / Swin from ``timm`` with
    a fresh multi-label head over AURA's finding set, for fine-tuning on your own
    labelled CXR (see ``ml/training/train_cnn.py``). Weights load from
    ``artifacts/vision_cnn.pt`` when present.

The module imports torch lazily and never raises on import, so the numpy feature
path keeps working when torch/weights are unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from common.config import ARTIFACTS
from schemas.clinical import Finding

# torchxrayvision pathology name -> AURA Finding. Hyperinflation has no MIMIC
# label; VisionEngine fills it from the feature model so all 7 findings stay live.
_XRV_TO_FINDING: dict[str, Finding] = {
    "Lung Opacity": Finding.OPACITY,
    "Consolidation": Finding.CONSOLIDATION,
    "Effusion": Finding.EFFUSION,
    "Cardiomegaly": Finding.CARDIOMEGALY,
    "Lung Lesion": Finding.NODULE,
    "Pneumothorax": Finding.PNEUMOTHORAX,
}

# Order of the fine-tuned timm head (one sigmoid per finding).
TIMM_HEAD_FINDINGS: list[Finding] = list(Finding)

# timm architectures we support out of the box for fine-tuning.
TIMM_ARCHES = {
    "densenet121": "densenet121",
    "efficientnetv2": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "swin": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def select_device(prefer: str | None = None) -> str:
    """Pick the compute device once. Honours AURA_DEVICE / explicit override."""
    import os

    import torch

    want = prefer or os.environ.get("AURA_DEVICE")
    if want:
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


class CXRBackbone:
    """A real CNN behind the vision contract. Holds the torch model + preprocessing.

    ``kind``:
      * ``"densenet_mimic"`` — torchxrayvision DenseNet-121 (MIMIC-CXR weights).
      * ``"timm"``           — a timm backbone + fine-tuned multi-label head.
    """

    def __init__(
        self,
        kind: str = "densenet_mimic",
        weights: str = "densenet121-res224-mimic_ch",
        arch: str = "densenet121",
        device: str | None = None,
        state_dict_path: Path | None = None,
    ):
        import torch  # lazy

        self.kind = kind
        self.device = device or select_device()
        self.arch = arch
        self._torch = torch
        self.model_version = f"vision-{kind}-{arch}"

        if kind == "densenet_mimic":
            import torchxrayvision as xrv

            self.model = xrv.models.DenseNet(weights=weights).to(self.device).eval()
            self.pathologies = list(self.model.pathologies)
            self.input_mode = "xrv"
            self.embed_dim = 1024
            # DenseNet feature map (B,1024,7,7) — the Grad-CAM target.
            self.cam_layer = self.model.features
            self._finding_index = {
                f: self.pathologies.index(name)
                for name, f in _XRV_TO_FINDING.items()
                if name in self.pathologies
            }
        elif kind == "timm":
            import timm

            arch_id = TIMM_ARCHES.get(arch, arch)
            self.model = timm.create_model(
                arch_id, pretrained=True, num_classes=len(TIMM_HEAD_FINDINGS), in_chans=1
            ).to(self.device).eval()
            self.pathologies = [f.value for f in TIMM_HEAD_FINDINGS]
            self.input_mode = "imagenet1ch"
            self.embed_dim = self.model.num_features
            self.cam_layer = self._timm_cam_layer()
            self._finding_index = {f: i for i, f in enumerate(TIMM_HEAD_FINDINGS)}
            sd = state_dict_path or (ARTIFACTS / "vision_cnn.pt")
            if Path(sd).exists():
                state = torch.load(sd, map_location=self.device, weights_only=True)  # safe unpickler (audit §11.5)
                self.model.load_state_dict(state.get("model", state), strict=False)
                self.model_version = state.get("version", self.model_version)
        else:
            raise ValueError(f"unknown backbone kind: {kind}")

    # ---- preprocessing ---------------------------------------------------- #
    def _to_tensor(self, img: np.ndarray):
        """numpy HxW in [0,1] (any size) -> (1,C,224,224) tensor on device."""
        import torch
        import torch.nn.functional as F

        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        t = torch.from_numpy(a)[None, None].to(self.device)  # (1,1,H,W)
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        if self.input_mode == "xrv":
            # torchxrayvision expects intensities in [-1024, 1024].
            t = (t.clamp(0, 1) * 2.0 - 1.0) * 1024.0
        else:  # imagenet, single channel
            mean = float(np.mean(_IMAGENET_MEAN))
            std = float(np.mean(_IMAGENET_STD))
            t = (t.clamp(0, 1) - mean) / std
        return t

    # ---- forward passes --------------------------------------------------- #
    def _pathology_logits(self, x):
        return self.model(x)

    def pathology_probs(self, img: np.ndarray) -> dict[str, float]:
        """Raw multi-label sigmoid outputs over every pathology the CNN predicts."""
        torch = self._torch
        with torch.no_grad():
            x = self._to_tensor(img)
            logits = self._pathology_logits(x)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        out = {}
        for name, p in zip(self.pathologies, probs):
            if name:
                out[name] = float(p)
        return out

    def score_findings(self, img: np.ndarray) -> dict[Finding, float]:
        """Map CNN outputs onto AURA findings. Returns only findings the CNN covers."""
        torch = self._torch
        with torch.no_grad():
            x = self._to_tensor(img)
            probs = torch.sigmoid(self._pathology_logits(x))[0].detach().cpu().numpy()
        return {f: float(probs[i]) for f, i in self._finding_index.items()}

    def embedding(self, img: np.ndarray) -> np.ndarray:
        """Pooled deep features (embed_dim,) — the real replacement for the 10 hand feats."""
        torch = self._torch
        import torch.nn.functional as F

        with torch.no_grad():
            x = self._to_tensor(img)
            if self.kind == "densenet_mimic":
                feats = self.model.features(x)
                feats = F.relu(feats)
                pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
            else:
                pooled = self.model.forward_features(x)
                if pooled.ndim == 4:
                    pooled = F.adaptive_avg_pool2d(pooled, (1, 1)).flatten(1)
                elif pooled.ndim == 3:  # swin/vit token grid
                    pooled = pooled.mean(dim=1)
            return pooled[0].detach().cpu().numpy().astype(float)

    def _timm_cam_layer(self):
        """Best-effort last-conv layer for Grad-CAM across timm arches."""
        m = self.model
        for attr in ("norm5",):
            if hasattr(getattr(m, "features", object()), attr):
                return getattr(m.features, attr)
        if hasattr(m, "features"):
            return m.features
        # generic: last module with 4-D conv output
        last = None
        for mod in m.modules():
            if mod.__class__.__name__.lower().endswith(("conv2d", "block", "stage")):
                last = mod
        return last


_CACHE: dict[str, "CXRBackbone"] = {}


def get_backbone(kind: str, **kw) -> Optional["CXRBackbone"]:
    """Load (and cache) a backbone; return None on any failure so callers fall back."""
    key = f"{kind}:{kw.get('arch','')}:{kw.get('weights','')}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        bb = CXRBackbone(kind=kind, **kw)
    except Exception as e:  # missing weights / torch / network — degrade gracefully
        print(f"[vision.cnn] backbone '{kind}' unavailable ({e}); using feature model.")
        return None
    _CACHE[key] = bb
    return bb
