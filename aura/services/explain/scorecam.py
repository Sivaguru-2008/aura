"""Score-CAM — gradient-free class-activation mapping for the CNN backbone.

Why this exists
---------------
Grad-CAM / Grad-CAM++ weight each activation channel by its *gradient*. Score-CAM
(Wang et al., CVPR-W 2020) instead weights each channel by the **actual change in
the target score** when the input is masked by that channel's (upsampled) activation
map. It needs no gradients — only forward passes — so it is robust to gradient
saturation and gives smoother, often better-localized maps. It complements the
gradient methods already in ``methods.py``.

The map is normalized to [0,1] on the display grid so the console/overlays consume
it unchanged. torch is imported lazily; nothing here runs without a CNN backbone.
"""
from __future__ import annotations

import numpy as np

from schemas.clinical import Finding
from services.explain.methods import _resize01, _target_index


def score_cam(backbone, img: np.ndarray, finding: Finding,
              batch: int = 64, max_channels: int | None = None,
              out_size: int = 64) -> np.ndarray:
    """Score-CAM heatmap for one finding.

    Parameters
    ----------
    batch        : masked inputs are forwarded in mini-batches (GPU-friendly).
    max_channels : cap the number of activation channels scored (top-activation
                   channels first) to bound cost on very wide feature maps; None
                   uses every channel.
    """
    import torch
    import torch.nn.functional as F

    idx = _target_index(backbone, finding)
    if idx is None:
        return np.zeros((out_size, out_size), dtype=float)

    model = backbone.model
    acts: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o.detach()

    h = backbone.cam_layer.register_forward_hook(fwd_hook)
    try:
        with torch.no_grad():
            x = backbone._to_tensor(img)               # (1,C_in,224,224)
            base_logits = backbone._pathology_logits(x)
            A = acts["v"][0]                            # (C,h,w)
            C, Hf, Wf = A.shape
            H, W = x.shape[-2], x.shape[-1]

            # Upsample every activation channel to input size, min-max per channel.
            masks = F.interpolate(A[None], size=(H, W), mode="bilinear",
                                  align_corners=False)[0]      # (C,H,W)
            flat = masks.view(C, -1)
            mn = flat.min(dim=1, keepdim=True).values
            mx = flat.max(dim=1, keepdim=True).values
            norm = (flat - mn) / (mx - mn + 1e-9)
            masks = norm.view(C, 1, H, W)

            # Optionally keep only the most active channels (cost control).
            if max_channels is not None and C > max_channels:
                energy = flat.sum(dim=1)
                keep = torch.topk(energy, max_channels).indices
                masks = masks[keep]
                A = A[keep]
                C = masks.shape[0]

            base_score = torch.sigmoid(base_logits[0, idx])
            weights = torch.zeros(C, device=x.device)
            for start in range(0, C, batch):
                mb = masks[start:start + batch]                # (b,1,H,W)
                masked = x * mb                                # broadcast over channels
                logits = backbone._pathology_logits(masked)    # (b,n_findings)
                s = torch.sigmoid(logits[:, idx])
                # Score-CAM increment: channel importance = target activation drop-in.
                weights[start:start + batch] = torch.relu(s - 0.0)
            weights = weights * base_score / (base_score + 1e-9)

            cam = torch.relu((weights[:, None, None] * A).sum(0)).cpu().numpy()
    finally:
        h.remove()

    return _resize01(cam, out_size)
