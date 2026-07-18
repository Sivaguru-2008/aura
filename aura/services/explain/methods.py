"""Gradient-based visual explanations for the CNN vision backbone.

Why this exists
---------------
The shipped explainer offers *only* occlusion saliency — slide a grey patch,
measure the probability drop. It is model-agnostic (a virtue we keep) but slow
(hundreds of forward passes) and coarse. This module adds the real,
publication-standard attribution methods that operate on a differentiable CNN:

  * Grad-CAM        — class-discriminative localization from conv gradients.
  * Grad-CAM++      — better localization for multiple / small lesions.
  * Integrated Gradients — axiomatic pixel attribution (completeness).
  * SmoothGrad      — noise-averaged gradients, sharper and less brittle.
  * Occlusion       — kept as the model-agnostic cross-check.

All return a map normalized to [0,1] on the display grid so the console overlays
them unchanged. torch is imported lazily; nothing here runs unless a CNN backbone
is active.
"""
from __future__ import annotations

import numpy as np

from schemas.clinical import Finding


def _resize01(m: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D map to size×size (bilinear via torch) and scale to [0,1]."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(m, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    a = t[0, 0].numpy()
    a = a - a.min()
    mx = a.max()
    return a / mx if mx > 1e-9 else a


def _target_index(backbone, finding: Finding) -> int | None:
    idx = backbone._finding_index.get(finding)
    return int(idx) if idx is not None else None


def grad_cam(backbone, img: np.ndarray, finding: Finding, plusplus: bool = True,
             out_size: int = 64) -> np.ndarray:
    """Grad-CAM / Grad-CAM++ heatmap for one finding's logit."""
    import torch

    idx = _target_index(backbone, finding)
    if idx is None:
        return np.zeros((out_size, out_size), dtype=float)

    model = backbone.model
    acts: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o
        o.register_hook(lambda g: grads.__setitem__("v", g))

    h = backbone.cam_layer.register_forward_hook(fwd_hook)
    try:
        x = backbone._to_tensor(img).requires_grad_(True)
        logits = backbone._pathology_logits(x)
        model.zero_grad(set_to_none=True)
        logits[0, idx].backward()
        A = acts["v"].detach()[0]          # (C,h,w)
        G = grads["v"].detach()[0]         # (C,h,w)
    finally:
        h.remove()

    if plusplus:
        g2, g3 = G ** 2, G ** 3
        denom = 2 * g2 + (A * g3).sum(dim=(1, 2), keepdim=True)
        alpha = g2 / torch.clamp(denom, min=1e-8)
        weights = (alpha * torch.clamp(G, min=0)).sum(dim=(1, 2))
    else:
        weights = G.mean(dim=(1, 2))
    cam = torch.clamp((weights[:, None, None] * A).sum(0), min=0).cpu().numpy()
    return _resize01(cam, out_size)


def _input_grad(backbone, x, idx):
    """Gradient of target logit wrt input tensor x (requires_grad set by caller)."""
    logits = backbone._pathology_logits(x)
    backbone.model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    logits[0, idx].backward()
    return x.grad.detach()


def integrated_gradients(backbone, img: np.ndarray, finding: Finding,
                         steps: int = 32, out_size: int = 64) -> np.ndarray:
    """Integrated Gradients with a black baseline (satisfies completeness)."""
    import torch

    idx = _target_index(backbone, finding)
    if idx is None:
        return np.zeros((out_size, out_size), dtype=float)
    x = backbone._to_tensor(img)
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        total += _input_grad(backbone, xi, idx)
    attr = ((x - baseline) * total / steps)[0].abs().sum(0).cpu().numpy()
    return _resize01(attr, out_size)


def smoothgrad(backbone, img: np.ndarray, finding: Finding,
               n: int = 25, sigma: float = 0.15, out_size: int = 64) -> np.ndarray:
    """SmoothGrad: average |∂logit/∂x| over Gaussian-noised inputs."""
    import torch

    idx = _target_index(backbone, finding)
    if idx is None:
        return np.zeros((out_size, out_size), dtype=float)
    x0 = backbone._to_tensor(img)
    scale = sigma * float(x0.abs().max().clamp(min=1e-3))
    acc = torch.zeros_like(x0)
    for _ in range(n):
        xi = (x0 + torch.randn_like(x0) * scale).clone().requires_grad_(True)
        acc += _input_grad(backbone, xi, idx).abs()
    attr = (acc / n)[0].sum(0).cpu().numpy()
    return _resize01(attr, out_size)


def occlusion(score_fn, img: np.ndarray, finding: Finding,
              window: int = 12, stride: int = 6, out_size: int = 64,
              baseline_val: float = 0.18) -> np.ndarray:
    """Model-agnostic occlusion saliency for one finding (works without gradients)."""
    from services.vision.features import _resize_to

    g = _resize_to(img, out_size).astype(float).copy()
    base_p = score_fn(g)[finding]
    sal = np.zeros((out_size, out_size))
    cnt = np.zeros((out_size, out_size))
    for r in range(0, out_size - 1, stride):
        for c in range(0, out_size - 1, stride):
            r1, c1 = min(out_size, r + window), min(out_size, c + window)
            patch = g[r:r1, c:c1].copy()
            g[r:r1, c:c1] = baseline_val
            drop = max(0.0, base_p - score_fn(g)[finding])
            g[r:r1, c:c1] = patch
            sal[r:r1, c:c1] += drop
            cnt[r:r1, c:c1] += 1
    cnt[cnt == 0] = 1
    sal /= cnt
    m = sal.max()
    return sal / m if m > 1e-9 else sal


def all_methods(backbone, img: np.ndarray, finding: Finding,
                out_size: int = 64, include_scorecam: bool = False) -> dict[str, np.ndarray]:
    """Compute the attribution methods for one finding. Robust to per-method failure.

    ``include_scorecam`` adds gradient-free Score-CAM. It is off by default because
    Score-CAM forwards one masked input per activation channel (heavier than the
    gradient methods); the explain/overlay path turns it on, the live serve path
    leaves it off to keep latency unchanged.
    """
    methods = [
        ("grad_cam", lambda: grad_cam(backbone, img, finding, plusplus=False, out_size=out_size)),
        ("grad_cam++", lambda: grad_cam(backbone, img, finding, plusplus=True, out_size=out_size)),
        ("integrated_gradients", lambda: integrated_gradients(backbone, img, finding, out_size=out_size)),
        ("smoothgrad", lambda: smoothgrad(backbone, img, finding, out_size=out_size)),
    ]
    if include_scorecam:
        from services.explain.scorecam import score_cam
        methods.append(
            ("score_cam", lambda: score_cam(backbone, img, finding, out_size=out_size))
        )
    out: dict[str, np.ndarray] = {}
    for name, fn in methods:
        try:
            out[name] = fn()
        except Exception as e:  # never let one method sink the explanation
            print(f"[explain.methods] {name} failed: {e}")
    return out
