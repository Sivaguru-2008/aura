"""Learnable evidence-weighting fusion — attention-gated log-linear model.

Why this exists
---------------
The classical baseline is a product-of-experts: a *fixed* weight per (diagnosis,
channel). Its own docstring notes it "by construction cannot represent higher-order
evidence interactions." Clinically that matters — the weight the model should place
on an effusion depends on whether cardiomegaly is also present.

This model keeps the honest, inspectable log-linear head but adds a **context gate**:
each evidence channel is scaled by ``g_j(x) = sigmoid(a_j·x + c_j)`` before the
diagnosis weights apply, so the influence of a channel is modulated by the rest of
the evidence. Trained end-to-end (torch); served in pure numpy so it satisfies the
same ``logits(x)`` / ``fuse(x)`` contract as the other backends and drops into the
existing engines unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS
from common.mathx import sigmoid, softmax
from schemas.clinical import DIAGNOSES

MODEL_VERSION = "fusion-gated-v1"


class LearnableFusion:
    def __init__(self, A: np.ndarray, c: np.ndarray, W: np.ndarray, b: np.ndarray):
        self.A = np.asarray(A, dtype=float)      # (n_channels, n_channels) gate weights
        self.c = np.asarray(c, dtype=float)      # (n_channels,) gate bias
        self.W = np.asarray(W, dtype=float)      # (n_dx, n_channels)
        self.b = np.asarray(b, dtype=float)      # (n_dx,)
        self.backend = "learnable"
        self.model_version = MODEL_VERSION

    @classmethod
    def load(cls, path: Path | None = None) -> "LearnableFusion | None":
        path = path or (ARTIFACTS / "fusion_learnable.npz")
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["A"], d["c"], d["W"], d["b"])

    def gates(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.A @ np.asarray(x, dtype=float) + self.c)

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = self.gates(x)
        return self.W @ (g * x) + self.b

    def fuse(self, x: np.ndarray, n_shots: int | None = None):
        posterior = softmax(self.logits(x))
        posterior_d = {d: float(posterior[i]) for i, d in enumerate(DIAGNOSES)}
        std_d = {d: 0.0 for d in DIAGNOSES}
        return posterior_d, std_d


def train_learnable(X, y, epochs: int = 300, lr: float = 0.05, l2: float = 1e-4,
                    seed: int = 7):
    """Train the gated fusion head with torch autograd; return numpy params."""
    import torch

    torch.manual_seed(seed)
    n_dx = len(DIAGNOSES)
    d = X.shape[1]
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)

    A = torch.nn.Parameter(0.05 * torch.randn(d, d, dtype=torch.float64))
    c = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
    W = torch.nn.Parameter(0.1 * torch.randn(n_dx, d, dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(n_dx, dtype=torch.float64))
    opt = torch.optim.Adam([A, c, W, b], lr=lr, weight_decay=l2)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        opt.zero_grad()
        g = torch.sigmoid(Xt @ A.T + c)          # (n, d)
        logits = (g * Xt) @ W.T + b              # (n, n_dx)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return (A.detach().numpy(), c.detach().numpy(),
            W.detach().numpy(), b.detach().numpy())
