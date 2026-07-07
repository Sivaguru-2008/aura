"""Small numerical helpers shared across engines. Pure numpy, no framework lock-in."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temperature, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy in bits."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    return float(-(p * np.log2(p)).sum())


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    return p / p.sum()


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
    """Free-energy OOD score: -T * logsumexp(logits / T).

    Lower (more negative) energy = more in-distribution; higher = more OOD.
    """
    logits = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    m = logits.max()
    lse = m + np.log(np.exp(logits - m).sum())
    return float(-temperature * lse)
