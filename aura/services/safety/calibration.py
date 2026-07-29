"""Calibration primitives, used by training to fit and by serving to apply.

  * Temperature scaling  — one scalar T minimizing NLL on a held-out set.
  * Conformal threshold  — split-conformal quantile giving marginal coverage.
  * OOD statistics       — energy mean/std on in-distribution data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from aura.common.config import ARTIFACTS
from aura.common.mathx import energy_score, softmax


@dataclass
class Calibration:
    temperature: float = 1.0
    conformal_qhat: float = 0.9        # nonconformity threshold (1 - p_true quantile)
    coverage: float = 0.90
    ood_mean: float = 0.0
    ood_std: float = 1.0
    ece: float = 0.0                    # reported expected calibration error post-scaling

    def save(self, path: Path | None = None, backend: str | None = None) -> None:
        """Persist. With ``backend`` set, writes the per-backend file as well.

        Both are written so an older reader that only knows ``safety.npz`` keeps
        working while a backend-aware reader gets the right numbers.
        """
        path = path or (ARTIFACTS / "safety.npz")
        payload = {k: np.array(v) for k, v in asdict(self).items()}
        np.savez(path, **payload)
        if backend:
            np.savez(backend_calibration_path(backend), **payload)

    @classmethod
    def load(cls, path: Path | None = None, backend: str | None = None
             ) -> "Calibration":
        """Load calibration, preferring the file fitted for ``backend``.

        Temperature, conformal threshold, and OOD energy statistics are all properties
        of *one model's* logit distribution, not of the pipeline. Serving the quantum
        VQC with the classical product-of-experts' temperature is not a small error:
        the two differ by roughly a factor of two on this system (0.94 vs 0.46), which
        is a large, systematic distortion of every downstream probability, conformal
        set, and abstention decision.

        Falls back to the shared ``safety.npz`` when no per-backend file exists, so a
        deployment that has not run the per-backend fit still starts.
        """
        if path is None and backend:
            candidate = backend_calibration_path(backend)
            if candidate.exists():
                path = candidate
        path = path or (ARTIFACTS / "safety.npz")
        if not path.exists():
            return cls()
        d = np.load(path)
        return cls(**{k: float(d[k]) for k in d.files})


def backend_calibration_path(backend: str) -> Path:
    """Artifact path holding the calibration fitted for ``backend``."""
    return ARTIFACTS / f"safety_{backend}.npz"


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """1-D search for T minimizing multiclass NLL."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def nll(logT: float) -> float:
        T = float(np.exp(logT))
        loss = 0.0
        for row, y in zip(logits, labels):
            p = softmax(row / T)
            loss -= np.log(max(p[y], 1e-12))
        return loss / len(labels)

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def fit_conformal(probs: np.ndarray, labels: np.ndarray, coverage: float) -> float:
    """Split-conformal: qhat = ceil((n+1)(coverage))/n empirical quantile of
    nonconformity scores s_i = 1 - p_i[true]. Serving set = {d : 1 - p[d] <= qhat}.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    scores = 1.0 - probs[np.arange(len(labels)), labels]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def ood_stats(logits: np.ndarray, temperature: float) -> tuple[float, float]:
    energies = np.array([energy_score(row, temperature) for row in logits])
    return float(energies.mean()), float(energies.std() + 1e-6)
