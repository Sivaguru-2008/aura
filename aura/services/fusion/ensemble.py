"""Deep-ensemble fusion — K independently-trained evidence→diagnosis heads.

Why this exists
---------------
A single fusion head gives a point posterior with no handle on *model*
uncertainty. A deep ensemble — the same architecture trained from different seeds
on bootstrap resamples — is the field-standard, well-calibrated way to get real
epistemic uncertainty: where the members *disagree*, the model is unsure.

The ensemble mean is itself a better-calibrated posterior, and it exposes the same
``logits(x)`` contract as the other fusion backends, so it drops straight into the
safety / explain / recommend engines. Member disagreement feeds
``uncertainty.ensemble_decomposition`` for the aleatoric/epistemic split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS
from common.mathx import softmax
from schemas.clinical import DIAGNOSES

MODEL_VERSION = "fusion-ensemble-v1"


class DeepEnsemble:
    def __init__(self, Ws: np.ndarray, bs: np.ndarray):
        self.Ws = np.asarray(Ws, dtype=float)     # (K, n_dx, n_channels)
        self.bs = np.asarray(bs, dtype=float)     # (K, n_dx)
        self.backend = "ensemble"
        self.model_version = MODEL_VERSION

    @classmethod
    def load(cls, path: Path | None = None) -> "DeepEnsemble | None":
        path = path or (ARTIFACTS / "fusion_ensemble.npz")
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["Ws"], d["bs"])

    @property
    def n_members(self) -> int:
        return len(self.Ws)

    def member_logits(self, x: np.ndarray) -> np.ndarray:
        """(K, n_dx) logits, one row per ensemble member."""
        x = np.asarray(x, dtype=float)
        return np.einsum("kdc,c->kd", self.Ws, x) + self.bs

    def member_posteriors(self, x: np.ndarray) -> np.ndarray:
        return np.array([softmax(r) for r in self.member_logits(x)])

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Bayesian model-averaged logits: log of the mean member posterior.

        Averaging in probability space (then back to logits) is the correct BMA;
        it keeps the ``logits(x)`` contract other engines depend on.
        """
        mean_p = self.member_posteriors(x).mean(axis=0)
        return np.log(np.clip(mean_p, 1e-12, 1.0))

    def fuse(self, x: np.ndarray, n_shots: int | None = None):
        P = self.member_posteriors(x)
        mean = P.mean(axis=0)
        std = P.std(axis=0)
        posterior_d = {d: float(mean[i]) for i, d in enumerate(DIAGNOSES)}
        std_d = {d: float(std[i]) for i, d in enumerate(DIAGNOSES)}
        return posterior_d, std_d
