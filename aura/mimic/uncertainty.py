"""Step 12 — Uncertainty quantification for the MIMIC-CXR tabular models.

Wires AURA's existing, tested safety primitives (``services/safety``) to the new
tabular models rather than reimplementing them:

    * Deep ensembles      — multi-seed models + ``ensemble_decomposition`` (epistemic
                            / aleatoric / mutual-information BALD)
    * Monte-Carlo dropout — ``enable_dropout`` + repeated MLP forward passes
    * Temperature scaling — ``fit_temperature`` on validation logits
    * Conformal prediction — ``fit_conformal`` (marginal) + Mondrian (class-conditional)

Calibration quality is measured with the existing ``expected_calibration_error``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from services.safety.calibration import (
    expected_calibration_error,
    fit_conformal,
    fit_temperature,
)
from services.safety.uncertainty import (
    ensemble_decomposition,
    mondrian_qhats,
    mondrian_set,
)

log = logging.getLogger("mimic.uncertainty")

_EPS = 1e-9


def _to_logits(proba: np.ndarray) -> np.ndarray:
    """Pseudo-logits from probabilities (log), so temperature scaling applies to
    any classifier that only exposes ``predict_proba`` (e.g. gradient boosting)."""
    return np.log(np.clip(proba, _EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Deep ensembles
# --------------------------------------------------------------------------- #
@dataclass
class EnsemblePrediction:
    proba: np.ndarray                  # mean over members (N, C)
    member_proba: np.ndarray           # (M, N, C)
    epistemic: float                   # mean top-class disagreement across members
    predictive_entropy: float
    mutual_information: float           # BALD


class DeepEnsemble:
    """Averages several fitted estimators and reports epistemic uncertainty."""

    def __init__(self, models: list) -> None:
        if not models:
            raise ValueError("DeepEnsemble needs at least one model")
        self.models = models

    def predict(self, X: np.ndarray, n_classes: int) -> EnsemblePrediction:
        members = np.stack([self._member_proba(m, X, n_classes) for m in self.models])
        mean = members.mean(axis=0)
        # per-sample uncertainty decomposition, averaged over the batch
        epi, ent, mi = [], [], []
        for i in range(members.shape[1]):
            d = ensemble_decomposition(members[:, i, :])
            epi.append(d.get("epistemic", 0.0))
            ent.append(d.get("predictive_entropy", 0.0))
            mi.append(d.get("mutual_information", d.get("epistemic_mi", 0.0)))
        return EnsemblePrediction(
            proba=mean, member_proba=members,
            epistemic=float(np.mean(epi)),
            predictive_entropy=float(np.mean(ent)),
            mutual_information=float(np.mean(mi)),
        )

    @staticmethod
    def _member_proba(model, X, n_classes: int) -> np.ndarray:
        p = model.predict_proba(X)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        if n_classes > 2 and p.shape[1] != n_classes and hasattr(model, "classes_"):
            full = np.zeros((len(X), n_classes))
            for j, c in enumerate(model.classes_):
                full[:, int(c)] = p[:, j]
            p = full
        return p


def train_gbm_ensemble(train_ds, val_ds, k: int = 5, backend: str = "auto") -> DeepEnsemble:
    """Train ``k`` gradient boosters with different seeds → a deep ensemble."""
    from mimic.training import GBMTrainer
    models = []
    for s in range(k):
        t = GBMTrainer(train_ds, backend=backend, seed=100 + s)
        t.train(val_ds)
        models.append(t.model)
    log.info("trained %d-member GBM ensemble (backend=%s)", k, models and backend)
    return DeepEnsemble(models)


# --------------------------------------------------------------------------- #
# Monte-Carlo dropout (MLP)
# --------------------------------------------------------------------------- #
def mc_dropout_proba(trainer, X_raw: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """MC-dropout predictive mean + per-sample epistemic std for the MLP.

    Returns (mean_proba (N,C), epistemic_std (N,)). Reuses ``enable_dropout``.
    """
    torch = trainer.torch
    from services.safety.uncertainty import enable_dropout

    X = ((np.asarray(X_raw, float) - trainer.mu) / trainer.sd).astype("float32")
    Xt = torch.tensor(X, device=trainer.device)
    trainer.net.eval()
    enable_dropout(trainer.net)                      # keep dropout active at inference
    passes = []
    with torch.no_grad():
        for _ in range(k):
            passes.append(torch.softmax(trainer.net(Xt), dim=1).cpu().numpy())
    stack = np.stack(passes)                          # (k, N, C)
    mean = stack.mean(0)
    top = mean.argmax(1)
    epi = stack.std(0)[np.arange(len(top)), top]
    return mean, epi


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
class TemperatureScaler:
    """Fit one scalar T on a validation set; divides logits by T to calibrate."""

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, proba_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        self.T = float(fit_temperature(_to_logits(proba_val), np.asarray(y_val, int)))
        log.info("fitted temperature T=%.3f", self.T)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return _softmax(_to_logits(proba) / max(self.T, _EPS))

    def calibration_gain(self, proba: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y = np.asarray(y, int)
        before = expected_calibration_error(proba, y)
        after = expected_calibration_error(self.transform(proba), y)
        return {"ece_before": round(before, 4), "ece_after": round(after, 4), "T": round(self.T, 3)}


# --------------------------------------------------------------------------- #
# Conformal prediction
# --------------------------------------------------------------------------- #
@dataclass
class ConformalResult:
    qhat: float
    coverage_target: float
    empirical_coverage: float
    avg_set_size: float
    method: str


class ConformalPredictor:
    """Split-conformal prediction sets (marginal or Mondrian class-conditional)."""

    def __init__(self, coverage: float = 0.9, mondrian: bool = False) -> None:
        self.coverage = coverage
        self.mondrian = mondrian
        self.qhat: Optional[float] = None
        self.qhats: Optional[np.ndarray] = None
        self.n_classes: int = 0

    def fit(self, proba_calib: np.ndarray, y_calib: np.ndarray) -> "ConformalPredictor":
        y = np.asarray(y_calib, int)
        self.n_classes = proba_calib.shape[1]
        if self.mondrian:
            self.qhats = mondrian_qhats(proba_calib, y, self.coverage, self.n_classes)
        else:
            self.qhat = fit_conformal(proba_calib, y, self.coverage)
        return self

    def predict_set(self, proba_row: np.ndarray) -> list[int]:
        if self.mondrian:
            return mondrian_set(proba_row, self.qhats)
        keep = [c for c in range(len(proba_row)) if (1 - proba_row[c]) <= self.qhat]
        return keep or [int(np.argmax(proba_row))]

    def evaluate(self, proba: np.ndarray, y: np.ndarray) -> ConformalResult:
        y = np.asarray(y, int)
        sets = [self.predict_set(p) for p in proba]
        covered = float(np.mean([y[i] in sets[i] for i in range(len(y))]))
        size = float(np.mean([len(s) for s in sets]))
        return ConformalResult(
            qhat=float(self.qhat) if self.qhat is not None else float("nan"),
            coverage_target=self.coverage, empirical_coverage=round(covered, 4),
            avg_set_size=round(size, 4), method="mondrian" if self.mondrian else "marginal",
        )
