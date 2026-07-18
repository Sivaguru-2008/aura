"""Step 11 — Explainability for the MIMIC-CXR tabular models.

Complements AURA's existing image/evidence explainers (``services/explain``) with
attributions for the new tabular models:

    * feature importance   — native (gain) + model-agnostic permutation importance
    * SHAP                  — the ``shap`` library when installed; otherwise a
                              transparent occlusion-to-baseline fallback (same idea:
                              marginal contribution of each feature vs a baseline)
    * integrated gradients — for the torch MLP (Sundararajan et al.)
    * counterfactuals      — greedy minimal feature moves that flip the prediction

All model-agnostic (permutation / occlusion / counterfactual) work on any fitted
estimator exposing ``predict_proba``. Integrated gradients needs the MLP.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("mimic.explain")


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def native_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Native feature importance (gain/split) if the estimator exposes it."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp, float)
    return dict(sorted(zip(feature_names, imp.tolist()), key=lambda kv: -kv[1]))


def permutation_importance(
    model, X: np.ndarray, y: np.ndarray, feature_names: list[str],
    n_repeats: int = 5, seed: int = 7,
) -> dict[str, float]:
    """Model-agnostic permutation importance (drop in accuracy when a col is shuffled)."""
    from sklearn.inspection import permutation_importance as _pi

    r = _pi(model, X, y, n_repeats=n_repeats, random_state=seed, scoring="accuracy")
    return dict(sorted(zip(feature_names, r.importances_mean.tolist()), key=lambda kv: -kv[1]))


def occlusion_attribution(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray, baseline: np.ndarray, target: int,
) -> np.ndarray:
    """Local attribution: change in P(target) when each feature is set to baseline.

    A transparent, SHAP-style marginal-contribution estimate that needs no extra
    dependency. Positive value => the feature's actual value *raises* P(target).
    """
    x = np.asarray(x, float)
    base_p = float(predict_proba(x[None, :])[0, target])
    attr = np.zeros_like(x)
    for j in range(len(x)):
        xj = x.copy()
        xj[j] = baseline[j]
        attr[j] = base_p - float(predict_proba(xj[None, :])[0, target])
    return attr


def shap_values(model, X_background: np.ndarray, X_explain: np.ndarray):
    """SHAP values via the ``shap`` library when available, else occlusion fallback.

    Returns (values, method) where method is "shap" or "occlusion".
    """
    if shap_available():
        import shap
        try:
            explainer = shap.Explainer(model.predict_proba, X_background)
            return explainer(X_explain).values, "shap"
        except Exception as e:  # pragma: no cover - shap backend quirks
            log.warning("shap failed (%s); using occlusion fallback", e)
    # fallback: occlusion vs the background mean, for the predicted class of each row
    base = X_background.mean(0)
    proba = model.predict_proba(X_explain)
    out = np.zeros_like(X_explain, dtype=float)
    for i, row in enumerate(X_explain):
        tgt = int(proba[i].argmax())
        out[i] = occlusion_attribution(model.predict_proba, row, base, tgt)
    return out, "occlusion"


def integrated_gradients_mlp(
    trainer, x_raw: np.ndarray, target: int, steps: int = 64,
    baseline_raw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Integrated gradients for the MLP w.r.t. the (standardized) input features.

    ``trainer`` is a fitted :class:`mimic.training.MLPTrainer`. Attribution is
    returned in the original feature space (rescaled by the trainer's std).
    """
    torch = trainer.torch
    net = trainer.net
    mu, sd = trainer.mu, trainer.sd
    x = (np.asarray(x_raw, float)[None, :] - mu) / sd
    base = (np.zeros_like(x_raw)[None, :] if baseline_raw is None
            else (np.asarray(baseline_raw, float)[None, :] - mu) / sd)
    x_t = torch.tensor(x, dtype=torch.float32, device=trainer.device)
    base_t = torch.tensor(base, dtype=torch.float32, device=trainer.device)

    net.eval()
    total = torch.zeros_like(x_t)
    for a in np.linspace(0, 1, steps):
        pt = (base_t + a * (x_t - base_t)).clone().requires_grad_(True)
        out = torch.softmax(net(pt), dim=1)[0, target]
        grad = torch.autograd.grad(out, pt)[0]
        total = total + grad
    ig = ((x_t - base_t) * total / steps).detach().cpu().numpy()[0]
    return ig / (sd[0] + 1e-9)          # back to original feature units


@dataclass
class Counterfactual:
    changed: dict[str, tuple[float, float]]   # feature -> (from, to)
    orig_class: int
    new_class: int
    steps: int


def counterfactual(
    model, x: np.ndarray, feature_names: list[str], baseline: np.ndarray,
    max_features: int = 6,
) -> Optional[Counterfactual]:
    """Greedy counterfactual: move the most-attributed features toward baseline
    until the predicted class changes (or give up after ``max_features``)."""
    x = np.asarray(x, float).copy()
    proba = model.predict_proba(x[None, :])[0]
    orig = int(proba.argmax())
    attr = np.abs(occlusion_attribution(model.predict_proba, x, baseline, orig))
    order = np.argsort(-attr)

    changed: dict[str, tuple[float, float]] = {}
    for step, j in enumerate(order[:max_features], start=1):
        frm = float(x[j])
        x[j] = baseline[j]
        changed[feature_names[j]] = (round(frm, 4), round(float(baseline[j]), 4))
        new = int(model.predict_proba(x[None, :])[0].argmax())
        if new != orig:
            return Counterfactual(changed, orig, new, step)
    return None
