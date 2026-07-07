"""Train per-finding logistic detectors from ground-truth findings.

Each finding gets its own binary logistic regression over standardized features;
label combinations that a single feature can't separate become separable in the
joint feature space (e.g. cardiomegaly = wide heart AND basal effusion AND low
asymmetry). Saves artifacts/vision.npz consumed by VisionEngine.load().
"""
from __future__ import annotations

import numpy as np

from common.config import ARTIFACTS, ensure_dirs
from common.mathx import sigmoid
from schemas.clinical import Finding
from services.vision.features import FEATURE_NAMES, extract_features
from ml.data import Sample, make_dataset


def _features(samples: list[Sample]) -> np.ndarray:
    return np.array(
        [[extract_features(s.image)[n] for n in FEATURE_NAMES] for s in samples],
        dtype=float,
    )


def _fit_logistic(X, yb, epochs=600, lr=0.2, l2=1e-3, seed=7):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.05, size=d + 1)
    Xb = np.hstack([X, np.ones((n, 1))])
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        g = Xb.T @ (p - yb) / n
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w


def run(n_samples: int = 700, seed: int = 7) -> dict:
    ensure_dirs()
    samples = make_dataset(n_samples, seed=seed)
    Xraw = _features(samples)
    mean, std = Xraw.mean(axis=0), Xraw.std(axis=0) + 1e-6
    X = (Xraw - mean) / std

    out: dict[str, np.ndarray] = {"_mean": mean, "_std": std}
    accs = {}
    for finding in Finding:
        yb = np.array([1.0 if s.findings[finding] >= 0.5 else 0.0 for s in samples])
        if yb.sum() == 0:
            out[finding.value] = np.zeros(X.shape[1] + 1) - 5.0  # never fires
            continue
        w = _fit_logistic(X, yb, seed=seed)
        out[finding.value] = w
        pred = sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w) >= 0.5
        accs[finding.value] = round(float((pred == yb).mean()), 3)

    np.savez(ARTIFACTS / "vision.npz", **out)
    print(f"[train] vision detectors fit; per-finding train acc: {accs}")
    return accs


if __name__ == "__main__":
    run()
