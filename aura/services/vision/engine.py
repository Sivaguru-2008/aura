"""Vision engine: features -> per-finding probabilities + embedding.

Detectors are per-finding logistic regressions over standardized anatomical
features, learned from ground-truth findings (see ml/training/train_vision.py)
and loaded from artifacts/vision.npz. Before training, a conservative fallback
keeps the engine functional. The key platform property: `score_findings` is a
pure callable, which is what makes model-agnostic occlusion saliency valid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import ARTIFACTS
from common.mathx import sigmoid
from schemas.clinical import Finding
from schemas.contracts import FindingScore, VisionResult
from services.vision.features import FEATURE_NAMES, extract_features, feature_vector

MODEL_VERSION = "vision-cxr-region-v1"
N_FEAT = len(FEATURE_NAMES)

# Region each finding localizes to, for overlay boxes (normalized coords).
_FINDING_REGION: dict[Finding, tuple[float, float, float, float]] = {
    Finding.OPACITY: (0.15, 0.08, 0.75, 0.92),
    Finding.CONSOLIDATION: (0.15, 0.08, 0.75, 0.92),
    Finding.EFFUSION: (0.72, 0.10, 0.92, 0.90),
    Finding.CARDIOMEGALY: (0.42, 0.34, 0.82, 0.66),
    Finding.NODULE: (0.15, 0.08, 0.75, 0.92),
    Finding.PNEUMOTHORAX: (0.15, 0.08, 0.75, 0.92),
    Finding.HYPERINFLATION: (0.12, 0.06, 0.86, 0.94),
}


class VisionEngine:
    """weights: {Finding: array(N_FEAT+1)} logistic weights over standardized feats+bias.
    mean/std: feature standardization vectors (length N_FEAT).
    """

    def __init__(self, weights: dict[Finding, np.ndarray] | None = None,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.weights = weights
        self.mean = mean if mean is not None else np.zeros(N_FEAT)
        self.std = std if std is not None else np.ones(N_FEAT)
        self.model_version = MODEL_VERSION

    @classmethod
    def load(cls, path: Path | None = None) -> "VisionEngine":
        path = path or (ARTIFACTS / "vision.npz")
        if path.exists():
            d = np.load(path, allow_pickle=True)
            w = {Finding(k): d[k] for k in d.files if k not in ("_mean", "_std")}
            return cls(weights=w, mean=d["_mean"], std=d["_std"])
        return cls()

    def _std_feats(self, img: np.ndarray) -> np.ndarray:
        f = extract_features(img)
        x = np.array([f[n] for n in FEATURE_NAMES], dtype=float)
        return (x - self.mean) / self.std

    def score_findings(self, img: np.ndarray) -> dict[Finding, float]:
        """Pure callable used both for serving and for occlusion saliency."""
        if self.weights is None:
            return self._fallback_scores(img)
        xs = np.append(self._std_feats(img), 1.0)
        return {f: float(sigmoid(float(np.dot(w, xs)))) for f, w in self.weights.items()}

    def _fallback_scores(self, img: np.ndarray) -> dict[Finding, float]:
        """Untrained heuristic so the engine is never dead. Uses raw features."""
        f = extract_features(img)
        lung_bright = 0.5 * (f["right_lung_bright"] + f["left_lung_bright"])
        return {
            Finding.OPACITY: float(sigmoid(12 * lung_bright - 2.0)),
            Finding.CONSOLIDATION: float(sigmoid(10 * lung_bright + 4 * f["lung_asymmetry"] - 2.6)),
            Finding.EFFUSION: float(sigmoid(20 * f["cp_bright"] - 2.6)),
            Finding.CARDIOMEGALY: float(sigmoid(8 * f["heart_width"] + 8 * f["cp_bright"] - 6.0)),
            Finding.NODULE: float(sigmoid(30 * f["nodule_tophat"] - 12 * f["vertical_line"] - 6.0)),
            Finding.PNEUMOTHORAX: float(sigmoid(14 * f["vertical_line"] + 6 * f["lung_asymmetry"] - 4.2)),
            Finding.HYPERINFLATION: float(sigmoid(-26 * f["lung_mean"] + 4.2)),
        }

    def embedding(self, img: np.ndarray) -> np.ndarray:
        """Compact evidence embedding: raw feature vector, used by memory/similarity."""
        return feature_vector(img)

    def analyze(self, study_id: str, img: np.ndarray) -> VisionResult:
        scores = self.score_findings(img)
        findings = [
            FindingScore(finding=f, probability=round(p, 4), region=_FINDING_REGION[f])
            for f, p in sorted(scores.items(), key=lambda kv: -kv[1])
        ]
        return VisionResult(
            study_id=study_id,
            findings=findings,
            embedding=[round(float(v), 5) for v in self.embedding(img)],
            model_version=self.model_version,
        )
