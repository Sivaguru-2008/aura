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

    ``backbone`` (optional): a production CNN (``services.vision.cnn.CXRBackbone``).
    When present, findings and embeddings come from the CNN; findings the CNN does
    not cover (e.g. hyperinflation, absent from MIMIC labels) fall back to the
    feature model, so ``score_findings`` always returns the full finding set. The
    call stays a pure, deterministic callable, keeping occlusion saliency valid.
    """

    def __init__(self, weights: dict[Finding, np.ndarray] | None = None,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 backbone=None):
        self.weights = weights
        self.mean = mean if mean is not None else np.zeros(N_FEAT)
        self.std = std if std is not None else np.ones(N_FEAT)
        self.backbone = backbone
        self.model_version = backbone.model_version if backbone is not None else MODEL_VERSION

    @classmethod
    def load(cls, path: Path | None = None) -> "VisionEngine":
        # PRODUCTION CONTRACT: serve the trained DenseNet-121 at artifacts/best_model.pt
        # and nothing else. We never substitute another model or fall back to heuristic
        # feature scores — a missing/broken checkpoint is a hard error, not a silent
        # downgrade (a fabricated finding on a real patient is worse than an outage).
        # Set AURA_ALLOW_FALLBACK_VISION=1 for dev/test only.
        import os

        from common.config import ARTIFACTS
        best_model_path = ARTIFACTS / "best_model.pt"
        allow_fallback = os.environ.get("AURA_ALLOW_FALLBACK_VISION", "0") == "1"
        backbone, load_err = None, None
        if best_model_path.exists():
            try:
                from ml.vision_cxr.inference import VisionModel
                backbone = VisionModel(str(best_model_path))
            except Exception as e:
                load_err = e
                print(f"[VisionEngine] failed to load production backbone from {best_model_path}: {e}")
        else:
            load_err = FileNotFoundError(f"missing trained model: {best_model_path}")

        if backbone is None:
            if not allow_fallback:
                raise RuntimeError(
                    f"AURA requires the trained model at {best_model_path} and refuses to "
                    f"serve substitute/heuristic outputs ({load_err}). Train it "
                    f"(`aura_cli train-cnn`) or set AURA_ALLOW_FALLBACK_VISION=1 for dev only."
                )
            backbone = cls._maybe_backbone()

        path = path or (ARTIFACTS / "vision.npz")
        weights = mean = std = None
        if path.exists():
            d = np.load(path, allow_pickle=True)
            weights = {Finding(k): d[k] for k in d.files if k not in ("_mean", "_std")}
            mean, std = d["_mean"], d["_std"]
        return cls(weights=weights, mean=mean, std=std, backbone=backbone)


    @staticmethod
    def _maybe_backbone():
        """Build the configured CNN backbone, or None to keep the feature path."""
        from common.config import get_settings

        kind = get_settings().vision_backend
        if kind in (None, "", "features"):
            return None
        from services.vision.cnn import get_backbone

        s = get_settings()
        if kind == "densenet_mimic":
            return get_backbone("densenet_mimic", weights=s.vision_weights)
        if kind == "timm":
            return get_backbone("timm", arch=s.vision_arch)
        return None

    def _std_feats(self, img: np.ndarray) -> np.ndarray:
        f = extract_features(img)
        x = np.array([f[n] for n in FEATURE_NAMES], dtype=float)
        return (x - self.mean) / self.std

    def _feature_scores(self, img: np.ndarray) -> dict[Finding, float]:
        """Findings from the numpy feature model (trained logistic or heuristic)."""
        if self.weights is None:
            return self._fallback_scores(img)
        xs = np.append(self._std_feats(img), 1.0)
        return {f: float(sigmoid(float(np.dot(w, xs)))) for f, w in self.weights.items()}

    def score_findings(self, img: np.ndarray) -> dict[Finding, float]:
        """Pure callable used both for serving and for occlusion saliency.

        With a CNN backbone: CNN findings, with feature-model fill for any finding
        the CNN doesn't predict. Without one: the feature model alone.
        """
        if self.backbone is None:
            return self._feature_scores(img)
        cnn = self.backbone.score_findings(img)
        # The production DenseNet-121 covers all findings: return its outputs verbatim,
        # with NO heuristic feature scores mixed in. Feature-fill is only for a partial
        # backbone (e.g. torchxrayvision, which lacks hyperinflation) in dev mode.
        if all(f in cnn for f in Finding):
            return {f: float(cnn[f]) for f in Finding}
        scores = self._feature_scores(img)          # baseline for uncovered findings only
        scores.update(cnn)                            # CNN wins where it has a label
        return {f: scores[f] for f in Finding}

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
        """Evidence embedding used by memory/similarity.

        CNN backbone -> pooled deep features (1024-d for DenseNet); otherwise the
        raw hand-feature vector. Either way a fixed-width vector cosine can rank.
        """
        if self.backbone is not None:
            return self.backbone.embedding(img)
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

    def predict(self, img: np.ndarray, study_id: str = "default") -> VisionResult:
        """Alias for analyze to satisfy predictability of prediction invocation."""
        return self.analyze(study_id, img)
