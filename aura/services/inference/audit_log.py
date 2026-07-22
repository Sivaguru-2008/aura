"""Per-inference audit log (production requirement).

Every real prediction appends one JSON line to ``artifacts/inference_log.jsonl`` with
the full provenance needed to reproduce and defend it: image hash, model + calibration
versions, raw logits, calibrated probabilities, threshold decisions, the quantum fusion
posterior, the safety decision, the final diagnosis, and the inference time.

No synthetic data path ever calls this — it logs real model output only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from common.config import ARTIFACTS, finding_present_threshold
from schemas.clinical import FINDINGS

LOG_PATH = ARTIFACTS / "inference_log.jsonl"
_CAL_PATH = ARTIFACTS / "vision_serving_calibration.json"


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except Exception:
        return "sha256:unavailable"


def _calibration_version() -> dict:
    """Identify the exact calibration applied: content hash + method + fit size."""
    try:
        raw = _CAL_PATH.read_bytes()
        d = json.loads(raw)
        return {
            "file": _CAL_PATH.name,
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()[:16],
            "method": d.get("method"),
            "n_images_fit": d.get("n_images"),
        }
    except Exception:
        return {"file": _CAL_PATH.name, "sha256": None}


def log_inference(bundle, image_path: str | Path, inference_time_s: float,
                  backbone=None) -> Optional[Path]:
    """Append one audit record for a completed prediction. Best-effort: a logging
    failure must never sink a real prediction, but it is surfaced, not swallowed."""
    try:
        vision = bundle.vision
        safety = getattr(bundle, "safety", None)

        # Raw logits + calibrated probs straight from the served model (one clean
        # forward pass on the backbone, so what we log is exactly what was served).
        raw_logits = None
        if backbone is not None and hasattr(backbone, "model"):
            try:
                import torch, cv2
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    with torch.no_grad():
                        x = backbone._to_tensor(img)
                        raw_logits = backbone.model(x)[0].detach().cpu().numpy().tolist()
            except Exception:
                raw_logits = None

        calibrated = {fs.finding.value: round(float(fs.probability), 6)
                      for fs in (vision.findings if vision else [])}
        thresholds = {f.value: round(finding_present_threshold(f.value), 4) for f in FINDINGS}
        decisions = {f.value: bool(calibrated.get(f.value, 0.0) >= thresholds[f.value])
                     for f in FINDINGS}

        quantum_posterior = None
        if safety is not None and getattr(safety, "predictions", None):
            quantum_posterior = {p.diagnosis.value: round(float(p.probability), 6)
                                 for p in safety.predictions}

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "study_id": getattr(bundle, "study_id", None),
            "case_id": getattr(bundle, "case_id", None),
            "image_sha256": _sha256(image_path),
            "model_version": vision.model_version if vision else None,
            "model_file": "artifacts/best_model.pt",
            "calibration_version": _calibration_version(),
            "raw_logits": {FINDINGS[i].value: round(float(v), 6) for i, v in enumerate(raw_logits)}
                          if raw_logits is not None else None,
            "calibrated_probabilities": calibrated,
            "thresholds": thresholds,
            "threshold_decisions": decisions,
            "quantum_posterior": quantum_posterior,
            "safety": {
                "top_diagnosis": safety.top.value if safety else None,
                "top_probability": round(float(safety.top_probability), 6) if safety else None,
                "abstained": bool(safety.abstained) if safety else None,
                "conformal_set": [d.value for d in safety.conformal_set] if safety else None,
                "epistemic_uncertainty": round(float(safety.epistemic_uncertainty), 6) if safety else None,
            },
            "final_diagnosis": safety.top.value if safety else None,
            "inference_time_s": round(float(inference_time_s), 4),
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return LOG_PATH
    except Exception as e:  # never sink a prediction on a logging failure
        print(f"[audit_log] failed to log inference: {e!r}")
        return None
