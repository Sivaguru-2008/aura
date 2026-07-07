"""Vision Intelligence Engine — image -> findings + embedding.

Deliberately commodity (the moat is downstream). Implemented as region feature
extraction + per-finding detectors so it runs on CPU with zero downloads and
exposes a pure `score_findings(image)` callable for occlusion saliency. The
`VisionEngine` contract is drop-in replaceable by a torchxrayvision/ONNX model.
"""
from services.vision.engine import VisionEngine

__all__ = ["VisionEngine"]
