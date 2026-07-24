"""Modality detection and engine selection.

    fingerprint(path) -> ImageFingerprint      measure the image once
    signatures        -> SignatureScore        per-modality scoring units
    detector          -> DetectionResult       ranking + commit policy
    ModalityRouter    -> RoutingMetadata       decision + engine selection

Extension points, in the order you are likely to need them:

* **new modality, same detection style** — add a ``ModalitySignature`` in
  ``signatures.py`` and list it in ``default_signatures()``;
* **new detection strategy** — implement the ``ModalityDetector`` protocol and pass
  it to ``ModalityRouter(detector=...)``. A trained classifier drops in here;
* **new engine for an existing modality** — register it; the router needs no change.
"""

from backend.core.router.detector import (
    DetectionResult,
    ModalityDetector,
    SignatureModalityDetector,
)
from backend.core.router.features import ImageFingerprint, fingerprint
from backend.core.router.router import ModalityRouter
from backend.core.router.signatures import (
    ModalitySignature,
    SignatureScore,
    default_signatures,
)

__all__ = [
    "DetectionResult",
    "ImageFingerprint",
    "ModalityDetector",
    "ModalityRouter",
    "ModalitySignature",
    "SignatureModalityDetector",
    "SignatureScore",
    "default_signatures",
    "fingerprint",
]
