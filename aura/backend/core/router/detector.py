"""Modality detection: run every signature, rank the results, decide whether to commit.

The detector is deliberately thin. All domain knowledge lives in the signatures; this
module only owns the *decision policy* — the thresholds that turn a set of scores into
"commit to this modality" or "decline to guess".

:class:`ModalityDetector` is a Protocol, so the whole detection strategy is swappable.
The shipped :class:`SignatureModalityDetector` is rule-based; replacing it with a
trained classifier is a one-line change at the router's construction site, and nothing
downstream — router, dispatch, API, engines — needs to know.

Precedence without special cases
--------------------------------
There is exactly one ordering rule, and it is enforced by the confidence scale rather
than by ``if`` statements: a signature's score states how well *measured* its claim is,
and plain argmax does the rest. An unmeasured signature is capped at
``UNCALIBRATED_CONFIDENCE_CAP`` (0.70) and so can never outrank a measured one. Keeping
this in the numbers instead of in branching logic means a new signature automatically
obeys the same precedence.

That ordering was previously load-bearing in a way it should not have been. The
cross-sectional-head signature was uncalibrated and therefore capped at 0.70, below the
chest pixel-accept at 0.88 — so a brain MRI PNG could not reach the brain engine no
matter how obviously it was a brain, and the only thing standing between it and the
chest model was the chest gate's willingness to reject it. The gate was not willing:
it had no test for head anatomy, and a brain's bright midline passes its mediastinum
check. CASE-UPLOAD-27 is the recorded consequence — an axial brain MRI analysed by
DenseNet-121 and reported as pneumonia. Both halves are now fixed: the gate rejects
head geometry outright, and the head signature is fitted against real corpora
(BraTS2020 vs MIMIC-CXR) so it is calibrated and scores above the chest accept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from .features import ImageFingerprint, fingerprint
from .signatures import (
    ModalitySignature,
    SignatureScore,
    default_signatures,
)
from ..shared.logging import get_logger
from ..shared.types import ImageAsset, ImagingModality

log = get_logger("router.detector")


@dataclass
class DetectionResult:
    """Ranked detection outcome for one asset."""

    #: Winning modality. ``UNKNOWN`` when nothing cleared the commit thresholds.
    modality: ImagingModality
    confidence: float
    calibrated: bool
    source: str
    reason: str
    #: Every signature's score, best first — including the ones that lost.
    candidates: list[SignatureScore]
    #: True when the decision is weak, provisional, or contested by a close runner-up.
    requires_review: bool
    detector: str
    fingerprint: ImageFingerprint

    @property
    def committed(self) -> bool:
        return self.modality is not ImagingModality.UNKNOWN

    @property
    def label(self) -> str:
        """Human-readable name of the detected modality."""
        from ..shared.types import MODALITY_LABELS

        return MODALITY_LABELS.get(self.modality, self.modality.value)

    @property
    def margin(self) -> float:
        """Gap between the winner and the runner-up. 1.0 when unopposed."""
        if len(self.candidates) < 2:
            return 1.0
        return round(self.candidates[0].confidence - self.candidates[1].confidence, 4)


@runtime_checkable
class ModalityDetector(Protocol):
    """Strategy interface for deciding an image's modality."""

    name: str

    def detect(self, asset: ImageAsset) -> DetectionResult:
        """Identify ``asset``'s modality. Must not raise for bad input."""
        ...


class SignatureModalityDetector:
    """Rule-based detector: score every signature, commit if the winner is clear."""

    name = "signature-detector/1.0"

    #: A winner below this does not get routed anywhere.
    MIN_COMMIT_CONFIDENCE = 0.55
    #: The winner must beat the runner-up by at least this much. Two modalities
    #: scoring near-identically is genuine ambiguity, and guessing between them is
    #: exactly the failure mode this whole layer exists to prevent.
    MIN_MARGIN = 0.10
    #: Below this — or on any uncalibrated or narrowly-won decision — the route is
    #: still taken, but flagged for human confirmation.
    REVIEW_CONFIDENCE = 0.85
    REVIEW_MARGIN = 0.20

    def __init__(self, signatures: Sequence[ModalitySignature] | None = None) -> None:
        self._signatures = list(signatures) if signatures is not None else default_signatures()
        log.info(
            "detector ready",
            extra={"context": {"signatures": [type(s).__name__ for s in self._signatures]}},
        )

    def detect(self, asset: ImageAsset) -> DetectionResult:
        import zipfile
        import tempfile
        import shutil
        from pathlib import Path
        
        path_to_detect = asset.path
        temp_dir = None
        
        if zipfile.is_zipfile(asset.path):
            try:
                with zipfile.ZipFile(asset.path, 'r') as zf:
                    for name in zf.namelist():
                        if not name.endswith('/') and not name.startswith('__MACOSX'):
                            temp_dir = Path(tempfile.mkdtemp(prefix="aura-router-zip-"))
                            extracted_path = temp_dir / Path(name).name
                            with open(extracted_path, 'wb') as f_out:
                                f_out.write(zf.read(name))
                            path_to_detect = extracted_path
                            break
            except Exception as zip_exc:
                log.warning(f"Failed to inspect zip file: {zip_exc}")

        try:
            fp = fingerprint(path_to_detect)
        finally:
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

        scores: list[SignatureScore] = []
        for signature in self._signatures:
            try:
                scores.append(signature.score(fp))
            except Exception:
                # One broken signature must not sink detection: log it, skip it, and
                # let the remaining signatures decide.
                log.exception(
                    "signature raised; excluding it from this decision",
                    extra={"context": {"signature": type(signature).__name__}},
                )
        scores.sort(key=lambda s: s.confidence, reverse=True)

        if not scores:
            return self._undetermined(fp, scores, "no modality signature could be evaluated")

        best = scores[0]
        runner_up = scores[1].confidence if len(scores) > 1 else 0.0
        margin = best.confidence - runner_up

        if best.confidence < self.MIN_COMMIT_CONFIDENCE:
            return self._undetermined(
                fp, scores,
                f"no modality scored above the {self.MIN_COMMIT_CONFIDENCE:.2f} "
                f"commit threshold (best: {best.label} at {best.confidence:.2f} — "
                f"{best.reason})")

        if margin < self.MIN_MARGIN:
            return self._undetermined(
                fp, scores,
                f"{best.label} ({best.confidence:.2f}) and {scores[1].label} "
                f"({runner_up:.2f}) are too close to separate; AURA declines to guess")

        requires_review = (
            not best.calibrated
            or best.review_required
            or best.confidence < self.REVIEW_CONFIDENCE
            or margin < self.REVIEW_MARGIN
        )
        result = DetectionResult(
            modality=best.modality,
            confidence=round(best.confidence, 4),
            calibrated=best.calibrated,
            source=best.source,
            reason=best.reason,
            candidates=scores,
            requires_review=requires_review,
            detector=self.name,
            fingerprint=fp,
        )
        log.info(
            "modality detected",
            extra={"context": {"modality": best.modality.value,
                               "confidence": result.confidence,
                               "margin": round(margin, 4),
                               "source": best.source,
                               "calibrated": best.calibrated,
                               "requires_review": requires_review,
                               "sha256": asset.sha256[:12]}},
        )
        return result

    def _undetermined(self, fp: ImageFingerprint, scores: list[SignatureScore],
                      reason: str) -> DetectionResult:
        if fp.error:
            reason = fp.error
        log.info("modality undetermined", extra={"context": {"reason": reason}})
        return DetectionResult(
            modality=ImagingModality.UNKNOWN,
            confidence=round(scores[0].confidence, 4) if scores else 0.0,
            calibrated=True,
            source="none",
            reason=reason,
            candidates=scores,
            requires_review=True,
            detector=self.name,
            fingerprint=fp,
        )

    def describe(self) -> dict[str, Any]:
        """Detector configuration, for the model card / diagnostics endpoints."""
        return {
            "name": self.name,
            "signatures": [type(s).__name__ for s in self._signatures],
            "min_commit_confidence": self.MIN_COMMIT_CONFIDENCE,
            "min_margin": self.MIN_MARGIN,
            "confidence_semantics": (
                "ordinal decision score in [0,1], not a calibrated posterior; see "
                "backend/core/router/signatures.py"
            ),
        }
