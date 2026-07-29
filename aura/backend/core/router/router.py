"""The Intelligent Modality Router.

Takes a staged upload, asks the detector what it is, asks the registry who serves
that, and returns the decision as :class:`RoutingMetadata`. It does not analyse
anything and it does not construct engines — deciding *whether* a study is routable
must never cost a model load.

The router deliberately does not raise on an unroutable study. ``supported=false``
with a filled-in ``reason`` is a more useful answer than an exception, because the
caller still learns what AURA thought the image was. Only a genuinely undetectable
upload (unreadable bytes, nothing above threshold) becomes an error, and the dispatch
service decides whether that error reaches the client.
"""
from __future__ import annotations

from .detector import (
    DetectionResult,
    ModalityDetector,
    SignatureModalityDetector,
)
from ..shared.errors import ModalityUndetermined
from ..shared.logging import correlation_id, get_logger
from ..shared.types import MODALITY_LABELS, ImageAsset, ImagingModality
from aura.backend.engines.base.registry import EngineRegistry, default_registry
from aura.backend.models.routing import ModalityCandidate, RoutingMetadata

log = get_logger("router")


class ModalityRouter:
    """Detect an upload's modality and select the engine that serves it.

    Both collaborators are injected so the router is testable without models and
    swappable in deployment: pass a trained classifier as ``detector``, or a scoped
    registry to expose only some engines.
    """

    def __init__(
        self,
        detector: ModalityDetector | None = None,
        registry: EngineRegistry | None = None,
    ) -> None:
        self.detector = detector or SignatureModalityDetector()
        self.registry = registry or default_registry

    # ------------------------------------------------------------------ #
    def route(self, asset: ImageAsset) -> RoutingMetadata:
        """Full routing decision for ``asset``. Never raises."""
        detection = self.detector.detect(asset)
        return self._to_metadata(detection, asset)

    def route_or_raise(self, asset: ImageAsset) -> RoutingMetadata:
        """As :meth:`route`, but raise when the modality is not supported or undetermined.

        Used by the analyse path, where there is nothing sensible to run. The
        inspect-only endpoint uses :meth:`route` instead, so a client debugging a
        rejected upload still gets the detector's full reasoning back.
        """
        metadata = self.route(asset)
        from ..shared.errors import UnsupportedModality
        from ..shared.types import ImagingModality

        if metadata.modality not in (ImagingModality.CHEST_XRAY.value, ImagingModality.BRAIN_MRI.value):
            raise UnsupportedModality(
                "Unsupported modality.",
                detail={
                    "candidates": [c.model_dump() for c in metadata.candidates],
                    "confidence": metadata.confidence,
                    "reason": metadata.reason,
                },
            )
        return metadata

    # ------------------------------------------------------------------ #
    @staticmethod
    def render_candidates(detection: DetectionResult) -> list[ModalityCandidate]:
        """Every signature's score as wire-format candidates, best first.

        Public because a caller that builds its own :class:`RoutingMetadata` — the
        dispatch service does, on the client-declared path — must still report the
        losing candidates. Dropping them there would make exactly the responses a human
        reaches for when a route looks wrong the ones with no evidence in them.
        """
        return [
            ModalityCandidate(
                modality=score.modality.value,
                label=score.label,
                confidence=round(score.confidence, 4),
                calibrated=score.calibrated,
                source=score.source,
                reason=score.reason,
                evidence=score.evidence,
            )
            for score in detection.candidates
        ]

    def _to_metadata(self, detection: DetectionResult,
                     asset: ImageAsset) -> RoutingMetadata:
        candidates = self.render_candidates(detection)
        common = {
            "modality": detection.modality.value,
            "modality_label": MODALITY_LABELS.get(detection.modality,
                                                  detection.modality.value),
            "confidence": detection.confidence,
            "request_id": correlation_id(),
            "detector": detection.detector,
            "calibrated": detection.calibrated,
            "candidates": candidates,
            "asset_sha256": asset.sha256,
        }

        if not detection.committed:
            log.warning(
                "upload could not be routed",
                extra={"context": {"reason": detection.reason,
                                   "sha256": asset.sha256[:12]}},
            )
            return RoutingMetadata(
                selected_engine=None,
                supported=False,
                reason=detection.reason,
                requires_review=True,
                engine_status=None,
                **common,
            )

        engine_id = self.registry.engine_id_for_modality(detection.modality)
        if engine_id is None:
            supported_labels = sorted(
                MODALITY_LABELS.get(m, m.value)
                for m in self.registry.supported_modalities()
            )
            reason = (
                f"identified as {common['modality_label']} ({detection.reason}), but no "
                f"analysis engine is registered for it. Currently served: "
                f"{', '.join(supported_labels) or 'none'}."
            )
            log.info(
                "routed to an unsupported modality",
                extra={"context": {"modality": detection.modality.value,
                                   "confidence": detection.confidence}},
            )
            return RoutingMetadata(
                selected_engine=None,
                supported=False,
                reason=reason,
                requires_review=detection.requires_review,
                engine_status=None,
                **common,
            )

        descriptor = self.registry.descriptor_for(engine_id)
        engine_status = descriptor.status.value if descriptor else None
        engine_name = descriptor.display_name if descriptor else engine_id
        reason = f"{detection.reason}; dispatching to {engine_name}"
        if detection.requires_review:
            reason += (
                " — confidence is provisional, so confirm the modality before "
                "relying on the result"
            )

        return RoutingMetadata(
            selected_engine=engine_id,
            supported=True,
            reason=reason,
            requires_review=detection.requires_review,
            engine_status=engine_status,
            **common,
        )
