"""Dispatch service — the one place the full intake -> route -> analyse flow lives.

Sequencing only. Every decision it makes is delegated: the intake owns transport
guards, the router owns modality detection, the registry owns engine resolution, and
the engine owns clinical interpretation. What this service adds is the *policy* about
which outcomes are errors and which are structured results:

* unroutable upload            -> raise (the caller asked for analysis; there is none)
* routable but unsupported     -> ``UNSUPPORTED`` result, HTTP 200
* routable, engine unavailable -> ``FAILED`` result, HTTP 200
* routable, engine placeholder -> ``NOT_IMPLEMENTED`` result, HTTP 200
* routable, engine ran         -> ``COMPLETED`` (or ``FAILED``) result, HTTP 200

Everything except the first returns 200 with a filled-in envelope on purpose: the
routing metadata is the valuable part of those responses, and burying it in an HTTP
error body would throw away exactly what the caller needs to see.
"""
from __future__ import annotations

import time
from typing import Any

from backend.core.router.router import ModalityRouter
from backend.core.shared.errors import AuraBackendError, EngineNotAvailable
from backend.core.shared.logging import get_logger, use_correlation_id
from backend.core.shared.types import ImageAsset
from backend.engines.base.registry import EngineRegistry, default_registry
from backend.models.routing import (
    AnalysisEnvelope,
    EngineOutcome,
    ResultStatus,
    RoutingMetadata,
)

log = get_logger("dispatch")


class _Stopwatch:
    """Accumulates per-stage wall-clock timings for the response envelope."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._mark = self._start
        self.timings: dict[str, float] = {}

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = round((now - self._mark) * 1000.0, 2)
        self._mark = now

    def finish(self) -> dict[str, float]:
        self.timings["total"] = round((time.perf_counter() - self._start) * 1000.0, 2)
        return self.timings


class DispatchService:
    """Routes an asset and runs the selected engine."""

    def __init__(
        self,
        router: ModalityRouter | None = None,
        registry: EngineRegistry | None = None,
    ) -> None:
        self.registry = registry or default_registry
        self.router = router or ModalityRouter(registry=self.registry)

    # ------------------------------------------------------------------ #
    def inspect(self, asset: ImageAsset) -> RoutingMetadata:
        """Detect and select without analysing. Backs the routing-only endpoint."""
        return self.router.route(asset)

    async def dispatch(self, asset: ImageAsset,
                       request_id: str | None = None,
                       declared_modality: str | None = None,
                       clinical_context: dict | None = None) -> AnalysisEnvelope:
        """Route ``asset`` and run the selected engine.

        ``declared_modality`` is the caller's statement of what the study is. See
        :meth:`_declared_routing` for how it is weighed against detection — in short,
        it breaks ties and fills silences but never overrules a confident measurement.

        Raises :class:`~backend.core.shared.errors.ModalityUndetermined` (from the
        router) and :class:`~backend.core.shared.errors.ModalityConflict` (from the
        declaration check). Engine-level problems are reported inside the envelope.
        """
        with use_correlation_id(request_id) as cid:
            watch = _Stopwatch()
            from backend.core.shared.errors import UnsupportedModality, ModalityConflict
            try:
                if declared_modality:
                    routing = self._declared_routing(asset, declared_modality, cid)
                else:
                    routing = self.router.route_or_raise(asset)
            except (UnsupportedModality, ModalityConflict) as exc:
                try:
                    from gateway.app import store
                    detail = {
                        "reason": exc.reason,
                        "confidence": exc.detail.get("confidence", 0.0),
                        "triggered_rule": exc.detail.get("reason", ""),
                        "asset_sha256": asset.sha256,
                    }
                    if isinstance(exc, ModalityConflict):
                        detail["declared"] = exc.detail.get("declared", "")
                        detail["detected"] = exc.detail.get("detected", "")
                        detail["confidence"] = exc.detail.get("detected_confidence", 0.0)
                        detail["triggered_rule"] = exc.detail.get("detector_reason", "")
                    
                    store().audit(
                        action="modality.rejected",
                        entity_type="upload",
                        entity_id=asset.sha256[:12],
                        detail=detail,
                    )
                except Exception as audit_exc:
                    log.warning(f"Failed to audit modality rejection: {audit_exc}")
                raise

            watch.lap("routing")

            if not routing.supported or routing.selected_engine is None:
                outcome = EngineOutcome(
                    status=ResultStatus.UNSUPPORTED,
                    engine="",
                    message=routing.reason,
                    payload={"modality": routing.modality,
                             "supported_modalities": sorted(
                                 m.value for m in self.registry.supported_modalities())},
                )
                return AnalysisEnvelope(request_id=cid, routing=routing,
                                        result=outcome, timings_ms=watch.finish())

            try:
                engine = self.registry.resolve(routing.selected_engine)
            except EngineNotAvailable as exc:
                log.error(
                    "selected engine could not be resolved",
                    extra={"context": {"engine": routing.selected_engine,
                                       "reason": exc.reason}},
                )
                outcome = EngineOutcome(
                    status=ResultStatus.FAILED,
                    engine=routing.selected_engine,
                    message=exc.reason,
                    payload={"error": exc.code},
                )
                return AnalysisEnvelope(request_id=cid, routing=routing,
                                        result=outcome, timings_ms=watch.finish())

            log.info(
                "dispatching study",
                extra={"context": {"engine": routing.selected_engine,
                                   "modality": routing.modality,
                                   "confidence": routing.confidence,
                                   "sha256": asset.sha256[:12]}},
            )
            # ``run`` is the contract's template method: it never propagates an
            # engine exception, it converts one into a FAILED outcome.
            outcome = await engine.run(asset, clinical_context=clinical_context)
            watch.lap("analysis")

            return AnalysisEnvelope(request_id=cid, routing=routing, result=outcome,
                                    timings_ms=watch.finish())

    # ------------------------------------------------------------------ #
    def _declared_routing(self, asset: ImageAsset, declared: str,
                          cid: str) -> RoutingMetadata:
        """Route on a caller-declared modality, cross-checked against the detector.

        The declaration is evidence, not authority. It is the strongest signal
        available when a clinician picks "brain MRI" from the console — they can see
        the study and the pixels cannot always prove what it is — so it decides the
        route whenever detection is silent, weak, or agrees. What it may never do is
        contradict a confident, calibrated detection: that is
        :class:`ModalityConflict`, and it is refused rather than resolved.

        Refusing matters in both directions. Trusting the declaration blindly is how a
        chest radiograph uploaded through the MRI button comes back as a glioma —
        NeuroMind's own validation accepts any 2D image, so nothing further downstream
        would catch it. Ignoring the declaration is how CASE-UPLOAD-27 happened.
        """
        from backend.core.shared.errors import ModalityConflict
        from backend.core.shared.types import MODALITY_LABELS, ImagingModality

        wanted = declared.strip().lower()
        modality = next(
            (m for m in ImagingModality
             if m.value.lower() == wanted or m.name.lower() == wanted), None)
        if modality is None:
            raise ModalityConflict(
                f"{declared!r} is not a modality AURA knows",
                detail={"declared": declared,
                        "known": sorted(m.value for m in ImagingModality)},
            )

        detected = self.router.detector.detect(asset)
        agrees = detected.modality is modality
        contradicts = (detected.committed and not agrees and detected.calibrated)
        if contradicts:
            log.warning(
                "declared modality contradicted by the detector; refusing to dispatch",
                extra={"context": {"declared": modality.value,
                                   "detected": detected.modality.value,
                                   "confidence": detected.confidence,
                                   "sha256": asset.sha256[:12]}},
            )
            raise ModalityConflict(
                f"this upload was submitted as {MODALITY_LABELS.get(modality, modality.value)}, "
                f"but AURA identifies it as {detected.label} "
                f"({detected.confidence:.2f}) — {detected.reason}. No analysis was run: "
                f"running a model on the wrong modality produces a confident, "
                f"plausible, and entirely meaningless report",
                detail={"declared": modality.value,
                        "detected": detected.modality.value,
                        "detected_confidence": round(detected.confidence, 4),
                        "detector_reason": detected.reason,
                        "candidates": [
                            {"modality": c.modality.value,
                             "confidence": round(c.confidence, 4),
                             "calibrated": c.calibrated,
                             "reason": c.reason}
                            for c in detected.candidates]},
            )

        engine_id = self.registry.engine_id_for_modality(modality)
        descriptor = self.registry.descriptor_for(engine_id) if engine_id else None
        engine_name = descriptor.display_name if descriptor else (engine_id or "")

        if agrees:
            reason = (f"submitted as {MODALITY_LABELS.get(modality, modality.value)} and "
                      f"confirmed by the detector ({detected.confidence:.2f}) — "
                      f"{detected.reason}; dispatching to {engine_name}")
        else:
            reason = (f"submitted as {MODALITY_LABELS.get(modality, modality.value)}; the "
                      f"detector could not identify this image on its own "
                      f"({detected.reason}), so the declaration stands and the route is "
                      f"flagged for review; dispatching to {engine_name}")

        return RoutingMetadata(
            modality=modality.value,
            modality_label=MODALITY_LABELS.get(modality, modality.value),
            # The declaration is not a measurement, so an agreeing detection reports
            # its own confidence and a silent one reports the detector's best score.
            # Neither is ever 1.0: nothing here measured certainty.
            confidence=round(detected.confidence, 4) if agrees else round(
                min(detected.confidence, 0.70), 4),
            supported=bool(engine_id),
            reason=reason,
            request_id=cid,
            detector=f"{detected.detector}+client-declaration",
            calibrated=bool(agrees and detected.calibrated),
            requires_review=bool(detected.requires_review or not agrees),
            engine_status=descriptor.status.value if descriptor else None,
            selected_engine=engine_id,
            candidates=self.router.render_candidates(detected),
            asset_sha256=asset.sha256,
        )

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        """Platform capability summary for the engines/diagnostics endpoint."""
        detector = getattr(self.router.detector, "describe", None)
        return {
            "engines": [d.to_dict() for d in self.registry.descriptors()],
            "supported_modalities": sorted(
                m.value for m in self.registry.supported_modalities()),
            "detector": detector() if callable(detector)
                        else {"name": getattr(self.router.detector, "name", "unknown")},
        }


def to_http_error(exc: AuraBackendError) -> tuple[int, dict[str, Any]]:
    """Render a backend error as ``(status_code, body)`` for the API layer."""
    return exc.http_status, exc.to_payload()
