"""The contract every AURA analysis engine implements.

Four methods, always in this order::

    validate_input(asset)   -> ValidationOutcome    is this engine's kind of image?
    preprocess(asset)       -> PreparedStudy        bytes -> model-ready study
    analyze(prepared)       -> AnalysisResult       run the models
    generate_report(result) -> EngineReport         clinical narrative

:meth:`AnalysisEngine.run` is the template method that calls them in order and turns
the result into the wire envelope. Engines normally inherit it untouched; overriding
it is allowed but means re-implementing the error handling, so it is rarely right.

Why the split matters
---------------------
Each stage has a different failure mode and a different reviewer:

* ``validate_input`` is a *safety gate* — it is the engine's own opinion about
  whether it should be looking at this image at all, independent of the router. The
  router can be wrong; a chest model must still refuse a knee film. This double
  check is deliberate defence in depth, not redundancy.
* ``preprocess`` is where fidelity is won or lost (resampling, windowing,
  normalisation). Isolating it keeps those decisions reviewable and testable without
  loading a model.
* ``analyze`` is the only stage allowed to be expensive or stochastic.
* ``generate_report`` must be a pure function of ``AnalysisResult`` — no model calls,
  no new evidence. That constraint is what makes a report *grounded*: every sentence
  has to trace to something ``analyze`` already produced.

``analyze`` is async because real engines do I/O and may await a shared executor; the
other three are synchronous because they must stay cheap and deterministic.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.core.shared.errors import (
    AuraBackendError,
    EngineExecutionError,
    EngineNotImplemented,
)
from backend.core.shared.logging import get_logger
from backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
from backend.models.routing import EngineOutcome, ResultStatus


@dataclass(frozen=True)
class EngineDescriptor:
    """Static identity of an engine. Read by the registry, the router, and the API.

    ``modalities`` is the routing claim: the registry indexes engines by it, so this
    tuple is the single place a domain says "I handle these studies".
    """

    engine_id: str
    display_name: str
    version: str
    modalities: tuple[ImagingModality, ...]
    status: EngineStatus = EngineStatus.AVAILABLE
    capabilities: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "version": self.version,
            "status": self.status.value,
            "modalities": [m.value for m in self.modalities],
            "capabilities": list(self.capabilities),
            "description": self.description,
        }


@dataclass
class ValidationOutcome:
    """An engine's verdict on whether it should analyse this asset."""

    accepted: bool
    reason: str = ""
    #: Named checks and their measured values — surfaced to the client on rejection
    #: so "we refused your upload" is always accompanied by *why*.
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedStudy:
    """Model-ready study handed from ``preprocess`` to ``analyze``.

    ``payload`` is engine-owned and opaque to the routing layer: the thorax engine
    puts a ``StudyInput`` in it, a future CT engine might put a volume handle. The
    surrounding fields are the parts the platform needs for logging and audit.
    """

    study_id: str
    modality: ImagingModality
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Raw analysis output, before narrative generation."""

    study_id: str
    #: Engine-specific clinical result (a ``CaseBundle`` for thorax, etc.).
    payload: Any
    #: Persisted case identifier, when the engine stores its result.
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineReport:
    """Clinical narrative derived *only* from an :class:`AnalysisResult`."""

    sections: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    #: Traceability map: section -> evidence node ids that produced it. AURA's
    #: existing report engine already emits this; new engines are expected to match.
    grounding: dict[str, Any] = field(default_factory=dict)


class AnalysisEngine(ABC):
    """Base class for every engine.

    Subclasses declare :attr:`descriptor` and implement the four stages. Construction
    should load models; it happens once, lazily, when the registry first resolves the
    engine.
    """

    #: Set by every concrete engine.
    descriptor: EngineDescriptor

    def __init__(self) -> None:
        self.log = get_logger(f"engine.{self.descriptor.engine_id}")

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def validate_input(self, asset: ImageAsset) -> ValidationOutcome:
        """Decide whether this engine is willing to analyse ``asset``.

        Must not raise for bad input — an undecodable or wrong-domain file is a
        rejection with a reason, not an exception.
        """

    @abstractmethod
    def preprocess(self, asset: ImageAsset) -> PreparedStudy:
        """Turn a validated file into a model-ready study.

        Raise :class:`~backend.core.shared.errors.UnreadableImage` if decoding fails.
        """

    @abstractmethod
    async def analyze(self, prepared: PreparedStudy) -> AnalysisResult:
        """Run the engine's models over a prepared study."""

    @abstractmethod
    def generate_report(self, result: AnalysisResult) -> EngineReport:
        """Compose the clinical narrative. Pure function of ``result``."""

    # ------------------------------------------------------------------ #
    # Template method
    # ------------------------------------------------------------------ #
    async def run(self, asset: ImageAsset, clinical_context: dict | None = None) -> EngineOutcome:
        """Execute the four stages and package the outcome.

        Error policy, applied uniformly so no engine has to reproduce it:

        * a validation rejection becomes a ``FAILED`` outcome carrying the engine's
          own checks — the router chose this engine, so the *engine's* refusal is the
          interesting signal;
        * :class:`EngineNotImplemented` becomes a ``NOT_IMPLEMENTED`` outcome, which
          is how a placeholder engine confirms a route without pretending to work;
        * any other exception is logged with its traceback and returned as an opaque
          ``FAILED`` — internal text never reaches the client.
        """
        engine_id = self.descriptor.engine_id
        stage = "validate_input"
        try:
            verdict = self.validate_input(asset)
            if not verdict.accepted:
                self.log.warning(
                    "rejected upload at validate_input",
                    extra={"context": {"reason": verdict.reason,
                                       "sha256": asset.sha256[:12]}},
                )
                return EngineOutcome(
                    status=ResultStatus.FAILED,
                    engine=engine_id,
                    engine_version=self.descriptor.version,
                    message=verdict.reason or "engine rejected the input",
                    payload={"validation": verdict.checks},
                )

            stage = "preprocess"
            t0 = time.perf_counter()
            prepared = self.preprocess(asset)
            if clinical_context:
                prepared.metadata["clinical_context"] = clinical_context

            stage = "analyze"
            result = await self.analyze(prepared)

            stage = "generate_report"
            report = self.generate_report(result)
            elapsed = time.perf_counter() - t0

            self.log.info(
                "analysis completed",
                extra={"context": {"study_id": prepared.study_id,
                                   "case_id": result.case_id,
                                   "seconds": round(elapsed, 4)}},
            )
            return EngineOutcome(
                status=ResultStatus.COMPLETED,
                engine=engine_id,
                engine_version=self.descriptor.version,
                message=report.summary,
                case_id=result.case_id,
                payload={**result.metadata, "analysis_seconds": round(elapsed, 4)},
                report={"summary": report.summary, "sections": report.sections,
                        "grounding": report.grounding},
            )

        except EngineNotImplemented as exc:
            # Expected for placeholder engines — the route resolved correctly.
            self.log.info(
                "engine is a placeholder; routing confirmed but analysis unavailable",
                extra={"context": {"stage": stage, "reason": exc.reason}},
            )
            return EngineOutcome(
                status=ResultStatus.NOT_IMPLEMENTED,
                engine=engine_id,
                engine_version=self.descriptor.version,
                message=exc.reason,
                payload=exc.detail,
            )
        except AuraBackendError as exc:
            self.log.error(
                f"engine error during {stage}",
                extra={"context": {"stage": stage, "code": exc.code,
                                   "reason": exc.reason}},
            )
            return EngineOutcome(
                status=ResultStatus.FAILED,
                engine=engine_id,
                engine_version=self.descriptor.version,
                message=exc.reason,
                payload={"stage": stage, "error": exc.code, **exc.detail},
            )
        except Exception:
            # Full traceback to the server log; opaque message to the client.
            self.log.exception(
                f"unhandled engine failure during {stage}",
                extra={"context": {"stage": stage, "sha256": asset.sha256[:12]}},
            )
            safe = EngineExecutionError(
                "the study could not be analysed", detail={"stage": stage}
            )
            return EngineOutcome(
                status=ResultStatus.FAILED,
                engine=engine_id,
                engine_version=self.descriptor.version,
                message=safe.reason,
                payload={"stage": stage, "error": safe.code},
            )
