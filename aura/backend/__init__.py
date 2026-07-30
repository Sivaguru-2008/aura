"""AURA Medical AI Operating System — modular backend layer.

This package is the *routing architecture* that turns AURA from a single chest-X-ray
product into a multi-modality platform. Nothing here re-implements clinical analysis:
each imaging domain lives behind an engine that satisfies one small contract
(``backend.engines.base.AnalysisEngine``), and the router's only job is deciding which
engine an upload belongs to.

Layout (mirrors the requested Medical-AI-OS structure)::

    backend/
      core/
        router/    modality detection + engine selection
        upload/    intake: allowlist, size cap, temp staging, guaranteed cleanup
        shared/    logging, error taxonomy, primitive types
      engines/
        base/      abstract contract + plug-in registry
        thorax/    adapter over the existing (unmodified) chest-X-ray pipeline
        neuro/     AURA NeuroMind — trained brain MRI engine (BraTS2020, gliomas)
      services/    orchestration (intake -> route -> dispatch -> envelope)
      api/         FastAPI routes mounted onto the existing gateway app
      models/      wire contracts (pydantic) for routing + analysis responses

Design rules this package holds itself to:

1. **The Thorax engine is not modified.** ``engines/thorax`` is a thin adapter that
   calls the existing ``gateway.pipeline.Pipeline`` through its public surface. The
   legacy ``POST /v1/studies/upload`` endpoint keeps working byte-for-byte.
2. **Adding a modality is additive.** A new engine is a class plus one registry
   entry; no router, API, or service code changes. Third-party engines can arrive
   through the ``aura.engines`` entry-point group without touching this repo.
3. **Detection confidence is honest.** The detector reports how a decision was
   reached and whether the thresholds behind it were calibrated on real data.
   Uncalibrated paths are capped and flagged, never dressed up as certainty.

See ``backend/README.md`` for the full architecture note and the measured evidence
behind the detector thresholds.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
