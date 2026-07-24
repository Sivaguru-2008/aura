"""Wire the routing layer into a running gateway.

One function, called once from the gateway's lifespan. It is the *only* integration
point between the new backend and the existing app — requirement 4 in one place, so
"what did the router change about Thorax?" has a short, checkable answer: it binds to
the pipeline and store the gateway already built, and mounts three new endpoints.

Engine registration happens in two ways here, both deliberate:

* **NeuroMind** self-registers on import (``@register_engine``) because it needs no
  runtime handles.
* **Thorax** is registered by this function because it must be bound to the *live*
  ``Pipeline`` and ``Store``. Constructing its own would duplicate the loaded models
  and split the memory index.

Third-party engines are then discovered from the ``aura.engines`` entry-point group,
so a plug-in installed into the environment appears with no code change at all.
"""
from __future__ import annotations

from typing import Any, Callable

from backend.core.shared.logging import get_logger
from backend.engines.base.registry import EngineRegistry, default_registry
from backend.services.dispatch import DispatchService

log = get_logger("bootstrap")


def install_router(
    app: Any,
    pipeline: Any,
    store: Any,
    on_case_created: Callable[[str], None] | None = None,
    registry: EngineRegistry | None = None,
    discover_plugins: bool = True,
) -> DispatchService:
    """Register engines and mount the routing endpoints onto ``app``.

    Args:
        app: the FastAPI application to mount onto.
        pipeline: live ``gateway.pipeline.Pipeline`` for the Thorax engine.
        store: live ``gateway.storage.Store`` for persistence and audit.
        on_case_created: hook fired with each new case id, so router-created cases
            join the console worklist exactly like legacy uploads.
        registry: engine registry to use (defaults to the process-wide one).
        discover_plugins: whether to scan the ``aura.engines`` entry-point group.

    Returns:
        The :class:`DispatchService` backing the mounted routes — handy for tests and
        for any in-process caller that wants to route without going through HTTP.
    """
    reg = registry or default_registry

    from backend.engines.neuro.engine import register_neuromind_engine
    from backend.engines.thorax.engine import register_thorax_engine

    register_thorax_engine(pipeline, store, on_case_created, registry=reg)
    register_neuromind_engine(pipeline, store, on_case_created, registry=reg)

    if discover_plugins:
        added = reg.discover_plugins()
        if added:
            log.info("loaded plug-in engines", extra={"context": {"engines": added}})

    dispatch = DispatchService(registry=reg)

    from backend.api.routes import build_router

    app.include_router(build_router(dispatch))

    log.info(
        "modality router installed",
        extra={"context": {
            "engines": [d.engine_id for d in reg.descriptors()],
            "modalities": sorted(m.value for m in reg.supported_modalities()),
        }},
    )
    return dispatch
