"""Engine registry — the plug-in seam.

Registration stores a **factory**, not an instance. Engines load model weights in
their constructor (the thorax engine pulls a DenseNet-121 and a fusion model), so
building every engine at import time would make process startup pay for modalities
that may never be uploaded. The registry therefore constructs lazily on first
resolution and caches from then on.

Three ways an engine gets registered, in increasing order of decoupling:

1. ``@register_engine`` on the class — used by the built-in engines.
2. ``registry.register(engine_id, factory, descriptor)`` — for engines whose
   construction needs runtime handles (the thorax adapter needs the gateway's live
   ``Pipeline`` and ``Store``).
3. :meth:`EngineRegistry.discover_plugins` — third-party packages advertise an entry
   point in the ``aura.engines`` group and are picked up with no code change here::

       [project.entry-points."aura.engines"]
       retina = "aura_retina.engine:RetinaEngine"

A failure to construct one engine is contained: it is logged, that engine is marked
``UNAVAILABLE``, and every other route keeps working. A missing optional dependency
in a new engine must never take the chest-X-ray path down with it.
"""
from __future__ import annotations

import threading
from typing import Callable, Iterable

from aura.backend.core.shared.errors import EngineNotAvailable
from aura.backend.core.shared.logging import get_logger
from aura.backend.core.shared.types import EngineStatus, ImagingModality
from .contract import AnalysisEngine, EngineDescriptor

log = get_logger("registry")

#: Entry-point group third-party engines advertise themselves under.
ENTRY_POINT_GROUP = "aura.engines"

EngineFactory = Callable[[], AnalysisEngine]


class _Entry:
    """One registered engine: its descriptor, its factory, and the cached instance."""

    __slots__ = ("descriptor", "factory", "instance", "load_error")

    def __init__(self, descriptor: EngineDescriptor, factory: EngineFactory) -> None:
        self.descriptor = descriptor
        self.factory = factory
        self.instance: AnalysisEngine | None = None
        self.load_error: str | None = None


class EngineRegistry:
    """Thread-safe map of engine id -> engine, indexed by modality for routing."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        descriptor: EngineDescriptor,
        factory: EngineFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` under ``descriptor.engine_id``.

        ``replace=True`` is how a deployment swaps an implementation (e.g. binding
        the thorax adapter to live gateway handles after startup) without a second
        registry. Re-registering without it is a programming error and raises.
        """
        with self._lock:
            existing = self._entries.get(descriptor.engine_id)
            if existing is not None and not replace:
                raise ValueError(
                    f"engine {descriptor.engine_id!r} is already registered; "
                    f"pass replace=True to override"
                )
            self._entries[descriptor.engine_id] = _Entry(descriptor, factory)
            log.info(
                "registered engine",
                extra={"context": {"engine": descriptor.engine_id,
                                   "status": descriptor.status.value,
                                   "modalities": [m.value for m in descriptor.modalities],
                                   "replaced": existing is not None}},
            )

    def unregister(self, engine_id: str) -> None:
        with self._lock:
            self._entries.pop(engine_id, None)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def descriptors(self) -> list[EngineDescriptor]:
        """Every registered engine's descriptor, with live availability applied.

        An engine that previously failed to construct reports ``UNAVAILABLE`` here
        even though it registered as ``AVAILABLE`` — the descriptor is the claim, the
        load result is the fact.
        """
        with self._lock:
            out = []
            for entry in self._entries.values():
                d = entry.descriptor
                if entry.load_error is not None:
                    d = EngineDescriptor(
                        engine_id=d.engine_id, display_name=d.display_name,
                        version=d.version, modalities=d.modalities,
                        status=EngineStatus.UNAVAILABLE,
                        capabilities=d.capabilities,
                        description=f"{d.description} (load failed: {entry.load_error})",
                    )
                out.append(d)
            return out

    def descriptor_for(self, engine_id: str) -> EngineDescriptor | None:
        with self._lock:
            entry = self._entries.get(engine_id)
            return entry.descriptor if entry else None

    def engine_id_for_modality(self, modality: ImagingModality) -> str | None:
        """Engine id claiming ``modality``, or ``None`` when nothing does.

        Deliberately does *not* construct the engine: the router calls this to decide
        ``supported``, and answering "is this routable?" must not cost a model load.
        When several engines claim a modality the first registered wins, and the
        collision is logged — silent shadowing would be a nasty production surprise.
        """
        with self._lock:
            matches = [e.descriptor.engine_id for e in self._entries.values()
                       if modality in e.descriptor.modalities]
        if len(matches) > 1:
            log.warning(
                "multiple engines claim the same modality; using the first registered",
                extra={"context": {"modality": modality.value, "engines": matches}},
            )
        return matches[0] if matches else None

    def supported_modalities(self) -> set[ImagingModality]:
        with self._lock:
            return {m for e in self._entries.values() for m in e.descriptor.modalities}

    def resolve(self, engine_id: str) -> AnalysisEngine:
        """Return the live engine, constructing it on first use.

        Raises :class:`EngineNotAvailable` if the engine is unknown or its factory
        fails. The failure is cached so a broken engine is not re-constructed on
        every request (a model load can take seconds).
        """
        with self._lock:
            entry = self._entries.get(engine_id)
            if entry is None:
                raise EngineNotAvailable(
                    f"no engine registered as {engine_id!r}",
                    detail={"engine": engine_id},
                )
            if entry.instance is not None:
                return entry.instance
            if entry.load_error is not None:
                raise EngineNotAvailable(
                    "the engine for this modality is not available on this deployment",
                    detail={"engine": engine_id},
                )
            try:
                log.info("constructing engine", extra={"context": {"engine": engine_id}})
                entry.instance = entry.factory()
                return entry.instance
            except Exception as exc:
                entry.load_error = f"{type(exc).__name__}: {exc}"
                log.exception(
                    "engine construction failed; marking unavailable",
                    extra={"context": {"engine": engine_id}},
                )
                raise EngineNotAvailable(
                    "the engine for this modality is not available on this deployment",
                    detail={"engine": engine_id},
                ) from exc

    # ------------------------------------------------------------------ #
    # Plug-in discovery
    # ------------------------------------------------------------------ #
    def discover_plugins(self, group: str = ENTRY_POINT_GROUP) -> list[str]:
        """Load engines advertised by installed packages. Returns the ids added.

        Each entry point must resolve to an :class:`AnalysisEngine` subclass carrying
        a ``descriptor``. A plug-in that fails to import is logged and skipped — one
        bad third-party package must not prevent the platform from starting.
        """
        from importlib.metadata import entry_points

        added: list[str] = []
        try:
            points: Iterable = entry_points(group=group)
        except Exception:                                  # pragma: no cover - stdlib shim
            log.warning("entry-point discovery unavailable on this interpreter")
            return added

        for point in points:
            try:
                cls = point.load()
                descriptor = getattr(cls, "descriptor", None)
                if not (isinstance(cls, type) and issubclass(cls, AnalysisEngine)
                        and isinstance(descriptor, EngineDescriptor)):
                    log.warning(
                        "skipping plug-in: not an AnalysisEngine subclass with a descriptor",
                        extra={"context": {"entry_point": point.name}},
                    )
                    continue
                self.register(descriptor, cls, replace=True)
                added.append(descriptor.engine_id)
            except Exception:
                log.exception(
                    "plug-in engine failed to load; skipping",
                    extra={"context": {"entry_point": getattr(point, 'name', '?')}},
                )
        return added


#: Process-wide registry the API and dispatch service read from.
default_registry = EngineRegistry()


def register_engine(cls: type[AnalysisEngine]) -> type[AnalysisEngine]:
    """Class decorator registering an engine in :data:`default_registry`.

    Suits engines constructible with no arguments::

        @register_engine
        class NeuroMindEngine(AnalysisEngine):
            descriptor = EngineDescriptor(...)
    """
    descriptor = getattr(cls, "descriptor", None)
    if not isinstance(descriptor, EngineDescriptor):
        raise TypeError(f"{cls.__name__} must define a class-level EngineDescriptor")
    default_registry.register(descriptor, cls, replace=True)
    return cls
