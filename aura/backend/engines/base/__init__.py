"""The engine contract and the plug-in registry that resolves it."""

from backend.engines.base.contract import (
    AnalysisEngine,
    AnalysisResult,
    EngineDescriptor,
    EngineReport,
    PreparedStudy,
    ValidationOutcome,
)
from backend.engines.base.registry import (
    EngineRegistry,
    default_registry,
    register_engine,
)

__all__ = [
    "AnalysisEngine",
    "AnalysisResult",
    "EngineDescriptor",
    "EngineReport",
    "EngineRegistry",
    "PreparedStudy",
    "ValidationOutcome",
    "default_registry",
    "register_engine",
]
