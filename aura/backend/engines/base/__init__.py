"""The engine contract and the plug-in registry that resolves it."""

from .contract import (
    AnalysisEngine,
    AnalysisResult,
    EngineDescriptor,
    EngineReport,
    PreparedStudy,
    ValidationOutcome,
)
from .registry import (
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
