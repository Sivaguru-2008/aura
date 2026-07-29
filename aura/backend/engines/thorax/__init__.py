"""AURA Thorax — chest radiograph analysis.

This package contains an *adapter only*. The chest-X-ray intelligence itself lives
where it always has (``gateway.pipeline``, ``services.vision``, ``services.fusion``,
``services.safety``, ``services.report``) and is not modified, wrapped, or
reimplemented here.
"""

from .engine import ThoraxEngine, register_thorax_engine

__all__ = ["ThoraxEngine", "register_thorax_engine"]
