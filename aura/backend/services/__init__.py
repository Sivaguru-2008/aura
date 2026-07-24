"""Orchestration services — they compose ``core`` and ``engines``, and own no logic
of their own beyond sequencing and error translation.
"""

from backend.services.dispatch import DispatchService

__all__ = ["DispatchService"]
