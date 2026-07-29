"""AURA NeuroMind — brain MRI analysis.

Currently a **placeholder**. It implements the full engine contract and participates
in routing, but ``analyze`` deliberately refuses rather than returning a fabricated
result. See ``engine.py`` for what is real today and what must be built.
"""

from .engine import NeuroMindEngine

__all__ = ["NeuroMindEngine"]
