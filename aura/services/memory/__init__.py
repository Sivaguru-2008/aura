"""Clinical Memory Engine (P0 seed).

Longitudinal + associative memory: retrieve similar prior cases by evidence
embedding (cosine), and compute a prior-vs-current delta when a prior study for
the same patient exists. In production this is backed by Qdrant + SimpleITK
registration; here it is an in-process vector store over seeded cases.
"""
from services.memory.engine import MemoryEngine

__all__ = ["MemoryEngine"]
