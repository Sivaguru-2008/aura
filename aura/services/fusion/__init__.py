"""Quantum Evidence Fusion Engine.

Fuses a compressed evidence vector (vision findings + structured priors) into a
joint diagnosis posterior. Two interchangeable backends live here:

  * quantum  — a variational quantum circuit (PennyLane) whose entangling layers
               model higher-order interactions between evidence sources.
  * classical — a Bayesian product-of-experts fusion (log-linear).

Backend is chosen by config; both expose the identical `fuse()` contract, and
`ml/evaluation` benchmarks them head-to-head. This is where quantum earns its
place — small, structured, correlation-rich reasoning, not image processing.
"""
from services.fusion.engine import FusionEngine

__all__ = ["FusionEngine"]
