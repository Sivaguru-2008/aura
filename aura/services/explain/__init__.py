"""Explainability Engine — why did AURA say what it said?

Three grounded explanation types:
  * occlusion saliency   — model-agnostic pixel importance for the top finding.
  * evidence attribution — each evidence node's contribution to the top diagnosis.
  * counterfactuals      — "if this evidence were absent, the top prob shifts by X".
"""
from services.explain.engine import ExplainEngine

__all__ = ["ExplainEngine"]
