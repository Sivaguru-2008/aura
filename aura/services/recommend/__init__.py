"""Missing Evidence Recommendation Engine.

Answers "what should we do next?" by value-of-information: for each candidate
acquisition, how much would it reduce diagnostic entropy, per unit cost and risk.
This is the output no classifier can produce, and it directly attacks over-ordering.
"""
from services.recommend.engine import RecommendEngine

__all__ = ["RecommendEngine"]
