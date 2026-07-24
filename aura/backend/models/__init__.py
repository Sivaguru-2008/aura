"""Wire contracts for the routing layer.

These are the pydantic models the API returns. They are deliberately separate from
``schemas.contracts`` (the clinical domain models): those describe *what a chest case
contains*, these describe *how an upload was routed and what came back*. Keeping them
apart is what lets a future engine return a completely different clinical payload
without touching the routing schema.
"""

from backend.models.routing import (
    AnalysisEnvelope,
    EngineDescriptorModel,
    EngineOutcome,
    ModalityCandidate,
    ResultStatus,
    RoutingMetadata,
)

__all__ = [
    "AnalysisEnvelope",
    "EngineDescriptorModel",
    "EngineOutcome",
    "ModalityCandidate",
    "ResultStatus",
    "RoutingMetadata",
]
