"""Clinical reasoning service — fuse image, metadata, labs, symptoms, history,
and guidelines into an evidence-grounded, adjusted differential."""
from .engine import ClinicalReasoner

__all__ = ["ClinicalReasoner"]
