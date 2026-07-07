"""AURA shared contracts — the single source of truth for every service.

Every engine consumes and produces these Pydantic models. Changing a contract
here is the *only* way to change an inter-service interface; implementations
behind the contract are freely replaceable. Import from `schemas` everywhere.
"""
from schemas.clinical import (
    DIAGNOSES,
    FINDINGS,
    Diagnosis,
    Finding,
    Modality,
)
from schemas.contracts import (
    AbstentionReason,
    CaseBundle,
    CaseState,
    EvidenceItem,
    EvidenceKind,
    Explanation,
    FeedbackVerdict,
    FusionResult,
    Prediction,
    Recommendation,
    ReportDraft,
    SafetyAssessment,
    StudyInput,
    VisionResult,
)

__all__ = [
    "DIAGNOSES",
    "FINDINGS",
    "Diagnosis",
    "Finding",
    "Modality",
    "AbstentionReason",
    "CaseBundle",
    "CaseState",
    "EvidenceItem",
    "EvidenceKind",
    "Explanation",
    "FeedbackVerdict",
    "FusionResult",
    "Prediction",
    "Recommendation",
    "ReportDraft",
    "SafetyAssessment",
    "StudyInput",
    "VisionResult",
]
