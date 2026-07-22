"""Inter-service data contracts.

Flow of a case through the engines, each stage adding to the bundle:

    StudyInput
      -> VisionResult        (vision)
      -> FusionResult        (fusion: quantum or classical)
      -> SafetyAssessment    (safety: calibration, conformal, OOD, abstention)
      -> [Explanation]       (explain)
      -> [Recommendation]    (recommend: expected information gain)
      -> ReportDraft         (report)
    == CaseBundle (everything the dashboard renders)
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from schemas.clinical import Diagnosis, Finding, Modality


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class StructuredPriors(BaseModel):
    """De-identified structured context that legitimately shifts priors."""
    age_band: str = "unknown"          # e.g. "18-40", "40-65", "65+"
    sex: str = "unknown"
    smoker: bool = False
    fever: bool = False
    prior_cancer: bool = False
    immunocompromised: bool = False


class LabPanel(BaseModel):
    """De-identified lab values. None = not resulted (missing, not zero)."""
    wbc: Optional[float] = None            # x10^9/L  (>11 leukocytosis)
    neutrophil_pct: Optional[float] = None
    crp: Optional[float] = None            # mg/L    (inflammation)
    procalcitonin: Optional[float] = None  # ng/mL   (bacterial)
    bnp: Optional[float] = None            # pg/mL   (>400 cardiac)
    troponin: Optional[float] = None       # ng/L
    d_dimer: Optional[float] = None        # ng/mL
    spo2: Optional[float] = None           # %


class Symptoms(BaseModel):
    dyspnea: bool = False
    productive_cough: bool = False
    fever: bool = False
    pleuritic_chest_pain: bool = False
    hemoptysis: bool = False
    orthopnea: bool = False
    acute_onset: bool = False


class ClinicalHistory(BaseModel):
    copd: bool = False
    heart_failure: bool = False
    prior_cancer: bool = False
    recent_surgery: bool = False
    immunosuppression: bool = False
    smoking_pack_years: float = 0.0


class MultimodalContext(BaseModel):
    """Non-imaging evidence the reasoning engine fuses with the image posterior."""
    labs: LabPanel = Field(default_factory=LabPanel)
    symptoms: Symptoms = Field(default_factory=Symptoms)
    history: ClinicalHistory = Field(default_factory=ClinicalHistory)


class StudyInput(BaseModel):
    study_id: str
    modality: Modality = Modality.CXR
    # Normalized image as a flat grayscale array + shape (kept simple/offline).
    image: list[float] = Field(default_factory=list)
    image_shape: tuple[int, int] = (64, 64)
    priors: StructuredPriors = Field(default_factory=StructuredPriors)
    # Optional non-imaging evidence (labs / symptoms / history).
    multimodal: Optional[MultimodalContext] = None
    # Optional ground-truth diagnosis — present only for synthetic/seed data.
    ground_truth: Optional[Diagnosis] = None


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
class FindingScore(BaseModel):
    finding: Finding
    probability: float
    # Bounding region (row0,col0,row1,col1) in normalized [0,1] coords for overlay.
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


class VisionResult(BaseModel):
    study_id: str
    findings: list[FindingScore]
    embedding: list[float]                     # for memory / similarity
    model_version: str
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# Evidence graph
# --------------------------------------------------------------------------- #
class EvidenceKind(str, Enum):
    IMAGING_FINDING = "imaging_finding"
    STRUCTURED_PRIOR = "structured_prior"
    PRIOR_DELTA = "prior_delta"
    CLINICIAN_INPUT = "clinician_input"
    ABSENT_EVIDENCE = "absent_evidence"        # evidence we *don't* have yet


class EvidenceItem(BaseModel):
    kind: EvidenceKind
    name: str
    value: float                               # normalized strength in [0,1]
    probability: float = 0.5                   # calibrated belief this evidence holds
    uncertainty: float = 0.0                   # engine-reported uncertainty
    source_service: str = ""


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
class FusionResult(BaseModel):
    study_id: str
    backend: str                               # "quantum" | "classical"
    # Joint posterior over diagnoses (sums to 1).
    posterior: dict[Diagnosis, float]
    # Fusion-level uncertainty (e.g. shot-noise std for quantum).
    posterior_std: dict[Diagnosis, float] = Field(default_factory=dict)
    evidence_vector: list[float] = Field(default_factory=list)
    n_shots: int = 0
    model_version: str = ""
    # Conflict-resolution (Wasserstein tie-breaker) telemetry. Defaults keep the
    # field backward-compatible with bundles written before Module 5 landed.
    resolved_backend: str = ""                 # backend actually trusted after tie-break
    conflict_distance: float = 0.0             # EMD(VQC, PoE) on the severity axis
    conflict_threshold: float = 0.0            # dynamic τ used for the decision
    fallback_triggered: bool = False           # True when the PoE fallback fired
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# Safety & uncertainty
# --------------------------------------------------------------------------- #
class AbstentionReason(str, Enum):
    NONE = "none"
    LOW_CONFIDENCE = "low_confidence"
    LARGE_CONFORMAL_SET = "large_conformal_set"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    HIGH_EPISTEMIC = "high_epistemic_uncertainty"


class Prediction(BaseModel):
    diagnosis: Diagnosis
    probability: float                         # calibrated
    ci_low: float
    ci_high: float


class SafetyAssessment(BaseModel):
    study_id: str
    predictions: list[Prediction]              # calibrated, ranked
    top: Diagnosis
    top_probability: float
    # Conformal prediction set at the configured coverage (e.g. 90%).
    conformal_set: list[Diagnosis]
    conformal_coverage: float
    conformal_method: str = "marginal"         # "marginal" | "mondrian" (class-conditional)
    epistemic_uncertainty: float = 0.0         # ensemble top-class disagreement (std)
    aleatoric_uncertainty: float = 0.0
    epistemic_mi: float = 0.0                  # mutual information / BALD (bits)
    predictive_entropy: float = 0.0            # total predictive entropy (bits)
    uncertainty_method: str = "input_perturbation"   # or "deep_ensemble"
    n_ensemble: int = 0
    ood_energy: float = 0.0
    is_ood: bool = False
    abstained: bool = False
    abstention_reason: AbstentionReason = AbstentionReason.NONE
    model_version: str = ""


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
class Explanation(BaseModel):
    study_id: str
    # Primary saliency heatmap (Grad-CAM++ for the CNN, occlusion otherwise),
    # flattened, same shape as the input image. Existing consumers read this.
    saliency: list[float] = Field(default_factory=list)
    saliency_shape: tuple[int, int] = (64, 64)
    # Additional attribution maps keyed by method name (grad_cam, grad_cam++,
    # integrated_gradients, smoothgrad, occlusion) — each flattened to saliency_shape.
    saliency_methods: dict[str, list[float]] = Field(default_factory=dict)
    saliency_target: str = ""                  # finding the maps localize
    # Shapley-style contribution of each evidence node to the top diagnosis.
    evidence_attribution: dict[str, float] = Field(default_factory=dict)
    # Counterfactual: "if this evidence were removed, top prob changes by ..."
    counterfactuals: dict[str, float] = Field(default_factory=dict)
    method: str = "occlusion+shapley"


# --------------------------------------------------------------------------- #
# Missing-evidence recommendation
# --------------------------------------------------------------------------- #
class Recommendation(BaseModel):
    action: str                                # e.g. "acquire_lateral_view"
    display: str                               # human phrasing
    expected_info_gain: float                  # bits of diagnostic entropy reduced
    cost_tier: str                             # "low" | "medium" | "high"
    risk_tier: str                             # "none" | "low" | "medium"
    utility: float                             # EIG / (cost * risk) composite
    rationale: str


# --------------------------------------------------------------------------- #
# Clinical reasoning
# --------------------------------------------------------------------------- #
class ReasoningStep(BaseModel):
    """One evidence-grounded inference: a statement, the evidence it rests on, its
    effect on the differential (log-likelihood-ratio per diagnosis), and a citation."""
    statement: str
    evidence: list[str] = Field(default_factory=list)
    effect: dict[Diagnosis, float] = Field(default_factory=dict)   # log-LR nudges
    guideline: str = ""
    modality: str = ""                         # "imaging" | "labs" | "symptoms" | "history"


class DifferentialItem(BaseModel):
    diagnosis: Diagnosis
    probability: float
    supporting: list[str] = Field(default_factory=list)
    opposing: list[str] = Field(default_factory=list)


class ReasoningTrace(BaseModel):
    study_id: str
    prior_posterior: dict[Diagnosis, float] = Field(default_factory=dict)
    adjusted_posterior: dict[Diagnosis, float] = Field(default_factory=dict)
    steps: list[ReasoningStep] = Field(default_factory=list)
    differential: list[DifferentialItem] = Field(default_factory=list)
    guideline_citations: list[str] = Field(default_factory=list)
    model_version: str = "reasoning-v1"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
class ReportDraft(BaseModel):
    study_id: str
    findings_text: str
    impression_text: str
    recommendation_text: str
    differential_text: str = ""                 # ranked alternatives with evidence
    confidence_text: str = ""                   # calibrated confidence + why
    # Every sentence maps to the evidence nodes that grounded it.
    grounding: dict[str, list[str]] = Field(default_factory=dict)
    generator: str = "structured+template"


# --------------------------------------------------------------------------- #
# Case bundle & feedback
# --------------------------------------------------------------------------- #
class CaseState(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    READY = "ready"
    IN_REVIEW = "in_review"
    SIGNED = "signed"
    ABSTAINED = "abstained"


class FeedbackVerdict(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class CaseBundle(BaseModel):
    """Everything the dashboard renders for one case."""
    case_id: str
    study_id: str
    state: CaseState
    priority_score: float = 0.0
    priors: StructuredPriors = Field(default_factory=StructuredPriors)
    image: list[float] = Field(default_factory=list)
    image_shape: tuple[int, int] = (64, 64)
    vision: Optional[VisionResult] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    fusion: Optional[FusionResult] = None
    safety: Optional[SafetyAssessment] = None
    explanation: Optional[Explanation] = None
    reasoning: Optional[ReasoningTrace] = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    report: Optional[ReportDraft] = None
    multimodal: Optional[MultimodalContext] = None
    ground_truth: Optional[Diagnosis] = None
    created_at: datetime = Field(default_factory=_now)
    dx_labels: dict[str, str] = Field(default_factory=dict)
    ev_labels: dict[str, str] = Field(default_factory=dict)
