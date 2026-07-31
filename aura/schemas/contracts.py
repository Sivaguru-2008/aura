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

from aura.schemas.clinical import Diagnosis, Finding, Modality


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


class ClinicalContext(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: Optional[str] = None
    previous_diagnosis: Optional[str] = None
    previous_surgery: Optional[str] = None
    radiotherapy: Optional[bool] = None
    chemotherapy: Optional[bool] = None
    clinical_notes: Optional[str] = None


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
    # ``None`` where the producing engine has no calibrated belief or no uncertainty
    # estimate for this item — e.g. a segmented-voxel count, which is a measurement
    # rather than a probabilistic claim. The 0.5 / 0.0 defaults are what a caller that
    # does not set them gets; they are not stand-ins for "unknown".
    probability: float | None = 0.5            # calibrated belief this evidence holds
    uncertainty: float | None = 0.0            # engine-reported uncertainty
    source_service: str = ""
    source: Optional[str] = None
    model: Optional[str] = None
    validation_status: Optional[str] = None


# --------------------------------------------------------------------------- #
# Evidence graph (structured directed graph)
# --------------------------------------------------------------------------- #
class RelationType(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    MEDIATES = "mediates"
    CONTRADICTS = "contradicts"


class EvidenceNode(BaseModel):
    id: str
    kind: EvidenceKind = EvidenceKind.IMAGING_FINDING
    label: str = ""
    value: float = 0.0
    modality: str = ""                         # "imaging"|"labs"|"symptoms"|"history"|"hypothesis"
    confidence: float = 0.5
    features: dict[str, float] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    source_id: str
    target_id: str
    relation: RelationType = RelationType.SUPPORTS
    weight: float = 1.0
    rationale: str = ""


class EvidenceGraph(BaseModel):
    nodes: dict[str, EvidenceNode] = Field(default_factory=dict)
    edges: list[EvidenceEdge] = Field(default_factory=list)

    def add_node(self, node: EvidenceNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        self.edges.append(edge)

    def edges_from(self, node_id: str) -> list[EvidenceEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def edges_to(self, node_id: str) -> list[EvidenceEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def contradicts_edges(self) -> list[EvidenceEdge]:
        return [e for e in self.edges if e.relation == RelationType.CONTRADICTS]

    def supporting_edges(self, node_id: str) -> list[EvidenceEdge]:
        return [e for e in self.edges
                if e.target_id == node_id and e.relation == RelationType.SUPPORTS]


# --------------------------------------------------------------------------- #
# Mitigation actions (uncertainty → workflow)
# --------------------------------------------------------------------------- #
class MitigationAction(str, Enum):
    SHOW_COMPETING_HYPOTHESES = "show_competing_hypotheses"
    ORDER_CONFIRMATORY_EVIDENCE = "order_confirmatory_evidence"
    ESC_HUMAN_EXPERT_REVIEW = "esc_human_expert_review"
    RE_ACQUIRE_SHOTS = "re_acquire_shots"


# --------------------------------------------------------------------------- #
# Clinical Safety Controller (Layer 1)
# --------------------------------------------------------------------------- #
class SafetyControllerCheck(BaseModel):
    """One veto check and its result."""
    name: str                                 # e.g. "data_integrity", "ood_energy", "epistemic"
    passed: bool
    measured: float = 0.0
    threshold: float = 0.0
    severity: float = 0.0                     # sigmoid-scaled violation severity in [0, 1]
    detail: str = ""


class SafetyControllerOutput(BaseModel):
    """Structured output of the ClinicalSafetyController (Layer 1).

    Emitted after the initial veto checks and before reasoning/recommendation.
    If ``state`` is ``FAILED``, the pipeline aborts immediately.
    """
    state: str = "PASSED"                     # "PASSED" | "FAILED" | "WARNING"
    policy_name: str = "community_conservative"
    checks: list[SafetyControllerCheck] = Field(default_factory=list)
    safety_confidence: float = 1.0            # overall safety confidence in [0, 1]
    recommendation: str = ""                  # direct clinical recommendation text
    mitigations: list[MitigationAction] = Field(default_factory=list)
    model_version: str = ""


# --------------------------------------------------------------------------- #
# Clinical Decision Readiness Engine (CDRE) — Layer 2
# --------------------------------------------------------------------------- #
class ReadinessDimension(BaseModel):
    """One dimension of the Decision Readiness Profile."""
    name: str                                 # e.g. "coverage", "quality", "consistency"
    score: float = 0.0                        # [0, 1] where 1 = fully ready
    detail: str = ""


class ReadinessState(str, Enum):
    READY = "ready"
    CONDITIONALLY_READY = "conditionally_ready"
    NOT_READY = "not_ready"


class DecisionReadinessProfile(BaseModel):
    """Layer 2 output — multi-dimensional decision readiness assessment.

    Computed after reasoning and recommendation, this profile tells the
    clinician *why* the system is or is not ready to commit to a diagnosis.
    """
    state: ReadinessState = ReadinessState.NOT_READY
    dimensions: list[ReadinessDimension] = Field(default_factory=list)
    limiting_factor: str = ""                 # the dimension with the lowest score
    limiting_score: float = 1.0
    s_coverage: float = 0.0
    s_quality: float = 0.0
    s_consistency: float = 0.0
    s_robustness: float = 0.0
    s_consensus: float = 0.0
    expected_decision_value: float = 0.0
    consensus_agreement_index: float = 0.0
    recommendation_summary: str = ""
    model_version: str = "cdre-v1"


# --------------------------------------------------------------------------- #
# Patient context (context-aware recommendations)
# --------------------------------------------------------------------------- #
class PatientContext(BaseModel):
    allergies: list[str] = Field(default_factory=list)
    renal_impairment: bool = False
    icu_queue_full: bool = False


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
    #: Entanglement telemetry from services/fusion/qmeasure.measure_entanglement:
    #: measurement entropy, entropy shift vs the product-state baseline, total and
    #: differential channel coupling, the top coupled pairs, and the correlation /
    #: differential matrices. None on classical backends, which have no register to
    #: measure. FusionEngine computed this on every quantum study and passed it here
    #: while the field did not exist — pydantic's default extra='ignore' dropped it
    #: silently, so 31 ms/study of real work was discarded and the console's
    #: entanglement panel (gated on this key) could never render.
    quantum_entanglement: Optional[dict] = None
    #: Did the quantum autoencoder ACTUALLY compress the vision embedding for this
    #: study? Not the `qae_enabled` setting. The compression path additionally needs
    #: a vision embedding and loaded QAE weights, and `QuantumAutoencoder.load()`
    #: returns None when the weights are absent — which is the shipped state. A flag
    #: reporting configuration intent would tell the console a quantum autoencoder
    #: ran when the engine had silently taken the plain-encoding branch.
    qae_applied: bool = False
    created_at: datetime = Field(default_factory=_now)


class MeasurementBudget(BaseModel):
    """What this decision cost in measurements, and what limited it.

    Quantum inference has a knob classical inference does not: precision is bought
    with shots, ``Var[<Z>] = (1 - <Z>^2) / n_shots``. So "how many measurements is
    this patient worth?" is a real question, and answering it sequentially splits
    *unresolved* into two states that call for opposite actions:

    * ``limiting_factor == "measurement"`` — the analytic margin is genuinely
      non-zero; this run just had not bought enough precision. ``predicted_shots``
      says how many more would settle it. **Action: run the circuit longer.**
    * ``limiting_factor == "model"`` — the top two diagnoses are tied at *infinite*
      measurement precision, or the lead is below the clinical-significance floor.
      No budget resolves it. **Action: escalate to a human.**

    A classical softmax of 0.55 means "unsure" and cannot say which kind. Populated
    only when the served fusion backend is quantum; ``None`` otherwise, because a
    product-of-experts has no measurement budget to vary and pretending it does
    would be the same category of fiction the rest of this codebase avoids.
    """
    committed: bool
    top: Diagnosis
    runner_up: Diagnosis
    shots_spent: int
    #: Decision margin p(top) - p(runner_up) at the final budget, and its shot-noise
    #: spread. ``separation_z = margin / margin_std``.
    margin: float
    margin_std: float
    separation_z: float
    #: The margin at infinite shots — the analytic value the serving path uses. This
    #: is the reference that decides whether more measurement could ever have helped.
    analytic_margin: float
    predicted_shots: Optional[int] = None
    limiting_factor: Optional[str] = None      # "measurement" | "model" | None
    floor_limited: bool = False
    reason: str = ""
    #: Per-stage schedule (shots, margin, separation) for the console trace.
    trajectory: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)

    @property
    def measurement_limited(self) -> bool:
        return self.limiting_factor == "measurement"

    @property
    def model_limited(self) -> bool:
        return self.limiting_factor == "model"


# --------------------------------------------------------------------------- #
# Safety & uncertainty
# --------------------------------------------------------------------------- #
class AbstentionReason(str, Enum):
    NONE = "none"
    LOW_CONFIDENCE = "low_confidence"
    LARGE_CONFORMAL_SET = "large_conformal_set"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    HIGH_EPISTEMIC = "high_epistemic_uncertainty"
    # Conditions of *validity*, not of confidence: the model may be perfectly certain
    # and still be outside the regime it was fitted in. Raised by the brain path, where
    # a study missing one of the four required sequences produces a precise number
    # from an input the network was never trained on.
    INCOMPLETE_STUDY = "incomplete_study"
    LOW_QUALITY = "low_quality"


class Prediction(BaseModel):
    diagnosis: Diagnosis
    probability: float                         # calibrated
    # ``None`` where no interval was computed. Not every engine produces one, and a
    # zero-width interval reads as a claim of exactness that was never made.
    ci_low: float | None = None
    ci_high: float | None = None


class SafetyAssessment(BaseModel):
    study_id: str
    predictions: list[Prediction]              # calibrated, ranked
    top: Diagnosis
    top_probability: float
    # Conformal prediction set at the configured coverage (e.g. 90%).
    #
    # Every quantity below is ``| None`` because not every engine measures every one of
    # them, and the alternative is worse than a gap: a 0.0 in ``ood_energy`` or
    # ``epistemic_uncertainty`` renders on the console as a dial reading zero, which a
    # clinician reads as "measured, and reassuring". The chest pipeline sets all of
    # them; the brain path runs a single network with no ensemble, no conformal
    # calibration set and no OOD scorer, and says so by leaving them null.
    conformal_set: list[Diagnosis]
    conformal_coverage: float | None = None
    conformal_method: str = "marginal"         # "marginal" | "mondrian" (class-conditional)
    epistemic_uncertainty: float | None = 0.0  # ensemble top-class disagreement (std)
    aleatoric_uncertainty: float | None = 0.0
    epistemic_mi: float | None = 0.0           # mutual information / BALD (bits)
    predictive_entropy: float | None = 0.0     # total predictive entropy (bits)
    uncertainty_method: str = "input_perturbation"   # or "deep_ensemble"
    n_ensemble: int = 0
    ood_energy: float | None = 0.0
    is_ood: bool | None = False
    abstained: bool = False
    abstention_reason: AbstentionReason = AbstentionReason.NONE
    recommended_mitigations: list[MitigationAction] = Field(default_factory=list)
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
    #: Drawable geometry for the primary saliency map — simplified region polygons,
    #: bounding boxes, intensity-weighted centroids, iso-level contours and an RGBA
    #: overlay PNG data URI (services/explain/geometry.heatmap_geometry). This is the
    #: default view; GET /v1/cases/{id}/geometry recomputes it when the frontend wants
    #: a different threshold, method or overlay size. ExplainEngine computed it on
    #: every study and passed it here while the field did not exist, so pydantic
    #: dropped it — 74 ms/study discarded, and the console's polygon overlay
    #: (console.js reads explanation.geometry) had nothing to draw.
    geometry: dict = Field(default_factory=dict)
    # Shapley-style contribution of each evidence node to the top diagnosis.
    evidence_attribution: dict[str, float] = Field(default_factory=dict)
    # Counterfactual: "if this evidence were removed, top prob changes by ..."
    counterfactuals: dict[str, float] = Field(default_factory=dict)
    method: str = "occlusion+shapley"
    clinical_warning: Optional[str] = None


# --------------------------------------------------------------------------- #
# Missing-evidence recommendation
# --------------------------------------------------------------------------- #
class Recommendation(BaseModel):
    action: str                                # e.g. "acquire_lateral_view"
    display: str                               # human phrasing
    # ``None`` when the producing engine does not run an expected-information-gain
    # computation. The chest path does (see services/eig); the brain path does not, and
    # inventing a utility so the card has a number to sort by would make a recommendation
    # look ranked when it was not.
    expected_info_gain: float | None = None    # bits of diagnostic entropy reduced
    cost_tier: str                             # "low" | "medium" | "high"
    risk_tier: str                             # "none" | "low" | "medium"
    utility: float | None = None               # EIG / (cost * risk) composite
    rationale: str
    reason: Optional[str] = None
    evidence: Optional[list[str]] = None
    confidence: Optional[float] = None


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
    panel_discussion_text: str = ""             # multi-agent panel discussion
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
    evidence_graph: Optional[EvidenceGraph] = None
    fusion: Optional[FusionResult] = None
    #: Measurement economics of the fusion decision. Quantum backends only — see
    #: MeasurementBudget for why a classical backend correctly leaves this None.
    measurement: Optional[MeasurementBudget] = None
    safety: Optional[SafetyAssessment] = None
    explanation: Optional[Explanation] = None
    reasoning: Optional[ReasoningTrace] = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    report: Optional[ReportDraft] = None
    volumes: Optional[dict[str, dict[str, float]]] = None
    clinical_context: Optional[ClinicalContext] = None
    multimodal: Optional[MultimodalContext] = None
    ground_truth: Optional[Diagnosis] = None
    safety_controller: Optional[SafetyControllerOutput] = None
    decision_readiness: Optional[DecisionReadinessProfile] = None
    drp: Optional[DecisionReadinessProfile] = None
    consensus_result: Optional[dict] = None               # multi-agent panel consensus
    created_at: datetime = Field(default_factory=_now)
    dx_labels: dict[str, str] = Field(default_factory=dict)
    ev_labels: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Clinical Q&A Copilot
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ClinicalChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


class ClinicalChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    correlation_id: str = ""
