# AURA — REST API Specifications

This document catalogs the REST API endpoints and data schemas exposed by AURA's FastAPI gateway.

---

## 1. Core Endpoints

### Health check
* **Method**: `GET`
* **Path**: `/health`
* **Response**: `{"status": "ok", "version": "1.0.0"}`

### Get Cases
* **Method**: `GET`
* **Path**: `/v1/cases`
* **Description**: Returns all patient cases in the triage worklist sorted by priority score.
* **Response**: `List[CaseBundle]`

### Get Case Details
* **Method**: `GET`
* **Path**: `/v1/cases/{case_id}`
* **Description**: Returns the full analysis bundle for a specific case (findings, posterior, conformal sets, saliency maps, report draft).
* **Response**: `CaseBundle`

### Upload Study (Intake Gate)
* **Method**: `POST`
* **Path**: `/v1/studies/upload`
* **Description**: Stream-uploads a chest radiograph (PNG, JPEG, or DICOM). Enforces file-type and size security validation, executes the 9-stage analysis pipeline, and saves the case.
* **Payload**: Multipart form file.
* **Response**: `CaseBundle`

### Submit Clinician Feedback
* **Method**: `POST`
* **Path**: `/v1/cases/{case_id}/feedback`
* **Description**: Logs clinician verdicts (accept, edit, reject) per prediction. Triggers an update to the online Adaptive Conformal Inference (ACI) threshold.
* **Payload**: `{"finding": str, "verdict": "accept" | "edit" | "reject", "corrected_value": float}`
* **Response**: `{"status": "saved", "aci_qhat": float}`

### Sign Clinical Report
* **Method**: `POST`
* **Path**: `/v1/cases/{case_id}/report/sign`
* **Description**: Signs off the report draft. Transitions case status from `READY` to `SIGNED`.
* **Response**: `{"status": "signed", "case_id": str}`

### Get Model Registry
* **Method**: `GET`
* **Path**: `/v1/models`
* **Description**: Lists active and fallback model versions, calibration JSON metrics, and dataset provenances.
* **Response**: `ModelRegistry`

### Admin Safety Dashboard
* **Method**: `GET`
* **Path**: `/v1/admin/safety`
* **Description**: Returns safety metrics (ECE, Brier, ACI conformal threshold, OOD energy statistics) and the system append-only audit log.
* **Response**: `SafetyDashboardResponse`

---

## 2. Core Pydantic Contracts

The schemas are defined in `aura/schemas/contracts.py`.

### `StudyInput`
```python
class StudyInput(BaseModel):
    study_id: str
    image: List[float]  # Flat normalized pixel array
    image_shape: Tuple[int, int]
    priors: StructuredPriors
    multimodal: Optional[MultimodalContext] = None
    ground_truth: Optional[Dict[str, int]] = None
```

### `CaseBundle`
```python
class CaseBundle(BaseModel):
    case_id: str
    study_id: str
    state: CaseState  # QUEUED | ANALYZING | READY | SIGNED | ABSTAINED
    priority_score: float
    priors: StructuredPriors
    image: List[float]
    image_shape: Tuple[int, int]
    vision: VisionResult
    evidence: List[EvidenceItem]
    fusion: FusionResult
    safety: SafetyAssessment
    explanation: ExplanationBundle
    reasoning: ReasoningSteps
    recommendations: List[Recommendation]
    report: GroundedReport
```
