# AURA — Clinical Intelligence Copilot
### Product & System Architecture v1.0 (pre-implementation)

> **AURA** — *Adaptive Uncertainty-aware Reasoning Assistant*
> "The copilot that knows what it doesn't know."
>
> Status: **DRAFT — awaiting founder approval before implementation begins.**

---

## 1. Product vision

Every diagnostic AI on the market today answers one question: *"What disease is in this image?"*
That is the wrong question. Radiologists are already good at pattern recognition. What breaks
clinical workflows is **uncertainty management**: incomplete evidence, ambiguous findings,
missed priors, no time to compare studies, and reports that take longer to write than to read.

AURA creates a new category: **Clinical Epistemic Intelligence** — software whose core job is
to model *what is known, how confidently it is known, what is missing, and what to do next*.

The core primitive is the **Evidence Graph**: every case is a living graph of evidence nodes
(imaging findings, prior studies, structured priors, clinician annotations), each carrying a
calibrated probability and an uncertainty estimate. AURA's engines operate on this graph:

1. **Fuse** heterogeneous evidence into a joint diagnostic posterior (quantum-assisted).
2. **Quantify** aleatoric vs. epistemic uncertainty, and *abstain* when unsure.
3. **Explain** every output — visually, statistically, and in clinical language.
4. **Identify gaps** — which missing evidence, if acquired, would most reduce uncertainty
   (expected information gain / value-of-information).
5. **Recommend** the next best diagnostic step, ranked by information gain per cost/risk.
6. **Compare** against the patient's prior studies (longitudinal delta analysis).
7. **Generate** a structured, grounded, clinician-editable report.
8. **Learn** from every clinician correction (feedback → calibration → retraining queue).

The doctor is always the decision-maker. AURA is decision *support* with an auditable
epistemic trail — which is also exactly what regulators (FDA SaMD, EU MDR/AI Act) reward.

**North-star metric:** minutes of radiologist time saved per case at equal-or-better
diagnostic accuracy, with zero silent failures (every low-confidence output flagged).

---

## 2. User personas

| Persona | Who | Goals | Frustrations | AURA touchpoints |
|---|---|---|---|---|
| **Dr. Meera, attending radiologist (11 yrs)** | Reads 80–120 studies/day | Speed without missed findings; defensible reports | Alert fatigue from binary AI tools; no context from priors | Copilot panel, report generator, prior comparison |
| **Dr. James, ER physician (nights)** | Orders imaging, acts before radiology reads | Fast triage; know when to escalate | Long turnaround at 3am; over-ordering "to be safe" | Worklist triage, next-best-test recommendation |
| **Priya, radiology resident (PGY-3)** | Learning to read | Understand *why*, not just *what* | Black-box AI teaches nothing | Explainability engine as teaching mode |
| **Dr. Chen, CMIO / medical director** | Buys and governs hospital AI | Safety, auditability, ROI, liability | Vendors can't answer "when is your model wrong?" | Safety dashboard, calibration reports, audit log |
| **Alex, PACS/IT administrator** | Runs imaging infrastructure | Zero-friction integration, on-prem option, no PHI leakage | Cloud-only AI vendors | DICOM gateway, dockerized on-prem deploy, RBAC |

---

## 3. Hospital workflow

### 3.1 Today (without AURA)
```
Order placed (EHR/RIS) → Scan acquired (modality) → Images to PACS
→ Study sits in FIFO worklist (minutes–hours)
→ Radiologist opens study cold (priors often unfetched)
→ Reads, dictates free-text report (8–15 min/case)
→ Report to EHR → referring MD interprets prose
→ Follow-up recommendations frequently lost ("failure to close the loop")
```

### 3.2 With AURA (copilot inserted, never a gatekeeper)
```
Images to PACS ──tap──> AURA DICOM Gateway (de-identify, normalize)
    → Vision Engine pre-analyzes (async, before human opens study)
    → Memory Engine auto-fetches & registers priors, computes deltas
    → Fusion Engine produces calibrated differential + uncertainty
    → Worklist re-prioritized: high-suspicion + high-confidence cases float up;
      high-uncertainty cases flagged "needs senior read", never auto-cleared
Radiologist opens study
    → Copilot panel: findings, saliency overlays, confidence intervals,
      prior-scan comparison, missing-evidence checklist, next-best-step
    → One-click accept/edit/reject per finding  (feedback captured)
    → Draft report pre-generated; radiologist edits & signs (2–4 min/case)
    → Structured report + epistemic audit trail to EHR
Feedback loop: every correction → calibration update + retraining queue
```

Integration points: DICOM C-STORE/DICOMweb (images in), HL7 FHIR `DiagnosticReport` /
`ImagingStudy` (reports out), OAuth2/SMART-on-FHIR (identity), no change to the
radiologist's primary viewer required (AURA runs as a companion panel; PACS plugin later).

---

## 4. Pain points (ranked by willingness-to-pay)

1. **Report turnaround & radiologist burnout** — global shortage; 100+ studies/day; dictation
   is the bottleneck. *(AURA: pre-drafted grounded reports.)*
2. **Silent AI failure** — existing CAD tools give a score with no notion of "I don't know",
   causing alert fatigue and mistrust. *(AURA: calibrated confidence + explicit abstention.)*
3. **Priors ignored** — comparing previous scans is manual and often skipped under load.
   *(AURA: automatic longitudinal registration + delta report.)*
4. **Incomplete workups & over-ordering** — next test chosen by habit, not information value.
   *(AURA: value-of-information ranked next-best-step.)*
5. **Lost follow-ups** — incidental findings with recommended follow-up never tracked.
   *(AURA: Clinical Memory Engine tracks open recommendations.)*
6. **Unaccountable AI** — hospitals can't audit why a model said what it said.
   *(AURA: per-prediction explanation + immutable audit log.)*

---

## 5. Unique value proposition

> **Every other tool sells answers. AURA sells calibrated judgment.**

- Only platform whose *primary* outputs are **uncertainty, evidence gaps, and next actions** —
  the diagnosis probability is one field among them, not the product.
- **Abstention as a feature**: AURA is contractually allowed to say "insufficient evidence,
  here is what would resolve it" — this is what makes it safe, trustworthy, and regulable.
- **Quantum evidence fusion** where it genuinely helps: modeling higher-order correlations
  between evidence sources in a compact variational model — not gimmick quantum CNNs.
- **Closed learning loop**: clinician feedback is a first-class data asset that continuously
  improves calibration — the data moat compounds per hospital.
- Deployable **on-prem** (single GPU box) — the deciding factor for most hospital IT teams.

---

## 6. Feature hierarchy

**P0 — demo-critical (hackathon cut)**
- F1 Image ingestion (DICOM + PNG/JPEG dev mode), de-identification
- F2 Vision Engine: chest X-ray multi-label findings (backbone: pretrained CXR model)
- F3 Uncertainty: MC-dropout + deep-ensemble variance, temperature-scaled calibration,
  conformal prediction sets, OOD detection (energy score)
- F4 Quantum Evidence Fusion: variational circuit fusing vision logits + structured priors
  → joint posterior (simulated backend, classical fallback switch)
- F5 Explainability: Grad-CAM++ overlays, per-evidence contribution attribution, counterfactual
  sensitivity ("if this region were absent, probability drops 0.34")
- F6 Missing-evidence engine: expected-information-gain ranking of candidate next tests
- F7 Report generator: structured JSON → clinician-style report (LLM, grounded, no PHI in prompt)
- F8 Doctor dashboard: worklist, case view (image + overlays + confidence UI), accept/edit/reject
- F9 Feedback capture + audit log

**P1 — post-hackathon**
- Prior-study comparison (registration + delta maps), longitudinal timeline
- Model registry UI, calibration drift monitoring, per-site fine-tuning queue
- FHIR DiagnosticReport export; multi-modality (CT slices)

**P2 — venture scale**
- Multimodal fusion (labs, notes via clinical LLM embeddings, genomics)
- Federated learning across hospitals; quantum hardware backends
- Follow-up tracking & closure ("no incidental finding left behind")

---

## 7. System architecture (high level)

```
                        ┌────────────────────────────────────────────┐
                        │              Doctor Dashboard (Next.js)     │
                        └───────────────▲────────────────────────────┘
                                        │ HTTPS / WebSocket
┌──────────────┐   DICOM    ┌───────────┴───────────┐
│ PACS / RIS / │──────────► │   API Gateway (FastAPI)│  AuthN/Z, rate limit,
│  Modalities  │  FHIR      │   + async job queue    │  audit middleware
└──────────────┘ ◄──────────└───────────┬───────────┘
                                        │ events (Redis Streams / NATS)
        ┌──────────┬──────────┬─────────┼──────────┬───────────┬──────────┐
        ▼          ▼          ▼         ▼          ▼           ▼          ▼
   ┌─────────┐┌─────────┐┌─────────┐┌────────┐┌──────────┐┌─────────┐┌─────────┐
   │ Vision  ││ Quantum ││ Explain ││ Safety ││ Evidence ││ Report  ││ Memory  │
   │ Engine  ││ Fusion  ││ Engine  ││ Engine ││ Recommend││ Gen     ││ Engine  │
   └────┬────┘└────┬────┘└────┬────┘└───┬────┘└────┬─────┘└────┬────┘└────┬────┘
        └──────────┴──────────┴─────────┴──────────┴───────────┴──────────┘
                                        │
              ┌─────────────┬───────────┴────────┬──────────────┐
              ▼             ▼                    ▼              ▼
        PostgreSQL     Object store          Qdrant         MLflow
        (clinical      (images, overlays,   (embeddings,    (model registry,
         records)       artifacts — S3/MinIO) case similarity) versions, metrics)
```

Communication: synchronous REST for reads; **event-driven async pipeline** for analysis
(`study.received → vision.completed → fusion.completed → case.ready`). Every service is
independently deployable, versioned, and replaceable behind a typed contract
(`packages/schemas` — single source of truth, Pydantic models shared by all services).

---

## 8. Microservice architecture

| # | Service | Responsibility | Tech | Scaling |
|---|---------|----------------|------|---------|
| 1 | `gateway` | AuthN/Z (OIDC + RBAC), REST/WS API, job orchestration, audit middleware | FastAPI, Redis | horizontal, stateless |
| 2 | `ingestion` | DICOM/DICOMweb listener, de-identification (PHI strip), normalization, storage | pydicom, MinIO | per-site |
| 3 | `vision` | Multi-label finding detection, feature/embedding extraction | PyTorch/MONAI → ONNX Runtime | GPU pool, batch |
| 4 | `fusion` | Quantum evidence fusion: evidence vectors → joint posterior | PennyLane (default.qubit sim; hw-pluggable), classical fallback (product-of-experts Bayes) | CPU, async |
| 5 | `explain` | Grad-CAM++/attention maps, evidence attribution (Shapley over evidence nodes), counterfactuals | Captum, custom | GPU/CPU |
| 6 | `safety` | Calibration (temp scaling), conformal prediction sets, OOD detection, abstention policy, drift monitors | scikit-learn, custom | CPU |
| 7 | `recommend` | Missing-evidence detection + expected-information-gain ranking of next tests | Bayesian VoI over evidence graph | CPU |
| 8 | `report` | Structured findings JSON → clinician report; strictly grounded (template + LLM polish, de-identified) | LLM API / local LLM adapter | CPU |
| 9 | `memory` | Longitudinal patient timeline, prior registration & delta, case-similarity retrieval, open-recommendation tracking | Qdrant, SimpleITK | CPU |
| 10 | `models` | Model registry, versioning, canary rollout, feedback-driven retraining queue | MLflow | control plane |
| 11 | `web` | Doctor dashboard | Next.js 14, TypeScript, Tailwind, shadcn/ui | CDN/edge |

Every inference service implements the same contract: `POST /analyze {case_id, inputs} →
job_id`, emits `<service>.completed` events, writes artifacts to object store, metadata to
Postgres. Swap any implementation without touching the rest — the modularity requirement.

---

## 9. AI architecture

### 9.1 Vision Intelligence Engine
- Backbone: pretrained CXR foundation encoder (e.g. torchxrayvision DenseNet-121 for P0;
  swappable to a ViT/foundation model later) → multi-label finding heads.
- Outputs per finding: logit, embedding (for memory/similarity), saliency tensor handle.
- Exported to ONNX for deterministic, backend-agnostic serving.
- **The vision model is deliberately commodity.** The moat is everything downstream.

### 9.2 Confidence & Safety Engine (the credibility core)
- **Epistemic uncertainty**: deep ensemble (3–5 seeds) + MC-dropout variance.
- **Aleatoric uncertainty**: learned heteroscedastic head (P1).
- **Calibration**: temperature scaling per finding, ECE tracked per model version and per site.
- **Conformal prediction**: distribution-free prediction *sets* with coverage guarantee
  (e.g. "90% coverage: {pneumonia, atelectasis}") — statistically rigorous, demo-friendly.
- **OOD detection**: energy score + Mahalanobis distance on embeddings → "this image is
  outside my training distribution" banner.
- **Abstention policy**: configurable per-hospital thresholds; abstained cases route to
  senior worklist. No silent failure, ever.

### 9.3 Missing Evidence Recommendation Engine
- Evidence graph: nodes = evidence items (present or *absent*), edges = dependencies.
- For each candidate acquisition `a` (lateral view, CT, labs, prior retrieval):
  `EIG(a) = H(D | E) − E_outcomes[H(D | E ∪ a)]` — expected reduction in diagnostic entropy,
  computed via the fusion engine's posterior. Ranked by EIG / (cost × risk).
- Output: "A lateral view would reduce diagnostic uncertainty by 41%; recommended next."

### 9.4 Clinical Report Generator
- Two-stage: (1) deterministic structured composer builds the findings/impression skeleton
  from fused, thresholded outputs — *every sentence traceable to an evidence node*;
  (2) LLM polishes phrasing only, with a validator that rejects any generated claim lacking
  a source node (hallucination gate). De-identified input only.

### 9.5 Learning loop
- Every accept/edit/reject → `feedback` table → nightly calibration refresh (cheap, safe)
  → periodic fine-tune candidates in `models` registry → canary → promote. Site-local first;
  federated aggregation at P2.

---

## 10. Quantum architecture

**Where quantum is used — and where it is not.** No quantum image processing; classical CNNs
won that. Quantum is applied to the small, structured, correlation-rich core of the problem:
**evidence fusion and probabilistic reasoning** over ~8–16 compressed evidence features.

### 10.1 Quantum Evidence Fusion Engine (service `fusion`)
- **Input**: compressed evidence vector `e ∈ R^n` (n=8–12): top vision logits, prior-delta
  score, structured risk priors (age band, history flags), OOD score.
- **Encoding**: angle encoding, one qubit per evidence feature.
- **Ansatz**: 3–4 layers of parameterized rotations + ring entanglement (`CNOT` ladder).
  Entangling gates model **higher-order interactions between evidence sources** — e.g.
  "opacity + effusion + fever prior" jointly shifting the posterior in a way linear/pairwise
  classical fusion misses — in an exponentially large Hilbert space with O(n·layers) params.
- **Readout**: Pauli-Z expectations per diagnosis head → softmax → joint posterior.
- **Uncertainty for free**: finite-shot measurement statistics give a natural sampling
  distribution over the posterior; shot-variance is reported as fusion-level uncertainty.
- **Training**: hybrid — parameter-shift gradients through PennyLane, jointly trained with a
  classical projection layer on frozen vision embeddings.

### 10.2 Quantum kernel case similarity (P1)
- Fidelity kernel `k(x,x') = |⟨φ(x)|φ(x')⟩|²` on evidence vectors for rare-presentation
  similarity search in low-data regimes, feeding the Memory Engine.

### 10.3 Honest engineering posture
- Backend abstraction: `QuantumDevice` interface → `default.qubit` simulator today;
  `qiskit.ibmq` / Braket adapters later. **Classical fallback (Bayesian product-of-experts
  fusion) ships in the same service behind a config flag** — the demo never depends on a
  simulator quirk, and A/B comparison quantum-vs-classical is a built-in benchmark we show
  judges rather than hand-wave.

---

## 11. Folder structure (monorepo)

```
aura/
├── docs/                          # this file, ADRs, api spec, demo script
│   ├── ARCHITECTURE.md
│   └── adr/                       # architecture decision records
├── packages/
│   ├── schemas/                   # Pydantic + TS types — single source of truth
│   └── clients/                   # generated API clients
├── services/
│   ├── gateway/                   # FastAPI app, auth, orchestration, audit
│   │   ├── app/{api,core,jobs}/
│   │   └── tests/
│   ├── ingestion/                 # dicom listener, de-id, normalize
│   ├── vision/                    # model wrappers, onnx serving
│   │   └── app/{models,serving}/
│   ├── fusion/                    # quantum + classical fusion
│   │   └── app/{circuits,classical,device}/
│   ├── explain/
│   ├── safety/                    # calibration, conformal, ood, abstain
│   ├── recommend/                 # evidence graph + EIG
│   ├── report/
│   ├── memory/
│   └── models/                    # registry integration, retrain queue
├── apps/
│   └── web/                       # Next.js dashboard
│       └── src/{app,components,lib,hooks}/
├── ml/
│   ├── notebooks/                 # experiments only, never imported by services
│   ├── training/                  # ensemble training, calibration fitting
│   └── evaluation/                # ECE, coverage, AUROC, quantum-vs-classical bench
├── infra/
│   ├── docker-compose.yml         # full local/hackathon stack
│   ├── k8s/                       # helm charts (P1)
│   └── scripts/
├── data/                          # gitignored; sample de-identified studies
├── Makefile
└── README.md
```

---

## 12. Database schema (PostgreSQL)

```sql
-- Identity & access
users(id PK, email UNIQUE, name, role ENUM(radiologist,physician,resident,admin),
      org_id FK, auth_provider_sub, created_at)
organizations(id PK, name, settings JSONB)          -- per-hospital config, thresholds

-- Clinical core (all PHI encrypted at column level; de-id map held separately)
patients(id PK, org_id FK, mrn_hash, demographics_enc BYTEA, created_at)
studies(id PK, patient_id FK, modality, body_part, study_uid UNIQUE,
        acquired_at, received_at, status ENUM(received,processing,ready,read,signed))
images(id PK, study_id FK, series_uid, sop_uid UNIQUE, storage_uri, meta JSONB)

-- Epistemic core
cases(id PK, study_id FK, state ENUM(queued,analyzed,in_review,signed,abstained),
      priority_score REAL, assigned_to FK users, created_at)
evidence_items(id PK, case_id FK, kind ENUM(imaging_finding,prior_delta,structured_prior,
      clinician_input, absent_evidence), payload JSONB, source_service,
      probability REAL, uncertainty REAL, created_at)
predictions(id PK, case_id FK, model_version_id FK, label, probability REAL,
      ci_low REAL, ci_high REAL, conformal_set JSONB, abstained BOOL, created_at)
explanations(id PK, prediction_id FK, method ENUM(gradcam,attribution,counterfactual),
      artifact_uri, summary JSONB)
recommendations(id PK, case_id FK, action, expected_info_gain REAL, cost_tier,
      rationale TEXT, status ENUM(open,ordered,completed,dismissed), created_at)
reports(id PK, case_id FK, draft JSONB, final_text TEXT, signed_by FK users,
      signed_at, fhir_exported BOOL)

-- Learning loop & governance
feedback(id PK, prediction_id FK, user_id FK, verdict ENUM(accept,edit,reject),
      correction JSONB, created_at)
model_versions(id PK, service, name, version, registry_uri, calibration JSONB,
      metrics JSONB, status ENUM(canary,active,retired), promoted_at)
audit_log(id PK, actor_id, action, entity_type, entity_id, detail JSONB,
      created_at)  -- append-only, no UPDATE/DELETE grants
```

Vector store (Qdrant): `case_embeddings` collection — image/evidence embeddings keyed by
`case_id` for similarity retrieval and longitudinal linking. Object store (MinIO/S3):
raw DICOM, normalized tensors, saliency overlays, report PDFs.

---

## 13. API design (gateway, `/v1`)

Async-first: heavy analysis returns `202 + job_id`; clients follow WebSocket or poll.

```
POST   /v1/studies                       # register/upload study (DICOM or dev PNG)
GET    /v1/studies/{id}
POST   /v1/cases/{id}/analyze            # kick full pipeline → 202 {job_id}
GET    /v1/jobs/{job_id}                 # {status, progress, stage}
GET    /v1/cases                         # worklist: ?state=&priority=&assigned_to=
GET    /v1/cases/{id}                    # full case bundle:
                                         #   predictions + uncertainty + conformal sets,
                                         #   evidence graph, explanations, recommendations,
                                         #   prior comparison, report draft
GET    /v1/cases/{id}/explanations/{pred_id}
GET    /v1/cases/{id}/recommendations
POST   /v1/cases/{id}/feedback           # {prediction_id, verdict, correction?}
POST   /v1/cases/{id}/report/sign
GET    /v1/patients/{id}/timeline        # memory engine longitudinal view
GET    /v1/models                        # registry: versions, calibration, status
POST   /v1/models/{id}/promote           # admin only
GET    /v1/admin/safety                  # ECE, coverage, drift, abstention rate
WS     /v1/ws/cases                      # live worklist + job events
```

Conventions: OAuth2 bearer (OIDC), org-scoped RBAC enforced at gateway; every mutating call
writes `audit_log`; all responses carry `model_version` provenance; errors RFC 7807.
Internal service-to-service messages ride Redis Streams with the shared Pydantic event
schemas (`study.received`, `vision.completed`, `fusion.completed`, `case.ready`,
`feedback.recorded`).

---

## 14. Deployment architecture

**Hackathon / dev** — single `docker-compose up`:
gateway, all engines, web, Postgres, Redis, MinIO, Qdrant, MLflow. Vision on CPU (ONNX) or
local GPU. Seeded demo dataset (de-identified public CXR, e.g. NIH ChestX-ray14 samples).

**Production (per hospital, on-prem first)**
```
Hospital network (no PHI egress)
┌─────────────────────────────────────────────────────┐
│  k3s / K8s cluster                                   │
│   • gateway ×2  • web  • engines (CPU pool)          │
│   • vision (GPU node, T4/L4)  • fusion (CPU)         │
│   • Postgres (HA) • Redis • MinIO • Qdrant           │
│   DICOM gateway VM ↔ PACS VLAN                       │
└─────────────────────────────────────────────────────┘
Cloud control plane (no PHI): model registry mirror, license,
telemetry (aggregated metrics only), fleet upgrade channel.
```
- GitOps (ArgoCD) upgrades; canary per model version via `models` service.
- Cloud-hosted SaaS variant (AWS HIPAA-eligible stack) for research/teaching customers.
- Everything Terraform'd; images signed (cosign); SBOM published per release.

---

## 15. Security & compliance

- **PHI boundary at ingestion**: DICOM de-identification (HIPAA Safe Harbor tags) before
  anything touches ML services; re-identification map encrypted (AES-256-GCM, KMS/Vault),
  accessible only to gateway under RBAC.
- **Encryption**: TLS 1.3 everywhere (mTLS service-to-service); at-rest encryption for
  Postgres, MinIO, backups; column-level encryption for demographics.
- **AuthN/Z**: OIDC (hospital IdP / SMART-on-FHIR), short-lived JWTs, org- and role-scoped
  permissions, break-glass audit alerts.
- **Auditability**: append-only `audit_log`; every prediction stores model version, input
  hash, calibration snapshot — full reproducibility of any historical output.
- **LLM safety**: report LLM receives only de-identified structured findings; grounding
  validator blocks unsourced claims; no PHI in any third-party API call.
- **Supply chain**: pinned deps, image scanning, signed releases.
- **Regulatory posture**: architected for FDA SaMD Class II (predicate: CADe/CADx triage
  devices) and EU AI Act "high-risk" documentation duties — the epistemic audit trail *is*
  the technical documentation. Positioning at launch: decision support / research use,
  clinician always in the loop, no autonomous diagnosis.

---

## 16. Future roadmap

| Phase | Timeline | Milestones |
|---|---|---|
| **0 — Hackathon** | now → demo | P0 features end-to-end on CXR; quantum-vs-classical fusion benchmark; live dashboard demo |
| **1 — Pilot** | 3–6 mo | Prior comparison + longitudinal memory; FHIR export; on-prem installer; 2 academic pilot sites (research agreements); calibration drift dashboard |
| **2 — Product** | 6–18 mo | CT/MR modalities; multimodal evidence (labs, notes); follow-up closure tracking; SOC 2 Type II; FDA pre-submission (Q-Sub) |
| **3 — Scale** | 18–36 mo | Federated learning across sites; quantum hardware backends for fusion research; teaching mode for residency programs; marketplace of specialty evidence models |

Key risks & mitigations: *quantum adds no measurable lift* → classical fallback is
first-class, quantum framed as research benchmark until proven; *regulatory drag* → launch
as research/decision-support, revenue from workflow (reporting, triage) not diagnosis;
*data access* → public datasets + academic partnerships first, feedback moat later.

---

## 17. Approval gate

Implementation begins only after sign-off on:
1. This architecture (sections 7–13 especially — service contracts and schema).
2. P0 scope (section 6) as the hackathon cut.
3. Naming (`AURA`) and monorepo location (`aura/` in this repo, or a fresh repo).

*— The AURA founding team*
