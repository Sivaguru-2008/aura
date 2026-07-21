# AURA — Complete Reverse-Engineering & Audit Report

**Method:** every claim below is grounded in a file read, a line reference, or a command executed against this repository on 2026-07-20. Where I ran code, the output is quoted. Where something is absent, it says **Not implemented**. Where I am inferring rather than observing, it says **Assumption**.

**Repository:** `E:\AURA\aura-main` · git `cf8356b` (working tree dirty: 13 modified, 10 untracked) · remote `github.com/Sivaguru-2008/aura`

---

## 0. Executive summary — the ten things that matter

| # | Finding | Severity | Evidence |
|---|---|---|---|
| F1 | **The fusion layer is trained on a distribution that never occurs in production.** Fusion training builds its dataset with a bare `VisionEngine()` (heuristic fallback) over synthetic 64×64 images; serving uses `VisionEngine.load()` (DenseNet-121) over real films. Measured on one image: evidence vector `[0.63, 0.66, 0.48, 0.80, 0.45, 0.00, 0.26, 0.62]` at training vs `[0.80, 0.68, 0.87, 0.54, 0.46, 0.87, 0.78, 0.62]` at serving. | **Critical** | `ml/training/dataset.py:18`, measured |
| F2 | **The Wasserstein conflict guard has zero effect on the diagnosis shown to the clinician.** It writes telemetry into `FusionResult`, then `SafetyEngine` recomputes the posterior from the quantum model and ignores it. Measured: guard chose *normal 0.446*, report said *malignancy 55.1%*. | **Critical** | `gateway/pipeline.py:75`, `services/safety/engine.py:79`, measured |
| F3 | **Pneumothorax sensitivity is 0.14 and nodule AUROC is 0.44 (below chance)** on honest per-study labels — the two most dangerous findings. Pneumothorax carries urgency 1.0 in the worklist ranker. | **Critical** | `artifacts/evaluation_perstudy/EVALUATION_SUMMARY.md` |
| F4 | **The served checkpoint is the weaker of two trained models.** `best_model.pt` = epoch 1, macro-AUROC 0.6962. `retrain_v2/best_model.pt` = 0.7859, never promoted. | High | `torch.load` metadata |
| F5 | **Uploaded radiographs are destroyed to 64×64 before the CNN sees them**, then upsampled to 224. The docstring claims the opposite. | High | `services/vision/io.py:79–88` |
| F6 | **The README's headline quantum-vs-classical table is an artifact of unequal temperature scaling.** The benchmark gives quantum a fitted temperature and classical `T=1.0`. | High | `ml/evaluation/benchmark.py:76–78` |
| F7 | **Mondrian conformal thresholds are degenerate.** The quantile level saturates at 1.0 for n ≤ 19, so q̂ becomes the *maximum* nonconformity score. Malignancy q̂ = 0.9889 → malignancy appears in ~78 % of confidence sets. | High | `services/safety/uncertainty.py:148`, measured |
| F8 | **The OOD detector was deliberately widened until it stopped firing on real films.** `ood_mean` moved −5.76 → −2.77; real films now score z ∈ [−0.43, +1.19] against a 1.5 threshold. | High | `artifacts/safety*.npz`, `ml/training/recalibrate_ood.py` |
| F9 | **Adaptive Conformal Inference updates a threshold nothing reads.** State is written to SQLite on feedback; `SafetyEngine` never loads it. | Medium | `gateway/storage.py:193`, grep |
| F10 | **The clinical reasoning layer cannot change the diagnosis.** Labs/symptoms produce `adjusted_posterior`, but the impression is built from `safety.top` (imaging only). Impression and differential can contradict each other. | Medium | `services/report/engine.py:91` |

**One-sentence verdict:** AURA is an unusually well-architected, well-documented, genuinely modular clinical-AI *scaffold* with a real CNN, a real quantum circuit, real conformal machinery, and a real reproducible audit harness — but its safety-critical wiring is broken in three independent places, its centerpiece quantum claim does not survive a fair test (the project's own audit says so), and its fusion brain is trained on a world that does not exist.

---

# PHASE 1 — Project understanding

## 1.1 What problem is this solving?

Chest radiography is the highest-volume imaging study in medicine and the interpretation bottleneck is radiologist time. The standard AI response is a classifier that emits a label. AURA's stated thesis (`README.md:5–9`) is that a label is the wrong product:

> "Most medical AI gives you a diagnosis. AURA's product is **calibrated doubt** — how sure it is, what evidence is missing, which test to run next, and the license to say *'I don't know'* instead of failing silently."

So the problem statement is: **an automated chest-X-ray reader that quantifies and communicates its own unreliability, abstains when it should, and tells you what evidence would resolve the ambiguity** — running entirely offline so no PHI leaves the machine.

## 1.2 Objective

Concretely, the system aims to produce, per study:
1. Seven imaging **findings** with probabilities (observations).
2. A calibrated posterior over six **diagnoses** (interpretations).
3. A **conformal prediction set** with a distribution-level coverage guarantee.
4. An **abstention decision** with a named reason.
5. **Saliency** maps and per-evidence attribution.
6. A ranked **next-test recommendation** by expected value of information.
7. A **grounded report** where each sentence traces to the evidence that produced it.

## 1.3 Intended users

Inferred from the interface and the data model (`schemas/contracts.py`, `apps/web/`):

- **Primary:** a radiologist or emergency clinician working a triage worklist. `CaseBundle.priority_score` and `CaseState` (`queued → analyzing → ready → in_review → signed | abstained`) describe a reading-queue workflow, and the console exposes accept/edit/reject verdicts plus a "sign report" action.
- **Secondary:** an ML/clinical-safety reviewer. `/v1/admin/safety` returns the model registry, benchmark, feedback stats, abstention rate, and audit tail.
- **Actual, today:** a hackathon/competition judge. The landing page, cinematic console transition, and `presentation/` deck set are built for demonstration. This is not a criticism — it is the honest reading of where the effort went.

## 1.4 The real-world workflow implemented

```
Clinician opens http://localhost:8000
  → worklist pre-seeded at startup (12 real MIMIC-CXR patients, or synthetic fallback)
  → clicks a case → sees findings, posterior, conformal set, saliency, differential, report
  → or uploads a new film (PNG/JPG/DICOM) → intake gate → full pipeline → new case
  → accepts / edits / rejects → feedback + ACI update + audit row
  → signs the report → state = SIGNED
```

## 1.5 Inputs accepted

| Input | Where | Notes |
|---|---|---|
| PNG / JPG / TIFF | `POST /v1/studies/upload` | via `services/vision/io.load_image` |
| DICOM | same | VOI-LUT applied, MONOCHROME1 inverted (`io.load_dicom`) |
| Synthetic study | `POST /v1/studies/simulate` | generated by `ml/data.make_sample` |
| Structured priors | `StructuredPriors` | age band, sex, smoker, fever, prior cancer, immunocompromised |
| Labs / symptoms / history | `MultimodalContext` | **only populated for synthetic cases** — see §7.6 |

## 1.6 Outputs generated

`CaseBundle` (`schemas/contracts.py:289`) carries: `vision` (7 finding scores + embedding), `evidence` (8 nodes), `fusion` (posterior + std + conflict telemetry), `safety` (calibrated predictions with CIs, conformal set, epistemic/aleatoric split, OOD energy, abstention), `explanation` (5 saliency maps + attribution + counterfactuals), `reasoning` (steps, log-LRs, differential, guideline citations), `recommendations`, `report` (5 text sections + grounding map).

The CLI additionally writes: `clinical_report.{md,json,html}`, `overlay.png`, `overlay_hires.png`, `heatmap.png`, `evidence.png`, `method_*.png` per attribution method, and `explanation.html`.

## 1.7 Capabilities — what it genuinely does

- Loads and preprocesses real DICOM and JPEG radiographs offline.
- Runs a real fine-tuned DenseNet-121 (7-label multi-label head) on GPU or CPU.
- Runs a real 8-qubit PennyLane variational circuit for evidence fusion.
- Fits and applies temperature scaling, split-conformal and Mondrian conformal thresholds.
- Computes Grad-CAM, Grad-CAM++, Integrated Gradients, SmoothGrad, Score-CAM, and occlusion saliency.
- Rejects non-radiographs at intake with a named reason (`xray_gate.validate_cxr`).
- Persists cases, feedback, outcomes, and an append-only audit log in SQLite.
- Extracts CheXpert-style structured labels from free-text reports with negation scoping.
- Ships a reproducible audit harness (`audit_all.py`) with DeLong, McNemar, bootstrap and permutation tests that **falsifies the project's own headline claim**.

## 1.8 Limitations — what it does not do

- **Not implemented:** authentication, authorization, encryption, PHI handling, multi-user isolation, rate limiting.
- **Not implemented:** any horizontal scaling. In-process singleton pipeline, in-memory memory index rebuilt each boot, SQLite.
- **Not implemented:** quantum services Q2–Q6 described in `docs/QUANTUM_STACK.md`.
- **Not implemented:** lateral-view fusion, prior-study comparison (`MemoryEngine.prior_delta` exists but no caller), segmentation, localization ground truth.
- The six-diagnosis vocabulary is closed and small; anything outside it is silently mapped or missed.
- Labs/symptoms/history exist in the schema and drive the reasoning engine, but MIMIC-CXR carries none of them (`mimic/patient.py:multimodal_context` returns `None` by design).

## 1.9 Assumptions the project makes

1. **That the evidence vector is modality-invariant.** It is not — F1.
2. **That 8 hand-chosen channels are a sufficient statistic** for the diagnosis. Everything the CNN saw is compressed to 8 scalars before fusion; the 1024-d embedding is used only for similarity.
3. **That findings map cleanly to diagnoses.** The mapping is a learned 6×8 linear head; there is no anatomical or causal model.
4. **That report text is ground truth.** Labels come from a regex labeler over radiology reports (`mimic/labeling.py`) — a proxy for a proxy.
5. **That study-order equals chronology.** `mimic/timeline.py` states this explicitly: the aug CSV has no timestamps, so `study_id` order is the time axis.
6. **That exchangeability holds** for the conformal guarantee — which ACI was built to relax, but ACI is not wired in (F9).

---

# PHASE 2 — Complete architecture

## 2.1 The actual runtime topology

```
┌─────────────────────────────────────────────────────────────────────┐
│ BROWSER  — zero-dependency SPA, no framework, no build step         │
│ index.html (landing + console)   history.html (report portal)       │
│ js/fx.js (api() + canvas fx) · landing.js · console.js · main.js    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ fetch() → /v1/*
┌───────────────────────────────▼─────────────────────────────────────┐
│ FASTAPI GATEWAY  — gateway/app.py                                   │
│   lifespan: ensure_dirs → Store → Pipeline → ModelRegistry → seed   │
│   middleware: audit_mw (Cache-Control + audit row on POST/PUT/DEL)  │
│   ⚠ NO AUTHENTICATION — x-aura-user header is trusted verbatim      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ INTAKE                                                              │
│   upload → xray_gate.validate_cxr()   [hard gates + structure score]│
│          → vision/io.study_from_cxr() [⚠ downsamples to 64×64 — F5] │
│   simulate → ml/data.make_sample()    [synthetic generator]         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ StudyInput
┌───────────────────────────────▼─────────────────────────────────────┐
│ PIPELINE  — gateway/pipeline.py :: Pipeline.run()                   │
│  (holds ONE instance of each engine; process-wide singleton)        │
└───────────────────────────────┬─────────────────────────────────────┘
   │
   ├─1─► VisionEngine.analyze()          services/vision/engine.py
   │      └─ VisionModel (DenseNet-121)  ml/vision_cxr/inference.py
   │         └─ 64×64 → resize 224 → 7 sigmoids + 1024-d embedding
   │
   ├─2─► evidence.encode()               services/fusion/evidence.py
   │      └─ 7 findings + prior_risk → x ∈ [0,1]^8
   │
   ├─3─► FusionEngine.fuse_vector()      services/fusion/engine.py
   │      ├─ QuantumFusion (8q VQC)      services/fusion/quantum.py
   │      ├─ WassersteinTieBreaker       services/fusion/conflict.py
   │      └─ ⚠ guard result is recorded but DISCARDED downstream — F2
   │
   ├─4─► SafetyEngine.assess()           services/safety/engine.py
   │      ├─ ⚠ recomputes logits from the QUANTUM model, ignoring the guard
   │      ├─ temperature scaling         services/safety/calibration.py
   │      ├─ DeepEnsemble (5 classical heads) → epistemic MI
   │      ├─ Mondrian conformal set      services/safety/uncertainty.py  ⚠ F7
   │      ├─ energy-score OOD            common/mathx.energy_score       ⚠ F8
   │      └─ abstention policy (4 reasons)
   │
   ├─5─► ExplainEngine.explain()         services/explain/engine.py
   │      └─ Grad-CAM / ++ / IG / SmoothGrad + leave-one-out attribution
   │
   ├─6─► RecommendEngine.recommend()     services/recommend/engine.py
   │      └─ EVOI + greedy panel + causal redundancy masking
   │
   ├─7─► ClinicalReasoner.reason()       services/reasoning/engine.py
   │      └─ 8 guideline log-LR rules → adjusted_posterior  ⚠ never used — F10
   │
   ├─8─► ReportEngine.compose()          services/report/engine.py
   │      └─ impression from safety.top; differential from reasoning
   │
   └─9─► MemoryEngine.index()            services/memory/engine.py
          └─ in-process list, cosine similarity, lost on restart
                                │ CaseBundle
┌───────────────────────────────▼─────────────────────────────────────┐
│ PERSISTENCE  — gateway/storage.py (SQLAlchemy → SQLite)             │
│   cases · feedback · conformal_state · outcomes · audit_log         │
└─────────────────────────────────────────────────────────────────────┘
```

## 2.2 Layer-by-layer

### Frontend
- **Purpose:** landing narrative + clinical console + history/report portal.
- **Implementation:** hand-written HTML/CSS/JS, no framework, no bundler. Canvas-based particle/warp effects.
- **Files:** `apps/web/index.html`, `history.html`, `css/main.css` (33 KB), `css/history.css` (19 KB), `js/{fx,landing,console,history,main}.js` (~87 KB total).
- **Dependencies:** none. Served as static files.
- **Data flow:** `fx.js:272` defines the single `api()` helper wrapping `fetch`; every other module calls it.

### API gateway
- **Purpose:** HTTP surface, static hosting, audit middleware, lifespan wiring.
- **Files:** `gateway/app.py`.
- **Algorithms:** none — pure orchestration.
- **Note:** cache headers are forced to `no-cache` for `/`, `/app`, `/static` (`app.py:72`) because stale JS previously left buttons unbound.

### Authentication
- **Not implemented.** `app.py:3–5` states auth is "stubbed for the P0 demo (a header-based principal)". The only use of identity is `request.headers.get("x-aura-user", "anonymous")` written into the audit log. Any client can set it to anything. Upload, feedback, and sign endpoints are all unauthenticated.

### Upload pipeline & image validation
- **Purpose:** stop non-radiographs before a case exists.
- **File:** `services/vision/xray_gate.py`.
- **Algorithm:** layered cheapest-first. Hard gates: decodability, DICOM modality ∈ {CR,DX,RG,XA}, body part contains CHEST/THORAX, aspect ratio ∈ [0.4, 2.5], mean chroma ≤ 0.08, colored-pixel fraction ≤ 0.10, grayscale σ ≥ 0.04, 256-bin histogram entropy ≥ 4.0 bits. Chest-structure gates: central-third / lateral-thirds brightness ratio ≥ 1.05, column-profile CV ≥ 0.09. Then a soft score requiring 2 of 3: edge density ≤ 0.10, midtone mass ≥ 0.60, ≥ 96 occupied gray levels.
- **Honest self-assessment in the file** (`xray_gate.py:32`): "A statistically radiograph-like impostor can still slip through heuristics."

### Vision model
- **Purpose:** image → 7 finding probabilities + embedding.
- **Files:** `ml/vision_cxr/model.py` (architecture), `inference.py` (serving wrapper), `services/vision/engine.py` (contract), `services/vision/cnn.py` (alternative backbones), `services/vision/features.py` (numpy fallback).
- **Algorithm:** DenseNet-121, ImageNet-pretrained, `conv0` collapsed 3→1 channel by BT.601 luminance weights, classifier replaced with `Linear(1024, 7)`.
- **Selection logic** (`engine.py:57–78`): if `artifacts/best_model.pt` exists → `VisionModel`; else `_maybe_backbone()` per config; else numpy feature model; else hard-coded heuristic.

### Quantum fusion
- **Purpose:** 8-channel evidence → 6-class posterior.
- **Files:** `services/fusion/{device,quantum,classical,learnable,ensemble,conflict,engine,evidence}.py`.
- **Algorithm:** see Phase 6.

### Safety / calibration / uncertainty
- **Files:** `services/safety/{engine,calibration,uncertainty,aci}.py`.
- **Algorithms:** temperature scaling (bounded 1-D NLL minimization), split-conformal, Mondrian conformal, deep-ensemble MI decomposition, free-energy OOD score, 4-reason abstention.

### Explainability
- **Files:** `services/explain/{engine,methods,scorecam,overlays}.py`.
- **Algorithms:** Grad-CAM, Grad-CAM++, Integrated Gradients (32 steps, black baseline), SmoothGrad (25 samples, σ=0.15), Score-CAM, occlusion (12-px window, 6-px stride), leave-one-out evidence attribution.

### Recommendation engine
- **File:** `services/recommend/engine.py`, `causal.py`.
- **Algorithm:** severity-weighted Bayes risk; EVOI = R(p₀) − E[R(p′)] over 2^k resolved outcomes; greedy forward panel selection under a cost·risk budget; chained-MI redundancy masking over a hand-authored causal graph.

### Report generation
- **Files:** `services/report/engine.py` (compact `ReportDraft`), `clinical_report.py` (full document, 444 lines).
- **Algorithm:** deterministic templating. **No LLM anywhere in this repository.** This is a strength for hallucination risk.

### Database & history
- **File:** `gateway/storage.py`. Five tables; case bundles stored as JSON documents with indexed columns for worklist queries.

---

# PHASE 3 — Complete file audit

Legend — **Prod** = on the request-serving path · **Exp** = experimental/research · **Obs** = obsolete · **Unused** = no importer · **Rm?** = safe to remove.

## 3.1 Core package (`aura/`)

| File | Purpose | Interacts with | Class | Rm? | Importance |
|---|---|---|---|---|---|
| `aura_cli.py` | Single entry point: train/bench/serve/demo/predict/evaluate/explain/benchmark/calibrate | everything | Prod | No | **Critical** |
| `pyproject.toml` | `[tool.aura]` config + pytest config | `common/config.py` | Prod | No | High |
| `requirements.txt` | Pinned P0 deps; CNN stack commented out as optional | — | Prod | No | High |
| `run.bat` / `run.sh` | One-shot launchers (install → train → bench → serve) | CLI | Prod | No | Medium |
| `sample.jpg` | 73 KB demo radiograph | CLI predict | Prod | No | Low |
| `_final_test_result.txt` | **Empty file, 0 bytes** | none | Obs | **Yes** | None |

## 3.2 `common/`

| File | Purpose | Class | Notes |
|---|---|---|---|
| `config.py` | `Settings` dataclass; pyproject → env override chain; `ROOT/ARTIFACTS/DATA/DB_PATH` | Prod | 20 settings. `@lru_cache` on `get_settings()` means env changes need a restart. |
| `eventbus.py` | In-process async pub/sub | Prod | **No subscribers exist.** `Pipeline` publishes 4 topics; nothing subscribes. Pure instrumentation seam today. |
| `mathx.py` | `softmax`, `entropy` (bits), `sigmoid`, `normalize`, `energy_score` | Prod | 38 lines, correct, no issues found. |

## 3.3 `schemas/`

| File | Purpose | Class | Notes |
|---|---|---|---|
| `clinical.py` | `Modality`, `Finding` (7), `Diagnosis` (6), display labels | Prod | The closed vocabulary the whole system rests on. |
| `contracts.py` | 20 Pydantic models = every inter-service interface | Prod | Excellent. Documented flow diagram in the module docstring. Backward-compatible defaults on the Module-5 conflict fields. |

## 3.4 `gateway/`

| File | Purpose | Class | Notes |
|---|---|---|---|
| `app.py` | 11 endpoints + middleware + lifespan | Prod | No auth. `/v1/studies/upload` catches broad `Exception` → 500 with raw message (info leak). |
| `pipeline.py` | 9-stage orchestration | Prod | **Site of F2** (line 75). |
| `storage.py` | 5 tables, repository API | Prod | Site of F9. |
| `seed.py` | Synthetic worklist seeder | Prod (fallback) | Used when `AURA_DATA_SOURCE=synthetic` or MIMIC absent. |

## 3.5 `services/fusion/`

| File | Purpose | Class | Rm? | Notes |
|---|---|---|---|---|
| `engine.py` | Backend resolution + conflict guard | Prod | No | |
| `quantum.py` | VQC serving, 102 params | Prod | No | |
| `classical.py` | Product-of-experts, 54 params | Prod | No | The honest baseline. |
| `ensemble.py` | 5 bootstrap heads | Prod | No | Powers epistemic uncertainty. |
| `learnable.py` | Attention-gated log-linear + torch trainer | Prod (selectable) | No | **Beats the VQC** on accuracy/NLL/Brier/AUROC in the audit. |
| `conflict.py` | Wasserstein tie-breaker | Prod (neutered) | No | Correct code, **ignored downstream**. |
| `evidence.py` | 7 findings + prior_risk → x ∈ [0,1]^8 | Prod | No | The information bottleneck of the whole system. |
| `device.py` | `make_qnode` (used) + `make_reuploading_qnode` (**unused**) | Mixed | Partial | Re-uploading ansatz has zero importers. |
| `projection.py` | `JointProjection` 1024→8 bottleneck | **Unused** | **Yes** | **Zero importers.** `docs/ARCHITECTURE_REFACTOR.md:17` claims it is "wired in `services/fusion/engine.py`" — it is not. |

## 3.6 `services/` (remaining)

| File | Purpose | Class | Notes |
|---|---|---|---|
| `vision/engine.py` | Vision contract + backbone selection | Prod | |
| `vision/cnn.py` | `CXRBackbone` (torchxrayvision / timm) | **Unused in practice** | `best_model.pt` exists → `VisionModel` always wins at `engine.py:62`. The strong `densenet121-res224-mimic_ch` weights are never loaded. |
| `vision/features.py` | 10 anatomical numpy features | Prod (fallback) | Also used by `mimic/features.py`. |
| `vision/io.py` | DICOM/PNG intake | Prod | **Site of F5.** |
| `vision/xray_gate.py` | Intake gate | Prod | Well-built. |
| `safety/engine.py` | Assessment | Prod | **Site of F2.** |
| `safety/calibration.py` | T, q̂, OOD stats | Prod | |
| `safety/uncertainty.py` | Ensemble MI, Brier, reliability, Mondrian | Prod | **Site of F7** (line 148). |
| `safety/aci.py` | Adaptive conformal inference | **Written, not wired** | Mathematically correct and beautifully documented. No serving reader. |
| `explain/engine.py` | Orchestration | Prod | |
| `explain/methods.py` | 4 gradient methods + occlusion | Prod | |
| `explain/scorecam.py` | Gradient-free CAM | Prod (CLI only) | Off by default in serving for latency. |
| `explain/overlays.py` | PNG/HTML rendering, bbox extraction | Prod (CLI) | Requires matplotlib. |
| `recommend/engine.py` | EVOI + panel | Prod | **Thread-unsafe** (`self._panel_evoi`). |
| `recommend/causal.py` | Chained-MI redundancy | Prod | `LAB_MARKERS` (troponin, bnp, d_dimer, wbc, crp) are in the graph but **never in `EVIDENCE_CHANNELS`**, so those edges can never fire from the current catalog. |
| `reasoning/engine.py` | 8 guideline rules | Prod (decorative) | **Site of F10.** |
| `report/engine.py` | `ReportDraft` | Prod | |
| `report/clinical_report.py` | Full document + md/json/html renderers | Prod (CLI) | |
| `memory/engine.py` | Cosine similarity | Prod | In-memory only; `prior_delta()` has **no caller**. |
| `models/registry.py` | Reads `registry.json` | Prod | Minimal. |

## 3.7 `ml/`

| File | Purpose | Class | Notes |
|---|---|---|---|
| `data.py` | Synthetic CXR world | Prod (training + simulate) | 218 lines. Well-designed generator. |
| `training/dataset.py` | Evidence dataset builder | Prod | **Site of F1** (line 18). |
| `training/train_fusion.py` | Trains 4 fusion backends + calibration | Prod | **Site of F6 root cause** (line 152–181): one temperature fit on the configured backend, applied to both in the metric table. |
| `training/train_vision.py` | Fits `vision.npz` logistic detectors | Prod | Output is loaded at serving but **overridden by the CNN** for 6 of 7 findings. |
| `training/train_cnn.py` | timm fine-tuning harness | Exp | Produced `vision_cnn.zip`; not the served path. |
| `training/cxr_dataset.py` | Manifest + synthetic datasets for `train_cnn` | Exp | |
| `training/prepare_mimic_manifest.py` | Builds a manifest CSV | Exp | |
| `training/recalibrate_ood.py` | Refits OOD stats on real+synthetic | Prod (run once) | **Site of F8.** |
| `vision_cxr/model.py` | `DenseNet121CXR` + luminance conv0 | Prod | The luminance-weighted conv0 init is a genuinely good, well-reasoned piece of work. |
| `vision_cxr/train.py` | The training loop that produced the served model | Prod | AMP, cosine LR, early stopping, resume, TB, post-training E2E validation. |
| `vision_cxr/dataset.py` | MIMIC loading + per-study labels | Prod | Contains the F3 fix (`per_study=True`). |
| `vision_cxr/{config,losses,metrics,checkpoint,utils,validate}.py` | Support | Prod | `losses.py` TV regulariser is well-documented incl. its own caveat. |
| `vision_cxr/inference.py` | `VisionModel` serving wrapper | Prod | |
| `evaluation/clinical_eval.py` | Full validation battery + bootstrap CIs | Prod | Excellent. |
| `evaluation/vision_calibration.py` | Temperature/MC-dropout/conformal for vision | Prod | |
| `evaluation/benchmark.py` | Quantum-vs-classical | Prod | **Site of F6.** |
| `evaluation/metrics.py` | Per-class + macro clinical metrics | Prod | |
| `evaluation/perf_benchmark.py` | Latency/throughput/memory | Prod | |

## 3.8 `mimic/` (16 files)

All are real, working, well-tested code. The subset that runs today: `config`, `parsing`, `loaders`, `cleaning`, `labeling`, `timeline`, `patient`, `seed`, `verify`.

The subset that **has no data**: `features`, `splits`, `tasks`, `training`, `evaluation`, `explain`, `performance`, `uncertainty` depend on the MIMIC-IV tabular tables. Verified: `datasets/mimiciv/{ed,hosp,icu,notes}` are all **empty directories**. These modules are honest about it — `features.py:16` emits `*_missing = 1.0` indicators rather than fabricating values, and `tasks.py` registers unavailable tasks with `available=False` and a reason. That is exemplary engineering discipline.

## 3.9 Root-level & non-code

| Path | Purpose | Class | Rm? |
|---|---|---|---|
| `audit_all.py` (31 KB) | Reproducible audit harness: DeLong, McNemar, bootstrap, permutation, Wilcoxon, Shapiro | Exp (high value) | No |
| `_probe_audit.py` | Scratch probe used while building `audit_all.py` | Obs | **Yes** |
| `test_image.py` | Ad-hoc CLI predictor; sets `AURA_VISION_BACKEND=timm` which contradicts the served path | Obs | **Yes** — superseded by `aura_cli predict` |
| `scientific_audit.md`, `audit_pipeline_fix_report.md`, `vision_audit.md`, `PROJECT_STATUS.md` | Prior audit reports | Doc | No |
| `presentation/` (~20 MB) | 7 pptx decks + 12 build/patch scripts + unpacked OOXML + a full OOXML schema tree | Build asset | Consider extracting to a separate repo |
| `media/` (~3 MB) | 5 screenshots | Doc | No |
| `aura/artifacts/` (~500 MB) | Checkpoints, optimizer states, plots, DB | Artifact | See §3.10 |

## 3.10 Artifact bloat

`last_checkpoint.pt` (84 MB) + `optimizer.pt` (56 MB) exist in **three** copies (`artifacts/`, `retrain_v2/`, `_smoke/`) = ~420 MB. `vision_cnn.zip` is a further 77 MB. The git pack is 171 MB with a 24 MB temp pack. `.gitignore` exists (425 bytes) but these are clearly tracked or were.

**Recommendation:** keep `best_model.pt` (27 MB) only; move the rest to release assets or Git LFS.

---

# PHASE 4 — Directory walkthrough

```
E:\AURA\
├── datasets/
│   ├── mimiciv/{ed,hosp,icu,notes}/        ← ALL EMPTY. Blocks aura/mimic tabular ML.
│   └── simhadrisadaram/mimic-cxr-dataset/versions/2/
│       ├── mimic_cxr_aug_train.csv         236 MB · 64,586 subjects
│       ├── mimic_cxr_aug_validate.csv      1.9 MB · 500 subjects
│       ├── official_data_iccv_final/files/ 261,137 JPEGs on disk
│       └── _aura_cache/                    parquet cache (mimic/loaders.py)
├── venv/                                   global venv
└── aura-main/                              THE REPOSITORY
    ├── .venv/                              second, project-local venv (py3.14)
    ├── aura/                               ← the Python package root; run everything from here
    │   ├── apps/web/                        zero-dependency SPA (why: no build step, offline)
    │   ├── artifacts/                       trained weights, calibration, plots, aura.db
    │   ├── common/                          config, event bus, math (why: no engine imports another)
    │   ├── data/                            (empty; created by ensure_dirs)
    │   ├── docs/                            11 markdown docs, 190 KB
    │   ├── gateway/                         API + orchestration + persistence
    │   ├── mimic/                           real-corpus loading, cleaning, labeling, tabular ML
    │   ├── ml/                              data generation, training, evaluation
    │   │   ├── evaluation/                  metric batteries + benchmarks
    │   │   ├── training/                    fusion + vision training entry points
    │   │   └── vision_cxr/                  the DenseNet training package (the real one)
    │   ├── schemas/                         Pydantic contracts — the single source of truth
    │   ├── services/                        the 9 engines, each independently replaceable
    │   │   ├── explain/ fusion/ inference/ memory/ models/
    │   │   └── reasoning/ recommend/ report/ safety/ vision/
    │   └── tests/                           18 files, 117 tests
    ├── audit_artifacts/run_20260719T175647Z/  seeded reproducible audit output
    ├── media/                               screenshots
    └── presentation/                        pitch decks + generators
```

**Why each exists:**

- **`services/` split by engine** — `services/__init__.py` states the rule: "No engine imports another; they compose only through the gateway pipeline and the event bus." I verified this holds, with one exception: `services/explain/engine.py:10–11` imports `services.vision.features._resize_to` and `ml.data.IMG`. That is a private cross-service import and a layering violation.
- **`schemas/` separate** — so a contract change is the only way to change an interface.
- **`common/` separate** — so config/math have no service dependencies.
- **`ml/` separate from `services/`** — `ml/__init__.py`: "Never imported by request-serving code paths beyond loading trained artifacts." **This rule is violated:** `services/vision/features.py:11` imports `IMG, REGIONS, _px` from `ml.data`, and `services/explain/engine.py:11` imports `ml.data.IMG`. The synthetic data module is therefore on the serving path.
- **`mimic/` separate from `ml/`** — real-corpus code kept apart from the synthetic world.
- **`aura/` nested inside `aura-main/`** — an artifact of the GitHub zip download. You must `cd aura/` before running anything, because `pyproject.toml` sets `pythonpath = ["."]` relative to that directory.

---

# PHASE 5 — AI & ML analysis

## 5.1 Models present

| # | Model | Type | Params | Trained on | Served? |
|---|---|---|---|---|---|
| 1 | `DenseNet121CXR` | CNN, 7-label multi-label | ~7.0 M | MIMIC-CXR, 259,038 images | **Yes** |
| 2 | `QuantumFusion` | 8-qubit VQC + linear head | 102 | Synthetic evidence, 420 samples | **Yes** (default) |
| 3 | `ClassicalFusion` | Product-of-experts | 54 | same | Fallback + guard |
| 4 | `DeepEnsemble` | 5 × PoE, bootstrapped | 270 | same | Yes (uncertainty only) |
| 5 | `LearnableFusion` | Attention-gated log-linear | 118 | same | Selectable |
| 6 | Vision logistic detectors | 7 × logistic on 10 feats | 77 | Synthetic images | Yes (hyperinflation only) |
| 7 | `diag_gbm.joblib` | Gradient boosting | — | MIMIC image features, n=338 | No |
| 8 | `diag_mlp.pt` / `_r.pt` | Small MLP | — | same | No |

## 5.2 Why these models

- **DenseNet-121** is the CheXNet lineage standard for CXR; dense connectivity gives strong gradient flow and a 1024-d pooled embedding at 7×7 spatial resolution, which is a good Grad-CAM target. Reasonable, conventional, defensible.
- **VQC for fusion, not for imaging** — this is the single best architectural judgement in the project. `services/fusion/__init__.py` states it plainly: "This is where quantum earns its place — small, structured, correlation-rich reasoning, not image processing." Angle-encoding 8 clinically meaningful channels into 8 qubits is tractable and honest; angle-encoding pixels would not be.
- **Product-of-experts as the twin** — deliberately chosen because it *cannot* represent higher-order interactions (`classical.py:7`), which is exactly the null hypothesis the VQC should beat.

## 5.3 Architecture — `DenseNet121CXR`

```
input (B,1,224,224)
  → conv0: Conv2d(1,64,7,2,3)   ← luminance-initialised from pretrained RGB filters
  → DenseNet-121 features       → (B,1024,7,7)
  → ReLU → AdaptiveAvgPool2d(1) → flatten → (B,1024)
  → Linear(1024,7)              → logits
```

The `luminance_init_conv0` function (`model.py:10–54`) is the most thoughtful ML code in the repository. Instead of the usual `weight.sum(dim=1)` when collapsing 3→1 channel, it applies BT.601 luma weights (0.299, 0.587, 0.114):

```python
weighted = a * w[:, 0:1] + b * w[:, 1:2] + c * w[:, 2:3]
```

The docstring explains why: naive summation "inflates the filter gain by treating three correlated channels as independent, shifting the activation statistics away from what BatchNorm downstream was calibrated to." Correct, and the reasoning is sound.

## 5.4 Loss functions

**Vision:** `RegularizedMultiLabelLoss` = `BCEWithLogitsLoss(pos_weight)` + λ·TV(features), λ = 1e-4.

`pos_weight` is computed per class as `clip(neg/pos, 0.1, 20.0)` (`train.py:165–168`). With pneumothorax prevalence 0.0325, that gives pos_weight ≈ 20 (clipped) — a 20× penalty on missed positives.

The TV term penalises high-frequency noise in the (B,1024,7,7) latent map:

TV(F) = Σ |F[i+1,j] − F[i,j]| + |F[i,j+1] − F[i,j]|, averaged.

**Critical observation:** the feature map is **7×7**. Total variation over a 7×7 grid is a very weak spatial prior — there are only 42 horizontal + 42 vertical differences per channel. The documented motivation ("Grad-CAM++ locks onto anatomy") is plausible in principle, but at this resolution the effect is near-negligible. The file's own caveat is honest ("too large a weight … can erase genuinely small, high-frequency lesions"), but the effect size at 7×7 is not discussed.

**Fusion:** `CrossEntropyLoss` for the VQC and the learnable head; hand-rolled softmax-regression gradient for the PoE (`train_fusion.py:37–50`).

## 5.5 Optimizers & schedulers

| Model | Optimizer | LR | Schedule | Epochs |
|---|---|---|---|---|
| DenseNet | AdamW, wd=1e-4 | 3e-4 | CosineAnnealingLR(T_max=epochs) | 15 requested, **2 completed** |
| VQC | Adam | 0.05 | none | 30 |
| Learnable | Adam, wd=1e-4 | 0.05 | none | 300 |
| PoE | manual GD | 0.3 | none | 400 |

Grad clipping at norm 1.0; AMP via `GradScaler` on CUDA.

## 5.6 Augmentations

`ml/vision_cxr/dataset.py:28–38`, Albumentations:

```python
A.Resize(224,224), A.HorizontalFlip(p=0.5), A.Rotate(limit=12, border_mode=0, p=0.5),
A.RandomBrightnessContrast(0.15, 0.15, p=0.5), A.GaussNoise(p=0.2)
```

**Clinical concern:** `HorizontalFlip(p=0.5)` mirrors chest radiographs left-to-right. This destroys situs — the heart is on the left, the right lung has three lobes, the aortic arch is left-sided. For cardiomegaly and effusion laterality this is actively harmful, and it is a known anti-pattern in CXR literature. Rotation ±12° is also generous for a modality acquired in a standardised position.

## 5.7 Preprocessing & normalization

Two different paths exist, and they agree — which is good:

| Path | Resize | Scale | Normalize |
|---|---|---|---|
| Training (`ChestXrayDataset`) | Albumentations → 224 | /255 | (t − 0.449)/0.226 |
| Serving (`VisionModel._to_tensor`) | cv2 → 224 | /255 if max>1 | (t − 0.449)/0.226 |

0.449 and 0.226 are the channel-averaged ImageNet mean/std — correct for a grayscale adaptation of an ImageNet-pretrained net.

**But** `services/vision/cnn.CXRBackbone._to_tensor` uses a *different* normalization (`[-1024, 1024]` for xrv, or ImageNet mean for timm). Since `CXRBackbone` is never actually loaded (F: `best_model.pt` always wins), this divergence is latent rather than active — but it is a trap for the next engineer.

**The real preprocessing defect is resolution.** `study_from_cxr(grid=64)` reduces e.g. a 2544×3056 film to 64×64 by *index selection* (`io.py:80–82`):

```python
rows = _np.linspace(0, full.shape[0]-1, grid).astype(int)
cols = _np.linspace(0, full.shape[1]-1, grid).astype(int)
small = full[_np.ix_(rows, cols)]
```

This is nearest-neighbour subsampling with no anti-aliasing — it takes 4,096 individual pixels out of ~7.8 million and discards the rest. `full` is then never used again. The docstring says "The full image drives the CNN (it resizes internally to 224)" — **that is false**; only `small` reaches `StudyInput`.

At 64×64, a pneumothorax pleural line (typically 1–2 px at full resolution) and a 5 mm nodule are not merely blurred — they are gone. This alone explains a large part of the nodule AUROC 0.44.

`services/inference/predict.py:26` uses `_PREDICT_GRID = 224` instead, so the **CLI and the web upload disagree about image fidelity**.

## 5.8 Output decoding & thresholding

- Vision: `sigmoid(logits)` per finding, threshold 0.5 for "present" in reports and metrics. No per-class threshold tuning despite prevalences spanning 0.03–0.25.
- Fusion: `softmax(W·z + b)`.
- Safety: `softmax(logits / T)`, T = 0.7725.
- Abstention thresholds: OOD z > 1.5; top-p < 0.45; conformal set > 3; epistemic > 0.20.

**Note that T = 0.77 < 1.0 means the temperature scaling makes the model *more* confident, not less.** That is unusual — it indicates the raw fusion logits were *under*-confident on the synthetic calibration split.

## 5.9 Calibration

`fit_temperature` (`calibration.py:41`) minimises multiclass NLL over log T ∈ [−3, 3] via `scipy.optimize.minimize_scalar(method="bounded")`. Correct, standard, single-parameter — cannot overfit meaningfully.

**The defect is where it is applied.** In `train_fusion.run()`:

```python
backend = s.fusion_backend                    # "quantum"
cal_logits = _quantum_logits(...)             # fit T on QUANTUM
T = fit_temperature(cal_logits, ycal)
...
q_metrics = metrics(_quantum_logits(...), yte) # uses T
c_metrics = metrics(_classical_logits(...), yte) # ALSO uses quantum's T
```

and in `benchmark.run()`:

```python
"quantum":   _eval_backend(q_logits, yte, coverage, cal.temperature),
"classical": _eval_backend(c_logits, yte, coverage),   # default temperature=1.0
```

So the README's "ECE 0.020 vs 0.276, 13.8× better calibrated" compares a temperature-scaled model against an unscaled one. The project's own `audit_all.py` fits classical its own temperature (T = 0.31) and gets **ECE 0.0274** — the gap collapses from 0.256 to 0.007.

## 5.10 Training methodology

**Vision:** 259,038 MIMIC-CXR images, patient-disjoint train/validate (verified: 0 subject overlap), labels from a regex report labeler. 15 epochs requested, early-stopping patience 4. The served model completed **2 epochs**.

**Fusion:** 700 synthetic samples → 60/20/20 train/calibration/test split (`ml/training/dataset.py:27–34`). Trains PoE, ensemble (5 bootstrap members), learnable head, and the VQC (30 epochs, batch 50). Then fits T, q̂, Mondrian q̂ per class, and OOD statistics on the calibration split.

## 5.11 Inference pipeline

See Phase 9.

## 5.12 Evaluation metrics

`ml/evaluation/metrics.evaluate` produces: accuracy, NLL, Brier, ECE (10-bin), macro AUROC/AUPRC/sensitivity/specificity/PPV/NPV/F1, per-class one-vs-rest table, reliability curve.

`ml/evaluation/clinical_eval` adds: per-label 2×2 confusion matrices, operating-point rates, percentile bootstrap 95 % CIs (1000 iterations), ROC/PR/calibration/histogram/confusion plots.

`audit_all.py` adds: DeLong correlated-ROC test (Sun & Xu 2014 fast algorithm), exact McNemar, paired bootstrap difference CI, permutation test, Wilcoxon signed-rank, paired t, Shapiro-Wilk normality gate, Cohen's d.

**This is a genuinely strong evaluation stack** — more rigorous than most published medical-AI work.

## 5.13 Confidence estimation

`Prediction.ci_low/ci_high` (`safety/engine.py:97`):

```python
half = min(0.5, epistemic + 0.5 * p * (1 - p))
```

**This is not a statistical confidence interval.** It is a heuristic width: ensemble std plus half the Bernoulli variance. It has no coverage property and should not be labelled "CI" in a clinical UI. The `0.5 * p(1-p)` term peaks at p = 0.5 giving ±0.125, which is arbitrary.

## 5.14 Uncertainty estimation

`ensemble_decomposition` (`uncertainty.py:30`) implements the standard BALD decomposition correctly:

- Predictive entropy H[E[p]] — total
- Aleatoric E[H[p]] — expected member entropy
- Epistemic = MI = H[E[p]] − E[H[p]]

**But the ensemble is 5 classical PoE heads while the served posterior is the quantum VQC.** The epistemic uncertainty therefore measures disagreement among models that are *not the model being used*. It is a proxy, not a decomposition of the served predictor's uncertainty.

Also: `aleatoric` is normalised by log₂(6) at `engine.py:91` while `epistemic_mi` and `predictive_entropy` are not — so the two numbers printed side by side in the report are on different scales.

## 5.15 Clinical reasoning

8 rules with hand-set log-likelihood ratios (`reasoning/engine.py:65–180`): BNP thresholds (400/100 pg/mL, ACC/AHA), infection composite (WBC>11, PCT≥0.5, CRP>50, fever + consolidation ≥0.4, IDSA/ATS), malignancy (nodule + pack-years/prior cancer/haemoptysis/age, Fleischner), COPD (GOLD), CHF signs (orthopnoea + cardiomegaly/effusion), pneumothorax (BTS), immunosuppression, hypoxia (SpO₂<92).

The thresholds and citations are clinically accurate. The log-LR magnitudes (0.3–1.8) are hand-chosen, not derived from data, which the module states openly.

**The fatal flaw is that none of it reaches the impression** — F10.

---

# PHASE 6 — Quantum analysis

## 6.1 Exactly where quantum is used

**One place only:** `services/fusion/quantum.py`, reached from `FusionEngine.fuse_vector` when `fusion_backend == "quantum"` (the default) and `artifacts/fusion_quantum.npz` exists.

Nowhere else. Vision, safety, explainability, recommendation, reasoning, and reporting are entirely classical. `docs/QUANTUM_STACK.md` (34 KB) describes six quantum services Q1–Q6; **only Q1 (evidence fusion) exists in code.** `PROJECT_STATUS.md` states this honestly.

## 6.2 Framework

PennyLane, `default.qubit` simulator, `diff_method="best"`. Training uses the `torch` interface with parameter broadcasting; serving uses the `numpy` interface analytically.

```python
dev = qml.device(device_name, wires=n_qubits, shots=shots)   # shots=None at serving
@qml.qnode(dev, interface=interface, diff_method="best")
```

## 6.3 Data encoding

Angle encoding, one channel per qubit (`device.py:30–31`):

```python
for i in range(n_qubits):
    qml.RY(np.pi * x[..., i], wires=i)
```

Each evidence channel x_i ∈ [0,1] becomes a rotation of π·x_i about Y, so x=0 → |0⟩ and x=1 → |1⟩. Clean and appropriate.

## 6.4 The circuit

```
|0⟩ ─RY(πx₀)─┬─RY(θ)─RZ(θ)─●───────────────X─┬─ … ×3 layers … ─┤⟨Z₀⟩
|0⟩ ─RY(πx₁)─┼─RY(θ)─RZ(θ)─X─●─────────────  ┼─                ─┤⟨Z₁⟩
|0⟩ ─RY(πx₂)─┼─RY(θ)─RZ(θ)───X─●───────────  ┼─                ─┤⟨Z₂⟩
   ⋮                             ⋱             ⋮
|0⟩ ─RY(πx₇)─┴─RY(θ)─RZ(θ)─────────────●───X─┴─                ─┤⟨Z₇⟩
             └──────── repeated n_layers = 3 times ────────┘
```

- **Qubits:** 8 (= `len(EVIDENCE_CHANNELS)`)
- **Layers:** 3
- **Per layer:** RY(θ) + RZ(θ) on every qubit, then a CNOT ring `i → (i+1) mod 8`
- **Variational parameters:** 3 × 8 × 2 = **48**
- **Classical head:** W (6×8) + b (6) = **54**
- **Total trainable: 102 parameters**

Verified from the artifact:
```
quantum params: {'theta': (3, 8, 2), 'W': (6, 8), 'b': (6,), 'n_qubits': 8.0, 'n_layers': 3.0}
```

## 6.5 Measurement & classical decoding

Readout is single-qubit `⟨Z_i⟩` for each of the 8 qubits — a **local** observable, which is the correct choice for barren-plateau resistance (Cerezo et al. 2021). Then:

```
z ∈ [-1,1]^8  →  logits = W·z + b ∈ R^6  →  posterior = softmax(logits)
```

## 6.6 The uncertainty story

`QuantumFusion.fuse` claims finite-shot uncertainty:

```python
var_z = np.clip((1.0 - z**2) / max(n_shots,1), 0.0, None)   # Var[⟨Z⟩] with n shots
rng = np.random.default_rng(0)
for _ in range(128):
    zz = z + rng.normal(0.0, std_z)
    samples.append(softmax(self.W @ zz + self.b))
post_std = samples.std(axis=0)
```

**What is real:** the variance formula Var[⟨Z⟩] = (1 − ⟨Z⟩²)/N is the correct binomial variance for a Pauli-Z measurement.

**What is not:** no shots are actually taken. The device is created with `shots=None`, so the expectation is exact. The reported `n_shots=512` in `FusionResult` is metadata describing a hypothetical, and the "uncertainty" is a Gaussian Monte-Carlo propagation with a **fixed seed (`default_rng(0)`)**, making it fully deterministic. It is a simulation of what shot noise *would* be, not shot noise.

This is defensible as a modelling choice but the framing in `quantum.py:4–7` ("propagates *finite-shot* measurement variance through the readout") oversells it.

## 6.7 Does quantum actually contribute?

**Measured answer: no, not significantly, and not in production.**

From the project's own `audit_artifacts/run_20260719T175647Z/`, n = 100 held-out:

| backend | accuracy | NLL | Brier | ECE | macro AUROC |
|---|---|---|---|---|---|
| quantum (T=0.77) | 0.960 | 0.092 | 0.060 | **0.020** | 0.9989 |
| **learnable (classical)** | **0.970** | **0.062** | **0.033** | 0.024 | **0.9993** |
| classical_fair (T=0.31) | 0.930 | 0.189 | 0.103 | 0.027 | 0.9977 |
| classical_raw (T=1.0) | 0.930 | 0.488 | 0.203 | 0.276 | 0.9970 |

Statistical verdicts (`metrics/claim_verdicts.json`):

- **Accuracy:** "NOT VALIDATED" — 3 discordant samples, McNemar exact **p = 0.25**, permutation **p = 0.25**, diff CI95 [0.00, 0.07] touching zero.
- **Calibration:** "CONFOUNDED" — the 0.276 figure is the *uncalibrated* classical.
- **AUROC:** "NOT VALIDATED" — DeLong finds **0 of 6** classes significant; 4 classes are degenerate at AUROC 1.000 for both models.

The one genuine edge: per-sample log-loss is lower for quantum vs fairly-calibrated classical (Wilcoxon p = 2.4e-5) — but Cohen's d = −0.29, a *small* effect.

**And AURA's own best model is classical.** `LearnableFusion` — an attention-gated log-linear head, 118 parameters, trained in seconds with torch — beats the VQC on accuracy, NLL, Brier, and AUROC.

## 6.8 Is it genuine quantum ML?

**Yes, mechanically.** This is a real parameterised quantum circuit with real entanglement (the CNOT ring creates genuine multi-qubit correlations in a 2⁸ = 256-dimensional Hilbert space), trained by real gradient descent through a real quantum simulator. It is not a fake, not a random-number stand-in, not a classical function named "quantum".

**But it is demonstrative rather than advantageous.** Three reasons:

1. **The task is too easy.** Four of six classes hit AUROC 1.000 for *both* backends on synthetic data. There is no headroom in which a quantum advantage could appear.
2. **The model is tiny.** 48 variational parameters over 8 qubits at depth 3. A 256-dimensional Hilbert space explored by 48 parameters is not where quantum advantage lives.
3. **`default.qubit` is a state-vector simulator.** Everything is classically computed; the circuit is a specific, constrained function family, not a computational speedup.

## 6.9 Would a classical model perform similarly?

**It performs better.** That is the measured result, not a hypothesis — `LearnableFusion` wins on 4 of 5 metrics. The VQC's one lead (ECE 0.020 vs 0.024) is within noise at n = 100.

**Assessment:** the quantum component is architecturally well-placed (fusion, not imaging), honestly implemented, and correctly benchmarked by the project's own harness — which concluded against it. The engineering is sound; the *claim* is not.

---

# PHASE 7 — Dataset analysis

## 7.1 Datasets used

| # | Dataset | Status | Used by |
|---|---|---|---|
| 1 | MIMIC-CXR (aug CSVs + JPEGs) | **Present**, 261,137 images | Vision training, evaluation, worklist seeding |
| 2 | Synthetic CXR world | Generated at runtime | **All fusion training**, calibration, benchmark, audit |
| 3 | MIMIC-IV (ed/hosp/icu/notes) | **Directories empty** | Nothing — blocks `mimic/{features,tasks,training}` |

## 7.2 MIMIC-CXR — origin, size, structure

Source: a Kaggle redistribution (`simhadrisadaram/mimic-cxr-dataset/versions/2`) of MIMIC-CXR with augmented report paraphrases.

Measured:
```
train rows(subjects): 64,586    validate rows(subjects): 500
subject overlap train/validate: 0
JPEGs on disk: 261,137
TRAIN images resolvable on disk: 259,038
VAL   images resolvable on disk: 2,099
```

Schema: one row per `subject_id`, with parallel stringified-list columns `image`, `view`, `AP`, `PA`, `Lateral`, `text`, `text_augment`. Images live at `files/pXX/pXXXXXXX/sYYYYYYYY/<dicom-uuid>.jpg`.

## 7.3 Labels — how they are generated

**No official labels are used.** Labels are extracted from free-text reports by `mimic/labeling.py`, a rule-based CheXpert/NegBio-style labeler:

1. Lowercase, strip `___` de-identification placeholders, drop `findings:`/`impression:` headers.
2. Split into sentences on `[.!?\n]+`.
3. For each of 10 concept regexes, find a match.
4. Determine polarity by **forward scoping**: find the nearest scope-breaker (`but|however|except|although|though|otherwise|aside from|other than|;`) before the match; within that window, negation cues (`no|not|without|never|free of|negative for|…`) → 0, uncertainty cues (`may|possible|cannot exclude|suspicious|likely|…`) → −1, else 1.
5. Aggregate across mentions: positive > uncertain > negative.
6. Map to `Finding` enum; binary vector uses `1 if label == 1 else 0` (so uncertain **and** absent both become 0).

Measured class balance:

| class | train pos | train rate | val pos | val rate |
|---|---|---|---|---|
| opacity | 65,167 | 0.2516 | 496 | 0.2363 |
| consolidation | 12,947 | 0.0500 | 120 | 0.0572 |
| pleural_effusion | 54,288 | 0.2096 | 524 | 0.2496 |
| cardiomegaly | 32,259 | 0.1245 | 284 | 0.1353 |
| nodule | 11,425 | 0.0441 | 95 | 0.0453 |
| pneumothorax | 8,431 | 0.0325 | 63 | 0.0300 |
| hyperinflation | 14,856 | 0.0574 | 120 | 0.0572 |

Train and validation prevalences match closely — a good sign that the split is not distributionally skewed.

## 7.4 Advantages

- Large (259 k usable images), real, clinically diverse.
- Patient-disjoint train/validate, verified.
- Reports are genuine radiologist text.
- The labeler is transparent, auditable, and dependency-free.

## 7.5 Weaknesses, biases, and leakage

1. **Label noise from the regex labeler.** No validation against CheXpert-labeled ground truth exists in this repository. **Not implemented:** any measurement of labeler precision/recall. Every downstream metric inherits an unmeasured error rate.

2. **Uncertain collapsed to negative.** `_label_vec` maps CheXpert `-1` → 0. "Cannot exclude pneumothorax" trains the model that there is no pneumothorax. For rare, dangerous findings this systematically suppresses positives.

3. **The label-smearing bug (fixed but still in the served model).** The legacy path concatenated *all* of a subject's reports and applied one label vector to every image of that patient — cross-study contamination. `dataset.py:76–130` now labels per-study by parsing the study id from the path. **The served `best_model.pt` predates this fix.** The impact is quantified by the two evaluation runs:

| metric | smeared labels | per-study labels | inflation |
|---|---|---|---|
| macro AUROC | 0.7019 | **0.6665** | +5 % |
| macro AUPRC | 0.5582 | **0.2140** | **+161 %** |
| macro F1 | 0.5472 | **0.2549** | **+115 %** |
| micro F1 | 0.6228 | **0.3169** | **+97 %** |

4. **MIMIC-CXR's known population bias:** Beth Israel Deaconess, Boston, 2011–2016 emergency department. Skewed toward acute presentations and portable AP films. No external validation set of any kind. **Not implemented:** any cross-institution evaluation.

5. **View mixing.** AP, PA, and lateral films are pooled into one training set with no view conditioning. AP portables have magnified cardiac silhouettes, which directly confounds cardiomegaly.

6. **No data leakage found at the split level.** Subject overlap is 0, and `mimic/splits.py` derives split assignment deterministically from `md5(salt:subject_id)` so a patient can never straddle splits. This is correctly done.

7. **A leak does exist in the tabular path, and is correctly avoided:** `mimic/features.py:6–10` explicitly forbids using report-derived findings as features since they are the labels. Good discipline.

## 7.6 The synthetic dataset

`ml/data.py` generates 64×64 images by planting Gaussian blobs into anatomical regions of a procedural thorax. Prevalence: normal 0.34, pneumonia 0.18, heart failure 0.15, COPD 0.13, malignancy 0.12, pneumothorax 0.08.

**This is the dataset that trains every fusion model, fits every calibration constant, and produces every number in the README table and the audit.** The vision model is real; the reasoning brain that consumes it is not.

`make_multimodal` generates correlated labs/symptoms/history — which is why the reasoning engine can fire on synthetic cases but never on real MIMIC ones (`mimic/patient.py:multimodal_context` returns `None`).

## 7.7 Train/validation/test splits

| Path | Split | Method |
|---|---|---|
| Vision | train / validate | MIMIC's own CSVs, patient-disjoint (verified) |
| Fusion | 60/20/20 | Random shuffle of 700 synthetic samples, seed 7 |
| Audit | fresh test | `make_splits(500, seed=7+101)` — different seed, genuinely held out |
| Tabular | train/test/validation | `md5` hash of subject_id, 15 % test |

**No test set exists for the vision model.** MIMIC's `validate` split is used for early stopping (`best_macro_auroc` model selection) *and* reported as the evaluation result. That is model selection on the reported set — the 0.6665 macro AUROC is optimistically biased, though only mildly given 2 epochs.

---

# PHASE 8 — Training analysis

## 8.1 The vision training pipeline

`python -m ml.vision_cxr.train --epochs 15 --batch 16 --lr 3e-4 --num-workers N`

```
TrainConfig → build_loaders → DenseNet121CXR(7) → AdamW + CosineAnnealingLR
  → pos_weight from train label frequencies
  → RegularizedMultiLabelLoss(pos_weight, tv_weight=1e-4)
  → GradScaler(enabled = amp and cuda)
  → [resume from last_checkpoint.pt if --resume]
  → per epoch: train_one_epoch → evaluate_model → scheduler.step
              → csv_logger → tensorboard → save_best_model if improved
              → save_model_checkpoint (always) → plot_training_history
              → early stop if no improvement for `patience` epochs
  → run_post_training_validation()  [full E2E pipeline check on a real MIMIC film]
```

## 8.2 Checkpointing

`save_model_checkpoint` writes three files per epoch:
- `last_checkpoint.pt` — model + optimizer + scheduler + scaler + epoch + best_metric (84 MB)
- `optimizer.pt` — optimizer state again (56 MB)
- `scheduler.pt` — scheduler state again (1.5 KB)

The optimizer state is therefore stored **twice**, costing 56 MB per run per epoch. Three run directories × this = the ~420 MB of artifact bloat.

`save_best_model` writes `best_model.pt` — model + epoch + best_metric only (27 MB). This is the served file.

## 8.3 Resume

`load_model_checkpoint` restores model, optimizer, scheduler, scaler, and returns `(epoch+1, best_metric)`. Correct and complete. Enabled by `--resume`.

## 8.4 Best-model selection & early stopping

Selection metric: **validation macro-AUROC**. Patience default 4. Both correct.

**However — the wrong checkpoint is served.** Measured:

```
artifacts/best_model.pt          -> epoch 1, best_metric 0.6962, densenet121
artifacts/retrain_v2/best_model.pt -> epoch 0, best_metric 0.7848, densenet121
artifacts/_smoke/best_model.pt     -> epoch 0, best_metric 0.7373, densenet121
```

And `retrain_v2_train.log` shows it went further:
```
Epoch 00/15 | Train Loss: 0.9924 | Val Loss: 0.9689 | Macro-AUROC: 0.7848 | Time: 1258.0s
Epoch 01/15 | Train Loss: 0.9378 | Val Loss: 0.9462 | Macro-AUROC: 0.7859 | Time: 1086.4s
```

The retrained model (per-study labels, 0.7859) is **12.9 % better** than the served one (0.6962) and sits unpromoted in a subdirectory.

## 8.5 Logging

- CSV: `HistoryLogger` → `history.csv`, 40 columns (7 classes × 5 metrics + macro + loss).
- TensorBoard: `Loss/train`, `Loss/val`, `Metric/val_macro_auroc`, `LR`.
- Plots regenerated every epoch: `plots/loss_curves.png`, `plots/auroc_curves.png`.

## 8.6 Artifacts produced

`best_model.pt`, `last_checkpoint.pt`, `optimizer.pt`, `scheduler.pt`, `history.csv`, `tensorboard/`, `plots/`.

## 8.7 Training speed & hardware

From `artifacts/performance/PERFORMANCE_SUMMARY.md`:
```
Device: NVIDIA GeForce RTX 5050 Laptop GPU · torch 2.11.0+cu128
CPU latency (1 img): 83.171 ms (12.02 img/s)
GPU latency (1 img): 29.102 ms (34.36 img/s)
Peak GPU memory: 693.45 MB
Batch throughput: 32 → 618 img/s
```

From `retrain_v2_train.log`: **~1,100–1,260 s per epoch** over 259,038 images = ~210 img/s during training.

## 8.8 Bottlenecks

1. **JPEG decode.** 259 k `cv2.imread` calls per epoch. `num_workers` defaults to **0** (`config.py:21`), meaning decode runs on the main thread, serialised with GPU compute. The code knows this — `dataset.py:169` comments "JPEG decode dominates step time; workers overlap it with GPU compute" — but the default leaves the optimisation off.
2. **Mixed precision is a net loss at small batch.** Measured: `Mixed precision: 0.823x (25.648 → 31.167 ms/batch of 8)` — AMP made it *slower*, because at batch 8 the FP16 conversion overhead exceeds the tensor-core gain.
3. **Full-resolution JPEG → 224 resize every epoch.** No decode cache, no pre-resized dataset. This is the single highest-ROI training fix.

## 8.9 Windows-specific issues

- `num_workers=0` default is a rational Windows choice: `spawn` multiprocessing re-imports the module and pickles the dataset, which is slow and fragile. But it costs ~2× throughput.
- `aura_cli._utf8_stdout()` reconfigures stdout/stderr to UTF-8 because Windows `cp1252` crashes on report glyphs. I observed the symptom in live output: the em-dash in `Chronic obstructive pulmonary disease 87% — supported by…` rendered as a U+FFFD replacement character.
- Absolute Windows paths are baked into artifacts: `artifacts/vision_cnn_train.json` contains `C:\\Users\\aruls\\Desktop\\aura\\...` (a *different* user's machine), and `evaluation/metrics.json` contains `E:\AURA\aura\aura\artifacts\best_model.pt` (a stale path). These break reproducibility.
- `mimic/config.py:28` hardcodes `r"E:\AURA\datasets\..."` as the default root — env-overridable, but the default is machine-specific.

## 8.10 Performance limitations

- Two epochs of training on a 7 M-parameter network over 259 k images is **severe undertraining**. The loss was still falling (0.9924 → 0.9378) and val AUROC still rising (0.7848 → 0.7859) when training stopped.
- No learning-rate warmup; cosine annealing with `T_max=epochs` means if you stop at epoch 2 of 15, the LR barely decayed.
- No gradient accumulation in `vision_cxr/train.py` (it exists in `ml/training/train_cnn.py` but that is a different harness).

---

# PHASE 9 — Inference pipeline: what happens on upload

A clinician drags `chest.jpg` (2544×3056) onto the console.

### Step 1 — Browser
`apps/web/js/console.js:626` posts `multipart/form-data` to `/v1/studies/upload` via `api()` in `fx.js:272`.

### Step 2 — Gateway receives
`gateway/app.py:199 upload_study()`. Writes the bytes to a `NamedTemporaryFile` preserving the suffix.

### Step 3 — Intake gate
`services/vision/xray_gate.validate_cxr(tmp_path)`:
- Extension not DICOM → `_load_rgb`: PIL open, convert RGB, `thumbnail((256,256))`, /255.
- Chroma gate: `chroma = rgb.max(2) − rgb.min(2)`; reject if `mean > 0.08` or `mean(chroma>0.15) > 0.10`.
- `_gate_gray(rgb.mean(axis=2))`: aspect ∈ [0.4,2.5]; σ ≥ 0.04; 256-bin entropy ≥ 4.0 bits; centre/lateral brightness ratio ≥ 1.05; column-profile CV ≥ 0.09; then ≥ 2 of 3 soft signals.
- Failure → **HTTP 422** `{"error":"not_a_cxr","reason":…,"checks":…}` and an audit row. No case created.

### Step 4 — Load & downsample
`services/vision/io.study_from_cxr(tmp_path)`:
- `load_cxr` → `cv2.imread(IMREAD_GRAYSCALE)` → `_normalize01`: clip to the 0.5–99.5 percentile window, scale to [0,1]. (For DICOM: `apply_voi_lut`, MONOCHROME1 inversion.)
- **`grid=64`**: `linspace` index selection → 64×64. `full` discarded. **← F5**
- Returns `StudyInput(image=[4096 floats], image_shape=(64,64), priors=StructuredPriors())` — all priors default to `unknown/False`.

### Step 5 — Pipeline entry
`gateway/pipeline.py:60 run()`. `img = np.array(study.image).reshape((64,64))`. Publishes `study.received` (no subscribers).

### Step 6 — Vision
`VisionEngine.analyze` → `score_findings`:
- `backbone` is a `VisionModel` (because `best_model.pt` exists), so:
- `_to_tensor`: `cv2.resize(64→224, INTER_LINEAR)`, `/255` skipped (already ≤1), `(t − 0.449)/0.226`.
- `self.model(x)` → 7 logits → `sigmoid` → 7 probabilities.
- `_feature_scores(img)` runs the numpy path too, and CNN values overwrite it where present. Since `VisionModel._finding_index` covers **all 7** findings, the feature model contributes nothing here.
- `embedding`: `densenet.features(x)` → ReLU → global avg pool → 1024-d.
- Emits `VisionResult` with findings sorted descending.

**Measured on real MIMIC films** (5 validation patients):
```
subj 10003502  evidence: [0.82 0.92 0.97 0.93 0.75 0.72 0.15 0.  ]
subj 10013502  evidence: [0.66 0.77 0.83 0.77 0.2  0.62 0.32 0.  ]
subj 10072167  evidence: [0.31 0.36 0.51 0.44 0.62 0.19 0.07 0.  ]
```
Patient 10003502 is simultaneously assigned opacity 0.82, consolidation 0.92, effusion 0.97, cardiomegaly 0.93, nodule 0.75 and hyperinflation 0.72 — a clinically incoherent combination, and the signature of an undertrained network with a 20× `pos_weight` pushing every sigmoid up.

### Step 7 — Evidence encoding
`services/fusion/evidence.encode(vision, priors)` → `x ∈ [0,1]^8`. Channel 8 is `prior_risk_score(priors)` = 0.0 for every upload (no priors are collected by the UI).

### Step 8 — Fusion
`FusionEngine.fuse_vector(x)`:
1. `posterior, std = QuantumFusion.fuse(x)` → circuit → 8 `⟨Z⟩` → `W·z + b` → softmax; std by seeded MC.
2. Guard: `p_poe = softmax(classical.logits(x))`; `W₁(p_vqc, p_poe)` on the severity axis `[0.0, 0.4, 0.6, 0.7, 0.95, 1.0]`; dynamic τ = max(0.12, mean + 3·std of the last 128 distances).
3. If `W₁ > τ`: `posterior ← p_poe`, `std ← 0`, `fallback_triggered = True`.

Measured: the guard fired on **3 of 5** real films (distances 0.184, 0.312, 0.261 vs τ = 0.12).

### Step 9 — Safety ⚠
`SafetyEngine.assess(study_id, x, self.fusion.model)`:

```python
logits = fusion_model.logits(x)      # ← the QUANTUM model, unconditionally
probs  = softmax(logits / 0.7725)
```

**The guarded posterior from step 8 is never passed in and never used.** Measured consequence:

| subject | guard fired | fusion.posterior top | safety.top → the report |
|---|---|---|---|
| 10003502 | yes | pneumonia 0.546 | **copd 0.830** |
| 10013502 | no | pneumonia 0.359 | pneumonia 0.416 |
| 10072167 | yes | **normal 0.446** | **malignancy 0.551** |
| 10075925 | no | copd 0.307 | copd 0.337 |
| 10174198 | yes | **normal 0.374** | **copd 0.867** |

Then:
- Epistemic: 5-member `DeepEnsemble` (classical) → BALD decomposition.
- Conformal: Mondrian set from `conformal_mondrian.npy` = `[0.1166, 0.0360, 0.0120, 0.0111, 0.9889, 0.1435]`.
- OOD: `z = (energy − (−2.7746)) / 0.7044`; abstain if `z > 1.5`. Measured range on real films: **−0.43 to +1.19** → never fires.
- Abstention cascade: OOD → low confidence (p<0.45) → conformal set >3 → epistemic >0.20.

### Step 10 — Explainability
`ExplainEngine.explain` → `M.all_methods(backbone, img, top_finding, out_size=64)` → Grad-CAM, Grad-CAM++, IG (32 steps), SmoothGrad (25 samples). Primary = Grad-CAM++. Then leave-one-out attribution over the 8 evidence channels.

### Step 11 — Recommendations
`RecommendEngine.recommend`: for each of 5 catalog items, EVOI over 2^k resolved outcomes of the *resolvable* channels (those with `0.08 < x_j < 0.92`); then greedy panel selection; then chained-MI joint EIG for the panel headline.

### Step 12 — Reasoning
`ClinicalReasoner.reason(...)` with `multimodal=None` for uploads → **all 8 rules return `None`** → `adjusted_posterior == prior_posterior`. The differential is built purely from the imaging posterior.

### Step 13 — Report
`ReportEngine.compose(vision, safety, recommendations, reasoning)`. Impression from `safety.top`. Differential from `reasoning.differential`.

Observed output for subject 10174198:
```
Impression: findings most consistent with chronic obstructive pulmonary disease
            (calibrated probability 87%). 90% mondrian confidence set:
            Suspicious pulmonary malignancy, ...
Differential: Chronic obstructive pulmonary disease 87% — supported by hyperinflation;
              Pneumonia 9%; Suspicious pulmonary malignancy 2%; Pneumothorax 2%.
```

Note the confidence set contains *malignancy* while the differential assigns it 2 % — the F7 Mondrian degeneracy surfacing directly in the clinician-facing text.

### Step 14 — Memory, persistence, response
`MemoryEngine.index(case_id, embedding, dx)` (in-process). `store.save_case(bundle)` → JSON document + indexed columns. `store.audit(...)`. Returns `{"case_id": …}`. The browser refetches `/v1/cases` and `/v1/cases/{id}` and renders.

**Measured end-to-end latency:** ~29 ms GPU / ~83 ms CPU for the vision forward pass; the full pipeline including 4 gradient attribution methods and EVOI enumeration is substantially more.

---

# PHASE 10 — Web application

## 10.1 Frontend architecture

No framework, no build step, no `node_modules`. Four scripts loaded in order with cache-busting query strings (`?v=3`):

| File | Size | Role |
|---|---|---|
| `js/fx.js` | 11.7 KB | Canvas effects + the single `api()` fetch wrapper |
| `js/landing.js` | 18.3 KB | Landing page, `/v1/admin/safety` panel |
| `js/console.js` | 32.2 KB | The clinical console — worklist, case detail, upload, feedback, sign |
| `js/main.js` | 4.2 KB | Boot sequence, `/v1/health` polling |
| `js/history.js` | 21.2 KB | Separate history/report portal (`history.html`) |

Rendering uses `<canvas>` for the X-ray view and saliency overlay (`#xray`, `#xray-sal`), plus ECG, particle-field, portal, warp, and ambient canvases for the transition choreography.

## 10.2 Backend architecture

FastAPI with a `lifespan` context manager. Single-process, single `Pipeline` instance.

## 10.3 Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | backend, trained flag, case count |
| GET | `/v1/cases` | worklist rows (lightweight, not full bundles) |
| GET | `/v1/cases/{id}` | full `CaseBundle` JSON |
| POST | `/v1/cases/{id}/feedback` | verdict + correction; triggers ACI |
| POST | `/v1/cases/{id}/report/sign` | state → SIGNED |
| POST | `/v1/studies/simulate` | generate + analyse a synthetic study |
| POST | `/v1/studies/upload` | gate + analyse an uploaded film |
| GET | `/v1/cases/{id}/similar` | top-3 cosine neighbours |
| GET | `/v1/models` | registry contents |
| GET | `/v1/admin/safety` | registry + benchmark + feedback + abstention rate + audit tail |
| GET | `/`, `/app`, `/history` | SPA entry points |

## 10.4 Routing
`/` and `/app` both return `index.html`; the SPA decides which view to mount. `/history` returns `history.html`.

## 10.5 Image upload
Covered in Phase 9 steps 2–4. Note the temp file is cleaned in a `finally` block — correct.

## 10.6 Database
SQLite via SQLAlchemy 2.0 declarative mapping. Tables: `cases`, `feedback`, `conformal_state`, `outcomes`, `audit_log`. Case bundles stored as a JSON column with denormalised indexed columns (`state`, `priority_score`, `top_diagnosis`, `abstained`) for worklist queries. `aura.db` is currently 4.3 MB.

## 10.7 History
`history.html` + `history.js` provide a report portal reading `/v1/cases` and `/v1/cases/{id}`, with a sign action.

## 10.8 Visualization
Canvas-rendered radiograph with a saliency heat overlay; charts for the posterior and evidence attribution; the CLI additionally writes matplotlib PNGs and a self-contained `explanation.html` (1 MB with embedded images).

## 10.9 Error handling

Mixed quality:

- **Good:** `xray_gate` never raises — undecodable files return a `GateResult(False, …)`. `explain/methods.all_methods` wraps each method in try/except so one failure cannot sink the explanation. `get_backbone` returns `None` on any failure so callers degrade to the feature model. The MIMIC seeder catches everything so a data problem never blocks startup.
- **Poor:** `app.py:241` catches bare `Exception` and returns `HTTPException(500, f"Failed to process custom image: {str(e)}")` — leaking internal exception text (which can include filesystem paths) to the client.
- **Silent:** `audit_mw` swallows audit-write failures with a bare `except: pass` (`app.py:81`). An audit log that can silently fail to write is not an audit log.

## 10.10 Caching
Only HTTP `Cache-Control: no-cache` on `/`, `/app`, `/static` plus `?v=3` query strings. `get_settings()` is `@lru_cache`'d. `get_backbone` has a module-level `_CACHE` dict. **No response caching, no model output caching, no CDN.**

## 10.11 Security

| Control | Status |
|---|---|
| Authentication | **Not implemented** |
| Authorization / RBAC | **Not implemented** |
| Transport encryption | **Not implemented** (uvicorn on `127.0.0.1:8000`, HTTP) |
| Encryption at rest | **Not implemented** (plain SQLite) |
| CSRF protection | **Not implemented** |
| Rate limiting | **Not implemented** |
| Upload size limit | **Not implemented** — `await file.read()` loads the whole body into memory |
| Upload type allowlist | Partial — the gate is content-based, not MIME/extension-based (arguably better) |
| Audit log | Present, append-only *by convention* only — no DB constraint prevents UPDATE/DELETE |
| Secrets management | N/A — no secrets exist |
| Dependency pinning | Lower bounds only (`>=`), no lockfile |

The binding to `127.0.0.1` is the only thing standing between this and an open service. `docs/DEPLOYMENT.md` exists; **Assumption:** it describes intended rather than implemented controls.

## 10.12 Deployment
`run.bat` / `run.sh` → pip install → train → bench → serve. Single process, single machine, offline. **Not implemented:** Dockerfile, CI, health probes beyond `/v1/health`, migrations, horizontal scaling.

---

# PHASE 11 — Code quality audit

## 11.1 Dead code & unused files

| Item | Evidence |
|---|---|
| `services/fusion/projection.py` (entire module) | grep: zero importers |
| `device.make_reuploading_qnode` + `n_params_reuploading` | grep: zero importers |
| `conflict.WassersteinTieBreaker.distance_cost_matrix` | no caller |
| `memory/engine.prior_delta` | no caller |
| `common/eventbus` subscribers | 4 topics published, 0 subscribed |
| `services/vision/cnn.CXRBackbone` | unreachable while `best_model.pt` exists |
| `_probe_audit.py`, `test_image.py`, `aura/_final_test_result.txt` (0 bytes) | scratch/obsolete |
| `causal.LAB_MARKERS` edges | markers not in `EVIDENCE_CHANNELS`; can never fire |
| `artifacts/safety.synthetic-ood.bak.npz` | backup, not loaded |

## 11.2 Duplicate logic

1. **`_EV_LABEL` dictionary defined twice** — `services/report/engine.py:42` and `services/report/clinical_report.py:43`, with the second having 3 extra keys. Guaranteed to drift.
2. **Nearest-neighbour resize implemented twice** — `services/vision/features._resize_to` and inline in `services/vision/io.study_from_cxr`.
3. **`_study_of` study-id regex implemented twice** — `ml/vision_cxr/dataset.py:14` (`[\\/](s\d+)[\\/]`) and `mimic/timeline.py:35` (`/(s\d+)/`) with different escaping.
4. **ECE implemented three times** — `calibration.expected_calibration_error`, `uncertainty.reliability_curve`, `clinical_eval.binary_ece`.
5. **Severity ordering defined twice** — `recommend/engine._SEVERITY` (NORMAL 0.20) and `fusion/conflict._SEVERITY` (NORMAL 0.00). They disagree. `conflict.py:41` claims "Same ordering intent as `recommend.engine._SEVERITY`" — the ordering matches but the values differ, which changes the EMD.
6. **Optimizer state saved twice per checkpoint** — inside `last_checkpoint.pt` and again as `optimizer.pt`.

## 11.3 Architectural issues

1. **Layering violation:** `services/` imports from `ml/`, contradicting `ml/__init__.py`'s stated rule. Specifically `services/vision/features.py:11` → `ml.data` and `services/explain/engine.py:11` → `ml.data.IMG`. The synthetic-data module is a hard dependency of the serving path.
2. **Cross-service private import:** `services/explain/engine.py:10` imports `services.vision.features._resize_to` — a leading-underscore private, across a service boundary that `services/__init__.py` declares must not exist.
3. **`IMG = 64` is a global constant in the synthetic data module** and is used as the saliency grid size everywhere, coupling explanation resolution to the toy generator.
4. **The guard/safety split** (F2) — the fusion engine computes a decision that the safety engine structurally cannot see, because the interface passes a *model* rather than a *posterior*.

## 11.4 Code smells

- `gateway/app.py:44` imports `os` *inside* the lifespan function; `app.py:161` imports `CaseState` inside a handler; `app.py:207` imports `tempfile, os` inside a handler. Function-local imports used as a style, not for lazy-loading.
- `app.py:269` `get_settings_artifacts()` is a one-line function wrapping a module import, called once.
- `VisionEngine.predict` (`engine.py:160`) is an alias for `analyze` with the comment "Alias for analyze to satisfy predictability of prediction invocation" — meaningless.
- `services/inference/predict.py:36` has `# pragma: no cover - not hit from CLI` on a branch that *is* hit whenever called from an async context.
- `RecommendEngine._greedy_panel` references `best_marginal` which is only assigned inside the loop; safe only because `best is None` short-circuits first. Fragile.
- Bare `except Exception` at `app.py:81`, `app.py:241`, `engine.py:66`, `cnn.py:222`, `predict.py:95`.

## 11.5 Security issues

Beyond §10.11:
1. **Unauthenticated file upload with no size limit** — `await file.read()` into memory. A large file is a trivial DoS.
2. **Exception text returned to the client** (`app.py:242`).
3. **`torch.load` without `weights_only=True`** — `ml/vision_cxr/inference.py:18`, `checkpoint.py:31`, `services/vision/cnn.py:126`. Loading an untrusted checkpoint executes arbitrary code. The checkpoints here are self-generated, so exposure is low, but the pattern is wrong and torch now warns about it.
4. **Audit log is append-only by convention only** (`storage.py:90` comment) — no trigger or constraint enforces it.
5. **PHI risk:** case bundles store the full image as a float array inside a JSON column in an unencrypted SQLite file.

## 11.6 Maintainability

**Strengths — genuinely unusual quality here:**
- Every module has a substantial docstring explaining *why it exists*, not just what it does.
- `conflict.py`, `aci.py`, `causal.py`, `projection.py`, `losses.py`, and `model.py` include mathematical derivations with citations (Gibbs & Candès 2021, McClean et al. 2018, Cerezo et al. 2021, Pérez-Salinas et al. 2020, Sun & Xu 2014).
- Several modules include an explicit **"honest caveat"** section documenting their own failure modes. `losses.py` warns that TV can erase small lesions. `xray_gate.py` warns that impostors can slip through. `mimic/timeline.py` flags the timestamp proxy. This is rare and admirable.
- Type hints throughout; Pydantic contracts as the single interface source of truth.
- 117 tests.

**Weaknesses:**
- Documentation describes intent that the code does not implement (`ARCHITECTURE_REFACTOR.md` claims `projection.py` is "wired in `services/fusion/engine.py`"; it is not). Documentation that is wrong is worse than absent, because it stops the next engineer from checking.
- No CI configuration of any kind.
- No linter/formatter config (no ruff, black, or flake8 settings).
- Absolute machine-specific paths embedded in committed artifacts.

## 11.7 Scalability

- Single-process, single `Pipeline`, in-memory `MemoryEngine` (unbounded list, lost on restart, O(n) similarity scan).
- SQLite with full-image JSON blobs — `list_cases` deserialises `r.bundle` for every row to read two nested fields (`storage.py:140–151`), so the "lightweight worklist" query is O(rows × bundle size).
- No queue, no worker pool, no batching. Concurrent uploads serialise on the GIL and on the single CUDA context.

## 11.8 Memory

- `MemoryEngine._store` grows without bound.
- `WassersteinTieBreaker._hist` is a bounded `deque(maxlen=128)` — correct.
- `services/vision/cnn._CACHE` is unbounded but keyed by a small finite set.
- Upload reads the entire file into RAM.
- Each `CaseBundle` carries `image` (4,096 floats), `saliency` (4,096), and `saliency_methods` (4 × 4,096) = ~24 k floats, JSON-serialised into SQLite. At 224 grid (CLI) that becomes ~300 k floats per case.

## 11.9 Thread safety & race conditions

FastAPI runs **sync** `def` endpoints in a threadpool. `Pipeline` and its engines are process-wide singletons. Therefore:

1. **`RecommendEngine._panel_evoi`** (`engine.py:194`, mutated in `_greedy_panel`) — an instance attribute used as loop state across a multi-step computation. Two concurrent requests interleave and corrupt each other's panel utility. **Real race.**
2. **`WassersteinTieBreaker._hist`** — `deque.append` is atomic, but `threshold()` reads mean/std while another thread appends, so the τ used can reflect a partially-updated history. Benign but nondeterministic.
3. **`MemoryEngine._store.append`** — `list.append` is atomic under the GIL; safe.
4. **The torch model** — inference under `torch.no_grad()` is thread-safe for reads, but concurrent CUDA calls on one context serialise.
5. **`ExplainEngine.occlusion_saliency` mutates `img` in place** (`engine.py:37`) and restores it. If the same array were shared across threads this would corrupt; currently each request has its own array, so it is safe *by accident*.

Note `/v1/studies/upload` and `/v1/studies/simulate` are `async def`, so they run on the event loop and block it during the entire synchronous pipeline — meaning a single upload freezes all other requests. That is a different, equally real problem: **blocking the event loop with CPU/GPU work**.

## 11.10 GPU inefficiencies

1. **`num_workers=0` default** — GPU idles during JPEG decode.
2. **AMP is a measured net loss at batch 8** (0.823×).
3. **Batch size 1 at serving.** No request batching; measured 26 img/s at batch 1 vs 618 img/s at batch 32 — a **23× throughput gap** left on the table.
4. **Four separate gradient attribution methods per request**, each a full forward+backward: Grad-CAM (1), Grad-CAM++ (1), IG (32 steps), SmoothGrad (25) = ~59 forward/backward passes per case, on the live serving path.
5. **`model.zero_grad(set_to_none=True)` called inside the attribution loop** rather than once.
6. No `torch.compile`, no TensorRT, no ONNX export, no half-precision inference.

---

# PHASE 12 — Medical validation

## 12.1 Clinical realism

**Reasonable:** the finding/diagnosis separation (observations vs interpretations) mirrors real radiology reporting. `NORMAL` as a first-class label is correct. The report structure (Findings / Impression / Differential / Confidence / Recommendation) matches clinical convention. Guideline citations are real and appropriate (ACC/AHA, IDSA/ATS, GOLD, BTS, Fleischner/Lung-RADS).

**Unrealistic:** six diagnoses cannot cover chest radiography. There is no "other", no "technically inadequate", no "comparison required". A patient with pulmonary embolism, TB, ARDS, sarcoidosis, rib fracture, or a misplaced line gets forced into one of six boxes. Findings are reported without laterality despite `_FINDING_REGION` hardcoding a right-sided box for effusion.

## 12.2 Medical correctness

| Item | Assessment |
|---|---|
| BNP >400 / <100 thresholds | Correct |
| Procalcitonin ≥0.5 ng/mL for bacterial | Correct |
| WBC >11 ×10⁹/L leukocytosis | Correct |
| SpO₂ <92 % hypoxaemia | Correct |
| Nodule + ≥20 pack-years + age → malignancy | Correct direction |
| Severity ordering (pneumothorax > malignancy > HF > pneumonia > COPD > normal) | Clinically sound |
| Horizontal flip augmentation | **Medically wrong** — destroys situs |
| Effusion region fixed to rows 0.72–0.92, cols 0.10–0.90 | Bilateral box regardless of actual side |
| `prior_risk` collapses smoker+cancer+fever+age+immunosuppression into one scalar | Loses all specificity; fever and prior cancer point at different diagnoses |

## 12.3 Diagnosis quality — measured

On honest per-study labels (`artifacts/evaluation_perstudy/`, n = 2,099):

| finding | AUROC | sens | spec | F1 | ECE | support |
|---|---|---|---|---|---|---|
| opacity | 0.624 | 0.653 | 0.526 | 0.410 | 0.283 | 496 |
| consolidation | 0.710 | 0.858 | 0.420 | 0.150 | 0.517 | 120 |
| pleural_effusion | 0.759 | 0.855 | 0.528 | 0.522 | 0.320 | 524 |
| cardiomegaly | 0.744 | 0.859 | 0.507 | 0.343 | 0.401 | 284 |
| **nodule** | **0.444** | 0.232 | 0.709 | 0.063 | 0.349 | 95 |
| **pneumothorax** | **0.611** | **0.143** | 0.885 | 0.059 | 0.184 | 63 |
| hyperinflation | 0.775 | 0.642 | 0.771 | 0.237 | 0.306 | 120 |
| **macro** | **0.667** | 0.606 | 0.621 | 0.255 | **0.337** | — |

## 12.4 Failure cases

1. **Pneumothorax: sensitivity 0.143.** The model misses **86 % of pneumothoraces**. Tension pneumothorax is a minutes-to-death diagnosis. This is the single most dangerous number in the system, and it sits behind a UI that assigns pneumothorax the highest worklist urgency (1.0) — creating the impression of vigilance where there is none.
2. **Nodule: AUROC 0.444.** Below chance. The model's nodule score is *anti-correlated* with nodule presence. Ranking by it is worse than random. Lung cancer screening is the highest-value use of CXR AI.
3. **Saturated multi-finding output.** Real films produce 5–6 simultaneous high-probability findings (measured §9 step 6), which is clinically incoherent and drives the fusion into arbitrary regions.
4. **Guard-vs-report contradiction** (F2): a case where the trusted classical estimator said *normal* was reported as *malignancy 55 %*.
5. **Malignancy in ~78 % of confidence sets** (F7): measured over 2,000 random posteriors, the Mondrian set includes malignancy 77.8 % of the time vs 3.5–5.2 % for other classes.

## 12.5 False positives

`prevalence`-driven. Specificity averages 0.621 — roughly **4 in 10 healthy patients get a false positive finding**. Combined with the malignancy conformal defect, a large fraction of normal patients would receive a report whose confidence set names malignancy. In a screening context this generates enormous downstream CT utilisation and patient anxiety.

## 12.6 False negatives

86 % of pneumothoraces, 77 % of nodules, 36 % of hyperinflation. The abstention machinery does not compensate: on 5 real films, 3 produced confident committed diagnoses and OOD never fired.

## 12.7 Safety mechanisms — status

| Mechanism | Implemented | Effective |
|---|---|---|
| Temperature scaling | Yes | Yes (on synthetic) |
| Marginal conformal | Yes | Superseded by Mondrian |
| Mondrian conformal | Yes | **No — degenerate (F7)** |
| Adaptive conformal (ACI) | Yes | **No — not wired (F9)** |
| Deep-ensemble epistemic | Yes | Partially — measures the wrong model |
| Energy OOD | Yes | **No — recalibrated until it stopped firing (F8)** |
| Abstention policy | Yes | Partially — low-confidence gate fires; others do not |
| Wasserstein conflict guard | Yes | **No — output discarded (F2)** |
| X-ray intake gate | Yes | **Yes** — this one works |

**Seven of nine safety mechanisms are non-functional in production.** The system's entire differentiating claim is safety.

## 12.8 Calibration

Vision ECE 0.337 macro on per-study labels — severely miscalibrated. `vision_calibration.py` exists to fix this and wrote `vision_calibration_perfinding.json`, but the served `VisionModel.score_findings` applies **raw sigmoid with no temperature**. The per-finding calibration is computed and not applied.

Fusion ECE 0.0147 on *synthetic* data. Meaningless for real inputs given F1.

## 12.9 Explainability

Technically strong: five attribution methods, correct Grad-CAM++ α-weighting, IG with a proper black baseline and completeness, leave-one-out counterfactuals, bounding boxes from thresholded heatmaps, an HTML report bundling everything.

Clinically limited: attributions are computed at **64×64** on an image the CNN saw at 224 after upsampling from 64 — so the saliency resolution is 64×64 over a film that was 2544×3056. No localisation ground truth exists, so no pointing-game or IoU validation is possible. **Not implemented:** any quantitative explanation faithfulness metric.

## 12.10 Regulatory readiness

**FDA (510(k) / De Novo, SaMD):** not ready. Missing: a defined intended-use statement, predicate device analysis, a locked algorithm, design controls under 21 CFR 820.30, a documented risk analysis (ISO 14971), clinical validation on a prospectively-defined multi-site dataset, reader studies, a cybersecurity plan, and labeling. The current sensitivity for pneumothorax alone would fail any reasonable acceptance criterion.

**CE / EU MDR:** not ready. Under MDR Rule 11, software providing diagnostic information is Class IIa at minimum and likely IIb given the mortality-relevant findings. Requires a notified body, a full technical file, IEC 62304 software lifecycle compliance, clinical evaluation under MEDDEV 2.7/1 rev 4, and post-market surveillance. None exists.

**GDPR / HIPAA:** the offline-by-default design is genuinely favourable — no data egress, no cloud. But there is no encryption at rest, no access control, and no audit-log integrity guarantee.

**Positively:** `docs/ARCHITECTURE_REFACTOR.md` includes an explicit regulatory framing paragraph stating these are "research-grade components. None is a validated medical device function." That disclaimer is correct and should be far more prominent.

## 12.11 Deployment risks

1. **Automation bias.** A confident "COPD 87 %" from a model with macro AUROC 0.667 will anchor a tired clinician.
2. **Missed pneumothorax with an authoritative-looking report.**
3. **The safety theatre problem.** The UI displays conformal sets, epistemic uncertainty, OOD status, and abstention reasons — all of which look like working safety machinery. Seven of nine are not. A user cannot distinguish a displayed guarantee from an enforced one.
4. **No versioned audit of which model produced which report** beyond a `model_version` string.
5. **No mechanism to recall or invalidate prior reports** if a model defect is found.

## 12.12 Hallucination risk

**Low, and this deserves credit.** There is no LLM anywhere. All text is deterministic templating over structured fields. A sentence can only exist if the field backing it exists. The `grounding` map explicitly ties each section to the evidence nodes that produced it.

The residual risk is not hallucination but **misgrounding**: the impression is grounded in `safety.top`, which — per F2 — can contradict the guarded fusion posterior it purports to summarise. The text is faithful to a number; the number is wrong.

---

# PHASE 13 — SIH Grand Finale judge review

Scoring as an evaluator who reads the code, not just the deck.

| Criterion | Score | Reasoning |
|---|---|---|
| **Innovation** | 7.5 / 10 | Placing quantum at *evidence fusion* rather than imaging is a genuinely smart, defensible insight. Conformal + EVOI + abstention as the product (rather than accuracy) is a strong, differentiated thesis. Points lost because the novel modules (ACI, conflict guard, causal EIG) are not actually connected. |
| **Technical depth** | 8.5 / 10 | Very high. Real VQC, real DenseNet, DeLong/McNemar/bootstrap/permutation testing, Mondrian conformal, BALD decomposition, luminance-weighted channel collapse, TV regularisation, EMD tie-breaking. The mathematics in the docstrings is correct and cited. This is well above typical hackathon depth. |
| **AI quality** | 4 / 10 | Macro AUROC 0.667, pneumothorax sensitivity 0.143, nodule AUROC 0.444 (below chance), ECE 0.337. Two epochs of training. The better checkpoint was not promoted. |
| **Quantum integration** | 5 / 10 | Real and correctly placed, but 102 parameters, no demonstrated advantage, and the project's own audit returns NOT VALIDATED on all three headline claims. Q2–Q6 are documentation only. |
| **Medical usefulness** | 3 / 10 | At current sensitivity it would miss most of what matters. The reasoning engine — the most clinically interesting component — cannot influence the diagnosis. |
| **Explainability** | 8 / 10 | Five methods, evidence attribution, counterfactuals, grounded reports, no LLM. Held back by 64×64 resolution and no faithfulness validation. |
| **Presentation** | 9.5 / 10 | Exceptional. Cinematic console, coherent narrative, 7 pitch decks, screenshots, a self-contained prototype HTML. Best-in-class for this format. |
| **Scalability** | 3 / 10 | Single process, SQLite with image blobs, in-memory index, no batching, event loop blocked by inference. |
| **Novelty** | 7 / 10 | The "calibrated doubt as the product" framing is genuinely novel positioning. The individual techniques are established. |
| **Real-world impact** | 3.5 / 10 | The offline/no-PHI-egress design is a real advantage for Indian district hospitals. The model performance is not deployable. |
| **Deployment readiness** | 2 / 10 | No auth, no encryption, no CI, no container, no migrations, no monitoring. |
| **Maintainability** | 8 / 10 | Outstanding docstrings, clean contracts, real modularity, 117 tests. Docked for documentation that describes unimplemented wiring, and for the `ml`↔`services` layering violation. |

### Overall: **58 / 100**

**Weighted view (Innovation 15 %, Technical depth 15 %, AI quality 15 %, Quantum 10 %, Medical 10 %, Explainability 10 %, Presentation 10 %, Scalability 5 %, Impact 5 %, Deployment 5 %): 5.86 → 59/100.**

### What a judge would say

> "This is one of the most *architecturally* impressive submissions here, and the honesty in the codebase — the self-audit that falsifies your own headline, the 'honest caveat' sections, the `available=False` task registry — is genuinely rare and admirable. It shows real scientific character.
>
> But three things will sink you in cross-examination. First, your README claims a 13.8× calibration advantage that your own `audit_all.py` labels CONFOUNDED. Second, your fusion model is trained on a distribution your production system never produces — I can reproduce this in one command. Third, your Wasserstein conflict guard, which you present as your safety innovation, has no effect on the diagnosis your report prints; I found a case where it said *normal* and your report said *malignancy 55 %*.
>
> Fix the wiring, retrain properly, and lead with 'calibration discipline and a working conformal guarantee' instead of 'quantum beats classical'. That story is true, defensible, and still differentiated. The current story is not, and one informed judge will find it."

### Realistic placement
Strong regional finalist. Top-3 nationally on presentation and architecture alone. Would not survive a technical panel containing one practising radiologist or one quantum-ML researcher.

---

# PHASE 14 — What is real?

## 14.1 Genuinely implemented and working

| Component | Evidence |
|---|---|
| DenseNet-121 CXR classifier | 27 MB checkpoint, loads, infers, measured 29 ms GPU |
| 8-qubit PennyLane VQC | 102 params in `fusion_quantum.npz`, real circuit, real entanglement |
| Classical PoE / ensemble / learnable backends | 4 `.npz` artifacts, all load and infer |
| Temperature scaling | T = 0.7725 fitted and applied |
| Split-conformal + Mondrian | Implemented; Mondrian is degenerate but real |
| BALD uncertainty decomposition | Correct implementation |
| Energy-score OOD | Correct formula, mis-calibrated |
| Grad-CAM / ++ / IG / SmoothGrad / Score-CAM / occlusion | All six work; artifacts on disk |
| X-ray intake gate | Works; measured multi-gate rejection |
| Report labeler | Real regex labeler with negation scoping |
| MIMIC-CXR loading | 259,038 images resolved on disk |
| FastAPI gateway + SPA | Runs; 11 endpoints |
| SQLite persistence + audit | 4.3 MB DB with real cases |
| Reproducible audit harness | Full run directory with seeds, env capture, git hash, plots, verdicts |
| Test suite | **114 passed, 3 failed** (failures are a missing `matplotlib` in the global interpreter, not code defects) |
| Performance benchmark | Real GPU measurements |

## 14.2 Simulated

| Component | Nature |
|---|---|
| Quantum hardware | `default.qubit` state-vector simulator. Honest — never claimed otherwise. |
| Shot noise | **Not sampled.** Analytic expectation + seeded Gaussian MC with `default_rng(0)`. `n_shots=512` is metadata. |
| Event bus | Real pub/sub implementation, zero subscribers. A seam, not a system. |

## 14.3 Synthetic

| Component | Nature |
|---|---|
| All fusion training data | `ml/data.make_dataset` — 700 procedural 64×64 images |
| All calibration constants | Fit on that synthetic data |
| Labs / symptoms / history | `ml/data.make_multimodal` — generated for synthetic cases only |
| Ground-truth diagnosis for fusion | Assigned by the generator |
| README benchmark table | Synthetic, and unfairly temperature-scaled |
| `_SEVERITY` values, causal graph edges, log-LR magnitudes | Hand-authored priors, openly labelled as such |
| `HISTORICAL_CORRELATION` matrix | `causal.py:69` — "a plausible clinical prior; replace with an estimate from your local outcome log" |

## 14.4 Mocked / stubbed

| Component | Status |
|---|---|
| Authentication | Stub — `x-aura-user` header, no verification |
| RBAC / OIDC | Comment marking a "seam"; no code |
| PostgreSQL | SQLite with a repository boundary; the swap is claimed, not done |
| Redis Streams / NATS | In-process `EventBus` |
| MIMIC-IV tabular data | Directories exist, are empty |

## 14.5 Placeholder / aspirational

| Component | Status |
|---|---|
| `services/fusion/projection.py` | Complete, correct, **zero importers** |
| `device.make_reuploading_qnode` | Complete, **zero importers** |
| ACI | Complete, correct, **updates a value nothing reads** |
| Quantum services Q2–Q6 | 34 KB of design documentation, no code |
| `CLINICAL_NA_FEATURES` (12 scores) | Emitted as 0.0 + missing indicator — **honest placeholder, correctly done** |
| Unavailable ML tasks (mortality, readmission, sepsis, LOS, shock, ICU transfer) | Registered `available=False` with a reason — **honest, correctly done** |

## 14.6 The honesty ledger

It is worth stating plainly: this project contains **more self-correcting honesty than most production medical AI**. `PROJECT_STATUS.md` uses a four-level status legend and openly marks components as proxies. `scientific_audit.md` concludes against the project's own headline claim. `mimic/features.py` refuses to fabricate 12 clinical scores. `mimic/tasks.py` registers six tasks as unavailable with reasons. Multiple modules carry explicit "honest caveat" sections.

The problem is not deception. It is that the **README and the code have drifted apart**, and the README is what a judge reads.

---

# PHASE 15 — What is missing?

Ranked by importance. Difficulty: S (hours), M (days), L (weeks), XL (months).

| # | Missing | Importance | Difficulty | Impact |
|---|---|---|---|---|
| 1 | Fusion trained on the real vision distribution | **Critical** | M | Everything downstream becomes meaningful |
| 2 | Guarded posterior actually reaching the report | **Critical** | **S** | Removes a contradictory-diagnosis defect |
| 3 | Adequate vision training (15+ epochs, full data) | **Critical** | M | AUROC 0.67 → ~0.80+ |
| 4 | Full-resolution image path (drop the 64×64 bottleneck) | **Critical** | **S** | Largest single lever on nodule/pneumothorax |
| 5 | Promote the better checkpoint | **Critical** | **S (minutes)** | +12.9 % macro AUROC free |
| 6 | Fix Mondrian quantile saturation | High | **S** | Removes spurious malignancy from most reports |
| 7 | Per-finding threshold + temperature at serving | High | S | ECE 0.337 → ~0.05 |
| 8 | Honest README aligned with `scientific_audit.md` | High | **S** | Removes the biggest cross-examination risk |
| 9 | Authentication + authorization | High | M | Any deployment prerequisite |
| 10 | Wire ACI into `SafetyEngine` | High | S | Makes the coverage claim real |
| 11 | Reasoning posterior feeding the impression | High | M | Makes multimodal reasoning matter |
| 12 | Held-out test set separate from validation | High | S | Removes optimistic bias |
| 13 | Labeler validation against CheXpert labels | High | M | Quantifies the label-noise floor |
| 14 | Remove horizontal-flip augmentation | High | **S** | Removes a medically wrong prior |
| 15 | Request batching at serving | Medium | M | 26 → 618 img/s |
| 16 | Move inference off the event loop | Medium | S | Concurrency |
| 17 | Fix `RecommendEngine` thread safety | Medium | **S** | Removes a real race |
| 18 | CI pipeline | Medium | S | Prevents regression |
| 19 | Dockerfile + deployment manifest | Medium | M | Reproducibility |
| 20 | Uncertain-label handling (CheXpert −1) | Medium | M | Rare-finding recall |
| 21 | View conditioning (AP/PA/lateral) | Medium | M | Cardiomegaly confounding |
| 22 | External validation set (CheXpert/NIH/PadChest) | Medium | L | Generalisation evidence |
| 23 | Encryption at rest + PHI handling | Medium | M | Regulatory prerequisite |
| 24 | Localisation ground truth + faithfulness metrics | Medium | L | Explainability validation |
| 25 | Reader study | Low (now) | XL | Regulatory |
| 26 | Quantum services Q2–Q6 | Low | XL | Currently documentation |
| 27 | Prospective clinical validation | Low (now) | XL | Regulatory |

---

# PHASE 16 — Improvement roadmap

## Tier 0 — Do this week (highest ROI in the repository)

Ordered strictly by impact-per-hour.

**1. Promote the better checkpoint — 5 minutes, +12.9 % macro AUROC.**
```bash
cp artifacts/retrain_v2/best_model.pt artifacts/best_model.pt
```
Then re-run `py -m aura_cli evaluate` to regenerate the model card.

**2. Fix the conflict-guard/safety disconnect — ~1 hour, removes a contradictory-diagnosis defect.**

The interface passes a model where it should pass a posterior. In `services/safety/engine.py`, accept an optional pre-computed posterior:

```python
def assess(self, study_id, x, fusion_model, posterior=None):
    logits = (np.log(np.clip(posterior, 1e-12, 1.0)) if posterior is not None
              else fusion_model.logits(x))
```

and in `gateway/pipeline.py`:

```python
fusion = self.fusion.fuse_vector(x, study_id=study.study_id)
p_guarded = np.array([fusion.posterior[d] for d in DIAGNOSES]) if fusion.fallback_triggered else None
safety = self.safety.assess(study.study_id, x, self.fusion.model, posterior=p_guarded)
```

**3. Remove the 64×64 bottleneck — ~30 minutes, largest single lever on small findings.**

In `gateway/app.py:225`, change `study_from_cxr(tmp_path)` to `study_from_cxr(tmp_path, grid=224)` to match `services/inference/predict.py`. Better still, make `grid` a setting and use proper area-averaging instead of `linspace` index selection:

```python
import cv2
small = cv2.resize(full, (grid, grid), interpolation=cv2.INTER_AREA)
```

Also correct the docstring, which currently claims the full image reaches the CNN.

**4. Fix the Mondrian quantile saturation — ~20 minutes.**

`services/safety/uncertainty.py:144`:
```python
def _quantile_hi(scores, coverage):
    n = len(scores)
    if n == 0: return 1.0
    level = np.ceil((n + 1) * coverage) / n
    if level > 1.0:            # too few calibration points for this class
        return None            # signal: fall back to the marginal threshold
    return float(np.quantile(scores, level, method="higher"))
```
Raise the `m.sum() >= 10` gate to `>= 1/(1-coverage) + 1` (i.e. ≥ 20 at 90 % coverage), which is the standard requirement for a valid conformal quantile.

**5. Rewrite the README table — ~1 hour, removes the biggest presentation risk.**

Replace the quantum-vs-classical table with the fairly-calibrated numbers from `scientific_audit.md`, and lead with what *is* defensible: a working conformal guarantee, an abstention policy, an intake gate, and a self-audit that falsified the team's own hypothesis. That last point is a genuine strength — present it as one.

**6. Remove `HorizontalFlip` from the training augmentation — 1 line.**

## Tier 1 — Next two weeks

**7. Retrain fusion on real evidence vectors.** The root fix for F1. In `ml/training/dataset.py`:
```python
def build_evidence_dataset(samples, vision=None):
    vision = vision or VisionEngine.load()   # was: VisionEngine()
```
Then build the fusion training set from **real MIMIC studies** with report-derived diagnoses (`mimic.patient.iter_patients` already yields exactly this), not synthetic ones. Refit T, q̂, Mondrian, and OOD statistics on the same real distribution. This makes every calibration number downstream meaningful for the first time.

**8. Train the vision model properly.** 15+ epochs, `--num-workers 4`, pre-resized 256×256 JPEG cache to kill the decode bottleneck, per-study labels. Expected macro AUROC ~0.80–0.83 based on the epoch-1 trajectory (0.7848 → 0.7859 still rising).

**9. Apply the per-finding calibration that already exists.** `artifacts/vision_calibration_perfinding.json` is computed and ignored. Load it in `VisionModel.score_findings` and divide logits by the per-finding temperature.

**10. Carve a real test set.** Hold out 20 % of the MIMIC validate split, never touch it during model selection, report on it once.

**11. Wire ACI.** `SafetyEngine.__init__` loads `store.load_aci_state()`; use `aci.qhat` in place of `cal.conformal_qhat`. Makes the coverage claim survive distribution shift, which is the entire point of the module.

**12. Fix thread safety.** Make `_panel_evoi` a local threaded through `_greedy_panel` rather than instance state. Convert the two `async def` study endpoints to `def` (so FastAPI runs them in the threadpool) or wrap the pipeline in `run_in_threadpool`.

## Tier 2 — Toward each target

### Publication-quality research
- Validate the report labeler against CheXpert ground truth; publish precision/recall per concept.
- External validation on CheXpert, NIH ChestX-ray14, or PadChest.
- Power analysis for the quantum comparison. At the observed effect size (3 discordant of 100), detecting a 3 % accuracy difference at 80 % power needs roughly **n ≈ 1,700**, not 100. Run it at that scale or withdraw the claim.
- Ablate the components honestly: conformal vs no conformal, VQC vs learnable vs PoE, TV vs no TV.
- Explanation faithfulness: deletion/insertion curves, pointing game against any available localisation labels.
- The genuinely publishable contribution is not quantum. It is **"an offline clinical copilot with layered abstention and a self-falsifying audit harness"** — the negative result is the interesting result.

### Production-grade software
- CI: pytest + ruff + mypy on every push.
- Dockerfile with a CUDA base; pin dependencies with a lockfile.
- Replace SQLite with PostgreSQL; move images out of the JSON blob to object storage with a reference.
- Structured logging, Prometheus metrics, OpenTelemetry traces.
- Request batching with a queue; move inference to a worker process.
- Delete `projection.py`, `make_reuploading_qnode`, `_probe_audit.py`, `test_image.py`, `_final_test_result.txt`, and the duplicated optimizer states, or wire them in.

### Hospital-ready
- OIDC/SAML authentication, RBAC, per-user audit.
- Encryption at rest and in transit; PHI de-identification at intake.
- DICOM C-STORE / DICOMweb integration; HL7 FHIR `DiagnosticReport` output.
- A model-recall mechanism and immutable report versioning.
- Shadow-mode deployment: run silently alongside radiologists for 3–6 months, measure agreement, never surface output.

### SIH-winning
- Do Tier 0 in full. It is a week of work and moves the honest headline metrics substantially.
- Rebuild the pitch around **"we built a system that told us our own hypothesis was wrong, and we published it"**. Show `claim_verdicts.json` on a slide. Judges reward intellectual honesty far more than an unverifiable quantum claim, and it is unattackable.
- Lead the demo with the abstention path: upload a non-radiograph, get a named rejection; upload an ambiguous film, get an abstention with a reason. That demonstrates the actual product.

### Startup-ready
- Narrow to one finding with a real clinical buyer. Pneumothorax triage on ICU portables is the classic wedge — but only after sensitivity moves from 0.14 to >0.95, which is the whole engineering problem.
- The defensible moat is **offline + calibrated + abstaining**, which suits low-connectivity district hospitals. That is a real market and a real differentiator.
- Budget 18–24 months and a clinical partner for regulatory work. Class IIa/IIb is not a side quest.

## ROI ranking

| Rank | Action | Effort | Impact |
|---|---|---|---|
| 1 | Promote `retrain_v2` checkpoint | 5 min | +12.9 % AUROC |
| 2 | `grid=224` on upload | 30 min | Large on small findings |
| 3 | Remove HorizontalFlip | 1 min | Removes wrong prior |
| 4 | Fix Mondrian saturation | 20 min | Removes spurious malignancy |
| 5 | Wire the conflict guard | 1 hr | Removes contradictory diagnoses |
| 6 | Honest README | 1 hr | Removes the top judge risk |
| 7 | Apply existing per-finding calibration | 2 hr | ECE 0.34 → ~0.05 |
| 8 | Retrain fusion on real distribution | 2–3 d | Makes the system coherent |
| 9 | Train vision 15 epochs with workers | 1 d compute | AUROC → ~0.80 |
| 10 | Wire ACI | 3 hr | Makes coverage real |

**Items 1–7 total under one working day and would move the project from "impressive but broken" to "modest but sound."**

---

# PHASE 17 — Explain it like I'm the creator

You built this months ago. Here is your own system, from scratch.

## 17.1 The mental model

You built a **pipeline of nine independent engines** connected by **Pydantic contracts**. No engine imports another. The only thing that knows about all of them is `gateway/pipeline.py`. That is why you can replace the vision model without touching fusion, and why extracting an engine into its own service would be a deployment change rather than a rewrite.

The core idea you were chasing: **a diagnosis without a calibrated uncertainty is not a clinical product.** So you built layers of doubt around the prediction — conformal sets, ensembles, OOD, abstention — and made the *doubt* the deliverable.

## 17.2 Walk the modules

**`schemas/`** — your vocabulary. `Finding` (7 observations) and `Diagnosis` (6 interpretations) are deliberately separate: the vision model sees *findings*, fusion infers *diagnoses*. `contracts.py` is the flow diagram; read its docstring first when you come back to this.

**`common/config.py`** — one `Settings` dataclass, three-level override: dataclass defaults → `[tool.aura]` in `pyproject.toml` → `AURA_*` env vars. It is `@lru_cache`'d, so env changes need a restart.

**`ml/data.py`** — your synthetic world. Procedural 64×64 thoraxes with pathology planted in anatomical regions. You built it so every stage would have ground truth — that was the right instinct. But it became the *only* thing fusion ever trained on, and that is the deepest problem in the project.

**`services/vision/`** — three-tier fallback: `VisionModel` (your trained DenseNet) → `CXRBackbone` (torchxrayvision/timm) → numpy feature model → hard-coded heuristic. Because `best_model.pt` exists, tier 1 always wins and tiers 2–3 are dead code in practice.

**`services/fusion/evidence.py`** — the bottleneck. Seven finding probabilities plus one composite `prior_risk` scalar = 8 numbers. **Everything the CNN saw is compressed to 8 floats here.** Eight because that is `n_qubits`. This is the most consequential design decision in the system and it deserves revisiting: the 1024-d embedding you compute is used only for similarity search.

**`services/fusion/quantum.py`** — the VQC. Angle-encode 8 channels as RY(πxᵢ), three layers of (RY, RZ) + CNOT ring, read `⟨Zᵢ⟩`, feed a 6×8 linear head. 102 trainable parameters. At serving it is analytic (no shots); the `posterior_std` is a seeded Gaussian MC simulating what shot noise would look like.

**`services/fusion/conflict.py`** — your Wasserstein tie-breaker. Embed the six diagnoses on a severity axis, compute EMD between the VQC and PoE posteriors, and fall back to PoE when they disagree by more than a dynamic threshold. **Beautiful idea, correctly implemented, and completely disconnected.** The safety engine recomputes the posterior from the quantum model and never sees your decision. Fixing this is a one-hour change and it is the single most important thing to do.

**`services/safety/engine.py`** — temperature-scale, decompose uncertainty with the ensemble, build a Mondrian conformal set, score OOD energy, then run a four-reason abstention cascade. This is the heart of your thesis.

**`services/safety/aci.py`** — your adaptive conformal inference. Read the docstring; it is the best writing in the repository and the maths is right. It updates q̂ online so coverage survives distribution shift without exchangeability. It writes to SQLite on every feedback event. Nothing reads it back.

**`services/recommend/engine.py`** — you replaced information-gain-in-bits with **EVOI in clinical-loss units**, because reducing entropy about a benign distinction is worthless while a small shift on a pneumothorax changes management. That reasoning is correct and it is one of the smartest calls in the project. Then `causal.py` handles redundancy between correlated tests via the MI chain rule with a causal gate.

**`services/reasoning/engine.py`** — eight guideline rules applying log-likelihood ratios to the imaging posterior. Real thresholds, real citations. Its `adjusted_posterior` is computed, stored, displayed in the differential — and never used for the impression.

**`services/report/`** — deterministic templating with a `grounding` map from each section to the evidence nodes behind it. No LLM. That is why hallucination risk is low.

**`gateway/`** — FastAPI, 11 endpoints, SQLite through a repository boundary, audit middleware. No auth.

**`mimic/`** — your real-corpus layer. `loaders` streams the CSVs in chunks with schema validation; `cleaning` dedupes and flags outliers; `labeling` is your CheXpert-style regex labeler with forward-scoped negation; `timeline` orders studies; `patient` is the unified object that emits a `StudyInput`. Read `labeling.py` carefully — everything the vision model knows comes from those ten regexes.

## 17.3 The three design decisions to re-examine

1. **Eight evidence channels.** You chose 8 because that is a tractable qubit count. But that means a 1024-d deep embedding is thrown away before the diagnosis is made. `projection.py` — which you wrote and never wired — is exactly the fix: a trainable 1024→8 bottleneck. Wire it in, or widen the channel set.

2. **Synthetic fusion training.** It gave you ground truth, which was the right reason. But it created a train/serve gap you can measure in one command. `mimic.patient.iter_patients` already gives you real studies with report-derived diagnoses — the replacement dataset already exists in your own codebase.

3. **Passing a model instead of a posterior.** `SafetyEngine.assess(study_id, x, fusion_model)` takes a *model* and recomputes. That single interface choice is what silently disconnects your conflict guard. Passing the posterior would have made F2 impossible.

## 17.4 Every dependency

Runtime: `numpy`, `scipy`, `scikit-learn`, `pennylane`, `fastapi`, `uvicorn`, `pydantic`, `SQLAlchemy`, `aiosqlite`, `pillow`, `python-multipart`, `jinja2`, `matplotlib`.
Vision (optional but actually required for the served path): `torch`, `torchvision`, `opencv-python`, `albumentations`, `tensorboard`, `pydicom`.
Dev: `pytest`, `httpx`.

## 17.5 How to run it

```bash
cd E:\AURA\aura-main\aura          # you MUST be in this directory
py -m aura_cli serve               # gateway on :8000, trains first if needed
py -m aura_cli train 700           # retrain vision detectors + all 4 fusion backends
py -m aura_cli bench 500           # quantum-vs-classical (note: unfair temperature)
py -m aura_cli predict --image sample.jpg
py -m aura_cli evaluate --limit 500 --calibrate
py -m aura_cli benchmark --iters 50
python -m ml.vision_cxr.train --epochs 15 --num-workers 4 --out-dir artifacts/run3
python ../audit_all.py             # the reproducible scientific audit
py -m pytest tests/ -q             # 117 tests
```

## 17.6 Where to look when something breaks

| Symptom | Look at |
|---|---|
| Everything abstains | `artifacts/safety.npz` OOD stats; `SafetyEngine.assess` cascade |
| Upload rejected | `xray_gate.validate_cxr` → the `checks` dict in the 422 body |
| Diagnosis looks wrong | `fusion.posterior` vs `safety.top` — they can differ (F2) |
| Confidence set always names malignancy | `artifacts/conformal_mondrian.npy` index 4 (F7) |
| Vision probabilities all high | `pos_weight` clipping at 20 + undertrained checkpoint |
| Saliency looks like noise | 64×64 grid upsampled from a full-res film (F5) |
| Reasoning has no effect | `multimodal` is `None` for real studies (F10) |
| Buttons do nothing | Stale cached JS — bump the `?v=` query strings |
| Mojibake in reports | `aura_cli._utf8_stdout()`; Windows cp1252 |

---

## Closing assessment

AURA is a **strong architectural achievement with a broken safety spine and an overstated headline**.

The engineering craft is real: contract-driven modularity that actually holds, docstrings that derive their own mathematics and cite sources, self-documented caveats, an honest status report, and — most unusually — a reproducible audit harness that the team built and then published against their own hypothesis. That last fact says more about the people involved than any metric here.

The failures are also real, and they cluster in one place: **things that were built correctly but never connected.** The conflict guard, adaptive conformal inference, the joint projection, the re-uploading ansatz, the per-finding calibration, the better checkpoint, the reasoning posterior — seven completed components sitting one wire away from mattering. That is a recoverable failure mode, and an unusually cheap one: Tier 0 of the roadmap is under a day of work.

The one thing that must change before this is shown to anyone technical is the README. `scientific_audit.md` already contains the honest version. Publish that one.

---

*Report produced 2026-07-20 by direct source reading, artifact inspection, and live execution against `E:\AURA\aura-main` at git `cf8356b`. All measured values are reproducible with the commands shown.*
