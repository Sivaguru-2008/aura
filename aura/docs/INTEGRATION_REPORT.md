# AURA — Integration Report (Step 1)

**Date:** 2026-07-18
**Scope:** Repository inspection prior to the production-evaluation hardening pass.
**Rule followed:** No API/route/schema redesign; minimum-footprint additions only.

---

## 1. Repository topology

The importable Python package lives at `aura/aura/` (the outer `aura/` is a
container). It is a namespace-style layout — every top-level directory is
importable (`pyproject.toml` sets `pythonpath = ["."]`). Commands run from
`aura/aura/`.

```
aura/aura/
├── aura_cli.py              # CLI entry point (train / train-cnn / bench / serve / demo)
├── common/                  # config, eventbus, mathx
├── schemas/                 # clinical vocab (Finding/Diagnosis) + pydantic contracts
├── services/                # the engines (vision, fusion, safety, explain, recommend,
│                            #   reasoning, report, memory, models)
├── ml/                      # data, training, evaluation, vision_cxr (DenseNet stack)
├── mimic/                   # real MIMIC-CXR loaders, labeling, evaluation, uncertainty…
├── gateway/                 # FastAPI app + pipeline orchestration + storage/seed
├── apps/web/                # dashboard SPA (static)
├── artifacts/               # trained weights + calibration + benchmark + plots
└── tests/                   # pytest suite (100 tests, all green at baseline)
```

## 2. Component inventory (what was found, where)

| Capability | Location | Status at inspection |
|---|---|---|
| **VisionEngine** | `services/vision/engine.py` | Auto-loads `artifacts/best_model.pt` (production DenseNet121) via `ml/vision_cxr/inference.py:VisionModel`; falls back to numpy feature model. `score_findings` is a pure callable. |
| **Production CNN** | `ml/vision_cxr/{model,inference}.py` | `DenseNet121CXR` — grayscale, 7-finding multi-label head. `best_model.pt` present (arch=densenet121, epoch=1, best_metric≈0.696). Loads on **CUDA** (torch 2.11+cu128). |
| **Reasoning Engine** | `services/reasoning/engine.py` | `ClinicalReasoner` — fuses imaging posterior with labs/symptoms/history + guideline citations; emits differential. |
| **Safety Engine** | `services/safety/engine.py` + `calibration.py` + `uncertainty.py` | Temperature scaling, marginal + Mondrian conformal, deep-ensemble / perturbation epistemic split, energy-OOD, abstention. |
| **Recommend Engine** | `services/recommend/engine.py` | EVOI-in-loss-units test recommendation + greedy panel selection. |
| **Explainability** | `services/explain/{engine,methods}.py` | Occlusion (model-agnostic) **plus** Grad-CAM, Grad-CAM++, Integrated Gradients, SmoothGrad on the CNN. |
| **Report Generator** | `services/report/engine.py` | `ReportDraft`: findings/impression/differential/confidence/recommendation, each grounded to evidence nodes. |
| **Dashboard** | `apps/web/` + `gateway/app.py` routes `/`, `/app`, `/static` | Static SPA reads `explanation.saliency` + `saliency_methods`. |
| **Gateway** | `gateway/app.py` (`/v1/*`) + `gateway/pipeline.py` | FastAPI. Pipeline wires all engines per study. |
| **Evaluation scripts** | `ml/evaluation/{metrics,benchmark}.py`, `mimic/evaluation.py` | Multiclass diagnosis metrics + multilabel/binary evaluators + quantum-vs-classical benchmark. |
| **Testing framework** | `tests/` + `[tool.pytest.ini_options]` | 100 tests, pass in ~93s. |
| **Real dataset** | `datasets/…/mimic-cxr-dataset/versions/2/` | `mimic_cxr_aug_validate.csv` (500 patients) + 261k JPGs under `official_data_iccv_final/files/`. Labels derived from report text by `mimic/labeling.py` (CheXpert-style). |

## 3. Key integration facts that shaped the additions

1. **The DenseNet is already the live backbone.** `VisionEngine.load()` checks
   `artifacts/best_model.pt` first, so the whole pipeline already runs on the
   trained model. No retraining, no wiring changes required.
2. **The vision task is multi-label** (7 findings), and validation labels come
   from `mimic/labeling.py` applied per patient (the exact convention
   `ml/vision_cxr/dataset.py:load_mimic_samples` used for training). The
   inference validation suite reuses that loader so labels match training.
3. **`ml/evaluation/metrics.py:evaluate` is for the multiclass *diagnosis*
   (fusion) head**, not the vision head. Vision needs the *multilabel*
   evaluator (`mimic/evaluation.py:evaluate_multilabel`) — reused, not
   reinvented — extended with ROC/PR/calibration curves and bootstrap CIs.
4. **`artifacts/benchmark.json` is read by the admin API** (`/v1/admin/safety`)
   and produced by the quantum-vs-classical `bench` command. To avoid clobbering
   it, the new performance benchmark writes `artifacts/performance/benchmark.json`.
5. **`ReportDraft` and all pydantic contracts are frozen.** The full structured
   clinical report (Step 5) is produced by a *new renderer* that reads the
   existing `CaseBundle`; the schema is untouched.
6. **The explain suite already has Grad-CAM++/IG.** Only **Score-CAM** and the
   **overlay/PNG/HTML artifact writers** were missing.

## 4. Files added / modified (minimum footprint)

**Added (new modules — no existing behavior touched):**
- `services/explain/scorecam.py` — Score-CAM.
- `services/explain/overlays.py` — heatmap overlays, bounding-box localization, PNG + high-res + HTML writers.
- `services/report/clinical_report.py` — full structured clinical report renderer (reads `CaseBundle`).
- `ml/evaluation/clinical_eval.py` — MIMIC-CXR inference validation suite (all metrics + plots + bootstrap CIs).
- `ml/evaluation/vision_calibration.py` — temperature scaling, MC-dropout, conformal, reliability/histogram/coverage plots.
- `ml/evaluation/perf_benchmark.py` — latency/memory/throughput/AMP benchmark.
- `services/inference/predict.py` — single-image predict orchestration (load → pipeline → report → overlay → latency).
- `tests/test_production_*.py` — regression tests for the new surface.

**Modified (additive only):**
- `aura_cli.py` — new `predict` / `evaluate` / `explain` / `benchmark` subcommands; every existing command preserved.
- `services/explain/methods.py` — register Score-CAM in `all_methods` (behind the existing per-method try/except).

**Explicitly NOT changed:** FastAPI routes, request/response schemas, pydantic
contracts, DenseNet weights, existing engine logic, dashboard routes.

## 5. Baseline verification (before changes)

- `pytest` → **100 passed** in ~93 s.
- `best_model.pt` loads on CUDA; `score_findings` returns all 7 findings.
- Full `Pipeline.run` on a sample produces vision → fusion → safety → explain
  (grad_cam/grad_cam++/integrated_gradients/smoothgrad) → recommend → reasoning
  → report end-to-end with `model_version = vision-cxr-densenet121-production`.
