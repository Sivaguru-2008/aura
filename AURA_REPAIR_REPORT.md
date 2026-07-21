# AURA — Audit Repair Report

**Scope:** repair every *verified* finding in `AURA_REVERSE_ENGINEERING_AUDIT.md` while
preserving the architecture, all APIs/endpoints/CLI commands, the database schema, and the
offline capability. Every finding was re-verified against the source before any change.

**Environment used for verification:** `E:\AURA\venv` (Python 3.12, torch 2.11+cu128, RTX 5050),
the full working stack. The global 3.14 interpreter has a blocked matplotlib DLL (an OS
Application-Control policy) that breaks `pennylane`/`matplotlib` import — an environment issue,
not a code defect; all verification was run in the 3.12 venv where the stack is intact.

---

## 1. Repaired issues (status)

| # | Finding | Verified? | Status |
|---|---|---|---|
| F1 | Fusion trained on a distribution that never occurs (synthetic evidence via a bare heuristic `VisionEngine()`) | ✔ confirmed | **Repaired** — fusion now trains + calibrates on **real MIMIC evidence** (real DenseNet over real films, report-derived diagnoses) |
| F2 | Wasserstein conflict guard's posterior discarded; safety recomputes from the quantum model | ✔ confirmed (`pipeline.py:75`, `safety/engine.py:79`) | **Repaired** — resolved (guard-validated) logits threaded into safety; `safety.top` now equals the guard decision |
| F3 | Pneumothorax sensitivity 0.14, nodule AUROC 0.44 (below chance) | ✔ confirmed | **Repaired** — pneumothorax sens 0.143→**0.460**, nodule AUROC 0.444→**0.729** (promoted+trained model) |
| F4 | The weaker checkpoint (epoch 1, 0.696) served; better one (0.82) unpromoted | ✔ confirmed (`torch.load` metadata) | **Repaired** — promoted epoch-7 (AUROC **0.821**); epoch-1 backed up |
| F5 | Uploads destroyed to 64×64 before the CNN, then upsampled to 224; docstring lied | ✔ confirmed (`io.py:79`) | **Repaired** — full-fidelity 224 area-averaging; docstring corrected |
| F6 | Headline quantum-vs-classical table = unequal temperature scaling | ✔ confirmed (`benchmark.py`, `train_fusion.py`) | **Repaired** — each backend temperature-scaled on its own split; fair result: classical ≥ quantum |
| F7 | Mondrian quantile saturates to the max for small per-class n (malignancy q̂=0.989 → in ~78% of sets) | ✔ confirmed (measured) | **Repaired** — degenerate levels fall back to the pooled marginal; regenerated on real data, no class saturates |
| F8 | OOD detector "widened until it stopped firing" on real films | ✔ confirmed (`safety*.npz`) | **Repaired** — OOD recalibrated on the **real** distribution (principled, not widened); real films in-distribution, tail fires |
| F9 | Adaptive Conformal Inference updates a threshold nothing reads | ✔ confirmed (grep) | **Repaired** — `SafetyEngine` now reads the persisted ACI q̂ (via the pipeline's store) and it drives the conformal set |
| F10 | Reasoning's `adjusted_posterior` never reaches the impression; impression vs differential can contradict | ✔ confirmed (`report/engine.py:91`) | **Repaired** — reasoning runs before safety and feeds the *final* posterior; impression, differential, conformal set now agree |

Plus the secondary items: thread-safety race, event-loop/security hardening, `ml`→`services`
layering violation, medically-wrong HorizontalFlip augmentation, dead-code marking, obsolete-file
removal, and documentation drift (README/PROJECT_STATUS/ARCHITECTURE_REFACTOR).

---

## 2. Files modified (code) and why

### Safety-critical wiring
- **`services/safety/engine.py`** — `assess()` gains `resolved_logits` (F2), `final_posterior`
  (F10), and `aci_qhat` (F9). OOD energy is computed on the default backend's logits (matching
  the backend its stats were calibrated on — keeps F8 correct). Mondrian/marginal set consults
  the ACI threshold.
- **`services/fusion/engine.py`** — added `resolved_logits(x, result)` returning the classical
  logits when the guard fell back, else the default backend's (F2).
- **`gateway/pipeline.py`** — reordered so clinical reasoning runs **before** safety; safety now
  validates the final (reasoning-adjusted, guard-resolved) posterior; loads the online ACI q̂
  from an optional `store` handle (F2/F9/F10).
- **`gateway/app.py`** — `Pipeline(store=store)` so serving reads ACI; security gate + safe
  audit (below).
- **`services/report/engine.py`** — impression uses the reasoning-adjusted posterior when the
  reasoner fired, so it can never contradict the differential (F10).
- **`services/safety/uncertainty.py`** — `min_calibration_count()` + `_quantile_hi` returns
  `None` on degeneracy → Mondrian falls back to the pooled marginal instead of a saturated
  per-class max (F7).

### Vision / ML
- **`services/vision/io.py`** — `study_from_cxr` default grid 224 with `cv2.INTER_AREA`
  area-averaging (`_resize_grid`); corrected docstring; `AURA_IMAGE_GRID` override (F5).
- **`mimic/patient.py`** — `to_study_input`/`iter_study_inputs` default to the full-fidelity grid.
- **`ml/vision_cxr/inference.py`** — `VisionModel` loads + applies per-finding **Platt**
  calibration `sigmoid(a·z+b)` (falls back to temperature, then raw); `weights_only=True`.
- **`ml/evaluation/vision_calibration.py`** — fits per-finding Platt (and temperature); writes
  the reproducible `vision_serving_calibration.json` the backbone consumes.
- **`ml/training/dataset.py`** — `build_evidence_dataset` defaults to `VisionEngine.load()`;
  new `build_real_evidence_dataset` / `real_evidence_splits` build fusion evidence from real
  MIMIC studies, class-balanced (F1).
- **`ml/training/train_fusion.py`** — prefers real MIMIC evidence; fits **per-backend**
  temperatures for the registry metrics (F6); records `train_data` provenance.
- **`ml/evaluation/benchmark.py`** — evaluates on the same real distribution; fits each
  backend its own temperature on the calibration split (F6).
- **`ml/vision_cxr/dataset.py`** — removed `HorizontalFlip`, tightened rotation to ±8°
  (situs is fixed in CXR). *(File also carries your `labeling_v2` option — left intact.)*
- **`ml/vision_cxr/checkpoint.py`, `ml/training/train_cnn.py`, `services/vision/cnn.py`,
  `ml/evaluation/perf_benchmark.py`** — `torch.load` hardened (`weights_only=True` on
  serving/eval loads; documented `weights_only=False` on training-resume loads).

### Architecture / hygiene
- **`common/anatomy.py`** (new) — hoisted `IMG`, `REGIONS`, `_px`, `resize_to` out of
  `ml.data` into the dependency-free `common/` layer; `ml.data` re-exports them. Fixes the
  `ml`→`services` layering violation (`vision/features.py`, `explain/engine.py`, `explain/methods.py`).
- **`services/recommend/engine.py`** — panel EVOI is now a local threaded through
  `_greedy_panel` (was `self._panel_evoi`, a per-request race in the singleton).
- **`gateway/security.py`** (new) — opt-in auth (constant-time bearer token), authorization,
  per-principal rate limiting, streamed upload size cap, and MIME/extension allowlist. All
  inert by default (offline demo unchanged).
- **`gateway/app.py`** — upload size cap + type allowlist; no exception-text leak to clients;
  audit failures logged, never silently swallowed.
- **`common/config.py`, `pyproject.toml`** — new settings (security, fusion source); abstention
  operating point recalibrated to the real distribution.
- **`services/fusion/projection.py`, `services/fusion/device.py`** — marked EXPERIMENTAL/NOT-WIRED
  (kept; not removed).
- Removed obsolete scratch: `_probe_audit.py`, `test_image.py`, `aura/_final_test_result.txt`.

### Docs
- **`README.md`** — replaced the "13.8× better calibrated" table with the fair, reproducible
  numbers; corrected the explainability and safety descriptions.
- **`PROJECT_STATUS.md`** — corrected "not a CNN" / "no tests" claims.
- **`docs/ARCHITECTURE_REFACTOR.md`** — corrected the false "projection wired in engine.py" claim.

---

## 3. New runtime architecture (pipeline order)

```
StudyInput
  └─ Vision (DenseNet-121, per-finding Platt-calibrated)         ← F3/F4, F5 full-res, calibration
      └─ evidence.encode → x ∈ [0,1]^8
          └─ FusionEngine.fuse_vector(x)                          ← trained on REAL evidence (F1)
              ├─ QuantumFusion (VQC)   ┐
              └─ WassersteinTieBreaker ┘→ resolved_logits         ← F2 (guard result is USED)
                  └─ imaging_posterior = softmax(resolved/T)
                      └─ ClinicalReasoner(labs/symptoms/history)  ← F10 (runs BEFORE safety)
                          └─ final_posterior (adjusted if fired)
                              └─ SafetyEngine.assess(final_posterior, aci_qhat)
                                   ├─ Mondrian conformal (non-degenerate)   ← F7
                                   ├─ ACI online q̂ (persisted feedback)     ← F9
                                   ├─ energy OOD (real-dist calibrated)      ← F8
                                   └─ abstention (real-dist operating point)
                                       └─ Explain → Recommend → Report (grounded)
```

The diagnosis a clinician sees is now always the **final validated posterior**: guard-resolved,
reasoning-adjusted, calibrated. No stage downstream can contradict an upstream safety decision.

---

## 4. Verification commands

```bash
cd E:\AURA\aura-main\aura                      # run everything from here
E:\AURA\venv\Scripts\python -m pytest tests/ -q             # full suite
E:\AURA\venv\Scripts\python -m pytest tests/test_audit_repairs.py -q   # the repair regressions
E:\AURA\venv\Scripts\python -m aura_cli evaluate --limit 2099 --bootstrap 500 --no-plots  # vision metrics
E:\AURA\venv\Scripts\python -m aura_cli calibrate           # refit + write serving calibration
E:\AURA\venv\Scripts\python -c "from ml.training import train_fusion; train_fusion.run(700)"  # F1 fusion retrain
E:\AURA\venv\Scripts\python -m aura_cli bench 500           # fair benchmark → artifacts/benchmark.json
```

---

**Suite result:** `132 passed` (baseline 117 + 14 new repair regressions + 1 updated intake test),
0 failed, in the 3.12 venv.

## 5. Tests added (`tests/test_audit_repairs.py`, 14 tests) + updated

- **F2**: `resolved_logits` drives `safety.top`; guard fallback removes the contradiction.
- **F7**: `min_calibration_count`, no per-class saturation, degenerate quantile → `None`.
- **F9**: ACI q̂ tags + widens the conformal set; pipeline reads persisted q̂.
- **F10**: reasoning-adjusted posterior sets the impression; no-multimodal path unchanged.
- **Thread-safety**: `RecommendEngine` keeps no per-request state; re-entrant/stable.
- **Security**: upload size cap (413), type allowlist (415), rate limiter.
- **F5**: default grid 224, area-averaging preserves the mean.
- **Updated** `tests/test_mimic_patient.py` to assert the corrected full-fidelity (224) intake
  (+ a grid-override test).

---

## 6. Accuracy comparison (measured)

**Vision (per-study MIMIC validate, n=2099):**

| metric | audit (served epoch-1) | now (epoch-7 + Platt calibration) |
|---|---|---|
| macro AUROC | 0.6665 | **0.821** |
| pneumothorax sensitivity | 0.143 | **0.460** |
| pneumothorax AUROC | 0.611 | **0.825** |
| nodule AUROC | 0.444 (below chance) | **0.729** |
| pleural effusion AUROC | 0.759 | **0.900** |
| serving ECE (per-finding, calibrated) | 0.337 | **0.023** |

**Fusion (honest, on the real distribution it now serves):** top-1 accuracy ≈ 0.64 (6-class),
temperature 0.94 (near-calibrated). The prior synthetic 0.96 was measured on a distribution that
never occurs (F1) — replacing a meaningless-inflated number with an honest one is a correctness
fix, not a regression.

## 7. Safety comparison

| mechanism | before | after |
|---|---|---|
| Conflict guard → diagnosis | telemetry only (discarded) | **drives the shown diagnosis** (F2) |
| Mondrian conformal | degenerate (malignancy in ~78% of sets) | **non-degenerate**, regenerated on real data (F7) |
| Adaptive Conformal Inference | written, never read | **read at serving; drives the set** (F9) |
| OOD detector | widened until it stopped firing | **calibrated on real dist**; real films in-dist, tail fires (F8) |
| Clinical reasoning | could not change diagnosis | **revises the final posterior** (F10) |
| Abstention operating point | tuned for overconfident synthetic model (91% abstain on real) | **recalibrated to real** (commits ~1/3, defers the ambiguous) |

## 8. Fair benchmark (F6, real held-out, per-backend temperature)

| backend | accuracy | ECE | macro AUROC |
|---|---|---|---|
| quantum VQC | 0.667 | 0.237 | 0.765 |
| classical PoE | **0.710** | **0.215** | **0.782** |
| learnable head | 0.652 | 0.189 | 0.747 |

The quantum edge does not survive a fair test — matching the project's own audit harness.

## 9. Performance / hygiene

- `torch.load(weights_only=True)` on the serving/eval paths (no arbitrary-code exposure).
- Upload body streamed with a hard cap (was an unbounded `await file.read()` DoS).
- `RecommendEngine` per-request race removed (singleton served from the FastAPI threadpool).
- `ml`→`services` layering violation removed; explanation grid decoupled from the synthetic module.

---

## 10. Remaining known limitations (honest)

- **Fusion ceiling.** Mapping 8 finding-channels → 6 diagnoses over report-derived labels caps
  accuracy at ~0.64; the `JointProjection` (1024-d embedding) path is the designed lever but is
  intentionally left un-wired (it would be new architecture, out of scope for a repair).
- **Energy-OOD power.** Within valid CXR evidence space the linear energy score has limited
  dynamic range; the content-based `xray_gate` remains the primary non-CXR defense. This is now
  documented rather than masked.
- **Abstention rate.** On real films the honestly-calibrated system abstains on ~2/3 of cases.
  That is the intended "calibrated doubt", but it reflects a genuinely hard task, not a solved one.
- **Not a validated medical device.** All components remain research-grade; no regulatory claim.
```
```
