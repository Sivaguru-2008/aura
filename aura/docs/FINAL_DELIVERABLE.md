# AURA — Production-Evaluation Pass: Final Deliverable

A polish-and-validate pass over the existing AURA repository. **No** FastAPI route,
frontend route, pydantic schema, or engine contract was changed; the DenseNet was
**not** retrained. All new capability is additive and reuses existing modules.

---

## 1. Files changed

### Added (new files — nothing existing touched)
| File | Purpose |
|---|---|
| `services/explain/scorecam.py` | Gradient-free **Score-CAM** for the CNN backbone |
| `services/explain/overlays.py` | Heatmap overlays, **bounding-box localization**, evidence bars, PNG / high-res / **HTML** writers |
| `services/report/clinical_report.py` | Full **structured clinical report** renderer (reads `CaseBundle`; md/html/json) |
| `services/inference/__init__.py`, `services/inference/predict.py` | Single-image **predict** orchestration (load → pipeline → report → overlays → latency) |
| `ml/evaluation/clinical_eval.py` | **Inference validation suite** on MIMIC-CXR (all metrics + plots + bootstrap CIs) |
| `ml/evaluation/vision_calibration.py` | **Calibration suite**: temperature scaling, MC-dropout, TTA, conformal, reliability/coverage |
| `ml/evaluation/perf_benchmark.py` | **Performance benchmark** (CPU/GPU latency, memory, throughput, AMP) |
| `tests/test_production_explain.py` | Regression tests: Score-CAM, overlays, boxes, HTML |
| `tests/test_production_report.py` | Regression tests: report has all 13 sections + renderers |
| `tests/test_production_predict.py` | Regression tests: end-to-end predict + artifacts |
| `tests/test_production_eval.py` | Regression tests: eval/calibration/perf (MIMIC-guarded) |
| `docs/INTEGRATION_REPORT.md`, `DEPLOYMENT.md`, `MODEL_CARD.md`, `EVALUATION_REPORT.md`, `PERFORMANCE_REPORT.md` | Step 1 & 9 documentation |
| `sample.jpg` (working dir) | A real MIMIC image so `predict --image sample.jpg` works out of the box |

### Modified (additive only)
| File | Change | Why |
|---|---|---|
| `aura_cli.py` | Added `predict` / `evaluate` / `explain` / `benchmark` / `calibrate` subcommands via argparse; UTF-8 stdout; **all legacy commands (`train`, `train-cnn`, `bench`, `serve`, `demo`) unchanged** | Step 7 |
| `services/explain/methods.py` | `all_methods(..., include_scorecam=False)` — registers Score-CAM behind an opt-in flag | Step 3 without changing serve latency |
| `requirements.txt` | Added `matplotlib` (plots/overlays); noted optional `psutil` | New plotting dependency |

## 2. Why each change (design rationale)

- **Reuse over reinvention.** Metrics reuse `mimic/evaluation.py` +
  `services/safety/uncertainty.py`; explainability builds on the existing Grad-CAM++/IG
  in `services/explain/methods.py`; predict drives the existing `gateway/pipeline.py`.
- **Schemas frozen.** The full clinical report (Step 5) is a *renderer over the existing
  `CaseBundle`*, not a schema change — so no contract or API is affected.
- **Vision task is multi-label.** Validation labels come from the same report labeler
  (`mimic/labeling.py`) via `load_mimic_samples`, so metrics measure the model, not a
  label mismatch. The multiclass `ml/evaluation/metrics.py` remains the fusion path.
- **Non-clobbering benchmark.** Performance results go to
  `artifacts/performance/benchmark.json`; the fusion `artifacts/benchmark.json` read by
  `/v1/admin/safety` is left intact.
- **Windows safety.** The CLI reconfigures stdout to UTF-8 so reports with clinical
  glyphs never crash cp1252 consoles.

## 3. Validation results

**Regression:** `pytest` → **117 passed** (100 existing + 17 new), 0 regressions.
`python -m compileall` clean; all new modules import. Legacy `bench`, `serve`
dispatch verified.

**Inference validation (full MIMIC validation split, n = 2,099 images):**
macro **AUROC 0.702** (95% CI 0.688–0.716), AUPRC 0.558, F1 0.547, sensitivity 0.584,
specificity 0.719, Brier 0.202, ECE 0.147. Micro F1 0.623. Strong on
opacity/effusion/cardiomegaly/consolidation/hyperinflation (AUROC 0.72–0.78); weak on
nodule (0.485) and pneumothorax (0.638). Matches the checkpoint's `best_metric≈0.696`.
→ `artifacts/evaluation/` (metrics.json + 5 plots).

**Calibration:** mean ECE **0.145 → 0.130** after per-finding temperature scaling;
conformal **coverage 0.906** at target 0.90 (mean set size 1.53). MC-dropout degenerate
(no dropout layers) → TTA epistemic proxy (mean std 0.040). → `artifacts/calibration/`.

**Performance (RTX 5050 Laptop, torch 2.11+cu128):** GPU 29 ms/img (34 img/s),
CPU 83 ms/img; peak batch throughput **618 img/s** (batch 32); peak GPU memory 693 MB;
AMP 0.82× (honestly slower on this small workload). → `artifacts/performance/benchmark.json`.

**Success criteria:** `python aura_cli.py predict --image sample.jpg` returns vision
findings, calibrated confidence + conformal set, differential, full clinical report,
evidence recommendations, saliency heatmaps (Grad-CAM/Grad-CAM++/IG/SmoothGrad/Score-CAM
overlays + HTML), and inference latency. `python aura_cli.py evaluate` scores the whole
validation set and writes every metric + visualization.

## 4. Remaining limitations

1. **Diagnosis fusion is a synthetic-trained prototype.** On real multi-finding images
   (e.g. `sample.jpg`, which has effusion + consolidation + cardiomegaly) the fusion can
   emit a confident but clinically wrong diagnosis (COPD in that case). The **validated
   output is the vision findings**, not the fused diagnosis. Per the constraints, fusion
   was neither modified nor retrained.
2. **1-epoch checkpoint.** Real but modest skill (macro AUROC 0.70); **near-chance on
   nodule** and low sensitivity on pneumothorax — do not use to exclude either.
3. **Label quality ceiling.** Rule-based, patient-level report labels bound achievable
   metrics and inflate rare-finding error.
4. **No subgroup/fairness audit** (age/sex/device) and no external-site validation.
5. **AMP** is not beneficial on this GPU/workload (measured, not assumed).

## 5. Recommended future work

1. **Retrain the fusion engine on real evidence** — pairs of (real DenseNet evidence
   vector → labeler diagnosis) — to close the synthetic→real domain gap; re-run
   `aura_cli bench`. Highest-impact item.
2. **Train the DenseNet longer** (more epochs, focal/class-balanced loss for
   nodule/pneumothorax) and re-validate; refresh `MODEL_CARD.md` from `metrics.json`.
3. **Better labels:** swap the rule labeler for the official CheXpert labeler or a
   fine-tuned report classifier; move to per-study (not per-patient) labels.
4. **Add dropout** to the classifier head so MC-dropout epistemic uncertainty is
   first-class, or train a true **deep ensemble** (the calibration suite already supports it).
5. **Fairness & external validation:** subgroup metrics + a second-site test set.
6. **CI gate:** wire `compileall` + `pytest` + a `predict` smoke into CI; add `ruff`/`mypy`.
