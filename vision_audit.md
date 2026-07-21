# AURA Vision Pipeline — Scientific Audit (DenseNet121 / MIMIC-CXR)

**Scope:** the image → finding → report → explainability path
(`ml/vision_cxr/*`, `services/vision/*`, `services/explain/*`, `services/report/*`,
`ml/evaluation/clinical_eval.py`). This is distinct from the earlier *fusion-stack*
audit (`scientific_audit.md`).

**Discipline:** every claim below is tagged **[VERIFIED]** (backed by code lines, the
trained checkpoint, or a probe I ran), or **[NEEDS EXPERIMENT]** (a hypothesis that
requires a training/inference run I have not executed). No result is invented.

**Primary evidence artifacts**
- `aura/artifacts/best_model.pt` → `torch.load` metadata: `epoch=1`, `best_metric=0.6962`, `arch=densenet121`, 7.0M params.
- `aura/artifacts/history.csv` → 2 rows (epoch 0, 1); `val_loss 1.336 → 1.186` (still falling).
- `aura/artifacts/evaluation/metrics.json` → n=2099, macro AUROC **0.7019**, macro ECE **0.1465**; nodule AUROC **0.4848** (CI 0.456–0.515).
- Probe (`venv` python) on the raw CSVs: train **64,586** subjects, val **500** subjects, **subject overlap = 0**; val = 2,991 images (mean **5.98**, max **114** images/subject); **48.2%** of val subjects have >1 report.

---

## Executive summary

The vision model that AURA **actually serves** is a DenseNet121 trained for **2 epochs**
against **cross-study-smeared labels**, and at inference it is fed a **64×64 upscaled**
image although it was trained at 224 from full resolution. A genuinely pretrained
MIMIC model (`torchxrayvision densenet121-res224-mimic_ch`) is present in the codebase
but sits **unused** behind the weak checkpoint. The reported **macro AUROC 0.70** is an
*optimistic* number (clean preprocessing, label-noisy denominator, model-selection set)
— the deployed number is lower and currently unmeasured.

None of these are architecture problems. The DenseNet121 adaptation itself is done well
(luminance conv0, TV feature regulariser, AMP, grad-clip). The losses are in the
**data, the training budget, the serving preprocessing, and the evaluation rigor**.

Highest-ROI fixes, **in order (revised after running E0/E1 below)**: **(1) de-smear
labels to per-study — this is the binding constraint**; **(2)** adopt/ensemble the
pretrained backbone and retrain to convergence *after* labels are fixed; **(3)**
decouple the double imbalance correction and calibrate per finding; **(4)** fix the
64×64 serve-path skew for correctness (measured AUROC impact ≈ 0, so not urgent).

---

## Measured results (experiments I ran — not assumptions)

Subset: first 140 val subjects → **700 images** (prevalence opacity 511, effusion 446,
cardiomegaly 411, consolidation 374, nodule 271, pneumothorax 81, hyperinflation 65).
Labels are the current (smeared) labels — this is a *relative* comparison on identical
images/labels. Low-support findings (pneumothorax, hyperinflation) are noisy at this n.

**E1 — `best_model.pt`: clean 224 vs served 64×64→224**

| | macro AUROC | opacity | consolid. | effusion | cardiom. | nodule | pneumoTx | hyperinf. |
|---|---|---|---|---|---|---|---|---|
| clean (224) | **0.678** | 0.776 | 0.679 | 0.760 | 0.689 | 0.474 | 0.649 | 0.719 |
| served (64) | **0.685** | 0.764 | 0.600 | 0.729 | 0.700 | 0.561 | 0.741 | 0.697 |

**Δ macro = −0.006 (within noise).** ⇒ the serve-path skew does **not** measurably
degrade ranking for this model. My prior hypothesis that "served is strictly worse" is
**refuted**. Fix it for consistency, not for score. (F1 downgraded P0→P2.)

**E0 — pretrained `torchxrayvision densenet121-res224-mimic_ch` vs `best_model.pt`** (6 shared findings)

| | macro AUROC (6) | opacity | consolid. | effusion | cardiom. | nodule | pneumoTx |
|---|---|---|---|---|---|---|---|
| best_model.pt | 0.671 | 0.776 | 0.679 | 0.760 | 0.689 | 0.474 | 0.649 |
| xrv_mimic | **0.702** | 0.786 | 0.685 | 0.780 | 0.737 | 0.587 | 0.638 |

**Δ macro = +0.031** in favour of the pretrained model — real but **modest**, and far
below its published ~0.80–0.90. **Key inference:** a strong, properly-trained external
model *also* plateaus at ~0.70 against these labels. ⇒ **label quality (F3), not model
capacity, is the ceiling.** This makes F3 the #1 priority and means retraining (E3) is
low-value until labels are de-smeared (E2).

### E2 — De-smearing IMPLEMENTED, validated, and measured (full val, n=2099)

**Change shipped:** `ml/vision_cxr/dataset.py` `load_mimic_samples(..., per_study=True)`
now labels each image with **its own study's** report (study id parsed from the path;
studies align 1:1 to reports — verified 100% of rows, 0 fallbacks). Legacy behaviour
kept behind `per_study=False`. Used by both training and `clinical_eval`.

**Prevalence was inflated 3–13× by smearing** (per-study is realistic for MIMIC ICU/ED):

| finding | smeared | per-study | true rate | note |
|---|---|---|---|---|
| opacity | 1430 (68%) | 496 | 17% | |
| consolidation | 832 | 120 | 4% | 6.9× inflated |
| effusion | 1169 | 524 | 18% | |
| cardiomegaly | 872 | 284 | 9% | |
| nodule | 542 | 95 | 3% | 5.7× inflated |
| pneumothorax | 249 | 63 | 2.1% | matches literature |
| hyperinflation | 415 | 120 | 4% | |

**25.9% of all (image,finding) labels flip.** Re-running the *real* eval pipeline
(`clinical_eval`, 1000-bootstrap, plots → `artifacts/evaluation_perstudy/`):

| macro metric | smeared (shipped) | **per-study (honest)** |
|---|---|---|
| AUROC | 0.702 | **0.667** (CI 0.650–0.683) |
| AUPRC | 0.558 | **0.214** |
| F1 @0.5 | 0.547 | **0.255** |
| ECE | 0.147 | **0.337** |
| Brier | 0.202 | 0.245 |

**Interpretation (evidence-based, reframes the audit):** smearing didn't just add noise —
it **leaked patient-level signal** ("any film of a patient who ever had effusion →
effusion+"), making an *easier* task than per-image diagnosis and inflating **every**
metric. The honest per-study numbers are the real deployment performance: **AUROC ~0.67,
and critically AUPRC 0.21 / F1 0.25 / ECE 0.34** — the model is badly **over-confident**
(trained to fire at 40–68% base rates; true rates 2–18%). De-smearing is a *correctness /
honesty* fix and a prerequisite for retraining, not itself an AUROC win.

**Labeler spot-check (14 real reports):** the regex `mimic/labeling.py` labeler is
~80% correct but has systematic gaps — misses cardiomegaly synonyms ("heart size is
prominent"), and forward-scoping uncertainty over-fires ("possibly … mild cardiomegaly"
→ wrongly uncertain → false negative). This is the residual label ceiling once smearing
is removed (why xrv can't beat ~0.67 on these labels).

**Revised next levers, in order:**
1. **E4 (cheap, no retrain): per-finding calibration + threshold optimization.** — DONE, see below.
2. **E3: retrain on de-smeared labels** (sampler OFF, pos_weight ON, ≥10 epochs) so the
   model learns the true per-image task instead of the patient-level shortcut.
3. **Better labeler** (official CheXpert / validated NLP) to lift the residual ceiling.

### E4 — Per-finding calibration DONE (held-out patient split, no retrain)

Fit on 205 calib patients, evaluated on **133 disjoint test patients** (888 images).
**Temperature-only scaling failed** (ECE 0.338→0.313) — it can only shrink toward 0.5,
but the defect is a base-rate mismatch (model fires ~0.5 for 2–4%-prevalence findings).
**Platt scaling** (affine `a·logit+b`, the intercept absorbs the prior) fixed it:

| TEST macro (held-out patients) | before (raw, thr 0.5) | after (Platt + opt thr) |
|---|---|---|
| ECE | 0.3382 | **0.0363** (−90%) |
| F1 | 0.2592 | 0.2631 |
| AUROC | 0.6534 | 0.6549 (preserved) |

Calibrated means now track prevalence (consolidation 0.56→0.05 = prev 0.05; pneumothorax
0.22→0.03 = prev 0.03). Artifact: `artifacts/vision_calibration_perfinding.json`.

**Two findings:**
- **Calibration ≠ discrimination.** ECE is fixed and probabilities are now honest, but
  **F1 barely moves** — it is capped by weak ranking (AUROC 0.65). No threshold rescues a
  weak discriminator; only **E3 (retrain)** lifts F1/AUPRC.
- **Nodule is anti-predictive:** Platt fit a **negative** slope (a=−0.23; AUROC 0.44 <
  chance). Recommend suppressing/flagging nodule in serving until retrained.

**Bottom line of this session:** two shipped fixes — de-smeared labels (correctness) and
per-finding Platt calibration (ECE −90%) — plus an honest model card. The remaining
weakness is **discrimination**, which requires E3 (retrain on the corrected labels) and,
for the ceiling, a better report labeler. Nothing else is a major weakness.

---

## Component dependency graph (as-built)

```
StudyInput.image (64x64!) ─┐
                           │  gateway/pipeline.py:61  img = reshape(study.image, (64,64))
PACS/JPG ─ services/vision/io.py:load_cxr ─ _normalize01 (0.5/99.5 pct) ─ study_from_cxr ─ 64x64 downsample ─┘
                           │
                           ▼
services/vision/engine.py VisionEngine.load()
   ├─ if artifacts/best_model.pt exists → ml/vision_cxr/inference.py VisionModel   ◀── SERVED (2-epoch DenseNet121)
   ├─ elif settings.vision_backend=="densenet_mimic" → services/vision/cnn.py CXRBackbone (torchxrayvision)  ◀── UNUSED
   ├─ elif "timm" → CXRBackbone(timm) + artifacts/vision_cnn.pt (absent)
   └─ else → services/vision/features.py hand-crafted logistic  (final fallback)
                           │ score_findings(img)  [pure callable]
                           ▼
   fusion → safety(calibration/conformal/OOD) → explain(occlusion + Grad-CAM) →
   recommend → reasoning → report (services/report/*  deterministic template)

TRAINING PATH (offline):
  ml/vision_cxr/train.py ─ dataset.build_loaders ─ load_mimic_samples
        │                       ├─ mimic/labeling.py label_report (forward-scoping)   ◀── labels here
        │                       └─ WeightedRandomSampler (rare-positive oversample)
        ├─ losses.RegularizedMultiLabelLoss = BCE(pos_weight) + λ·TV(features)
        ├─ AdamW 3e-4 / wd 1e-4 / CosineAnnealingLR(T_max=10) / AMP / clip 1.0
        └─ checkpoint by val macro-AUROC (early stop patience 4)

EVAL PATH: ml/evaluation/clinical_eval.py ─ load_validation(validate_csv) [SAME smeared labels] ─ VisionModel(best_model.pt)
DEAD/PARALLEL: ml/training/prepare_mimic_manifest.py has a SECOND, cruder label_report (divergent)
```

---

## Findings (ranked; required format)

### F1 — Deployed model is fed 64×64 images; trained on 224 from full-res  ·  Priority ~~P0~~ → P2 (measured)
- **MEASURED (E1):** Δ macro AUROC = −0.006 (clean 0.678 vs served 0.685). The skew is a real correctness/consistency defect but has **no measurable AUROC impact** for the current model. Fix it, but do not expect a score gain.
- **Problem:** train/serve preprocessing skew. The served model never sees the resolution it was trained on.
- **Evidence [VERIFIED]:** `gateway/pipeline.py:61` `img = np.array(study.image).reshape(study.image_shape)` where `study.image` is the 64×64 downsample built in `services/vision/io.py:79-88` (`grid=64`). `VisionModel._to_tensor` (`ml/vision_cxr/inference.py:40-41`) then `cv2.resize(64²→224²)`. Training used `A.Resize(224)` on full-res JPGs (`dataset.py:22`). The benchmark (`clinical_eval.py:56`) *also* uses full-res→224, so **metrics.json's 0.70 does not reflect the served path.**
- **Root cause:** `study_from_cxr` stores a 64×64 thumbnail as the canonical image for the whole pipeline; the CNN branch was added later and inherited that thumbnail instead of re-reading full-res.
- **Fix:** pass the full-res `[0,1]` array to the vision engine (keep the 64×64 only for the legacy overlay/feature fallback). Either carry `full` on `StudyInput` or have `VisionModel` re-load from `path`. Then align normalization (see F10).
- **Expected improvement [NEEDS EXPERIMENT]:** recovers the gap between served and benchmarked AUROC; largest effect on fine-structure findings (nodule, pneumothorax). Hypothesis to measure, not assumed.
- **Difficulty:** Low. **Priority: P0.**

### F2 — Model is undertrained (2 epochs)  ·  Priority P0
- **Problem:** the production checkpoint stopped at epoch 1 of a 10-epoch schedule.
- **Evidence [VERIFIED]:** `best_model.pt` `epoch=1`, `best_metric=0.6962`; `history.csv` has only epochs 0–1; `val_loss` still falling 1.336→1.186; `train_loss` 0.898→0.647. No plateau ⇒ underfit, not overfit. CosineAnnealingLR `T_max=10` (`train.py:145`) barely decayed.
- **Root cause:** training run was interrupted/short (2 epochs written), yet the checkpoint was shipped.
- **Fix:** train to convergence (≥10–20 epochs) with early stopping on a *clean* val metric; keep the current AdamW/cosine setup (it is sound).
- **Expected improvement [NEEDS EXPERIMENT]:** standard MIMIC DenseNet121 reaches ~0.80–0.85 macro AUROC on these findings; converging from 0.70 is plausible but must be measured on de-smeared labels (F3).
- **Difficulty:** Medium (compute). **Priority: P0.**

### F3 — Labels are smeared across all of a patient's studies  ·  Priority P0 (THE binding constraint)
- **MEASURED (E0):** a strong external model (`torchxrayvision`, published ~0.85) reaches only **0.702** macro AUROC against these labels — essentially the same ~0.70 wall as the local model. Two very different models hitting the same ceiling is direct evidence the **labels**, not the model, cap performance.
- **Problem:** every image of a subject gets ONE label vector built from **all** the subject's reports concatenated. An early normal film inherits a later film's pathology (and vice-versa). This corrupts both training targets and the AUROC denominator.
- **Evidence [VERIFIED]:** `dataset.py:69-90` — `report_text = " ".join(sentences)` over the row's entire `text` list, one `y_vec`, then `for img in img_list: labels.append(y_vec)`. Probe: val subjects average **3.62 reports / 5.98 images** (max 95 reports / 114 images); **48.2%** have >1 report. Sampled subject 10003502: 8 reports describing *different* states ("effusions unchanged" vs "small effusion may be present" vs "moderate bilateral effusions") → all 12 images labelled identically. `clinical_eval.py:42` evaluates against these same labels.
- **Root cause:** the CSV is one-row-per-subject with parallel image/report lists; the loader never aligns an image to *its own* study's report.
- **Fix:** build per-study (ideally per-image) labels. MIMIC image paths encode `.../pXXXX/sYYYY/<dicom>.jpg` (study `sYYYY`); align each image to the report of its study. If the CSV lacks a per-study mapping, regenerate from the MIMIC `mimic-cxr-2.0.0-*.csv` metadata + CheXpert/NegBio labels.
- **Expected improvement [NEEDS EXPERIMENT]:** primarily *correctness of the metric* and cleaner training signal; direction of the AUROC change is unknown until measured (smearing can both inflate — easy positives — and deflate — mislabeled negatives).
- **Difficulty:** Medium. **Priority: P0.**

### F4 — A pretrained MIMIC model exists but is not served  ·  Priority P0
- **Problem:** `services/vision/cnn.py` wraps `torchxrayvision densenet121-res224-mimic_ch` (real MIMIC-trained weights, published AUROCs ~0.80–0.90), but `VisionEngine.load()` prefers `best_model.pt` whenever it exists (`engine.py:60-67`), so the strong model is dead code by default.
- **Evidence [VERIFIED]:** `engine.py:56-70`; `cnn.py:98-111`.
- **Root cause:** load precedence favors the local checkpoint unconditionally.
- **Fix (cheap, decisive):** benchmark `densenet_mimic` vs `best_model.pt` on the *same* val set (no training needed) and serve whichever wins per-finding; or ensemble. Note it covers 6/7 findings (no hyperinflation) — the existing feature-model fill already handles that gap (`engine.py:119-121`).
- **MEASURED (E0):** +0.031 macro AUROC over `best_model.pt` on the 6 shared findings (0.702 vs 0.671) — real but modest, and capped by label noise (see F3). Biggest per-finding gains: nodule +0.11, cardiomegaly +0.05.
- **Expected improvement:** a cheap ~+0.03 now; the gain will grow once labels are fixed (F3). Serve xrv for its 6 findings + feature-fill for hyperinflation, or ensemble.
- **Difficulty:** Low. **Priority: P0 (cheap win) — but do E2/F3 first for the real ceiling lift.**

### F5 — Double imbalance correction miscalibrates minority findings  ·  Priority P1
- **Problem:** class imbalance is corrected **twice** — `WeightedRandomSampler` oversamples rare-positive images (`dataset.py:116-125`) *and* `pos_weight` up-weights rare positives in BCE (`train.py:148-153`). Stacking both over-predicts positives on rare classes → low precision + over-confidence (high ECE).
- **Evidence [VERIFIED]:** the two mechanisms above; `metrics.json` nodule precision 0.26 / recall 0.29 (still bad because also undertrained), pneumothorax precision 0.27, macro ECE 0.146.
- **Root cause:** two independently-added balancing strategies were never reconciled.
- **Fix:** keep **one**. Recommend `pos_weight` only (leave sampling natural), or a mild focal loss; re-measure precision/ECE.
- **Expected improvement [NEEDS EXPERIMENT]:** better precision + calibration on rare findings; controlled A/B (sampler on/off) required.
- **Difficulty:** Low. **Priority: P1.**

### F6 — Per-finding probabilities are served raw and thresholded at 0.5  ·  Priority P1
- **Problem:** vision sigmoids are neither temperature-calibrated nor threshold-optimized per finding. The safety engine calibrates the *diagnosis* posterior, not the per-finding vision outputs the report prints.
- **Evidence [VERIFIED]:** `inference.py:63-70` returns raw `sigmoid`; `clinical_report.py:111` `present = probability >= 0.5` (fixed); `clinical_eval.py:74` uses threshold 0.5 for all findings; macro ECE 0.146.
- **Root cause:** no per-finding calibration/threshold layer between the CNN and the report.
- **Fix:** fit per-finding temperature (Platt/temperature scaling) + per-finding operating point (maximize F1 or fix sensitivity) on a held-out split; store beside `best_model.pt`.
- **Expected improvement [NEEDS EXPERIMENT]:** ECE↓ and F1↑ at no AUROC cost (monotone transforms); measure ECE/Brier before/after.
- **Difficulty:** Low. **Priority: P1.**

### F7 — Metrics are on the model-selection set; CIs ignore patient clustering  ·  Priority P1
- **Problem:** (a) `clinical_eval.py` reports on `validate_csv`, the *same* set used for early-stopping/checkpoint selection — no independent test set. (b) The bootstrap resamples the 2,099 images i.i.d. (`clinical_eval.py:128`) although they come from ≤500 patients, so the 95% CIs are too narrow.
- **Evidence [VERIFIED]:** `train.py:193,213-217` selects best by val AUROC; `clinical_eval.py:42` evaluates the same `validate_csv`; `_bootstrap_ci` resamples image rows independently.
- **Root cause:** the `DatasetBuilder` test split (`mimic/splits.py`, patient-disjoint) is unused by the vision eval.
- **Fix:** evaluate on the untouched `test.csv`; switch to a **patient-clustered** (block) bootstrap.
- **Expected improvement:** honest generalization estimate + correctly-wide CIs (rigor, not a score gain).
- **Difficulty:** Low–Medium. **Priority: P1.**

### F8 — Nodule is at chance; pneumothorax weak  ·  Priority P1 (dependent)
- **Problem:** nodule AUROC 0.4848 (CI 0.456–0.515) — indistinguishable from random; pneumothorax 0.6375.
- **Evidence [VERIFIED]:** `metrics.json`. **Confound:** these are the findings most destroyed by 64×64 serving (F1), most corrupted by smearing (F3, "nodule/mass" is a rare, transient mention), and most distorted by double-balancing (F5). Cannot attribute a single cause from current evidence.
- **Fix:** re-measure after F1+F3+F2; if still poor, add higher input resolution (320/448) and lesion-preserving augmentation; consider excluding `mass/lesion` from the nodule regex (over-broad, see F9).
- **Expected improvement [NEEDS EXPERIMENT].** **Difficulty:** Medium. **Priority: P1.**

### F9 — Duplicate, divergent `label_report` implementations  ·  Priority P2
- **Problem:** two label engines exist: the forward-scoping `mimic/labeling.py` (used by training) and a cruder 35-char-window keyword matcher in `ml/training/prepare_mimic_manifest.py:12-84`. They disagree (uncertainty handling, negation scope, nodule includes "lesion"), a maintenance/leakage hazard if the manifest path is ever used.
- **Evidence [VERIFIED]:** both files. The deployed model uses the former (`dataset.py:10`). The manifest path (+ `train_cnn.py`) appears parallel/unused for `best_model.pt`.
- **Fix:** delete or clearly quarantine the manifest labeler; single-source labels. Long-term: adopt the official CheXpert labeler (the docstring already says it "maps to" it).
- **Difficulty:** Low. **Priority: P2.**

### F10 — Aspect-ratio stretch, no windowing/CLAHE; serve-vs-train normalization differs  ·  Priority P2
- **Problem:** `A.Resize(224,224)` distorts aspect ratio; no CLAHE/histogram normalization. Separately, serving normalizes via `_normalize01` (0.5/99.5 percentile stretch, `io.py:19-24`) while training uses raw `/255` (`dataset.py:49`) — different intensity distributions even before the 64×64 issue.
- **Evidence [VERIFIED]:** `dataset.py:15-22,49`; `io.py:19-24`; `inference.py:47-56` skips `/255` when input already ≤1.0.
- **Fix:** pick one intensity pipeline and use it in both train and serve (recommend percentile-windowing in both — it is the clinically correct one); optionally letterbox-pad to preserve aspect ratio.
- **Difficulty:** Low. **Priority: P2.**

### F11 — Report faithfully prints a weak, uncalibrated model (no hallucination)  ·  Priority P2
- **Problem:** the report generator is a deterministic template over the computed `CaseBundle` — it does **not** hallucinate or invent findings (Phase 6 concern resolved). But it renders `p≥0.5 → "present"` verbatim, so upstream false positives become clinical assertions.
- **Evidence [VERIFIED]:** `services/report/clinical_report.py` (no LLM, all fields read from the bundle); `_vision_findings` at line 102-114.
- **Fix:** gate finding statements on the calibrated threshold (F6) and surface per-finding uncertainty already available on the bundle. The requested "verified findings → evidence → reasoning → report" flow is *already* the architecture; it just needs calibrated inputs.
- **Difficulty:** Low. **Priority: P2.**

### F12 — Grad-CAM quality cannot be trusted from a 2-epoch model  ·  Priority P3
- **Problem:** Grad-CAM/Grad-CAM++ differentiates through `norm5` (`inference.py:32`); a 2-epoch, mislabeled-data model has weak features, so heatmap grounding is unreliable. The TV regulariser (`losses.py:15-58`) is a good idea but cannot compensate for undertraining.
- **Evidence [VERIFIED (code)] / [NEEDS EXPERIMENT (heatmaps)]:** target layer + TV present; heatmap quality not measured here. The occlusion saliency path is model-agnostic and methodologically valid regardless.
- **Fix:** re-evaluate CAM localization (e.g., pointing-game / IoU vs. finding regions) after F1–F3; only then tune λ_TV.
- **Difficulty:** Medium. **Priority: P3.**

---

## What is actually healthy (do not "fix")
- DenseNet121 grayscale adaptation via **luminance-weighted conv0** (`model.py:10-54`) — correct and well-reasoned.
- **No patient leakage** train↔val (probe: overlap 0). ✅
- AMP, grad-clip, AdamW, cosine schedule, checkpoint-by-AUROC — standard and correct.
- TV feature regulariser, occlusion saliency, deterministic grounded report — sound designs.
- Eval battery (`clinical_eval.py`): per-label ROC/PR/calibration/confusion, Brier, ECE, bootstrap — genuinely thorough (fix only the two rigor items in F7).

---

## Cannot conclude from available evidence (explicit)
- The *served* (64×64) AUROC — never benchmarked (F1).
- The *label-corrected* AUROC ceiling — requires de-smeared labels (F3).
- Whether nodule/pneumothorax fail from resolution, labels, or balancing — confounded (F8).
- Grad-CAM anatomical grounding — not quantified (F12).
- External-dataset generalization (Phase 11) — no external CXR set is present on disk.

---

## Proposed experiment ladder (each isolates ONE variable; measured, not assumed)

| # | Experiment | Compute | Decides |
|---|-----------|---------|---------|
| E0 | Benchmark `torchxrayvision densenet_mimic` vs `best_model.pt` on val (no training) | minutes (weights DL) | F4 — quick-win baseline |
| E1 | Re-eval `best_model.pt` with full-res serve preprocessing vs 64×64 | minutes | F1 magnitude |
| E2 | Rebuild per-study labels; re-eval same checkpoint | ~1 hr | F3 metric correctness |
| E3 | Retrain to convergence on de-smeared labels (sampler OFF, pos_weight ON) | GPU-hours | F2+F5 |
| E4 | Fit per-finding temperature + thresholds on held-out; measure ECE/Brier/F1 | minutes | F6 |
| E5 | Report on patient-disjoint `test.csv` w/ patient-clustered bootstrap | minutes | F7 |

Only E3 is expensive. E0/E1/E4/E5 are runnable now and would already move the honest
headline. Recommend running **E0 → E1 → E2** first (all cheap-to-moderate) before
committing GPU-hours to E3.
