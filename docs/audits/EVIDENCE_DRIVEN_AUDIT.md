# AURA — Evidence-Driven Inference Audit

*Placed in `E:\AURA\` (outside the `aura-main` git repo) because a concurrent window is
consolidating the repo's root `.md` docs into `AURA_MASTER_AUDIT.md` and repeatedly removing
new root markdown files. Merge into that master doc when the concurrent work settles.*

**Objective:** every output shown to the user must derive from the uploaded chest radiograph +
learned model outputs — not defaults, scripted logic, synthetic priors, placeholders, or
hardcoded rankings.

**Method:** full source read of `upload → gate → vision → evidence → fusion → reasoning →
safety → explain → recommend → report → dashboard`, cross-checked against fitted artifacts and
**measured** runs of the real pipeline on real MIMIC-CXR films (`E:\AURA\venv`, torch 2.11+cu128).

---

## 0. Headline verdict
**The core diagnosis path is genuinely evidence-driven and learned. No Critical fabrication, no
static/placeholder probabilities, no image-independent fallback in the served path.**
- Findings: real DenseNet-121 forward pass (`ml/vision_cxr/inference.py:101`) → Platt calibration
  fitted on n=2099 (`vision_serving_calibration.json`) — real fits, not the degenerate `b≈−10`
  clamp a prior audit flagged (repaired 2026-07-21).
- Diagnosis: trained `W·x+b` fusion (`services/fusion/classical.py:37`).
- Synthetic studies disabled (410); model-card returns `null` not placeholders; abstains to
  `INDETERMINATE` when weak.

Empirically (req #5): 4 distinct MIMIC films → 4/4 distinct finding vectors, distinct posteriors,
max per-finding delta **0.432**, 3/4 abstained rather than forcing a call.

---

## 1. Findings (severity = impact on a served prediction)

### 🔴 CRITICAL — none.

### 🟠 HIGH
- **H1 — dashboard hid detections behind hardcoded `0.5`** → **FIXED + verified.** `console.js`
  filtered findings at 0.5 while report used calibrated thresholds (0.13–0.29), so a diagnosis
  could render with supporting findings invisible. Fix: `gateway/app.py get_case()` attaches
  calibrated `present`/`threshold`; `console.js` filters on `present`, label uses
  `EV_LABEL[f.finding]` (fixes `pleural_effusion` mislabel), stale `CXR 64×64` → real
  `image_shape`. Verified live: effusion 0.437 (was hidden) now renders as "Pleural effusion ·
  0.44", meta "CXR 224×224".
- **H2 — served default backend** → **flipped to `classical`, recalibrated, verified.** See §3.

### 🟡 MEDIUM
- **M1** static overlay boxes (`vision/engine.py:25`) — image-independent; Grad-CAM++ is the real
  localization. *Open.*
- **M2** `prior_risk_score` hand-weights (`fusion/evidence.py:36`) — 1/8 channel, **inert for
  image-only uploads**.
- **M3** reasoner guideline LRs (`reasoning/engine.py:65`) — **inert for uploads** (needs labs).
- **M4 — DOWNGRADED.** `registry.json` was NOT stale/inverted (my initial claim was wrong); it was
  a legitimate n=173-split result. Now regenerated with served-role + provenance labels.

### 🟢 LOW (legitimate specs / decorative / dev-only)
`_priority` sort; recommender `_SEVERITY`/`CATALOG`/`causal.py` (image-driven selection);
`landing.js` decorative animation; "lungs are clear" template (image-dependent negative);
feature fallback (dev-only, production hard-fails without DenseNet).

---

## 2. Measured — Requirement #5 (PASS)
Real pipeline, trained DenseNet, `AURA_ALLOW_FALLBACK_VISION=0`, 4 patients — 4/4 distinct finding
vectors + posteriors, max delta 0.432, 3/4 abstained. Outputs are genuinely image-dependent.

## 3. H2 — HONEST picture + implementation

⚠️ **Correction to an earlier overclaim.** I first said "classical wins all six metrics, robust,
registry.json stale/inverted." Regenerating calibration on `train_fusion`'s own split exposed this
as overstated. Three measured splits:

| eval split | n(test) | quantum acc | classical acc | quantum ECE | classical ECE |
|---|---|---|---|---|---|
| `bench` (benchmark.json) | 69 | 0.638 | **0.696** | 0.238 | **0.219** |
| `train_fusion`/recal (registry) | 173 | **0.520** | 0.480 | 0.061 | **0.044** |
| repair report §8 | — | 0.667 | **0.710** | — | — |

- **Accuracy: no robust winner** — classical wins 2 of 3 splits, quantum 1; all within noise at
  n≈70–170 on a hard 6-class task.
- **Calibration: classical consistently better** (lower ECE on every split) — the one robust
  signal, and the one that matters most for a clinical probability-emitter.
- `registry.json` was a real n=173 result, not stale (M4 corrected).

**Decision (user-approved): serve `classical`, keep `quantum` selectable (research)** — justified
by calibration + simplicity/determinism/interpretability + user choice, NOT by a raw-accuracy claim
the data doesn't support. Implemented correctly:
- `common/config.py` + `pyproject.toml`: `fusion_backend = "classical"`.
- **Recalibrated for classical** (serving classical logits through quantum's T=0.94 would be
  miscalibrated): `safety.npz` T 0.9386→**0.4574**, OOD + Mondrian conformal refit on classical
  logits, `registry.json` regenerated (classical=served, quantum=research). **Models untouched**;
  old artifacts → `*.prequantumflip.bak`. Reused `train_fusion`'s exact helpers + same seeded split.
- **Verified:** req #5 PASS on classical path; full suite green (~135 tests, 1 skip).

## 4. Still open
- **M1** — overlay boxes from Grad-CAM++.
- **M2/M3** — only with a real prior/labs→outcome cohort; else quarantine + label (currently inert).
- Merge this report into `AURA_MASTER_AUDIT.md` once the concurrent doc-consolidation settles.
