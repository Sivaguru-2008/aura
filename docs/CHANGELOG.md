# AURA Changelog

All notable changes to the AURA project are documented here.

---

## [1.3.0] — 2026-08-21 (Manuscript remediation: submission version 5.0)

Closes every blocking issue identified in a Tier-1 readiness review of the manuscript. Four
new experiments, seven artifact/documentation defects closed, and a submission-length
manuscript with Supplementary Information and a reporting checklist.

### New experiments
* **Matched optimiser protocol** (`ml/evaluation/matched_optimiser.py`, `fusion_matched_optimiser.json`).
  Refits all four fusion arms under one optimiser, batch size, epoch cap, early-stopping rule,
  weight decay, seed and per-arm learning-rate grid. **Every significant ordering in the
  published Table 6 dissolves**; the quantum circuit's likelihood advantage over the
  product-of-experts reverses in sign. Budget adequacy is measured on an L-BFGS anchor
  (residual gap 0.0052 NLL) and reported rather than assumed. Adds protocol rule **P8**.
* **Nonlinear positive control** (`ml/evaluation/nonlinear_control.py`, `nonlinear_control.json`).
  Holds inputs, splits, arms, protocol and tests fixed and varies only the target's structure,
  establishing whether the paper's nulls measure the task or the apparatus. Measures the linear
  ceiling of the clinical target at **0.9991**, empirically confirming a claim previously
  asserted from the labeller's source.
* **Full-coverage quantum kernel** (`ml/evaluation/qkl_full_coverage.py`, `brain/qkl_full_coverage.json`).
  Replaces an 11-test-subject single split with nested subject-level cross-validation over all
  **185 subjects**, with symmetric 5x5 hyperparameter grids giving the quantum arm the bandwidth
  it previously lacked. ΔAUROC −0.0026, CI [−0.044, +0.039], DeLong p = 0.90 — an informative
  null in place of an absence of evidence.
* **DeLong consolidated** (`ml/evaluation/delong.py`). Single implementation, validated against
  `sklearn.metrics.roc_auc_score`, with `audit_all.py` delegating to it.

### Defects closed
* **D1** superseded kernel figures updated across README, grading, methodology and pitch docs.
* **D3/D4** `quantum_study.json` annotated in place with the §2.8 retraction; its hardware
  statement corrected to scope itself to that artifact and name the `ibm_marrakesh` run.
* **D6** training lengths and split provenance recorded on the artifacts that lacked them;
  `fusion_ablation.json` marked superseded with its unknowns stated.
* **D8** the full-coverage embedding dump is now used, removing the power loss it caused.
* **D10 (safety-critical)** the served abstention envelope was published wrong on two of four
  criteria. `Settings.epistemic_abstention_threshold` replaces a hardcoded literal, the
  abstention branch now reads all four criteria from configuration, and a new source-level test
  fails if any criterion becomes a literal again. The build test that should have caught this
  compared documentation against settings the serving path never read.
* **D12 (new)** the published design-sweep claim ("four of five best cells at zero entanglement")
  does not match the artifact when read by negative log-likelihood; corrected to **three of five**.
* **D13 (new)** DeLong existed only in `audit_all.py`, outside every packaged evaluation module.
* **D14 (new, most serious)** `audit_all.py` — the harness Code Availability tells reviewers to
  run — drew its held-out split from the **synthetic** evidence world. Correct until audit
  finding F1 refitted the fusion heads on real MIMIC-CXR evidence (2026-07-21); never updated
  afterwards. Every run since evaluated real-trained models on out-of-distribution vectors and
  reported **below-chance accuracy (0.010 against a 1/6 floor)** while still printing
  `CLAIM [VALIDATED]` for one of three claims. Now reads the same cached real-evidence split the
  manuscript uses, records the data world in its manifest, warns loudly on synthetic fallback,
  and **aborts** rather than reporting metrics when a served head scores below chance. It
  reproduces the published table exactly (quantum 0.5324 / classical 0.5414, ECE 0.0896 /
  0.1266, T = 0.9057 / 0.4790). §2.1's ECE figures are re-reported on real data as a
  consequence.

### Manuscript
* `docs/RESEARCH_PAPER_v5.md` — submission version: ~5,000-word main text, 372-word structured
  abstract, 8 display items.
* `docs/RESEARCH_PAPER_v5_SUPPLEMENTARY.md` — extended methods, all remaining tables, the
  thirteen-item defect register, and the claim-to-artifact map.
* `docs/REPORTING_CHECKLIST.md` — TRIPOD+AI and CLAIM, with four material gaps stated plainly.
* Drift test extended to the v5 manuscript **including its abstract**, which was previously
  uncovered, plus a guard that fails on unfilled placeholders.

---

## [1.1.0] — 2026-07-21 (Audit Repairs Update)

This release implements fixes for 10 critical and high-severity findings identified during the reverse-engineering audit of the repository, establishing complete alignment between the clinical reasoning components.

### Repaired Safety-Critical Wiring
* **F2 (Wasserstein Conflict Guard)**: Threaded resolved logits directly from the conflict guard into the safety engine, ensuring `safety.top` and report summaries reflect the validated PoE fallback rather than the discarded VQC posterior.
* **F9 (Adaptive Conformal Inference)**: Connected `SafetyEngine` to the SQLite conformal prediction state (`aci_qhat`), allowing signed clinician feedback to drive the online coverage threshold.
* **F10 (Clinical Reasoner Integration)**: Re-ordered the pipeline execution so the `ClinicalReasoner` runs before the `SafetyEngine`. Safety now validates the reasoning-adjusted posterior, resolving contradictions between the differential and the report impression.
* **Epistemic Engine Thread-Safety**: Resolved a thread-safety race in `RecommendEngine` by refactoring panel calculations to utilize local variables rather than shared instance attributes.

### Improved Vision & ML Pipelines
* **F1 (Fusion Distribution Alignment)**: Refactored `ml/training/dataset.py` to train and calibrate the fusion models on real MIMIC-CXR evidence distributions (obtained from the DenseNet model) instead of synthetic heuristics.
* **F3/F4 (Checkpoint Promotion)**: Promoted the Epoch 7 DenseNet checkpoint (macro-AUROC 0.821, pneumothorax sensitivity 0.460, nodule AUROC 0.729) to `artifacts/best_model.pt`, replacing the weaker Epoch 1 model (macro-AUROC 0.696).
* **F5 (Image Downsampling Fix)**: Removed the 64x64 pixel downsampling step in `services/vision/io.py`, restoring full-fidelity $224 \times 224$ area-averaging for CNN inputs.
* **F6 (Fair Fusion Benchmark)**: Standardized temperature scaling in `benchmark.py` to evaluate each fusion backend (Quantum VQC, Classical PoE, Learnable head) on its own calibration split, eliminating the classical calibration bias.
* **F7 (Mondrian Conformal Quantiles)**: Hardened the Mondrian prediction sets to fall back to pooled marginal quantiles when class-specific sample counts are too low, preventing quantile saturation.
* **F8 (OOD Energy Calibration)**: Recalibrated energy-score OOD parameters against real MIMIC evidence distributions instead of artificial distributions, allowing tail cases to fire correctly while maintaining typical films in-distribution.

---

## [1.0.0] — 2026-07-18 (P0 Hackathon Release)

Initial offline clinical intelligence copilot implementation.
* 8-qubit variational quantum circuit (PennyLane) for evidence fusion.
* FastAPI gateway serving static doctor dashboard.
* 9-stage analysis pipeline (vision, fusion, safety, explainability, recommender, and reporting).
* SQLite persistence and append-only audit ledger.
