# AURA Changelog

All notable changes to the AURA project are documented here.

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
