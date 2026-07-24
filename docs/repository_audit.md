# AURA — Repository Audit

**Audit date:** 2026-07-24
**Scope:** Full source tree under `aura-main/` (excluding `.venv/`, `__pycache__/`, and vendored `site-packages/`).
**Method:** Static inventory of the tree plus a live test-collection and execution pass against the trained checkpoints. This audit records what is *actually wired and verified*, and separates it explicitly from what is *declared but planned*.

---

## 1. Inventory

| Metric | Count |
| --- | --- |
| Python source files (excl. venv / pycache) | 261 |
| Test modules (`aura/tests/test_*.py`) | 26 |
| Documentation files (`docs/`) | 20+ (see `docs/`) |
| Model cards | 4 (`docs/model_cards/`) |
| Demo cases (`demo_data/`) | 9, each with `expected_output.json` |

### Top-level package layout (`aura/`)

| Path | Responsibility |
| --- | --- |
| `backend/core/` | Modality router, upload intake, shared types/errors/logging |
| `backend/foundation/mri/` | MRI Foundation Layer — DICOM/NIfTI/NRRD readers, geometry, QC, standardisation |
| `backend/vision/brain/` | BraTS ResU-Net trainer + inference (NeuroMind's network) |
| `backend/engines/neuro/` | NeuroMind engine: validate → preprocess → analyze → report, calibration, bundle |
| `services/` | Legacy microservice engines (Thorax/CXR, fusion, safety, ACI) coordinated by the event bus |
| `gateway/` | HTTP/API surface |
| `common/`, `schemas/` | Config surface and shared contracts |
| `mimic/`, `ml/`, `data/` | MIMIC-CXR loaders, feature/training pipelines |
| `artifacts/` | Generated evaluation and benchmark outputs |
| `demo_data/` | Structured demo dataset (see Task 3) |

---

## 2. Test status

The suite runs on Python 3.14 with the trained checkpoints present.

- **`tests/test_mri_foundation.py`: 113 collected, 113 passing.** This includes the four `NeuroMind engine integration` tests (`§10`) that were the subject of the most recent repair.
- The three previously-failing NeuroMind tests are **fixed and verified**:
  1. `test_neuromind_preprocess_runs_the_foundation_layer` — a single-sequence NIfTI phantom now preprocesses through the real foundation pipeline and the engine abstains (rather than being rejected at `validate_input`).
  2. `test_neuromind_foundation_evidence_carries_no_clinical_claim` — direct `preprocess()` calls succeed for single-sequence phantoms; the missing/duplicate-sequence rejection now lives in `analyze()`, so `run()` still returns `FAILED` with the right validation error.
  3. `test_neuromind_declines_a_single_slice_with_a_specific_reason` — a single-slice NIfTI now reaches `preprocess`, raising `MRIFoundationError`, and the payload carries `error="unreadable_image"` with `foundation_error` attached.

### Environment caveats found during audit
- The repo carries an in-tree venv (`aura-main/.venv`, Python 3.14) where **`pandas` cannot load** — its compiled extension is blocked by a Windows *Application Control* policy (`DLL load failed … blocked by Application Control policy`). The nine `tests/test_mimic_*.py` modules therefore fail *collection* (not assertion) in that venv.
- The **global** Python 3.14 interpreter (the documented serving interpreter) loads `pandas` and `torch` cleanly and is the environment used for the full-suite run.
- Optional imaging dependencies (`pydicom`, `nibabel`, `pynrrd`) are intentionally commented out of `requirements.txt`; the MRI tests guard on them with `pytest.importorskip`. They must be installed for the foundation/NeuroMind tests to execute rather than skip.

**Actionable:** pin `pandas`, `pydicom`, `nibabel`, `pynrrd` in a `requirements-dev.txt` (or an `[dev]` extra) so a clean checkout can run the full suite without manual dependency archaeology, and document the Application-Control constraint next to the venv.

---

## 3. Real vs. planned — the honesty ledger

AURA's design deliberately separates *capabilities it has* from *capabilities it names on a roadmap*. This audit confirms the separation holds in code.

### Verified real (image-driven, learned, tested)
- **NeuroMind brain MRI** serves the actual BraTS ResU-Net checkpoint (epoch 24). Output is a function of the pixels: single-sequence Dice degradation and the presence-head behaviour are measured, not asserted. FLAIR-dominant behaviour (T1-only whole-tumour Dice ≈ 0.02) is a documented measured fact.
- **Presence head is Platt-calibrated** (ECE 0.095 → 0.018); the engine abstains on *validity* violations (missing sequences, low quality, OOD energy, low confidence) rather than on a raw threshold.
- **Chest/CXR (DenseNet-121)** path is repaired and served at 224 px full fidelity with per-finding Platt calibration; grounded report generation (no hallucinated findings).
- **Modality router** rejects unsupported modalities before they reach an engine, with an explicit NIfTI/NRRD bypass so volumetric MR is not misclassified from pixel geometry.

### Declared-but-planned (correctly excluded from live capabilities)
`NeuroMindEngine.PLANNED_CAPABILITIES` names `tumor_subtype_classification`, `intracranial_hemorrhage`, `midline_shift_quantification`, `white_matter_burden`, and `longitudinal_comparison`. These are asserted *absent* from the live descriptor by `test_neuromind_only_claims_capabilities_it_has`. The roadmap is not a claim.

### Known, disclosed weaknesses (not defects — documented boundaries)
- Quantum-vs-classical accuracy is split-dependent; the classical Product-of-Experts backend is the *served* winner only on **calibration**, and that is what ships. (See `EVIDENCE_DRIVEN_AUDIT.md`.)
- Head CT vs. brain MRI cannot be separated from raw PNG/JPEG pixel geometry; DICOM metadata routes head CT safely, raw exports route to NeuroMind flagged-for-review.
- Nodule detection fails the torchxrayvision cross-check and is disclosed as such in the model card and UI.

---

## 4. Cross-references
- Full narrative audit & single source of truth: [`AURA_MASTER_AUDIT.md`](../AURA_MASTER_AUDIT.md)
- Evidence-driven upload→report audit: [`EVIDENCE_DRIVEN_AUDIT.md`](../../EVIDENCE_DRIVEN_AUDIT.md) *(at the `E:\AURA` root, above `aura-main`)*
- Clinical boundaries: [`docs/KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
- Deployment readiness: [`docs/deployment_readiness.md`](deployment_readiness.md)
- Model cards: [`docs/model_cards/`](model_cards/)

---

*This audit reflects the state of the tree on 2026-07-24. Test counts are from a live run against the trained checkpoints; re-run `pytest tests/test_mri_foundation.py -q` to reproduce the NeuroMind integration results.*
