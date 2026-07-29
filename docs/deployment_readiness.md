# AURA — Deployment Readiness Assessment

**Assessment date:** 2026-07-24
**Verdict:** **Conditional GO** for offline demonstration / evaluation deployment. **NOT** for autonomous clinical use.

This document is a go/no-go readiness checklist. For installation mechanics see [`DEPLOYMENT.md`](DEPLOYMENT.md); this file records whether each readiness dimension is actually satisfied today.

---

## 1. Readiness scorecard

| Dimension | Status | Evidence / Note |
| --- | --- | --- |
| Core inference path (brain MRI) | ✅ Ready | Serves the real BraTS ResU-Net (epoch 24); output is image-driven and tested. |
| Chest/CXR path | ✅ Ready | DenseNet-121 served at 224 px, per-finding Platt calibration, grounded reports. |
| Modality routing & rejection | ✅ Ready | Router rejects unsupported modalities pre-engine; NIfTI/NRRD bypass verified. |
| Calibration & abstention | ✅ Ready | Presence head Platt-calibrated (ECE 0.018); validity-based abstention wired. |
| Test suite | ✅ Ready* | `test_mri_foundation.py` 113/113 pass. *See dependency caveat below. |
| Offline operation | ✅ Ready | No network calls on the inference path; runs on local CPU or GPU. |
| Reproducible environment | ⚠️ Conditional | Optional imaging deps & `pandas` not pinned; see §3. |
| Authentication / access control | ⚠️ Off by default | `AURA_SEC_AUTH_ENABLED=false`; enable bearer auth before any shared deployment. |
| Clinical validation | ⛔ Not for autonomous use | Research/demo grade; boundaries in `KNOWN_LIMITATIONS.md`. |
| Regulatory clearance | ⛔ None | Not an FDA/CE-cleared device. Decision-support/education only. |

---

## 2. Pre-flight checklist

- [x] Trained checkpoints present (`artifacts/best_model.pt`, brain vision v2 @ epoch 24).
- [x] Config surface resolves (`common/config.py` ← `pyproject.toml` + `AURA_*` env).
- [x] Served fusion backend is `classical` (fair-benchmark calibration winner) — confirm `fusion_backend = "classical"` in `pyproject.toml`.
- [x] Abstention thresholds set (`ood_energy_threshold`, `low_confidence_threshold`).
- [ ] **Enable authentication** (`AURA_SEC_AUTH_ENABLED=true`) before exposing beyond localhost.
- [ ] **Pin runtime deps** (see §3) so a clean host reproduces the tested environment.
- [x] Demo dataset staged (`demo_data/`, 9 cases with `expected_output.json`).
- [x] Export paths verified (PDF / JSON / CSV / NIfTI — see `docs/export_validation.md`).

---

## 3. Environment hardening (required before a fresh-host deploy)

The audit found the tested environment depends on packages that are **not pinned** in `requirements.txt`:

- `pydicom`, `nibabel`, `pynrrd` — required for MRI ingest and the foundation/NeuroMind tests (currently `importorskip`-guarded, so their absence silently *skips* rather than fails).
- `pandas` — required by the MIMIC pipeline and `tests/test_mimic_*.py`.

**Windows note:** on the in-tree venv, `pandas`' compiled extension is blocked by a host *Application Control* policy (`DLL load failed … blocked by Application Control policy`). Deploy on the interpreter where `pandas` loads (the documented global Python 3.14) or on a host without that policy. Validate with:

```bash
python -c "import pandas, torch, pydicom, nibabel, nrrd; print('deps ok')"
```

**Recommendation:** add a `requirements-dev.txt` pinning the four packages above and run the full suite on the target host as an acceptance gate.

---

## 4. Smoke test (run on the target host)

```bash
# 1. Dependency check
python -c "import pandas, torch, pydicom, nibabel, nrrd; print('deps ok')"

# 2. Foundation + NeuroMind integration tests must pass
python -m pytest tests/test_mri_foundation.py -q

# 3. Bring up the gateway
python -m aura.aura_cli serve 8000
#    → open http://127.0.0.1:8000 and confirm the dashboard renders live counts

# 4. End-to-end demo
python demo.py
```

A deployment is **GO for demo** when steps 1–4 succeed and authentication is enabled for any non-localhost exposure.

---

## 5. Explicit non-goals / do-not-deploy conditions

- Do **not** deploy for autonomous diagnosis or triage without a clinician in the loop.
- Do **not** rely on capabilities listed in `NeuroMindEngine.PLANNED_CAPABILITIES` (subtype classification, hemorrhage, midline shift, etc.) — they are roadmap, not shipped.
- Do **not** feed 2D PNG/JPEG exports of MR studies; they are refused by design.
- Do **not** treat quantum-fusion accuracy claims as deployment-relevant — the served backend is classical.

---

*Cross-references: [`repository_audit.md`](repository_audit.md), [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md), [`export_validation.md`](export_validation.md), [`DEPLOYMENT.md`](DEPLOYMENT.md).*
