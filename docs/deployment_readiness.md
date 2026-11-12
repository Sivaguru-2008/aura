# AURA — Deployment Readiness Assessment

**Assessment date:** 2026-09-13
**Verdict:** **GO** for an offline, single-site research pilot with a clinician reviewing every
output. **NOT** for autonomous clinical use. **NOT** for sale until the data-licence position is
resolved (`docs/DATA_LICENSES.md`).

For installation see [`HOSPITAL_IT_INSTALL.md`](HOSPITAL_IT_INSTALL.md). This file records
whether each readiness dimension is satisfied today, and what the evidence is.

---

## 1. Scorecard

| Dimension | Status | Evidence / note |
| --- | --- | --- |
| Chest inference | ✅ Research-ready | DenseNet-121 (epoch 7), macro AUROC 0.821 on 2,099 MIMIC-CXR validation images. The same images selected the epoch, so there is no untouched test set. |
| Brain inference | ✅ Research-ready | BraTS 2020 residual U-Net; needs all four sequences. |
| Served fusion backend | ✅ `classical` | `aura/pyproject.toml`; chosen on the preregistered CheXpert external validation (`docs/MARKET_COMPARISON.md` §3.2). |
| Access control | ✅ Secure by default | No token: this machine only. Token: header or signed session cookie. Accounts: scrypt, none seeded. Covered by `test_secure_defaults.py`, `test_accounts_and_validation.py` and `test_rbac.py`. Sign-in, sign-out and the HttpOnly cookie were also checked in a browser on 2026-09-13. |
| Separating sites | ✅ | `AURA_SITE_ID` binding, checked at startup and by preflight. |
| Case store: SQLite | ✅ | Default. |
| Case store: PostgreSQL | ⚠️ Written, not yet run live | The PostgreSQL leg of `tests/test_storage_backends.py` was skipped: Docker Desktop failed to start on 2026-09-13. |
| Container image | ⚠️ Not rebuilt since the 2026-09-13 changes | Same Docker failure. The last verified build was 2026-07-29. |
| Pinned dependencies | ⛔ Not done | Planned as a lock file taken from a verified image build. |
| Test suite | ✅ | 841 passed, 49 skipped, 0 failed (non-slow), 2026-09-13. |
| Local validation of results | ✅ Honest by construction | Computed from verified outcomes; refuses below 20 (`services/assurance/local_validation.py`). |
| Clinical validation | ⛔ | No radiologist-read test set, no reader study, no prospective use. |
| Regulatory clearance | ⛔ None | Research and decision support only. |
| Data licences | ⛔ Unresolved for commercial use | `docs/DATA_LICENSES.md`. |

---

## 2. Before a pilot goes live

- [ ] Docker working again: rebuild the image, then
      `python deploy/smoke_test.py --url http://localhost:8000 --token "$AURA_AUTH_TOKEN"`.
- [ ] PostgreSQL, if used:
      `AURA_TEST_DATABASE_URL=postgresql+psycopg://<user>:<pw>@<host>:5432/<db> python -m pytest tests/test_storage_backends.py`,
      with no skips.
- [ ] Every item in `HOSPITAL_IT_INSTALL.md` §13: token, site id, bootstrap admin removed, TLS
      proxy, firewall, backups with a tested restore.
- [ ] A database that ran the quarantined pilot script has been cleaned
      (`E:\AURA\_quarantine_fabricated_pilot_20260913\cleanup_live_db.py`), or replaced.
- [ ] A research or pilot agreement and ethics approval are in place.

---

## 3. Do-not-deploy conditions

- Autonomous diagnosis or triage without a clinician in the loop.
- Capabilities listed in `NeuroMindEngine.PLANNED_CAPABILITIES` (subtype classification,
  haemorrhage, midline shift, …): roadmap, not shipped.
- 2D PNG/JPEG exports of MR studies: refused by design.
- `AURA_ALLOW_OPEN_ACCESS=1` on any network a patient-data workstation can reach.
- `AURA_FUSION_BACKEND=quantum` as the served model: it was the least accurate of four
  backends on the external validation.
- Any figure from `artifacts/brain/qkl_full_coverage.json`: withdrawn; see the provenance
  audit in `DATA_LICENSES.md`.

---

*Cross-references: [`HOSPITAL_IT_INSTALL.md`](HOSPITAL_IT_INSTALL.md), [`SECURITY.md`](SECURITY.md),
[`THREAT_MODEL.md`](THREAT_MODEL.md), [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md),
[`DATA_LICENSES.md`](DATA_LICENSES.md).*
