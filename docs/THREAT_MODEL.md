# AURA — Threat Model

**Status (2026-09-13).** Each row names the control that exists in the code today, or the gap
where there is none. `docs/SECURITY.md` has the detail.

## 1. STRIDE

| Category | Threat | Control today | Gap |
|---|---|---|---|
| **Spoofing** | Impersonating a clinician, script or PACS | Shared token or account sign-in; network access refused with no token configured; open access capped at `CLINICAL_USER` | Shared-token callers self-declare their name and role. No SSO or MFA. The DICOM listener does not check AE titles. |
| **Tampering** | Editing audit rows or cases | SHA-256 hash chain over audit rows, serialised appends; per-case provenance hash | `actor` and `tenant_id` are not covered by the chain hash. Model files are not re-verified at load. |
| **Repudiation** | Denying a sign-off or a change | `report.signed` and every mutating request are written to the audit log with the resolved principal | Principals behind the shared token are self-asserted. |
| **Information disclosure** | Patient data leaving through the API | Every `/v1` data route is gated; FHIR/HL7 exports need a credential; per-site databases with site binding | No TLS in the service itself; no encryption at rest; no de-identification of chest uploads. |
| **Denial of service** | Oversized or malformed uploads, request floods | 25 MB upload cap enforced while streaming; extension and MIME allowlist; optional per-principal rate limit; 10 failed sign-ins per minute per address | The rate limit is per process and off by default. |
| **Elevation of privilege** | Reaching administrative endpoints | `require_role` on organisations, users, the registry, validation and audit; hospital admins cannot grant `AURA_ADMIN`; unknown role headers refused | Core clinical endpoints do not check role. |

## 2. Threats specific to clinical AI

**Out-of-distribution input.** A non-chest image or an unfamiliar scanner produces a
plausible-looking diagnosis. *Control:* a modality router and X-ray gate reject non-radiographs
before inference, and an energy-based OOD score above z = 1.5 triggers abstention. *Limit:*
thresholds were set on MIMIC-CXR and have not been re-checked on another site's data.

**Overconfident error.** *Control:* per-backend temperature scaling and split-conformal
prediction sets (90% marginal coverage, valid when a site's cases resemble the calibration
data). AURA abstains when the set holds more than 4 diagnoses or the top probability is below
0.30. *Limit:* coverage is marginal, not per-diagnosis, and rare diagnoses have few
calibration cases.

**Drift after deployment.** *Control:* adaptive conformal inference updates the threshold
from verified outcomes (`POST /v1/cases/{id}/outcome`), and local cohort validation computes
metrics from those outcomes (`services/assurance/local_validation.py`). *Limit:* both depend
on the site actually recording verified outcomes. With none, nothing adapts, and validation
refuses to report.

**Training-data provenance and labels.** Every served model was trained on research-licensed
data with rule-derived labels, and the chest model has no untouched test set.
*Control:* none in code. The position is documented in `docs/DATA_LICENSES.md`.
