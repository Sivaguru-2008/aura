# AURA — Security Model

**Status (2026-09-13).** This describes what the code does today. Where a control people
often expect is missing, it says so in §6 rather than implying it.

## 1. Access modes

The gateway (`aura/gateway/security.py`) decides every request's principal, role and tenant
in one place, before any handler runs.

| Mode | When | Who gets in | Role |
|---|---|---|---|
| **Shared token** | `AURA_AUTH_TOKEN` set | Callers presenting the token (`x-aura-token` or `Authorization: Bearer`) with an `x-aura-user` name, and browsers holding a signed session cookie from `POST /v1/auth/session` | Token callers choose any valid role and tenant with `x-aura-role` / `x-aura-tenant`: the token is a root credential. Browser sessions are `CLINICAL_USER`. |
| **User accounts** | always available once an account exists | `POST /v1/auth/login` with username and password, which returns an `aura_tok_*` session token (also set as a cookie) | The account's role and organisation |
| **Loopback** | no token configured | Direct, un-proxied requests from the machine itself | Full rights, for on-box maintenance and the offline demo |
| **Open access** | no token and `AURA_ALLOW_OPEN_ACCESS=1` | Anyone who can reach the port | Capped at `CLINICAL_USER` in the default organisation. Isolated demo networks only. |

With no token and no open access, a network request gets a 403 that says to set a token.
A request relayed by a proxy (`X-Forwarded-For`, `Forwarded`, `X-Real-IP`) always counts as
remote.

## 2. Roles

| Role | Can additionally |
|---|---|
| `READ_ONLY` | Read the model registry, validation runs and monitoring data |
| `AUDITOR` | Read and verify the audit ledger (`/v1/audit/records`, `/v1/audit/verify`) |
| `CLINICAL_USER` | Queue and execute validation runs; write monitoring metrics and alerts |
| `HOSPITAL_ADMIN` | Everything above within its organisation; create and disable users (not `AURA_ADMIN`); register models and change their status |
| `AURA_ADMIN` | Everything, across organisations; create organisations |

Inheritance (`ROLE_HIERARCHY`): `AURA_ADMIN` ⊃ `HOSPITAL_ADMIN` ⊃ {`CLINICAL_USER`, `AUDITOR`} ⊃
`READ_ONLY`. `CLINICAL_USER` does not include `AUDITOR`.

**Gap:** the core clinical endpoints (upload, case review, feedback, FHIR/HL7 export) require
authentication but do not yet check role. Any authenticated principal can use them.

## 3. Credentials

- **Shared token:** compared in constant time. Rotating it ends every browser session,
  because the session signing key is derived from it.
- **Browser sessions:** `v1.<name>.<expiry>.<HMAC-SHA256>` in an HttpOnly, SameSite=Strict
  cookie, `Secure` behind HTTPS, lifetime `AURA_SESSION_TTL_HOURS`. Nothing is stored
  server-side.
- **Passwords:** scrypt (N=2^14, r=8, p=1) with a per-user salt. Hashes written by builds
  before 2026-09-13 (one round of salted SHA-256) are still accepted and rewritten as scrypt
  at the next successful sign-in.
- **Account sessions:** random `aura_tok_*` tokens. Only their SHA-256 is stored. They expire,
  can be revoked (`POST /v1/auth/logout`), and all of a user's sessions are revoked when the
  account is disabled.
- **No default accounts.** The first administrator comes from
  `AURA_BOOTSTRAP_ADMIN_PASSWORD`, used once while no active account exists. Any account
  still accepting one of the passwords that earlier builds published in the source is
  disabled at startup and recorded in the audit log.
- **Brute force:** failed sign-ins are limited to 10 per minute per client address.

## 4. Separating hospitals

Separation is physical: one server and one database (or schema) per site. `AURA_SITE_ID` is
stamped into the database on first start, and a server configured for another site, or for
none, refuses to open it (`storage.SiteBindingRow`, checked again by preflight). Within one
database, `tenant_id` scopes cases, validation runs, metrics and alerts in application
queries. The database does not enforce it, so it is not the boundary between hospitals.

## 5. Audit and provenance

- **Audit ledger (`audit_log`):** each row stores the SHA-256 of
  `previous_hash ‖ action ‖ entity_id ‖ detail JSON ‖ timestamp`. Appends are serialised by
  a process lock and, on PostgreSQL, a transaction-scoped advisory lock, so concurrent writers
  cannot fork the chain. Verify it with `python -m aura.aura_cli audit-verify` or
  `GET /v1/audit/verify`. **Gap:** `actor` and `tenant_id` are stored but not yet covered by
  the hash, so editing either is not detected.
- **Case provenance:** `compute_provenance_hash` covers the image, vision result, evidence,
  reasoning steps, safety assessment and readiness profile, and is recorded per case in
  `decision_provenance`.

## 6. Not provided — plan for these

- **TLS.** AURA serves plain HTTP. Terminate TLS at a reverse proxy
  (`docs/HOSPITAL_IT_INSTALL.md` §6).
- **Single sign-on, SAML/OIDC, MFA.**
- **De-identification.** Uploads, stored cases and FHIR/HL7 exports can contain PHI. Only the
  MRI metadata layer drops dates; nothing strips identifiers from chest images or reports.
- **Encryption at rest.** Use disk or volume encryption on the host.
- **Model file integrity at load.** Weight checksums are verified when models are fetched
  during the image build (`deploy/fetch_models.py`), not each time they are loaded.
- **DICOM allowlisting.** The listener does not check calling AE titles. Restrict port 11112
  at the firewall.
