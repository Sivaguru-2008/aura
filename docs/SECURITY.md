# AURA Enterprise — Security Policy & RBAC Governance

## 1. Security Architecture & Core Tenets

AURA is engineered to satisfy the stringent cybersecurity, auditability, and data governance standards demanded by clinical hospital enterprises:
* **Zero Trust Boundary**: All incoming requests require strong cryptographic authentication.
* **Tenant Isolation**: Patient cases, model evaluations, and audit records are partitioned by `tenant_id`. Cross-tenant data leakage is strictly prohibited at both the database and gateway layer.
* **Cryptographic Tamper-Evidence**: Inferences, clinician feedback, and administrative actions are committed to an append-only SHA-256 hash-chained audit ledger.
* **Safe Harbor De-identification**: DICOM and tabular evidence vectors strip 18 HIPAA Safe Harbor direct identifiers at the intake gate.

---

## 2. Role-Based Access Control (RBAC)

AURA enforces a 5-tier role hierarchy with explicit privileges:

| Role | Description | Permitted Capabilities |
|---|---|---|
| `AURA_ADMIN` | Global Platform Administrator | Organization provisioning, system-wide configuration, security policy management, global model deployments. |
| `HOSPITAL_ADMIN` | Hospital IT / Department Admin | User provisioning within hospital organization, local model approvals, validation dataset uploads, department audit inspection. |
| `CLINICAL_USER` | Radiologist / Attending Physician | Diagnostic review, study uploads, signing clinical reports, recording confirmed diagnostic outcomes, clinical copilot chat. |
| `AUDITOR` | Compliance Officer / Clinical Auditor | Read-only inspection of tamper-evident audit ledger, model cards, validation run metrics, and calibration histories. |
| `READ_ONLY` | Clinical Observer / Research Fellow | Read-only access to de-identified cases, published model cards, and aggregated performance metrics. Cannot view audit logs or sign reports. |

### Role Hierarchy Inheritance
```
AURA_ADMIN
   └── HOSPITAL_ADMIN
          └── CLINICAL_USER
                 └── AUDITOR
                        └── READ_ONLY
```

---

## 3. Session & Token Management

1. **API Tokens**: Bearer tokens formatted as `aura_tok_*` with database-backed expiration, revocation status, and role scoping.
2. **Session Cookies**: Signed, `HttpOnly`, `SameSite=Strict`, and `Secure` cookies with configurable TTL (default: 12.0 hours, matching one hospital shift).
3. **Revocation**: Disabling a user account or calling `POST /v1/auth/logout` immediately invalidates all active sessions.

---

## 4. Cryptographic Audit Ledger & Provenance

Every case analysis produces a cryptographic provenance hash computed across:
$$	ext{Provenance Hash} = 	ext{SHA256}(	ext{ImageHash} \parallel 	ext{VisionFeatures} \parallel 	ext{ReasoningTrace} \parallel 	ext{SafetyAssessment} \parallel 	ext{DecisionAssurance})$$

The audit ledger (`audit_log` table) maintains an unbroken SHA-256 hash chain:
$$H_t = 	ext{SHA256}(H_{t-1} \parallel 	ext{Timestamp} \parallel 	ext{Action} \parallel 	ext{Actor} \parallel 	ext{EntityID} \parallel 	ext{TenantID} \parallel 	ext{DetailJSON})$$

Any modification, deletion, or re-ordering of historical records breaks the hash chain, immediately alerting auditors during validation runs or CLI checks:
```bash
python -m aura.aura_cli audit-verify
```
