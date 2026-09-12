# AURA Enterprise — Threat Model & Security Controls

## 1. Threat Classification (STRIDE Framework)

| Category | Threat Description | Attack Vector | AURA Defensive Control |
|---|---|---|---|
| **Spoofing** | Adversary attempts to impersonate a clinician or external PACS. | Forged HTTP requests or unauthorized DICOM association. | Database session verification, `AURA_AUTH_TOKEN` gating, DICOM AE Title white-listing, and TLS 1.3 encryption. |
| **Tampering** | Malicious alteration of diagnostic models or historical audit records. | Modifying model weights on disk or tampering with database rows. | SHA-256 model binary checksum validation at load time; append-only SHA-256 hash-chained audit ledger. |
| **Repudiation** | Clinician or technician denies signing a report or changing a diagnosis. | Contesting historical interaction. | Cryptographic provenance hash recorded on report signing (`report.signed`), bound to user identity and timestamp. |
| **Information Disclosure** | Unauthorized exposure of PHI across hospital tenants. | Cross-tenant parameter injection (`GET /v1/cases/{id}`). | Strict database-level `tenant_id` scoping; explicit `CrossTenantAccessError` raising 403 Forbidden. |
| **Denial of Service** | Exhaustion of server memory via massive image uploads. | Flooding gateway with gigabyte-sized files or malformed DICOMs. | Hard upload size cap (`AURA_MAX_UPLOAD_MB=25.0`), token bucket rate limiting (`AURA_RATE_LIMIT_RPM`), and synchronous intake cleanup. |
| **Elevation of Privilege** | Clinical user attempts to provision organizations or deploy models. | Invoking administrative endpoints (`/v1/organizations`). | Server-side dependency injection enforcing `require_role(Role.AURA_ADMIN)`. |

---

## 2. Clinical AI-Specific Threat Vectors

### Threat Vector 1: Out-of-Distribution (OOD) Input & Scanner Shift
* **Risk**: An image from an unvalidated scanner, foreign body, or non-chest radiograph is processed, leading the model to hallucinate a plausible diagnosis.
* **Control**: Free-energy OOD detector computes energy score $E(x, T) = -T \log \sum_i e^{z_i / T}$ and standardizes to in-distribution z-score. If $z_{	ext{ood}} \ge 1.5$, the system triggers hard abstention (`ABSTAIN`).

### Threat Vector 2: Overconfident AI Error & Ambiguity
* **Risk**: Model makes a high-consequence misclassification with high probability on an ambiguous clinical film.
* **Control**: Platt-scaled temperature calibration aligns probabilities with empirical prevalence. Split-conformal and Adaptive Conformal Inference (ACI) produce prediction sets guaranteed to cover the true diagnosis $\ge 90\%$ of the time. When the conformal set contains $> 4$ diagnoses or confidence $< 0.30$, the system abstains (`ABSTAIN`).

### Threat Vector 3: Post-Deployment Covariate Shift
* **Risk**: Demographic or seasonal case-mix shifts silently degrade conformal coverage below the $90\%$ threshold.
* **Control**: ACI updates nonconformity threshold $q̂$ online from confirmed pathologist/PCR outcomes:
  $$q̂_{t+1} = 	ext{clip}(q̂_t + \gamma \cdot (\mathbf{1}\{Y_t 
otin \hat{C}_t\} - lpha))$$
  Long-run empirical miscoverage is mathematically guaranteed to converge to target $lpha$ regardless of drift.
