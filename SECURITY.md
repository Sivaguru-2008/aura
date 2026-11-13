# Security & Responsible Clinical Disclosure Policy

The AURA project takes clinical safety, HIPAA safe-harbor compliance, PHI de-identification, and data privacy with utmost seriousness.

## Scope & Zero-PHI Design
AURA is architected to operate **100% offline**:
- Zero telemetry transmission to external cloud services or public LLM APIs.
- No protected health information (PHI) ever leaves the local compute environment.
- Cryptographically pinned artifact checks guard against tampered checkpoints or poisoned weights.

## Reporting a Vulnerability
If you discover a security vulnerability, clinical data leakage issue, or cryptographic weakness:
1. **Do NOT open a public GitHub issue.**
2. Send an encrypted or confidential report directly to:
   - **Lead Researcher & Maintainer:** Sivaguru Murugan
   - **Email:** [sivagurumurugan1@gmail.com](mailto:sivagurumurugan1@gmail.com)
3. Include detailed steps to reproduce the vulnerability, sample inputs (strictly de-identified), and environment configuration.
4. We acknowledge receipt within 48 hours and provide a remediation timeline.

For hospital IT deployment constraints, FHIR/DICOM sandboxing, and enterprise security models, see [docs/HOSPITAL_IT_INSTALL.md](docs/HOSPITAL_IT_INSTALL.md) and [docs/SECURITY.md](docs/SECURITY.md).
