# AURA Enterprise — Hospital IT Installation & Deployment Guide

## 1. Executive Summary & Architecture
**AURA** is an enterprise AI assurance, monitoring, and validation platform for healthcare systems. Designed for on-premise deployment in clinical environments, AURA operates entirely within the hospital's secured network boundary, guaranteeing HIPAA compliance, patient privacy, and high-availability clinical decision support.

```
                     +---------------------------------------+
                     | Hospital Network (VLAN / VPN / LAN)   |
                     +---------------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                |                               |
       v                                v                               v
+--------------+               +------------------+             +---------------+
| PACS / VNA   |               | Modality Scanner |             | EHR (Epic /   |
| (DICOM SCU)  |               | (CXR / MRI)      |             | Cerner / FHIR)|
+--------------+               +------------------+             +---------------+
       |                                |                               |
       | DICOM C-STORE (Port 11112)     | DICOM C-STORE                 | HL7 / FHIR
       v                                v                               v
+-------------------------------------------------------------------------------+
| AURA Edge Appliance / Virtual Cluster                                         |
|                                                                               |
|  +---------------------------+        +------------------------------------+  |
|  | DICOM SCP Gateway         |        | Enterprise REST API Gateway        |  |
|  | (Port 11112)              |        | (Port 8000 / Reverse Proxy 443)    |  |
|  +-------------+-------------+        +-----------------+------------------+  |
|                |                                        |                     |
|                +-------------------+--------------------+                     |
|                                    |                                          |
|                                    v                                          |
|                    +-------------------------------+                          |
|                    | AURA Diagnostic Pipeline      |                          |
|                    | - Modality Auto-Detection     |                          |
|                    | - Vision Feature Extraction   |                          |
|                    | - Calibrated Fusion Posterior |                          |
|                    | - OOD Energy Detector         |                          |
|                    | - Conformal Sets (>=90%)      |                          |
|                    | - Decision Assurance Engine   |                          |
|                    |   (ACCEPT / REVIEW / ABSTAIN) |                          |
|                    +---------------+---------------+                          |
|                                    |                                          |
|                +-------------------+--------------------+                     |
|                v                                        v                     |
|  +---------------------------+        +------------------------------------+  |
|  | PostgreSQL 15+ Enterprise |        | Tamper-Evident Audit Ledger        |  |
|  | Multi-Tenant Case Store   |        | SHA-256 Hash Chain                 |  |
|  +---------------------------+        +------------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## 2. System Prerequisites & Hardware Specifications

### Minimum Server Requirements
* **CPU**: 8 physical cores (AMD EPYC or Intel Xeon 2.8 GHz+)
* **Memory**: 32 GB DDR4/DDR5 ECC RAM
* **Disk**: 100 GB SSD (OS & system binaries) + 500 GB NVMe SSD for study staging and database storage
* **GPU**: NVIDIA Ada Lovelace / Ampere (RTX 4090, RTX A4000/A5000, L4, or A10G) with at least 16 GB VRAM and CUDA 12.x drivers. *(CPU fallback mode is supported for non-GPU hospital VMs)*
* **Operating System**: Ubuntu 22.04 LTS, Red Hat Enterprise Linux (RHEL) 9, or Windows Server 2022.

### Network Firewall & Port Allocations
| Port | Protocol | Direction | Source | Purpose |
|---|---|---|---|---|
| **8000** | TCP | Inbound | LAN / Reverse Proxy | AURA Gateway REST API (HTTP) |
| **443** | TCP | Inbound | Hospital Workstations | AURA Web Console & API (HTTPS/TLS 1.3) |
| **11112** | TCP | Inbound | PACS / Modality Scanners | DICOM C-STORE and C-FIND SCP Service |
| **5432** | TCP | Outbound/Local | AURA Application Server | PostgreSQL 15+ Enterprise Database |

---

## 3. Database Setup & Schema Migrations

AURA supports two database engines:
1. **SQLite (`sqlite:///artifacts/aura.db`)**: For zero-configuration single-box deployments, offline field clinics, and isolated evaluations.
2. **PostgreSQL 15+**: Production certified engine for hospital deployments requiring concurrent multi-user access and high throughput.

### PostgreSQL Configuration
Create the database and service user:
```sql
CREATE USER aura WITH ENCRYPTED PASSWORD 'secure_production_password_here';
CREATE DATABASE aura OWNER aura;
GRANT ALL PRIVILEGES ON DATABASE aura TO aura;
```

### Running Schema Migrations
AURA manages database schema versions automatically via its built-in idempotent migration runner.

Run all forward migrations:
```bash
python -m aura.gateway.migrations.runner upgrade
```
To verify current migration status:
```bash
python -m aura.gateway.migrations.runner status
```

Applied migrations:
* `001_initial_core`: Core tables (`cases`, `audit_log`, `feedback`, `outcomes`, `conformal_state`, `site_binding`, `intake_baselines`).
* `002_multi_tenancy_rbac`: Tenancy and user governance (`organizations`, `users`, `sessions`, `projects`).
* `003_model_registry_validation`: Model governance (`model_registry`, `validation_runs`).
* `004_monitoring_alerts`: Real-time telemetry (`system_metrics`, `system_alerts`).

---

## 4. Environment Variables & Secret Configuration

Create `/etc/aura/aura.env` (or `.env` in the application directory) with mode `600`:

```ini
# --- Database Configuration ---
AURA_DATABASE_URL=postgresql+psycopg://aura:secure_production_password_here@localhost:5432/aura
AURA_SITE_ID=hospital_saint_jude_campus

# --- Security & Authentication ---
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
AURA_AUTH_TOKEN=k8J2n_F9x1aZbQ7vL0pYe3wR6mTu4iOc8sH5dG1vK2M=
AURA_AUTH_HEADER=x-aura-token
AURA_SESSION_TTL_HOURS=12.0
AURA_ALLOW_OPEN_ACCESS=false
AURA_MAX_UPLOAD_MB=25.0
AURA_RATE_LIMIT_RPM=600

# --- Clinical Safety & Decision Thresholds ---
AURA_FUSION_BACKEND=classical
AURA_ACTIVE_POLICY=community_conservative
AURA_CONFORMAL_COVERAGE=0.90
AURA_OOD_ENERGY_THRESHOLD=1.5
AURA_LOW_CONFIDENCE_THRESHOLD=0.30
AURA_ABSTENTION_CONFORMAL_SIZE=4
AURA_EPISTEMIC_ABSTENTION_THRESHOLD=0.20
AURA_ACI_ENABLED=true
AURA_ACI_GAMMA=0.02

# --- Enterprise Integrations ---
AURA_DICOM_PORT=11112
AURA_DICOM_AE_TITLE=AURA_PACS
```

---

## 5. Systemd Service Deployment

Create systemd services for Linux deployments.

### 1. AURA Gateway (`/etc/systemd/system/aura-gateway.service`)
```ini
[Unit]
Description=AURA AI Enterprise Gateway
After=network.target postgresql.service

[Service]
Type=simple
User=aura
Group=aura
WorkingDirectory=/opt/aura/aura-main
EnvironmentFile=/etc/aura/aura.env
ExecStart=/opt/aura/venv/bin/uvicorn aura.gateway.app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable aura-gateway
sudo systemctl start aura-gateway
sudo systemctl status aura-gateway
```

---

## 6. Preflight & Smoke Verification

Always run the deployment preflight verification tool before opening clinical access:

```bash
# 1. Verify dependencies, model weights, database connectivity, and security configuration
python deploy/preflight.py

# 2. Run end-to-end integration smoke test with live model inference
python deploy/smoke_test.py --token <AURA_AUTH_TOKEN>
```

Expected output:
```
==========================================================================
PREFLIGHT OK
==========================================================================
SMOKE TEST OK - 18 checks passed.
```

---

## 7. PACS (DICOM) & EHR (HL7 / FHIR) Integration

### DICOM Configuration
Configure your Hospital PACS / VNA or Modality Worklist to send studies to AURA:
* **Called AE Title**: `AURA_PACS`
* **Port**: `11112`
* **Supported Presentation Contexts**:
  * Digital X-Ray Image Storage (`1.2.840.10008.5.1.4.1.1.1.1`)
  * Computed Radiography Image Storage (`1.2.840.10008.5.1.4.1.1.1`)
  * MR Image Storage (`1.2.840.10008.5.1.4.1.1.4`)
* **Query / Retrieve**: Supports Patient Root & Study Root C-FIND SCP.

### FHIR R4 Integration
AURA emits standard HL7 FHIR R4 `DiagnosticReport` and `Observation` resources with embedded tamper-evident provenance hashes and Decision Assurance extensions:
* `GET /v1/cases/{case_id}/fhir/report` -> Returns FHIR DiagnosticReport.
* `GET /v1/cases/{case_id}/fhir/observations` -> Returns LOINC-coded observations.

### HL7 v2.5 Integration
AURA generates HL7 `ORU^R01` unsolicited transmission observation messages:
* Header MSH includes sending facility `AURA^{tenant_id}`.
* OBX segments carry `DECISION_VERDICT`, `CONFORMAL_SET`, calibrated findings, and the SHA-256 `PROVENANCE_HASH`.

---

## 8. Backup & Disaster Recovery

### Database Backup
Execute automated nightly backups via cron:
```bash
pg_dump -U aura -Fc aura > /backup/aura/aura_db_$(date +\%Y\%m\%d).dump
```

### Cryptographic Audit Ledger Verification
To audit the integrity of all logged inferences and verify that no audit records have been manipulated:
```bash
python -m aura.aura_cli audit-verify
```
