# AURA — Hospital IT Installation Guide

**Status (2026-09-13).** AURA is research software. It is **not** a cleared medical device,
and it does not make an installation compliant with HIPAA, GDPR or any other regime by
itself: compliance is a property of how your organisation runs it. Install it only under a
research or pilot agreement, with a clinician reviewing every output. The data-licence
position of the models it serves is in `docs/DATA_LICENSES.md`, and the clinical limits are
in `docs/KNOWN_LIMITATIONS.md`.

## 1. What you are installing

One AURA server per hospital or site. Each server holds its own case store and its own
hash-chained audit log. Nothing leaves the building on the inference path. The one optional
feature that makes a network call, the chat copilot, only talks to an Ollama server you run
yourself.

| Component | Where | Notes |
|---|---|---|
| Gateway and web console | container `aura-copilot`, port 8000 | Plain HTTP. Put TLS in front of it (§6). |
| DICOM listener | same container, port 11112 | Pilot-grade; read §7 before exposing it. |
| Case store and audit log | SQLite on the `aura-state` volume, or PostgreSQL (`--profile postgres`) | §4, §9 |
| Model weights | baked into the image at build time | Chest DenseNet-121, BraTS brain model, fusion and calibration files |

## 2. Hardware and operating system

- A Linux x86-64 host with Docker Engine 24+ and Docker Compose v2. Docker Desktop is fine
  for evaluation.
- 4+ CPU cores, 8 GB RAM, 20 GB free disk plus room for case data. The container is capped
  at 6 GB (`AURA_MEM_LIMIT`); brain MRI studies need at least 4 GB.
- No GPU is needed. The image runs inference on CPU.

## 3. Network

| Port | Direction | Allow from | Purpose |
|---|---|---|---|
| 8000/tcp | inbound | the reverse proxy only | AURA API and console (HTTP) |
| 443/tcp | inbound | clinical workstations | your reverse proxy (TLS) |
| 11112/tcp | inbound | your PACS only | DICOM listener |
| outbound | none needed at runtime | — | Building the image needs internet access (base image, Python packages, model weights). |

If the proxy runs on the same host, bind AURA to loopback with `AURA_PORT=127.0.0.1:8000`
in `.env`, so workstations cannot bypass the proxy.

## 4. Install

```bash
git clone <your AURA repository> aura && cd aura
cp .env.example .env
```

Edit `.env`:

| Variable | Set it to |
|---|---|
| `AURA_AUTH_TOKEN` | output of `python -c "import secrets;print(secrets.token_urlsafe(32))"`. Store it in your secrets manager. |
| `AURA_SITE_ID` | a stable name for this site, e.g. `general-hospital-main` |
| `AURA_PG_PASSWORD`, `AURA_DATABASE_URL` | only for PostgreSQL, as described in `.env.example` |
| `AURA_BOOTSTRAP_ADMIN_PASSWORD` | optional, 12+ characters; see §5. Remove it after the first start. |

Start it:

```bash
docker compose --profile postgres up -d --build   # PostgreSQL case store
# or
docker compose up -d --build                      # SQLite case store on the volume
```

The image build and every container start run `deploy/preflight.py`. It refuses to start on
a missing model file, an unreachable database, or a database that belongs to another site.
Then check the service end to end:

```bash
docker compose logs aura | grep -A3 "PREFLIGHT"
python deploy/smoke_test.py --url http://localhost:8000 --token "<AURA_AUTH_TOKEN>"
```

The smoke test uploads a real radiograph, checks the result, and confirms that a request
without the token is refused.

Without Docker, run exactly one worker: `uvicorn aura.gateway.app:app --workers 1`. The
gateway keeps per-process state and binds the DICOM port at startup, so a second worker
fails.

## 5. Accounts and access

There are three ways in.

1. **The shared token (`AURA_AUTH_TOKEN`).** Scripts and integrations send it as
   `x-aura-token: <token>` together with `x-aura-user: <name>`. Browsers show a sign-in card
   that asks for a name and the token, then keep a signed, HttpOnly session cookie for
   `AURA_SESSION_TTL_HOURS`. Treat the token as an administrator credential. Rotate it by
   changing `.env` and restarting, which signs every browser out.
2. **User accounts** with roles `AURA_ADMIN`, `HOSPITAL_ADMIN`, `CLINICAL_USER`, `AUDITOR`
   and `READ_ONLY`. A fresh install has **no accounts**. Create the first administrator by
   setting `AURA_BOOTSTRAP_ADMIN_PASSWORD` for the first start (and
   `AURA_BOOTSTRAP_ADMIN_USER` if you don't want the name `admin`), then remove the
   variable. Add further users with `POST /v1/users`. Passwords are stored with scrypt.
   Password sign-in is API-only today (`POST /v1/auth/login`); the web console's sign-in
   card uses the shared token.
3. **Loopback.** A request from the server itself, not relayed by a proxy, gets full access
   for on-box maintenance.

Builds before 2026-09-13 created five accounts whose passwords were published in the source
code. Any account still using one of those passwords is disabled automatically at startup,
and the audit log records it.

`AURA_ALLOW_OPEN_ACCESS=1` lets anyone who can reach port 8000 in without a credential, as a
clinical user in the default organisation. It exists for isolated demo networks only.

## 6. TLS and the reverse proxy

AURA does not terminate TLS. Put your standard reverse proxy in front of it:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto https;
    client_max_body_size 30m;   # AURA_MAX_UPLOAD_MB is 25
}
```

A request that arrives through the proxy counts as remote, so it needs a credential even if
the proxy is on the same host. With `X-Forwarded-Proto: https`, session cookies are marked
`Secure`.

## 7. PACS and DICOM (read before exposing port 11112)

The listener (`aura/services/enterprise/dicom_listener.py`, AE title `AURA_PACS`, port
11112) exists for integration testing. It has **not** been validated against a production
PACS. Allow port 11112 only from your PACS's address, and remove the `11112` port mapping in
`docker-compose.yml` if you are not using it. Treat any DICOM routing as a pilot activity,
done in your integration environment first.

## 8. EHR export

- FHIR R4 `DiagnosticReport` and `Observation` resources: `GET /v1/cases/{case_id}/export/fhir`
- HL7 v2 `ORU^R01`: `GET /v1/cases/{case_id}/export/hl7`

Neither has been validated against an EHR interface engine. Test them against your engine's
non-production channel before routing anything to a patient record.

## 9. Keeping hospitals' data apart

Separation is physical: one AURA server and one database, or one PostgreSQL schema, per
site. On first start `AURA_SITE_ID` is stamped into the database. A server configured for a
different site, or for none, then refuses to open it, and preflight reports why. The
`tenant_id` field inside a database scopes organisations within one site. It is **not** the
boundary between hospitals.

Several sites on one PostgreSQL server: create one schema per site
(`CREATE SCHEMA site_a AUTHORIZATION aura;`) and add `?options=-csearch_path%3Dsite_a` to that
site's `AURA_DATABASE_URL`.

## 10. Backups, audit checks, schema

- **SQLite:**
  `docker compose exec aura python -c "import sqlite3; s=sqlite3.connect('/var/lib/aura/aura.db'); d=sqlite3.connect('/var/lib/aura/backup.db'); s.backup(d)"`,
  then `docker cp aura-copilot:/var/lib/aura/backup.db ./aura-backup.db`.
- **PostgreSQL:** `docker compose exec -T postgres pg_dump -U aura -Fc aura > aura-backup.dump`
- **Restore:** test a restore onto a spare machine before go-live, not after the first
  incident.
- **Audit chain:** `docker compose exec aura python -m aura.aura_cli audit-verify`. The hash
  covers each record's predecessor, action, entity id, detail and timestamp. The actor and
  tenant fields are not yet inside the hash.
- **Schema:** migrations run automatically at startup, and a failed migration stops the
  server. `docker compose exec aura python -m aura.aura_cli migrate current` lists the
  applied versions. Back up before every upgrade: migrations have no tested rollback.

## 11. Upgrades and rollback

```bash
docker tag aura:local aura:before-upgrade    # keep the running image
git pull && docker compose build             # preflight runs inside the build
docker compose up -d
```

To roll back, `docker tag aura:before-upgrade aura:local`, then `docker compose up -d`. If
the upgrade migrated the schema, restore the database backup too.

## 12. Monitoring

- `GET /v1/health` is public and returns status, the fusion backend in use, and the site id.
  The container healthcheck uses it.
- Logs: `docker compose logs aura`. Rotated at 3 × 10 MB.

## 13. Go-live checklist

- [ ] `AURA_AUTH_TOKEN` set and stored in your secrets manager
- [ ] `AURA_SITE_ID` set
- [ ] First administrator created, and `AURA_BOOTSTRAP_ADMIN_PASSWORD` removed from `.env`
- [ ] TLS proxy in front; port 8000 not reachable from workstations directly
- [ ] Port 11112 limited to the PACS, or not mapped
- [ ] Backup job running, and a restore tested
- [ ] Preflight OK; smoke test OK with `--token`
- [ ] Research or pilot agreement and ethics approval in place; clinicians briefed that AURA
      is decision support, not a diagnosis
