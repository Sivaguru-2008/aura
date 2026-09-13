"""Persistence — SQLite or PostgreSQL behind a small repository API.

SQLite at ``DB_PATH`` is the default and needs no setup, which is what a single-box
deployment wants. Setting ``AURA_DATABASE_URL`` moves the case store, outcomes, ACI
state and the hash-chained audit ledger to PostgreSQL with no other change. Case
bundles are stored as JSON documents with indexed columns for worklist queries;
feedback and audit_log are first-class.

Two engine differences are handled here rather than left to callers: PostgreSQL
sessions are pinned to UTC (the audit hash covers naive timestamps), and hash-chain
appends take a lock so concurrent writers cannot fork the chain (``_chain_append``).

``AURA_DATABASE_URL`` does not move the demo EHR fixture (``services/enterprise/
ehr.py``) or the SQLite embedding cache (``services/memory/vectorstore.py``); both keep
their local file at ``DB_PATH``. Neither is the record of what AURA decided.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    text,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from aura.schemas.contracts import CaseBundle

_GENESIS_HASH = "0" * 64

#: pg_advisory_xact_lock key for hash-chain appends ("AURA" in ASCII).
_CHAIN_LOCK_KEY = 0x41555241
_CHAIN_MUTEX = threading.Lock()


class SiteMismatchError(RuntimeError):
    """This server's AURA_SITE_ID does not match the site the database belongs to."""


class CrossTenantAccessError(RuntimeError):
    """Attempted cross-tenant access to an unauthorized organization's resource."""


class AuthenticationError(RuntimeError):
    """Authentication failed or credentials invalid."""


class AuthorizationError(RuntimeError):
    """Principal lacks required permissions/roles."""


def engine_for(url: str) -> Engine:
    """Create the SQLAlchemy engine for a case-store URL.

    ``postgresql://`` and ``postgres://`` are rewritten to the psycopg 3 driver the
    image ships (SQLAlchemy's bare ``postgresql`` means psycopg2, which it does not).
    PostgreSQL sessions are pinned to UTC, appended to any ``options`` already in the
    URL so a per-site ``search_path`` survives. ``created_at`` is stored without a
    zone and the audit hash covers its naive ISO form, so a server whose TimeZone is
    not UTC would shift every timestamp on read-back and fail verification.
    """
    u = make_url(url)
    if u.drivername in ("postgresql", "postgres"):
        u = u.set(drivername="postgresql+psycopg")
    if u.get_backend_name() == "postgresql":
        existing = u.query.get("options", "")
        if isinstance(existing, tuple):
            existing = " ".join(existing)
        u = u.update_query_dict({"options": f"{existing} -c timezone=UTC".strip()})
        return create_engine(u, future=True, pool_pre_ping=True)
    return create_engine(u, future=True)


@contextmanager
def _chain_append(engine: Engine):
    """A session in which reading the chain head and appending to it is atomic.

    Every append reads the previous record's hash and writes the next one. Two
    writers doing that at once both link to the same predecessor and fork the chain,
    which ``verify_audit_trail`` then reports as tampering. FastAPI runs sync handlers
    on a thread pool, so this can happen inside one process: the mutex covers threads,
    and on PostgreSQL a transaction-scoped advisory lock covers other processes.
    """
    with _CHAIN_MUTEX, Session(engine) as ses:
        if ses.get_bind().dialect.name == "postgresql":
            ses.execute(text("SELECT pg_advisory_xact_lock(CAST(:k AS bigint))"),
                        {"k": _CHAIN_LOCK_KEY})
        yield ses


def _compute_audit_hash(previous_hash: str, action: str, entity_id: str,
                        detail: dict, timestamp: datetime) -> str:
    ts = timestamp.replace(tzinfo=None).isoformat()
    payload = previous_hash + action + entity_id + json.dumps(detail, sort_keys=True, default=str) + ts
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_audit_trail(store=None, *, db_path: Path | None = None) -> bool:
    if store is None and db_path is not None:
        store = Store(db_path)
    if store is None:
        return True
    with Session(store.engine) as ses:
        rows = ses.execute(
            select(AuditRow).order_by(AuditRow.id.asc())
        ).scalars().all()
    prev_hash = _GENESIS_HASH
    for row in rows:
        if row.previous_hash != prev_hash:
            return False
        expected = _compute_audit_hash(
            prev_hash, row.action, row.entity_id, row.detail, row.created_at
        )
        if row.record_hash != expected:
            return False
        prev_hash = row.record_hash
    return True


class Base(DeclarativeBase):
    pass


class CaseRow(Base):
    __tablename__ = "cases"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    study_id: Mapped[str] = mapped_column(String, index=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True, default="default")
    model_version: Mapped[str] = mapped_column(String, default="vision-cxr-densenet121-production")
    decision: Mapped[str] = mapped_column(String, default="ACCEPT")
    decision_reason: Mapped[str] = mapped_column(Text, default="")
    state: Mapped[str] = mapped_column(String, index=True)
    priority_score: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    top_diagnosis: Mapped[str] = mapped_column(String, default="")
    top_probability: Mapped[float] = mapped_column(Float, default=0.0)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    ground_truth: Mapped[str] = mapped_column(String, default="")
    bundle: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String, index=True)
    diagnosis: Mapped[str] = mapped_column(String, default="")
    verdict: Mapped[str] = mapped_column(String)
    correction: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ConformalStateRow(Base):
    """Persisted Adaptive Conformal Inference (ACI) state — one row per stream.

    ``stream`` is "global" for the marginal q̂, or a class name for a Mondrian-ACI
    hybrid. The whole ACIState (q̂, α, γ, t, rolling window) lives in ``state`` as
    JSON so the online coverage guarantee survives process restarts on the edge box.
    """
    __tablename__ = "conformal_state"
    stream: Mapped[str] = mapped_column(String, primary_key=True, default="global")
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class OutcomeRow(Base):
    """Confirmed patient outcomes — the ground-truth stream that drives ACI.

    Append-only log of (case, emitted set, confirmed diagnosis, coverage hit/miss)
    so the adaptation is auditable and replayable."""
    __tablename__ = "outcomes"
    __table_args__ = (UniqueConstraint("case_id", name="uq_outcome_case_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String, index=True)
    true_diagnosis: Mapped[str] = mapped_column(String, default="")
    covered: Mapped[bool] = mapped_column(Boolean, default=False)
    qhat_after: Mapped[float] = mapped_column(Float, default=0.0)
    localized_coverage: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class AuditRow(Base):
    __tablename__ = "audit_log"          # append-only by convention
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String, default="system")
    action: Mapped[str] = mapped_column(String)
    entity_type: Mapped[str] = mapped_column(String, default="")
    entity_id: Mapped[str] = mapped_column(String, default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    previous_hash: Mapped[str] = mapped_column(String, default=_GENESIS_HASH)
    record_hash: Mapped[str] = mapped_column(String, default=_GENESIS_HASH)
    tenant_id: Mapped[str] = mapped_column(String, index=True, default="default")
    model_version: Mapped[str] = mapped_column(String, default="")
    aura_version: Mapped[str] = mapped_column(String, default="0.1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class IntakeBaselineRow(Base):
    """A clinical intake recorded *before* any imaging arrives.

    Deliberately NOT a ``CaseRow``. A pre-imaging intake has no study, no image
    and no posterior, so it is not a case bundle: putting it in ``cases`` would
    surface an imaging-less row in the worklist and in ``/history``, and would
    move ``Store.count()``, which the upload adapters use to mint
    ``CASE-UPLOAD-{n}`` (deleting the row on attach would then let the next
    upload collide with an existing case id and overwrite it).

    Rows here are short-lived: created when the clinician submits the intake
    form, consumed by ``POST /v1/cases/{case_id}/attach`` once the scan has been
    analysed, or dropped by ``DELETE /v1/cases/{case_id}``.
    """

    __tablename__ = "intake_baselines"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    attached_case_id: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class NeuroViewRow(Base):
    """MRI-only visualization artifact keyed by case id."""

    __tablename__ = "neuroview_artifacts"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    study_id: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class MemoryRecordRow(Base):
    __tablename__ = "memory_records"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    diagnosis: Mapped[str] = mapped_column(String, default="")
    embedding_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatMessageRow(Base):
    __tablename__ = "chat_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class DecisionProvenanceRow(Base):
    """Decision provenance graph — complete pipeline path for one case.

    Records the chain:
        Image Hash → Vision Result → Evidence Graph → Reasoning Steps →
        Safety Verification → Safety Controller → DRP
    """
    __tablename__ = "decision_provenance"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance_hash: Mapped[str] = mapped_column(String, default=_GENESIS_HASH)
    previous_hash: Mapped[str] = mapped_column(String, default=_GENESIS_HASH)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class SiteBindingRow(Base):
    """Which hospital/site this database belongs to. At most one row (id = 1).

    Data separation between hospitals is physical: one AURA server and one database
    (or one PostgreSQL schema) per site, never rows from several sites in one table.
    This row makes that arrangement fail safe. The first server started with
    ``AURA_SITE_ID`` stamps its site here; a later server configured for a different
    site, or for none, refuses to open the database rather than serve one hospital's
    cases under another's name.
    """
    __tablename__ = "site_binding"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str] = mapped_column(String)
    bound_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class OrganizationRow(Base):
    __tablename__ = "organizations"
    org_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class UserRow(Base):
    __tablename__ = "users"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="CLINICAL_USER")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SessionRow(Base):
    __tablename__ = "sessions"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))
    token_hash: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class ProjectRow(Base):
    __tablename__ = "projects"
    project_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class RegisteredModelRow(Base):
    __tablename__ = "model_registry"
    model_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True, default="system")
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    modality: Mapped[str] = mapped_column(String(64))
    intended_use: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(255), default="")
    training_data_provenance: Mapped[str] = mapped_column(Text, default="")
    validation_dataset: Mapped[str] = mapped_column(String(255), default="")
    validation_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    checksum_sha256: Mapped[str] = mapped_column(String(64), default="")
    approval_status: Mapped[str] = mapped_column(String(32), default="APPROVED")
    deployment_status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    retired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ValidationRunRow(Base):
    __tablename__ = "validation_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    dataset_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")  # QUEUED, RUNNING, COMPLETED, FAILED
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    subgroup_analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    failure_analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    report_markdown: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SystemMetricRow(Base):
    __tablename__ = "system_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[float] = mapped_column(Float)
    dimensions: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class AlertRow(Base):
    __tablename__ = "system_alerts"
    alert_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    model_id: Mapped[str] = mapped_column(String(128), default="")
    severity: Mapped[str] = mapped_column(String(32), default="WARNING")  # INFO, WARNING, CRITICAL
    category: Mapped[str] = mapped_column(String(64))  # OOD_SPIKE, DRIFT, UNCERTAINTY, SYSTEM
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Store:
    def __init__(self, db_path: Path | None = None, url: str | None = None, site_id: str | None = None):
        """Open (creating if needed) the case store.

        ``db_path`` is the SQLite file used when ``url`` is empty; it stays the home of
        the local EHR fixture and embedding cache either way (see the module docstring).
        ``url`` is any SQLAlchemy URL, normally ``AURA_DATABASE_URL``. ``site_id`` is
        ``AURA_SITE_ID``; see :class:`SiteBindingRow`.
        """
        if db_path is None:
            from aura.common.config import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        if url:
            self.engine = engine_for(url)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        Base.metadata.create_all(self.engine)
        # Schema upgrades for databases created by an older AURA. A failed migration
        # used to be swallowed here, which left a half-upgraded schema that broke later
        # on some unrelated query, far from the cause. Stop at startup instead.
        from aura.gateway.migrations import MigrationRunner
        try:
            MigrationRunner(self.engine).apply_all()
        except Exception as exc:
            raise RuntimeError(
                "database schema migration failed on "
                f"{self.engine.url.render_as_string(hide_password=True)}: "
                f"{type(exc).__name__}: {exc}") from exc
        self.site_id = self._bind_site((site_id or "").strip())
        self._seed_defaults()

    # ---- passwords ----
    # scrypt (RFC 7914) from the standard library: memory-hard, so offline guessing
    # against a stolen users table is expensive. Stored as "scrypt$n$r$p$<hex>" in
    # password_hash, with the salt in its own column.
    _SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

    #: Passwords that earlier code published in its source: the five accounts builds
    #: before 2026-09-13 seeded on every fresh database, and the one password the
    #: quarantined pilot script gave every account it created. An account that still
    #: accepts one is a known credential rather than an account, so it is disabled at
    #: startup. All of them were hashed by the legacy salted-SHA-256 code, so only legacy
    #: hashes are checked, which keeps the scan cheap.
    _PUBLISHED_PASSWORDS = (
        "AdminSecret123!", "HospitalAdmin123!", "Clinician123!", "Auditor123!",
        "Observer123!", "HospitalPilotSecure2026!",
    )

    @staticmethod
    def hash_password(password: str) -> tuple[str, str]:
        salt = secrets.token_hex(16)
        n, r, p = Store._SCRYPT_N, Store._SCRYPT_R, Store._SCRYPT_P
        digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt),
                                n=n, r=r, p=p, dklen=32).hex()
        return f"scrypt${n}${r}${p}${digest}", salt

    @staticmethod
    def verify_password(password: str, salt: str, expected_hash: str) -> bool:
        if expected_hash.startswith("scrypt$"):
            try:
                _, n, r, p, digest = expected_hash.split("$")
                got = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt),
                                     n=int(n), r=int(r), p=int(p), dklen=32).hex()
            except (ValueError, TypeError):
                return False
            return secrets.compare_digest(got, digest)
        # Legacy: one round of salted SHA-256, written by builds before 2026-09-13.
        # Still accepted so existing accounts can sign in; the login endpoint rewrites
        # the row as scrypt on the first successful sign-in (upgrade_password_hash).
        legacy = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
        return secrets.compare_digest(legacy, expected_hash)

    @staticmethod
    def password_needs_upgrade(expected_hash: str) -> bool:
        return not expected_hash.startswith("scrypt$")

    def upgrade_password_hash(self, user_id: str, password: str) -> None:
        pw_hash, salt = self.hash_password(password)
        with Session(self.engine) as ses:
            u = ses.get(UserRow, user_id)
            if u is not None:
                u.password_hash, u.salt = pw_hash, salt
                ses.commit()

    def _disable_known_default_accounts(self) -> None:
        disabled: list[str] = []
        with Session(self.engine) as ses:
            for u in ses.execute(
                select(UserRow).where(UserRow.is_active.is_(True))
            ).scalars().all():
                if not self.password_needs_upgrade(u.password_hash):
                    continue                          # scrypt: written by current code
                if any(self.verify_password(pw, u.salt, u.password_hash)
                       for pw in self._PUBLISHED_PASSWORDS):
                    u.is_active = False
                    u.disabled_at = datetime.now(timezone.utc)
                    for sess in ses.execute(
                        select(SessionRow).where(SessionRow.user_id == u.user_id)
                    ).scalars():
                        sess.revoked = True
                    disabled.append(u.username)
            if disabled:
                ses.commit()
        for username in disabled:
            self.audit("security.default_account_disabled", entity_type="user",
                       entity_id=username,
                       detail={"reason": "password was a published default"})

    def _bootstrap_admin(self) -> None:
        """Create the first administrator from AURA_BOOTSTRAP_ADMIN_PASSWORD, once.

        Only while no active account exists, so the variable is inert after the first
        start and should then be removed. Username: AURA_BOOTSTRAP_ADMIN_USER
        (default ``admin``).
        """
        import os

        password = os.environ.get("AURA_BOOTSTRAP_ADMIN_PASSWORD", "")
        if not password:
            return
        with Session(self.engine) as ses:
            if ses.execute(select(UserRow.user_id).where(UserRow.is_active.is_(True))).first():
                return
        if len(password) < 12:
            raise RuntimeError("AURA_BOOTSTRAP_ADMIN_PASSWORD must be at least 12 characters")
        username = os.environ.get("AURA_BOOTSTRAP_ADMIN_USER", "").strip() or "admin"
        try:
            user = self.create_user(username=username, password=password,
                                    role="AURA_ADMIN", org_id="default")
        except ValueError as exc:
            raise RuntimeError(
                f"cannot create the bootstrap administrator: {exc}. If that is a disabled "
                f"default account, set AURA_BOOTSTRAP_ADMIN_USER to another name.") from exc
        self.audit("security.bootstrap_admin_created", entity_type="user",
                   entity_id=user["user_id"], detail={"username": username})

    def _seed_defaults(self) -> None:
        """Seed the default organization and the model registry cards.

        No user accounts are seeded. Builds before 2026-09-13 created five accounts
        with passwords written in this file, so every fresh database could be opened
        by anyone who had read the source. The first administrator now comes from
        ``AURA_BOOTSTRAP_ADMIN_PASSWORD`` (:meth:`_bootstrap_admin`); without it the
        server runs on the shared-token and loopback modes only.
        """
        self._disable_known_default_accounts()
        self._bootstrap_admin()
        with Session(self.engine) as ses:
            # 1. Default Organization
            if not ses.get(OrganizationRow, "default"):
                ses.add(OrganizationRow(
                    org_id="default", name="Default Hospital", slug="default", is_active=True
                ))

            # Registry cards for the models this build serves, derived from the artifacts
            # on disk (services/models/cards.py) rather than typed in, so a card cannot
            # claim a figure the checkout does not contain. They are written
            # PENDING_REVIEW: deployed is a fact, approved is a claim nobody has made. A
            # person approves through POST /v1/models/registry/{id}/status, which is
            # audited. Cards the 2026-09-12 seed left APPROVED are rewritten.
            from aura.services.models.cards import (
                SERVED_MODEL_IDS, SUPERSEDED_SEED_DATASETS, served_model_cards)
            existing = {r.model_id: r for r in ses.execute(
                select(RegisteredModelRow).where(
                    RegisteredModelRow.model_id.in_(SERVED_MODEL_IDS))).scalars()}
            stale = [mid for mid, r in existing.items()
                     if r.approval_status == "APPROVED" and r.org_id == "system"
                     and r.validation_dataset in SUPERSEDED_SEED_DATASETS]
            corrected: list[str] = []
            if stale or len(existing) < len(SERVED_MODEL_IDS):
                for mid, card in served_model_cards().items():
                    row = existing.get(mid)
                    if row is None:
                        ses.add(RegisteredModelRow(**card))
                    elif mid in stale:
                        for key, value in card.items():
                            setattr(row, key, value)
                        corrected.append(mid)

            ses.commit()
        for mid in corrected:
            self.audit("registry.card_corrected", entity_type="model", entity_id=mid,
                       detail={"reason": "seeded APPROVED with figures that did not "
                                         "trace to an artifact"})

    def _bind_site(self, configured: str) -> str:
        """Stamp or check the site this database belongs to (see SiteBindingRow)."""
        with Session(self.engine) as ses:
            row = ses.get(SiteBindingRow, 1)
            bound = row.site_id if row else ""
            if not bound and configured:
                ses.add(SiteBindingRow(id=1, site_id=configured))
                try:
                    ses.commit()
                except IntegrityError:               # another server bound it first
                    ses.rollback()
                    bound = ses.get(SiteBindingRow, 1).site_id
                else:
                    self.audit("site.bound", entity_type="system", entity_id=configured,
                               detail={"site_id": configured})
                    return configured
        if bound and bound != configured:
            where = self.engine.url.render_as_string(hide_password=True)
            if configured:
                raise SiteMismatchError(
                    f"refusing to open {where}: it belongs to site '{bound}', and this "
                    f"server is configured as AURA_SITE_ID='{configured}'. Point "
                    f"AURA_DATABASE_URL / AURA_DB_PATH at this site's own database.")
            raise SiteMismatchError(
                f"refusing to open {where}: it belongs to site '{bound}', and this server "
                f"has no AURA_SITE_ID. Set AURA_SITE_ID='{bound}' if this server is that site.")
        return bound

    # ---- cases ----
    def save_case(self, bundle: CaseBundle, tenant_id: str = "default",
                  model_version: str = "vision-cxr-densenet121-production",
                  decision: str = "ACCEPT", decision_reason: str = "") -> None:
        data = json.loads(bundle.model_dump_json())
        s = bundle.safety
        with Session(self.engine) as ses:
            row = ses.get(CaseRow, bundle.case_id)
            if row is None:
                row = CaseRow(case_id=bundle.case_id)
                ses.add(row)
            row.study_id = bundle.study_id
            row.state = bundle.state.value
            row.priority_score = bundle.priority_score
            row.top_diagnosis = s.top.value if s else ""
            row.top_probability = s.top_probability if s else 0.0
            row.abstained = bool(s.abstained) if s else False
            row.ground_truth = bundle.ground_truth.value if bundle.ground_truth else ""
            row.tenant_id = tenant_id or "default"
            row.model_version = model_version or (bundle.vision.model_version if bundle.vision else "") or "vision-cxr-densenet121-production"
            if getattr(bundle, "decision_assurance", None) is not None:
                row.decision = bundle.decision_assurance.verdict.value
                row.decision_reason = bundle.decision_assurance.summary_reason()
            elif s and s.abstained:
                row.decision = "ABSTAIN"
                row.decision_reason = getattr(s, "abstention_reason", "") or "High uncertainty / OOD detected"
            else:
                row.decision = decision or "ACCEPT"
                row.decision_reason = decision_reason
            row.bundle = data
            ses.commit()

    def get_case(self, case_id: str, tenant_id: str | None = None) -> CaseBundle | None:
        with Session(self.engine) as ses:
            row = ses.get(CaseRow, case_id)
            if not row:
                return None
            if tenant_id and tenant_id not in ("default", "system", "*") and getattr(row, "tenant_id", "default") != tenant_id:
                raise CrossTenantAccessError(
                    f"Access denied: case {case_id} belongs to tenant '{getattr(row, 'tenant_id', 'default')}', not '{tenant_id}'"
                )
            b = CaseBundle.model_validate(row.bundle)
            from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
            b.dx_labels = {d.value: l for d, l in DIAGNOSIS_LABELS.items()}
            b.ev_labels = {f.value: l for f, l in FINDING_LABELS.items()}
            return b

    def list_cases(self, state: str | None = None, limit: int = 200, tenant_id: str | None = None) -> list[dict]:
        """Lightweight worklist rows (not full bundles)."""
        with Session(self.engine) as ses:
            stmt = select(CaseRow)
            if state:
                stmt = stmt.where(CaseRow.state == state)
            if tenant_id and tenant_id not in ("system", "*"):
                stmt = stmt.where(CaseRow.tenant_id == tenant_id)
            stmt = stmt.order_by(CaseRow.priority_score.desc()).limit(limit)
            rows = ses.execute(stmt).scalars().all()
            out = []
            from aura.schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
            for r in rows:
                b = r.bundle
                try:
                    dx_enum = Diagnosis(r.top_diagnosis)
                    label = DIAGNOSIS_LABELS.get(dx_enum, r.top_diagnosis)
                except ValueError:
                    label = r.top_diagnosis
                out.append({
                    "case_id": r.case_id,
                    "study_id": r.study_id,
                    "tenant_id": getattr(r, "tenant_id", "default"),
                    "patient_id": b.get("patient_id", "") or "",
                    "state": r.state,
                    "priority_score": r.priority_score,
                    "top_diagnosis": r.top_diagnosis,
                    "top_diagnosis_label": label,
                    "top_probability": r.top_probability,
                    "abstained": r.abstained,
                    "decision": getattr(r, "decision", "ACCEPT"),
                    "decision_reason": getattr(r, "decision_reason", ""),
                    "model_version": getattr(r, "model_version", ""),
                    "backend": (b.get("fusion") or {}).get("backend", ""),
                    "conformal_set": (b.get("safety") or {}).get("conformal_set", []),
                    "priors": b.get("priors", {}),
                    "created_at": r.created_at.isoformat(),
                })
            return out

    def count(self) -> int:
        with Session(self.engine) as ses:
            return ses.query(CaseRow).count()

    # ---- pre-imaging intake baselines ----
    def next_intake_id(self) -> str:
        """Mint the next ``CASE-INTAKE-{n}`` id.

        Numbered off the highest suffix present rather than off a row count, so an
        attached baseline (which is retained) can never have its id handed out
        again. A discarded baseline's id may be reused — nothing references it.
        """
        with Session(self.engine) as ses:
            rows = ses.execute(select(IntakeBaselineRow.case_id)).scalars().all()
        top = 0
        for cid in rows:
            try:
                top = max(top, int(str(cid).rsplit("-", 1)[-1]))
            except ValueError:
                continue
        return f"CASE-INTAKE-{top + 1}"

    def save_intake(self, case_id: str, payload: dict, priority_score: float = 0.0) -> None:
        with Session(self.engine) as ses:
            row = ses.get(IntakeBaselineRow, case_id)
            if row is None:
                row = IntakeBaselineRow(case_id=case_id)
                ses.add(row)
            row.payload = payload
            row.priority_score = float(priority_score)
            ses.commit()

    def get_intake(self, case_id: str) -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(IntakeBaselineRow, case_id)
            return dict(row.payload) if row and row.payload else None

    def mark_intake_attached(self, case_id: str, attached_case_id: str) -> None:
        """Record which case consumed this baseline, keeping the row as evidence.

        The intake is a clinical record in its own right — the priors it carries
        are what the reasoner fused — so it is retained and marked rather than
        deleted once it has been merged into an analysed case.
        """
        with Session(self.engine) as ses:
            row = ses.get(IntakeBaselineRow, case_id)
            if row is not None:
                row.attached_case_id = attached_case_id
                ses.commit()

    def delete_intake(self, case_id: str) -> bool:
        """Drop a baseline the clinician discarded. True if a row was removed."""
        with Session(self.engine) as ses:
            row = ses.get(IntakeBaselineRow, case_id)
            if row is None:
                return False
            ses.delete(row)
            ses.commit()
            return True

    # ---- MRI NeuroView artifacts ----
    def save_neuroview(self, case_id: str, study_id: str, payload: dict) -> None:
        with Session(self.engine) as ses:
            row = ses.get(NeuroViewRow, case_id)
            if row is None:
                row = NeuroViewRow(case_id=case_id)
                ses.add(row)
            row.study_id = study_id
            row.payload = payload
            ses.commit()

    def get_neuroview(self, case_id: str) -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(NeuroViewRow, case_id)
            return dict(row.payload) if row and row.payload else None

    # ---- feedback ----
    def add_feedback(self, case_id: str, diagnosis: str, verdict: str,
                     correction: str = "") -> None:
        with Session(self.engine) as ses:
            ses.add(FeedbackRow(case_id=case_id, diagnosis=diagnosis,
                                verdict=verdict, correction=correction))
            ses.commit()

    def feedback_stats(self) -> dict:
        with Session(self.engine) as ses:
            rows = ses.execute(select(FeedbackRow)).scalars().all()
            counts: dict[str, int] = {}
            for r in rows:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
            return {"total": len(rows), "by_verdict": counts}

    def list_feedback(self) -> list[dict]:
        """Every clinician feedback row, oldest first, as plain dicts."""
        with Session(self.engine) as ses:
            rows = ses.execute(
                select(FeedbackRow).order_by(FeedbackRow.id.asc())
            ).scalars().all()
            return [{"case_id": r.case_id, "diagnosis": r.diagnosis,
                     "verdict": r.verdict, "correction": r.correction} for r in rows]

    def verified_outcome_pairs(self, org_id: str | None = None,
                               model_version: str | None = None) -> list[dict]:
        """Cases that carry a verified outcome, with the posterior AURA stored for each.

        The input to local cohort validation (``services/assurance/local_validation``):
        ``{"case_id", "posterior": {dx: p}, "truth", "covered", "abstained"}`` per case.
        """
        with Session(self.engine) as ses:
            stmt = select(CaseRow, OutcomeRow).join(
                OutcomeRow, OutcomeRow.case_id == CaseRow.case_id)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where(CaseRow.tenant_id == org_id)
            if model_version:
                stmt = stmt.where(CaseRow.model_version == model_version)
            out = []
            for case, outcome in ses.execute(stmt).all():
                safety = (case.bundle or {}).get("safety") or {}
                posterior = {
                    p["diagnosis"]: float(p.get("probability", 0.0))
                    for p in safety.get("predictions") or [] if p.get("diagnosis")
                }
                out.append({"case_id": case.case_id, "posterior": posterior,
                            "truth": outcome.true_diagnosis,
                            "covered": bool(outcome.covered),
                            "abstained": bool(case.abstained)})
            return out

    # ---- adaptive conformal inference (Module 8) ----
    def load_aci_state(self, stream: str = "global") -> dict | None:
        """Return the persisted ACIState row (as a plain dict) or None if unset."""
        with Session(self.engine) as ses:
            row = ses.get(ConformalStateRow, stream)
            return dict(row.state) if row and row.state else None

    def save_aci_state(self, state_row: dict, stream: str = "global") -> None:
        with Session(self.engine) as ses:
            row = ses.get(ConformalStateRow, stream)
            if row is None:
                row = ConformalStateRow(stream=stream)
                ses.add(row)
            row.state = state_row
            row.updated_at = datetime.now(timezone.utc)
            ses.commit()

    def record_outcome(self, case_id: str, probs, true_index: int,
                       true_diagnosis: str = "", stream: str = "global") -> dict:
        """Fold one **verified** outcome into the online ACI threshold and persist it.

        Loads q̂, runs the ACI step, writes q̂ back plus an append-only OutcomeRow.
        Returns the transition telemetry so the caller can surface live coverage.
        Import of the ACI engine is local to keep ``storage`` free of a service
        dependency at import time.

        The **only** legitimate caller is ``POST /v1/cases/{id}/outcome``, which first
        checks that ``source`` is one of ``VALID_OUTCOME_SOURCES`` (pcr, biopsy,
        pathology, expert_consensus, clinical_course) and that the diagnosis parses.
        This docstring used to say "called whenever a clinician confirms a case's
        diagnosis (see the feedback endpoint)", and the feedback endpoint did call it —
        with a diagnosis defaulting to *AURA's own top prediction*. Conformal coverage
        was therefore scored against the model's own output, ``covered`` was true by
        construction, and q̂ shrank monotonically until the 90% guarantee meant nothing.
        Do not wire this to any signal that is not ground truth: an ACI update is a
        statistical claim about reality, not a record that a clinician clicked accept.

        ``outcomes.case_id`` is UNIQUE — one verified outcome per case. Callers must
        handle IntegrityError (or pre-check :meth:`has_outcome`) rather than letting a
        duplicate surface as a 500.
        """
        from aura.common.config import get_settings
        from aura.services.safety.aci import AdaptiveConformalInference, ACIState

        s = get_settings()
        row = self.load_aci_state(stream)
        if row is not None:
            aci = AdaptiveConformalInference(state=ACIState.from_row(row))
        else:
            aci = AdaptiveConformalInference(
                coverage=s.conformal_coverage, gamma=s.aci_gamma, window=s.aci_window
            )

        info = aci.update(probs, true_index)
        self.save_aci_state(aci.state.to_row(), stream=stream)

        with Session(self.engine) as ses:
            ses.add(OutcomeRow(
                case_id=case_id,
                true_diagnosis=true_diagnosis,
                covered=info["covered"],
                qhat_after=info["qhat"],
                localized_coverage=info["localized_coverage"],
            ))
            ses.commit()
        return info

    # ---- audit ----
    def audit(self, action: str, entity_type: str = "", entity_id: str = "",
              actor: str = "system", detail: dict | None = None,
              tenant_id: str = "default", model_version: str = "",
              aura_version: str = "0.1.0") -> None:
        now = datetime.now(timezone.utc)
        d = detail or {}
        with _chain_append(self.engine) as ses:
            last = ses.execute(
                select(AuditRow).order_by(AuditRow.id.desc()).limit(1)
            ).scalars().first()
            prev_hash = last.record_hash if last else _GENESIS_HASH
            rec_hash = _compute_audit_hash(prev_hash, action, entity_id, d, now)
            ses.add(AuditRow(actor=actor, action=action, entity_type=entity_type,
                             entity_id=entity_id, detail=d,
                             previous_hash=prev_hash, record_hash=rec_hash,
                             tenant_id=tenant_id or "default",
                             model_version=model_version or "",
                             aura_version=aura_version or "0.1.0",
                             created_at=now))
            ses.commit()

    def recent_audit(self, limit: int = 50, tenant_id: str | None = None) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(AuditRow)
            if tenant_id and tenant_id not in ("system", "*"):
                stmt = stmt.where(AuditRow.tenant_id == tenant_id)
            rows = ses.execute(
                stmt.order_by(AuditRow.id.desc()).limit(limit)
            ).scalars().all()
            return [{
                "id": r.id,
                "actor": r.actor, "action": r.action, "entity_type": r.entity_type,
                "entity_id": r.entity_id, "detail": r.detail,
                "tenant_id": getattr(r, "tenant_id", "default"),
                "model_version": getattr(r, "model_version", ""),
                "aura_version": getattr(r, "aura_version", "0.1.0"),
                "previous_hash": r.previous_hash, "record_hash": r.record_hash,
                "created_at": r.created_at.isoformat(),
            } for r in rows]

    # ---- organizations ----
    def create_organization(self, org_id: str, name: str, slug: str) -> dict:
        with Session(self.engine) as ses:
            if ses.get(OrganizationRow, org_id):
                raise ValueError(f"Organization '{org_id}' already exists")
            org = OrganizationRow(org_id=org_id, name=name, slug=slug, is_active=True)
            ses.add(org)
            ses.commit()
            return {
                "org_id": org.org_id,
                "name": org.name,
                "slug": org.slug,
                "is_active": org.is_active,
                "created_at": org.created_at.isoformat(),
            }

    def get_organization(self, org_id: str) -> dict | None:
        with Session(self.engine) as ses:
            org = ses.get(OrganizationRow, org_id)
            if not org:
                return None
            return {
                "org_id": org.org_id,
                "name": org.name,
                "slug": org.slug,
                "is_active": org.is_active,
                "created_at": org.created_at.isoformat(),
            }

    def list_organizations(self) -> list[dict]:
        with Session(self.engine) as ses:
            rows = ses.execute(select(OrganizationRow).order_by(OrganizationRow.created_at.asc())).scalars().all()
            return [
                {
                    "org_id": r.org_id,
                    "name": r.name,
                    "slug": r.slug,
                    "is_active": r.is_active,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # ---- users ----
    def create_user(self, username: str, password: str, role: str = "CLINICAL_USER",
                    org_id: str = "default", email: str = "") -> dict:
        pw_hash, salt = self.hash_password(password)
        uid = f"usr-{uuid.uuid4().hex[:12]}"
        with Session(self.engine) as ses:
            existing = ses.execute(select(UserRow).where(UserRow.username == username)).scalars().first()
            if existing:
                raise ValueError(f"Username '{username}' already exists")
            user = UserRow(
                user_id=uid,
                org_id=org_id,
                username=username,
                email=email,
                password_hash=pw_hash,
                salt=salt,
                role=role,
                is_active=True,
            )
            ses.add(user)
            ses.commit()
            return {
                "user_id": user.user_id,
                "org_id": user.org_id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
            }

    def get_user(self, user_id: str) -> dict | None:
        with Session(self.engine) as ses:
            u = ses.get(UserRow, user_id)
            if not u:
                return None
            return {
                "user_id": u.user_id,
                "org_id": u.org_id,
                "username": u.username,
                "email": u.email,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "disabled_at": u.disabled_at.isoformat() if u.disabled_at else None,
            }

    def get_user_by_username(self, username: str) -> dict | None:
        with Session(self.engine) as ses:
            u = ses.execute(select(UserRow).where(UserRow.username == username)).scalars().first()
            if not u:
                return None
            return {
                "user_id": u.user_id,
                "org_id": u.org_id,
                "username": u.username,
                "email": u.email,
                "password_hash": u.password_hash,
                "salt": u.salt,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "disabled_at": u.disabled_at.isoformat() if u.disabled_at else None,
            }

    def list_users(self, org_id: str | None = None) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(UserRow)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where(UserRow.org_id == org_id)
            rows = ses.execute(stmt.order_by(UserRow.created_at.asc())).scalars().all()
            return [{
                "user_id": u.user_id,
                "org_id": u.org_id,
                "username": u.username,
                "email": u.email,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "disabled_at": u.disabled_at.isoformat() if u.disabled_at else None,
            } for u in rows]

    def disable_user(self, user_id: str) -> bool:
        with Session(self.engine) as ses:
            u = ses.get(UserRow, user_id)
            if not u:
                return False
            u.is_active = False
            u.disabled_at = datetime.now(timezone.utc)
            # ORM rather than raw SQL: "SET revoked = 1" is an integer-into-boolean
            # error on PostgreSQL.
            for sess in ses.execute(
                select(SessionRow).where(SessionRow.user_id == user_id)
            ).scalars():
                sess.revoked = True
            ses.commit()
            return True

    # ---- sessions / auth tokens ----
    def create_session(self, user_id: str, org_id: str, role: str, duration_hours: int = 24) -> tuple[str, dict]:
        raw_token = f"aura_tok_{secrets.token_urlsafe(32)}"
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        sid = f"ses-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        expires_at = now + timedelta(hours=duration_hours)
        with Session(self.engine) as ses:
            session_row = SessionRow(
                session_id=sid,
                user_id=user_id,
                org_id=org_id,
                role=role,
                token_hash=token_hash,
                created_at=now,
                expires_at=expires_at,
                revoked=False,
            )
            ses.add(session_row)
            ses.commit()
            return raw_token, {
                "session_id": sid,
                "user_id": user_id,
                "org_id": org_id,
                "role": role,
                "expires_at": expires_at.isoformat(),
            }

    def validate_session(self, token: str) -> dict | None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        with Session(self.engine) as ses:
            stmt = select(SessionRow).where(SessionRow.token_hash == token_hash, SessionRow.revoked == False)
            row = ses.execute(stmt).scalars().first()
            if not row:
                return None
            exp = row.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if now > exp:
                return None
            user = ses.get(UserRow, row.user_id)
            if not user or not user.is_active:
                return None
            return {
                "session_id": row.session_id,
                "user_id": row.user_id,
                "username": user.username,
                "org_id": row.org_id,
                "role": row.role,
                "expires_at": row.expires_at.isoformat(),
            }

    def revoke_session(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with Session(self.engine) as ses:
            stmt = select(SessionRow).where(SessionRow.token_hash == token_hash)
            row = ses.execute(stmt).scalars().first()
            if not row:
                return False
            row.revoked = True
            ses.commit()
            return True

    # ---- projects ----
    def create_project(self, project_id: str, org_id: str, name: str, description: str = "") -> dict:
        with Session(self.engine) as ses:
            proj = ProjectRow(project_id=project_id, org_id=org_id, name=name, description=description)
            ses.add(proj)
            ses.commit()
            return {
                "project_id": proj.project_id,
                "org_id": proj.org_id,
                "name": proj.name,
                "description": proj.description,
                "created_at": proj.created_at.isoformat(),
            }

    def get_project(self, project_id: str, org_id: str | None = None) -> dict | None:
        with Session(self.engine) as ses:
            p = ses.get(ProjectRow, project_id)
            if not p:
                return None
            if org_id and org_id not in ("system", "*") and p.org_id != org_id:
                raise CrossTenantAccessError(f"Project '{project_id}' does not belong to organization '{org_id}'")
            return {
                "project_id": p.project_id,
                "org_id": p.org_id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat(),
            }

    def list_projects(self, org_id: str | None = None) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(ProjectRow)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where(ProjectRow.org_id == org_id)
            rows = ses.execute(stmt.order_by(ProjectRow.created_at.asc())).scalars().all()
            return [{
                "project_id": p.project_id,
                "org_id": p.org_id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat(),
            } for p in rows]

    # ---- model registry ----
    def register_model(self, model_id: str, name: str, version: str, modality: str,
                       org_id: str = "system", intended_use: str = "", source: str = "",
                       training_data_provenance: str = "", validation_dataset: str = "",
                       validation_metrics: dict | None = None, checksum_sha256: str = "",
                       approval_status: str = "APPROVED", deployment_status: str = "ACTIVE") -> dict:
        with Session(self.engine) as ses:
            row = ses.get(RegisteredModelRow, model_id)
            if row is not None:
                raise ValueError(f"Model '{model_id}' already registered")
            row = RegisteredModelRow(
                model_id=model_id,
                org_id=org_id,
                name=name,
                version=version,
                modality=modality,
                intended_use=intended_use,
                source=source,
                training_data_provenance=training_data_provenance,
                validation_dataset=validation_dataset,
                validation_metrics=validation_metrics or {},
                checksum_sha256=checksum_sha256,
                approval_status=approval_status,
                deployment_status=deployment_status,
            )
            ses.add(row)
            ses.commit()
            return self._model_row_to_dict(row)

    def get_model(self, model_id: str) -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(RegisteredModelRow, model_id)
            if not row:
                return None
            return self._model_row_to_dict(row)

    def list_models(self, org_id: str | None = None, deployment_status: str | None = None) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(RegisteredModelRow)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where((RegisteredModelRow.org_id == org_id) | (RegisteredModelRow.org_id == "system"))
            if deployment_status:
                stmt = stmt.where(RegisteredModelRow.deployment_status == deployment_status)
            rows = ses.execute(stmt.order_by(RegisteredModelRow.created_at.desc())).scalars().all()
            return [self._model_row_to_dict(r) for r in rows]

    def update_model_status(self, model_id: str, approval_status: str | None = None,
                            deployment_status: str | None = None) -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(RegisteredModelRow, model_id)
            if not row:
                return None
            if approval_status is not None:
                row.approval_status = approval_status
            if deployment_status is not None:
                row.deployment_status = deployment_status
                if deployment_status == "RETIRED":
                    row.retired_at = datetime.now(timezone.utc)
            ses.commit()
            return self._model_row_to_dict(row)

    @staticmethod
    def _model_row_to_dict(row: RegisteredModelRow) -> dict:
        return {
            "model_id": row.model_id,
            "org_id": row.org_id,
            "name": row.name,
            "version": row.version,
            "modality": row.modality,
            "intended_use": row.intended_use,
            "source": row.source,
            "training_data_provenance": row.training_data_provenance,
            "validation_dataset": row.validation_dataset,
            "validation_metrics": row.validation_metrics,
            "checksum_sha256": row.checksum_sha256,
            "approval_status": row.approval_status,
            "deployment_status": row.deployment_status,
            "created_at": row.created_at.isoformat(),
            "retired_at": row.retired_at.isoformat() if row.retired_at else None,
        }

    # ---- validation runs ----
    def create_validation_run(self, run_id: str, org_id: str, model_id: str,
                              dataset_name: str, config: dict | None = None,
                              created_by: str = "system") -> dict:
        with Session(self.engine) as ses:
            row = ValidationRunRow(
                run_id=run_id,
                org_id=org_id,
                model_id=model_id,
                dataset_name=dataset_name,
                status="QUEUED",
                config=config or {},
                created_by=created_by,
            )
            ses.add(row)
            ses.commit()
            return self._validation_run_to_dict(row)

    def update_validation_run(self, run_id: str, status: str, metrics: dict | None = None,
                              subgroup_analysis: dict | None = None, failure_analysis: dict | None = None,
                              report_markdown: str = "", error: str = "") -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(ValidationRunRow, run_id)
            if not row:
                return None
            row.status = status
            if metrics is not None:
                row.metrics = metrics
            if subgroup_analysis is not None:
                row.subgroup_analysis = subgroup_analysis
            if failure_analysis is not None:
                row.failure_analysis = failure_analysis
            if report_markdown:
                row.report_markdown = report_markdown
            if error:
                row.error = error
            if status in ("COMPLETED", "FAILED"):
                row.completed_at = datetime.now(timezone.utc)
            ses.commit()
            return self._validation_run_to_dict(row)

    def get_validation_run(self, run_id: str, org_id: str | None = None) -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(ValidationRunRow, run_id)
            if not row:
                return None
            if org_id and org_id not in ("system", "*") and row.org_id != org_id:
                raise CrossTenantAccessError(f"Validation run '{run_id}' does not belong to organization '{org_id}'")
            return self._validation_run_to_dict(row)

    def list_validation_runs(self, org_id: str | None = None, model_id: str | None = None, limit: int = 50) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(ValidationRunRow)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where(ValidationRunRow.org_id == org_id)
            if model_id:
                stmt = stmt.where(ValidationRunRow.model_id == model_id)
            rows = ses.execute(stmt.order_by(ValidationRunRow.created_at.desc()).limit(limit)).scalars().all()
            return [self._validation_run_to_dict(r) for r in rows]

    @staticmethod
    def _validation_run_to_dict(row: ValidationRunRow) -> dict:
        return {
            "run_id": row.run_id,
            "org_id": row.org_id,
            "model_id": row.model_id,
            "dataset_name": row.dataset_name,
            "status": row.status,
            "config": row.config,
            "metrics": row.metrics,
            "subgroup_analysis": row.subgroup_analysis,
            "failure_analysis": row.failure_analysis,
            "report_markdown": row.report_markdown,
            "error": row.error,
            "created_by": row.created_by,
            "created_at": row.created_at.isoformat(),
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    # ---- monitoring & alerts ----
    def log_metric(self, metric_name: str, value: float, org_id: str = "default",
                   dimensions: dict | None = None) -> None:
        with Session(self.engine) as ses:
            row = SystemMetricRow(
                org_id=org_id,
                metric_name=metric_name,
                value=float(value),
                dimensions=dimensions or {},
                timestamp=datetime.now(timezone.utc),
            )
            ses.add(row)
            ses.commit()

    def get_metrics(self, metric_name: str | None = None, org_id: str = "default",
                    limit: int = 100) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(SystemMetricRow)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where(SystemMetricRow.org_id == org_id)
            if metric_name:
                stmt = stmt.where(SystemMetricRow.metric_name == metric_name)
            rows = ses.execute(stmt.order_by(SystemMetricRow.timestamp.desc()).limit(limit)).scalars().all()
            return [{
                "id": r.id,
                "org_id": r.org_id,
                "metric_name": r.metric_name,
                "value": r.value,
                "dimensions": r.dimensions,
                "timestamp": r.timestamp.isoformat(),
            } for r in rows]

    def create_alert(self, alert_id: str, category: str, message: str,
                     severity: str = "WARNING", org_id: str = "default",
                     model_id: str = "", details: dict | None = None) -> dict:
        with Session(self.engine) as ses:
            row = AlertRow(
                alert_id=alert_id,
                org_id=org_id,
                model_id=model_id,
                severity=severity,
                category=category,
                message=message,
                details=details or {},
                acknowledged=False,
            )
            ses.add(row)
            ses.commit()
            return {
                "alert_id": row.alert_id,
                "org_id": row.org_id,
                "model_id": row.model_id,
                "severity": row.severity,
                "category": row.category,
                "message": row.message,
                "details": row.details,
                "acknowledged": row.acknowledged,
                "created_at": row.created_at.isoformat(),
            }

    def list_alerts(self, org_id: str = "default", acknowledged: bool | None = None,
                    limit: int = 50) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = select(AlertRow)
            if org_id and org_id not in ("system", "*"):
                stmt = stmt.where(AlertRow.org_id == org_id)
            if acknowledged is not None:
                stmt = stmt.where(AlertRow.acknowledged == acknowledged)
            rows = ses.execute(stmt.order_by(AlertRow.created_at.desc()).limit(limit)).scalars().all()
            return [{
                "alert_id": r.alert_id,
                "org_id": r.org_id,
                "model_id": r.model_id,
                "severity": r.severity,
                "category": r.category,
                "message": r.message,
                "details": r.details,
                "acknowledged": r.acknowledged,
                "created_at": r.created_at.isoformat(),
            } for r in rows]

    def acknowledge_alert(self, alert_id: str, org_id: str | None = None) -> bool:
        with Session(self.engine) as ses:
            row = ses.get(AlertRow, alert_id)
            if not row:
                return False
            if org_id and org_id not in ("system", "*") and row.org_id != org_id:
                raise CrossTenantAccessError(f"Alert '{alert_id}' does not belong to organization '{org_id}'")
            row.acknowledged = True
            ses.commit()
            return True

    # ---- memory indexing ----
    def add_memory_record(self, case_id: str, embedding: list[float] | np.ndarray, diagnosis: str) -> None:
        emb_list = list(embedding) if hasattr(embedding, "__iter__") else embedding
        emb_str = json.dumps(emb_list)
        with Session(self.engine) as ses:
            row = ses.get(MemoryRecordRow, case_id)
            if row is None:
                row = MemoryRecordRow(case_id=case_id)
                ses.add(row)
            row.diagnosis = diagnosis
            row.embedding_json = emb_str
            ses.commit()

    def list_memory_records(self) -> list[dict]:
        with Session(self.engine) as ses:
            rows = ses.execute(select(MemoryRecordRow)).scalars().all()
            return [
                {
                    "case_id": r.case_id,
                    "diagnosis": r.diagnosis,
                    "embedding": json.loads(r.embedding_json),
                }
                for r in rows
            ]

    def has_outcome(self, case_id: str) -> bool:
        with Session(self.engine) as ses:
            row = ses.execute(
                select(OutcomeRow).where(OutcomeRow.case_id == case_id)
            ).scalars().first()
            return row is not None

    def save_chat_message(self, case_id: str, role: str, content: str) -> None:
        with Session(self.engine) as ses:
            row = ChatMessageRow(case_id=case_id, role=role, content=content)
            ses.add(row)
            ses.commit()

    def get_chat_history(self, case_id: str) -> list[dict]:
        with Session(self.engine) as ses:
            stmt = (
                select(ChatMessageRow)
                .where(ChatMessageRow.case_id == case_id)
                .order_by(ChatMessageRow.id.asc())
            )
            rows = ses.execute(stmt).scalars().all()
            return [
                {"role": r.role, "content": r.content, "created_at": r.created_at.isoformat()}
                for r in rows
            ]

    def log_decision_provenance(self, case_id: str, provenance: dict) -> None:
        """Record the complete decision provenance path for a case.

        Serializes the pipeline execution chain and computes a SHA-256 hash
        so the provenance can be verified later.
        """
        now = datetime.now(timezone.utc)
        with _chain_append(self.engine) as ses:
            last = ses.execute(
                select(DecisionProvenanceRow).order_by(DecisionProvenanceRow.case_id.desc())
                .limit(1)
            ).scalars().first()
            prev_hash = last.provenance_hash if last else _GENESIS_HASH
            prov_hash = _compute_audit_hash(prev_hash, "decision_provenance", case_id, provenance, now)
            row = ses.get(DecisionProvenanceRow, case_id)
            if row is None:
                row = DecisionProvenanceRow(case_id=case_id)
                ses.add(row)
            row.provenance = provenance
            row.provenance_hash = prov_hash
            row.previous_hash = prev_hash
            row.created_at = now
            ses.commit()

    def get_decision_provenance(self, case_id: str) -> dict | None:
        with Session(self.engine) as ses:
            row = ses.get(DecisionProvenanceRow, case_id)
            return dict(row.provenance) if row and row.provenance else None

    def verify_decision_provenance(self, case_id: str) -> bool:
        """Verify the provenance hash chain for a specific case."""
        with Session(self.engine) as ses:
            row = ses.get(DecisionProvenanceRow, case_id)
            if row is None:
                return True
            expected = _compute_audit_hash(
                row.previous_hash, "decision_provenance", case_id,
                row.provenance, row.created_at,
            )
            return row.provenance_hash == expected

    def verify_audit_trail(self) -> bool:
        return verify_audit_trail(self)

def compute_provenance_hash(bundle: CaseBundle) -> str:
    import hashlib
    import json
    
    # 1. Image Hash
    img_str = "".join(map(str, bundle.image))
    image_hash = hashlib.sha256(img_str.encode()).hexdigest()
    
    # 2. Vision Result
    vision_dict = bundle.vision.model_dump(mode="json") if bundle.vision else None
    
    # 3. Evidence Graph
    evidence_list = [e.model_dump(mode="json") for e in bundle.evidence] if bundle.evidence else []
    
    # 4. Reasoning Steps
    reasoning_list = [s.model_dump(mode="json") for s in bundle.reasoning.steps] if bundle.reasoning else []
    
    # 5. Safety Verification
    safety_dict = bundle.safety.model_dump(mode="json") if bundle.safety else None
    
    # 6. DRP
    drp_dict = bundle.drp.model_dump(mode="json") if bundle.drp else None

    # 7. Decision Assurance
    da_dict = (
        bundle.decision_assurance.model_dump(mode="json")
        if getattr(bundle, "decision_assurance", None) else None
    )
    
    path_dict = {
        "image_hash": image_hash,
        "vision_result": vision_dict,
        "evidence_graph": evidence_list,
        "reasoning_steps": reasoning_list,
        "safety_verification": safety_dict,
        "drp": drp_dict,
        "decision_assurance": da_dict,
    }
    
    serialized = json.dumps(path_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
