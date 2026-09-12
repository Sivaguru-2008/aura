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
import json
import threading
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


class Store:
    def __init__(self, db_path: Path, url: str | None = None, site_id: str | None = None):
        """Open (creating if needed) the case store.

        ``db_path`` is the SQLite file used when ``url`` is empty; it stays the home of
        the local EHR fixture and embedding cache either way (see the module docstring).
        ``url`` is any SQLAlchemy URL, normally ``AURA_DATABASE_URL``. ``site_id`` is
        ``AURA_SITE_ID``; see :class:`SiteBindingRow`.
        """
        self.db_path = Path(db_path)
        if url:
            self.engine = engine_for(url)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self.site_id = self._bind_site((site_id or "").strip())

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
    def save_case(self, bundle: CaseBundle) -> None:
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
            row.bundle = data
            ses.commit()

    def get_case(self, case_id: str) -> CaseBundle | None:
        with Session(self.engine) as ses:
            row = ses.get(CaseRow, case_id)
            if not row:
                return None
            b = CaseBundle.model_validate(row.bundle)
            from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
            b.dx_labels = {d.value: l for d, l in DIAGNOSIS_LABELS.items()}
            b.ev_labels = {f.value: l for f, l in FINDING_LABELS.items()}
            return b

    def list_cases(self, state: str | None = None, limit: int = 200) -> list[dict]:
        """Lightweight worklist rows (not full bundles)."""
        with Session(self.engine) as ses:
            stmt = select(CaseRow)
            if state:
                stmt = stmt.where(CaseRow.state == state)
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
                    # Empty when the study arrived without one — C-FIND returns a
                    # zero-length PatientID in that case rather than inventing one.
                    "patient_id": b.get("patient_id", "") or "",
                    "state": r.state,
                    "priority_score": r.priority_score,
                    "top_diagnosis": r.top_diagnosis,
                    "top_diagnosis_label": label,
                    "top_probability": r.top_probability,
                    "abstained": r.abstained,
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
              actor: str = "system", detail: dict | None = None) -> None:
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
                             created_at=now))
            ses.commit()

    def recent_audit(self, limit: int = 50) -> list[dict]:
        with Session(self.engine) as ses:
            rows = ses.execute(
                select(AuditRow).order_by(AuditRow.id.desc()).limit(limit)
            ).scalars().all()
            return [{
                "actor": r.actor, "action": r.action, "entity_type": r.entity_type,
                "entity_id": r.entity_id, "detail": r.detail,
                "previous_hash": r.previous_hash, "record_hash": r.record_hash,
                "created_at": r.created_at.isoformat(),
            } for r in rows]

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
    vision_dict = bundle.vision.model_dump() if bundle.vision else None
    
    # 3. Evidence Graph
    evidence_list = [e.model_dump() for e in bundle.evidence] if bundle.evidence else []
    
    # 4. Reasoning Steps
    reasoning_list = [s.model_dump() for s in bundle.reasoning.steps] if bundle.reasoning else []
    
    # 5. Safety Verification
    safety_dict = bundle.safety.model_dump() if bundle.safety else None
    
    # 6. DRP
    drp_dict = bundle.drp.model_dump() if bundle.drp else None
    
    path_dict = {
        "image_hash": image_hash,
        "vision_result": vision_dict,
        "evidence_graph": evidence_list,
        "reasoning_steps": reasoning_list,
        "safety_verification": safety_dict,
        "drp": drp_dict
    }
    
    serialized = json.dumps(path_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()
