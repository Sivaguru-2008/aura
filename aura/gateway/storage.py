"""Persistence — SQLite (P0) behind a small repository API.

The architecture targets PostgreSQL; SQLAlchemy + a repository boundary keep the
swap to Postgres a config change. Case bundles are stored as JSON documents with
indexed columns for worklist queries; feedback and audit_log are first-class.
"""
from __future__ import annotations

import json
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
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from aura.schemas.contracts import CaseBundle
import hashlib

_GENESIS_HASH = "0" * 64


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


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)

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
        with Session(self.engine) as ses:
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
        with Session(self.engine) as ses:
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
