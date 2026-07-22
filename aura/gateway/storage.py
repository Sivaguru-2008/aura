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
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from schemas.contracts import CaseBundle


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
            from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
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
            from schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
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
        """Fold one confirmed outcome into the online ACI threshold and persist it.

        This is the offline-tracking hook: called whenever a clinician confirms a
        case's diagnosis (see the feedback endpoint). Loads q̂, runs the ACI step,
        writes q̂ back plus an append-only OutcomeRow. Returns the transition
        telemetry so the caller can surface the live coverage. Import of the ACI
        engine is local to keep ``storage`` free of a service dependency at import.
        """
        from common.config import get_settings
        from services.safety.aci import AdaptiveConformalInference, ACIState

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
        with Session(self.engine) as ses:
            ses.add(AuditRow(actor=actor, action=action, entity_type=entity_type,
                             entity_id=entity_id, detail=detail or {}))
            ses.commit()

    def recent_audit(self, limit: int = 50) -> list[dict]:
        with Session(self.engine) as ses:
            rows = ses.execute(
                select(AuditRow).order_by(AuditRow.id.desc()).limit(limit)
            ).scalars().all()
            return [{
                "actor": r.actor, "action": r.action, "entity_type": r.entity_type,
                "entity_id": r.entity_id, "detail": r.detail,
                "created_at": r.created_at.isoformat(),
            } for r in rows]
