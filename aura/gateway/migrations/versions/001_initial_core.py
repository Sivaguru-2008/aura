"""Migration 001: Initial Core Schema."""
from sqlalchemy import text
from sqlalchemy.engine import Engine

VERSION = "001"
NAME = "initial_core_schema"


def upgrade(engine: Engine) -> None:
    # Use SQLAlchemy metadata reflection/creation for core tables
    from aura.gateway.storage import Base
    Base.metadata.create_all(engine)


def downgrade(engine: Engine) -> None:
    # Clean rollback if needed
    with engine.begin() as conn:
        for tbl in [
            "decision_provenance", "chat_history", "memory_records",
            "neuroview_artifacts", "intake_baselines", "outcomes",
            "conformal_state", "feedback", "cases", "site_binding"
        ]:
            conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
