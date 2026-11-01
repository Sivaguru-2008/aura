"""A database created by an older AURA opens, and keeps working, under this one.

create_all() adds missing tables but never adds a column, so every column added to an
existing table needs a migration. On 2026-09-14 one had none: the July container volume
opened, reported healthy, and then failed every upload and every audit write with
``no such column: audit_log.previous_hash``. The schema below is that volume's, copied
from its sqlite_master, so these tests fail again the next time a model column ships
without a migration.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from aura.gateway.migrations import MigrationRunner
from aura.gateway.migrations.runner import SchemaMigrationRow
from aura.gateway.storage import Base, Store, verify_audit_trail

# The July 2026 release's schema, verbatim.
JULY_SCHEMA = """
CREATE TABLE audit_log (
    id INTEGER NOT NULL, actor VARCHAR NOT NULL, action VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL, entity_id VARCHAR NOT NULL, detail JSON NOT NULL,
    created_at DATETIME NOT NULL, PRIMARY KEY (id));
CREATE TABLE cases (
    case_id VARCHAR NOT NULL, study_id VARCHAR NOT NULL, state VARCHAR NOT NULL,
    priority_score FLOAT NOT NULL, top_diagnosis VARCHAR NOT NULL,
    top_probability FLOAT NOT NULL, abstained BOOLEAN NOT NULL,
    ground_truth VARCHAR NOT NULL, bundle JSON NOT NULL, created_at DATETIME NOT NULL,
    PRIMARY KEY (case_id));
CREATE TABLE conformal_state (
    stream VARCHAR NOT NULL, state JSON NOT NULL, updated_at DATETIME NOT NULL,
    PRIMARY KEY (stream));
CREATE TABLE feedback (
    id INTEGER NOT NULL, case_id VARCHAR NOT NULL, diagnosis VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL, correction TEXT NOT NULL, created_at DATETIME NOT NULL,
    PRIMARY KEY (id));
CREATE TABLE memory_records (
    case_id VARCHAR NOT NULL, diagnosis VARCHAR NOT NULL, embedding_json TEXT NOT NULL,
    created_at DATETIME NOT NULL, PRIMARY KEY (case_id));
CREATE TABLE neuroview_artifacts (
    case_id VARCHAR NOT NULL, study_id VARCHAR NOT NULL, payload JSON NOT NULL,
    created_at DATETIME NOT NULL, PRIMARY KEY (case_id));
CREATE TABLE outcomes (
    id INTEGER NOT NULL, case_id VARCHAR NOT NULL, true_diagnosis VARCHAR NOT NULL,
    covered BOOLEAN NOT NULL, qhat_after FLOAT NOT NULL,
    localized_coverage FLOAT NOT NULL, created_at DATETIME NOT NULL, PRIMARY KEY (id));
CREATE INDEX ix_cases_priority_score ON cases (priority_score);
CREATE INDEX ix_cases_state ON cases (state);
CREATE INDEX ix_cases_study_id ON cases (study_id);
CREATE INDEX ix_feedback_case_id ON feedback (case_id);
CREATE INDEX ix_neuroview_artifacts_study_id ON neuroview_artifacts (study_id);
CREATE INDEX ix_outcomes_case_id ON outcomes (case_id);
"""


def _july_db(path):
    """A July database holding what the volume held: two audit rows and one case."""
    con = sqlite3.connect(path)
    con.executescript(JULY_SCHEMA)
    con.executemany(
        "INSERT INTO audit_log (actor, action, entity_type, entity_id, detail, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [("system", "case.uploaded", "case", "CASE-UPLOAD-1",
          json.dumps({"top": "heart_failure", "abstained": False}), "2026-07-29 12:43:22.081268"),
         ("anonymous", "POST /v1/studies/upload", "http", "", "{}", "2026-07-29 12:43:22.201928")])
    con.execute(
        "INSERT INTO cases (case_id, study_id, state, priority_score, top_diagnosis,"
        " top_probability, abstained, ground_truth, bundle, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("CASE-UPLOAD-1", "STUDY-UPLOAD-1", "ready", 0.5, "heart_failure", 0.497, 0, "",
         json.dumps({"case_id": "CASE-UPLOAD-1"}), "2026-07-29 12:43:22.081268"))
    con.commit()
    con.close()
    return path


def test_july_database_gains_every_model_column(tmp_path):
    store = Store(db_path=_july_db(tmp_path / "july.db"))
    insp = inspect(store.engine)
    missing = [f"{t.name}.{c.name}" for t in Base.metadata.sorted_tables
               for c in t.columns
               if c.name not in {col["name"] for col in insp.get_columns(t.name)}]
    assert missing == []


def test_july_audit_history_verifies_and_the_chain_continues(tmp_path):
    store = Store(db_path=_july_db(tmp_path / "july.db"))
    # The two July rows are linked, not left at the genesis default the new column had.
    assert verify_audit_trail(store)
    store.audit("case.uploaded", entity_type="case", entity_id="CASE-UPLOAD-2",
                detail={"top": "normal"})
    assert verify_audit_trail(store)
    actions = [r["action"] for r in store.recent_audit(limit=100)]
    assert {"POST /v1/studies/upload", "case.uploaded"} <= set(actions)
    assert actions.count("case.uploaded") == 2


def test_july_case_stays_in_the_default_tenant(tmp_path):
    path = _july_db(tmp_path / "july.db")
    store = Store(db_path=path)
    assert store.count() == 1
    con = sqlite3.connect(path)
    assert con.execute("SELECT tenant_id FROM cases").fetchall() == [("default",)]
    con.close()


def test_a_model_column_without_a_migration_stops_startup(tmp_path):
    """Record every migration as applied without running any, as if the one that adds
    the chain columns had never been written: the store must refuse to open, naming
    the column, rather than start and fail on the first write."""
    path = _july_db(tmp_path / "july.db")
    engine = create_engine(f"sqlite:///{path}")
    runner = MigrationRunner(engine)
    with Session(engine) as ses:
        ses.add_all(SchemaMigrationRow(version=m.version, name=m.name)
                    for m in runner.discover_migrations())
        ses.commit()
    engine.dispose()
    with pytest.raises(RuntimeError, match=r"audit_log\.previous_hash"):
        Store(db_path=path)
