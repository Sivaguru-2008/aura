"""Migration 005: the audit hash chain on an audit_log that predates it.

The hash-chained audit ledger added ``previous_hash`` and ``record_hash`` to
``audit_log``. 001's create_all() never alters a table that already exists, and 002
added the tenancy columns but not these two, so on a database created before the
chain every audit write, and with it every upload, failed with ``no such column:
audit_log.previous_hash``. Found upgrading the July 2026 container volume.

Adds both columns, then links the rows already there in id order with the same hash
``verify_audit_trail`` recomputes, so the old history verifies instead of reading as
tampering from its first row. A table created with the chain is left alone.
"""
from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, inspect, select, text, update
from sqlalchemy.engine import Connection, Engine

VERSION = "005"
NAME = "audit_log_hash_chain"


def upgrade(engine: Engine) -> None:
    from aura.gateway.storage import _GENESIS_HASH

    insp = inspect(engine)
    if not insp.has_table("audit_log"):
        return
    cols = {c["name"] for c in insp.get_columns("audit_log")}
    if {"previous_hash", "record_hash"} <= cols:
        return
    with engine.begin() as conn:
        for name in ("previous_hash", "record_hash"):
            if name not in cols:
                conn.execute(text(f"ALTER TABLE audit_log ADD COLUMN {name} "
                                  f"VARCHAR NOT NULL DEFAULT '{_GENESIS_HASH}'"))
        _link_existing_rows(conn)


def _link_existing_rows(conn: Connection) -> None:
    from aura.gateway.storage import _GENESIS_HASH, _compute_audit_hash

    # Only the columns this step needs, typed as the model types them, so ``detail`` and
    # ``created_at`` reach the hash exactly as verify_audit_trail sees them. Not the model
    # itself: a column that a later migration adds would not exist yet at this point.
    audit = Table("audit_log", MetaData(),
                  Column("id", Integer, primary_key=True),
                  Column("action", String), Column("entity_id", String),
                  Column("detail", JSON), Column("created_at", DateTime),
                  Column("previous_hash", String), Column("record_hash", String))
    rows = conn.execute(select(audit.c.id, audit.c.action, audit.c.entity_id,
                               audit.c.detail, audit.c.created_at)
                        .order_by(audit.c.id)).all()
    prev = _GENESIS_HASH
    for row in rows:
        rec = _compute_audit_hash(prev, row.action, row.entity_id, row.detail, row.created_at)
        conn.execute(update(audit).where(audit.c.id == row.id)
                     .values(previous_hash=prev, record_hash=rec))
        prev = rec


def downgrade(engine: Engine) -> None:
    # Deliberately keeps the columns. On a database created with the chain this
    # migration changed nothing, so dropping them would destroy a ledger it never
    # touched; and code that predates the chain simply leaves them at their default.
    return None
