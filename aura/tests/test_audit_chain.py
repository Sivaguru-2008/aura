"""Tests for the cryptographic audit chain (tamper-evident audit log).

Verifies:
- Fresh audit rows are correctly chained with SHA-256 hashes.
- Modifying a historical row's detail field breaks verify_audit_trail().
- The genesis row links to the zero-hash sentinel.
- Multiple rows form an unbroken chain.
- verify_audit_trail() returns True on an untouched chain.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from aura.gateway.storage import AuditRow, Store, verify_audit_trail, _GENESIS_HASH, _compute_audit_hash


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Create a fresh SQLite database in a temp directory."""
    db = tmp_path / "test_audit.db"
    store = Store(db)
    return store, db


# --------------------------------------------------------------------------- #
# Chain integrity
# --------------------------------------------------------------------------- #
def test_single_audit_row_has_correct_hash(tmp_db):
    store, db = tmp_db
    store.audit("test.action", "test", "id-1", detail={"msg": "hello"})

    with Session(store.engine) as ses:
        row = ses.execute(
            select(AuditRow).order_by(AuditRow.id.asc())
        ).scalars().first()

    assert row.previous_hash == _GENESIS_HASH
    expected = _compute_audit_hash(
        _GENESIS_HASH, "test.action", "id-1", {"msg": "hello"}, row.created_at
    )
    assert row.record_hash == expected


def test_chain_links_multiple_rows(tmp_db):
    store, db = tmp_db
    for i in range(5):
        store.audit(f"action.{i}", "test", f"id-{i}", detail={"n": i})

    assert verify_audit_trail(store) is True


def test_genesis_hash_is_64_zeros():
    assert len(_GENESIS_HASH) == 64
    assert _GENESIS_HASH == "0" * 64


def test_verify_returns_true_on_fresh_chain(tmp_db):
    store, db = tmp_db
    store.audit("a.b", "e", "1")
    store.audit("c.d", "e", "2", detail={"key": "val"})
    store.audit("e.f", "e", "3")
    assert store.verify_audit_trail() is True


def test_verify_returns_true_on_empty_log(tmp_db):
    store, db = tmp_db
    assert verify_audit_trail(store) is True


# --------------------------------------------------------------------------- #
# Tamper detection
# --------------------------------------------------------------------------- #
def test_modifying_detail_in_historical_row_breaks_chain(tmp_db):
    """The core invariant: tampering with a past row causes verify to fail."""
    store, db = tmp_db
    store.audit("action.first", "test", "id-1", detail={"value": "original"})
    store.audit("action.second", "test", "id-2", detail={"value": "keep"})
    store.audit("action.third", "test", "id-3", detail={"value": "keep"})

    # Verify intact first
    assert verify_audit_trail(store) is True

    # Tamper: change the detail of the first row
    with Session(store.engine) as ses:
        row = ses.execute(
            select(AuditRow).order_by(AuditRow.id.asc()).limit(1)
        ).scalars().first()
        row.detail = {"value": "TAMPERED"}
        ses.commit()

    # Chain is now broken
    assert verify_audit_trail(store) is False


def test_modifying_action_in_historical_row_breaks_chain(tmp_db):
    store, db = tmp_db
    store.audit("original.action", "test", "id-1")
    store.audit("second.action", "test", "id-2")

    assert verify_audit_trail(store) is True

    # Tamper with the action
    with Session(store.engine) as ses:
        row = ses.execute(
            select(AuditRow).order_by(AuditRow.id.asc()).limit(1)
        ).scalars().first()
        row.action = "tampered.action"
        ses.commit()

    assert verify_audit_trail(store) is False


def test_modifying_previous_hash_breaks_chain(tmp_db):
    store, db = tmp_db
    store.audit("a", "e", "1")
    store.audit("b", "e", "2")

    assert verify_audit_trail(store) is True

    with Session(store.engine) as ses:
        row = ses.execute(
            select(AuditRow).order_by(AuditRow.id.desc()).limit(1)
        ).scalars().first()
        row.previous_hash = "a" * 64
        ses.commit()

    assert verify_audit_trail(store) is False


def test_modifying_record_hash_breaks_chain(tmp_db):
    store, db = tmp_db
    store.audit("a", "e", "1")

    assert verify_audit_trail(store) is True

    with Session(store.engine) as ses:
        row = ses.execute(
            select(AuditRow).order_by(AuditRow.id.asc()).limit(1)
        ).scalars().first()
        row.record_hash = "b" * 64
        ses.commit()

    assert verify_audit_trail(store) is False


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_detail_is_consistent(tmp_db):
    store, db = tmp_db
    store.audit("x", "e", "1")
    store.audit("y", "e", "2")
    assert verify_audit_trail(store) is True


def test_special_characters_in_detail(tmp_db):
    store, db = tmp_db
    store.audit("x", "e", "1", detail={"unicode": "\u00e9\u00e8\u00ea", "json": '{"nested": true}'})
    store.audit("y", "e", "2", detail={"emoji": "test", "path": "C:\\Users\\test"})
    assert verify_audit_trail(store) is True


def test_verify_with_db_path(tmp_db):
    store, db = tmp_db
    store.audit("x", "e", "1")
    assert verify_audit_trail(db_path=db) is True


def test_recent_audit_includes_hashes(tmp_db):
    store, db = tmp_db
    store.audit("test.action", "test", "id-1")
    rows = store.recent_audit(10)
    assert len(rows) == 1
    assert "previous_hash" in rows[0]
    assert "record_hash" in rows[0]
    assert len(rows[0]["record_hash"]) == 64
    assert rows[0]["previous_hash"] == _GENESIS_HASH


# Need select import
from sqlalchemy import select
