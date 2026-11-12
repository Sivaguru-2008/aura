"""Tests for AURA schema migration engine and versioned migrations."""
from __future__ import annotations

import tempfile
from pathlib import Path
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool

from aura.gateway.migrations.runner import MigrationRunner


def test_migration_discovery():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    runner = MigrationRunner(engine)
    migrations = runner.discover_migrations()
    assert len(migrations) >= 4
    versions = [m.version for m in migrations]
    assert "001" in versions
    assert "002" in versions
    assert "003" in versions
    assert "004" in versions
    engine.dispose()


def test_migration_application_and_idempotency():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    runner = MigrationRunner(engine)

    assert runner.applied_versions() == []

    applied = runner.apply_all()
    assert len(applied) >= 4
    assert set(applied) == set(runner.applied_versions())

    # Verify tables exist
    insp = inspect(engine)
    tables = insp.get_table_names()
    assert "schema_migrations" in tables
    assert "cases" in tables
    assert "audit_log" in tables
    assert "organizations" in tables
    assert "users" in tables
    assert "sessions" in tables
    assert "projects" in tables
    assert "model_registry" in tables
    assert "validation_runs" in tables
    assert "system_metrics" in tables
    assert "system_alerts" in tables

    # Idempotency check: running again should apply 0 migrations
    reapplied = runner.apply_all()
    assert reapplied == []
    assert runner.applied_versions() == applied

    engine.dispose()
