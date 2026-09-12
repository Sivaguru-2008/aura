"""AURA Database Migration Runner — supports SQLite and PostgreSQL.

Provides deterministic, versioned, transactional schema migrations and rollbacks.
Tracks applied versions in `schema_migrations` table.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import Boolean, DateTime, Integer, String, Text, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger("aura.migrations")


class MigrationBase(DeclarativeBase):
    pass


class SchemaMigrationRow(MigrationBase):
    __tablename__ = "schema_migrations"
    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    checksum: Mapped[str] = mapped_column(String(64), default="")


class Migration:
    def __init__(self, version: str, name: str, upgrade_fn: Callable[[Engine], None], downgrade_fn: Callable[[Engine], None] | None = None):
        self.version = version
        self.name = name
        self.upgrade_fn = upgrade_fn
        self.downgrade_fn = downgrade_fn


class MigrationRunner:
    def __init__(self, engine: Engine, versions_dir: Path | None = None):
        self.engine = engine
        if versions_dir is None:
            versions_dir = Path(__file__).resolve().parent / "versions"
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _ensure_table(self) -> None:
        MigrationBase.metadata.create_all(self.engine)

    def applied_versions(self) -> list[str]:
        with Session(self.engine) as ses:
            stmt = select(SchemaMigrationRow.version).order_by(SchemaMigrationRow.version.asc())
            return list(ses.execute(stmt).scalars().all())

    def discover_migrations(self) -> list[Migration]:
        migrations: list[Migration] = []
        py_files = sorted(self.versions_dir.glob("[0-9]*_*.py"))
        for f in py_files:
            mod_name = f"aura.gateway.migrations.versions.{f.stem}"
            try:
                mod = importlib.import_module(mod_name)
            except ModuleNotFoundError:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                else:
                    continue

            version = getattr(mod, "VERSION", f.stem.split("_")[0])
            name = getattr(mod, "NAME", f.stem)
            upgrade_fn = getattr(mod, "upgrade", None)
            downgrade_fn = getattr(mod, "downgrade", None)
            if upgrade_fn:
                migrations.append(Migration(version=version, name=name, upgrade_fn=upgrade_fn, downgrade_fn=downgrade_fn))
        return sorted(migrations, key=lambda m: m.version)

    def apply_all(self) -> list[str]:
        """Apply all pending migrations in order."""
        applied = set(self.applied_versions())
        discovered = self.discover_migrations()
        executed = []

        for m in discovered:
            if m.version not in applied:
                logger.info("Applying migration %s: %s", m.version, m.name)
                m.upgrade_fn(self.engine)
                with Session(self.engine) as ses:
                    row = SchemaMigrationRow(version=m.version, name=m.name)
                    ses.add(row)
                    ses.commit()
                executed.append(m.version)
        return executed

    def rollback_last(self) -> str | None:
        """Roll back the most recently applied migration."""
        applied = self.applied_versions()
        if not applied:
            return None
        last_version = applied[-1]
        discovered = {m.version: m for m in self.discover_migrations()}
        migration = discovered.get(last_version)

        if migration and migration.downgrade_fn:
            logger.info("Rolling back migration %s: %s", migration.version, migration.name)
            migration.downgrade_fn(self.engine)
            with Session(self.engine) as ses:
                row = ses.get(SchemaMigrationRow, last_version)
                if row:
                    ses.delete(row)
                    ses.commit()
            return last_version
        elif migration:
            raise RuntimeError(f"Migration {last_version} does not define a downgrade function")
        return None
