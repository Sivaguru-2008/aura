"""Migration 002: Multi-Tenancy and Role-Based Access Control (RBAC)."""
from sqlalchemy import Boolean, DateTime, Integer, String, Text, inspect, text
from sqlalchemy.engine import Engine

VERSION = "002"
NAME = "multi_tenancy_and_rbac"


def upgrade(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        dialect = engine.dialect.name

        # 1. Organizations
        if not insp.has_table("organizations"):
            conn.execute(text("""
                CREATE TABLE organizations (
                    org_id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(64) UNIQUE NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP
                )
            """))

        # 2. Users
        if not insp.has_table("users"):
            conn.execute(text("""
                CREATE TABLE users (
                    user_id VARCHAR(64) PRIMARY KEY,
                    org_id VARCHAR(64) NOT NULL,
                    username VARCHAR(64) UNIQUE NOT NULL,
                    email VARCHAR(255) DEFAULT '',
                    password_hash VARCHAR(128) NOT NULL,
                    salt VARCHAR(64) NOT NULL,
                    role VARCHAR(32) DEFAULT 'CLINICAL_USER',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP,
                    disabled_at TIMESTAMP
                )
            """))

        # 3. Sessions
        if not insp.has_table("sessions"):
            conn.execute(text("""
                CREATE TABLE sessions (
                    session_id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    org_id VARCHAR(64) NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    token_hash VARCHAR(128) NOT NULL,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    revoked BOOLEAN DEFAULT 0
                )
            """))

        # 4. Projects
        if not insp.has_table("projects"):
            conn.execute(text("""
                CREATE TABLE projects (
                    project_id VARCHAR(64) PRIMARY KEY,
                    org_id VARCHAR(64) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TIMESTAMP
                )
            """))

        # 5. Add tenant_id to cases if missing
        if insp.has_table("cases"):
            cols = [c["name"] for c in insp.get_columns("cases")]
            if "tenant_id" not in cols:
                conn.execute(text("ALTER TABLE cases ADD COLUMN tenant_id VARCHAR(64) DEFAULT 'default'"))
            if "model_version" not in cols:
                conn.execute(text("ALTER TABLE cases ADD COLUMN model_version VARCHAR(128) DEFAULT 'vision-cxr-densenet121-production'"))
            if "decision" not in cols:
                conn.execute(text("ALTER TABLE cases ADD COLUMN decision VARCHAR(32) DEFAULT 'ACCEPT'"))
            if "decision_reason" not in cols:
                conn.execute(text("ALTER TABLE cases ADD COLUMN decision_reason TEXT DEFAULT ''"))

        # 6. Add tenant_id to audit_log if missing
        if insp.has_table("audit_log"):
            cols = [c["name"] for c in insp.get_columns("audit_log")]
            if "tenant_id" not in cols:
                conn.execute(text("ALTER TABLE audit_log ADD COLUMN tenant_id VARCHAR(64) DEFAULT 'default'"))
            if "model_version" not in cols:
                conn.execute(text("ALTER TABLE audit_log ADD COLUMN model_version VARCHAR(128) DEFAULT ''"))
            if "aura_version" not in cols:
                conn.execute(text("ALTER TABLE audit_log ADD COLUMN aura_version VARCHAR(32) DEFAULT '0.1.0'"))


def downgrade(engine: Engine) -> None:
    with engine.begin() as conn:
        for tbl in ["projects", "sessions", "users", "organizations"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
