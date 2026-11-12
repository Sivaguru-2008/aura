"""Migration 003: Model Registry and Validation Runs."""
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

VERSION = "003"
NAME = "model_registry_and_validation"


def upgrade(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        if not insp.has_table("model_registry"):
            conn.execute(text("""
                CREATE TABLE model_registry (
                    model_id VARCHAR(128) PRIMARY KEY,
                    org_id VARCHAR(64) DEFAULT 'system',
                    name VARCHAR(128) NOT NULL,
                    version VARCHAR(64) NOT NULL,
                    modality VARCHAR(64) NOT NULL,
                    intended_use TEXT DEFAULT '',
                    source VARCHAR(255) DEFAULT '',
                    training_data_provenance TEXT DEFAULT '',
                    validation_dataset VARCHAR(255) DEFAULT '',
                    validation_metrics JSON,
                    checksum_sha256 VARCHAR(64) DEFAULT '',
                    approval_status VARCHAR(32) DEFAULT 'APPROVED',
                    deployment_status VARCHAR(32) DEFAULT 'ACTIVE',
                    created_at TIMESTAMP,
                    retired_at TIMESTAMP
                )
            """))

        if not insp.has_table("validation_runs"):
            conn.execute(text("""
                CREATE TABLE validation_runs (
                    run_id VARCHAR(64) PRIMARY KEY,
                    org_id VARCHAR(64) NOT NULL,
                    model_id VARCHAR(128) NOT NULL,
                    dataset_name VARCHAR(128) NOT NULL,
                    status VARCHAR(32) DEFAULT 'QUEUED',
                    config JSON,
                    metrics JSON,
                    subgroup_analysis JSON,
                    failure_analysis JSON,
                    report_markdown TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    created_by VARCHAR(64) DEFAULT 'system',
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """))


def downgrade(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS validation_runs"))
        conn.execute(text("DROP TABLE IF EXISTS model_registry"))
