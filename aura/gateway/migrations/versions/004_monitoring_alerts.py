"""Migration 004: Post-Deployment Monitoring and Alerts."""
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

VERSION = "004"
NAME = "monitoring_and_alerts"


def upgrade(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        if not insp.has_table("system_metrics"):
            conn.execute(text("""
                CREATE TABLE system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id VARCHAR(64) DEFAULT 'default',
                    metric_name VARCHAR(128) NOT NULL,
                    value FLOAT NOT NULL,
                    dimensions JSON,
                    timestamp TIMESTAMP
                )
            """))

        if not insp.has_table("system_alerts"):
            conn.execute(text("""
                CREATE TABLE system_alerts (
                    alert_id VARCHAR(64) PRIMARY KEY,
                    org_id VARCHAR(64) DEFAULT 'default',
                    model_id VARCHAR(128) DEFAULT '',
                    severity VARCHAR(32) DEFAULT 'WARNING',
                    category VARCHAR(64) NOT NULL,
                    message TEXT NOT NULL,
                    details JSON,
                    acknowledged BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP
                )
            """))


def downgrade(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS system_alerts"))
        conn.execute(text("DROP TABLE IF EXISTS system_metrics"))
