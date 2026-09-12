"""The case store behaves the same on SQLite and PostgreSQL.

SQLite runs everywhere. The PostgreSQL leg runs when ``AURA_TEST_DATABASE_URL`` points
at a server where the user can create schemas. Each test gets its own schema, dropped
afterwards, so the suite never touches data it did not create:

    AURA_TEST_DATABASE_URL=postgresql+psycopg://aura:aura@127.0.0.1:55432/aura \\
        python -m pytest tests/test_storage_backends.py

What is tested here is what differs between the engines, or would lose data if it did:
the audit hash chain under concurrent writers (a fork reads as tampering), timestamps
surviving the round trip the hash depends on, the one-outcome-per-case constraint, the
site binding that keeps hospitals' databases apart, and URL handling.
"""
from __future__ import annotations

import os
import threading
import uuid

import numpy as np
import pytest
from sqlalchemy import text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aura.gateway.storage import (
    AuditRow,
    SiteMismatchError,
    Store,
    engine_for,
    verify_audit_trail,
)
from aura.schemas.clinical import Diagnosis
from aura.schemas.contracts import (
    CaseBundle,
    FusionResult,
    SafetyAssessment,
    StructuredPriors,
    VisionResult,
)

PG_URL = os.environ.get("AURA_TEST_DATABASE_URL", "")


@pytest.fixture(params=["sqlite", "postgresql"])
def open_store(request, tmp_path):
    """A factory ``open_store(site_id=None) -> Store``. Every call opens the same DB."""
    stores: list[Store] = []

    if request.param == "sqlite":
        db = tmp_path / "store.db"

        def _open(site_id=None):
            s = Store(db, site_id=site_id)
            stores.append(s)
            return s

        yield _open
        for s in stores:
            s.engine.dispose()                    # Windows keeps the file locked otherwise
        return

    if not PG_URL:
        pytest.skip("set AURA_TEST_DATABASE_URL to run the PostgreSQL leg")
    schema = f"aura_test_{uuid.uuid4().hex[:12]}"
    admin = engine_for(PG_URL)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    url = make_url(PG_URL).update_query_dict({"options": f"-csearch_path={schema}"})
    url_text = url.render_as_string(hide_password=False)

    def _open(site_id=None):
        s = Store(tmp_path / "unused.db", url=url_text, site_id=site_id)
        stores.append(s)
        return s

    try:
        yield _open
    finally:
        for s in stores:
            s.engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _bundle(case_id: str) -> CaseBundle:
    study = f"STU-{case_id}"
    return CaseBundle(
        case_id=case_id,
        study_id=study,
        state="ready",
        priors=StructuredPriors(),
        vision=VisionResult(study_id=study, findings=[], embedding=[], model_version=""),
        fusion=FusionResult(study_id=study, backend="classical", posterior={},
                            evidence_vector=[0.1] * 8),
        safety=SafetyAssessment(study_id=study, predictions=[], conformal_set=[],
                                top=Diagnosis.NORMAL, top_probability=0.9),
    )


def test_cases_round_trip(open_store):
    store = open_store()
    store.save_case(_bundle("CASE-1"))
    store.save_case(_bundle("CASE-2"))
    assert store.count() == 2
    got = store.get_case("CASE-1")
    assert got is not None and got.fusion.evidence_vector == [0.1] * 8
    assert {r["case_id"] for r in store.list_cases()} == {"CASE-1", "CASE-2"}


def test_feedback_is_readable_through_the_store(open_store):
    """recalibrate.py reads feedback here, not through a raw sqlite3 handle."""
    store = open_store()
    store.add_feedback("CASE-1", "normal", "incorrect", "pneumonia")
    assert store.list_feedback() == [{"case_id": "CASE-1", "diagnosis": "normal",
                                      "verdict": "incorrect", "correction": "pneumonia"}]


def test_audit_chain_verifies_after_a_round_trip(open_store):
    """The hash covers created_at, so timestamps must come back exactly as written."""
    store = open_store()
    for i in range(5):
        store.audit(f"action-{i}", entity_type="test", entity_id=str(i), detail={"i": i})
    assert verify_audit_trail(store) is True


def test_audit_chain_detects_tampering(open_store):
    store = open_store()
    for i in range(3):
        store.audit(f"action-{i}", entity_id=str(i))
    with Session(store.engine) as ses:
        ses.execute(update(AuditRow).where(AuditRow.entity_id == "1").values(action="edited"))
        ses.commit()
    assert verify_audit_trail(store) is False


def test_concurrent_audit_writers_do_not_fork_the_chain(open_store):
    """FastAPI serves sync handlers on a thread pool, so audit writes genuinely race.

    Without the append lock two writers read the same chain head, both link to it, and
    verification then reports the fork as tampering.
    """
    store = open_store()
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(10):
                store.audit(f"w{n}-{i}", entity_id=f"{n}-{i}")
        except Exception as exc:                  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(store.recent_audit(100)) == 60
    assert verify_audit_trail(store) is True


def test_one_verified_outcome_per_case(open_store):
    store = open_store()
    probs = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    store.record_outcome("CASE-1", probs, 0, true_diagnosis="normal")
    with pytest.raises(IntegrityError):
        store.record_outcome("CASE-1", probs, 0, true_diagnosis="normal")


def test_a_fresh_database_binds_to_the_first_site(open_store):
    store = open_store(site_id="hospital-a")
    assert store.site_id == "hospital-a"
    assert any(r["action"] == "site.bound" for r in store.recent_audit(10))
    assert open_store(site_id="hospital-a").site_id == "hospital-a"


def test_a_server_for_another_site_refuses_the_database(open_store):
    open_store(site_id="hospital-a")
    with pytest.raises(SiteMismatchError, match="hospital-a"):
        open_store(site_id="hospital-b")


def test_a_server_with_no_site_refuses_a_bound_database(open_store):
    open_store(site_id="hospital-a")
    with pytest.raises(SiteMismatchError, match="AURA_SITE_ID='hospital-a'"):
        open_store()


def test_an_unbound_database_stays_unbound_without_a_site(open_store):
    """The single-box demo sets no site and must keep working exactly as before."""
    assert open_store().site_id == ""
    assert open_store().site_id == ""


def test_postgres_urls_get_the_psycopg3_driver_and_utc():
    pytest.importorskip("psycopg")
    for raw in ("postgresql://u:p@db:5432/aura", "postgres://u:p@db:5432/aura"):
        url = engine_for(raw).url
        assert url.drivername == "postgresql+psycopg"
        assert "timezone=UTC" in url.query["options"]


def test_a_per_site_search_path_survives_the_utc_pin():
    pytest.importorskip("psycopg")
    url = engine_for("postgresql+psycopg://u:p@db/aura?options=-csearch_path%3Dsite_a").url
    assert url.query["options"].startswith("-csearch_path=site_a")
    assert url.query["options"].endswith("-c timezone=UTC")
