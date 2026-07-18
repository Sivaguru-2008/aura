"""Tests for Step 13 — integration of real MIMIC-CXR into the app."""
from __future__ import annotations

import asyncio

import pytest

from mimic.config import get_mimic_paths

PATHS = get_mimic_paths()
HAS_DATA = PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def test_app_imports_with_integration_wired():
    # The gateway must import cleanly with the new seeder branch in place.
    import gateway.app as app
    assert hasattr(app, "app")
    assert hasattr(app, "seed")           # synthetic fallback still imported


def test_seed_mimic_is_drop_in_signature():
    import inspect
    from mimic.seed import seed_mimic
    sig = inspect.signature(seed_mimic)
    # same leading params as gateway.seed.seed (store, pipeline, ...)
    params = list(sig.parameters)
    assert params[:2] == ["store", "pipeline"]
    assert inspect.iscoroutinefunction(seed_mimic)


@needs_data
def test_seed_mimic_populates_real_cases(tmp_path):
    from gateway.pipeline import Pipeline
    from gateway.storage import Store
    from mimic.seed import seed_mimic

    store = Store(tmp_path / "t.db")
    pipe = Pipeline()
    n = asyncio.run(seed_mimic(store, pipe, n=3))
    assert n == 3
    assert store.count() == 3
    cases = store.list_cases(limit=5)
    assert all(c["case_id"].startswith("CASE-MIMIC-") for c in cases)
    # every case carries a full bundle with a report-derived ground truth
    b = store.get_case(cases[0]["case_id"])
    assert b.vision is not None and b.safety is not None and b.report is not None
    assert b.ground_truth is not None


@needs_data
def test_seed_mimic_is_idempotent(tmp_path):
    from gateway.pipeline import Pipeline
    from gateway.storage import Store
    from mimic.seed import seed_mimic

    store = Store(tmp_path / "t.db")
    pipe = Pipeline()
    asyncio.run(seed_mimic(store, pipe, n=2))
    again = asyncio.run(seed_mimic(store, pipe, n=2))   # already populated
    assert again == store.count() == 2
