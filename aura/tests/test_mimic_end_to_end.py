"""Step 15 — end-to-end smoke test tying the whole MIMIC-CXR pipeline together.

Exercises, on real data, the full chain the plan lists under "prediction pipeline":
    loaders -> patient merge -> timeline -> features -> split -> task dataset
    -> training -> prediction -> evaluation.
Bounded (small limits) so it runs in CI-reasonable time when the corpus is mounted.
"""
from __future__ import annotations

import numpy as np
import pytest

from mimic.config import get_mimic_paths

PATHS = get_mimic_paths()
HAS_DATA = PATHS.train_csv.is_file() and PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")

SPLITS_BUILT = (get_mimic_paths().cache_dir / "splits" / "test.csv").is_file()
needs_splits = pytest.mark.skipif(not SPLITS_BUILT, reason="run DatasetBuilder.build_and_write first")


@needs_data
def test_full_prediction_pipeline_diagnosis():
    from mimic.tasks import TaskDatasetBuilder
    from mimic.training import GBMTrainer
    from mimic.evaluation import evaluate_multiclass

    tdb = TaskDatasetBuilder()
    train = tdb.build("diagnosis_prediction", "train", limit=1500)
    val = tdb.build("diagnosis_prediction", "validation")

    # train -> predict -> evaluate
    res = GBMTrainer(train).train(val)
    assert res.proba_val.shape == (val.n_samples, 6)
    assert np.allclose(res.proba_val.sum(1), 1.0, atol=1e-3)

    rep = evaluate_multiclass(res.proba_val, res.y_val, train.classes)
    # sanity: better-than-random ranking overall, valid calibration numbers
    assert rep["macro_auroc"] > 0.55
    assert 0.0 <= rep["ece"] <= 1.0
    assert np.array(rep["confusion_matrix"]).sum() == val.n_samples


@needs_data
@needs_splits
def test_test_split_is_predictable_and_disjoint():
    import pandas as pd
    from mimic.tasks import TaskDatasetBuilder

    out = get_mimic_paths().cache_dir / "splits"
    train_ids = set(pd.read_csv(out / "train.csv")["subject_id"])
    test_ids = set(pd.read_csv(out / "test.csv")["subject_id"])
    assert train_ids.isdisjoint(test_ids)             # no leakage into evaluation

    # a small test-split materialization yields usable feature rows
    ds = TaskDatasetBuilder().build("diagnosis_prediction", "test", limit=300)
    assert ds.n_samples > 0
    assert ds.X.shape[1] == len(ds.feature_names)


@needs_data
def test_patient_merge_to_pipeline_bundle():
    """Patient Object -> StudyInput -> full pipeline -> CaseBundle (no synthetic data)."""
    import asyncio
    from gateway.pipeline import Pipeline
    from mimic.patient import iter_patients

    patient = next(p for p in iter_patients("validate", limit=10) if p.n_studies >= 1)
    study = patient.to_study_input(-1)
    bundle = asyncio.run(Pipeline().run(study, case_id=f"CASE-{patient.subject_id}"))
    assert bundle.vision is not None
    assert bundle.safety is not None
    assert bundle.report is not None
    assert bundle.ground_truth is not None           # from the real report
