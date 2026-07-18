"""Tests for Step 8 — ML task datasets."""
from __future__ import annotations

import numpy as np
import pytest

from mimic.config import get_mimic_paths
from mimic.tasks import (
    TASK_REGISTRY,
    TaskDatasetBuilder,
    available_tasks,
    list_tasks,
)

PATHS = get_mimic_paths()
HAS_DATA = PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def test_registry_covers_plan_tasks():
    plan = {
        "Mortality Prediction", "Readmission Prediction", "Length of Stay Prediction",
        "Sepsis Prediction", "Shock Prediction", "ICU Transfer Prediction",
        "Disease Prediction",
    }
    registered_plan_names = {t.plan_name for t in list_tasks()}
    assert plan.issubset(registered_plan_names)


def test_unavailable_tasks_have_reasons():
    for t in list_tasks():
        if not t.available:
            assert t.reason, f"{t.name} unavailable without a reason"


def test_available_tasks_are_the_three_imaging_tasks():
    names = {t.name for t in available_tasks()}
    assert names == {
        "diagnosis_prediction", "finding_classification", "pneumothorax_detection"
    }


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        TaskDatasetBuilder().build("nope", "validation")


@needs_data
def test_unavailable_task_build_raises():
    with pytest.raises(ValueError):
        TaskDatasetBuilder().build("mortality_prediction", "validation")


@needs_data
def test_multiclass_dataset_shapes():
    ds = TaskDatasetBuilder().build("diagnosis_prediction", "validation")
    assert ds.X.shape[0] == ds.y.shape[0]
    assert ds.X.shape[1] == len(ds.feature_names)
    assert ds.y.ndim == 1
    assert set(np.unique(ds.y)).issubset(set(range(6)))
    # no label columns leaked into X
    assert not any(c.startswith("label_") for c in ds.X.columns)


@needs_data
def test_multilabel_dataset_shapes():
    ds = TaskDatasetBuilder().build("finding_classification", "validation")
    assert ds.y.shape == (ds.X.shape[0], 7)
    assert set(np.unique(ds.y)).issubset({0, 1})


@needs_data
def test_binary_dataset_is_two_class():
    ds = TaskDatasetBuilder().build("pneumothorax_detection", "validation")
    assert ds.y.ndim == 1
    assert set(np.unique(ds.y)).issubset({0, 1})
