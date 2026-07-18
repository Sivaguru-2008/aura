"""Step 8 — ML task datasets for real MIMIC-CXR patients.

Turns the engineered feature matrix (Step 6) + leakage-safe splits (Step 7) into
supervised ``(X, y)`` datasets for concrete learning tasks.

Honesty about task availability:
    Backed by data (labels come from radiology reports):
        * ``diagnosis_prediction``      — 6-class (the plan's "Disease Prediction")
        * ``finding_classification``    — 7-way multi-label
        * ``pneumothorax_detection``    — binary (acute, safety-relevant)
    NOT backed by data (require MIMIC-IV outcomes/vitals/ICU/admissions):
        * mortality / readmission / length_of_stay / sepsis / shock / icu_transfer

The unavailable tasks are registered with ``available=False`` and a reason, so
callers can enumerate the full plan and see precisely what is missing and why —
instead of a silently fabricated label column.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from mimic.config import MimicPaths, get_mimic_paths
from mimic.features import FeatureEngineer, feature_names
from mimic.splits import DatasetBuilder
from schemas.clinical import DIAGNOSES, FINDINGS

log = logging.getLogger("mimic.tasks")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str                 # "multiclass" | "multilabel" | "binary"
    target: str                    # label column, or comma-joined for multilabel
    n_classes: int
    available: bool
    reason: str = ""               # why unavailable (when available is False)
    plan_name: str = ""            # the name used in the Step-8 plan


# Source split for each task split (train/test share the 'train' CSV source).
_SOURCE = {"train": "train", "test": "train", "validation": "validate"}

FINDING_LABEL_COLS = [f"label_finding_{f.value}" for f in FINDINGS]

TASK_REGISTRY: dict[str, TaskSpec] = {
    "diagnosis_prediction": TaskSpec(
        "diagnosis_prediction", "multiclass", "label_diagnosis",
        n_classes=len(DIAGNOSES), available=True, plan_name="Disease Prediction",
    ),
    "finding_classification": TaskSpec(
        "finding_classification", "multilabel", ",".join(FINDING_LABEL_COLS),
        n_classes=len(FINDINGS), available=True, plan_name="(imaging findings)",
    ),
    "pneumothorax_detection": TaskSpec(
        "pneumothorax_detection", "binary", "label_finding_pneumothorax",
        n_classes=2, available=True, plan_name="(acute finding)",
    ),
    # ---- Plan tasks with no backing data in MIMIC-CXR ----
    "mortality_prediction": TaskSpec(
        "mortality_prediction", "binary", "", 2, available=False,
        reason="no death/discharge outcome table (MIMIC-IV hosp.admissions/patients)",
        plan_name="Mortality Prediction",
    ),
    "readmission_prediction": TaskSpec(
        "readmission_prediction", "binary", "", 2, available=False,
        reason="no admissions table (MIMIC-IV hosp.admissions)",
        plan_name="Readmission Prediction",
    ),
    "length_of_stay_prediction": TaskSpec(
        "length_of_stay_prediction", "regression", "", 1, available=False,
        reason="no admit/discharge timestamps (MIMIC-IV hosp.admissions)",
        plan_name="Length of Stay Prediction",
    ),
    "sepsis_prediction": TaskSpec(
        "sepsis_prediction", "binary", "", 2, available=False,
        reason="no labs/vitals/cultures for Sepsis-3 (MIMIC-IV hosp.labevents, icu.chartevents)",
        plan_name="Sepsis Prediction",
    ),
    "shock_prediction": TaskSpec(
        "shock_prediction", "binary", "", 2, available=False,
        reason="no vitals/vasopressor data (MIMIC-IV icu.chartevents/inputevents)",
        plan_name="Shock Prediction",
    ),
    "icu_transfer_prediction": TaskSpec(
        "icu_transfer_prediction", "binary", "", 2, available=False,
        reason="no ICU/transfers table (MIMIC-IV icu.icustays, hosp.transfers)",
        plan_name="ICU Transfer Prediction",
    ),
}


def list_tasks() -> list[TaskSpec]:
    return list(TASK_REGISTRY.values())


def available_tasks() -> list[TaskSpec]:
    return [t for t in TASK_REGISTRY.values() if t.available]


@dataclass
class TaskDataset:
    task: TaskSpec
    split: str
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    classes: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    def class_balance(self) -> dict:
        if self.task.task_type == "multilabel":
            return {c: float(self.y[:, i].mean()) for i, c in enumerate(self.classes)}
        vals, counts = np.unique(self.y, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(vals, counts)}


class TaskDatasetBuilder:
    """Materializes ``(X, y)`` for a task+split from features + split manifests."""

    def __init__(self, paths: Optional[MimicPaths] = None):
        self.paths = paths or get_mimic_paths()
        self.fe = FeatureEngineer(self.paths)
        self._splitter = DatasetBuilder(self.paths)

    def _manifest_subject_ids(self, split: str) -> Optional[set[int]]:
        """subject_ids belonging to a split (from the written manifest CSV)."""
        if split == "validation":
            return None                              # whole 'validate' source is the val set
        fp = self._splitter.out_dir() / f"{split}.csv"
        if not fp.is_file():
            raise FileNotFoundError(
                f"split manifest missing: {fp} — run DatasetBuilder.build_and_write() first"
            )
        return set(pd.read_csv(fp)["subject_id"].astype(int))

    def build(
        self, task_name: str, split: str, limit: Optional[int] = None
    ) -> TaskDataset:
        if task_name not in TASK_REGISTRY:
            raise KeyError(f"unknown task {task_name!r}; known: {list(TASK_REGISTRY)}")
        spec = TASK_REGISTRY[task_name]
        if not spec.available:
            raise ValueError(
                f"task {task_name!r} is not available for MIMIC-CXR: {spec.reason}"
            )
        if split not in _SOURCE:
            raise ValueError(f"unknown split {split!r} (train|validation|test)")

        source = _SOURCE[split]
        subject_ids = self._manifest_subject_ids(split)
        frame = self.fe.build_frame(source, limit=limit, subject_ids=subject_ids)
        return self._to_xy(spec, split, frame)

    def _to_xy(self, spec: TaskSpec, split: str, frame: pd.DataFrame) -> TaskDataset:
        feats = [c for c in feature_names() if c in frame.columns]
        X = frame[feats].astype(float)

        if spec.task_type == "multilabel":
            cols = spec.target.split(",")
            y = frame[cols].astype(int).to_numpy()
            classes = [c.replace("label_finding_", "") for c in cols]
        elif spec.task_type == "binary":
            y = frame[spec.target].astype(int).to_numpy()
            classes = ["negative", "positive"]
        else:  # multiclass
            y = frame[spec.target].astype(int).to_numpy()
            classes = [d.value for d in DIAGNOSES]

        log.info("task=%s split=%s X=%s y=%s", spec.name, split, X.shape, y.shape)
        return TaskDataset(spec, split, X, y, feats, classes)
