"""Step 6 — Feature engineering for real MIMIC-CXR patients.

Two hard rules shape this module:

1. **No target leakage.** The report-derived findings/diagnosis are *labels*
   (see :mod:`mimic.labeling`), so they are NEVER used as features. Features come
   only from the image and non-label metadata. Labels live in ``label_*`` columns.

2. **Honesty about missing modalities.** The plan asks for Charlson, SOFA, qSOFA,
   NEWS2, BMI, lab trends, medication history, LOS, ICU duration, previous
   admissions. None of these have backing data in MIMIC-CXR. Rather than fake
   them, each is emitted as ``0.0`` with a companion ``*_missing = 1.0`` indicator
   — so the feature schema matches the plan and every model can learn to ignore
   them, and the day a MIMIC-IV linkage arrives they light up with no code change.

Real, non-leaking features produced:
    * image content   — 10 anatomical region features (reuses the vision extractor)
                        + intensity statistics of the radiograph
    * imaging metadata — study/image/view counts, frontal/lateral availability
    * temporal        — number of studies, longitudinal flag (LOS proxy)
    * demographics    — one-hot (all "unknown" here; missing-flagged)
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from mimic.config import MimicPaths, get_mimic_paths
from mimic.patient import Patient, iter_patients
from schemas.clinical import DIAGNOSES, FINDINGS, Diagnosis, Finding
from services.vision.features import FEATURE_NAMES as IMG_FEATURE_NAMES, extract_features

log = logging.getLogger("mimic.features")

# Tabular clinical features the plan lists but MIMIC-CXR cannot back — emitted as
# 0.0 + a *_missing=1.0 indicator so the schema is complete and honest.
CLINICAL_NA_FEATURES = [
    "charlson_comorbidity_index",
    "sofa_score",
    "qsofa_score",
    "news2_score",
    "bmi",
    "hospital_los_days",
    "icu_duration_hours",
    "previous_admissions",
    "medication_count",
    "lab_wbc_last",
    "lab_creatinine_last",
    "lab_lactate_last",
]


@dataclass
class FeatureRow:
    subject_id: int
    features: dict[str, float]
    labels: dict[str, float]

    def flat(self) -> dict[str, float]:
        row: dict[str, float] = {"subject_id": float(self.subject_id)}
        row.update(self.features)
        row.update(self.labels)
        return row


def _image_features(img: np.ndarray) -> dict[str, float]:
    """10 anatomical region features + intensity statistics of the radiograph."""
    feats = dict(extract_features(img))                 # non-leaking, reused
    flat = img.reshape(-1)
    feats.update(
        {
            "img_mean": float(flat.mean()),
            "img_std": float(flat.std()),
            "img_p10": float(np.percentile(flat, 10)),
            "img_p90": float(np.percentile(flat, 90)),
            "img_dark_frac": float((flat < 0.2).mean()),
            "img_bright_frac": float((flat > 0.8).mean()),
        }
    )
    return feats


def _demographic_features(p: Patient) -> dict[str, float]:
    sex = (p.sex or "unknown").upper()
    return {
        "age": 0.0,
        "age_missing": 1.0,                              # no patients.csv
        "sex_M": 1.0 if sex == "M" else 0.0,
        "sex_F": 1.0 if sex == "F" else 0.0,
        "sex_unknown": 1.0 if sex not in ("M", "F") else 0.0,
    }


def _metadata_features(p: Patient) -> dict[str, float]:
    n_ap = len(p.images_by_view.get("AP", []))
    n_pa = len(p.images_by_view.get("PA", []))
    n_lat = len(p.images_by_view.get("Lateral", []))
    n_studies = max(1, p.n_studies)
    return {
        "n_studies": float(p.n_studies),
        "n_images": float(len(p.images)),
        "n_ap": float(n_ap),
        "n_pa": float(n_pa),
        "n_lateral": float(n_lat),
        "has_frontal": 1.0 if (n_ap + n_pa) > 0 else 0.0,
        "has_lateral": 1.0 if n_lat > 0 else 0.0,
        "images_per_study": float(len(p.images)) / n_studies,
        "is_longitudinal": 1.0 if p.n_studies > 1 else 0.0,   # multi-study (LOS proxy)
    }


def _clinical_na_features() -> dict[str, float]:
    out: dict[str, float] = {}
    for name in CLINICAL_NA_FEATURES:
        out[name] = 0.0
        out[f"{name}_missing"] = 1.0
    return out


def _labels(p: Patient) -> dict[str, float]:
    """Report-derived targets — kept strictly separate from features."""
    labels: dict[str, float] = {}
    # multi-label finding classification targets (1 = present)
    for f in FINDINGS:
        labels[f"label_finding_{f.value}"] = 1.0 if p.label.findings.get(f) == 1 else 0.0
    # diagnosis target (categorical + per-class score)
    labels["label_diagnosis"] = float(DIAGNOSES.index(p.diagnosis))
    labels["label_diagnosis_name"] = p.diagnosis.value            # string, convenience
    return labels


def patient_feature_row(p: Patient, grid: int = 64) -> Optional[FeatureRow]:
    """Build one feature+label row for a patient from its most recent study.

    Returns ``None`` if the patient has no usable image (already rare — loader
    filters image-less patients).
    """
    if p.n_studies == 0:
        return None
    try:
        si = p.to_study_input(study_index=-1, grid=grid)
    except (ValueError, OSError) as e:
        log.warning("subject %s: no feature row (%s)", p.subject_id, e)
        return None
    img = np.asarray(si.image, dtype=float).reshape(si.image_shape)

    features: dict[str, float] = {}
    features.update(_image_features(img))
    features.update(_metadata_features(p))
    features.update(_demographic_features(p))
    features.update(_clinical_na_features())
    return FeatureRow(subject_id=p.subject_id, features=features, labels=_labels(p))


def feature_names() -> list[str]:
    """Canonical ordered feature-column names (excludes labels & subject_id)."""
    names: list[str] = []
    names += list(IMG_FEATURE_NAMES)
    names += ["img_mean", "img_std", "img_p10", "img_p90", "img_dark_frac", "img_bright_frac"]
    names += [
        "n_studies", "n_images", "n_ap", "n_pa", "n_lateral",
        "has_frontal", "has_lateral", "images_per_study", "is_longitudinal",
    ]
    names += ["age", "age_missing", "sex_M", "sex_F", "sex_unknown"]
    for n in CLINICAL_NA_FEATURES:
        names += [n, f"{n}_missing"]
    return names


class FeatureEngineer:
    """Streams patients → engineered feature rows; builds/caches a feature frame."""

    def __init__(self, paths: Optional[MimicPaths] = None) -> None:
        self.paths = paths or get_mimic_paths()

    def iter_rows(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        subject_ids: Optional[set[int]] = None,
        **loader_kwargs,
    ) -> Iterator[FeatureRow]:
        """Yield feature rows for a source split, optionally restricted to
        ``subject_ids`` (used to materialize a specific train/test manifest)."""
        for p in iter_patients(split, paths=self.paths, limit=limit, **loader_kwargs):
            if subject_ids is not None and p.subject_id not in subject_ids:
                continue
            row = patient_feature_row(p)
            if row is not None:
                yield row

    def build_frame(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        subject_ids: Optional[set[int]] = None,
        **loader_kwargs,
    ) -> pd.DataFrame:
        rows = [
            r.flat()
            for r in self.iter_rows(split, limit=limit, subject_ids=subject_ids, **loader_kwargs)
        ]
        df = pd.DataFrame(rows)
        log.info("built feature frame: split=%s rows=%d cols=%d", split, len(df), df.shape[1])
        return df

    def cache_path(self, split: str) -> "Path":  # type: ignore[name-defined]
        from pathlib import Path
        return self.paths.cache_dir / f"features_{split}.parquet"

    def build_frame_cached(
        self, split: str = "train", rebuild: bool = False
    ) -> pd.DataFrame:
        cp = self.cache_path(split)
        if cp.is_file() and not rebuild:
            log.info("loading cached features %s", cp)
            return pd.read_parquet(cp)
        df = self.build_frame(split)
        cp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp, index=False)
        log.info("cached features -> %s (%d rows)", cp, len(df))
        return df
