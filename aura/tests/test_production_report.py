"""Regression tests for Step 5 — full clinical report renderer (schema untouched)."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ml.data import IMG, make_multimodal, make_sample
from schemas.clinical import Diagnosis
from schemas.contracts import StudyInput
from services.report.clinical_report import (
    build_clinical_report,
    render_html,
    render_markdown,
    render_text,
    save_report,
)

REQUIRED_SECTIONS = [
    "patient_summary", "vision_findings", "confidence", "calibration",
    "differential_diagnosis", "evidence_used", "evidence_missing",
    "recommended_tests", "risk_level", "clinical_impression", "limitations",
    "model_version", "inference_time_s",
]


@pytest.fixture(scope="module")
def bundle():
    from gateway.pipeline import Pipeline

    pipe = Pipeline()
    rng = np.random.default_rng(1)
    s = make_sample(Diagnosis.HEART_FAILURE, rng)
    study = StudyInput(
        study_id="RPT1", image=[float(v) for v in s.image.flatten()],
        image_shape=(IMG, IMG), priors=s.priors,
        multimodal=make_multimodal(Diagnosis.HEART_FAILURE, rng),
        ground_truth=Diagnosis.HEART_FAILURE)
    return asyncio.run(pipe.run(study, "CASE-RPT1"))


def test_report_has_all_sections(bundle):
    rep = build_clinical_report(bundle, inference_time_s=0.5)
    for key in REQUIRED_SECTIONS:
        assert key in rep, f"missing section {key}"
    assert len(rep["vision_findings"]) == 7
    assert rep["risk_level"]["level"] in {
        "HIGH", "MODERATE", "LOW-MODERATE", "LOW", "REVIEW", "INDETERMINATE", "UNKNOWN"}
    assert rep["model_version"]["vision"].startswith("vision-cxr")
    assert rep["inference_time_s"] == 0.5


def test_renderers_produce_text(bundle):
    rep = build_clinical_report(bundle, inference_time_s=0.1)
    md = render_markdown(rep)
    assert "Patient Summary" in md and "Differential Diagnosis" in md
    assert "Limitations" in md and "Risk Level" in md
    txt = render_text(rep)
    assert "Patient Summary" in txt
    html = render_html(rep)
    assert html.startswith("<!doctype html>") and "Clinical Report" in html


def test_save_report_writes_files(bundle, tmp_path):
    rep = build_clinical_report(bundle, inference_time_s=0.2)
    paths = save_report(rep, tmp_path, stem="clin")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
