"""Tests for Step 3 — report labeling + data cleaning."""
from __future__ import annotations

import pytest

from mimic.cleaning import CleanedPatient, _clean_text, _dedup, clean_record
from mimic.labeling import label_report, label_patient_reports
from mimic.loaders import PatientRecord
from schemas.clinical import Diagnosis, Finding


# --------------------------------------------------------------------------- #
# Labeler — negation, uncertainty, diagnosis mapping
# --------------------------------------------------------------------------- #
def test_forward_negation_across_comma_list():
    rl = label_report("Lungs are clear without consolidation, effusion, or pneumothorax.")
    assert rl.concepts["consolidation"] == 0
    assert rl.concepts["effusion"] == 0
    assert rl.concepts["pneumothorax"] == 0
    assert rl.diagnosis == Diagnosis.NORMAL


def test_positive_effusion_and_edema_to_heart_failure():
    rl = label_report("Moderate cardiomegaly with pulmonary edema and pleural effusions.")
    assert rl.concepts["edema"] == 1
    assert rl.concepts["cardiomegaly"] == 1
    assert rl.diagnosis == Diagnosis.HEART_FAILURE
    assert Finding.CARDIOMEGALY in rl.positive_findings


def test_nodule_to_malignancy():
    rl = label_report("Spiculated nodule in the right upper lobe.")
    assert rl.diagnosis == Diagnosis.MALIGNANCY


def test_pneumothorax_positive():
    rl = label_report("Moderate right pneumothorax is present.")
    assert rl.diagnosis == Diagnosis.PNEUMOTHORAX


def test_hyperinflation_to_copd():
    rl = label_report("Hyperinflated lungs with flattened diaphragms.")
    assert rl.diagnosis == Diagnosis.COPD


def test_uncertainty_marked_not_positive():
    rl = label_report("Findings possibly represent early pneumonia.")
    assert rl.concepts["pneumonia"] == -1  # uncertain, not asserted positive


def test_no_acute_process_is_normal():
    rl = label_report("No acute cardiopulmonary process.")
    assert rl.diagnosis == Diagnosis.NORMAL
    assert rl.normal_cue is True


def test_empty_report_defaults_normal():
    assert label_report("").diagnosis == Diagnosis.NORMAL


def test_patient_aggregation_takes_highest_acuity():
    reports = [
        "No acute cardiopulmonary process.",
        "Small right pneumothorax.",
        "Lungs are clear.",
    ]
    summary, per = label_patient_reports(reports)
    assert len(per) == 3
    assert summary.diagnosis == Diagnosis.PNEUMOTHORAX  # highest acuity across studies


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def test_dedup_preserves_order():
    out, removed = _dedup(["a", "b", "a", "c", "b"])
    assert out == ["a", "b", "c"]
    assert removed == 2


def test_clean_text_collapses_ws_and_placeholders():
    assert _clean_text("compared  to  ___   prior") == "compared to prior"


def test_clean_record_dedupes_and_labels():
    rec = PatientRecord(
        subject_id=7,
        images=["x.jpg", "x.jpg", "y.jpg"],
        images_by_view={"AP": ["x.jpg"], "LL": ["y.jpg"]},
        reports=["Moderate right pneumothorax.", "Moderate right pneumothorax.", "  "],
        reports_aug=[],
        n_images_referenced=3,
        n_images_present=3,
    )
    cp, st = clean_record(rec)
    assert isinstance(cp, CleanedPatient)
    assert cp.images == ["x.jpg", "y.jpg"]          # dup image removed
    assert st.dup_images_removed == 1
    assert cp.reports == ["Moderate right pneumothorax."]  # dup + blank removed
    assert st.dup_reports_removed == 1
    assert st.blank_reports_removed == 1
    assert set(cp.images_by_view) == {"AP", "Lateral"}     # LL normalized to Lateral
    assert cp.diagnosis == Diagnosis.PNEUMOTHORAX
