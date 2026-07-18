"""Tests for Step 5 — unified Patient Object + StudyInput adapter."""
from __future__ import annotations

import pytest

from mimic.config import get_mimic_paths
from mimic.loaders import MimicCxrLoader, PatientRecord
from mimic.patient import Patient, build_patient, iter_patients
from schemas.clinical import Diagnosis
from schemas.contracts import StructuredPriors, StudyInput

PATHS = get_mimic_paths()
HAS_DATA = PATHS.validate_csv.is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIMIC-CXR corpus not mounted")


def _rec() -> PatientRecord:
    return PatientRecord(
        subject_id=99,
        images=["files/p10/p99/s100/a.jpg", "files/p10/p99/s200/b.jpg"],
        images_by_view={"PA": ["files/p10/p99/s100/a.jpg"], "AP": ["files/p10/p99/s200/b.jpg"]},
        reports=["Moderate cardiomegaly with pulmonary edema.", "No acute cardiopulmonary process."],
        reports_aug=[],
    )


def test_build_patient_shape():
    p = build_patient(_rec())
    assert isinstance(p, Patient)
    assert p.subject_id == 99
    assert p.patient_id == "MIMIC-99"
    assert p.n_studies == 2
    assert p.radiology == p.notes           # reports serve both roles
    assert len(p.radiology) == 2


def test_schema_parity_fields_present_but_none():
    # MIMIC-IV fields exist for schema parity, honestly empty (no such data).
    p = build_patient(_rec())
    assert p.hadm_id is None and p.stay_id is None
    assert p.age is None and p.sex == "unknown" and p.race == "unknown"
    assert p.labs == [] and p.vitals == [] and p.medications == [] and p.procedures == []


def test_priors_are_valid_contract():
    p = build_patient(_rec())
    pri = p.priors()
    assert isinstance(pri, StructuredPriors)
    assert pri.age_band == "unknown"


def test_representative_image_prefers_frontal():
    p = build_patient(_rec())
    ev = p.timeline.events[0]                # study s100, PA view
    assert p._representative_image(ev) == "files/p10/p99/s100/a.jpg"


def test_no_studies_raises_on_adapter():
    p = build_patient(PatientRecord(subject_id=5, images=[], reports=[], reports_aug=[]))
    with pytest.raises(ValueError):
        p.to_study_input()


# --------------------------------------------------------------------------- #
# Real-data: the adapter must load an actual JPG into a valid StudyInput.
# --------------------------------------------------------------------------- #
@needs_data
def test_to_study_input_loads_real_image():
    pt = next(p for p in iter_patients("validate", limit=20) if p.n_studies >= 1)
    si = pt.to_study_input(-1)
    assert isinstance(si, StudyInput)
    assert si.image_shape == (64, 64)
    assert len(si.image) == 64 * 64
    assert 0.0 <= min(si.image) and max(si.image) <= 1.0
    assert isinstance(si.ground_truth, Diagnosis)
    # not a constant image (real content)
    assert max(si.image) - min(si.image) > 0.1


@needs_data
def test_iter_study_inputs_covers_all_studies():
    pt = next(p for p in iter_patients("validate", limit=40) if p.n_studies >= 3)
    sis = list(pt.iter_study_inputs())
    assert len(sis) == pt.n_studies
    assert all(isinstance(s, StudyInput) for s in sis)
