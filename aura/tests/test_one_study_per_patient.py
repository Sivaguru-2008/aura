"""The fusion evidence set has one study per patient — verified by code, not assumed.

The paper's patient-level bootstrap (§2.5) treats each test study as a distinct patient.
Two rules make that true and both are pinned here, without the MIMIC-CXR corpus:

1. Within a patient, the timeline is ordered by study id, so ``study_index=-1`` selects
   the highest-id study and only that one is emitted.
2. Across rows, ``build_real_evidence_dataset`` emits at most one row per ``subject_id``
   even if the source yields the same subject twice.
"""
from types import SimpleNamespace

import numpy as np

from aura.mimic.loaders import PatientRecord
from aura.mimic.timeline import build_timeline
from aura.ml.training import dataset as ds
from aura.schemas.clinical import DIAGNOSES


def _img(subject: int, study: str) -> str:
    return f"files/p10/p{subject}/{study}/view1.jpg"


def test_timeline_orders_by_study_id_so_last_is_highest():
    rec = PatientRecord(
        subject_id=10000001,
        images=[_img(10000001, "s59000000"), _img(10000001, "s51000000")],  # out of order
        reports=["later report", "earlier report"],
    )
    tl = build_timeline(rec)
    assert [e.study_id for e in tl.events] == ["s51000000", "s59000000"]
    assert tl.events[-1].study_id == "s59000000"


class _FakePatient:
    """Two studies per patient; to_study_input(-1) returns the later one."""

    def __init__(self, subject_id: int, label):
        self.subject_id = subject_id
        self.n_studies = 2
        self._label = label
        self.emitted: list[str] = []

    def to_study_input(self, study_index: int = -1):
        study_id = ["early", "late"][study_index]
        self.emitted.append(study_id)
        return SimpleNamespace(study_id=f"{self.subject_id}-{study_id}",
                               ground_truth=self._label, image=[0.0] * 4,
                               image_shape=(2, 2), priors=None)


def test_builder_emits_one_row_per_subject_even_if_a_subject_repeats(monkeypatch, tmp_path):
    csv = tmp_path / "validate.csv"
    csv.write_text("subject_id\n")
    monkeypatch.setattr("aura.mimic.config.get_mimic_paths",
                        lambda: SimpleNamespace(validate_csv=csv, root=tmp_path))

    label = DIAGNOSES[0]
    a, a_again, b = _FakePatient(1, label), _FakePatient(1, label), _FakePatient(2, label)
    patients = [a, a_again, b] + [_FakePatient(100 + i, DIAGNOSES[i % len(DIAGNOSES)])
                                  for i in range(3 * len(DIAGNOSES))]
    monkeypatch.setattr("aura.mimic.patient.iter_patients", lambda split, limit=None: iter(patients))

    seen_ids: list[str] = []
    vision = SimpleNamespace(analyze=lambda sid, img: seen_ids.append(sid) or sid)
    monkeypatch.setattr(ds, "encode", lambda vr, priors: np.zeros(8))

    X, y = ds.build_real_evidence_dataset(n=10_000, vision=vision)

    subjects = [sid.split("-")[0] for sid in seen_ids]
    assert len(subjects) == len(set(subjects)), "a subject contributed two rows"
    assert subjects.count("1") == 1
    assert a_again.emitted == [], "the repeated subject must be skipped before any study is read"
    assert all(sid.endswith("-late") for sid in seen_ids), "only the later study may survive"
    assert len(X) == len(seen_ids)
