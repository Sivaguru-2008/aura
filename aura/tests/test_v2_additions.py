"""Unit and integration tests for the 5 implemented AURA V2 subsystems."""
from __future__ import annotations

import numpy as np
import pytest
import sqlite3
import pydicom

from aura.services.memory import MemoryEngine
from aura.services.enterprise.ehr import EHRClient
from aura.schemas.contracts import StudyInput, StructuredPriors, MultimodalContext, LabPanel, Symptoms, ClinicalHistory
from aura.schemas.clinical import Diagnosis, Finding, DIAGNOSES
from aura.services.reasoning import ClinicalReasoner
from aura.services.enterprise.dicom_listener import DicomListener
from aura.services.training.recalibrate import recalibrate_from_feedback
from aura.gateway.storage import Store
from aura.gateway.pipeline import Pipeline
from aura.common.config import get_settings


def test_persistent_memory_index(tmp_path):
    """Test that MemoryEngine persists and retrieves vectors correctly."""
    db_file = tmp_path / "test_memory.db"
    # SqliteVectorStore automatically creates the tables
    from aura.services.memory.vectorstore import SqliteVectorStore
    store = SqliteVectorStore(db_path=db_file)
    engine = MemoryEngine(store=store)

    emb1 = np.random.rand(1024)
    emb2 = np.random.rand(1024)

    engine.index("CASE-TEST-1", emb1, "pneumonia", modality="cxr", study_id="STU-1")
    engine.index("CASE-TEST-2", emb2, "heart_failure", modality="cxr", study_id="STU-2")

    # Verify we can find similar cases
    hits = engine.similar(emb1, k=1)
    assert len(hits) == 1
    assert hits[0]["case_id"] == "CASE-TEST-1"
    assert hits[0]["diagnosis"] == "pneumonia"
    assert hits[0]["similarity"] > 0.9  # self similarity is 1.0


def test_hospital_ehr_integration(tmp_path):
    """Test saving and retrieving clinical records via EHRClient and Pipeline integration."""
    db_file = tmp_path / "test_ehr.db"
    ehr = EHRClient(db_path=db_file)

    patient_id = "MIMIC-99999"
    labs = LabPanel(bnp=1200.0, wbc=14.5, procalcitonin=2.1, crp=80.0, spo2=91.0)
    symptoms = Symptoms(fever=True, orthopnea=True)
    history = ClinicalHistory(copd=True, smoking_pack_years=35.0)
    context = MultimodalContext(labs=labs, symptoms=symptoms, history=history)

    ehr.save_patient_context(patient_id, context)

    # Verify retrieval
    retrieved = ehr.get_patient_context(patient_id)
    assert retrieved is not None
    assert retrieved.labs.bnp == 1200.0
    assert retrieved.symptoms.fever is True
    assert retrieved.history.copd is True

    # Test Pipeline integration
    store = Store(db_file)
    pipeline = Pipeline(store=store)
    
    study = StudyInput(
        study_id="STU-99999",
        image=[0.5] * (64 * 64),
        image_shape=(64, 64),
        priors=StructuredPriors(age_band="65+", sex="M"),
        multimodal=None, # Trigger EHR lookup
        patient_id=patient_id,
    )
    
    # Check that pipeline.run triggers the ehr retrieval
    # Note: we mock pipeline.vision and pipeline.fusion to avoid running actual heavy inference
    class DummyVision:
        def analyze(self, study_id, img):
            from aura.schemas.contracts import VisionResult, FindingScore
            from aura.schemas.clinical import Finding
            return VisionResult(study_id=study_id, findings=[FindingScore(finding=Finding.CONSOLIDATION, probability=0.8)], embedding=[0.1]*1024, model_version="dummy")
        def score_findings(self, img):
            from aura.schemas.clinical import Finding
            return {Finding.CONSOLIDATION: 0.8}
    
    class DummyFusion:
        backend = "classical"
        class DummyModel:
            model_version = "dummy"
            def logits(self, x):
                return np.zeros(6)
        model = DummyModel()
        def fuse_vector(self, x, study_id):
            from aura.schemas.contracts import FusionResult
            return FusionResult(study_id=study_id, backend="classical", posterior={d: 1.0/6.0 for d in DIAGNOSES})
        def resolved_logits(self, x, fusion):
            return np.zeros(6)

    pipeline.vision = DummyVision()
    pipeline.fusion = DummyFusion()
    
    # We run pipeline.run which is async
    import asyncio
    bundle = asyncio.run(pipeline.run(study, "CASE-99999"))
    
    # Verify that study.multimodal was populated
    assert study.multimodal is not None
    assert study.multimodal.labs.bnp == 1200.0


@pytest.fixture
def quantum_reasoner_backend():
    """Select the quantum reasoner for the duration of a test."""
    settings = get_settings()
    orig = settings.reasoner_backend
    object.__setattr__(settings, "reasoner_backend", "quantum")
    yield
    object.__setattr__(settings, "reasoner_backend", orig)


def _reason_once(reasoner):
    priors = StructuredPriors(age_band="65+", sex="M")
    mm = MultimodalContext(
        labs=LabPanel(bnp=500.0, wbc=14.0),
        symptoms=Symptoms(fever=True),
        history=ClinicalHistory(),
    )
    findings = {Finding.CONSOLIDATION: 0.8, Finding.CARDIOMEGALY: 0.1}
    posterior = {d: 1.0 / 6.0 for d in DIAGNOSES}
    return reasoner.reason("STU-QBN-1", findings, posterior, priors, mm)


def test_qbn_does_not_run_untrained(quantum_reasoner_backend):
    """Selecting the quantum backend without a fitted artifact must change nothing.

    The constructor's default theta is six unfitted constants. Serving those would
    present hardcoded numbers as a quantum inference, so the reasoner declines.
    """
    from aura.services.reasoning.qbn import QuantumBayesianNetwork

    assert QuantumBayesianNetwork.load_trained() is None, (
        "a fitted reasoning_qbn.npz is present; this test asserts the untrained path"
    )

    trace, _ = _reason_once(ClinicalReasoner())

    assert trace.qbn_enabled is False
    assert not any(s.modality == "quantum" for s in trace.steps)
    # The guideline rules still ran and still carry the citations.
    assert len(trace.steps) >= 1
    assert trace.guideline_citations


def test_qbn_refines_without_replacing_the_rules(quantum_reasoner_backend, monkeypatch):
    """A trained QBN layers on top of the rules; it never displaces them."""
    from aura.services.reasoning import engine as reasoning_engine
    from aura.services.reasoning.qbn import QuantumBayesianNetwork

    classical, _ = _reason_once(ClinicalReasoner())
    assert classical.qbn_enabled is False
    classical_steps = [s.statement for s in classical.steps]
    assert classical_steps, "expected the guideline rules to fire for this context"

    trained = QuantumBayesianNetwork(theta=np.full(6, 0.8))
    monkeypatch.setattr(
        reasoning_engine.QuantumBayesianNetwork if hasattr(reasoning_engine, "QuantumBayesianNetwork")
        else QuantumBayesianNetwork,
        "load_trained",
        classmethod(lambda cls, path=None: trained),
    )

    trace, _ = _reason_once(ClinicalReasoner())

    assert trace.qbn_enabled is True
    quantum_steps = [s for s in trace.steps if s.modality == "quantum"]
    assert len(quantum_steps) == 1
    assert "Quantum Bayesian Network" in quantum_steps[0].statement
    # Every classical guideline step survives the refinement.
    for statement in classical_steps:
        assert statement in [s.statement for s in trace.steps]
    assert trace.guideline_citations
    assert abs(sum(trace.adjusted_posterior.values()) - 1.0) < 1e-3


def test_pacs_network_client(tmp_path):
    """Test that DicomListener can handle C-FIND requests."""
    db_file = tmp_path / "test_pacs.db"
    store = Store(db_file)
    
    # Populate a case to search for
    from aura.schemas.contracts import CaseBundle, VisionResult, FusionResult, SafetyAssessment
    bundle = CaseBundle(
        case_id="CASE-PACS-1",
        study_id="STU-PACS-1",
        state="ready",
        priors=StructuredPriors(),
        vision=VisionResult(study_id="STU-PACS-1", findings=[], embedding=[], model_version=""),
        fusion=FusionResult(study_id="STU-PACS-1", backend="classical", posterior={}),
        safety=SafetyAssessment(
            study_id="STU-PACS-1",
            predictions=[],
            conformal_set=[],
            top=Diagnosis.NORMAL,
            top_probability=0.9
        )
    )
    bundle.patient_id = "MIMIC-4242"
    store.save_case(bundle)

    class DummyDispatch:
        pass

    # Port 11115 for testing to avoid collisions
    listener = DicomListener(DummyDispatch(), store, port=11115)

    def _query(**keys):
        event = type("DummyEvent", (), {})()
        event.is_cancelled = False
        ds = pydicom.dataset.Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        for keyword, value in keys.items():
            setattr(ds, keyword, value)
        event.identifier = ds
        return list(listener.handle_c_find(event))

    # Wildcard match on the real patient id: one pending match + final success.
    responses = _query(PatientID="MIMIC-*", StudyDate="", PatientBirthDate="", PatientName="")
    assert len(responses) == 2
    assert responses[0][0] == 0xFF00
    assert responses[1][0] == 0x0000

    res = responses[0][1]
    assert res.PatientID == "MIMIC-4242"
    # Demographics AURA does not hold come back zero-length, never invented: an
    # SCU is entitled to treat a returned birth date as the patient's real one.
    assert res.PatientBirthDate == ""
    assert res.PatientName == ""
    # StudyDate is the stored creation date, not "today".
    assert res.StudyDate == "" or len(res.StudyDate) == 8

    # A non-matching patient id returns only the final success status.
    assert len(_query(PatientID="NOPE-*")) == 1

    # Universal (empty) matching returns the record.
    assert len(_query(PatientID="")) == 2

    # Attributes the SCU did not ask for are not fabricated into the response.
    minimal = _query(PatientID="MIMIC-4242")[0][1]
    assert "AccessionNumber" not in minimal


def test_nightly_recalibration(tmp_path, monkeypatch):
    """Test that nightly recalibration aggregates feedback and runs calibration."""
    db_file = tmp_path / "test_recal.db"
    store = Store(db_file)
    
    # Pre-populate feedback and corresponding cases
    from aura.schemas.contracts import CaseBundle, VisionResult, FusionResult, SafetyAssessment
    
    # Case 1: Incorrect -> Corrected to Pneumonia
    b1 = CaseBundle(
        case_id="CASE-FEED-1",
        study_id="STU-FEED-1",
        state="ready",
        priors=StructuredPriors(),
        vision=VisionResult(study_id="STU-FEED-1", findings=[], embedding=[], model_version=""),
        fusion=FusionResult(study_id="STU-FEED-1", backend="classical", posterior={}, evidence_vector=[0.1]*8),
        safety=SafetyAssessment(
            study_id="STU-FEED-1",
            predictions=[],
            conformal_set=[],
            top=Diagnosis.NORMAL,
            top_probability=0.9
        )
    )
    store.save_case(b1)
    store.add_feedback("CASE-FEED-1", "normal", "incorrect", "pneumonia")

    # Run recalibration from feedback
    # We mock _load_backend to avoid loading actual heavy ML files
    import aura.services.training.recalibrate as rc

    class DummyModel:
        model_version = "dummy"
        def logits(self, x):
            return np.ones(6)

    monkeypatch.setattr(rc, "_load_backend", lambda name: DummyModel())
    monkeypatch.setattr(rc, "calibrate", lambda backend, *a, **k: {"temperature": 1.0})

    # One sample is below the threshold: recalibration must not touch the served
    # safety envelope, because refitting adds no information but does move the
    # temperature and conformal threshold the abstention gate depends on.
    res = recalibrate_from_feedback(store)
    assert res["status"] == "skipped"
    assert res["feedback_samples_added"] == 1
    assert res["results"] == {}

    # Once enough clinician labels exist, it runs and reports what it changed.
    for i in range(2, rc.MIN_FEEDBACK_SAMPLES + 1):
        bn = b1.model_copy(update={"case_id": f"CASE-FEED-{i}", "study_id": f"STU-FEED-{i}"})
        store.save_case(bn)
        store.add_feedback(f"CASE-FEED-{i}", "normal", "incorrect", "pneumonia")

    res = recalibrate_from_feedback(store)
    assert res["status"] == "success"
    assert res["feedback_samples_added"] == rc.MIN_FEEDBACK_SAMPLES
    assert "classical" in res["results"]
