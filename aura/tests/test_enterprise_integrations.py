"""Tests for Enterprise Integrations: DICOM SCP tenant filtering, FHIR exports, and HL7 messages."""
from unittest.mock import MagicMock
import pydicom
import pytest

from aura.schemas.clinical import Diagnosis, Finding
from aura.schemas.contracts import (
    CaseBundle,
    CaseState,
    DecisionAssurance,
    DecisionVerdict,
    FindingScore,
    Prediction,
    ReportDraft,
    SafetyAssessment,
    StructuredPriors,
    VisionResult,
)
from aura.services.enterprise.dicom_listener import DicomListener
from aura.services.enterprise.fhir import (
    export_fhir_diagnostic_report,
    export_fhir_observations,
)
from aura.services.enterprise.hl7 import export_hl7_oru_r01


def _create_sample_bundle(
    case_id: str = "CASE-INTEG-01",
    verdict: DecisionVerdict = DecisionVerdict.ACCEPT,
    top_p: float = 0.88,
) -> CaseBundle:
    da = DecisionAssurance(
        verdict=verdict,
        primary_diagnosis=Diagnosis.NORMAL.value,
        confidence=top_p,
        conformal_set=[Diagnosis.NORMAL.value],
        conformal_coverage=0.90,
        ood_zscore=0.25,
        is_ood=False,
        epistemic_uncertainty=0.04,
        limiting_factor="none" if verdict == DecisionVerdict.ACCEPT else "confidence",
        reasons=["Assurance criteria fully satisfied."],
        mitigations=["routine_clinical_signoff"],
        requires_human_review=verdict != DecisionVerdict.ACCEPT,
    )
    safety = SafetyAssessment(
        study_id="STUDY-100",
        predictions=[
            Prediction(diagnosis=Diagnosis.NORMAL, probability=top_p),
            Prediction(diagnosis=Diagnosis.PNEUMONIA, probability=round(1.0 - top_p, 4)),
        ],
        top=Diagnosis.NORMAL,
        top_probability=top_p,
        conformal_set=[Diagnosis.NORMAL],
        conformal_coverage=0.90,
    )
    vision = VisionResult(
        study_id="STUDY-100",
        findings=[
            FindingScore(finding=Finding.OPACITY, probability=0.12),
        ],
        embedding=[0.1] * 8,
        model_version="vision-cxr-densenet121",
    )
    report = ReportDraft(
        study_id="STUDY-100",
        findings_text="Clear lung fields bilaterally.",
        impression_text="No acute cardiopulmonary disease.",
        recommendation_text="Routine clinical followup.",
    )
    return CaseBundle(
        case_id=case_id,
        study_id="STUDY-100",
        patient_id="PAT-999",
        state=CaseState.READY,
        priors=StructuredPriors(age_band="40-65", sex="F"),
        safety=safety,
        vision=vision,
        report=report,
        decision_assurance=da,
    )


class TestFHIRIntegration:
    def test_fhir_diagnostic_report_export(self):
        bundle = _create_sample_bundle(verdict=DecisionVerdict.ACCEPT)
        fhir_doc = export_fhir_diagnostic_report(bundle, tenant_id="saint_jude_hospital")

        assert fhir_doc["resourceType"] == "DiagnosticReport"
        assert fhir_doc["id"] == f"report-{bundle.case_id}"
        assert fhir_doc["meta"]["source"] == "urn:aura:tenant:saint_jude_hospital"

        # Check provenance hash tag
        tags = fhir_doc["meta"]["tag"]
        assert any(t["system"] == "https://aura.healthcare/provenance-hash" for t in tags)
        p_hash = next(t["code"] for t in tags if t["system"] == "https://aura.healthcare/provenance-hash")
        assert len(p_hash) == 64  # SHA-256 hex string

        # Check conclusion contains verdict
        assert "[ACCEPT]" in fhir_doc["conclusion"]

        # Check Decision Assurance Extension
        extensions = fhir_doc.get("extension", [])
        assert len(extensions) >= 1
        da_ext = next(
            e for e in extensions
            if e["url"] == "https://aura.healthcare/fhir/StructureDefinition/decision-assurance"
        )
        fields = {item["url"]: item for item in da_ext["extension"]}
        assert fields["verdict"]["valueString"] == "ACCEPT"
        assert fields["conformalCoverage"]["valueDecimal"] == 0.90
        assert fields["confidence"]["valueDecimal"] == 0.88
        assert not fields["requiresHumanReview"]["valueBoolean"]

    def test_fhir_observations_export(self):
        bundle = _create_sample_bundle()
        observations = export_fhir_observations(bundle)
        assert len(observations) == 1
        obs = observations[0]
        assert obs["resourceType"] == "Observation"
        assert obs["code"]["coding"][0]["code"] == "finding-opacity"
        assert obs["valueQuantity"]["value"] == 0.12


class TestHL7Integration:
    def test_hl7_oru_r01_export(self):
        bundle = _create_sample_bundle(
            verdict=DecisionVerdict.REVIEW_REQUIRED,
            top_p=0.55,
        )
        hl7_msg = export_hl7_oru_r01(bundle, tenant_id="metro_health")

        # Verify MSH header has tenant identity
        lines = hl7_msg.strip().split("\r")
        msh = lines[0]
        assert msh.startswith("MSH|^~\\&|AURA^metro_health|")

        # Verify OBX lines
        obx_lines = [line for line in lines if line.startswith("OBX|")]
        assert len(obx_lines) >= 4

        # Check Decision Verdict segment
        verdict_line = next(line for line in obx_lines if "DECISION_VERDICT" in line)
        assert "REVIEW_REQUIRED" in verdict_line
        assert "|A|" in verdict_line  # Flagged Abnormal for review

        # Check Provenance Hash segment
        hash_line = next(line for line in obx_lines if "PROVENANCE_HASH" in line)
        parts = hash_line.split("|")
        provenance_hex = parts[5]
        assert len(provenance_hex) == 64

        # Check Conformal Set segment
        conformal_line = next(line for line in obx_lines if "CONFORMAL_SET" in line)
        assert "Coverage 90%" in conformal_line
        assert Diagnosis.NORMAL.value in conformal_line


class TestDicomTenantResolution:
    def test_resolve_tenant_from_dataset(self):
        ds = pydicom.dataset.Dataset()
        ds.InstitutionName = "Saint Jude Childrens Research"
        tenant = DicomListener._resolve_tenant_id(None, ds)
        assert tenant == "saint_jude_childrens_research"

    def test_resolve_tenant_from_station_name(self):
        ds = pydicom.dataset.Dataset()
        ds.StationName = "MRI_SCANNER_BLDG4"
        tenant = DicomListener._resolve_tenant_id(None, ds)
        assert tenant == "mri_scanner_bldg4"

    def test_resolve_tenant_from_association(self):
        event = MagicMock()
        event.assoc.requestor.ae_title = b"PACSHOSPITALB   "
        tenant = DicomListener._resolve_tenant_id(event, ds=None)
        assert tenant == "pacshospitalb"

    def test_resolve_tenant_fallback_default(self):
        event = MagicMock()
        event.assoc = None
        ds = pydicom.dataset.Dataset()
        tenant = DicomListener._resolve_tenant_id(event, ds)
        assert tenant == "default"

    def test_handle_c_find_tenant_isolation(self):
        mock_dispatch = MagicMock()
        mock_store = MagicMock()
        listener = DicomListener(mock_dispatch, mock_store)

        # Mock list_cases to return only hospital_a cases when queried for hospital_a
        def fake_list_cases(tenant_id=None, **kwargs):
            if tenant_id == "hospital_a":
                return [
                    {"case_id": "CASE-A1", "study_id": "ST-A1", "patient_id": "P-A1", "created_at": "2026-09-12T10:00:00"},
                    {"case_id": "CASE-A2", "study_id": "ST-A2", "patient_id": "P-A2", "created_at": "2026-09-12T10:05:00"},
                ]
            elif tenant_id == "hospital_b":
                return [
                    {"case_id": "CASE-B1", "study_id": "ST-B1", "patient_id": "P-B1", "created_at": "2026-09-12T11:00:00"},
                ]
            return []

        mock_store.list_cases.side_effect = fake_list_cases

        # Create C-FIND event with InstitutionName = hospital_a
        event = MagicMock()
        event.is_cancelled = False
        ds = pydicom.dataset.Dataset()
        ds.InstitutionName = "hospital_a"
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID = ""
        ds.StudyInstanceUID = ""
        ds.PatientName = ""
        ds.AccessionNumber = ""
        event.identifier = ds

        results = list(listener.handle_c_find(event))
        # Status results should have 2 matching records (0xFF00) plus final 0x0000
        statuses = [st for st, _ in results]
        assert statuses == [0xFF00, 0xFF00, 0x0000]

        returned_patient_ids = [res.PatientID for _, res in results if res is not None]
        assert returned_patient_ids == ["P-A1", "P-A2"]
        # hospital_b patient should never appear
        assert "P-B1" not in returned_patient_ids
