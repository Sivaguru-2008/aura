from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from aura.schemas.contracts import CaseBundle
from aura.schemas.clinical import DIAGNOSIS_LABELS

def export_fhir_diagnostic_report(b: CaseBundle, tenant_id: str | None = None) -> dict[str, Any]:
    from aura.gateway.storage import compute_provenance_hash
    issued_str = b.created_at.isoformat() if hasattr(b.created_at, "isoformat") else datetime.now(timezone.utc).isoformat()
    p_hash = compute_provenance_hash(b)
    
    top_dx = b.safety.top.value if (b.safety and b.safety.top) else "unknown"
    top_dx_label = DIAGNOSIS_LABELS.get(b.safety.top, top_dx) if (b.safety and b.safety.top) else top_dx
    
    results = []
    if b.vision and b.vision.findings:
        for i in range(len(b.vision.findings)):
            results.append({"reference": f"Observation/{b.case_id}-finding-{i}"})

    extensions = []
    da = getattr(b, "decision_assurance", None)
    if da is not None:
        extensions.append({
            "url": "https://aura.healthcare/fhir/StructureDefinition/decision-assurance",
            "extension": [
                {"url": "verdict", "valueString": da.verdict.value},
                {"url": "confidence", "valueDecimal": float(da.confidence)},
                {"url": "conformalSet", "valueString": ", ".join(da.conformal_set)},
                {"url": "conformalCoverage", "valueDecimal": float(da.conformal_coverage)},
                {"url": "oodZScore", "valueDecimal": float(da.ood_zscore)},
                {"url": "isOod", "valueBoolean": bool(da.is_ood)},
                {"url": "epistemicUncertainty", "valueDecimal": float(da.epistemic_uncertainty)},
                {"url": "requiresHumanReview", "valueBoolean": bool(da.requires_human_review)},
                {"url": "limitingFactor", "valueString": str(da.limiting_factor)},
            ]
        })

    conclusion_text = b.report.impression_text if b.report else "No impression text"
    if da is not None:
        conclusion_text = f"[{da.verdict.value}] {conclusion_text}"

    report_dict = {
        "resourceType": "DiagnosticReport",
        "id": f"report-{b.case_id}",
        "meta": {
            "source": f"urn:aura:tenant:{tenant_id or 'default'}",
            "tag": [
                {
                    "system": "https://aura.healthcare/provenance-hash",
                    "code": p_hash,
                    "display": "Tamper-Evident SHA-256 Provenance",
                }
            ],
        },
        "identifier": [{
            "use": "official",
            "system": "http://aura.hospital.org/cases",
            "value": b.case_id
        }],
        "status": "final" if b.state.value == "signed" or b.state == "signed" else "registered",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "11524-6",
                "display": "Radiology Study Report"
            }]
        },
        "subject": {
            "reference": f"Patient/PAT-{b.case_id}",
            "display": f"Demographics: Sex {b.priors.sex}, Age band {b.priors.age_band}"
        },
        "issued": issued_str,
        "performer": [{
            "display": "AURA AI Clinical Reasoning Assistant",
            "actor": {
                "reference": "Device/aura-copilot-v2"
            }
        }],
        "conclusion": conclusion_text,
        "conclusionCode": [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "38341003",
                "display": top_dx_label
            }]
        }],
        "result": results
    }
    if extensions:
        report_dict["extension"] = extensions
    return report_dict

def export_fhir_observations(b: CaseBundle) -> list[dict[str, Any]]:
    obs_list = []
    if not b.vision:
        return obs_list
    
    issued_str = b.created_at.isoformat() if hasattr(b.created_at, "isoformat") else datetime.now(timezone.utc).isoformat()
    
    for i, f in enumerate(b.vision.findings):
        obs_list.append({
            "resourceType": "Observation",
            "id": f"{b.case_id}-finding-{i}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "imaging",
                    "display": "Imaging"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": f"finding-{f.finding.value}",
                    "display": f.finding.value
                }]
            },
            "subject": {
                "reference": f"Patient/PAT-{b.case_id}"
            },
            "effectiveDateTime": issued_str,
            "valueQuantity": {
                "value": round(float(f.probability), 4),
                "unit": "probability",
                "system": "http://unitsofmeasure.org",
                "code": "1"
            },
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "A" if f.probability >= 0.5 else "N",
                    "display": "Abnormal" if f.probability >= 0.5 else "Normal"
                }]
            }]
        })
    return obs_list
