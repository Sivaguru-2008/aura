from __future__ import annotations
from datetime import datetime, timezone
from aura.schemas.contracts import CaseBundle
from aura.schemas.clinical import DIAGNOSIS_LABELS

def export_hl7_oru_r01(b: CaseBundle, tenant_id: str | None = None) -> str:
    from aura.gateway.storage import compute_provenance_hash
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    p_hash = compute_provenance_hash(b)
    tenant = tenant_id or "default"
    top_dx = b.safety.top.value if (b.safety and b.safety.top) else "unknown"
    top_dx_label = DIAGNOSIS_LABELS.get(b.safety.top, top_dx) if (b.safety and b.safety.top) else top_dx
    
    # MSH (Message Header)
    msh = f"MSH|^~\\&|AURA^{tenant}|HOSPITAL|EPIC|HOSPITAL|{now_str}||ORU^R01^ORU_R01|MSG{b.case_id}|P|2.5"
    
    # PID (Patient Identification)
    sex_char = "U"
    if b.priors and b.priors.sex:
        sex_char = b.priors.sex[0].upper() if b.priors.sex else "U"
    pid = f"PID|1||PAT-{b.case_id}||Patient^Demo|||{sex_char}|||||||||||"
    
    # OBR (Observation Request)
    obr = f"OBR|1||{b.study_id}|11524-6^Radiology Study Report^LN|||{now_str}|||||||||||||||||F"
    
    # OBX (Observation Results)
    obx_segments = []
    idx = 1
    
    # Decision Assurance Verdict (if available)
    da = getattr(b, "decision_assurance", None)
    if da is not None:
        flag = "A" if da.verdict.value != "ACCEPT" else "N"
        obx_segments.append(
            f"OBX|{idx}|TX|DECISION_VERDICT^Decision Verdict^AURA|1|{da.verdict.value} (limiting: {da.limiting_factor})|||{flag}|||F"
        )
        idx += 1
        obx_segments.append(
            f"OBX|{idx}|TX|CONFORMAL_SET^Conformal Prediction Set (Coverage {da.conformal_coverage:.0%})^AURA|1|{', '.join(da.conformal_set)}|||N|||F"
        )
        idx += 1

    # Cryptographic Provenance Hash
    obx_segments.append(
        f"OBX|{idx}|ST|PROVENANCE_HASH^Tamper-Evident SHA256^AURA|1|{p_hash}|||N|||F"
    )
    idx += 1

    # Diagnostic Impression
    obx_segments.append(
        f"OBX|{idx}|TX|DX^Primary Diagnosis^AURA|1|{top_dx_label} (probability: {b.safety.top_probability:.4f})|||N|||F"
    )
    idx += 1
    
    # Individual findings
    if b.vision and b.vision.findings:
        for f in b.vision.findings:
            obx_segments.append(
                f"OBX|{idx}|NM|{f.finding.value}^Finding^AURA|1|{f.probability:.4f}|||{'A' if f.probability >= 0.5 else 'N'}|||F"
            )
            idx += 1
            
    # Textual findings
    if b.report and b.report.findings_text:
        findings_clean = b.report.findings_text.replace("\n", " ").replace("|", "\\|")
        obx_segments.append(
            f"OBX|{idx}|TX|FT^Findings Narrative^AURA|1|{findings_clean}|||N|||F"
        )
        idx += 1
        
    lines = [msh, pid, obr] + obx_segments
    return "\r".join(lines) + "\r"
