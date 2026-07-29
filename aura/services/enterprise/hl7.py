from __future__ import annotations
from datetime import datetime, timezone
from schemas.contracts import CaseBundle
from schemas.clinical import DIAGNOSIS_LABELS

def export_hl7_oru_r01(b: CaseBundle) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    top_dx = b.safety.top.value if (b.safety and b.safety.top) else "unknown"
    top_dx_label = DIAGNOSIS_LABELS.get(b.safety.top, top_dx) if (b.safety and b.safety.top) else top_dx
    
    # MSH (Message Header)
    msh = f"MSH|^~\\&|AURA|HOSPITAL|EPIC|HOSPITAL|{now_str}||ORU^R01^ORU_R01|MSG{b.case_id}|P|2.5"
    
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
