from __future__ import annotations

from aura.schemas.contracts import CaseBundle


def build_grounding_prompt(bundle: CaseBundle) -> str:
    sections: list[str] = []

    sections.append("You are a local clinical copilot assisting a physician.")
    sections.append(
        "Analyze the question using ONLY the provided case facts below."
    )
    sections.append(
        "If the question asks about something not present in the facts, "
        "state clearly: 'This information is not present in the scan's evidence graph.'"
    )
    sections.append(
        "Never hallucinate findings, lesions, or symptoms that are not "
        "explicitly present in the data."
    )
    sections.append(
        "Cite your sources inline, referring to specific agents "
        "(e.g., '[Radiologist Agent]') or findings."
    )

    sections.append("")

    study_id = bundle.study_id or "unknown"
    modality = "MRI" if study_id.startswith("STU-MR") else "Chest X-ray"
    sections.append(f"=== Study Information ===")
    sections.append(f"Study ID: {study_id}")
    sections.append(f"Modality: {modality}")

    sections.append("")
    sections.append("=== Patient Priors ===")
    p = bundle.priors
    sections.append(f"  Age band: {p.age_band}")
    sections.append(f"  Sex: {p.sex}")
    sections.append(f"  Smoker: {p.smoker}")
    sections.append(f"  Fever: {p.fever}")
    sections.append(f"  Prior cancer: {p.prior_cancer}")
    sections.append(f"  Immunocompromised: {p.immunocompromised}")

    ctx = bundle.multimodal
    if ctx is not None:
        sections.append("")
        sections.append("=== Clinical Context ===")
        sections.append(f"  Labs: WBC={ctx.labs.wbc}, BNP={ctx.labs.bnp}, "
                        f"CRP={ctx.labs.crp}, Procalcitonin={ctx.labs.procalcitonin}, "
                        f"Troponin={ctx.labs.troponin}, D-dimer={ctx.labs.d_dimer}, "
                        f"SpO2={ctx.labs.spo2}")
        sx = ctx.symptoms
        active_sx = []
        if sx.dyspnea: active_sx.append("dyspnea")
        if sx.productive_cough: active_sx.append("productive cough")
        if sx.fever: active_sx.append("fever")
        if sx.pleuritic_chest_pain: active_sx.append("pleuritic chest pain")
        if sx.hemoptysis: active_sx.append("hemoptysis")
        if sx.orthopnea: active_sx.append("orthopnea")
        if sx.acute_onset: active_sx.append("acute onset")
        sections.append(f"  Symptoms: {', '.join(active_sx) if active_sx else 'none reported'}")
        hx = ctx.history
        sections.append(f"  History: COPD={hx.copd}, HF={hx.heart_failure}, "
                        f"Prior cancer={hx.prior_cancer}, Smoking pack-years={hx.smoking_pack_years}")

    if bundle.vision is not None and bundle.vision.findings:
        sections.append("")
        sections.append("=== Vision Findings (from scan) ===")
        for f in bundle.vision.findings:
            if f.probability >= 0.1:
                sections.append(f"  - {f.finding.value}: {f.probability:.2f}")

    consensus = bundle.consensus_result
    if consensus is not None:
        posterior = consensus.get("posterior", {})
        if posterior:
            sections.append("")
            sections.append("=== Consensus Diagnosis ===")
            sorted_dx = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
            for dx, prob in sorted_dx:
                if prob >= 0.05:
                    sections.append(f"  - {dx}: {prob:.2%}")

        entropy = consensus.get("consensus_entropy", 0.0)
        sections.append(f"  Consensus entropy: {entropy:.3f}")

        verdicts = consensus.get("verdicts", [])
        if verdicts:
            sections.append("")
            sections.append("=== Agent Panel Verdicts ===")
            for v in verdicts:
                agent_name = v.get("agent_name", "unknown").replace("_", " ").title()
                conf = v.get("confidence", 0.0)
                sections.append(f"  [{agent_name}] confidence={conf:.2f}")
                agent_findings = v.get("findings", {})
                if agent_findings:
                    top_findings = sorted(agent_findings.items(),
                                          key=lambda x: x[1], reverse=True)[:3]
                    for dx, p in top_findings:
                        sections.append(f"    - {dx}: {p:.2%}")
                args = v.get("arguments", [])
                if args:
                    for arg in args:
                        sections.append(f"    - {arg}")
                refs = v.get("guideline_references", [])
                if refs:
                    sections.append(f"    Guidelines: {', '.join(refs)}")

    panel_discussion = (consensus or {}).get("panel_discussion", "")
    if panel_discussion:
        sections.append("")
        sections.append("=== Panel Discussion ===")
        sections.append(panel_discussion)

    return "\n".join(sections)
