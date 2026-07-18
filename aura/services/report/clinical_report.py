"""Full structured clinical report renderer (Step 5).

The shipped ``ReportEngine`` produces a compact, grounded ``ReportDraft`` (findings /
impression / differential / confidence / recommendation). This module does **not**
change that schema or engine — it *reads a finished ``CaseBundle``* and assembles the
full clinician-facing document the brief asks for:

    Patient Summary · Vision Findings · Confidence · Calibration ·
    Differential Diagnosis · Evidence Used · Evidence Missing ·
    Recommended Tests · Risk Level · Clinical Impression · Limitations ·
    Model Version · Inference Time

It emits a structured dict (for JSON/consumers) plus text, markdown, and HTML
renderings. Everything is derived from data already present on the bundle, so it is
purely additive and cannot break any existing consumer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schemas.clinical import (
    DIAGNOSIS_LABELS,
    FINDING_LABELS,
    Diagnosis,
    Finding,
)
from schemas.contracts import CaseBundle
from services.fusion.evidence import EVIDENCE_CHANNELS

# Clinical acuity → risk band for the top diagnosis.
_RISK_BAND: dict[Diagnosis, str] = {
    Diagnosis.PNEUMOTHORAX: "HIGH",
    Diagnosis.MALIGNANCY: "HIGH",
    Diagnosis.HEART_FAILURE: "MODERATE",
    Diagnosis.PNEUMONIA: "MODERATE",
    Diagnosis.COPD: "LOW-MODERATE",
    Diagnosis.NORMAL: "LOW",
}

_EV_LABEL = {
    "labs.bnp": "BNP", "labs.wbc": "white-cell count", "labs.crp": "CRP",
    "labs.procalcitonin": "procalcitonin", "labs.spo2": "oxygen saturation",
    "history.smoking_pack_years": "smoking history", "history.prior_cancer": "prior malignancy",
    "history.copd": "COPD history", "history.immunosuppression": "immunosuppression",
    "symptoms.fever": "fever", "symptoms.orthopnea": "orthopnoea",
    "symptoms.hemoptysis": "haemoptysis", "symptoms.pleuritic_chest_pain": "pleuritic chest pain",
    "symptoms.acute_onset": "acute onset", "symptoms.dyspnea": "dyspnoea",
    "priors.age_band": "age", "prior_risk": "structured risk factors",
}


def _pretty(ids) -> str:
    seen, out = set(), []
    for e in ids:
        lbl = _EV_LABEL.get(e, str(e).split(".")[-1].replace("_", " "))
        if lbl not in seen:
            seen.add(lbl); out.append(lbl)
    return ", ".join(out)


def _patient_summary(bundle: CaseBundle) -> dict:
    p = bundle.priors
    parts = [f"age {p.age_band}", f"sex {p.sex}"]
    flags = []
    if p.smoker:
        flags.append("smoker")
    if p.prior_cancer:
        flags.append("prior malignancy")
    if p.fever:
        flags.append("febrile")
    if p.immunocompromised:
        flags.append("immunocompromised")
    mm = bundle.multimodal
    labs_present = {}
    symptoms = []
    history = []
    if mm is not None:
        labs = mm.labs
        for name in ("wbc", "crp", "procalcitonin", "bnp", "troponin", "d_dimer", "spo2", "neutrophil_pct"):
            v = getattr(labs, name, None)
            if v is not None:
                labs_present[name] = v
        for name, on in vars(mm.symptoms).items():
            if on:
                symptoms.append(name)
        for name, v in vars(mm.history).items():
            if v:
                history.append(f"{name}={v}" if not isinstance(v, bool) else name)
    return {
        "demographics": ", ".join(parts),
        "risk_flags": flags,
        "labs": labs_present,
        "symptoms": symptoms,
        "history": history,
        "modality": "CXR",
        "study_id": bundle.study_id,
    }


def _vision_findings(bundle: CaseBundle) -> list[dict]:
    if bundle.vision is None:
        return []
    out = []
    for fs in bundle.vision.findings:
        out.append({
            "finding": FINDING_LABELS.get(fs.finding, fs.finding.value),
            "key": fs.finding.value,
            "probability": round(float(fs.probability), 4),
            "present": bool(fs.probability >= 0.5),
            "region": fs.region,
        })
    return out


def _confidence(bundle: CaseBundle) -> dict:
    s = bundle.safety
    if s is None:
        return {}
    top_pred = next((p for p in s.predictions if p.diagnosis == s.top), None)
    return {
        "top_diagnosis": DIAGNOSIS_LABELS.get(s.top, s.top.value),
        "top_probability": round(float(s.top_probability), 4),
        "ci_low": round(float(top_pred.ci_low), 4) if top_pred else None,
        "ci_high": round(float(top_pred.ci_high), 4) if top_pred else None,
        "epistemic_uncertainty": round(float(s.epistemic_uncertainty), 4),
        "aleatoric_uncertainty": round(float(s.aleatoric_uncertainty), 4),
        "predictive_entropy": round(float(s.predictive_entropy), 4),
        "uncertainty_method": s.uncertainty_method,
        "abstained": bool(s.abstained),
        "abstention_reason": s.abstention_reason.value,
    }


def _calibration(bundle: CaseBundle, calibration=None) -> dict:
    s = bundle.safety
    out = {
        "conformal_set": [DIAGNOSIS_LABELS.get(d, d.value) for d in (s.conformal_set if s else [])],
        "conformal_coverage": round(float(s.conformal_coverage), 4) if s else None,
        "conformal_method": s.conformal_method if s else None,
        "out_of_distribution": bool(s.is_ood) if s else None,
        "ood_energy_z": round(float(s.ood_energy), 4) if s else None,
    }
    if calibration is not None:
        out["temperature"] = round(float(getattr(calibration, "temperature", 1.0)), 4)
        out["reported_ece"] = round(float(getattr(calibration, "ece", 0.0)), 4)
    return out


def _differential(bundle: CaseBundle) -> list[dict]:
    r = bundle.reasoning
    if r is None or not r.differential:
        # Fall back to safety predictions.
        s = bundle.safety
        if s is None:
            return []
        return [{"diagnosis": DIAGNOSIS_LABELS.get(p.diagnosis, p.diagnosis.value),
                 "probability": round(float(p.probability), 4),
                 "supporting": [], "opposing": []} for p in s.predictions[:4]]
    out = []
    for item in r.differential:
        out.append({
            "diagnosis": DIAGNOSIS_LABELS.get(item.diagnosis, item.diagnosis.value),
            "probability": round(float(item.probability), 4),
            "supporting": _pretty(item.supporting).split(", ") if item.supporting else [],
            "opposing": _pretty(item.opposing).split(", ") if item.opposing else [],
        })
    return out


def _evidence_used(bundle: CaseBundle) -> list[str]:
    used: list[str] = []
    # Positive imaging findings.
    for fs in (bundle.vision.findings if bundle.vision else []):
        if fs.probability >= 0.5:
            used.append(f"{FINDING_LABELS.get(fs.finding, fs.finding.value)} (p={fs.probability:.2f})")
    # Reasoning-step evidence (labs/symptoms/history that fired).
    for st in (bundle.reasoning.steps if bundle.reasoning else []):
        for e in st.evidence:
            lbl = _EV_LABEL.get(e, str(e).split(".")[-1].replace("_", " "))
            if lbl not in used:
                used.append(lbl)
    return list(dict.fromkeys(used))


def _evidence_missing(bundle: CaseBundle) -> list[str]:
    missing: list[str] = []
    # Evidence flagged absent by the evidence graph.
    from schemas.contracts import EvidenceKind

    for ev in bundle.evidence:
        if ev.kind == EvidenceKind.ABSENT_EVIDENCE and 0.08 < ev.value < 0.5:
            missing.append(f"{ev.name} indeterminate (p={ev.value:.2f})")
    # Relevant labs not resulted.
    mm = bundle.multimodal
    if mm is not None:
        for name, why in (("bnp", "cardiac cause"), ("procalcitonin", "bacterial infection"),
                          ("d_dimer", "thromboembolism"), ("troponin", "cardiac injury")):
            if getattr(mm.labs, name, None) is None:
                missing.append(f"{name.upper()} not resulted ({why})")
    # Ambiguity: a large conformal set means competing diagnoses remain.
    s = bundle.safety
    if s is not None and len(s.conformal_set) > 1:
        missing.append(
            f"{len(s.conformal_set)} diagnoses remain in the {int(s.conformal_coverage*100)}% "
            "confidence set — discriminating evidence would narrow it")
    return list(dict.fromkeys(missing))


def _recommended_tests(bundle: CaseBundle) -> list[dict]:
    return [{
        "action": r.action,
        "test": r.display,
        "expected_info_gain_bits": round(float(r.expected_info_gain), 4),
        "cost": r.cost_tier,
        "risk": r.risk_tier,
        "utility": round(float(r.utility), 4),
        "rationale": r.rationale,
    } for r in bundle.recommendations]


def _risk_level(bundle: CaseBundle) -> dict:
    s = bundle.safety
    if s is None:
        return {"level": "UNKNOWN", "rationale": "no safety assessment"}
    if s.abstained:
        return {"level": "INDETERMINATE",
                "rationale": f"model abstained ({s.abstention_reason.value}); human review required"}
    band = _RISK_BAND.get(s.top, "MODERATE")
    conf = s.top_probability
    note = f"{DIAGNOSIS_LABELS.get(s.top, s.top.value)} at {conf:.0%} calibrated confidence"
    if s.is_ood:
        band = "REVIEW"
        note += "; input flagged out-of-distribution"
    return {"level": band, "rationale": note}


def _limitations(bundle: CaseBundle) -> list[str]:
    lim = [
        "Automated decision support only — not a substitute for a radiologist's read.",
        "DenseNet-121 trained on MIMIC-CXR; performance may degrade on out-of-distribution "
        "equipment, projections, or populations.",
        "Ground-truth training labels were derived from free-text reports by a rule-based "
        "labeler and inherit its error modes.",
        "Findings localize regions of interest; they are observations, not tissue diagnoses.",
    ]
    s = bundle.safety
    if s is not None:
        if s.is_ood:
            lim.append("This image was flagged out-of-distribution; interpret with heightened caution.")
        if s.abstained:
            lim.append(f"The system abstained from a committed impression ({s.abstention_reason.value}).")
    return lim


def build_clinical_report(bundle: CaseBundle, inference_time_s: Optional[float] = None,
                          calibration=None) -> dict:
    """Assemble the full structured clinical report from a finished CaseBundle."""
    versions = {}
    if bundle.vision is not None:
        versions["vision"] = bundle.vision.model_version
    if bundle.fusion is not None:
        versions["fusion"] = f"{bundle.fusion.backend}:{bundle.fusion.model_version}"
    if bundle.safety is not None:
        versions["safety"] = bundle.safety.model_version
    if bundle.reasoning is not None:
        versions["reasoning"] = bundle.reasoning.model_version
    if bundle.report is not None:
        versions["report"] = bundle.report.generator

    return {
        "study_id": bundle.study_id,
        "case_id": bundle.case_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient_summary": _patient_summary(bundle),
        "vision_findings": _vision_findings(bundle),
        "confidence": _confidence(bundle),
        "calibration": _calibration(bundle, calibration),
        "differential_diagnosis": _differential(bundle),
        "evidence_used": _evidence_used(bundle),
        "evidence_missing": _evidence_missing(bundle),
        "recommended_tests": _recommended_tests(bundle),
        "risk_level": _risk_level(bundle),
        "clinical_impression": (bundle.report.impression_text if bundle.report else ""),
        "findings_text": (bundle.report.findings_text if bundle.report else ""),
        "limitations": _limitations(bundle),
        "model_version": versions,
        "inference_time_s": round(float(inference_time_s), 4) if inference_time_s is not None else None,
        "ground_truth": bundle.ground_truth.value if bundle.ground_truth else None,
    }


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append(f"# AURA Clinical Report — {rep['study_id']}")
    L.append(f"_Generated {rep['generated_at']}_")
    ps = rep["patient_summary"]
    L.append("\n## Patient Summary")
    L.append(f"- **Study:** {ps['study_id']} ({ps['modality']})")
    L.append(f"- **Demographics:** {ps['demographics']}")
    if ps["risk_flags"]:
        L.append(f"- **Risk flags:** {', '.join(ps['risk_flags'])}")
    if ps["labs"]:
        L.append(f"- **Labs:** " + ", ".join(f"{k}={v}" for k, v in ps["labs"].items()))
    if ps["symptoms"]:
        L.append(f"- **Symptoms:** {', '.join(ps['symptoms'])}")
    if ps["history"]:
        L.append(f"- **History:** {', '.join(ps['history'])}")

    L.append("\n## Vision Findings")
    if rep["vision_findings"]:
        L.append("| Finding | Probability | Present |")
        L.append("|---|---|---|")
        for f in rep["vision_findings"]:
            L.append(f"| {f['finding']} | {f['probability']:.3f} | {'✓' if f['present'] else ''} |")
    else:
        L.append("_No vision result._")

    c = rep["confidence"]
    L.append("\n## Confidence")
    if c:
        ci = f" (95% CI {c['ci_low']:.2f}–{c['ci_high']:.2f})" if c.get("ci_low") is not None else ""
        L.append(f"- **Top diagnosis:** {c['top_diagnosis']} — {c['top_probability']:.0%}{ci}")
        L.append(f"- **Uncertainty:** epistemic {c['epistemic_uncertainty']:.3f}, "
                 f"aleatoric {c['aleatoric_uncertainty']:.3f} ({c['uncertainty_method']})")

    cal = rep["calibration"]
    L.append("\n## Calibration")
    L.append(f"- **{cal.get('conformal_coverage', 0) and int(cal['conformal_coverage']*100)}% "
             f"{cal.get('conformal_method','')} conformal set:** {', '.join(cal['conformal_set'])}")
    if "temperature" in cal:
        L.append(f"- **Temperature:** {cal['temperature']} · **reported ECE:** {cal['reported_ece']}")
    if cal.get("out_of_distribution") is not None:
        L.append(f"- **Out-of-distribution:** {'YES' if cal['out_of_distribution'] else 'no'} "
                 f"(energy z={cal['ood_energy_z']})")

    L.append("\n## Differential Diagnosis")
    for d in rep["differential_diagnosis"]:
        seg = f"- **{d['diagnosis']}** — {d['probability']:.0%}"
        if d["supporting"]:
            seg += f"; supported by {', '.join(d['supporting'])}"
        if d["opposing"]:
            seg += f"; against: {', '.join(d['opposing'])}"
        L.append(seg)

    L.append("\n## Evidence Used")
    L.append(", ".join(rep["evidence_used"]) or "_none asserted_")
    L.append("\n## Evidence Missing")
    for m in rep["evidence_missing"]:
        L.append(f"- {m}")
    if not rep["evidence_missing"]:
        L.append("_none — available evidence is sufficient._")

    L.append("\n## Recommended Tests")
    if rep["recommended_tests"]:
        for r in rep["recommended_tests"]:
            L.append(f"- **{r['test']}** (cost {r['cost']}, risk {r['risk']}, "
                     f"utility {r['utility']}) — {r['rationale']}")
    else:
        L.append("_No additional testing indicated by value-of-information analysis._")

    rl = rep["risk_level"]
    L.append(f"\n## Risk Level: **{rl['level']}**")
    L.append(rl["rationale"])

    L.append("\n## Clinical Impression")
    L.append(rep["clinical_impression"] or "_withheld_")

    L.append("\n## Limitations")
    for lim in rep["limitations"]:
        L.append(f"- {lim}")

    L.append("\n## Provenance")
    for k, v in rep["model_version"].items():
        L.append(f"- **{k}:** {v}")
    if rep["inference_time_s"] is not None:
        L.append(f"- **Inference time:** {rep['inference_time_s']*1000:.0f} ms")
    if rep["ground_truth"]:
        L.append(f"- **Ground truth (eval only):** {rep['ground_truth']}")
    return "\n".join(L)


def render_text(rep: dict) -> str:
    """Compact plain-text rendering for the terminal (no markdown markup)."""
    md = render_markdown(rep)
    # Strip longer markers before their prefixes (## before #) so headings survive.
    for token in ("**", "### ", "## ", "# ", "|---|", "---", "| ", " |", "_"):
        md = md.replace(token, "")
    return md


def render_html(rep: dict) -> str:
    body = render_markdown(rep)
    # Minimal markdown → HTML (headings, bold, list items) for a standalone file.
    html_lines = []
    for line in body.splitlines():
        s = line.rstrip()
        if s.startswith("## "):
            html_lines.append(f"<h2>{s[3:]}</h2>")
        elif s.startswith("# "):
            html_lines.append(f"<h1>{s[2:]}</h1>")
        elif s.startswith("- "):
            html_lines.append(f"<li>{_md_inline(s[2:])}</li>")
        elif s.startswith("|"):
            html_lines.append(f"<div class='row'>{_md_inline(s.strip('|'))}</div>")
        elif s:
            html_lines.append(f"<p>{_md_inline(s)}</p>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AURA Clinical Report — {rep['study_id']}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0f14;color:#e6edf3;
  max-width:820px;margin:0 auto;padding:32px;line-height:1.5}}
 h1{{font-size:22px}} h2{{font-size:16px;color:#22d3ee;border-bottom:1px solid #223;padding-bottom:5px;margin-top:26px}}
 li{{margin:3px 0}} .row{{font-family:ui-monospace,monospace;font-size:12px;color:#9fb2c8}}
 code{{background:#111823;padding:1px 4px;border-radius:4px}}
</style></head><body>{''.join(html_lines)}</body></html>"""


def _md_inline(s: str) -> str:
    import re

    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s


def save_report(rep: dict, out_dir: str | Path, stem: str = "report") -> dict[str, Path]:
    """Write the report as .md, .html, and .json. Returns the written paths."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / f"{stem}.md",
        "html": out / f"{stem}.html",
        "json": out / f"{stem}.json",
    }
    paths["markdown"].write_text(render_markdown(rep), encoding="utf-8")
    paths["html"].write_text(render_html(rep), encoding="utf-8")
    paths["json"].write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return paths
