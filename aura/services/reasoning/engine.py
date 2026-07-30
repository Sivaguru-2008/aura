"""ClinicalReasoner — combine imaging, metadata, labs, symptoms, history, guidelines.

Why this exists
---------------
Nothing in the shipped pipeline reasons *across modalities*. Fusion sees eight
imaging-derived channels plus a scalar prior-risk; labs, symptoms, and history
never enter, and no clinical guideline is applied. This engine adds that layer.

It treats the fusion posterior as the imaging prior and performs explicit Bayesian
updates with **likelihood ratios drawn from clinical guidelines** — BNP for heart
failure, procalcitonin/leukocytosis for pneumonia, smoking + nodule for malignancy,
and so on. Every update is a first-class ``ReasoningStep``: a statement, the exact
evidence it used, the log-LR it applied per diagnosis, and a guideline citation.
The result is an adjusted differential where each diagnosis carries its own
supporting and opposing evidence — which is what the report then narrates.

This is deliberately a transparent rule/LR engine, not a black box: in a clinical
setting the provenance of every nudge matters more than a fractional AUROC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from aura.common.mathx import softmax
from aura.schemas.clinical import CHEST_DIAGNOSES, DIAGNOSES, Diagnosis, Finding
from aura.schemas.contracts import (
    DifferentialItem,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceKind,
    EvidenceNode,
    MultimodalContext,
    RelationType,
    ReasoningStep,
    ReasoningTrace,
    StructuredPriors,
)

D = Diagnosis


@dataclass
class Rule:
    name: str
    modality: str                                   # imaging|labs|symptoms|history
    guideline: str
    fire: Callable[["Ctx"], Optional[ReasoningStep]]


@dataclass
class Ctx:
    findings: dict[Finding, float]
    priors: StructuredPriors
    mm: MultimodalContext


def _step(statement, evidence, effect, guideline, modality) -> ReasoningStep:
    return ReasoningStep(statement=statement, evidence=evidence,
                         effect={d: float(v) for d, v in effect.items()},
                         guideline=guideline, modality=modality)


# --------------------------------------------------------------------------- #
# Guideline knowledge base. Each rule returns a ReasoningStep or None.
# Log-LR magnitudes are intentionally modest (evidence nudges, not overrides).
# --------------------------------------------------------------------------- #
def _r_bnp(c: Ctx):
    bnp = c.mm.labs.bnp
    if bnp is None:
        return None
    if bnp >= 400:
        return _step(f"BNP {bnp:.0f} pg/mL is markedly elevated, supporting a cardiac "
                     "cause of dyspnoea/effusion.", ["labs.bnp"],
                     {D.HEART_FAILURE: 1.1, D.PNEUMONIA: -0.3, D.NORMAL: -0.5},
                     "ACC/AHA HF guideline (BNP >400 pg/mL)", "labs")
    if bnp < 100:
        return _step(f"BNP {bnp:.0f} pg/mL is low, arguing against decompensated heart "
                     "failure.", ["labs.bnp"],
                     {D.HEART_FAILURE: -1.0}, "ACC/AHA HF guideline (BNP <100 pg/mL)", "labs")
    return None


def _r_infection(c: Ctx):
    labs, sym = c.mm.labs, c.mm.symptoms
    ev, score = [], 0.0
    if labs.wbc is not None and labs.wbc > 11:
        ev.append("labs.wbc"); score += 0.5
    if labs.procalcitonin is not None and labs.procalcitonin >= 0.5:
        ev.append("labs.procalcitonin"); score += 0.7
    if labs.crp is not None and labs.crp > 50:
        ev.append("labs.crp"); score += 0.3
    if sym.fever or c.priors.fever:
        ev.append("symptoms.fever"); score += 0.4
    consol = c.findings.get(Finding.CONSOLIDATION, 0.0)
    if score >= 0.8 and consol >= 0.4:
        ev.append("imaging.consolidation")
        return _step("Systemic inflammatory markers with airspace consolidation are "
                     "consistent with bacterial pneumonia.", ev,
                     {D.PNEUMONIA: min(1.6, score + 0.4), D.NORMAL: -0.6},
                     "IDSA/ATS community-acquired pneumonia guideline", "labs")
    return None


def _r_malignancy(c: Ctx):
    hist = c.mm.history
    nodule = c.findings.get(Finding.NODULE, 0.0)
    if nodule < 0.4:
        return None
    ev, lr = ["imaging.nodule"], 0.3
    py = hist.smoking_pack_years or (20.0 if c.priors.smoker else 0.0)
    if py >= 20:
        ev.append("history.smoking_pack_years"); lr += 0.7
    if hist.prior_cancer or c.priors.prior_cancer:
        ev.append("history.prior_cancer"); lr += 0.8
    if c.mm.symptoms.hemoptysis:
        ev.append("symptoms.hemoptysis"); lr += 0.5
    if c.priors.age_band == "65+":
        ev.append("priors.age_band"); lr += 0.3
    return _step("Pulmonary nodule with malignancy risk factors (smoking burden / "
                 "prior malignancy / age) raises suspicion for neoplasm.", ev,
                 {D.MALIGNANCY: min(1.8, lr), D.NORMAL: -0.4},
                 "Fleischner Society / Lung-RADS nodule guidance", "history")


def _r_copd(c: Ctx):
    hyper = c.findings.get(Finding.HYPERINFLATION, 0.0)
    if c.mm.history.copd and hyper >= 0.4:
        return _step("Known COPD with radiographic hyperinflation supports an obstructive "
                     "process.", ["history.copd", "imaging.hyperinflation"],
                     {D.COPD: 1.2, D.NORMAL: -0.3}, "GOLD COPD report", "history")
    return None


def _r_chf_signs(c: Ctx):
    card = c.findings.get(Finding.CARDIOMEGALY, 0.0)
    eff = c.findings.get(Finding.EFFUSION, 0.0)
    if c.mm.symptoms.orthopnea and (card >= 0.4 or eff >= 0.4):
        return _step("Orthopnoea with cardiomegaly and/or pleural effusion supports "
                     "congestive heart failure.",
                     ["symptoms.orthopnea", "imaging.cardiomegaly", "imaging.pleural_effusion"],
                     {D.HEART_FAILURE: 0.9}, "ACC/AHA HF guideline (clinical signs)", "symptoms")
    return None


def _r_pneumothorax(c: Ctx):
    ptx = c.findings.get(Finding.PNEUMOTHORAX, 0.0)
    if ptx >= 0.4 and (c.mm.symptoms.pleuritic_chest_pain or c.mm.symptoms.acute_onset):
        return _step("Acute pleuritic chest pain with a visible pleural edge is "
                     "consistent with pneumothorax; assess for tension.",
                     ["symptoms.pleuritic_chest_pain", "imaging.pneumothorax"],
                     {D.PNEUMOTHORAX: 1.3, D.NORMAL: -0.5}, "BTS pleural disease guideline", "symptoms")
    return None


def _r_immunosupp(c: Ctx):
    if c.mm.history.immunosuppression or c.priors.immunocompromised:
        return _step("Immunosuppression widens the infectious differential and lowers the "
                     "threshold to treat.", ["history.immunosuppression"],
                     {D.PNEUMONIA: 0.5, D.NORMAL: -0.4}, "IDSA immunocompromised host guidance",
                     "history")
    return None


def _r_hypoxia(c: Ctx):
    spo2 = c.mm.labs.spo2
    if spo2 is not None and spo2 < 92:
        return _step(f"Hypoxaemia (SpO2 {spo2:.0f}%) indicates a clinically significant "
                     "cardiopulmonary process; against a normal study.", ["labs.spo2"],
                     {D.NORMAL: -0.9}, "Oxygenation red-flag threshold", "labs")
    return None


RULES: list[Rule] = [
    Rule("bnp", "labs", "ACC/AHA HF", _r_bnp),
    Rule("infection", "labs", "IDSA/ATS CAP", _r_infection),
    Rule("malignancy", "history", "Fleischner/Lung-RADS", _r_malignancy),
    Rule("copd", "history", "GOLD", _r_copd),
    Rule("chf_signs", "symptoms", "ACC/AHA HF", _r_chf_signs),
    Rule("pneumothorax", "symptoms", "BTS", _r_pneumothorax),
    Rule("immunosupp", "history", "IDSA", _r_immunosupp),
    Rule("hypoxia", "labs", "Oxygenation", _r_hypoxia),
]

# Imaging findings that intrinsically support each diagnosis (for the differential).
_FINDING_SUPPORT = {
    D.PNEUMONIA: [Finding.CONSOLIDATION, Finding.OPACITY],
    D.HEART_FAILURE: [Finding.CARDIOMEGALY, Finding.EFFUSION],
    D.COPD: [Finding.HYPERINFLATION],
    D.MALIGNANCY: [Finding.NODULE],
    D.PNEUMOTHORAX: [Finding.PNEUMOTHORAX],
    D.NORMAL: [],
}


class ClinicalReasoner:
    def __init__(self):
        self.model_version = "reasoning-v1"

    def reason(self, study_id: str, findings: dict[Finding, float],
               posterior: dict[Diagnosis, float], priors: StructuredPriors,
               multimodal: MultimodalContext | None) -> tuple[ReasoningTrace, EvidenceGraph]:
        mm = multimodal or MultimodalContext()
        ctx = Ctx(findings=findings, priors=priors, mm=mm)

        p0 = np.array([max(1e-9, posterior.get(d, 0.0)) for d in DIAGNOSES])
        p0 = p0 / p0.sum()
        logit = np.log(p0)

        steps: list[ReasoningStep] = []
        citations: list[str] = []
        for rule in RULES:
            step = rule.fire(ctx)
            if step is None:
                continue
            for i, d in enumerate(DIAGNOSES):
                logit[i] += step.effect.get(d, 0.0)
            steps.append(step)
            if step.guideline and step.guideline not in citations:
                citations.append(step.guideline)

        adjusted = softmax(logit)
        prior_d = {d: float(p0[i]) for i, d in enumerate(DIAGNOSES)}
        adj_d = {d: float(adjusted[i]) for i, d in enumerate(DIAGNOSES)}

        differential = self._differential(ctx, adj_d, steps)
        graph = self._build_graph(ctx, steps, adj_d)

        trace = ReasoningTrace(
            study_id=study_id,
            prior_posterior={d: round(v, 4) for d, v in prior_d.items()},
            adjusted_posterior={d: round(v, 4) for d, v in adj_d.items()},
            steps=steps,
            differential=differential,
            guideline_citations=citations,
            model_version=self.model_version,
        )
        return trace, graph

    # ---- evidence graph construction ------------------------------------ #
    def _build_graph(self, ctx: Ctx, steps: list[ReasoningStep],
                     adj: dict[Diagnosis, float]) -> EvidenceGraph:
        graph = EvidenceGraph()

        # Imaging finding nodes
        for f, prob in ctx.findings.items():
            nid = f"ev.{f.value}"
            kind = EvidenceKind.IMAGING_FINDING if prob >= 0.5 else EvidenceKind.ABSENT_EVIDENCE
            graph.add_node(EvidenceNode(
                id=nid, kind=kind, label=f.value, value=prob, modality="imaging",
                confidence=prob,
            ))

        # Lab nodes (raw values)
        labs = ctx.mm.labs
        for name in ("bnp", "wbc", "procalcitonin", "crp", "spo2", "troponin", "d_dimer"):
            val = getattr(labs, name, None)
            if val is not None:
                graph.add_node(EvidenceNode(
                    id=f"hx.{name}", kind=EvidenceKind.STRUCTURED_PRIOR,
                    label=name, value=float(val), modality="labs",
                ))

        # Symptom nodes
        for name in ("fever", "orthopnea", "pleuritic_chest_pain", "dyspnea",
                      "productive_cough", "hemoptysis", "acute_onset"):
            val = getattr(ctx.mm.symptoms, name, None)
            if val is not None:
                graph.add_node(EvidenceNode(
                    id=f"sym.{name}", kind=EvidenceKind.STRUCTURED_PRIOR,
                    label=name, value=1.0 if val else 0.0, modality="symptoms",
                ))

        # Hypothesis nodes for each chest diagnosis
        for dx in CHEST_DIAGNOSES:
            graph.add_node(EvidenceNode(
                id=f"dx.{dx.value}", kind=EvidenceKind.STRUCTURED_PRIOR,
                label=dx.value, value=adj.get(dx, 0.0), modality="hypothesis",
            ))

        # Edges from reasoning steps
        for step in steps:
            for dx, lr in step.effect.items():
                if abs(lr) < 0.01:
                    continue
                # Map evidence names to node IDs
                for ev_name in step.evidence:
                    src_id = self._ev_to_node_id(ev_name)
                    if src_id is None or src_id not in graph.nodes:
                        continue
                    tgt_id = f"dx.{dx.value}"
                    if lr > 0:
                        rel = RelationType.SUPPORTS
                    else:
                        rel = RelationType.REFUTES
                    graph.add_edge(EvidenceEdge(
                        source_id=src_id, target_id=tgt_id,
                        relation=rel, weight=abs(lr),
                        rationale=step.statement,
                    ))

        # Detect contradictions: high lab vs low imaging for same hypothesis
        self._detect_contradictions(graph, ctx)

        return graph

    def _ev_to_node_id(self, ev_name: str) -> str | None:
        if ev_name.startswith("imaging."):
            return f"ev.{ev_name[len('imaging.'):]}"
        if ev_name.startswith("labs."):
            return f"hx.{ev_name[len('labs.'):]}"
        if ev_name.startswith("symptoms."):
            return f"sym.{ev_name[len('symptoms.'):]}"
        if ev_name.startswith("history."):
            return None
        if ev_name.startswith("priors."):
            return None
        return None

    def _detect_contradictions(self, graph: EvidenceGraph, ctx: Ctx) -> None:
        bnp = ctx.mm.labs.bnp
        cardiac_findings = [
            ctx.findings.get(Finding.CARDIOMEGALY, 0.0),
            ctx.findings.get(Finding.EFFUSION, 0.0),
        ]
        max_cardiac = max(cardiac_findings) if cardiac_findings else 0.0
        if bnp is not None and bnp >= 400 and max_cardiac < 0.3:
            if "hx.bnp" in graph.nodes and "dx.heart_failure" in graph.nodes:
                graph.add_edge(EvidenceEdge(
                    source_id="hx.bnp", target_id="dx.heart_failure",
                    relation=RelationType.CONTRADICTS, weight=0.8,
                    rationale="Elevated BNP without corroborating cardiac imaging findings",
                ))

        # Consolidation + high WBC without fever
        consol = ctx.findings.get(Finding.CONSOLIDATION, 0.0)
        wbc = ctx.mm.labs.wbc
        if consol >= 0.5 and wbc is not None and wbc > 11 and not ctx.mm.symptoms.fever:
            if "ev.consolidation" in graph.nodes and "hx.wbc" in graph.nodes:
                graph.add_edge(EvidenceEdge(
                    source_id="ev.consolidation", target_id="hx.wbc",
                    relation=RelationType.CONTRADICTS, weight=0.5,
                    rationale="Consolidation with elevated WBC but no fever",
                ))

    def _differential(self, ctx: Ctx, adj: dict[Diagnosis, float],
                      steps: list[ReasoningStep]) -> list[DifferentialItem]:
        items: list[DifferentialItem] = []
        for d in sorted(DIAGNOSES, key=lambda x: -adj[x])[:4]:
            supporting, opposing = [], []
            for f in _FINDING_SUPPORT.get(d, []):
                if ctx.findings.get(f, 0.0) >= 0.5:
                    supporting.append(f"imaging.{f.value}")
            for st in steps:
                e = st.effect.get(d, 0.0)
                if e > 0.05:
                    supporting.extend(st.evidence)
                elif e < -0.05:
                    opposing.extend(st.evidence)
            items.append(DifferentialItem(
                diagnosis=d, probability=round(adj[d], 4),
                supporting=list(dict.fromkeys(supporting)),
                opposing=list(dict.fromkeys(opposing)),
            ))
        return items
