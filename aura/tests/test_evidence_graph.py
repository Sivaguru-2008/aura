"""Tests for the structured Evidence Graph (Step 2).

Verifies that the reasoning engine constructs a typed, directed evidence graph
with correct node types, edge relations, and contradiction detection.
"""
from __future__ import annotations

import pytest
import numpy as np

from aura.schemas.clinical import Diagnosis, Finding
from aura.schemas.contracts import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceKind,
    EvidenceNode,
    MultimodalContext,
    RelationType,
    StructuredPriors,
)
from aura.services.reasoning.engine import ClinicalReasoner


@pytest.fixture
def reasoner():
    return ClinicalReasoner()


def _make_mm(bnp=None, wbc=None, procalcitonin=None, crp=None, spo2=None,
             fever=False, orthopnea=False, pleuritic_chest_pain=False,
             copd=False, heart_failure=False, smoking_pack_years=0.0,
             immunosuppression=False):
    from aura.schemas.contracts import LabPanel, Symptoms, ClinicalHistory
    return MultimodalContext(
        labs=LabPanel(bnp=bnp, wbc=wbc, procalcitonin=procalcitonin,
                      crp=crp, spo2=spo2),
        symptoms=Symptoms(fever=fever, orthopnea=orthopnea,
                          pleuritic_chest_pain=pleuritic_chest_pain),
        history=ClinicalHistory(copd=copd, heart_failure=heart_failure,
                                smoking_pack_years=smoking_pack_years,
                                immunosuppression=immunosuppression),
    )


# ------------------------------------------------------------------ #
# Test 1: Graph is produced and has nodes/edges
# ------------------------------------------------------------------ #
def test_reasoner_produces_graph(reasoner):
    findings = {Finding.OPACITY: 0.8, Finding.CONSOLIDATION: 0.7}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(wbc=15.0, procalcitonin=1.2, fever=True)

    trace, graph = reasoner.reason(
        "STU-1", findings, posterior, StructuredPriors(), mm
    )

    assert isinstance(graph, EvidenceGraph)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


# ------------------------------------------------------------------ #
# Test 2: Imaging findings become nodes
# ------------------------------------------------------------------ #
def test_imaging_finding_nodes_present(reasoner):
    findings = {
        Finding.OPACITY: 0.9,
        Finding.CARDIOMEGALY: 0.6,
        Finding.PNEUMOTHORAX: 0.1,
    }
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    trace, graph = reasoner.reason(
        "STU-2", findings, posterior, StructuredPriors(), None
    )

    assert "ev.opacity" in graph.nodes
    assert "ev.cardiomegaly" in graph.nodes
    assert "ev.pneumothorax" in graph.nodes

    op_node = graph.nodes["ev.opacity"]
    assert op_node.kind == EvidenceKind.IMAGING_FINDING
    assert op_node.modality == "imaging"
    assert op_node.value >= 0.5

    ptx_node = graph.nodes["ev.pneumothorax"]
    assert ptx_node.kind == EvidenceKind.ABSENT_EVIDENCE
    assert ptx_node.value < 0.5


# ------------------------------------------------------------------ #
# Test 3: Hypothesis nodes for diagnoses
# ------------------------------------------------------------------ #
def test_hypothesis_nodes_present(reasoner):
    findings = {Finding.OPACITY: 0.3}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    trace, graph = reasoner.reason(
        "STU-3", findings, posterior, StructuredPriors(), None
    )

    from aura.schemas.clinical import CHEST_DIAGNOSES
    for dx in CHEST_DIAGNOSES:
        hypo_id = f"dx.{dx.value}"
        assert hypo_id in graph.nodes, f"missing hypothesis node {hypo_id}"
        assert graph.nodes[hypo_id].modality == "hypothesis"


# ------------------------------------------------------------------ #
# Test 4: Lab values become nodes
# ------------------------------------------------------------------ #
def test_lab_nodes_present(reasoner):
    findings = {}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(bnp=500.0, wbc=14.0, spo2=88.0)
    trace, graph = reasoner.reason(
        "STU-4", findings, posterior, StructuredPriors(), mm
    )

    assert "hx.bnp" in graph.nodes
    assert graph.nodes["hx.bnp"].value == 500.0
    assert graph.nodes["hx.bnp"].modality == "labs"

    assert "hx.wbc" in graph.nodes
    assert "hx.spo2" in graph.nodes


# ------------------------------------------------------------------ #
# Test 5: Symptom nodes present
# ------------------------------------------------------------------ #
def test_symptom_nodes_present(reasoner):
    findings = {}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(fever=True, orthopnea=True)
    trace, graph = reasoner.reason(
        "STU-5", findings, posterior, StructuredPriors(), mm
    )

    assert "sym.fever" in graph.nodes
    assert "sym.orthopnea" in graph.nodes
    assert graph.nodes["sym.fever"].value == 1.0
    assert graph.nodes["sym.fever"].modality == "symptoms"


# ------------------------------------------------------------------ #
# Test 6: Supports / refutes edges from reasoning steps
# ------------------------------------------------------------------ #
def test_step_edges_typed_correctly(reasoner):
    findings = {Finding.CARDIOMEGALY: 0.7, Finding.EFFUSION: 0.6}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(bnp=500.0, orthopnea=True)
    trace, graph = reasoner.reason(
        "STU-6", findings, posterior, StructuredPriors(), mm
    )

    # BNP 500 should produce a SUPPORTS edge to heart_failure
    bnp_hf_edges = [
        e for e in graph.edges
        if e.source_id == "hx.bnp" and e.target_id == "dx.heart_failure"
    ]
    assert len(bnp_hf_edges) >= 1
    assert bnp_hf_edges[0].relation == RelationType.SUPPORTS

    # BNP 500 should also produce a REFUTES edge to normal
    bnp_norm_edges = [
        e for e in graph.edges
        if e.source_id == "hx.bnp" and e.target_id == "dx.normal"
    ]
    assert len(bnp_norm_edges) >= 1
    assert bnp_norm_edges[0].relation == RelationType.REFUTES


# ------------------------------------------------------------------ #
# Test 7: Contradiction detection — elevated BNP + normal CXR
# ------------------------------------------------------------------ #
def test_contradiction_bnp_vs_normal_cxr(reasoner):
    """Elevated BNP supporting heart failure while no cardiac imaging findings
    (normal CXR) should produce a contradicts edge."""
    # No cardiomegaly, no effusion — CXR is essentially normal
    findings = {Finding.OPACITY: 0.2, Finding.CARDIOMEGALY: 0.15,
                Finding.EFFUSION: 0.1}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(bnp=600.0)
    trace, graph = reasoner.reason(
        "STU-7", findings, posterior, StructuredPriors(), mm
    )

    contradictions = graph.contradicts_edges()
    assert len(contradictions) >= 1, (
        f"Expected at least one contradicts edge, got {len(contradictions)}. "
        f"Edges: {[(e.source_id, e.target_id, e.relation) for e in graph.edges]}"
    )

    # The contradiction should be between BNP and an imaging finding
    contra_pairs = {(e.source_id, e.target_id) for e in contradictions}
    bnp_contras = [p for p in contra_pairs if "bnp" in p[0] or "bnp" in p[1]]
    assert len(bnp_contras) >= 1, (
        f"Expected BNP in a contradiction, pairs: {contra_pairs}"
    )


# ------------------------------------------------------------------ #
# Test 8: Contradiction — consolidation + high WBC vs no fever
# ------------------------------------------------------------------ #
def test_contradiction_infection_vs_no_fever(reasoner):
    """Strong infection markers (consolidation + elevated WBC) without fever
    creates tension between infectious and non-infectious aetiologies."""
    findings = {Finding.CONSOLIDATION: 0.8, Finding.OPACITY: 0.7}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(wbc=16.0, procalcitonin=2.0, crp=80.0, fever=False)
    trace, graph = reasoner.reason(
        "STU-8", findings, posterior, StructuredPriors(), mm
    )

    # There should be supporting edges from infection markers to pneumonia
    pneumonia_support = [
        e for e in graph.edges
        if e.target_id == "dx.pneumonia" and e.relation == RelationType.SUPPORTS
    ]
    assert len(pneumonia_support) >= 2, (
        f"Expected multiple supporting edges for pneumonia, "
        f"got {len(pneumonia_support)}"
    )


# ------------------------------------------------------------------ #
# Test 9: Graph helper methods
# ------------------------------------------------------------------ #
def test_graph_helpers():
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode(id="ev.a", kind=EvidenceKind.IMAGING_FINDING,
                                label="a", value=0.8))
    graph.add_node(EvidenceNode(id="ev.b", kind=EvidenceKind.IMAGING_FINDING,
                                label="b", value=0.2))
    graph.add_node(EvidenceNode(id="dx.pneumonia", kind=EvidenceKind.IMAGING_FINDING,
                                label="pneumonia", value=0.6, modality="hypothesis"))

    graph.add_edge(EvidenceEdge(source_id="ev.a", target_id="dx.pneumonia",
                                relation=RelationType.SUPPORTS))
    graph.add_edge(EvidenceEdge(source_id="ev.b", target_id="dx.pneumonia",
                                relation=RelationType.REFUTES))
    graph.add_edge(EvidenceEdge(source_id="ev.a", target_id="ev.b",
                                relation=RelationType.CONTRADICTS))

    assert len(graph.edges_from("ev.a")) == 2
    assert len(graph.edges_to("dx.pneumonia")) == 2
    assert len(graph.contradicts_edges()) == 1
    assert len(graph.supporting_edges("dx.pneumonia")) == 1
    assert len(graph.supporting_edges("dx.normal")) == 0


# ------------------------------------------------------------------ #
# Test 10: No contradictions when evidence is consistent
# ------------------------------------------------------------------ #
def test_no_contradictions_when_consistent(reasoner):
    """When imaging and labs agree, no contradicts edges should appear."""
    findings = {Finding.CONSOLIDATION: 0.9, Finding.OPACITY: 0.85}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(wbc=15.0, procalcitonin=2.0, fever=True)
    trace, graph = reasoner.reason(
        "STU-10", findings, posterior, StructuredPriors(), mm
    )

    contradictions = graph.contradicts_edges()
    assert len(contradictions) == 0, (
        f"Expected no contradictions for consistent evidence, "
        f"got {len(contradictions)}: {[(e.source_id, e.target_id) for e in contradictions]}"
    )


# ------------------------------------------------------------------ #
# Test 11: Backward compatibility — flat evidence list still works
# ------------------------------------------------------------------ #
def test_flat_evidence_list_preserved():
    """CaseBundle.evidence (list[EvidenceItem]) must still be usable."""
    from aura.schemas.contracts import CaseBundle, CaseState, EvidenceItem
    bundle = CaseBundle(
        case_id="CASE-BC",
        study_id="STU-BC",
        state=CaseState.READY,
        evidence=[
            EvidenceItem(kind=EvidenceKind.IMAGING_FINDING, name="opacity",
                         value=0.8),
        ],
    )
    assert len(bundle.evidence) == 1
    assert bundle.evidence_graph is None  # optional, not set here


# ------------------------------------------------------------------ #
# Test 12: Graph serialises to JSON round-trip
# ------------------------------------------------------------------ #
def test_graph_json_roundtrip(reasoner):
    findings = {Finding.OPACITY: 0.8, Finding.CARDIOMEGALY: 0.6}
    posterior = {d: 1.0 / 6 for d in Diagnosis}
    mm = _make_mm(bnp=500.0, fever=True)
    _, graph = reasoner.reason(
        "STU-12", findings, posterior, StructuredPriors(), mm
    )

    json_str = graph.model_dump_json()
    restored = EvidenceGraph.model_validate_json(json_str)
    assert len(restored.nodes) == len(graph.nodes)
    assert len(restored.edges) == len(graph.edges)
    for nid in graph.nodes:
        assert nid in restored.nodes
