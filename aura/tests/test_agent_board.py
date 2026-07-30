from __future__ import annotations

import asyncio

import numpy as np
import pytest

from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Finding, Modality
from aura.services.agent.base import AgentVerdict, ClinicalAgent
from aura.services.agent.registry import AgentRegistry
from aura.services.agent.consensus import ConsensusEngine, _js_divergence, AGENT_TIMEOUT


# --------------------------------------------------------------------------- #
# Agent contract
# --------------------------------------------------------------------------- #
class _DummyAgent(ClinicalAgent):
    agent_name = "dummy"
    def __init__(self, verdict: AgentVerdict | None = None, timeout: bool = False):
        self._verdict = verdict
        self._timeout = timeout
        self.agent_version = "1.0"
        if verdict is not None:
            self.agent_name = verdict.agent_name
    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        if self._timeout:
            await asyncio.sleep(AGENT_TIMEOUT + 1)
        return self._verdict or AgentVerdict(
            agent_name=self.agent_name, agent_version="1.0", confidence=0.0, findings={},
        )


def test_agent_verdict_contract():
    v = AgentVerdict(
        agent_name="test_agent",
        agent_version="1.0",
        confidence=0.85,
        findings={Diagnosis.PNEUMONIA: 0.7, Diagnosis.NORMAL: 0.3},
        arguments=["Visual analysis reveals opacity."],
        guideline_references=["Fleischner Society"],
    )
    assert v.agent_name == "test_agent"
    assert v.confidence == 0.85
    assert v.findings[Diagnosis.PNEUMONIA] == 0.7
    assert v.guideline_references[0] == "Fleischner Society"


def test_agent_empty_findings():
    v = AgentVerdict(agent_name="a", agent_version="1", confidence=0.0, findings={})
    assert len(v.findings) == 0


# --------------------------------------------------------------------------- #
# AgentRegistry
# --------------------------------------------------------------------------- #
def test_registry_register_and_list():
    AgentRegistry.clear()
    a1 = _DummyAgent(AgentVerdict(agent_name="alpha", agent_version="1", confidence=0.5, findings={}))
    a2 = _DummyAgent(AgentVerdict(agent_name="beta", agent_version="1", confidence=0.6, findings={}))
    AgentRegistry.register(a1)
    AgentRegistry.register(a2)
    names = [a.agent_name for a in AgentRegistry.all()]
    assert "alpha" in names
    assert "beta" in names


def test_registry_get():
    AgentRegistry.clear()
    a = _DummyAgent(AgentVerdict(agent_name="gamma", agent_version="1", confidence=0.5, findings={}))
    AgentRegistry.register(a)
    assert AgentRegistry.get("gamma") is a
    assert AgentRegistry.get("nonexistent") is None


def test_registry_clear():
    AgentRegistry.clear()
    assert len(list(AgentRegistry.all())) == 0


# --------------------------------------------------------------------------- #
# JS-divergence utility
# --------------------------------------------------------------------------- #
def test_js_divergence_identical():
    p = [0.5, 0.3, 0.2]
    assert _js_divergence(p, p) == 0.0


def test_js_divergence_orthogonal():
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    js = _js_divergence(p, q)
    assert 0.6 < js < 0.7


def test_js_divergence_clipping():
    p = [0.0, 0.0, 0.0]
    q = [1.0, 0.0, 0.0]
    assert _js_divergence(p, q) > 0.0


# --------------------------------------------------------------------------- #
# ClinicalAgent interface: supported() and get_base_weight()
# --------------------------------------------------------------------------- #
def test_clinical_agent_defaults():
    a = _DummyAgent()
    assert a.supported(Modality.CXR) is True
    assert a.get_base_weight() == 0.2


# --------------------------------------------------------------------------- #
# ConsensusEngine integration
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consensus_all_agents_agree():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    engine._register_defaults()

    evidence = {
        "findings": {
            Finding.OPACITY: 0.9,
            Finding.CONSOLIDATION: 0.8,
        },
        "labs": {"wbc": 15.0, "procalcitonin": 2.5, "crp": 120, "bnp": 50.0},
        "symptoms": {"dyspnea": True, "fever": True, "orthopnea": False},
    }
    priors = {"smoker": False, "prior_cancer": False}

    result = await engine.evaluate(evidence, priors)

    assert result.consensus_entropy >= 0.0
    assert 0.0 < result.confidence <= 1.0
    top = max(result.posterior, key=result.posterior.get)
    assert top in CHEST_DIAGNOSES
    assert len(result.verdicts) == 5
    assert result.panel_discussion != ""
    assert result.model_version == "consensus-v2"


@pytest.mark.asyncio
async def test_consensus_conflict_triggers_arbitration():
    AgentRegistry.clear()

    for i in range(3):
        AgentRegistry.register(_DummyAgent(
            AgentVerdict(
                agent_name=f"pro_pneumonia_{i}",
                agent_version="1", confidence=0.9,
                findings={Diagnosis.PNEUMONIA: 0.9, Diagnosis.HEART_FAILURE: 0.05},
            )
        ))
    for i in range(2):
        AgentRegistry.register(_DummyAgent(
            AgentVerdict(
                agent_name=f"pro_hf_{i}",
                agent_version="1", confidence=0.9,
                findings={Diagnosis.HEART_FAILURE: 0.85, Diagnosis.PNEUMONIA: 0.1},
            )
        ))

    engine = ConsensusEngine()
    result = await engine.evaluate({}, {})

    assert len(result.arbitration_history) > 0
    top = max(result.posterior, key=result.posterior.get)
    assert top == Diagnosis.PNEUMONIA


@pytest.mark.asyncio
async def test_consensus_high_entropy_single_agent():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    AgentRegistry.clear()
    AgentRegistry.register(_DummyAgent(
        AgentVerdict(
            agent_name="uncertain",
            agent_version="1", confidence=0.3,
            findings={d: 1.0 / len(CHEST_DIAGNOSES) for d in CHEST_DIAGNOSES},
        )
    ))
    result = await engine.evaluate({}, {})
    assert result.consensus_entropy == 0.0
    assert result.confidence == 0.3
    # A lone agent has no peers, so its mean peer-agreement is a mean over an empty
    # list. That used to be nan and poisoned every weight; the `or 1.0` fallbacks did
    # not catch it because nan is truthy, and this assertion did not exist, so the
    # posterior came back all-nan while the test still passed.
    assert all(np.isfinite(p) for p in result.posterior.values())
    assert result.posterior[Diagnosis.NORMAL] == pytest.approx(
        1.0 / len(CHEST_DIAGNOSES), abs=1e-9
    )
    assert sum(result.posterior.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.asyncio
async def test_consensus_guideline_agent_truth():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    engine._register_defaults()

    from aura.services.agent.specialists.guideline import GuidelineAgent
    AgentRegistry.clear()
    AgentRegistry.register(GuidelineAgent())

    evidence = {
        "findings": {Finding.NODULE: 0.8},
        "labs": {},
        "symptoms": {},
    }
    priors = {"smoker": True, "prior_cancer": False}

    result = await engine.evaluate(evidence, priors)
    guideline_verdict = [v for v in result.verdicts if v.agent_name == "guideline"][0]
    assert guideline_verdict.findings.get(Diagnosis.MALIGNANCY, 0.0) > 0.3
    assert len(guideline_verdict.guideline_references) > 0


# --------------------------------------------------------------------------- #
# Resilience: timeout and fallback
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consensus_timeout_fallback():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    AgentRegistry.clear()

    AgentRegistry.register(_DummyAgent(
        AgentVerdict(agent_name="fast", agent_version="1", confidence=0.8,
                     findings={Diagnosis.PNEUMONIA: 0.9}),
    ))
    AgentRegistry.register(_DummyAgent(
        AgentVerdict(agent_name="slow", agent_version="1", confidence=0.9,
                     findings={Diagnosis.COPD: 0.8}),
        timeout=True,
    ))

    result = await engine.evaluate({}, {})
    assert len(result.verdicts) == 2
    slow_verdict = [v for v in result.verdicts if v.agent_name == "slow"][0]
    assert slow_verdict.confidence == 0.0
    assert slow_verdict.metadata.get("status") == "failed_timeout"


@pytest.mark.asyncio
async def test_consensus_exception_fallback():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    AgentRegistry.clear()

    class _CrashAgent(ClinicalAgent):
        agent_name = "crasher"
        def __init__(self):
            self.agent_version = "1"
        async def evaluate(self, evidence, priors):
            raise ValueError("something broke")

    AgentRegistry.register(_CrashAgent())
    AgentRegistry.register(_DummyAgent(
        AgentVerdict(agent_name="ok", agent_version="1", confidence=0.8,
                     findings={Diagnosis.PNEUMONIA: 0.9}),
    ))

    result = await engine.evaluate({}, {})
    assert len(result.verdicts) == 2
    crash_v = [v for v in result.verdicts if v.agent_name == "crasher"][0]
    assert crash_v.confidence == 0.0
    assert "something broke" in " ".join(crash_v.arguments)


# --------------------------------------------------------------------------- #
# Escalation: requires_review
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consensus_requires_review():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    AgentRegistry.clear()
    # Three agents with completely disjoint diagnoses
    AgentRegistry.register(_DummyAgent(
        AgentVerdict(agent_name="pro_pna", agent_version="1", confidence=0.9,
                     findings={Diagnosis.PNEUMONIA: 0.95}),
    ))
    AgentRegistry.register(_DummyAgent(
        AgentVerdict(agent_name="pro_hf", agent_version="1", confidence=0.9,
                     findings={Diagnosis.HEART_FAILURE: 0.95}),
    ))
    AgentRegistry.register(_DummyAgent(
        AgentVerdict(agent_name="pro_malignancy", agent_version="1", confidence=0.9,
                     findings={Diagnosis.MALIGNANCY: 0.95}),
    ))
    result = await engine.evaluate({}, {})
    # Three-way split => entropy > 0.3 and top < 0.5 => requires review
    assert result.consensus_entropy > 0.3
    assert max(result.posterior.values()) < 0.5
    assert result.requires_review is True


# --------------------------------------------------------------------------- #
# Modality routing
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consensus_modality_filter():
    AgentRegistry.clear()

    class _ChestAgent(ClinicalAgent):
        agent_name = "chest_only"
        agent_version = "1"
        def supported(self, modality: Modality) -> bool:
            return modality == Modality.CXR
        async def evaluate(self, evidence, priors):
            return AgentVerdict(agent_name="chest_only", agent_version="1",
                                confidence=0.8, findings={Diagnosis.PNEUMONIA: 0.9})

    class _BrainAgent(ClinicalAgent):
        agent_name = "brain_only"
        agent_version = "1"
        def supported(self, modality: Modality) -> bool:
            return modality == Modality.MR
        async def evaluate(self, evidence, priors):
            return AgentVerdict(agent_name="brain_only", agent_version="1",
                                confidence=0.8, findings={Diagnosis.NORMAL: 0.9})

    AgentRegistry.register(_ChestAgent())
    AgentRegistry.register(_BrainAgent())

    engine = ConsensusEngine()
    AgentRegistry.clear()
    AgentRegistry.register(_ChestAgent())
    AgentRegistry.register(_BrainAgent())

    result = await engine.evaluate({}, {}, modality=Modality.CXR)
    assert len(result.verdicts) == 1
    assert result.verdicts[0].agent_name == "chest_only"

    result2 = await engine.evaluate({}, {}, modality=Modality.MR)
    assert len(result2.verdicts) == 1
    assert result2.verdicts[0].agent_name == "brain_only"


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consensus_to_dict():
    AgentRegistry.clear()
    engine = ConsensusEngine()
    engine._register_defaults()

    evidence = {
        "findings": {Finding.OPACITY: 0.7},
        "labs": {},
        "symptoms": {},
    }
    priors = {}
    result = await engine.evaluate(evidence, priors)
    d = result.to_dict()
    assert "posterior" in d
    assert "confidence" in d
    assert "consensus_entropy" in d
    assert "agreement_matrix" in d
    assert "verdicts" in d
    assert "arbitration_history" in d
    assert "panel_discussion" in d
    assert "requires_review" in d
    assert isinstance(d["posterior"], dict)
    assert all(isinstance(k, str) for k in d["posterior"])
