from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from aura.gateway.app import app, store
from aura.schemas.contracts import (
    CaseBundle,
    CaseState,
    ChatMessage,
    ClinicalChatRequest,
    ClinicalChatResponse,
    ClinicalHistory,
    LabPanel,
    MultimodalContext,
    StructuredPriors,
    Symptoms,
    VisionResult,
    FindingScore,
)
from aura.schemas.clinical import Diagnosis, Finding


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _patch_store(bundle: CaseBundle):
    """Mock store().get_case to return the given bundle."""
    mock_store = MagicMock()
    mock_store.get_case.return_value = bundle
    mock_store.save_chat_message = MagicMock()
    mock_store.audit = MagicMock()
    mock_store.get_chat_history.return_value = []
    return patch("aura.gateway.app.store", return_value=mock_store)


def _make_minimal_bundle() -> CaseBundle:
    return CaseBundle(
        case_id="COPILOT-TEST-001",
        study_id="STU-TEST-001",
        state=CaseState.READY,
        priors=StructuredPriors(
            age_band="40-65", sex="male", smoker=True,
            fever=True, prior_cancer=False, immunocompromised=False,
        ),
        multimodal=MultimodalContext(
            labs=LabPanel(wbc=14.5, bnp=85.0, crp=120.0, spo2=91.0),
            symptoms=Symptoms(dyspnea=True, productive_cough=True, fever=True),
            history=ClinicalHistory(copd=False, heart_failure=False, smoking_pack_years=30.0),
        ),
        vision=VisionResult(
            study_id="STU-TEST-001",
            findings=[
                FindingScore(finding=Finding.OPACITY, probability=0.87),
                FindingScore(finding=Finding.CONSOLIDATION, probability=0.72),
                FindingScore(finding=Finding.NODULE, probability=0.05),
            ],
            embedding=[0.1] * 64,
            model_version="test",
        ),
        consensus_result={
            "posterior": {"pneumonia": 0.88, "copd": 0.07},
            "consensus_entropy": 0.15,
            "confidence": 0.85,
            "verdicts": [
                {
                    "agent_name": "radiologist",
                    "confidence": 0.92,
                    "findings": {"pneumonia": 0.90},
                    "arguments": ["Bilateral opacities consistent with pneumonia."],
                    "guideline_references": ["IDSA/ATS CAP"],
                    "abstained": False,
                },
                {
                    "agent_name": "pulmonologist",
                    "confidence": 0.88,
                    "findings": {"pneumonia": 0.85, "copd": 0.10},
                    "arguments": ["Elevated WBC and productive cough support pneumonia."],
                    "guideline_references": ["GOLD COPD"],
                    "abstained": False,
                },
                {
                    "agent_name": "pathology",
                    "confidence": 0.65,
                    "findings": {"pneumonia": 0.70, "malignancy": 0.20},
                    "arguments": ["Nodule present but low suspicion for malignancy."],
                    "guideline_references": [],
                    "abstained": False,
                },
            ],
            "panel_discussion": "Radiologist detected bilateral opacities. "
                                "Pulmonologist confirms with clinical correlation. "
                                "Pathology notes low malignancy risk.",
        },
        dx_labels={},
        ev_labels={},
    )


# --------------------------------------------------------------------------- #
# Schema tests
# --------------------------------------------------------------------------- #

def test_chat_request_schema():
    req = ClinicalChatRequest(question="What is the primary diagnosis?")
    assert req.question == "What is the primary diagnosis?"
    assert req.history == []

    req2 = ClinicalChatRequest(
        question="What about the nodule?",
        history=[ChatMessage(role="user", content="What is the primary diagnosis?")],
    )
    assert len(req2.history) == 1
    assert req2.history[0].role == "user"


def test_chat_response_schema():
    resp = ClinicalChatResponse(answer="Pneumonia", sources=["Radiologist Agent"])
    assert resp.answer == "Pneumonia"
    assert resp.sources == ["Radiologist Agent"]
    assert resp.correlation_id == ""


# --------------------------------------------------------------------------- #
# Prompt builder tests
# --------------------------------------------------------------------------- #

def test_prompt_contains_case_facts():
    from aura.services.copilot.prompt_builder import build_grounding_prompt

    bundle = _make_minimal_bundle()
    prompt = build_grounding_prompt(bundle)

    assert "STU-TEST-001" in prompt
    assert "Chest X-ray" in prompt
    assert "40-65" in prompt
    assert "Smoker: True" in prompt
    assert "WBC=14.5" in prompt
    assert "dyspnea" in prompt
    assert "productive cough" in prompt
    assert "opacity" in prompt
    assert "consolidation" in prompt
    # Vision finding probabilities below 0.1 are excluded from the prompt
    assert "0.05" not in prompt
    assert "Nodule present but low suspicion" in prompt  # from agent arguments
    assert "pneumonia" in prompt
    assert "Radiologist" in prompt
    assert "Pulmonologist" in prompt
    assert "Bilateral opacities" in prompt


def test_prompt_no_consensus():
    from aura.services.copilot.prompt_builder import build_grounding_prompt

    bundle = _make_minimal_bundle()
    bundle.consensus_result = None
    bundle.vision = None
    prompt = build_grounding_prompt(bundle)

    assert "Consensus Diagnosis" not in prompt
    assert "Agent Panel Verdicts" not in prompt
    assert "Vision Findings" not in prompt
    assert "Patient Priors" in prompt


# --------------------------------------------------------------------------- #
# Ollama client tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_ollama_client_connection_error():
    from aura.services.copilot.ollama_client import OllamaCopilotClient, OllamaConnectionError

    client = OllamaCopilotClient(base_url="http://localhost:1")

    with pytest.raises(OllamaConnectionError, match="Cannot connect to Ollama"):
        await client.ask_copilot("system", "user question")


@pytest.mark.asyncio
async def test_ollama_client_success():
    from aura.services.copilot.ollama_client import OllamaCopilotClient

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "model": "llama3.1",
                "message": {"role": "assistant", "content": "The primary diagnosis is pneumonia."},
            }

    client = OllamaCopilotClient()
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=_FakeResponse())):
        answer, correlation_id = await client.ask_copilot(
            system_prompt="You are a clinical copilot.",
            user_message="What is the diagnosis?",
        )

    assert answer == "The primary diagnosis is pneumonia."
    assert len(correlation_id) == 12


@pytest.mark.asyncio
async def test_ollama_client_timeout():
    from aura.services.copilot.ollama_client import OllamaCopilotClient, OllamaConnectionError

    client = OllamaCopilotClient(base_url="http://localhost:1")

    with patch("httpx.AsyncClient.post", AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
        with pytest.raises(OllamaConnectionError, match="timed out"):
            await client.ask_copilot("system", "user question")


@pytest.mark.asyncio
async def test_ollama_client_http_error():
    from aura.services.copilot.ollama_client import OllamaCopilotClient, OllamaConnectionError

    class _FakeResponse500:
        status_code = 500
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("POST", "http://localhost:11434/api/chat"),
                response=self,
            )
        def json(self):
            return {}

    client = OllamaCopilotClient()
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=_FakeResponse500())):
        with pytest.raises(OllamaConnectionError, match="returned an error"):
            await client.ask_copilot("system", "user question")


# --------------------------------------------------------------------------- #
# API endpoint tests
# --------------------------------------------------------------------------- #

def test_chat_endpoint_case_not_found(client):
    resp = client.post("/v1/cases/NONEXISTENT/chat", json={"question": "test"})
    assert resp.status_code == 404


def test_chat_endpoint_ollama_down(client):
    bundle = _make_minimal_bundle()
    with _patch_store(bundle), \
         patch.object(httpx.AsyncClient, "post",
                      AsyncMock(side_effect=httpx.ConnectError("connection refused"))):
        resp = client.post(
            "/v1/cases/COPILOT-TEST-001/chat",
            json={"question": "What is the diagnosis?"},
        )
    assert resp.status_code == 503
    data = resp.json()
    assert data["detail"]["error"] == "ollama_unavailable"


def test_chat_endpoint_success(client):
    class _FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "model": "llama3.1",
                "message": {"role": "assistant", "content": "The primary diagnosis is pneumonia."},
            }

    bundle = _make_minimal_bundle()
    with _patch_store(bundle), \
         patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=_FakeResponse())):
        resp = client.post(
            "/v1/cases/COPILOT-TEST-001/chat",
            json={"question": "What is the primary diagnosis?"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The primary diagnosis is pneumonia."
    assert len(data["sources"]) > 0
    assert data["correlation_id"]


def test_chat_history_endpoint(client):
    bundle = _make_minimal_bundle()
    with _patch_store(bundle):
        resp = client.get("/v1/cases/COPILOT-TEST-001/chat")
    assert resp.status_code == 200
    data = resp.json()
    assert data["case_id"] == "COPILOT-TEST-001"
    assert "messages" in data


def test_chat_history_endpoint_not_found(client):
    resp = client.get("/v1/cases/NONEXISTENT/chat")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Prompt builder behavioral guardrails
# --------------------------------------------------------------------------- #

def test_prompt_contains_guardrails():
    from aura.services.copilot.prompt_builder import build_grounding_prompt

    bundle = _make_minimal_bundle()
    prompt = build_grounding_prompt(bundle)

    assert "ONLY the provided case facts" in prompt
    assert "Never hallucinate" in prompt
    assert "not present in the scan's evidence graph" in prompt
    assert "Cite your sources" in prompt
