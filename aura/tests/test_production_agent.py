"""Regression tests for Step 14/15 — Active Diagnostic Agent integration on real images."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gateway.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def synthetic_cxr(tmp_path):
    """A synthetic chest-like grayscale image written to a PNG file."""
    from ml.data import make_sample
    from schemas.clinical import Diagnosis

    s = make_sample(Diagnosis.PNEUMONIA, np.random.default_rng(3))
    img = (np.clip(s.image, 0, 1) * 255).astype(np.uint8)
    path = tmp_path / "synthetic.png"
    try:
        import cv2
        cv2.imwrite(str(path), img)
    except Exception:
        from PIL import Image
        Image.fromarray(img).save(path)
    return path


def test_agent_endpoint_returns_trajectory(client, synthetic_cxr):
    with open(synthetic_cxr, "rb") as f:
        response = client.post(
            "/v1/studies/agent",
            files={"file": (synthetic_cxr.name, f, "image/png")},
            params={"entropy_target": 0.6, "confidence": 0.85, "max_tests": 2}
        )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "committed" in data
    assert "status" in data
    assert "final_diagnosis" in data
    assert "final_probability" in data
    assert "initial_entropy" in data
    assert "final_entropy" in data
    assert "steps" in data
    
    steps = data["steps"]
    assert len(steps) > 0
    for s in steps:
        assert "step" in s
        assert "posterior" in s
        assert "entropy_bits" in s
        assert "confident" in s


def test_agent_endpoint_rejects_non_cxr(client, tmp_path):
    # Create a plain white image of 10x10 which fails the xray gate check
    img = np.ones((10, 10), dtype=np.uint8) * 255
    path = tmp_path / "non_cxr.png"
    try:
        import cv2
        cv2.imwrite(str(path), img)
    except Exception:
        from PIL import Image
        Image.fromarray(img).save(path)
        
    with open(path, "rb") as f:
        response = client.post(
            "/v1/studies/agent",
            files={"file": (path.name, f, "image/png")}
        )
        
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "not_a_cxr"


def test_agent_quantum_trajectory():
    from services.fusion.engine import FusionEngine
    from services.agent.active_diagnosis import ActiveDiagnosisAgent
    import numpy as np

    # Ensure we load with quantum backend
    fusion_engine = FusionEngine(backend="quantum")
    assert fusion_engine.backend == "quantum"

    # Make active diagnosis agent
    agent = ActiveDiagnosisAgent(
        fusion_model=fusion_engine,
        entropy_target_bits=1.5,
        confidence=0.85,
        max_tests=3
    )

    # An uncertain evidence vector
    x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2], dtype=float)
    
    trajectory = agent.diagnose(x0)
    
    assert trajectory.backend == "quantum"
    assert len(trajectory.steps) > 0
    assert trajectory.initial_entropy > 0.0
    
    for step in trajectory.steps:
        assert step.entropy_bits >= 0.0
        if step.action_display:
            assert step.action_eig_bits is not None
            assert step.action_eig_bits >= 0.0
