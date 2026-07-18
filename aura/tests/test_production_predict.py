"""Regression tests for Step 7/10 — single-image predict orchestration."""
from __future__ import annotations

import numpy as np
import pytest

from services.inference.predict import predict_image


@pytest.fixture(scope="module")
def pipeline():
    from gateway.pipeline import Pipeline

    return Pipeline()


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


def test_predict_returns_full_result(pipeline, synthetic_cxr, tmp_path):
    res = predict_image(synthetic_cxr, pipeline=pipeline, out_dir=tmp_path / "out",
                        include_scorecam=False)
    assert len(res["findings"]) == 7
    for f in res["findings"]:
        assert 0.0 <= f["probability"] <= 1.0
        assert set(f) >= {"finding", "key", "probability", "present"}
    assert res["top_diagnosis"]
    assert res["inference_time_s"] > 0
    # Success-criteria fields all present.
    rep = res["clinical_report"]
    for key in ("vision_findings", "confidence", "differential_diagnosis",
                "recommended_tests", "risk_level", "limitations"):
        assert key in rep


def test_predict_writes_artifacts(pipeline, synthetic_cxr, tmp_path):
    out = tmp_path / "art"
    res = predict_image(synthetic_cxr, pipeline=pipeline, out_dir=out,
                        include_scorecam=False)
    arts = res["artifacts"]
    assert "report_markdown" in arts and "overlay_png" in arts
    assert "explanation_html" in arts
    from pathlib import Path

    for k in ("report_markdown", "overlay_png", "explanation_html"):
        assert Path(arts[k]).exists()


def test_predict_missing_file():
    with pytest.raises(FileNotFoundError):
        predict_image("does_not_exist_1234.jpg")
