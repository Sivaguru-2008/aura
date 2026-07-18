"""Regression tests for Step 3 — Score-CAM, overlays, bounding boxes, HTML report."""
from __future__ import annotations

import numpy as np
import pytest

from schemas.clinical import Finding
from services.explain import overlays as O


def test_bboxes_and_norm():
    heat = np.zeros((64, 64), dtype=float)
    heat[20:40, 25:45] = 1.0
    boxes = O.heatmap_bboxes(heat, thresh_rel=0.5)
    assert len(boxes) == 1
    r0, c0, r1, c1 = boxes[0]["bbox"]
    assert 0.0 <= r0 < r1 <= 1.0 and 0.0 <= c0 < c1 <= 1.0
    assert boxes[0]["peak"] == pytest.approx(1.0, abs=1e-6)


def test_bboxes_empty_map():
    assert O.heatmap_bboxes(np.zeros((32, 32))) == []


def test_overlay_shape_and_range():
    base = np.clip(np.random.rand(100, 100), 0, 1)
    heat = np.zeros((64, 64)); heat[10:30, 10:30] = 1.0
    ov = O.overlay(base, heat)
    assert ov.shape == (100, 100, 3)
    assert ov.dtype == np.uint8


def test_png_and_html_writers(tmp_path):
    base = np.clip(np.random.rand(96, 96), 0, 1).astype("float32")
    heat = np.zeros((64, 64), "float32"); heat[20:40, 20:40] = 1.0
    boxes = O.heatmap_bboxes(heat)
    p1 = O.save_overlay_png(tmp_path / "ov.png", base, heat, boxes, "t")
    p2 = O.save_high_res_overlay(tmp_path / "hi.png", base, heat, boxes)
    p3 = O.save_raw_heatmap_png(tmp_path / "heat.png", heat)
    p4 = O.save_evidence_bar_png(tmp_path / "ev.png", {"opacity": 0.3, "effusion": -0.1})
    html = O.save_explanation_html(
        tmp_path / "r.html", base, {"grad_cam++": heat, "score_cam": heat * 0.7},
        "grad_cam++", boxes, "effusion", [("effusion", 0.8), ("opacity", 0.4)],
        evidence_attribution={"opacity": 0.3, "effusion": -0.1},
        meta={"model": "test"})
    for p in (p1, p2, p3, p4, html):
        assert p.exists() and p.stat().st_size > 0
    assert "Explainability" in html.read_text(encoding="utf-8")


@pytest.mark.parametrize("include_scorecam", [False, True])
def test_gradient_methods_on_backbone(include_scorecam):
    """Score-CAM + gradient methods run end-to-end on the production backbone."""
    from common.config import ARTIFACTS

    if not (ARTIFACTS / "best_model.pt").exists():
        pytest.skip("best_model.pt not present")
    from ml.vision_cxr.inference import VisionModel
    from services.explain import methods as M

    vm = VisionModel(str(ARTIFACTS / "best_model.pt"))
    img = np.clip(np.random.rand(224, 224), 0, 1).astype("float32")
    maps = M.all_methods(vm, img, Finding.EFFUSION, out_size=64,
                         include_scorecam=include_scorecam)
    assert "grad_cam++" in maps
    if include_scorecam:
        assert "score_cam" in maps
    for m in maps.values():
        assert m.shape == (64, 64)
        assert 0.0 <= float(np.min(m)) and float(np.max(m)) <= 1.0 + 1e-6
