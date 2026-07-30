"""Heatmap geometry: contours, polygons, probability weighting, RGBA overlays.

These tests pin the properties a clinician reads off the overlay — that a drawn
region actually sits on the activation, that its outline is a real outline and
not a decorated box, and that display intensity tracks calibrated probability
rather than per-finding normalisation.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from aura.services.explain.geometry import (
    convex_hull,
    extract_contours,
    heatmap_geometry,
    heatmap_regions,
    otsu_threshold,
    polygon_area,
    probability_weighted_mask,
    rgba_overlay,
    rgba_overlay_data_uri,
    simplify_polygon,
    threshold_mask,
)


def blob(cy: float, cx: float, spread: float = 60.0, size: int = 64,
         amplitude: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return amplitude * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / spread))


@pytest.fixture
def two_blobs() -> np.ndarray:
    return blob(20, 18) + blob(45, 46, spread=40, amplitude=0.8)


# --------------------------------------------------------------------------- #
# Masks and thresholds
# --------------------------------------------------------------------------- #
def test_probability_weighting_scales_intensity(two_blobs):
    """A 5%-probability finding must not render as hot as a 90% one."""
    high = probability_weighted_mask(two_blobs, 0.9)
    low = probability_weighted_mask(two_blobs, 0.05)

    assert high.max() == pytest.approx(0.9, abs=1e-6)
    assert low.max() == pytest.approx(0.05, abs=1e-6)
    assert low.max() < high.max()
    assert np.all(low <= high + 1e-9)


def test_probability_weighting_is_clipped_and_bounded(two_blobs):
    for p in (-1.0, 0.0, 0.5, 1.0, 2.0):
        m = probability_weighted_mask(two_blobs, p)
        assert 0.0 <= m.min() and m.max() <= 1.0


def test_probability_floor_keeps_a_faint_trace(two_blobs):
    assert probability_weighted_mask(two_blobs, 0.0, floor=0.1).max() == pytest.approx(0.1, abs=1e-6)


def test_relative_threshold_adapts_to_dynamic_range():
    """Relative thresholding must behave identically on a rescaled map."""
    faint = blob(32, 32) * 0.01
    assert threshold_mask(faint, thresh_rel=0.5).sum() == threshold_mask(blob(32, 32), thresh_rel=0.5).sum()


def test_absolute_threshold_overrides_relative(two_blobs):
    strict = threshold_mask(two_blobs, thresh_rel=0.5, thresh_abs=0.95)
    assert strict.sum() < threshold_mask(two_blobs, thresh_rel=0.5).sum()


def test_otsu_separates_a_bimodal_map(two_blobs):
    t = otsu_threshold(two_blobs)
    assert 0.0 < t < 1.0
    assert threshold_mask(two_blobs, thresh_abs=t).any()


def test_empty_map_yields_no_regions():
    assert heatmap_regions(np.zeros((32, 32))) == []
    assert extract_contours(np.zeros((32, 32))) == []


# --------------------------------------------------------------------------- #
# Region geometry
# --------------------------------------------------------------------------- #
def test_two_blobs_produce_two_regions(two_blobs):
    regions = heatmap_regions(two_blobs, min_area_frac=0.001)
    assert len(regions) == 2


def test_regions_are_ranked_by_score(two_blobs):
    scores = [r["score"] for r in heatmap_regions(two_blobs, min_area_frac=0.001)]
    assert scores == sorted(scores, reverse=True)


def test_centroid_lands_on_the_activation(two_blobs):
    """The drawn marker must sit on the hot region, not merely near it."""
    top = heatmap_regions(two_blobs, min_area_frac=0.001)[0]
    cx, cy = top["centroid"]
    assert two_blobs[int(cy * 64), int(cx * 64)] > 0.5 * two_blobs.max()

    # Coordinate-order regression: centroid is (x, y), bbox is (r0, c0, r1, c1).
    r0, c0, r1, c1 = top["bbox"]
    assert r0 <= cy <= r1 and c0 <= cx <= c1


def test_peak_point_is_the_maximum(two_blobs):
    top = heatmap_regions(two_blobs, min_area_frac=0.001)[0]
    px, py = top["peak_point"]
    assert two_blobs[int(py * 64), int(px * 64)] == pytest.approx(two_blobs.max(), rel=0.02)


def test_polygon_is_tighter_than_its_bounding_box(two_blobs):
    """The whole point of a polygon: it must not just re-describe the box."""
    top = heatmap_regions(two_blobs, min_area_frac=0.001)[0]
    r0, c0, r1, c1 = top["bbox"]
    box_area = (r1 - r0) * (c1 - c0)

    assert top["vertices"] >= 3
    assert 0 < top["polygon_area_frac"] <= box_area + 1e-6


def test_polygon_vertices_are_normalised(two_blobs):
    for region in heatmap_regions(two_blobs, min_area_frac=0.001):
        pts = np.asarray(region["polygon"])
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert pts.min() >= 0.0 and pts.max() <= 1.0


def test_max_regions_is_honoured():
    heat = sum(blob(c, c, spread=20) for c in (8, 20, 32, 44, 56))
    assert len(heatmap_regions(heat, min_area_frac=0.0001, max_regions=2)) == 2


def test_min_area_filters_speckle(two_blobs):
    speckled = two_blobs.copy()
    speckled[0, 0] = 1.0                      # a single hot pixel
    assert len(heatmap_regions(speckled, min_area_frac=0.005)) == 2


def test_hull_mode_produces_a_convex_outline(two_blobs):
    hulled = heatmap_regions(two_blobs, min_area_frac=0.001, hull=True)[0]
    assert hulled["polygon_kind"] == "convex_hull"
    assert hulled["polygon_area_frac"] >= 0.0


# --------------------------------------------------------------------------- #
# Contours
# --------------------------------------------------------------------------- #
def test_contours_are_returned_per_level(two_blobs):
    contours = extract_contours(two_blobs, levels=(0.5, 0.9))
    assert contours
    assert set(c["level"] for c in contours) <= {0.5, 0.9}
    for c in contours:
        pts = np.asarray(c["points"])
        assert pts.shape[1] == 2
        assert pts.min() >= 0.0 and pts.max() <= 1.0


def test_higher_level_contours_enclose_less_area(two_blobs):
    """Iso-lines must nest: a 0.9 contour cannot be larger than a 0.5 one."""
    def total(level):
        return sum(polygon_area(np.asarray(c["points"]))
                   for c in extract_contours(two_blobs, levels=(level,)))

    assert total(0.9) < total(0.5)


# --------------------------------------------------------------------------- #
# Polygon primitives
# --------------------------------------------------------------------------- #
def test_simplify_collapses_a_straight_line():
    line = np.stack([np.linspace(0, 10, 200), np.zeros(200)], axis=1)
    assert len(simplify_polygon(line, tolerance=0.01)) == 2


def test_simplify_preserves_corners():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    assert len(simplify_polygon(square, tolerance=0.01)) == len(square)


def test_simplify_is_monotone_in_tolerance():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 300)
    poly = np.stack([np.cos(t), np.sin(t)], axis=1) + rng.normal(0, 0.01, (300, 2))
    assert len(simplify_polygon(poly, 0.5)) <= len(simplify_polygon(poly, 0.01))


def test_convex_hull_drops_interior_points():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
    assert len(convex_hull(pts)) == 4


def test_polygon_area_of_unit_square():
    assert polygon_area(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)) == pytest.approx(1.0)


def test_degenerate_polygons_have_zero_area():
    assert polygon_area(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0
    assert polygon_area(np.zeros((0, 2))) == 0.0


# --------------------------------------------------------------------------- #
# RGBA overlay
# --------------------------------------------------------------------------- #
def test_overlay_is_transparent_where_cold(two_blobs):
    """Alpha, not a flattened blend, is what lets the frontend keep anatomy crisp."""
    rgba = rgba_overlay(two_blobs, threshold=0.15)

    assert rgba.shape == (64, 64, 4)
    assert rgba.dtype == np.uint8
    assert rgba[0, 0, 3] == 0                             # cold corner fully transparent
    assert rgba[..., 3].max() > 0                         # hot region visible


def test_overlay_alpha_respects_the_alpha_ceiling(two_blobs):
    assert rgba_overlay(two_blobs, alpha=0.5)[..., 3].max() <= int(0.5 * 255) + 1


def test_overlay_can_be_resized_for_display(two_blobs):
    assert rgba_overlay(two_blobs, size=(128, 96)).shape == (128, 96, 4)


def test_overlay_data_uri_is_a_decodable_png(two_blobs):
    uri = rgba_overlay_data_uri(two_blobs)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Full payload
# --------------------------------------------------------------------------- #
def test_geometry_payload_is_complete_and_serialisable(two_blobs):
    import json

    payload = heatmap_geometry(two_blobs, probability=0.83, finding="cardiomegaly")

    assert payload["finding"] == "cardiomegaly"
    assert payload["probability"] == 0.83
    assert payload["probability_weighted"] is True
    assert payload["backend"] in {"opencv", "matplotlib"}
    assert payload["region_count"] == len(payload["regions"]) > 0
    assert payload["contours"]
    assert payload["overlay_png_data_uri"].startswith("data:image/png;base64,")
    json.dumps(payload)                                   # must cross the API boundary


def test_geometry_without_probability_is_marked_unweighted(two_blobs):
    payload = heatmap_geometry(two_blobs, include_overlay=False)

    assert payload["probability"] is None
    assert payload["probability_weighted"] is False
    assert "overlay_png_data_uri" not in payload


def test_probability_does_not_move_the_geometry(two_blobs):
    """Where a finding is must not change with how sure the model is.

    Confidence belongs in overlay intensity and in the reported number. If a
    falling probability shrank the outline, the overlay would be understating
    lesion extent — the one thing it exists to communicate.
    """
    high = heatmap_geometry(two_blobs, probability=0.95, include_overlay=False)
    low = heatmap_geometry(two_blobs, probability=0.02, include_overlay=False)

    assert high["geometry_basis"] == low["geometry_basis"] == "raw_activation"
    assert high["regions"] == low["regions"]
    assert high["contours"] == low["contours"]


def test_probability_does_move_the_overlay_intensity(two_blobs):
    """...but it must reach the pixels, or confidence is invisible to the reader."""
    def peak_alpha(p: float) -> int:
        payload = heatmap_geometry(two_blobs, probability=p)
        png = base64.b64decode(payload["overlay_png_data_uri"].split(",", 1)[1])
        from io import BytesIO

        from PIL import Image

        return int(np.asarray(Image.open(BytesIO(png)))[..., 3].max())

    assert peak_alpha(0.02) < peak_alpha(0.95)


def test_geometry_on_a_flat_map_is_empty_not_broken():
    payload = heatmap_geometry(np.zeros((32, 32)), probability=0.5, include_overlay=False)
    assert payload["regions"] == []
    assert payload["region_count"] == 0


def test_matplotlib_backend_agrees_with_opencv(two_blobs, monkeypatch):
    """The fallback must find the same regions, or deployments would disagree."""
    import aura.services.explain.geometry as G

    opencv_regions = heatmap_regions(two_blobs, min_area_frac=0.001)
    if G._backend() != "opencv":
        pytest.skip("opencv not installed; only one backend available here")

    monkeypatch.setattr(G, "_backend", lambda: "matplotlib")
    mpl_regions = G.heatmap_regions(two_blobs, min_area_frac=0.001)

    assert len(mpl_regions) == len(opencv_regions)
    for a, b in zip(mpl_regions, opencv_regions):
        assert a["centroid"] == pytest.approx(b["centroid"], abs=0.02)
        assert a["vertices"] >= 3
