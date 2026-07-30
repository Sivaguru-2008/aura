"""Heatmap geometry — turning a saliency map into drawable, measurable regions.

``services.explain.methods`` produces genuine Grad-CAM++ activation maps. This
module converts one into the shapes a clinician and a frontend can both use:

* **contours** at chosen intensity levels (iso-lines of the activation),
* **bounding polygons** — the convex or simplified outline of each hot region,
  which follows lesion shape instead of the axis-aligned box that has been the
  only localisation output so far,
* **probability-weighted masks** — activation scaled by the model's calibrated
  probability for that finding, so a 4%-probability finding cannot render as
  vividly as a 90% one,
* **RGBA overlays** — transparent PNGs the frontend composites over the image
  itself, rather than server-rendered flattened JPEGs.

Backends, in preference order: OpenCV (exact, gives nesting hierarchy), then
matplotlib's contour engine (a hard dependency, so always present). The chosen
backend is reported in every payload — geometry that silently changed algorithm
between deployments would be untraceable.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Literal

import numpy as np

from .overlays import _norm01, _resize, colorize

ContourBackend = Literal["opencv", "matplotlib"]


def _backend() -> ContourBackend:
    try:
        import cv2  # noqa: F401

        return "opencv"
    except Exception:
        return "matplotlib"


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def probability_weighted_mask(
    heat: np.ndarray,
    probability: float,
    gamma: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Scale a normalised activation map by the finding's calibrated probability.

    A Grad-CAM++ map is normalised to [0,1] *per finding*, so a finding the model
    puts at 4% renders exactly as hot as one it puts at 95%. That is visually
    dishonest: it invites a reader to weight a near-negative finding as strongly
    as a confident one. Multiplying through by the calibrated probability makes
    display intensity track evidence.

    ``gamma`` shapes the weighting (``>1`` suppresses low-probability findings
    harder); ``floor`` keeps a faint trace visible so a region is not lost
    entirely.
    """
    h = _norm01(heat)
    p = float(np.clip(probability, 0.0, 1.0)) ** float(max(gamma, 1e-6))
    return np.clip(h * max(p, float(floor)), 0.0, 1.0)


def threshold_mask(
    heat: np.ndarray,
    thresh_rel: float = 0.5,
    thresh_abs: float | None = None,
) -> np.ndarray:
    """Binary mask of the hot region.

    ``thresh_rel`` is relative to the map's own maximum, which adapts to dynamic
    range; ``thresh_abs``, when given, wins and is absolute in [0,1] — needed when
    masks from different findings must be compared on one scale.
    """
    h = _norm01(heat)
    if thresh_abs is not None:
        return h >= float(thresh_abs)
    peak = float(h.max())
    return h >= (thresh_rel * peak) if peak > 0 else np.zeros_like(h, dtype=bool)


def otsu_threshold(heat: np.ndarray) -> float:
    """Otsu's between-class-variance threshold, for maps with no natural cut point."""
    h = _norm01(heat).ravel()
    hist, edges = np.histogram(h, bins=256, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.5
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return 0.5
    cum = np.cumsum(hist * centres)
    m0 = np.divide(cum, w0, out=np.zeros_like(cum), where=w0 > 0)
    m1 = np.divide(cum[-1] - cum, w1, out=np.zeros_like(cum), where=w1 > 0)
    variance = w0 * w1 * (m0 - m1) ** 2
    variance[~valid] = -1.0
    return float(centres[int(np.argmax(variance))])


# --------------------------------------------------------------------------- #
# Contours
# --------------------------------------------------------------------------- #
def extract_contours(
    heat: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.7, 0.9),
    normalized: bool = True,
    min_points: int = 4,
) -> list[dict[str, Any]]:
    """Iso-intensity contours of a saliency map.

    Returns one entry per closed curve: ``{"level", "points", "closed",
    "length"}``. ``points`` are ``(x, y)`` pairs — image convention, column
    first — normalised to [0,1] by default so a frontend can scale them to any
    render size without knowing the map's resolution.
    """
    h = _norm01(heat)
    H, W = h.shape
    out: list[dict[str, Any]] = []
    backend = _backend()

    for level in levels:
        mask = (h >= float(level)).astype(np.uint8)
        if not mask.any():
            continue
        for pts in _contours_for_mask(mask, backend):
            if len(pts) < min_points:
                continue
            arr = np.asarray(pts, dtype=float)
            if normalized:
                arr = arr / np.asarray([max(W - 1, 1), max(H - 1, 1)], dtype=float)
            out.append({
                "level": round(float(level), 4),
                "points": [[round(float(x), 5), round(float(y), 5)] for x, y in arr],
                "closed": True,
                "length": int(len(arr)),
            })
    return out


def _contours_for_mask(mask: np.ndarray, backend: ContourBackend) -> list[np.ndarray]:
    """Extract (x, y) boundary polylines from a binary mask."""
    if backend == "opencv":
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 3]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        cs = plt.contour(mask.astype(float), levels=[0.5])
        polys: list[np.ndarray] = []
        for path in cs.get_paths():
            verts = np.asarray(path.vertices, dtype=float)
            if len(verts) >= 3:
                polys.append(verts)            # already (x, y)
        return polys
    finally:
        plt.close(fig)


def simplify_polygon(points: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification.

    A raw contour can carry hundreds of vertices — too many to ship per finding
    per study. Tolerance is in the same units as ``points``.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts

    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a, b = pts[start], pts[end]
        seg = b - a
        seg_len = float(np.hypot(*seg))
        span = pts[start + 1 : end]
        if seg_len < 1e-12:
            dist = np.hypot(*(span - a).T)
        else:
            # Perpendicular distance from each interior point to segment a->b.
            # The 2-D cross product is written out because numpy 2 removed the
            # 2-vector overload of np.cross.
            rel = span - a
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
        i = int(np.argmax(dist))
        if dist[i] > tolerance:
            idx = start + 1 + i
            keep[idx] = True
            stack.extend([(start, idx), (idx, end)])

    return pts[keep]


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) — the tightest convex outline of a region."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross2(o, a, b) -> float:
        """2-D cross product; written out because numpy 2 removed the overload."""
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and cross2(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    return np.asarray(half(pts)[:-1] + half(pts[::-1])[:-1])


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area. Always non-negative (winding order is not meaningful here)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# --------------------------------------------------------------------------- #
# Region geometry
# --------------------------------------------------------------------------- #
def heatmap_regions(
    heat: np.ndarray,
    thresh_rel: float = 0.5,
    thresh_abs: float | None = None,
    min_area_frac: float = 0.005,
    max_regions: int = 5,
    simplify_tolerance: float = 0.01,
    hull: bool = False,
) -> list[dict[str, Any]]:
    """Per-connected-component geometry: box, polygon, centroid, and intensity stats.

    This is the payload the frontend draws and the report quotes. Coordinates are
    normalised to [0,1]; boxes stay in the historical ``(r0, c0, r1, c1)`` row/col
    order used by :func:`overlays.heatmap_bboxes`, while polygons are ``(x, y)``
    for direct SVG/canvas use. Both are labelled in the payload so the difference
    cannot be misread.
    """
    h = _norm01(heat)
    H, W = h.shape
    mask = threshold_mask(h, thresh_rel=thresh_rel, thresh_abs=thresh_abs)
    if not mask.any():
        return []

    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask)
    except Exception:
        from .overlays import _label_numpy

        labels, n = _label_numpy(mask)

    min_area = max(1, int(min_area_frac * H * W))
    regions: list[dict[str, Any]] = []

    for lab in range(1, n + 1):
        component = labels == lab
        ys, xs = np.where(component)
        if ys.size < min_area:
            continue

        r0, r1 = int(ys.min()), int(ys.max() + 1)
        c0, c1 = int(xs.min()), int(xs.max() + 1)
        values = h[ys, xs]

        polys = _contours_for_mask(component.astype(np.uint8), _backend())
        if polys:
            outline = max(polys, key=len)
            if hull:
                outline = convex_hull(outline)
            outline = simplify_polygon(
                outline, tolerance=simplify_tolerance * max(H, W)
            )
            norm_poly = outline / np.asarray([max(W - 1, 1), max(H - 1, 1)], dtype=float)
        else:
            norm_poly = np.zeros((0, 2))

        # Intensity-weighted centroid — the peak of evidence, not of area.
        weight = values.sum()
        cy = float((ys * values).sum() / weight) if weight > 0 else float(ys.mean())
        cx = float((xs * values).sum() / weight) if weight > 0 else float(xs.mean())
        peak_idx = int(np.argmax(values))

        regions.append({
            "bbox": [round(r0 / H, 4), round(c0 / W, 4), round(r1 / H, 4), round(c1 / W, 4)],
            "bbox_order": "r0,c0,r1,c1 (row/col, normalized)",
            "polygon": [[round(float(x), 5), round(float(y), 5)] for x, y in norm_poly],
            "polygon_order": "x,y (col/row, normalized)",
            "polygon_kind": "convex_hull" if hull else "simplified_contour",
            "centroid": [round(cx / W, 5), round(cy / H, 5)],
            "peak_point": [round(float(xs[peak_idx]) / W, 5), round(float(ys[peak_idx]) / H, 5)],
            "peak": round(float(values.max()), 4),
            "mean_activation": round(float(values.mean()), 4),
            "score": round(float(values.sum()), 4),
            "area_frac": round(float(ys.size) / (H * W), 5),
            "polygon_area_frac": round(polygon_area(norm_poly), 5) if len(norm_poly) else 0.0,
            "vertices": int(len(norm_poly)),
        })

    regions.sort(key=lambda r: -r["score"])
    return regions[:max_regions]


# --------------------------------------------------------------------------- #
# Transparent overlays for the frontend
# --------------------------------------------------------------------------- #
def rgba_overlay(
    heat: np.ndarray,
    cmap: str = "turbo",
    alpha: float = 0.65,
    threshold: float = 0.15,
    size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Colorized heatmap as (H, W, 4) uint8 RGBA, transparent where cold.

    Alpha rises with activation and is zero below ``threshold``, so the frontend
    composites this straight over the original image: the anatomy stays crisp and
    the user can toggle or fade the overlay client-side. A flattened server-side
    JPEG can do none of that.

    ``normalize`` must be ``False`` for a map whose absolute scale already carries
    meaning — a probability-weighted map above all. Rescaling one to full range
    would restore a 2%-probability finding to maximum brightness and silently
    undo the weighting.
    """
    h = _norm01(heat) if normalize else np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    if size is not None:
        resized = _resize(h, size)
        h = _norm01(resized) if normalize else np.clip(resized, 0.0, 1.0)
    # Colour always spans the full ramp so the palette stays comparable between
    # studies; only alpha carries the absolute magnitude.
    rgb = colorize(_norm01(h) if h.max() > 0 else h, cmap)
    a = np.clip((h - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0) * float(alpha)
    return np.dstack([rgb, (a * 255).astype(np.uint8)])


def rgba_overlay_png(heat: np.ndarray, **kwargs) -> bytes:
    """:func:`rgba_overlay` encoded as PNG bytes (alpha preserved)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba_overlay(heat, **kwargs), mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def rgba_overlay_data_uri(heat: np.ndarray, **kwargs) -> str:
    """PNG data URI, directly assignable to an ``<img src>``."""
    return "data:image/png;base64," + base64.b64encode(rgba_overlay_png(heat, **kwargs)).decode("ascii")


# --------------------------------------------------------------------------- #
# Full payload
# --------------------------------------------------------------------------- #
def heatmap_geometry(
    heat: np.ndarray,
    probability: float | None = None,
    finding: str | None = None,
    levels: tuple[float, ...] = (0.5, 0.7, 0.9),
    thresh_rel: float = 0.5,
    use_otsu: bool = False,
    include_overlay: bool = True,
    overlay_size: tuple[int, int] | None = None,
    max_regions: int = 5,
) -> dict[str, Any]:
    """Everything the API and the frontend need for one finding's heatmap.

    Probability enters the payload in exactly one place, deliberately:

    * **Geometry** (regions and contours) is extracted from the *raw* activation.
      Where a finding is does not change because the model is unsure that it is
      there — shrinking the outline with falling confidence would misstate the
      lesion's extent, which is the one thing a localisation overlay exists to
      convey.
    * **Overlay intensity** is probability-weighted. Confidence is carried by how
      strongly the region is painted (and by the reported number), so a 4%
      finding cannot render as vividly as a 95% one.

    ``probability_weighted`` in the payload refers to the overlay only, and
    ``geometry_basis`` records which map the shapes came from.
    """
    raw = _norm01(heat)
    threshold = otsu_threshold(raw) if use_otsu else None

    payload: dict[str, Any] = {
        "finding": finding,
        "probability": round(float(probability), 4) if probability is not None else None,
        "shape": list(raw.shape),
        "backend": _backend(),
        "threshold": {
            "mode": "otsu" if use_otsu else "relative",
            "relative": thresh_rel,
            "absolute": round(float(threshold), 4) if threshold is not None else None,
        },
        "geometry_basis": "raw_activation",
        "probability_weighted": probability is not None,   # applies to the overlay
        "peak_activation": round(float(raw.max()), 4),
        "regions": heatmap_regions(
            raw, thresh_rel=thresh_rel, thresh_abs=threshold, max_regions=max_regions
        ),
        "contours": extract_contours(raw, levels=levels),
    }
    payload["region_count"] = len(payload["regions"])
    if include_overlay:
        display = (
            probability_weighted_mask(raw, probability) if probability is not None else raw
        )
        payload["overlay_png_data_uri"] = rgba_overlay_data_uri(
            display, size=overlay_size, normalize=probability is None
        )
    return payload
