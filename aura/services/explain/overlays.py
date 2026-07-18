"""Visualization writers for clinical explainability.

Turns the normalized [0,1] saliency maps produced by ``services/explain`` into
the deliverables Step 3 asks for:

    * colorized heatmap overlays on the radiograph,
    * bounding-box localization (connected-component boxes around hot regions),
    * evidence visualization (bars of per-evidence attribution),
    * saved PNGs, high-resolution overlays, and a self-contained HTML report.

matplotlib (Agg) and scipy are imported lazily and only used here, so importing
this module never pulls a GUI backend into the serving path. Everything is written
under a caller-chosen directory (the pipeline uses ``artifacts/explain/``).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Core array ops
# --------------------------------------------------------------------------- #
def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _resize(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D map to ``size`` (rows, cols) with bilinear interpolation."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == size:
        return a
    try:
        import cv2

        return cv2.resize(a, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # numpy fallback: nearest-neighbour index mapping
        ri = np.linspace(0, a.shape[0] - 1, size[0]).astype(int)
        ci = np.linspace(0, a.shape[1] - 1, size[1]).astype(int)
        return a[np.ix_(ri, ci)]


def colorize(heat: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    """Map a [0,1] heatmap to an (H,W,3) uint8 RGB image via a matplotlib colormap."""
    import matplotlib

    matplotlib.use("Agg")

    h = _norm01(heat)
    try:                                    # matplotlib >= 3.9
        colormap = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):       # older API
        from matplotlib import cm

        colormap = cm.get_cmap(cmap)
    rgba = colormap(h)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(base_gray: np.ndarray, heat: np.ndarray, alpha: float = 0.45,
            cmap: str = "turbo") -> np.ndarray:
    """Blend a colorized heatmap over a grayscale radiograph. Returns (H,W,3) uint8."""
    base = _norm01(base_gray)
    heat_r = _resize(heat, base.shape)
    base_rgb = np.stack([base] * 3, axis=-1)
    heat_rgb = colorize(heat_r, cmap).astype(np.float32) / 255.0
    # Weight the heat by its own intensity so cold regions keep the anatomy visible.
    w = (alpha * _norm01(heat_r))[..., None]
    out = (1 - w) * base_rgb + w * heat_rgb
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Bounding-box localization
# --------------------------------------------------------------------------- #
def heatmap_bboxes(heat: np.ndarray, thresh_rel: float = 0.5,
                   min_area_frac: float = 0.01, max_boxes: int = 3) -> list[dict]:
    """Connected-component boxes around the hottest regions of a saliency map.

    Returns a list of dicts with normalized ``bbox`` (r0,c0,r1,c1 in [0,1]),
    ``score`` (summed saliency in the box), and ``peak`` (max saliency), ranked by
    score. Threshold is relative to the map max so it adapts to map dynamic range.
    """
    h = _norm01(heat)
    H, W = h.shape
    mask = h >= (thresh_rel * float(h.max()) if h.max() > 0 else 1.1)
    if not mask.any():
        return []
    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask)
    except Exception:
        labels, n = _label_numpy(mask)
    min_area = max(1, int(min_area_frac * H * W))
    boxes: list[dict] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        if ys.size < min_area:
            continue
        r0, r1 = ys.min(), ys.max() + 1
        c0, c1 = xs.min(), xs.max() + 1
        boxes.append({
            "bbox": (round(r0 / H, 4), round(c0 / W, 4), round(r1 / H, 4), round(c1 / W, 4)),
            "score": round(float(h[ys, xs].sum()), 4),
            "peak": round(float(h[ys, xs].max()), 4),
            "area_frac": round(float(ys.size) / (H * W), 4),
        })
    boxes.sort(key=lambda b: -b["score"])
    return boxes[:max_boxes]


def _label_numpy(mask: np.ndarray):
    """Tiny 4-connectivity connected-components fallback when scipy is absent."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
    return labels, cur


# --------------------------------------------------------------------------- #
# PNG writers
# --------------------------------------------------------------------------- #
def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def save_overlay_png(path: str | Path, base_gray: np.ndarray, heat: np.ndarray,
                     boxes: Optional[Iterable[dict]] = None, title: str = "",
                     dpi: int = 150) -> Path:
    """Write a heatmap-over-radiograph PNG with optional bounding boxes drawn on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ov = overlay(base_gray, heat)
    H, W = ov.shape[:2]
    fig, ax = plt.subplots(figsize=(W / dpi * 2.2, H / dpi * 2.2), dpi=dpi)
    ax.imshow(ov)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    for b in boxes or []:
        r0, c0, r1, c1 = b["bbox"]
        ax.add_patch(Rectangle((c0 * W, r0 * H), (c1 - c0) * W, (r1 - r0) * H,
                               fill=False, edgecolor="#00e5ff", linewidth=1.6))
        ax.text(c0 * W, max(0, r0 * H - 4), f"{b.get('peak', 0):.2f}",
                color="#00e5ff", fontsize=7)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def save_high_res_overlay(path: str | Path, base_gray: np.ndarray, heat: np.ndarray,
                          boxes: Optional[Iterable[dict]] = None, dpi: int = 300) -> Path:
    """High-resolution overlay at the radiograph's native size."""
    return save_overlay_png(path, base_gray, heat, boxes=boxes, dpi=dpi)


def save_raw_heatmap_png(path: str | Path, heat: np.ndarray, cmap: str = "turbo") -> Path:
    """Write the bare colorized heatmap (no anatomy) as a PNG."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize(heat, cmap)).save(path)
    return path


def save_evidence_bar_png(path: str | Path, attribution: dict[str, float],
                          title: str = "Evidence attribution", dpi: int = 130) -> Path:
    """Horizontal bar chart of per-evidence attribution to the top diagnosis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(attribution.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
    fig, ax = plt.subplots(figsize=(6, max(2.0, 0.4 * len(names))), dpi=dpi)
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Δ P(top diagnosis) if evidence removed")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _png_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _overlay_png_bytes(base_gray: np.ndarray, heat: np.ndarray,
                       boxes: Optional[Iterable[dict]] = None, title: str = "") -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ov = overlay(base_gray, heat)
    H, W = ov.shape[:2]
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=140)
    ax.imshow(ov)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    for b in boxes or []:
        r0, c0, r1, c1 = b["bbox"]
        ax.add_patch(Rectangle((c0 * W, r0 * H), (c1 - c0) * W, (r1 - r0) * H,
                               fill=False, edgecolor="#00e5ff", linewidth=1.6))
    return _fig_to_png_bytes(fig)


def save_explanation_html(path: str | Path, base_gray: np.ndarray,
                          method_maps: dict[str, np.ndarray], primary: str,
                          boxes: list[dict], target_finding: str,
                          findings: list[tuple[str, float]],
                          evidence_attribution: Optional[dict[str, float]] = None,
                          meta: Optional[dict] = None) -> Path:
    """Write a self-contained HTML explainability report (images embedded as data URIs)."""
    meta = meta or {}
    cards = []
    # Primary first, then the rest.
    ordered = [primary] + [m for m in method_maps if m != primary]
    for name in ordered:
        if name not in method_maps:
            continue
        b = boxes if name == primary else []
        uri = _png_data_uri(_overlay_png_bytes(base_gray, method_maps[name], b, name))
        cards.append(
            f'<div class="card"><img src="{uri}"/><div class="cap">{name}'
            f'{" (primary)" if name == primary else ""}</div></div>'
        )

    findings_rows = "".join(
        f"<tr><td>{n}</td><td>{p:.3f}</td>"
        f'<td><div class="bar"><div class="fill" style="width:{min(100, p*100):.0f}%"></div></div></td></tr>'
        for n, p in findings
    )
    box_rows = "".join(
        f"<tr><td>{i+1}</td><td>{b['bbox']}</td><td>{b['peak']}</td><td>{b['area_frac']}</td></tr>"
        for i, b in enumerate(boxes)
    ) or '<tr><td colspan="4">No region above threshold.</td></tr>'

    ev_html = ""
    if evidence_attribution:
        ev_uri = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                save_evidence_bar_png(tf.name, evidence_attribution)
                ev_uri = _png_data_uri(Path(tf.name).read_bytes())
                Path(tf.name).unlink(missing_ok=True)
        except Exception:
            ev_uri = None
        if ev_uri:
            ev_html = f'<h2>Evidence attribution</h2><img class="wide" src="{ev_uri}"/>'

    meta_html = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in meta.items())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>AURA Explainability — {target_finding}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0f14;color:#e6edf3;margin:0;padding:24px}}
 h1{{font-size:20px}} h2{{font-size:15px;margin-top:26px;border-bottom:1px solid #223;padding-bottom:6px}}
 .grid{{display:flex;flex-wrap:wrap;gap:14px}}
 .card{{background:#111823;border:1px solid #223;border-radius:10px;padding:8px}}
 .card img{{width:240px;border-radius:6px;display:block}}
 .cap{{text-align:center;font-size:12px;margin-top:6px;color:#9fb2c8}}
 table{{border-collapse:collapse;width:100%;max-width:640px;font-size:13px}}
 td,th{{border-bottom:1px solid #223;padding:6px 8px;text-align:left}}
 .bar{{background:#1c2733;border-radius:4px;height:10px;width:160px}}
 .fill{{background:linear-gradient(90deg,#22d3ee,#0ea5e9);height:10px;border-radius:4px}}
 ul{{color:#9fb2c8;font-size:13px}} img.wide{{max-width:640px;border-radius:8px}}
</style></head><body>
<h1>AURA Explainability Report — target finding: <span style="color:#22d3ee">{target_finding}</span></h1>
<ul>{meta_html}</ul>
<h2>Saliency overlays (primary = {primary})</h2>
<div class="grid">{''.join(cards)}</div>
<h2>Vision findings</h2>
<table><tr><th>finding</th><th>probability</th><th></th></tr>{findings_rows}</table>
<h2>Bounding-box localization ({target_finding})</h2>
<table><tr><th>#</th><th>bbox (r0,c0,r1,c1)</th><th>peak</th><th>area</th></tr>{box_rows}</table>
{ev_html}
</body></html>""", encoding="utf-8")
    return path
