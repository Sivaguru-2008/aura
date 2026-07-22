"""Task 2 (trust) — OOD intake gate showcase.

Runs services/vision/xray_gate.validate_cxr on a real chest radiograph (accepted)
and on a battery of non-CXR probes (rejected with a named reason), then renders a
verdict contact sheet. The probes exercise the gate's distinct rejection paths
(color content, high edge/text density, solid/low-range, non-radiograph structure).

Outputs (isolated): artifacts/explain_demo/ood_gate.{png,json}
"""
from __future__ import annotations
import os
os.environ.setdefault("AURA_LABELER", "v2")
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common.config import ARTIFACTS
from mimic.config import get_mimic_paths
from mimic.parsing import safe_str_list
from services.vision.xray_gate import validate_cxr

OUT = ARTIFACTS / "explain_demo"
PROBE_DIR = OUT / "ood_probes"


def _real_cxr() -> Path:
    import pandas as pd
    mp = get_mimic_paths()
    df = pd.read_csv(mp.validate_csv, dtype=str)
    for _, r in df.iterrows():
        for rel in safe_str_list(r.get("image", "")):
            p = mp.images_root / rel
            if p.is_file():
                return p
    raise FileNotFoundError("no readable val CXR")


def _make_probes() -> list[tuple[str, Path]]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    probes = []

    # 1) Colour photo — saturated RGB blobs (violates grayscale gate)
    a = np.zeros((256, 256, 3), np.uint8)
    for _ in range(60):
        y, x = rng.integers(0, 256, 2)
        col = rng.integers(60, 255, 3)
        rr = rng.integers(10, 40)
        yy, xx = np.ogrid[:256, :256]
        a[((yy - y) ** 2 + (xx - x) ** 2) <= rr * rr] = col
    p = PROBE_DIR / "colour_photo.png"; Image.fromarray(a).save(p); probes.append(("colour photo", p))

    # 2) Screenshot / document — white bg, black text-like strokes (high edge density)
    img = Image.new("RGB", (256, 256), (245, 245, 245)); d = ImageDraw.Draw(img)
    for row in range(12, 240, 16):
        x = 12
        while x < 240:
            w = int(rng.integers(6, 26)); d.rectangle([x, row, x + w, row + 8], fill=(20, 20, 20)); x += w + 6
    p = PROBE_DIR / "screenshot_text.png"; img.save(p); probes.append(("screenshot / text", p))

    # 3) Solid / near-empty (fails dynamic range)
    a = np.full((256, 256), 30, np.uint8) + rng.integers(0, 3, (256, 256), dtype=np.uint8)
    p = PROBE_DIR / "solid.png"; Image.fromarray(a).save(p); probes.append(("near-solid", p))

    # 4) White noise (fails smoothness / structure)
    a = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    p = PROBE_DIR / "noise.png"; Image.fromarray(a).save(p); probes.append(("white noise", p))

    # 5) Colour logo-like gradient (colour + flat bands)
    yy, xx = np.mgrid[0:256, 0:256]
    a = np.stack([(xx).astype(np.uint8), (yy).astype(np.uint8),
                  (255 - xx).astype(np.uint8)], -1)
    p = PROBE_DIR / "gradient_logo.png"; Image.fromarray(a).save(p); probes.append(("colour gradient", p))

    return probes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    cxr = _real_cxr()
    g = validate_cxr(cxr)
    cases.append({"name": "real chest X-ray", "path": str(cxr), "ok": bool(g.ok),
                  "reason": g.reason or "accepted", "expect": "ACCEPT"})
    for name, p in _make_probes():
        g = validate_cxr(p)
        cases.append({"name": name, "path": str(p), "ok": bool(g.ok),
                      "reason": g.reason or "accepted", "expect": "REJECT"})

    n_ok = sum(1 for c in cases if c["ok"] == (c["expect"] == "ACCEPT"))
    summary = {"n_cases": len(cases), "n_as_expected": n_ok, "cases": cases}
    (OUT / "ood_gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(cases)
    for c in cases:
        verdict = "ACCEPT" if c["ok"] else "REJECT"
        print(f"  {c['name']:20s} -> {verdict:6s} (expected {c['expect']:6s})  {c['reason']}")
    print(f"[ood] {n_ok}/{len(cases)} verdicts as expected -> {OUT/'ood_gate.png'}")


def _plot(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4))
    for ax, c in zip(np.atleast_1d(axes), cases):
        try:
            im = Image.open(c["path"]).convert("L"); ax.imshow(np.asarray(im), cmap="gray")
        except Exception:
            ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])
        ok = c["ok"]; good = ok == (c["expect"] == "ACCEPT")
        color = "#22d3ee" if ok else "#e74c3c"
        edge = "#2ecc71" if good else "#e74c3c"
        for s in ax.spines.values():
            s.set_edgecolor(edge); s.set_linewidth(3)
        verdict = "ACCEPT" if ok else "REJECT"
        reason = c["reason"] if not ok else "passes all gates"
        # short reason (before the em-dash) wrapped to the panel width
        import textwrap
        short = reason.split("—")[0].split(" - ")[0].strip()
        wrapped = "\n".join(textwrap.wrap(short, width=22)) or short
        ax.set_title(f"{c['name']}\n{verdict}", color=color, fontsize=10, weight="bold")
        ax.text(0.5, -0.06, wrapped, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#555")
    fig.suptitle("AURA intake OOD gate — accepts radiographs, rejects non-CXR with a named reason",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "ood_gate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
