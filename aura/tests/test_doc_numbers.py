"""The published benchmark tables must equal the artifact `bench` writes.

AURA's central claim is that its numbers are *regenerated, not asserted*. That claim is
only worth something if the tables a judge reads in README.md / WHAT_IS_AURA.md /
docs/BENCHMARKS.md still agree with `artifacts/benchmark.json`. They drifted once —
three docs carried an older run's accuracies while the artifact (and the model cards
derived from it) carried the current ones. This test is the guard against that.

If this fails, the fix is to re-copy the numbers out of `benchmark.json`, never to
loosen the tolerance.
"""
import json
import re
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]      # .../aura-main/aura
REPO_ROOT = PKG_ROOT.parent                          # .../aura-main
BENCHMARK = PKG_ROOT / "artifacts" / "benchmark.json"

# Markdown table row label -> key in benchmark.json["metrics_full"]
BACKEND_LABELS = {
    "quantum vqc": "quantum",
    "classical poe": "classical",
    "learnable head": "learnable",
    "ensemble": "ensemble",
}

# Every published table uses this column order.
COLUMNS = ("accuracy", "ece", "macro_auroc")

DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "WHAT_IS_AURA.md",
    REPO_ROOT / "docs" / "BENCHMARKS.md",
]

# Only match probabilities, so "(8-qubit)" and "n=69" in a label are never read as data.
NUMBER = re.compile(r"0\.\d{3,4}")

# Confidence intervals are published as "[0.581, 0.795]" next to the point estimates.
# They are derived from the accuracy column rather than stored in benchmark.json, so
# they must not be read as extra metric columns — strip the bracketed span first.
INTERVAL = re.compile(r"\[[^\]]*\]")


def _metrics():
    if not BENCHMARK.exists():
        pytest.skip(f"{BENCHMARK} not present; run `py -m aura.aura_cli bench` first")
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))["metrics_full"]


def _published_rows(doc: Path):
    """Yield (line_number, backend_key, [floats]) for each benchmark row in `doc`."""
    for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        label = cells[0].lower()
        for text, key in BACKEND_LABELS.items():
            if label.startswith(text):
                row = INTERVAL.sub(" ", " ".join(cells[1:]))
                values = [float(v) for v in NUMBER.findall(row)]
                yield lineno, key, values
                break


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_published_benchmark_tables_match_the_artifact(doc):
    assert doc.exists(), f"{doc} is missing"
    metrics = _metrics()

    rows = list(_published_rows(doc))
    assert rows, f"{doc.name} has no recognisable benchmark table rows"

    for lineno, backend, values in rows:
        truth = metrics[backend]
        expected = [truth[c] for c in COLUMNS]
        assert len(values) == len(expected), (
            f"{doc.name}:{lineno} ({backend}) has {len(values)} numbers, "
            f"expected {len(expected)} for columns {COLUMNS}"
        )
        for column, published, actual in zip(COLUMNS, values, expected):
            assert published == pytest.approx(actual, abs=5e-5), (
                f"{doc.name}:{lineno} publishes {backend}.{column} = {published}, "
                f"but artifacts/benchmark.json says {actual}. "
                f"Re-copy from benchmark.json; do not loosen this test."
            )


def test_every_backend_in_the_artifact_is_published_somewhere():
    """A backend that quietly stops being reported is its own kind of drift."""
    metrics = _metrics()
    published = {
        backend
        for doc in DOCS
        if doc.exists()
        for _, backend, _ in _published_rows(doc)
    }
    missing = set(metrics) - published
    assert not missing, f"benchmark.json reports {sorted(missing)} but no doc publishes it"


def test_conclusion_still_holds_classical_beats_quantum_on_accuracy():
    """The prose in all three docs says the classical head wins a fair fight.

    If a retrain ever flips this, the *words* need changing too — not just the table.
    """
    metrics = _metrics()
    assert metrics["classical"]["accuracy"] > metrics["quantum"]["accuracy"], (
        "Quantum now beats classical on accuracy. The published narrative "
        "('does not beat a fairly-calibrated classical head') is now wrong — "
        "update the prose in README.md, WHAT_IS_AURA.md and docs/BENCHMARKS.md."
    )


# --------------------------------------------------------------------------- #
# Attribution: a number nobody measured must say so
# --------------------------------------------------------------------------- #
BENCH_REPORT = REPO_ROOT / "docs" / "benchmark_report.md"

# Models AURA compares itself against but has never run. Every mention has to sit
# near an explicit disclaimer, or the table reads as a head-to-head that happened.
UNMEASURED_BASELINES = ("nnU-Net", "nnUNet", "SwinUNETR", "MONAI")

ATTRIBUTION_MARKERS = ("not measured", "not run", "published", "literature")


def test_competitor_rows_are_marked_as_unmeasured():
    """nnU-Net / SwinUNETR / MONAI numbers are literature values, not experiments.

    They were once bare literals in the same table as AURA's interpolated rows,
    under prose describing a comparison "against industry-standard baseline models".
    Nothing in the repo runs them. If they are cited, they must be *labelled* cited.
    """
    if not BENCH_REPORT.exists():
        pytest.skip(f"{BENCH_REPORT} not generated; run ml/evaluation/run_pipeline.py")
    text = BENCH_REPORT.read_text(encoding="utf-8")

    for line in text.splitlines():
        if not any(b in line for b in UNMEASURED_BASELINES):
            continue
        if line.strip().startswith("|") and "---" not in line:
            assert any(m in line.lower() for m in ATTRIBUTION_MARKERS), (
                f"benchmark_report.md publishes a competitor row with no provenance:\n"
                f"  {line.strip()}\n"
                f"Every such row must name its source and say it was not measured here."
            )


def test_benchmark_report_discloses_the_2d_3d_non_equivalence():
    """AURA's brain Dice is pooled 2-D; the cited baselines are per-case 3-D.

    Pooled-2-D scoring is systematically more generous — empty slices are easy and
    numerous — so presenting the two in one table without the caveat overstates AURA.
    """
    if not BENCH_REPORT.exists():
        pytest.skip(f"{BENCH_REPORT} not generated; run ml/evaluation/run_pipeline.py")
    text = BENCH_REPORT.read_text(encoding="utf-8").lower()
    assert "2-d" in text or "2d" in text, "no dimensionality caveat in benchmark_report.md"
    assert "per-case 3-d" in text or "per-case 3d" in text, (
        "benchmark_report.md must state that the cited baselines are per-case 3-D "
        "while AURA's figure is pooled per-slice 2-D."
    )


def test_fusion_gap_is_reported_with_its_uncertainty():
    """The headline accuracy gap is four cases out of sixty-nine.

    Publishing it bare implies a resolution the split does not have. The report has
    to carry the interval and the significance context next to the point estimates.
    """
    if not BENCH_REPORT.exists():
        pytest.skip(f"{BENCH_REPORT} not generated; run ml/evaluation/run_pipeline.py")
    text = BENCH_REPORT.read_text(encoding="utf-8").lower()
    assert "mcnemar" in text, "no significance test reported alongside the fusion gap"
    assert "95%" in text, "no confidence interval reported alongside the fusion gap"


def test_published_safety_thresholds_match_the_served_policy():
    """docs/KNOWN_LIMITATIONS.md must describe the envelope the app actually uses.

    It published epistemic 0.45 / OOD z 3.0 / conformal set > 3 long after those were
    recalibrated to 0.15 / 2.5 / > 4 — values the code comment in common/config.py
    explicitly calls superseded. The safety-boundary doc is the worst place to be
    stale, so the thresholds are pinned the same way the benchmark tables are.
    """
    import sys

    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
    from aura.common.config import get_settings

    doc = REPO_ROOT / "docs" / "KNOWN_LIMITATIONS.md"
    assert doc.exists(), f"{doc} is missing"
    text = doc.read_text(encoding="utf-8")
    s = get_settings()

    for label, value in (
        ("epistemic_threshold", f"{s.epistemic_threshold:g}"),
        ("ood_threshold", f"{s.ood_threshold:g}"),
        ("abstention_conformal_size", f"{s.abstention_conformal_size:g}"),
    ):
        assert f"`{value}`" in text, (
            f"KNOWN_LIMITATIONS.md does not publish the served {label} = {value}. "
            f"Re-copy from get_settings(); do not loosen this test."
        )
