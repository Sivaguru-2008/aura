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
                values = [float(v) for v in NUMBER.findall(" ".join(cells[1:]))]
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
