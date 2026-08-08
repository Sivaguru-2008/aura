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


def _manuscript_text():
    """The v4 manuscript, normalised for numeric matching."""
    manuscript = REPO_ROOT / "docs" / "RESEARCH_PAPER_v4.md"
    if not manuscript.exists():
        pytest.skip(f"{manuscript} not present")
    text = manuscript.read_text(encoding="utf-8")
    return text.replace("**", "").replace("_", "").replace("−", "-").replace("—", "-")


def _row(text, name):
    """The markdown table row whose first cell is exactly ``name``."""
    hits = [ln for ln in text.splitlines()
            if ln.strip().startswith("|") and len(ln.split("|")) > 1
            and ln.split("|")[1].strip() == name]
    assert hits, f"no table row named {name!r} in the manuscript"
    return hits[0]


def _any_fmt(row, value, label):
    """Assert ``value`` appears in ``row`` at 2, 3 or 4 decimal places, signed or not."""
    sign = "+" if value > 0 else ""
    cands = [f"{sign}{value:.4f}", f"{sign}{value:.3f}", f"{sign}{value:.2f}",
             f"{value:.4f}", f"{value:.3f}", f"{value:.2f}", f"{value:g}"]
    assert any(c in row for c in cands), (
        f"{label}: none of {cands} found in row -> {row.strip()}"
    )


def test_manuscript_table_8c_matches_the_7qubit_artifact():
    """Table 8c is the controlled ablation. It carries the paper's one surviving
    entanglement claim, so it must track its artifact exactly."""
    f = PKG_ROOT / "artifacts" / "entanglement_multiseed_7q.json"
    assert f.exists(), (
        "entanglement_multiseed_7q.json is missing. Section 8.3.1 is the controlled "
        "version of the headline ablation; without the artifact the claim is unbacked."
    )
    data = json.loads(f.read_text(encoding="utf-8"))
    text = _manuscript_text()
    section = text.split("#### 8.3.1")[1].split("###")[0]

    for key, name in (("delta_nll", "Negative log-likelihood"),
                      ("delta_accuracy", "Accuracy"),
                      ("delta_ece", "Expected calibration error")):
        m = data[key]
        row = _row(section, name)
        _any_fmt(row, m["mean"], f"Table 8c {name} mean")
        _any_fmt(row, m["sd_across_seeds"], f"Table 8c {name} sd")
        _any_fmt(row, m["wilcoxon_p"], f"Table 8c {name} Wilcoxon p")
        _any_fmt(row, m["wilcoxon_p_holm"], f"Table 8c {name} Holm p")
        seeds = m["n_seeds_favouring_entanglement"]
        assert f"{seeds} of 10" in row, (
            f"Table 8c {name}: artifact says {seeds} of 10 seeds favour entanglement"
        )


def test_manuscript_table_3_matches_the_kernel_split():
    """Table 3 and Section 8.5 disagreed about the grading split across two
    revisions because nothing checked Table 3. This checks Table 3."""
    f = PKG_ROOT / "artifacts" / "brain" / "qkl_training_report.json"
    if not f.exists():
        pytest.skip("qkl_training_report.json not present")
    split = json.loads(f.read_text(encoding="utf-8"))["split"]
    row = _row(_manuscript_text(), "Quantum kernel grading (§8.5)")
    for subj, sl in (("train_subjects", "train_slices_used"),
                     ("cal_subjects", "cal_slices_used"),
                     ("test_subjects", "test_slices_used")):
        assert f"{split[subj]} subjects" in row, (
            f"Table 3 does not publish {split[subj]} for {subj}: {row.strip()}"
        )
        assert f"{split[sl]:,}" in row, (
            f"Table 3 does not publish {split[sl]:,} for {sl}: {row.strip()}"
        )


def test_abstract_agrees_with_the_results_sections():
    """The abstract carried a superseded kernel result while Section 8.5 carried the
    current one. Both are read from artifacts here so they cannot diverge again."""
    text = _manuscript_text()
    abstract = text.split("## Abstract")[1].split("## 1.")[0]

    f7 = PKG_ROOT / "artifacts" / "entanglement_multiseed_7q.json"
    if f7.exists():
        nll = json.loads(f7.read_text(encoding="utf-8"))["delta_nll"]
        assert f"+{nll['mean']:.3f}" in abstract or f"+{nll['mean']:.4f}" in abstract, (
            "the abstract does not report the controlled 7-qubit NLL effect"
        )
        assert f"{nll['wilcoxon_p']:.3f}" in abstract, (
            "the abstract does not report the controlled 7-qubit Wilcoxon p"
        )

    fq = PKG_ROOT / "artifacts" / "brain" / "qkl_training_report.json"
    if fq.exists():
        d = json.loads(fq.read_text(encoding="utf-8"))["quantum_minus_classical_auroc"]
        sign = "+" if d["mean"] > 0 else "-"
        assert f"{sign}{abs(d['mean']):.4f}" in abstract, (
            f"the abstract does not report the current kernel delta "
            f"{d['mean']:+.4f}; it may be quoting a superseded run"
        )
        p = d["p_two_sided"]
        assert f"{p:.2f}" in abstract or f"{p:.3f}" in abstract, (
            "the abstract does not report the current kernel p-value"
        )


def test_murphy_decomposition_reconstructs_the_brier_score():
    """A decomposition that does not reconstruct the score it decomposes cannot
    support a claim about which of its terms dominates. The predecessor of this
    artifact reported terms that were off by more than twice the effect."""
    f = PKG_ROOT / "artifacts" / "murphy_decomposition.json"
    if not f.exists():
        pytest.skip("murphy_decomposition.json not present")
    data = json.loads(f.read_text(encoding="utf-8"))
    for width, block in data["by_width"].items():
        for entry in block["per_seed"]:
            for k, cell in entry["by_group_count"].items():
                for arm in ("ring", "product"):
                    d = cell[arm]
                    recon = d["reliability"] - d["resolution"] + d["uncertainty"]
                    assert abs(recon - d["reconstructed_brier"]) < 1e-4, (
                        f"{width} seed {entry['seed']} k={k} {arm}: REL-RES+UNC does not "
                        f"equal the reported reconstruction"
                    )
                    assert abs(d["actual_brier"] - recon - d["residual"]) < 1e-4, (
                        f"{width} seed {entry['seed']} k={k} {arm}: the residual does not "
                        f"close the identity actual = REL-RES+UNC + residual"
                    )


def test_manuscript_v4_tables_match_artifacts():
    """Verify that the primary tables in the manuscript (v4) match the stored artifacts."""
    manuscript = REPO_ROOT / "docs" / "RESEARCH_PAPER_v4.md"
    if not manuscript.exists():
         pytest.skip(f"{manuscript} not present")

    text = manuscript.read_text(encoding="utf-8")
    # Clean text to normalize formats
    clean_text = text.replace("**", "").replace("_", "").replace("−", "-").replace("—", "-")

    # 1. Validate Table 6 (Primary comparison on n=447 split)
    f_headline = PKG_ROOT / "artifacts" / "fusion_headline_n447.json"
    if f_headline.exists():
        headline_data = json.loads(f_headline.read_text(encoding="utf-8"))["backends"]
        for key, row_name in [("classical", "Classical product-of-experts"),
                              ("quantum", "Quantum variational circuit"),
                              ("learnable", "Learnable gated head")]:
            m = headline_data[key]
            row = [line for line in clean_text.splitlines() if line.strip().startswith("|") and len(line.split("|")) > 6 and line.split("|")[1].strip() == row_name][0]
            assert f"{m['temperature']:.3f}" in row or f"{m['temperature']:.2f}" in row or f"{m['temperature']:g}" in row, f"Temp mismatch for {row_name} in Table 6"
            assert f"{m['accuracy']:.4f}" in row or f"{m['accuracy']:.3f}" in row, f"Accuracy mismatch for {row_name} in Table 6"
            assert f"{m['nll']:.4f}" in row or f"{m['nll']:.3f}" in row, f"NLL mismatch for {row_name} in Table 6"
            assert f"{m['ece']:.4f}" in row or f"{m['ece']:.3f}" in row, f"ECE mismatch for {row_name} in Table 6"
            assert f"{m['macro_auroc']:.4f}" in row or f"{m['macro_auroc']:.3f}" in row, f"AUROC mismatch for {row_name} in Table 6"

    # 2. Validate Table 8 (Entanglement ablation across 10 seeds)
    f_multiseed = PKG_ROOT / "artifacts" / "entanglement_multiseed.json"
    if f_multiseed.exists():
        multiseed_data = json.loads(f_multiseed.read_text(encoding="utf-8"))
        for metric_key, row_name in [("delta_nll", "Negative log-likelihood"),
                                     ("delta_accuracy", "Accuracy"),
                                     ("delta_ece", "Expected calibration error")]:
            m = multiseed_data[metric_key]
            row = [line for line in clean_text.splitlines() if line.strip().startswith("|") and len(line.split("|")) > 6 and line.split("|")[1].strip() == row_name][0]
            sign = "+" if m["mean"] > 0 else ""
            assert f"{sign}{m['mean']:.4f}" in row or f"{sign}{m['mean']:.3f}" in row or f"{sign}{m['mean']:g}" in row, f"Mean mismatch for {row_name} in Table 8"
            assert f"{m['sd_across_seeds']:.4f}" in row or f"{m['sd_across_seeds']:.3f}" in row or f"{m['sd_across_seeds']:g}" in row, f"SD mismatch for {row_name} in Table 8"
            assert f"{m['wilcoxon_p']:.4f}" in row or f"{m['wilcoxon_p']:.3f}" in row or f"{m['wilcoxon_p']:g}" in row, f"Wilcoxon p mismatch for {row_name} in Table 8"

    # 3. Validate Table 9 (Entangler topology at 8 qubits, 3 layers)
    f_sweep = PKG_ROOT / "artifacts" / "design_sweep.json"
    if f_sweep.exists():
        sweep_cells = json.loads(f_sweep.read_text(encoding="utf-8"))["cells"]
        target_cells = {c["entangler"]: c for c in sweep_cells if c["n_qubits"] == 8 and c["n_layers"] == 3}
        for ent, row_name in [("ring", "Ring (served)"), ("linear", "Linear"), ("full", "Full"), ("none", "None")]:
            c = target_cells[ent]
            row = [line for line in clean_text.splitlines() if line.strip().startswith("|") and len(line.split("|")) > 1 and line.split("|")[1].strip() == row_name][0]
            assert f"{c['accuracy']:.4f}" in row or f"{c['accuracy']:.3f}" in row, f"Accuracy mismatch for {row_name} in Table 9"
            assert f"{c['nll']:.4f}" in row or f"{c['nll']:.3f}" in row, f"NLL mismatch for {row_name} in Table 9"
            assert f"{c['ece']:.4f}" in row or f"{c['ece']:.3f}" in row, f"ECE mismatch for {row_name} in Table 9"

    # 4. Validate Table 11 & 12 (Quantum and classical kernel performance on BraTS)
    f_qkl = PKG_ROOT / "artifacts" / "brain" / "qkl_training_report.json"
    if f_qkl.exists():
        qkl_data = json.loads(f_qkl.read_text(encoding="utf-8"))
        section_85 = clean_text.split("### 8.5")[1].split("### 8.6")[0]
        for key, row_name in [("test_quantum", "Quantum kernel (IQP, 10 qubits)"),
                              ("test_classical_rbf", "Classical radial basis function")]:
            m = qkl_data[key]
            row = [line for line in section_85.splitlines() if line.strip().startswith("|") and len(line.split("|")) > 1 and line.split("|")[1].strip() == row_name][0]
            assert f"{m['auroc']:.4f}" in row or f"{m['auroc']:.3f}" in row, f"AUROC mismatch for {row_name} in Table 11"
            assert f"{m['accuracy']:.4f}" in row or f"{m['accuracy']:.3f}" in row, f"Accuracy mismatch for {row_name} in Table 11"
            assert f"{m['ece']:.4f}" in row or f"{m['ece']:.3f}" in row, f"ECE mismatch for {row_name} in Table 11"
            assert f"{m['brier']:.4f}" in row or f"{m['brier']:.3f}" in row, f"Brier mismatch for {row_name} in Table 11"

        for key, row_name in [("quantum_minus_classical_auroc", "AUROC"),
                              ("quantum_minus_classical_ece", "Expected calibration error")]:
            m = qkl_data[key]
            sign = "+" if m["mean"] > 0 else ""
            row = [line for line in section_85.splitlines() if line.strip().startswith("|") and len(line.split("|")) > 1 and line.split("|")[1].strip() == row_name][0]
            assert f"{sign}{m['mean']:.4f}" in row or f"{sign}{m['mean']:.3f}" in row or f"{sign}{m['mean']:g}" in row, f"Mean mismatch for {row_name} in Table 12"
            p_val = m["p_two_sided"]
            assert f"{p_val:.3f}" in row or f"{p_val:.2f}" in row or f"{p_val:g}" in row or (p_val < 0.001 and "< 0.001" in row), f"p-value mismatch for {row_name} in Table 12"
