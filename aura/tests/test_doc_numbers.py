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

    History of this test, because it is the reason the next one exists. It first
    caught a doc publishing epistemic 0.45 / OOD z 3.0 / conformal set > 3 after
    those were recalibrated. It was then written to compare the doc against
    ``Settings.epistemic_threshold`` (0.15) and ``Settings.ood_threshold`` (2.5) —
    neither of which the abstention path reads. Those two fields belong to the CDRE
    *controller's* policy profile; the abstention decision in
    ``services/safety/engine.py`` reads ``ood_energy_threshold`` (1.5) and
    ``epistemic_abstention_threshold`` (0.20). So the test passed for four revisions
    while the published envelope was wrong on two of its four criteria.

    It now asserts the four values the abstention path actually reads.
    """
    import sys

    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
    from aura.common.config import get_settings

    doc = REPO_ROOT / "docs" / "KNOWN_LIMITATIONS.md"
    assert doc.exists(), f"{doc} is missing"
    text = doc.read_text(encoding="utf-8")
    s = get_settings()

    # Exactly the four criteria SafetyEngine.assess() branches on, in its order.
    for label, value in (
        ("ood_energy_threshold", f"{s.ood_energy_threshold:g}"),
        ("low_confidence_threshold", f"{s.low_confidence_threshold:g}"),
        ("abstention_conformal_size", f"{s.abstention_conformal_size:g}"),
        ("epistemic_abstention_threshold", f"{s.epistemic_abstention_threshold:g}"),
    ):
        assert f"`{value}`" in text, (
            f"KNOWN_LIMITATIONS.md does not publish the served {label} = {value}. "
            f"Re-copy from get_settings(); do not loosen this test."
        )


def test_abstention_criteria_are_read_from_settings_not_literals():
    """The mechanism check the previous test could not perform.

    A test that compares a document against a *setting* is satisfied whether or not
    the serving path reads that setting. That is how the epistemic criterion stayed
    hardcoded at 0.20 through four published revisions of a 0.15 figure. Verifying
    the number is not the same as verifying the wiring, so this reads the source of
    the abstention branch and requires every criterion to come from ``self.settings``.

    If you add a fifth criterion, add it here too.
    """
    engine = PKG_ROOT / "services" / "safety" / "engine.py"
    assert engine.exists(), f"{engine} is missing"
    src = engine.read_text(encoding="utf-8")

    # The window starts at the OOD energy computation rather than at the branch,
    # because the first criterion is evaluated into `is_ood` a few lines above the
    # if/elif chain and would otherwise fall outside the check.
    start = src.index("e = energy_score(")
    end = src.index("return SafetyAssessment", start)
    branch = src[start:end]

    for setting in ("ood_energy_threshold", "low_confidence_threshold",
                    "abstention_conformal_size", "epistemic_abstention_threshold"):
        assert f"self.settings.{setting}" in branch, (
            f"the abstention branch does not read self.settings.{setting}. "
            f"Every criterion must come from configuration, or the published safety "
            f"envelope cannot be verified against the served one."
        )

    # No bare numeric comparison may survive in the branch: `> 0.20` is exactly the
    # defect this test exists to prevent recurring.
    literals = re.findall(r"[<>]=?\s*[0-9]+\.?[0-9]*", branch)
    assert not literals, (
        f"the abstention branch compares against numeric literals {literals}. "
        f"Move them into Settings so they are configurable and inspectable."
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


def test_manuscript_table_8e_matches_the_decomposition_artifact():
    """Table 8e is the evidence for withdrawing the calibration/refinement mechanism.
    Its whole point is that the terms move with the group count, so the published
    values have to track the artifact row by row or the argument evaporates."""
    f = PKG_ROOT / "artifacts" / "murphy_decomposition.json"
    if not f.exists():
        raise AssertionError(
            "murphy_decomposition.json is missing; run "
            "`py -m aura.ml.evaluation.murphy_decomp`. This check skipped for one "
            "revision while the manuscript reported the result as computed, which is "
            "why it now fails instead."
        )
    widths = json.loads(f.read_text(encoding="utf-8"))["by_width"]
    text = _manuscript_text()
    section = text.split("#### 8.3.2")[1].split("\n### ")[0]

    # Table 8e is the 8-qubit width, Table 8f the 7-qubit one. Both are three rows
    # keyed by group count, so split the section at 8f to keep the rows apart.
    parts = section.split("Table 8f. Murphy decomposition")
    assert len(parts) == 2, "Section 8.3.2 should contain both Table 8e and Table 8f"
    for width, table, chunk in (("8q", "8e", parts[0]), ("7q", "8f", parts[1])):
        block = widths.get(width, {})
        if not (block.get("complete") and block.get("across_seeds")):
            pytest.skip(f"{width} decomposition incomplete")
        for k in ("5", "10", "20"):
            a = block["across_seeds"][k]
            row = _row(chunk, k)
            for metric in ("delta_reliability", "delta_resolution", "delta_brier_actual"):
                _any_fmt(row, a[metric]["mean"], f"Table {table} k={k} {metric}")
                assert f"{a[metric]['wilcoxon_p']:.4f}" in row, (
                    f"Table {table} k={k} {metric}: p={a[metric]['wilcoxon_p']:.4f} not in row"
                )
            for side in ("ring", "product"):
                v = a[f"mean_abs_residual_{side}"]
                assert f"{v:.4f}" in row, (
                    f"Table {table} k={k}: {side} residual {v:.4f} not in row"
                )


def test_murphy_decomposition_reconstructs_the_brier_score():
    """A decomposition that does not reconstruct the score it decomposes cannot
    support a claim about which of its terms dominates. The predecessor of this
    artifact reported terms that were off by more than twice the effect."""
    f = PKG_ROOT / "artifacts" / "murphy_decomposition.json"
    if not f.exists():
        raise AssertionError(
            "murphy_decomposition.json is missing; run "
            "`py -m aura.ml.evaluation.murphy_decomp`. This check skipped for one "
            "revision while the manuscript reported the result as computed, which is "
            "why it now fails instead."
        )
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


def test_table_12b_matches_the_external_validation_artifact():
    """Section 8.6's external result is the paper's strongest claim and the one a
    reader is least able to re-derive, so every published cell is pinned."""
    f = PKG_ROOT / "artifacts" / "chexpert_external.json"
    assert f.exists(), (
        "chexpert_external.json is missing; run "
        "`py -m aura.ml.evaluation.chexpert_external`. Do not weaken this to a skip - "
        "a skipped artifact check is why v4.1 published a Murphy result that was "
        "never computed."
    )
    data = json.loads(f.read_text(encoding="utf-8"))
    prim = data["primary_uncertain_as_negative"]
    text = _manuscript_text()

    rows = {
        "classical": "Classical product-of-experts",
        "quantum": "Quantum variational circuit",
        "learnable": "Learnable gated head",
        "logistic": "Multinomial logistic regression",
    }
    # Table 12b and Table 6 both carry these row names; pick the row holding the
    # external accuracy, which only Table 12b does.
    for arm, name in rows.items():
        b = prim["backends"][arm]
        hits = [ln for ln in text.splitlines()
                if ln.strip().startswith("|")
                and ln.split("|")[1].strip() == name
                and f"{b['accuracy']:.4f}" in ln]
        assert hits, (
            f"Table 12b has no row for {name} carrying the external accuracy "
            f"{b['accuracy']:.4f}"
        )
        row = hits[0]
        for label, value in (("NLL", b["nll"]), ("ECE", b["ece"]),
                             ("macro AUROC", b["macro_auroc"])):
            assert f"{value:.4f}" in row, (
                f"Table 12b {name}: {label} {value:.4f} not in row -> {row.strip()}"
            )
        assert str(b["copd_predictions"]) in row, (
            f"Table 12b {name}: COPD error count {b['copd_predictions']} not in row"
        )
        assert str(b["trainable_parameters"]) in row, (
            f"Table 12b {name}: parameter count not in row"
        )

    floor = prim["majority_class_floor"]["accuracy"]
    assert f"{floor:.4f}" in text, (
        f"the manuscript does not publish the external majority-class floor {floor:.4f}"
    )
    n_test = data["data"]["test_studies"]
    assert f"{n_test:,}" in text, (
        f"the manuscript does not publish the external test size {n_test:,}"
    )
    assert data["data"]["copd_derivable"] is False
    assert prim["majority_class_floor"]["support"]["copd"] == 0, (
        "COPD support must be zero on CheXpert; the pre-registration says the map "
        "cannot reach it without a hyperinflation label"
    )


def test_external_validation_was_preregistered_before_it_ran():
    """The pre-registration is the only reason Section 8.6's H1/H2 language is
    admissible. If the file goes missing the claim must not stand."""
    prereg = REPO_ROOT / "docs" / "PREREGISTRATION_CHEXPERT.md"
    assert prereg.exists(), "docs/PREREGISTRATION_CHEXPERT.md is missing"
    text = prereg.read_text(encoding="utf-8")
    for probe in ("H1", "H2", "H3", "H4", "frozen", "temperature"):
        assert probe in text, f"the pre-registration does not state {probe}"

    f = PKG_ROOT / "artifacts" / "chexpert_external.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        assert "PREREGISTRATION_CHEXPERT" in data.get("preregistration", ""), (
            "the external artifact does not name the pre-registration it ran under"
        )


def test_the_constant_channel_is_constant_in_both_corpora():
    """Section 8.6's H4. If prior_risk ever becomes informative on either corpus,
    the confound analysis in 5.1.3 and 8.3.1 needs revisiting, not quiet reuse."""
    import numpy as np

    ext = PKG_ROOT / "artifacts" / "chexpert_evidence.npz"
    mim = PKG_ROOT / "artifacts" / "quantum_study_evidence.npz"
    if not (ext.exists() and mim.exists()):
        pytest.skip("evidence tensors not present")

    X = np.load(ext)["X"]
    assert np.all(X[:, 7] == 0), "prior_risk is no longer constant on CheXpert"
    d = np.load(mim)
    for key in ("Xtr", "Xcal", "Xte"):
        assert np.all(d[key][:, 7] == 0), f"prior_risk is no longer constant in {key}"


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


# --------------------------------------------------------------------------- #
# The v5 submission manuscript
#
# Section 11.7 of v4 reported that the drift test "does not parse the abstract or
# Table 3 — a gap that allowed the abstract and Section 8.5 to disagree about the
# kernel result between two revisions". These tests close that gap for v5: the
# abstract's figures are checked against the same artifacts as the body's, so the
# two cannot diverge again.
# --------------------------------------------------------------------------- #
V5 = REPO_ROOT / "docs" / "RESEARCH_PAPER_v5.md"
V5_SUPP = REPO_ROOT / "docs" / "RESEARCH_PAPER_v5_SUPPLEMENTARY.md"


def _v5_text(path=None):
    p = path or V5
    if not p.exists():
        pytest.skip(f"{p} not present")
    return (p.read_text(encoding="utf-8")
            .replace("**", "").replace("−", "-").replace("—", "-"))


def _v5_abstract():
    text = _v5_text()
    return text.split("## Abstract")[1].split("## 1.")[0]


def test_v5_has_no_unfilled_placeholders():
    """A placeholder that reaches the manuscript is a fabricated number waiting to
    happen. Both documents are checked."""
    for p in (V5, V5_SUPP):
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        stray = re.findall(r"\[\[[A-Z_]+\]\]", text)
        assert not stray, f"{p.name} still contains placeholders: {sorted(set(stray))}"


def test_v5_table1_matches_the_matched_optimiser_artifact():
    """Table 1's lower block is the paper's central new result: under one optimiser
    protocol no ordering among the four backends survives."""
    f = PKG_ROOT / "artifacts" / "fusion_matched_optimiser.json"
    assert f.exists(), "fusion_matched_optimiser.json is missing; Table 1 is unbacked"
    data = json.loads(f.read_text(encoding="utf-8"))
    text = _v5_text()

    label = {"classical": "Product-of-experts", "quantum": "Quantum VQC",
             "learnable": "Gated head", "logistic": "Logistic regression"}
    # The matched block is the second occurrence of each row label.
    for arm, name in label.items():
        rows = [ln for ln in text.splitlines()
                if ln.strip().startswith("|") and len(ln.split("|")) > 2
                and ln.split("|")[2].strip() == name]
        assert len(rows) >= 2, f"expected published and matched rows for {name}"
        row = rows[1]
        b = data["backends"][arm]
        for key in ("accuracy", "nll", "ece", "macro_auroc"):
            _any_fmt(row, b[key], f"Table 1 matched / {name} / {key}")
        assert str(data["trainable_parameters"][arm]) in row, (
            f"Table 1 matched / {name}: parameter count "
            f"{data['trainable_parameters'][arm]} not in row"
        )


def test_v5_abstract_entanglement_figures_match_the_artifact():
    """The abstract's entanglement claim must read from the controlled 7-qubit run."""
    f = PKG_ROOT / "artifacts" / "entanglement_multiseed_7q.json"
    if not f.exists():
        pytest.skip("entanglement_multiseed_7q.json not present")
    data = json.loads(f.read_text(encoding="utf-8"))
    abstract = _v5_abstract()

    n_bad = data["delta_nll"]["n_seeds_favouring_entanglement"]
    assert n_bad == 0, "artifact no longer shows 0 of 10 seeds favouring entanglement"
    assert "ten of ten seeds" in abstract, (
        "the abstract no longer states the ten-of-ten seed count"
    )
    holm = data["delta_nll"]["wilcoxon_p_holm"]
    assert f"{holm:.3f}" in abstract or f"{holm:.4f}" in abstract, (
        f"abstract does not carry the Holm-adjusted p of {holm}"
    )


def test_v5_kernel_result_is_consistent_between_abstract_body_and_artifact():
    """The specific failure v4 disclosed: abstract and body disagreeing about the
    kernel. Both are now checked against the artifact."""
    f = PKG_ROOT / "artifacts" / "brain" / "qkl_full_coverage.json"
    if not f.exists():
        pytest.skip("qkl_full_coverage.json not present")
    data = json.loads(f.read_text(encoding="utf-8"))
    dl = data["pooled_out_of_fold"]["delong"]
    text = _v5_text()
    body = text.split("### 2.6")[1].split("### 2.7")[0]

    _any_fmt(body, dl["delta_auroc"], "section 2.6 / delta AUROC")
    _any_fmt(body, dl["auroc_a"], "section 2.6 / quantum AUROC")
    _any_fmt(body, dl["auroc_b"], "section 2.6 / rbf AUROC")
    assert str(data["corpus"]["subjects"]) in body, (
        "section 2.6 does not state the subject count it was computed on"
    )
    # The abstract must not contradict the body: it claims no difference, so the
    # artifact must actually show one that is not significant.
    assert dl["p"] > 0.05, (
        f"the artifact now shows a significant kernel difference (p={dl['p']}); "
        f"the abstract's 'did not differ' is no longer true"
    )


def test_v5_external_validation_matches_the_artifact():
    f = PKG_ROOT / "artifacts" / "chexpert_external.json"
    if not f.exists():
        pytest.skip("chexpert_external.json not present")
    prim = json.loads(f.read_text(encoding="utf-8"))["primary_uncertain_as_negative"]
    text = _v5_text()
    label = {"classical": "Product-of-experts", "quantum": "Quantum VQC",
             "learnable": "Gated head", "logistic": "Logistic regression"}
    table = text.split("Table 2.")[1].split("Pairwise:")[0]
    for arm, name in label.items():
        row = [ln for ln in table.splitlines()
               if ln.strip().startswith("|") and len(ln.split("|")) > 1
               and ln.split("|")[1].strip() == name]
        assert row, f"no Table 2 row for {name}"
        b = prim["backends"][arm]
        for key in ("accuracy", "nll", "ece", "macro_auroc"):
            _any_fmt(row[0], b[key], f"Table 2 / {name} / {key}")


def test_v5_design_sweep_claim_matches_the_artifact():
    """The corrected D12 claim: three of five best cells by NLL, not four."""
    f = PKG_ROOT / "artifacts" / "design_sweep.json"
    if not f.exists():
        pytest.skip("design_sweep.json not present")
    cells = json.loads(f.read_text(encoding="utf-8"))["cells"]
    top5 = sorted(cells, key=lambda c: c["nll"])[:5]
    n_none = sum(1 for c in top5 if c["entangler"] == "none")
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    body = _v5_text().split("### 2.5")[1].split("### 2.6")[0]
    assert f"{words[n_none]} of the five best" in body.lower(), (
        f"section 2.5 does not state that {words[n_none]} of the five best cells by "
        f"negative log-likelihood are at zero entanglement (artifact says {n_none})"
    )
    assert top5[0]["entangler"] == "none", (
        "the best cell by NLL is no longer the unentangled one; section 2.5 says it is"
    )


def test_v5_table4_matches_the_nonlinear_control_artifact():
    """Table 4 reports each arm's likelihood delta against the logistic anchor.

    Written by hand from the artifact, one of its Holm p-values was initially taken
    from the wrong pairwise comparison (classical-vs-quantum in place of
    classical-vs-logistic). Both are real numbers in the same artifact, so nothing
    downstream could have flagged it. This recomputes every cell.
    """
    f = PKG_ROOT / "artifacts" / "nonlinear_control.json"
    if not f.exists():
        pytest.skip("nonlinear_control.json not present")
    data = json.loads(f.read_text(encoding="utf-8"))
    section = _v5_text().split("### 2.3")[1].split("### 2.4")[0]

    for target, rec in data["targets"].items():
        b = rec["backends"]
        ceil = rec["ceilings"]
        _any_fmt(section, ceil["linear_ceiling"], f"{target} linear ceiling")
        _any_fmt(section, ceil["interaction_ceiling"], f"{target} interaction ceiling")

        for arm in ("classical", "learnable", "quantum"):
            delta = b[arm]["nll"] - b["logistic"]["nll"]
            _any_fmt(section, round(delta, 3), f"{target}/{arm} delta NLL vs logistic")

            key = f"classical_vs_logistic" if arm == "classical" else f"{arm}_vs_logistic"
            holm = rec["pairwise"][key]["nll_wilcoxon_p_holm"]
            # Only non-negligible p-values are written out in full; the tiny ones
            # are rendered as an inequality, which is checked separately below.
            if holm >= 1e-4:
                assert (f"{holm:.3f}" in section or f"{holm:.4f}" in section
                        or f"{holm:.1e}".replace("e-0", "x10-") in section
                        or f"{holm * 1e4:.1f}" in section), (
                    f"{target}/{arm}: Holm p={holm} not found in section 2.3"
                )

    # The section's headline claim, recomputed rather than trusted.
    q = data["targets"]["T2b_pairwise"]
    q_delta = (q["backends"]["quantum"]["nll"] - q["backends"]["logistic"]["nll"])
    assert q_delta < 0, (
        "section 2.3 states the quantum circuit beats the logistic regression on "
        "T2b; the artifact no longer shows that"
    )
    assert q["pairwise"]["quantum_vs_logistic"]["nll_wilcoxon_p_holm"] < 0.05, (
        "section 2.3 states that T2b win is significant; the artifact disagrees"
    )
