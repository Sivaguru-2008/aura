"""Every number in manuscript v6 must equal the artifact it is read from (protocol rule P6).

The registry ``docs/manuscript_v6_numbers.json`` maps each published number to an artifact and
a key path, with the format it is printed in. Three checks:

1. every registered number, formatted from its artifact, appears verbatim in the manuscript;
2. no unresolved placeholder (⟨…⟩) remains;
3. every decimal number in a manuscript *table* is registered, or listed as a literal with a
   reason (for example a figure quoted from another paper).

If this fails, fix the manuscript or the registry by re-reading the artifact. Never loosen a
format to make a number match.
"""
import json
import re
from functools import reduce
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANUSCRIPT = REPO / "docs" / "MANUSCRIPT_v6_npjDM.md"
REGISTRY = REPO / "docs" / "manuscript_v6_numbers.json"
DECIMAL = re.compile(r"(?<![\w.])[−-]?\d+\.\d+(?![\w.])")


def _registry():
    if not REGISTRY.exists():
        raise FileNotFoundError(
            f"Required numerical registry {REGISTRY} does not exist. Silent skipping is prohibited."
        )
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _get(obj, path):
    def step(o, k):
        if isinstance(o, list):
            return o[int(k)]
        return o[k]
    return reduce(step, path.split("/"), obj)


def _fmt(value, spec):
    """spec: e.g. '{:.4f}', '{:+.3f}', 'pct1' (×100, 1 dp), 'abs:{:.1f}'."""
    if spec.startswith("abs:"):
        value, spec = abs(value), spec[4:]
    if spec.startswith("pct"):
        s = f"{100 * value:.{int(spec[3:])}f}"
    else:
        s = spec.format(value)
    return s.replace("-", "−")


def _text():
    return MANUSCRIPT.read_text(encoding="utf-8").replace("-", "−")


def test_registered_numbers_match_artifacts():
    reg = _registry()
    text = _text()
    bad = []
    for item in reg["numbers"]:
        art_path = REPO / item["artifact"]
        if not art_path.exists():
            raise FileNotFoundError(f"Registered artifact {art_path} is missing. Silent skipping is prohibited.")
        art = json.loads(art_path.read_text(encoding="utf-8"))
        expected = _fmt(_get(art, item["path"]), item["format"])
        if expected not in text:
            bad.append(f"{item['id']}: expected '{expected}' from {item['artifact']}:{item['path']}")
    assert not bad, "manuscript numbers not found:\n" + "\n".join(bad)


def test_no_placeholders_remain():
    hits = re.findall(r"⟨[^⟩]*⟩", MANUSCRIPT.read_text(encoding="utf-8"))
    assert not hits, f"{len(hits)} unresolved placeholders, e.g. {hits[:3]}"


def test_every_table_number_is_registered():
    reg = _registry()
    known = set(reg.get("literals", {}))
    text = _text()
    for item in reg["numbers"]:
        art = json.loads((REPO / item["artifact"]).read_text(encoding="utf-8"))
        known.add(_fmt(_get(art, item["path"]), item["format"]).lstrip("+"))
    missing = []
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith("|"):
            continue
        for tok in DECIMAL.findall(line):
            if tok.lstrip("+") not in known and tok.lstrip("+−") not in known:
                missing.append(f"line {ln}: {tok}")
    assert not missing, "unregistered table numbers:\n" + "\n".join(missing[:40])


def test_corrupted_metric_fails_suite():
    """Priority 2: Deliberately corrupted metrics must cause assertion failure."""
    reg = _registry()
    sample_item = reg["numbers"][0]
    art_path = REPO / sample_item["artifact"]
    art = json.loads(art_path.read_text(encoding="utf-8"))
    true_val = _fmt(_get(art, sample_item["path"]), sample_item["format"])

    corrupted_text = _text().replace(true_val, "9.9999_CORRUPTED")

    bad = []
    for item in reg["numbers"]:
        art = json.loads((REPO / item["artifact"]).read_text(encoding="utf-8"))
        expected = _fmt(_get(art, item["path"]), item["format"])
        if expected not in corrupted_text:
            bad.append(f"{item['id']}: expected '{expected}' from {item['artifact']}:{item['path']}")

    assert len(bad) > 0, "Corrupted metric was not detected by verification suite!"
    assert sample_item["id"] in bad[0], f"Expected failure on {sample_item['id']}, got {bad}"

    # Verify that an unregistered decimal in a table raises missing error
    corrupted_table_text = _text() + "\n| Corrupted Row | 0.9999 |"
    known = set(reg.get("literals", {}))
    for item in reg["numbers"]:
        art = json.loads((REPO / item["artifact"]).read_text(encoding="utf-8"))
        known.add(_fmt(_get(art, item["path"]), item["format"]).lstrip("+"))

    missing = []
    for ln, line in enumerate(corrupted_table_text.splitlines(), 1):
        if not line.lstrip().startswith("|"):
            continue
        for tok in DECIMAL.findall(line):
            if tok.lstrip("+") not in known and tok.lstrip("+−") not in known:
                missing.append(f"line {ln}: {tok}")
    assert any("0.9999" in m for m in missing), "Corrupted table decimal was not flagged as missing!"


def test_missing_registry_raises_explicit_failure(monkeypatch):
    """Priority 2: Missing registry must raise FileNotFoundError, not skip."""
    nonexistent = REPO / "docs" / "nonexistent_registry_file.json"
    monkeypatch.setattr("aura.tests.test_manuscript_numbers.REGISTRY", nonexistent)
    with pytest.raises(FileNotFoundError, match="Silent skipping is prohibited"):
        _registry()
