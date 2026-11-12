"""D14 guard: evaluation inputs must be the pinned real-corpus tensors.

The drift test (test_doc_numbers.py) proves prose == artifacts. This proves artifacts were
computed from the evidence they claim — the gap D14 fell through.
"""
import shutil

import numpy as np
import pytest

from aura.common.config import ARTIFACTS
from aura.ml.evaluation import provenance
from aura.ml.evaluation.provenance import ProvenanceError


@pytest.mark.parametrize("name", provenance.PINNED)
def test_pinned_artifacts_match_ledger(name):
    path = ARTIFACTS / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this checkout")
    provenance.assert_npz(path)


def test_tampered_array_is_refused(tmp_path):
    src = ARTIFACTS / "quantum_study_evidence.npz"
    if not src.is_file():
        pytest.skip("evidence cache not present")
    with np.load(src) as d:
        arrays = {k: d[k] for k in d.files}
    arrays["Xte"] = arrays["Xte"].copy()
    arrays["Xte"][0, 0] += 1e-9                      # one ulp-scale edit must be caught
    fake = tmp_path / src.name
    np.savez(fake, **arrays)
    with pytest.raises(ProvenanceError, match="Xte"):
        provenance.assert_npz(fake)


def test_unpinned_file_is_refused(tmp_path):
    other = tmp_path / "something_else.npz"
    np.savez(other, x=np.zeros(3))
    with pytest.raises(ProvenanceError, match="not pinned"):
        provenance.assert_npz(other)


def test_synthetic_source_is_refused():
    with pytest.raises(ProvenanceError, match="synthetic"):
        provenance.assert_real_source("synthetic")
    provenance.assert_real_source("mimic-cxr")
