"""Input-provenance assertions for evaluation inputs (defect D14).

D14: real-trained models were scored on synthetic evidence and a "validated" verdict was
printed. The doc-number drift test passed, because it only checks that prose equals
artifacts — not that the artifacts were computed from the data they claim. This module
closes that gap for the cached tensors the paper's fusion results are computed from: the
SHA-256 of every array is pinned in ``evidence_ledger.json`` and checked on load, and a
cache whose ``source`` is not the real corpus is refused outright.

    python -m aura.ml.evaluation.provenance --write   # re-pin after a deliberate rebuild

Re-pinning is a deliberate act and shows up in the diff of the ledger; that is the point.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from aura.common.config import ARTIFACTS

LEDGER = ARTIFACTS / "evidence_ledger.json"

#: Files whose bytes the published fusion results depend on.
PINNED = (
    "quantum_study_evidence.npz",
    "fusion_matched_optimiser_predictions.npz",
)


class ProvenanceError(RuntimeError):
    """An evaluation input is not the tensor the published results were computed from."""


def array_digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())   # shape/dtype are part of identity
    h.update(a.tobytes())
    return h.hexdigest()


def npz_digests(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as d:
        return {k: array_digest(d[k]) for k in sorted(d.files)}


def write_ledger() -> dict:
    ledger = {name: npz_digests(ARTIFACTS / name) for name in PINNED}
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger


def assert_npz(path: Path) -> None:
    """Raise unless *path* matches its ledger entry array-for-array."""
    path = Path(path)
    if not LEDGER.is_file():
        raise ProvenanceError(f"no evidence ledger at {LEDGER}")
    ledger = json.loads(LEDGER.read_text())
    if path.name not in ledger:
        raise ProvenanceError(f"{path.name} is not pinned in {LEDGER.name}")
    actual, expected = npz_digests(path), ledger[path.name]
    bad = sorted(k for k in expected.keys() | actual.keys()
                 if expected.get(k) != actual.get(k))
    if bad:
        raise ProvenanceError(f"{path.name}: arrays {bad} differ from the pinned ledger — "
                              "this is not the evidence the published results used")


def assert_real_source(source: str) -> None:
    if source != "mimic-cxr":
        raise ProvenanceError(f"evidence source is {source!r}, not 'mimic-cxr'; refusing to "
                              "score real-trained models on it")


if __name__ == "__main__":
    if "--write" in sys.argv:
        print(json.dumps(write_ledger(), indent=2))
    else:
        for name in PINNED:
            assert_npz(ARTIFACTS / name)
            print(f"ok  {name}")
