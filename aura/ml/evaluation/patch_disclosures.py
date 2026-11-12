"""Apply the disclosures Section 11.7 owes to the published artifacts.

Why this exists rather than a regeneration
------------------------------------------
Three of the defects Section 11.7 reports are *missing disclosures on artifacts
whose numbers are correct*: an artifact that carries a retracted interpretation, an
artifact whose hardware statement was overtaken by events, and artifacts that record
results without recording the split or the training length that produced them.

Regenerating those artifacts to add a sentence would retrain the models and move the
numbers the manuscript cites, which trades a documentation defect for a reproducibility
one. So the disclosures are applied in place, from the same module-level constants the
generating scripts use, and ``tests/test_doc_numbers.py`` asserts that the artifact on
disk carries them. One definition, one writer, one test.

The script is idempotent: running it twice changes nothing the second time.

Usage::

    python -m aura.ml.evaluation.patch_disclosures
    python -m aura.ml.evaluation.patch_disclosures --check   # exit 1 if unpatched
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aura.common.config import ARTIFACTS
from aura.ml.evaluation.quantum_study import (
    BUDGET_NOTE_WITHDRAWN,
    BUDGET_RETRACTION,
    HARDWARE_DISCLOSURE,
)

# --------------------------------------------------------------------------- #
# Training lengths of the served fusion arms.
#
# Section 11.7 defect 6: ``fusion_headline_n447.json`` evaluates the served models
# but does not record how long they were trained, so a re-run cannot be attributed.
# The values are the defaults ``ml/training/train_fusion.run()`` calls its trainers
# with — it overrides none of them — and are recorded here with that provenance
# rather than as free-standing constants.
# --------------------------------------------------------------------------- #
SERVED_TRAINING = {
    "classical": {
        "epochs": 400,
        "optimiser": "full-batch gradient descent, fixed lr=0.3, l2=1e-3",
        "source": "ml/training/train_fusion.py::train_classical defaults",
    },
    "learnable": {
        "epochs": 300,
        "optimiser": "Adam, lr=0.05, weight_decay=1e-4, full batch",
        "source": "services/fusion/learnable.py::train_learnable defaults",
    },
    "quantum": {
        "epochs": 30,
        "optimiser": "Adam, lr=0.05, batch=50",
        "source": "ml/training/train_fusion.py::train_quantum defaults",
    },
    "_note": (
        "Recorded retrospectively from the trainer defaults that "
        "train_fusion.run() invokes; it overrides none of them. The arms differ in "
        "optimiser as well as in model class, which is the confound Section 11.5 "
        "reports and artifacts/fusion_matched_optimiser.json removes."
    ),
}

ABLATION_PROVENANCE = {
    "SUPERSEDED": (
        "This artifact must not be cited. It records neither its sample size nor "
        "its split identifier, so its numbers cannot be attributed to a cohort "
        "(Section 11.7, defect 6). The selective-prediction analysis it was "
        "published to support has been redone on the primary 447-study split and "
        "withdrawn on the evidence of that redo; see artifacts/risk_coverage_n447.json "
        "and Section 9.1. Retained only so that the earlier published numbers remain "
        "traceable to a file."
    ),
    "split": "UNRECORDED — not reconstructible from this artifact",
    "n": "UNRECORDED — not reconstructible from this artifact",
    "superseded_by": "artifacts/risk_coverage_n447.json",
}


def _patch_quantum_study(dry: bool) -> list[str]:
    path = ARTIFACTS / "quantum_study.json"
    if not path.exists():
        return [f"MISSING {path.name}"]
    d = json.loads(path.read_text(encoding="utf-8"))
    changes = []

    budget = d.get("q3_measurement_budget")
    if isinstance(budget, dict):
        if budget.get("RETRACTED") != BUDGET_RETRACTION:
            budget["RETRACTED"] = BUDGET_RETRACTION
            changes.append("q3_measurement_budget.RETRACTED")
        if budget.get("note_withdrawn") != BUDGET_NOTE_WITHDRAWN:
            budget["note_withdrawn"] = BUDGET_NOTE_WITHDRAWN
            changes.append("q3_measurement_budget.note_withdrawn")
        if "note" in budget:
            del budget["note"]
            changes.append("q3_measurement_budget.note (removed; superseded)")

    honesty = d.get("honesty")
    if isinstance(honesty, dict) and honesty.get("hardware") != HARDWARE_DISCLOSURE:
        honesty["hardware"] = HARDWARE_DISCLOSURE
        changes.append("honesty.hardware")

    if changes and not dry:
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return changes


def _patch_headline(dry: bool) -> list[str]:
    path = ARTIFACTS / "fusion_headline_n447.json"
    if not path.exists():
        return [f"MISSING {path.name}"]
    d = json.loads(path.read_text(encoding="utf-8"))
    changes = []
    if d.get("served_model_training") != SERVED_TRAINING:
        d["served_model_training"] = SERVED_TRAINING
        changes.append("served_model_training")
    if changes and not dry:
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return changes


def _patch_ablation(dry: bool) -> list[str]:
    path = ARTIFACTS / "fusion_ablation.json"
    if not path.exists():
        return [f"MISSING {path.name}"]
    d = json.loads(path.read_text(encoding="utf-8"))
    changes = []
    if d.get("provenance") != ABLATION_PROVENANCE:
        d["provenance"] = ABLATION_PROVENANCE
        changes.append("provenance")
    if changes and not dry:
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change and exit 1 if anything would")
    args = ap.parse_args()

    all_changes: dict[str, list[str]] = {
        "quantum_study.json": _patch_quantum_study(args.check),
        "fusion_headline_n447.json": _patch_headline(args.check),
        "fusion_ablation.json": _patch_ablation(args.check),
    }

    pending = {k: v for k, v in all_changes.items() if v}
    if not pending:
        print("[disclosures] all artifacts already carry their disclosures")
        return 0

    verb = "would patch" if args.check else "patched"
    for name, fields in pending.items():
        print(f"[disclosures] {verb} {name}: {', '.join(fields)}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
