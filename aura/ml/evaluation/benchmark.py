"""Quantum-vs-classical fusion benchmark — the head-to-head we show judges.

Evaluates both trained backends on a fresh held-out set with the same metrics,
including the properties that matter clinically: calibration (ECE), conformal
coverage vs. its guarantee, and average prediction-set size (efficiency).
"""
from __future__ import annotations

import json

import numpy as np

from common.config import ARTIFACTS, get_settings
from common.mathx import softmax
from schemas.clinical import DIAGNOSES
from services.fusion.classical import ClassicalFusion
from services.fusion.quantum import QuantumFusion
from services.safety.calibration import (
    Calibration,
    expected_calibration_error,
    fit_conformal,
)
from ml.training.dataset import build_evidence_dataset, make_splits


def _brier(P, y):
    Y = np.eye(len(DIAGNOSES))[y]
    return float(((P - Y) ** 2).sum(axis=1).mean())


def _conformal_stats(P, y, coverage):
    qhat = fit_conformal(P, y, coverage)
    sets = P >= (1.0 - qhat)
    covered = sets[np.arange(len(y)), y].mean()
    size = sets.sum(axis=1).mean()
    return float(covered), float(size), float(qhat)


def _eval_backend(logits, y, coverage, temperature=1.0):
    P = np.array([softmax(r / temperature) for r in logits])
    acc = float((P.argmax(1) == y).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())
    ece = expected_calibration_error(P, y)
    cov, size, qhat = _conformal_stats(P, y, coverage)
    return {
        "accuracy": round(acc, 4),
        "nll": round(nll, 4),
        "ece": round(ece, 4),
        "brier": round(_brier(P, y), 4),
        "conformal_coverage": round(cov, 4),
        "conformal_set_size": round(size, 3),
    }


def run(n_samples: int = 500) -> dict:
    s = get_settings()
    q = QuantumFusion.load()
    c = ClassicalFusion.load()
    if q is None or c is None:
        raise RuntimeError("Fusion models not trained. Run `aura_cli train` first.")
    cal = Calibration.load()

    # Fresh evaluation split (different seed so it's genuinely held out).
    _, _, te = make_splits(n_samples, seed=s.seed + 101)
    Xte, yte = build_evidence_dataset(te)

    q_logits = np.array([q.logits(x) for x in Xte])
    c_logits = np.array([c.logits(x) for x in Xte])

    result = {
        "n_eval": len(yte),
        "temperature": round(cal.temperature, 4),
        "quantum": _eval_backend(q_logits, yte, s.conformal_coverage, cal.temperature),
        "classical": _eval_backend(c_logits, yte, s.conformal_coverage),
    }

    # Full clinical evaluation suite (AUROC/AUPRC/sens/spec/PPV/NPV/F1/calibration).
    from ml.evaluation.metrics import evaluate, print_report

    Pq = np.array([softmax(r / cal.temperature) for r in q_logits])
    Pc = np.array([softmax(r) for r in c_logits])
    full = {"quantum": evaluate(Pq, yte), "classical": evaluate(Pc, yte)}

    # Include the extra backends when they are trained.
    from services.fusion.ensemble import DeepEnsemble
    from services.fusion.learnable import LearnableFusion
    ens, lrn = DeepEnsemble.load(), LearnableFusion.load()
    if ens is not None:
        full["ensemble"] = evaluate(
            np.array([softmax(ens.logits(x) / cal.temperature) for x in Xte]), yte)
    if lrn is not None:
        full["learnable"] = evaluate(
            np.array([softmax(lrn.logits(x) / cal.temperature) for x in Xte]), yte)

    result["metrics_full"] = full
    (ARTIFACTS / "benchmark.json").write_text(json.dumps(result, indent=2))
    _print_table(result)
    print_report(full["quantum"], title="Quantum backend — full clinical metrics")
    return result


def _print_table(r: dict) -> None:
    q, c = r["quantum"], r["classical"]
    rows = [
        ("accuracy", q["accuracy"], c["accuracy"], "higher"),
        ("nll", q["nll"], c["nll"], "lower"),
        ("ece (calibration)", q["ece"], c["ece"], "lower"),
        ("brier", q["brier"], c["brier"], "lower"),
        ("conformal coverage", q["conformal_coverage"], c["conformal_coverage"], "~0.90"),
        ("conformal set size", q["conformal_set_size"], c["conformal_set_size"], "lower"),
    ]
    print("\n  Quantum vs classical evidence fusion "
          f"(n={r['n_eval']}, held-out)\n")
    print(f"  {'metric':<22}{'quantum':>10}{'classical':>12}   better")
    print("  " + "-" * 56)
    for name, qv, cv, better in rows:
        print(f"  {name:<22}{qv:>10}{cv:>12}   {better}")
    print()


if __name__ == "__main__":
    run()
