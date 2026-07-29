"""AURA quantum decision path — a single, judge-facing demo.

Runs real held-out patients through the *quantum* backend end to end and narrates
what the quantum layer is actually doing:

    quantum VQC fusion  ->  posterior + shot-noise (measurement-native) uncertainty
    QMBA measurement budget  ->  COMMIT or ABSTAIN, with the reason
    (and cites the real IBM hardware run that executed this exact circuit)

The point it makes on stage: quantum here is not decoration and not an accuracy
claim. It is (1) a trained circuit that runs on real hardware, and (2) a
measurement-native way to decide *when to abstain* — something classical inference
does not give you for free. It deliberately shows one case that commits and one that
abstains, because the abstention is the part a classifier cannot do.

    python -m ml.evaluation.quantum_demo
    python -m ml.evaluation.quantum_demo --commit-case 144 --abstain-case 37
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from aura.common.config import ARTIFACTS
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.fusion.qmba import QuantumMeasurementBudget
from aura.services.fusion.quantum import QuantumFusion

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BAR = "=" * 66
RULE = "-" * 66


def _label(i: int) -> str:
    d = DIAGNOSES[i]
    return d.value if hasattr(d, "value") else str(d)


def _posterior(model: QuantumFusion, x: np.ndarray) -> np.ndarray:
    return softmax(model.W @ model._expectations(x) + model.b)


def _load_hardware_citation() -> str:
    path = ARTIFACTS / "ibm_hardware_run.json"
    if not path.exists():
        return "  (no hardware run on record — run run_ibm_hardware.py --mode hardware)"
    d = json.loads(path.read_text())
    hw = d.get("hardware", {})
    if not hw:
        return "  (hardware artifact present but incomplete)"
    agree = "top-1 diagnosis preserved" if hw.get("top1_agrees_with_analytic") else "top-1 diverged under noise"
    return (f"  executed on IBM {hw.get('backend','?')} "
            f"({hw.get('backend_qubits','?')} qubits) — job {hw.get('job_id','?')}\n"
            f"  {agree}; mean |d<Z>| = {hw.get('mean_abs_z_error_vs_analytic', float('nan')):.2f} "
            f"vs. the analytic circuit")


def _select_cases(model: QuantumFusion, budget: QuantumMeasurementBudget,
                  X: np.ndarray, y: np.ndarray, commit_idx: int | None,
                  abstain_idx: int | None) -> tuple[int, int]:
    """Auto-pick one case QMBA commits and one it abstains on, unless overridden."""
    if commit_idx is not None and abstain_idx is not None:
        return commit_idx, abstain_idx

    confidences = np.array([_posterior(model, x).max() for x in X])
    margins = np.array([
        (lambda p: float(np.sort(p)[-1] - np.sort(p)[-2]))(_posterior(model, x))
        for x in X])

    if commit_idx is None:
        # most confident case that QMBA actually commits
        for i in np.argsort(confidences)[::-1]:
            if budget.decide(X[i]).committed:
                commit_idx = int(i)
                break
        commit_idx = commit_idx if commit_idx is not None else int(confidences.argmax())

    if abstain_idx is None:
        # the smallest-margin case QMBA abstains on (the genuinely ambiguous patient)
        for i in np.argsort(margins):
            if not budget.decide(X[i]).committed:
                abstain_idx = int(i)
                break
        abstain_idx = abstain_idx if abstain_idx is not None else int(margins.argmin())

    return commit_idx, abstain_idx


def _show_case(tag: str, idx: int, model: QuantumFusion,
               budget: QuantumMeasurementBudget, X: np.ndarray, y: np.ndarray) -> None:
    x = X[idx].astype(float)
    post, post_std = model.fuse(x)
    order = sorted(range(len(DIAGNOSES)), key=lambda i: post[_label(i)], reverse=True)
    decision = budget.decide(x)

    print(f"\n {tag} — held-out patient #{idx}   (true dx: {_label(int(y[idx]))})")
    print("   quantum VQC posterior  (probability +/- shot-noise std):")
    for rank, i in enumerate(order[:4]):
        lab = _label(i)
        marker = "  <-- top" if rank == 0 else ""
        print(f"     {lab:<22} {post[lab]:.3f} +/- {post_std[lab]:.3f}{marker}")

    print("   QMBA measurement budget:")
    verdict = "COMMIT" if decision.committed else "ABSTAIN"
    print(f"     shots spent : {decision.shots_spent}")
    print(f"     margin      : {decision.margin:.3f} +/- {decision.margin_std:.3f} "
          f"({decision.separation_z:.1f} sigma of shot noise)")
    if not decision.committed:
        lim = decision.limiting_factor
        kind = ("model-limited — a human is needed, more shots would not help"
                if lim == "model" else
                "measurement-limited — a longer run on the SAME circuit would resolve it")
        print(f"     limiting    : {kind}")
        if decision.predicted_shots:
            print(f"     would need  : ~{decision.predicted_shots} shots to reach the bar")
    print(f"     >> {verdict}: {decision.top}"
          + (f"  (over {decision.runner_up})" if decision.committed else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit-case", type=int, default=None)
    ap.add_argument("--abstain-case", type=int, default=None)
    args = ap.parse_args()

    model = QuantumFusion.load()
    if model is None:
        raise SystemExit("no fusion_quantum.npz — the quantum backend is not trained")
    budget = QuantumMeasurementBudget(model)

    d = np.load(ARTIFACTS / "quantum_study_evidence.npz")
    X, y = d["Xte"], d["yte"]

    print(BAR)
    print(" AURA — QUANTUM DECISION PATH  (real held-out patients)")
    print(BAR)
    print(f" fusion backend : quantum VQC — {model.n_qubits} qubits, {model.n_layers} "
          f"layers, {model.entangler} entangler")
    print(" readout        : <Z_i> per qubit -> linear head -> diagnosis posterior")
    print(" uncertainty    : propagated from finite-shot measurement variance")
    print(" hardware proof :")
    print(_load_hardware_citation())
    print(RULE)

    commit_idx, abstain_idx = _select_cases(model, budget, X, y,
                                             args.commit_case, args.abstain_case)
    _show_case("CASE A (confident)", commit_idx, model, budget, X, y)
    _show_case("CASE B (ambiguous)", abstain_idx, model, budget, X, y)

    print("\n" + BAR)
    print(" Takeaway: the quantum layer runs on real hardware AND decides when NOT to")
    print(" answer — measurement-native abstention, not an accuracy claim.")
    print(BAR)


if __name__ == "__main__":
    main()
