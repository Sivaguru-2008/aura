"""Quantum evidence pack: does the quantum part do measurable work?

One script, one artifact (``artifacts/quantum_study.json``), four questions a
quantum-hackathon judge will actually ask. Every number is measured on the same real
MIMIC-CXR evidence splits, patient-disjoint, with each model scored at *its own*
fitted temperature.

**Q1 — Does entanglement do clinical work?**
Trains the shipped CNOT-ring VQC and an otherwise-identical product-state VQC
(``entangler="none"``): same qubits, same layers, same parameter count, same
encoding, same readout, same optimiser, same seed, same data. The only difference is
whether the two-qubit gates are there. Whatever the delta is, it is attributable to
entanglement and to nothing else — including the case where the delta is ~0, which is
a real and publishable answer.

**Q2 — Is the quantum backend competitive?**
Scores the entangled VQC, the product VQC, and the classical product-of-experts on
the held-out split at each model's own temperature. The audit already established
that scoring one backend at another's temperature is what produced the original
inflated quantum win (audit F6), so that mistake is not repeated here.

**Q3 — What does measurement budget buy?**
Runs :class:`~aura.services.fusion.qmba.QuantumMeasurementBudget` over the whole test set:
commit rate, accuracy at commit, shots spent, and the split between decisions that
were measurement-limited (more shots would settle it) and model-limited (nothing
would). This is the number with no classical analogue.

**Q4 — What does the entanglement encode?**
Aggregates the per-patient evidence-coupling differentials over the test set to find
which clinical evidence pairs this circuit consistently couples.

The evidence set is cached, because building it runs the DenseNet backbone over ~900
real films and the ablation is meaningless unless every model sees byte-identical
data.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.fusion.device import make_qnode
from aura.services.fusion.qmba import QuantumMeasurementBudget
from aura.services.fusion.qmeasure import measure_entanglement
from aura.services.fusion.quantum import QuantumFusion
from aura.services.safety.calibration import (
    expected_calibration_error,
    fit_temperature,
)

N_DX = len(DIAGNOSES)
EVIDENCE_CACHE = ARTIFACTS / "quantum_study_evidence.npz"
OUTPUT = ARTIFACTS / "quantum_study.json"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_evidence(n: int, seed: int, refresh: bool = False):
    """Real MIMIC evidence splits, cached to disk.

    Cached deliberately rather than rebuilt: the entanglement ablation compares two
    models that must see identical inputs, and rebuilding the set between runs would
    resample patients and silently confound the comparison.
    """
    if EVIDENCE_CACHE.exists() and not refresh:
        d = np.load(EVIDENCE_CACHE)
        print(f"[data] reusing cached evidence set from {EVIDENCE_CACHE.name} "
              f"(train={len(d['ytr'])} cal={len(d['ycal'])} test={len(d['yte'])})")
        return (d["Xtr"], d["ytr"], d["Xcal"], d["ycal"], d["Xte"], d["yte"],
                str(d["source"]))

    from ..training.dataset import build_evidence_dataset, make_splits, \
        real_evidence_splits

    print(f"[data] building real MIMIC evidence set (target n={n}) ...")
    cap = max(60, n // N_DX)
    real = real_evidence_splits(n=n, split="train", seed=seed, per_class_cap=cap)
    if real is not None:
        Xtr, ytr, Xcal, ycal, Xte, yte = real
        source = "mimic-cxr"
    else:
        print("[data] MIMIC unavailable — falling back to the synthetic evidence world")
        tr, cal, te = make_splits(n, seed=seed)
        Xtr, ytr = build_evidence_dataset(tr)
        Xcal, ycal = build_evidence_dataset(cal)
        Xte, yte = build_evidence_dataset(te)
        source = "synthetic"

    np.savez(EVIDENCE_CACHE, Xtr=Xtr, ytr=ytr, Xcal=Xcal, ycal=ycal, Xte=Xte,
             yte=yte, source=source)
    print(f"[data] cached to {EVIDENCE_CACHE.name} "
          f"(train={len(ytr)} cal={len(ycal)} test={len(yte)}, source={source})")
    return Xtr, ytr, Xcal, ycal, Xte, yte, source


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _quantum_logits(theta, W, b, X, n_qubits, n_layers, entangler):
    circuit = make_qnode(n_qubits, n_layers, interface="numpy", entangler=entangler)
    z = np.stack([np.asarray(v) for v in circuit(np.asarray(X), theta)], axis=-1)
    return z @ W.T + b


def bootstrap_delta(probs_a, probs_b, y, *, n_boot: int = 2000, seed: int = 7) -> dict:
    """Paired bootstrap CI for the metric differences between two models.

    Paired — the same resampled studies are scored under both models — because the
    two models saw identical data and the question is which is better *on the same
    patients*. An unpaired bootstrap would add between-sample variance that the
    comparison does not contain, and would widen every interval for no reason.

    This exists because the test split is 173 studies. An accuracy difference of
    0.006 on 173 studies is one patient, and reporting it without an interval would
    be exactly the kind of unsupported headline this project's audit was written to
    catch.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_all = np.arange(n)
    deltas = {"accuracy": [], "nll": [], "ece": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pa, pb = probs_a[idx], probs_b[idx]
        deltas["accuracy"].append(float((pa.argmax(1) == yb).mean()
                                        - (pb.argmax(1) == yb).mean()))
        rows = np.arange(len(yb))
        deltas["nll"].append(float(
            -np.log(np.clip(pa[rows, yb], 1e-12, 1)).mean()
            + np.log(np.clip(pb[rows, yb], 1e-12, 1)).mean()))
        deltas["ece"].append(float(expected_calibration_error(pa, yb)
                                   - expected_calibration_error(pb, yb)))

    out: dict = {}
    for metric, values in deltas.items():
        arr = np.asarray(values)
        low, high = np.percentile(arr, [2.5, 97.5])
        out[metric] = {
            "delta": round(float(arr.mean()), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            # "Significant" here means the 95% interval excludes zero. Stated as a
            # flag rather than a p-value because that is the claim being made.
            "excludes_zero": bool(low > 0 or high < 0),
        }
    out["n_bootstrap"] = int(n_boot)
    out["n_studies"] = int(n)
    return out


def score(logits_cal, y_cal, logits_test, y_test) -> dict:
    """Fit temperature on calibration, report held-out metrics at that temperature.

    Each model gets its own ``T``. This is the fix the audit demanded: applying one
    backend's temperature to another's logits understates the second and is what
    produced the original inflated quantum-vs-classical gap.
    """
    T = fit_temperature(logits_cal, y_cal)
    P = np.array([softmax(r / T) for r in logits_test])
    correct = P.argmax(1) == y_test
    nll = -np.log(np.clip(P[np.arange(len(y_test)), y_test], 1e-12, 1))
    return {
        "accuracy": round(float(correct.mean()), 4),
        "nll": round(float(nll.mean()), 4),
        "ece": round(float(expected_calibration_error(P, y_test)), 4),
        "temperature": round(float(T), 4),
        "mean_confidence": round(float(P.max(1).mean()), 4),
    }


# --------------------------------------------------------------------------- #
# Q1 + Q2: entanglement ablation
# --------------------------------------------------------------------------- #
def run_ablation(Xtr, ytr, Xcal, ycal, Xte, yte, s, epochs: int) -> dict:
    from ..training.train_fusion import train_classical, train_quantum

    results: dict = {}
    probs: dict[str, np.ndarray] = {}
    for entangler, label in (("ring", "vqc_entangled"), ("none", "vqc_product")):
        print(f"[ablation] training VQC entangler={entangler!r} ...")
        t0 = time.time()
        theta, W, b = train_quantum(Xtr, ytr, s.n_qubits, s.n_layers, epochs=epochs,
                                    seed=s.seed, entangler=entangler)
        cal = _quantum_logits(theta, W, b, Xcal, s.n_qubits, s.n_layers, entangler)
        test = _quantum_logits(theta, W, b, Xte, s.n_qubits, s.n_layers, entangler)
        metrics = score(cal, ycal, test, yte)
        T = metrics["temperature"]
        probs[label] = np.array([softmax(r / T) for r in test])
        metrics["train_seconds"] = round(time.time() - t0, 2)
        metrics["entangler"] = entangler
        metrics["trainable_parameters"] = int(theta.size + W.size + b.size)
        results[label] = metrics
        if entangler == "ring":
            np.savez(ARTIFACTS / "fusion_quantum_ablation_ring.npz",
                     theta=theta, W=W, b=b, n_qubits=s.n_qubits, n_layers=s.n_layers)
        print(f"[ablation]   {label}: acc={metrics['accuracy']} "
              f"nll={metrics['nll']} ece={metrics['ece']} T={metrics['temperature']}")

    print("[ablation] training classical product-of-experts reference ...")
    Wc, bc = train_classical(Xtr, ytr)
    results["classical_poe"] = score(Xcal @ Wc.T + bc, ycal, Xte @ Wc.T + bc, yte)
    results["classical_poe"]["trainable_parameters"] = int(Wc.size + bc.size)
    print(f"[ablation]   classical_poe: acc={results['classical_poe']['accuracy']} "
          f"nll={results['classical_poe']['nll']} ece={results['classical_poe']['ece']}")

    print("[ablation] bootstrapping the entangled-vs-product difference ...")
    ring, product = results["vqc_entangled"], results["vqc_product"]
    results["entanglement_effect"] = {
        "delta_accuracy": round(ring["accuracy"] - product["accuracy"], 4),
        "delta_nll": round(ring["nll"] - product["nll"], 4),
        "delta_ece": round(ring["ece"] - product["ece"], 4),
        "bootstrap": bootstrap_delta(probs["vqc_entangled"], probs["vqc_product"],
                                     np.asarray(yte), seed=s.seed),
        "note": (
            "Entangled minus product-state, both trained by the same function on the "
            "same data with the same seed and the same parameter count. Negative "
            "delta_nll and delta_ece favour entanglement; positive delta_accuracy "
            "favours entanglement. A delta near zero is a real result and means the "
            "CNOT ring is not earning its place on this 8-channel evidence vector. "
            "Read the bootstrap intervals before reading the point estimates: the "
            "test split is small enough that a point delta can be one patient."
        ),
    }
    return results


# --------------------------------------------------------------------------- #
# Q3: what does measurement budget buy?
# --------------------------------------------------------------------------- #
def run_budget_study(model: QuantumFusion, Xte, yte, seed: int) -> dict:
    budget = QuantumMeasurementBudget(model, seed=seed)
    committed, correct_at_commit, shots, limits = 0, 0, [], Counter()
    predicted: list[int] = []

    for x, y in zip(Xte, yte):
        d = budget.decide(x)
        shots.append(d.shots_spent)
        if d.committed:
            committed += 1
            label = DIAGNOSES[int(y)]
            if d.top == (label.value if hasattr(label, "value") else str(label)):
                correct_at_commit += 1
        else:
            limits[d.limiting_factor or "unknown"] += 1
            if d.predicted_shots:
                predicted.append(d.predicted_shots)

    n = len(yte)
    abstained = n - committed
    return {
        "studies": int(n),
        "commit_rate": round(committed / n, 4) if n else 0.0,
        "abstain_rate": round(abstained / n, 4) if n else 0.0,
        "accuracy_at_commit": round(correct_at_commit / committed, 4) if committed else None,
        "median_shots_spent": int(np.median(shots)) if shots else 0,
        "mean_shots_spent": int(np.mean(shots)) if shots else 0,
        "min_shots_spent": int(np.min(shots)) if shots else 0,
        "abstention_breakdown": {
            "measurement_limited": int(limits["measurement"]),
            "model_limited": int(limits["model"]),
        },
        "median_predicted_shots_to_resolve": (int(np.median(predicted))
                                              if predicted else None),
        "note": (
            "Measurement-limited abstentions would be resolved by running the same "
            "circuit longer. Model-limited abstentions would not — the top two "
            "diagnoses are tied at infinite measurement precision, so the case needs "
            "a human. No classical fusion model can make this distinction, because "
            "classical inference has no measurement budget to vary."
        ),
    }


# --------------------------------------------------------------------------- #
# Q4: what does the entanglement encode?
# --------------------------------------------------------------------------- #
def run_coupling_study(model: QuantumFusion, Xte, top_k: int = 6) -> dict:
    from aura.services.fusion.evidence import EVIDENCE_CHANNELS

    n_q = model.n_qubits
    accumulated = np.zeros((n_q, n_q), dtype=float)
    differentials, entropies, shifts = [], [], []

    for x in Xte:
        e = measure_entanglement(model, x)
        accumulated += np.abs(e.differential)
        differentials.append(e.differential_coupling)
        entropies.append(e.measurement_entropy_bits)
        shifts.append(e.entropy_shift_bits)

    accumulated /= max(1, len(Xte))
    pairs = []
    for i in range(n_q):
        for j in range(i + 1, n_q):
            pairs.append({"channels": [EVIDENCE_CHANNELS[i], EVIDENCE_CHANNELS[j]],
                          "mean_abs_differential": round(float(accumulated[i, j]), 6)})
    pairs.sort(key=lambda p: p["mean_abs_differential"], reverse=True)

    return {
        "channels": list(EVIDENCE_CHANNELS[:n_q]),
        "mean_abs_differential_matrix": [[round(float(v), 6) for v in row]
                                         for row in accumulated],
        "top_coupled_pairs": pairs[:top_k],
        "mean_differential_coupling": round(float(np.mean(differentials)), 4),
        "mean_measurement_entropy_bits": round(float(np.mean(entropies)), 4),
        "max_entropy_bits": float(n_q),
        "mean_entropy_shift_bits": round(float(np.mean(shifts)), 4),
        "note": (
            "Averaged |C_ij(x) - C_ij(no findings)| over the held-out set. Ranks the "
            "evidence pairs this circuit consistently reasons about jointly. Coupling "
            "within the model, not causation in the patient."
        ),
    }


# --------------------------------------------------------------------------- #
def run(n: int = 900, epochs: int = 30, refresh: bool = False,
        seed: int | None = None) -> dict:
    ensure_dirs()
    s = get_settings()
    seed = s.seed if seed is None else seed
    started = time.time()

    Xtr, ytr, Xcal, ycal, Xte, yte, source = load_evidence(n, seed, refresh)
    ablation = run_ablation(Xtr, ytr, Xcal, ycal, Xte, yte, s, epochs)

    served = QuantumFusion.load()
    if served is None:
        raise RuntimeError("no trained VQC at artifacts/fusion_quantum.npz — "
                           "run `aura_cli train` first")

    print("[budget] running measurement-budgeted abstention over the test set ...")
    budget = run_budget_study(served, Xte, yte, seed)
    print(f"[budget]   commit={budget['commit_rate']} "
          f"acc@commit={budget['accuracy_at_commit']} "
          f"median shots={budget['median_shots_spent']} "
          f"breakdown={budget['abstention_breakdown']}")

    print("[coupling] measuring evidence-entanglement over the test set ...")
    coupling = run_coupling_study(served, Xte)
    print(f"[coupling]   mean differential coupling="
          f"{coupling['mean_differential_coupling']} "
          f"top pair={coupling['top_coupled_pairs'][0]['channels']}")

    report = {
        "generated_seconds": round(time.time() - started, 1),
        "data": {
            "source": source,
            "train": int(len(ytr)), "calibration": int(len(ycal)),
            "test": int(len(yte)),
            "patient_disjoint": True,
        },
        "circuit": {
            "n_qubits": int(s.n_qubits), "n_layers": int(s.n_layers),
            "device": "default.qubit (PennyLane statevector simulator)",
            "encoding": "RY(pi * x_i), one evidence channel per qubit",
            "readout": "<Z_i> per qubit -> linear head -> diagnosis logits",
        },
        "q1_q2_ablation": ablation,
        "q3_measurement_budget": budget,
        "q4_evidence_coupling": coupling,
        "honesty": {
            "hardware": "simulated; no quantum hardware was used",
            "shot_noise": (
                "Shot noise is aleatoric readout noise. More shots converge to the "
                "analytic expectation, not to the truth — a confident wrong answer "
                "gets more confident, not more correct."
            ),
            "temperature": (
                "Every model is scored at its own fitted temperature. Scoring one "
                "backend at another's temperature produced the original inflated "
                "quantum win (audit F6) and is not repeated here."
            ),
        },
    }

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {OUTPUT}")
    return report


if __name__ == "__main__":                       # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the cached evidence set")
    args = parser.parse_args()
    run(n=args.n, epochs=args.epochs, refresh=args.refresh)
