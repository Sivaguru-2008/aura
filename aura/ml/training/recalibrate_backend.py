"""Fit temperature, conformal threshold, and OOD statistics per fusion backend.

Why this exists
---------------
``artifacts/safety.npz`` holds one temperature. That was fine while one backend was
served, and became a latent bug the moment a second one was selectable: the shipped
file held the **classical** temperature (0.4574), so selecting the quantum backend
scored the VQC's logits with a constant fitted for a different model.

The two are not close. On this system the classical product-of-experts needs
T ~ 0.46 and the VQC needs T ~ 0.94 — a factor of two. Applying the wrong one
systematically distorts every probability, every conformal set, and every abstention
threshold downstream, and it does so *silently*, because the output still looks like a
well-formed posterior.

This script fits all three calibration quantities separately for each backend and
writes ``artifacts/safety_<backend>.npz``, which
:meth:`aura.services.safety.calibration.Calibration.load` prefers when the corresponding
backend is served. The shared ``safety.npz`` is left in place for the default backend
so nothing that reads it breaks.

Run after any fusion retrain::

    python -m ml.training.recalibrate_backend
    python -m ml.training.recalibrate_backend --backends quantum
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES
from aura.services.fusion.classical import ClassicalFusion
from aura.services.fusion.learnable import LearnableFusion
from aura.services.fusion.quantum import QuantumFusion
from aura.services.safety.calibration import (
    Calibration,
    expected_calibration_error,
    fit_conformal,
    fit_temperature,
    ood_stats,
)

N_DX = len(DIAGNOSES)
EVIDENCE_CACHE = ARTIFACTS / "quantum_study_evidence.npz"
REPORT = ARTIFACTS / "backend_calibration.json"


def _load_backend(name: str):
    return {
        "quantum": QuantumFusion.load,
        "classical": ClassicalFusion.load,
        "learnable": LearnableFusion.load,
    }[name]()


def _evidence_splits(n: int, seed: int):
    """Reuse the cached study evidence when present.

    Sharing the cache with ``ml.evaluation.quantum_study`` is deliberate: the
    calibration split and the split the ablation reports on must be the same
    patients, or the published metrics describe a model calibrated on different data
    than the one that was measured.
    """
    if EVIDENCE_CACHE.exists():
        d = np.load(EVIDENCE_CACHE)
        print(f"[data] reusing {EVIDENCE_CACHE.name} "
              f"(cal={len(d['ycal'])} test={len(d['yte'])})")
        return d["Xcal"], d["ycal"], d["Xte"], d["yte"], str(d["source"])

    from .dataset import real_evidence_splits

    print(f"[data] building evidence set (n={n}) ...")
    real = real_evidence_splits(n=n, split="train", seed=seed,
                               per_class_cap=max(60, n // N_DX))
    if real is None:
        raise RuntimeError(
            "no cached evidence set and MIMIC-CXR is unavailable; run "
            "`python -m ml.evaluation.quantum_study` first to build the cache")
    _, _, Xcal, ycal, Xte, yte = real
    return Xcal, ycal, Xte, yte, "mimic-cxr"


def calibrate(backend: str, Xcal, ycal, Xte, yte, coverage: float) -> dict | None:
    model = _load_backend(backend)
    if model is None:
        print(f"[skip] {backend}: no trained artifact")
        return None

    cal_logits = np.array([model.logits(x) for x in Xcal], dtype=float)
    test_logits = np.array([model.logits(x) for x in Xte], dtype=float)

    T = fit_temperature(cal_logits, ycal)
    cal_probs = np.array([softmax(r / T) for r in cal_logits])
    qhat = fit_conformal(cal_probs, ycal, coverage)
    om, osd = ood_stats(cal_logits, T)

    test_probs = np.array([softmax(r / T) for r in test_logits])
    ece = float(expected_calibration_error(test_probs, yte))
    accuracy = float((test_probs.argmax(1) == yte).mean())
    nll = float(-np.log(np.clip(test_probs[np.arange(len(yte)), yte], 1e-12, 1)).mean())

    Calibration(temperature=T, conformal_qhat=qhat, coverage=coverage,
                ood_mean=om, ood_std=osd, ece=ece).save(backend=backend)

    # Mean conformal set size at this qhat — the number that actually decides how
    # often the safety engine abstains, and the one a temperature error moves most.
    set_sizes = (1.0 - test_probs <= qhat).sum(axis=1)
    result = {
        "backend": backend,
        "temperature": round(T, 4),
        "conformal_qhat": round(qhat, 4),
        "coverage_target": coverage,
        "empirical_coverage": round(float(
            (1.0 - test_probs[np.arange(len(yte)), yte] <= qhat).mean()), 4),
        "mean_conformal_set_size": round(float(set_sizes.mean()), 3),
        "ood_mean": round(om, 4),
        "ood_std": round(osd, 4),
        "test_accuracy": round(accuracy, 4),
        "test_nll": round(nll, 4),
        "test_ece": round(ece, 4),
    }
    print(f"[fit] {backend:10s} T={result['temperature']:<7} "
          f"qhat={result['conformal_qhat']:<7} ece={result['test_ece']:<7} "
          f"acc={result['test_accuracy']:<7} "
          f"coverage={result['empirical_coverage']} "
          f"set_size={result['mean_conformal_set_size']}")
    return result


def run(backends: tuple[str, ...] = ("classical", "quantum", "learnable"),
        n: int = 900) -> dict:
    ensure_dirs()
    s = get_settings()
    Xcal, ycal, Xte, yte, source = _evidence_splits(n, s.seed)

    results = {}
    for backend in backends:
        fitted = calibrate(backend, Xcal, ycal, Xte, yte, s.conformal_coverage)
        if fitted:
            results[backend] = fitted

    temperatures = {b: r["temperature"] for b, r in results.items()}
    report = {
        "data_source": source,
        "calibration_studies": int(len(ycal)),
        "test_studies": int(len(yte)),
        "backends": results,
        "temperature_spread": (round(max(temperatures.values())
                                     / min(temperatures.values()), 2)
                               if len(temperatures) > 1 else 1.0),
        "why": (
            "Each backend is calibrated on its own logits. The temperature spread "
            "across backends is the factor by which serving one backend with "
            "another's calibration would distort every downstream probability, "
            "conformal set, and abstention decision."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {REPORT}")
    print(f"[done] temperature spread across backends: "
          f"{report['temperature_spread']}x")
    return report


if __name__ == "__main__":                       # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="*",
                        default=["classical", "quantum", "learnable"])
    parser.add_argument("--n", type=int, default=900)
    args = parser.parse_args()
    run(tuple(args.backends), n=args.n)
