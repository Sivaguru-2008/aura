"""External validation of the fusion comparison on CheXpert.

Governed by ``docs/PREREGISTRATION_CHEXPERT.md``, deposited at commit fa8c2b6
before any CheXpert image was scored. Read that document before changing
anything here: the sampling rule, the freeze/refit boundary, the label map, the
uncertainty policy and the test family were all fixed in advance.

What this does
--------------
1. Scans ``train.csv`` in file order, taking the highest-numbered study with a
   frontal view for each patient, up to ``--patients``.
2. Scores every study through the *served* seam — ``VisionEngine.analyze``
   followed by ``evidence.encode`` — so preprocessing and per-finding Platt
   calibration are identical to production and to the MIMIC path.
3. Derives the six-way target from CheXpert's label columns with the identical
   weighted-score map and acuity tie-break used on MIMIC (``mimic/labeling.py``).
   COPD is unreachable: the map reaches it only through hyperinflation, which
   CheXpert does not label.
4. Refits **only** a per-arm temperature on the CheXpert calibration split (P1)
   and evaluates all four frozen backends on the test split.

Writes ``artifacts/chexpert_external.json`` and caches the extracted evidence to
``artifacts/chexpert_evidence.npz`` so the analysis can be re-run without
re-scoring images.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS, ensure_dirs
from aura.common.mathx import softmax
from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis
from aura.schemas.contracts import StructuredPriors
from aura.services.safety.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from aura.ml.evaluation.fusion_baselines import (
    _converged_logistic,
    _jeffreys,
    _macro_auroc,
    _mcnemar,
    _holm,
    _per_study_nll,
    _served_classical,
    _served_learnable,
    _served_quantum,
)
from aura.ml.evaluation.quantum_study import load_evidence

CHEXPERT_ROOT = Path(r"E:/AURA")
OUTPUT = ARTIFACTS / "chexpert_external.json"
EVIDENCE = ARTIFACTS / "chexpert_evidence.npz"

CAL_FRACTION = 0.40
SPLIT_SEED = 7

#: Pre-registered concept -> CheXpert column map. `hyperinflation` is absent by design.
CONCEPT_COLUMN = {
    "pneumothorax": "Pneumothorax",
    "nodule": "Lung Lesion",
    "edema": "Edema",
    "cardiomegaly": "Cardiomegaly",
    "effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "consolidation": "Consolidation",
    "opacity": "Lung Opacity",
}

_DX_PRIORITY = [
    Diagnosis.PNEUMOTHORAX, Diagnosis.MALIGNANCY, Diagnosis.HEART_FAILURE,
    Diagnosis.PNEUMONIA, Diagnosis.COPD, Diagnosis.NORMAL,
]


# --------------------------------------------------------------------------- #
# Label derivation — mirrors mimic/labeling.py::_map_diagnosis exactly
# --------------------------------------------------------------------------- #
def _positive(cell: str, uncertain_as: int) -> int:
    """CheXpert cells: 1.0 positive, 0.0 negative, -1.0 uncertain, blank absent."""
    if cell is None or cell == "":
        return 0
    v = float(cell)
    if v == 1.0:
        return 1
    if v == -1.0:
        return uncertain_as
    return 0


def map_diagnosis(row: dict, uncertain_as: int) -> int:
    concepts = {c: _positive(row.get(col, ""), uncertain_as)
                for c, col in CONCEPT_COLUMN.items()}
    normal_cue = _positive(row.get("No Finding", ""), uncertain_as) == 1

    scores = {d: 0.0 for d in CHEST_DIAGNOSES}
    if concepts["pneumothorax"]:
        scores[Diagnosis.PNEUMOTHORAX] += 1.0
    if concepts["nodule"]:
        scores[Diagnosis.MALIGNANCY] += 1.0
    if concepts["edema"]:
        scores[Diagnosis.HEART_FAILURE] += 0.7
    if concepts["cardiomegaly"]:
        scores[Diagnosis.HEART_FAILURE] += 0.5
    if concepts["effusion"]:
        scores[Diagnosis.HEART_FAILURE] += 0.2
    if concepts["pneumonia"]:
        scores[Diagnosis.PNEUMONIA] += 0.9
    if concepts["consolidation"]:
        scores[Diagnosis.PNEUMONIA] += 0.6
    if concepts["opacity"]:
        scores[Diagnosis.PNEUMONIA] += 0.3
    # hyperinflation has no CheXpert column, so COPD is unreachable (pre-registered)

    if max(scores.values()) <= 0.0:
        return CHEST_DIAGNOSES.index(Diagnosis.NORMAL)
    top = max(scores.values())
    for d in _DX_PRIORITY:
        if scores[d] == top:
            return CHEST_DIAGNOSES.index(d)
    return CHEST_DIAGNOSES.index(Diagnosis.NORMAL)


# --------------------------------------------------------------------------- #
# Cohort assembly
# --------------------------------------------------------------------------- #
def select_studies(n_patients: int) -> list[dict]:
    """One study per patient — the highest-numbered study carrying a frontal view.

    Rows are consumed in file order, as pre-registered, so there is no sampling
    seed and no opportunity to reselect.
    """
    best: dict[str, tuple[int, dict]] = {}
    order: list[str] = []
    with open(CHEXPERT_ROOT / "train.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Frontal/Lateral"] != "Frontal":
                continue
            parts = row["Path"].split("/")
            patient, study = parts[2], parts[3]
            if patient not in best:
                if len(best) >= n_patients:
                    continue
                order.append(patient)
            idx = int(study.replace("study", ""))
            if patient not in best or idx > best[patient][0]:
                best[patient] = (idx, row)
    return [best[p][1] for p in order]


def extract(rows: list[dict], uncertain_as: int) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    from aura.services.fusion.evidence import encode
    from aura.services.vision.engine import VisionEngine

    vision = VisionEngine.load()
    X, y = [], []
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        rel = row["Path"].split("CheXpert-v1.0-small/", 1)[-1]
        path = CHEXPERT_ROOT / rel
        if not path.is_file():
            continue
        try:
            img = np.array(Image.open(path).convert("L"), dtype=float)
        except OSError:
            continue
        vr = vision.analyze(f"chexpert-{i}", img)
        X.append(encode(vr, StructuredPriors()))
        y.append(map_diagnosis(row, uncertain_as))
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  [{i + 1}/{len(rows)}] {rate:.1f} img/s", flush=True)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


# --------------------------------------------------------------------------- #
def evaluate_cohort(Xcal, ycal, Xte, yte, Xtr_mimic, ytr_mimic,
                    Xcal_mimic, ycal_mimic) -> dict:
    """All four frozen backends, temperature refit on the CheXpert calibration split."""
    arms = {}

    def add(name, Lcal, Lte, params):
        T = fit_temperature(Lcal, ycal)
        P = np.array([softmax(r / T) for r in Lte])
        correct = P.argmax(axis=1) == yte
        nll = _per_study_nll(P, yte)
        pred = P.argmax(axis=1)
        arms[name] = {
            "n": int(len(yte)),
            "trainable_parameters": params,
            "temperature": round(float(T), 4),
            "accuracy": round(float(correct.mean()), 4),
            "correct": int(correct.sum()),
            "accuracy_ci95_jeffreys": _jeffreys(int(correct.sum()), len(yte)),
            "nll": round(float(nll.mean()), 4),
            "ece": round(float(expected_calibration_error(P, yte)), 4),
            "macro_auroc": round(_macro_auroc(P, yte), 4),
            "copd_predictions": int((pred == CHEST_DIAGNOSES.index(Diagnosis.COPD)).sum()),
            "_correct": correct,
            "_nll": nll,
        }

    Lc, Lt, p = _served_classical(Xcal, Xte)
    add("classical", Lc, Lt, p)
    Lc, Lt, p = _served_quantum(Xcal, Xte)
    add("quantum", Lc, Lt, p)
    Lc, Lt, p = _served_learnable(Xcal, Xte)
    add("learnable", Lc, Lt, p)

    # The logistic arm is fitted on MIMIC only, then frozen — same as the others.
    from sklearn.linear_model import LogisticRegression
    _, _, p, model, _, _ = _converged_logistic(
        Xtr_mimic, ytr_mimic, Xcal_mimic, ycal_mimic, Xcal_mimic
    )
    add("logistic", model.decision_function(Xcal), model.decision_function(Xte), p)

    counts = np.bincount(yte, minlength=len(CHEST_DIAGNOSES))
    floor = {
        "accuracy": round(float(counts.max() / len(yte)), 4),
        "class": CHEST_DIAGNOSES[int(counts.argmax())].value,
        "support": {CHEST_DIAGNOSES[i].value: int(counts[i])
                    for i in range(len(CHEST_DIAGNOSES))},
    }

    names = ["classical", "quantum", "learnable", "logistic"]
    pairwise, acc_p, nll_p = {}, {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}_vs_{b}"
            A, B = arms[a], arms[b]
            mc = _mcnemar(A["_correct"], B["_correct"])
            d = A["_nll"] - B["_nll"]
            try:
                w = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                w = 1.0
            pairwise[key] = {
                "delta_accuracy": round(float(A["accuracy"] - B["accuracy"]), 4),
                "mcnemar": mc,
                "delta_nll": round(float(A["nll"] - B["nll"]), 4),
                "nll_wilcoxon_p": float(f"{w:.3e}"),
            }
            acc_p[key], nll_p[key] = mc["p"], w
    for k, v in _holm(acc_p).items():
        pairwise[k]["mcnemar_p_holm"] = v
    for k, v in _holm(nll_p).items():
        pairwise[k]["nll_wilcoxon_p_holm"] = float(f"{v:.3e}")

    return {
        "majority_class_floor": floor,
        "backends": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in arms.items()},
        "pairwise": pairwise,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=8000)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    started = time.time()

    if EVIDENCE.exists() and not args.refresh:
        d = np.load(EVIDENCE)
        X, y_u0, y_u1 = d["X"], d["y_u0"], d["y_u1"]
        print(f"[data] reusing cached CheXpert evidence, n={len(y_u0)}")
    else:
        rows = select_studies(args.patients)
        print(f"[data] {len(rows)} patient-disjoint frontal studies selected")
        X, y_u0 = extract(rows, uncertain_as=0)
        y_u1 = np.asarray([map_diagnosis(r, 1) for r in rows[:len(y_u0)]], dtype=int)
        np.savez(EVIDENCE, X=X, y_u0=y_u0, y_u1=y_u1)
        print(f"[data] cached {len(y_u0)} studies to {EVIDENCE.name}")

    Xtr_m, ytr_m, Xcal_m, ycal_m, _, _, _ = load_evidence(n=1338, seed=7)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(X))
    n_cal = int(CAL_FRACTION * len(X))
    cal_idx, te_idx = perm[:n_cal], perm[n_cal:]

    result = {
        "purpose": (
            "External validation of the Section 8.2 fusion comparison on CheXpert "
            "(Stanford) with every model frozen and only per-arm temperature refit, "
            "under docs/PREREGISTRATION_CHEXPERT.md deposited at commit fa8c2b6."
        ),
        "preregistration": "docs/PREREGISTRATION_CHEXPERT.md (commit fa8c2b6)",
        "data": {
            "corpus": "CheXpert-v1.0-small train split",
            "patients_requested": args.patients,
            "studies_scored": int(len(X)),
            "selection": "one study per patient, highest-numbered with a frontal view, file order",
            "calibration_studies": int(len(cal_idx)),
            "test_studies": int(len(te_idx)),
            "patient_disjoint": True,
            "constant_channels": [int(i) for i in range(X.shape[1])
                                  if np.all(X[:, i] == 0)],
            "copd_derivable": False,
            "copd_note": ("The diagnosis map reaches COPD only through hyperinflation, "
                          "which CheXpert does not label. Five of six categories are "
                          "reachable; a COPD prediction can only be an error."),
        },
        "frozen": ["densenet121 + per-finding platt", "evidence encoder",
                   "all four fusion backends including the logistic regression"],
        "refit_on_chexpert": ["one temperature per backend (P1)"],
        "primary_uncertain_as_negative": evaluate_cohort(
            X[cal_idx], y_u0[cal_idx], X[te_idx], y_u0[te_idx],
            Xtr_m, ytr_m, Xcal_m, ycal_m),
        "sensitivity_uncertain_as_positive": evaluate_cohort(
            X[cal_idx], y_u1[cal_idx], X[te_idx], y_u1[te_idx],
            Xtr_m, ytr_m, Xcal_m, ycal_m),
    }

    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for label in ("primary_uncertain_as_negative", "sensitivity_uncertain_as_positive"):
        blk = result[label]
        print(f"\n=== {label} (floor {blk['majority_class_floor']['accuracy']}) ===")
        for k, v in blk["backends"].items():
            print(f"  {k:10s} p={v['trainable_parameters']:4d} T={v['temperature']:.4f} "
                  f"acc={v['accuracy']:.4f} nll={v['nll']:.4f} ece={v['ece']:.4f} "
                  f"auroc={v['macro_auroc']:.4f} copd_preds={v['copd_predictions']}")
        for k, v in blk["pairwise"].items():
            print(f"  {k:24s} dAcc={v['delta_accuracy']:+.4f} McNemar Holm={v['mcnemar_p_holm']:.4f}"
                  f" | dNLL={v['delta_nll']:+.4f} Wilcoxon Holm={v['nll_wilcoxon_p_holm']:.3e}")
    print(f"\n[done] wrote {OUTPUT} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
