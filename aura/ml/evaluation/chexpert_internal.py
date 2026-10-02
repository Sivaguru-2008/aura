"""Fusion comparisons trained and tested within CheXpert evidence (out-of-sample for the
image model), under ``docs/PREREGISTRATION_ADDENDUM_2026-10b_CHEXPERT_INTERNAL.md``.

    python -m aura.ml.evaluation.chexpert_internal e1        # matched-optimiser comparison
    python -m aura.ml.evaluation.chexpert_internal e2        # 7-qubit entanglement ablation
    python -m aura.ml.evaluation.chexpert_internal summarize

E1 runs ``matched_optimiser.main`` itself with only the data source swapped, so the arms,
grid, stopping rule and tests are the published ones, not a re-implementation.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from aura.common.config import ARTIFACTS, ensure_dirs

EVIDENCE = ARTIFACTS / "chexpert_evidence.npz"
PART = ARTIFACTS / "chexpert_internal"
OUT = ARTIFACTS / "chexpert_internal.json"
SPLIT_SEED = 7


def splits():
    d = np.load(EVIDENCE)
    X, y = d["X"], d["y_u0"]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(X))
    n_tr, n_cal = int(0.6 * len(X)), int(0.2 * len(X))
    tr, cal, te = perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]
    return X[tr], y[tr], X[cal], y[cal], X[te], y[te]


def run_e1():
    from aura.ml.evaluation import matched_optimiser as mo

    data = splits()
    mo.load_evidence = lambda **_: (*data, "chexpert-internal")
    mo.OUTPUT = PART / "E1_matched_optimiser.json"
    mo.PREDICTIONS = PART / "E1_predictions.npz"
    mo.main()


def run_e2():
    from aura.ml.evaluation import revision_v51 as rv

    Xtr, ytr, Xcal, ycal, Xte, yte = splits()
    Xtr, Xcal, Xte = (A[:, :rv.N_QUBITS] for A in (Xtr, Xcal, Xte))
    rows, Pr, Pp = [], [], []
    for seed in rv.SEEDS:
        print(f"[E2] seed {seed}", flush=True)
        mr, P_r = rv._plain_fit(Xtr, ytr, Xcal, ycal, Xte, yte, seed, "ring")
        mp, P_p = rv._plain_fit(Xtr, ytr, Xcal, ycal, Xte, yte, seed, "none")
        dn = round(mr["nll"] - mp["nll"], 4)
        print(f"    ring {mr['nll']:.4f} prod {mp['nll']:.4f} d={dn:+.4f}", flush=True)
        rows.append({"seed": seed, "entangled": mr, "product": mp, "delta_nll": dn})
        Pr.append(P_r); Pp.append(P_p)
    np.savez(PART / "E2_probs.npz", yte=yte, ring=np.array(Pr), none=np.array(Pp),
             seeds=np.array(rv.SEEDS))
    out = {"per_seed": rows, "seed_level": rv._seed_stats([r["delta_nll"] for r in rows])}
    (PART / "E2_ablation.json").write_text(json.dumps(out, indent=2))


def summarize():
    from aura.ml.evaluation import revision_v51 as rv

    out = {"preregistration": "docs/PREREGISTRATION_ADDENDUM_2026-10b_CHEXPERT_INTERNAL.md",
           "data": {"source": "chexpert_evidence.npz, y_u0", "split_seed": SPLIT_SEED,
                    "train/cal/test": [len(a) for a in splits()[1::2]]}}
    e1 = PART / "E1_matched_optimiser.json"
    if e1.exists():
        out["E1"] = json.loads(e1.read_text())
        d = np.load(PART / "E1_predictions.npz")
        diff = (rv._per_study_nll(d["probs_quantum"], d["yte"])
                - rv._per_study_nll(d["probs_classical"], d["yte"]))
        out["E1"]["tost_quantum_vs_poe"] = rv.tost(diff, rv.MARGIN)
    e2 = PART / "E2_ablation.json"
    if e2.exists():
        out["E2"] = json.loads(e2.read_text())
        out["E2"]["study_bootstrap"] = rv._study_bootstrap(PART / "E2_probs.npz")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ensure_dirs()
    PART.mkdir(parents=True, exist_ok=True)
    {"e1": run_e1, "e2": run_e2, "summarize": summarize}[sys.argv[1]]()
