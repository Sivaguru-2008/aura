"""Figure 1 of manuscript v6: one quantity under every evaluation choice.

    python -m aura.ml.evaluation.v6_figures

Panel A — the paired difference in per-study negative log-likelihood, quantum circuit minus
comparator (negative favours quantum), with 95% t-intervals over studies, under each
evaluation choice the paper examines, in the order a reader meets them.
Panel B — each audited published claim: the margin the paper reported (hollow) against the
symmetric-protocol difference with its Nadeau–Bengio 95% CI (filled).

Every number is read from a released artifact or recomputed from per-study predictions;
nothing is typed in except the published claims' reported figures, which are quoted from the
papers via ``bench/field_audit/results/*.json``. Writes ``docs/figures/fig_v6_forest.{png,pdf}``
and ``aura/artifacts/v6_forest.json`` (the plotted values, for the drift test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from aura.common.config import ARTIFACTS

ROOT = Path(__file__).resolve().parents[3]
FIGDIR = ROOT / "docs" / "figures"
FA = ROOT / "bench" / "field_audit" / "results"


def _pnll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def _diff(path, a="quantum", b="classical"):
    d = np.load(path)
    y = d["yte"]
    x = _pnll(d[f"probs_{a}"], y) - _pnll(d[f"probs_{b}"], y)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se)
    return {"delta": float(m), "ci95": [float(lo), float(hi)], "n": int(len(x))}


def _external_rows():
    """Frozen-transfer per-study predictions, recomputed exactly as chexpert_external does."""
    from aura.common.mathx import softmax
    from aura.ml.evaluation import chexpert_external as ce
    from aura.ml.evaluation.fusion_baselines import (_converged_logistic, _served_classical,
                                                     _served_learnable, _served_quantum)
    from aura.ml.evaluation.quantum_study import load_evidence
    from aura.services.safety.calibration import fit_temperature

    d = np.load(ce.EVIDENCE)
    X, y = d["X"], d["y_u0"]
    perm = np.random.default_rng(ce.SPLIT_SEED).permutation(len(X))
    n_cal = int(ce.CAL_FRACTION * len(X))
    ci, ti = perm[:n_cal], perm[n_cal:]
    P = {}
    for name, fn in (("classical", _served_classical), ("quantum", _served_quantum),
                     ("learnable", _served_learnable)):
        Lc, Lt, _ = fn(X[ci], X[ti])
        T = fit_temperature(Lc, y[ci])
        P[name] = np.array([softmax(r / T) for r in Lt])
    Xtr, ytr, Xc, yc, _, _, _ = load_evidence(n=1338, seed=7)
    _, _, _, model, _, _ = _converged_logistic(Xtr, ytr, Xc, yc, Xc)
    Lc, Lt = model.decision_function(X[ci]), model.decision_function(X[ti])
    T = fit_temperature(Lc, y[ci])
    P["logistic"] = np.array([softmax(r / T) for r in Lt])
    out = PART_EXT
    np.savez(out, yte=y[ti], **{f"probs_{k}": v for k, v in P.items()})
    return out


PART_EXT = ARTIFACTS / "v6" / "external_predictions.npz"


def rows_panel_a():
    R = []

    def add(group, label, path, b):
        if Path(path).exists():
            R.append({"group": group, "label": label, **_diff(path, "quantum", b)})

    pub = ARTIFACTS / "fusion_baselines_predictions.npz"
    mat = ARTIFACTS / "fusion_matched_optimiser_predictions.npz"
    oos = ARTIFACTS / "chexpert_internal" / "E1_predictions.npz"
    add("MIMIC-CXR, each arm's own optimiser (n=447)", "vs product-of-experts", pub, "classical")
    add("MIMIC-CXR, each arm's own optimiser (n=447)", "vs gated head", pub, "learnable")
    add("MIMIC-CXR, each arm's own optimiser (n=447)", "vs logistic regression", pub, "logistic")
    add("MIMIC-CXR, matched optimiser (n=447)", "vs product-of-experts", mat, "classical")
    add("MIMIC-CXR, matched optimiser (n=447)", "vs gated head", mat, "learnable")
    add("CheXpert, out-of-sample evidence (n=1,600)", "vs product-of-experts", oos, "classical")
    add("CheXpert, out-of-sample evidence (n=1,600)", "vs gated head", oos, "learnable")
    if not PART_EXT.exists():
        _external_rows()
    add("CheXpert, frozen transfer (n=4,800)", "vs product-of-experts", PART_EXT, "classical")
    add("CheXpert, frozen transfer (n=4,800)", "vs logistic regression", PART_EXT, "logistic")
    v6 = ARTIFACTS / "v6" / "C1_y_u0_probs.npz"
    if v6.exists():
        n = len(np.load(v6)["yte"])
        g = f"Held-out patients, scored without selection (n={n:,})"
        add(g, "matched optimiser, vs product-of-experts", v6, "classical")
        add(g, "matched optimiser, vs gated head", v6, "learnable")
        conf = ARTIFACTS / "v6_confirmatory.json"
        if conf.exists() and "C2" in json.loads(conf.read_text()):
            c2 = json.loads(conf.read_text())["C2"]["primary"]
            t = c2["tost"]
            R.append({"group": g, "label": "36-config search per arm, vs product-of-experts",
                      "delta": t["mean_difference"], "ci95": c2["ci95"], "n": t["n"]})
    return R


#: the margin each paper reported over the strongest comparator it reports beating
REPORTED = {
    "ckd": ("CKD (PeerJ CS 2025)", 0.992 - 0.963, "accuracy"),
    "glioma": ("Glioma grade (Sci Rep 2025)", 0.83 - 0.82, "accuracy"),
    "stack_parkinsons": ("Parkinson's (Sci Rep 2026)", 0.813 - 0.585, "specificity"),
    "stack_pima": ("Diabetes (Sci Rep 2026)", 0.621 - 0.553, "sensitivity"),
    "eeg_qsvm": ("Schizophrenia EEG (J Med Syst 2024)", 1.00 - 1.00, "accuracy"),
    "heart_qnn": ("Heart disease (Sci Rep 2026)", 0.8272 - 0.8094, "accuracy"),
}


def rows_panel_b():
    R = []
    for key, (label, rep, metric) in REPORTED.items():
        p = FA / f"{key}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        tag = "full" if "full" in d["symmetric"] else "train_10pct"
        v = d["symmetric"][tag]["verdict"]
        dd = v["difference_quantum_minus_classical"]
        R.append({"label": f"{label}\n{metric}" + ("" if tag == "full" else ", 10% training"),
                  "reported": rep, "delta": dd["mean"], "ci95": dd["ci95"],
                  "verdict": v["verdict"]})
    return R


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False})
    A, B = rows_panel_a(), rows_panel_b()
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(A) + 6, len(B) + 2], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    y, ticks, labels, last = 0, [], [], None
    for r in A:
        if r["group"] != last:
            y -= 0.6
            ax.text(-0.36, y, r["group"], fontsize=7.5, fontweight="bold", va="center",
                    transform=ax.get_yaxis_transform(), ha="left")
            y -= 1
            last = r["group"]
        lo, hi = r["ci95"]
        favours_q = hi < 0
        col = "#B23A48" if favours_q else ("#2F5D8A" if lo > 0 else "#555555")
        ax.plot([lo, hi], [y, y], color=col, lw=1.4)
        ax.plot(r["delta"], y, "s", color=col, ms=4)
        ticks.append(y)
        labels.append(r["label"])
        y -= 1
    ax.axvspan(-0.01, 0.01, color="#999999", alpha=0.15, lw=0)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ negative log-likelihood, quantum − comparator (nats)    "
                  "← favours quantum | favours classical →")
    ax.set_title("A  Fusion circuit vs classical comparators, under each evaluation choice",
                 loc="left", fontsize=9, fontweight="bold", pad=16)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25)

    bx = fig.add_subplot(gs[1])
    for i, r in enumerate(B):
        yy = -i
        lo, hi = r["ci95"]
        bx.plot([lo, hi], [yy, yy], color="#555555", lw=1.4)
        bx.plot(r["delta"], yy, "o", color="#555555", ms=4.5,
                label="symmetric protocol, 95% CI" if i == 0 else None)
        bx.plot(r["reported"], yy, "o", mfc="white", mec="#B23A48", ms=5.5, mew=1.2,
                label="margin reported in the paper" if i == 0 else None)
    bx.axvline(0, color="black", lw=0.7)
    bx.set_yticks([-i for i in range(len(B))])
    bx.set_yticklabels([r["label"] for r in B], fontsize=7)
    bx.set_xlabel("Quantum − classical, on each paper's primary metric "
                  "← favours classical | favours quantum →")
    bx.set_title("B  Published claims of quantum benefit: reported vs re-tested",
                 loc="left", fontsize=9, fontweight="bold", pad=16)
    bx.legend(loc="lower right", fontsize=7)
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="x", alpha=0.25)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v6_forest.{ext}", bbox_inches="tight")
    (ARTIFACTS / "v6_forest.json").write_text(json.dumps({"panel_a": A, "panel_b": B},
                                                         indent=2, default=float))
    print(json.dumps({"panel_a": A, "panel_b": B}, indent=1, default=float)[:3000])




def mechanism_figure():
    """Figure 2: interaction-order spectrum of the trained circuits (M1) and the glioma
    kernel's bandwidth sweep (M3). Reads ``v6_mechanism.json``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = json.loads((ARTIFACTS / "v6_mechanism.json").read_text())
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"wspace": 0.55})

    if "M1" in M:
        rows = M["M1"]["C1b_7q"]
        orders = list(range(0, 8))
        for ent, col, lab in (("ring", "#B23A48", "CNOT ring (entangled)"),
                              ("none", "#2F5D8A", "no entangler (product)")):
            E = np.array([[r[ent]["coefficient_energy_share_by_order"].get(str(o), 0.0)
                           for o in orders] for r in rows if ent in r])
            m = E.mean(0)
            a.bar(np.array(orders) + (-0.2 if ent == "ring" else 0.2), m, width=0.4,
                  color=col, label=f"{lab}, 10 seeds")
        a.set_xlabel("interaction order of term (number of inputs it multiplies)")
        a.set_ylabel("share of readout coefficient energy")
        a.set_title("A  Interaction spectrum (7 clinical channels, 10 seeds)", loc="left",
                    fontweight="bold", fontsize=8.5, pad=10)
        a.legend(fontsize=7, frameon=False)
        a.set_xticks(orders)

    sw = M["M3"]["glioma_qkl"]["bandwidth_sweep"]
    bws = sorted(sw, key=float)
    x = [float(k) for k in bws]
    b.plot(x, [sw[k]["kta_quantum"] for k in bws], "o-", color="#B23A48",
           label="kernel–target alignment")
    b2 = b.twinx()
    b2.plot(x, [sw[k]["effective_dimension_quantum"] for k in bws], "s--", color="#555555",
            label="effective dimension")
    b2.set_yscale("log")
    b2.spines["top"].set_visible(False)
    b.axvline(0.25, color="black", lw=0.7, ls=":")
    b.text(0.27, 0.25, "selected in\nall 5 folds", fontsize=7)
    b.set_xscale("log", base=2)
    b.set_xlabel("quantum-kernel bandwidth")
    b.set_ylabel("alignment with grade labels", color="#B23A48")
    b2.set_ylabel("effective dimension (log)", color="#555555")
    b.set_title("B  10-qubit glioma kernel across bandwidths", loc="left",
                fontweight="bold", fontsize=8.5, pad=10)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.18, wspace=0.55)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v6_mechanism.{ext}", bbox_inches="tight")
        fig.savefig(ROOT.parent / f"fig_v6_mechanism.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mechanism":
        mechanism_figure()
    else:
        main()
