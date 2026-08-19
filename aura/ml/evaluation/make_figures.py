"""Regenerate the manuscript figures that had no generator.

Figures 1-4 were produced ad hoc and no script for them survives in the
repository, which is a P6 gap of its own. This module covers the two figures
added at v4.3 and is written so the earlier four can be folded into it.

  * Figure 5 - external validation on CheXpert (Section 8.6).
  * Figure 6 - risk-coverage curves and the matched-coverage control (Section 9.1).

Both read from released artifacts; nothing is recomputed from raw data except
the risk-coverage curves, which need per-study probabilities.

Writes ``docs/figures/fig5_external_validation.{png,pdf}`` and
``docs/figures/fig6_risk_coverage.{png,pdf}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aura.common.config import ARTIFACTS

FIGDIR = Path(__file__).resolve().parents[3] / "docs" / "figures"

ARMS = ["classical", "quantum", "learnable", "logistic"]
LABEL = {
    "classical": "Product-of-experts\n(54 params)",
    "quantum": "Quantum VQC\n(102 params)",
    "learnable": "Gated head\n(126 params)",
    "logistic": "Logistic regression\n(54 params)",
}
SHORT = {"classical": "PoE", "quantum": "VQC", "learnable": "Gated", "logistic": "LogReg"}
COLOR = {"classical": "#4C72B0", "quantum": "#C44E52",
         "learnable": "#8172B2", "logistic": "#55A868"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    return plt


# --------------------------------------------------------------------------- #
def figure_external(plt) -> None:
    ext = json.loads((ARTIFACTS / "chexpert_external.json").read_text(encoding="utf-8"))
    base = json.loads((ARTIFACTS / "fusion_baselines_n447.json").read_text(encoding="utf-8"))
    prim = ext["primary_uncertain_as_negative"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    # -- Panel A: accuracy, internal vs external, each against its own floor -- #
    x = np.arange(len(ARMS))
    w = 0.36
    mimic = [base["backends"][a]["accuracy"] for a in ARMS]
    chex = [prim["backends"][a]["accuracy"] for a in ARMS]
    ax1.bar(x - w / 2, mimic, w, label="MIMIC-CXR (n = 447)",
            color=[COLOR[a] for a in ARMS], alpha=0.55, edgecolor="none")
    ax1.bar(x + w / 2, chex, w, label="CheXpert (n = 4,800)",
            color=[COLOR[a] for a in ARMS], edgecolor="none")
    ax1.axhline(base["majority_class_floor"]["accuracy"], ls="--", lw=1.1,
                color="0.35")
    ax1.axhline(prim["majority_class_floor"]["accuracy"], ls=":", lw=1.4, color="0.15")
    ax1.text(3.52, base["majority_class_floor"]["accuracy"] + 0.006, "MIMIC floor",
             ha="right", fontsize=7.5, color="0.35")
    ax1.text(3.52, prim["majority_class_floor"]["accuracy"] + 0.006, "CheXpert floor",
             ha="right", fontsize=7.5, color="0.15")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABEL[a] for a in ARMS], fontsize=7.5)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 0.68)
    ax1.set_title("A  Accuracy transfers unevenly", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, fc="0.5")]
    ax1.legend(handles, ["MIMIC-CXR (n = 447)", "CheXpert (n = 4,800)"],
               loc="upper left", fontsize=7.5)

    # -- Panel B: NLL degradation against capacity -- #
    params = [base["trainable_parameters"][a] for a in ARMS]
    dnll = [prim["backends"][a]["nll"] - base["backends"][a]["nll"] for a in ARMS]
    # PoE and LogReg share an x of 54, so PoE's label goes below to avoid the collision
    offsets = {"classical": (0, -19), "logistic": (0, 11),
               "quantum": (0, 11), "learnable": (0, 11)}
    for a, p, d in zip(ARMS, params, dnll):
        ax2.scatter(p, d, s=110, color=COLOR[a], zorder=3, edgecolor="white", lw=1.2)
        ax2.annotate(SHORT[a], (p, d), textcoords="offset points",
                     xytext=offsets[a], ha="center", fontsize=8.5, weight="bold",
                     color=COLOR[a])
    ax2.set_xlabel("Trainable parameters")
    ax2.set_ylabel("Increase in NLL under transfer\n(CheXpert − MIMIC, lower is better)")
    ax2.set_xlim(35, 145)
    ax2.set_ylim(0, max(dnll) * 1.32)
    ax2.set_title("B  Larger arms transfer worse", loc="left")

    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig5_external_validation.{ext_}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_external_validation")


# --------------------------------------------------------------------------- #
def figure_risk_coverage(plt) -> None:
    from aura.common.mathx import softmax
    from aura.services.safety.calibration import fit_temperature
    from aura.ml.evaluation.fusion_baselines import (
        _converged_logistic, _served_classical, _served_learnable, _served_quantum,
    )
    from aura.ml.evaluation.quantum_study import load_evidence
    from aura.ml.evaluation.risk_coverage import risk_coverage_curve

    rc = json.loads((ARTIFACTS / "risk_coverage_n447.json").read_text(encoding="utf-8"))
    Xtr, ytr, Xcal, ycal, Xte, yte, _ = load_evidence(n=1338, seed=7)

    builders = {
        "classical": lambda: _served_classical(Xcal, Xte)[:2],
        "quantum": lambda: _served_quantum(Xcal, Xte)[:2],
        "learnable": lambda: _served_learnable(Xcal, Xte)[:2],
        "logistic": lambda: _converged_logistic(Xtr, ytr, Xcal, ycal, Xte)[:2],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    for a in ARMS:
        Lcal, Lte = builders[a]()
        T = fit_temperature(Lcal, ycal)
        P = np.array([softmax(r / T) for r in Lte])
        correct = P.argmax(axis=1) == yte
        cov, risk = risk_coverage_curve(correct, P.max(axis=1))
        keep = cov >= 0.05
        ax1.plot(cov[keep], risk[keep], color=COLOR[a], lw=1.7,
                 label=f"{SHORT[a]}  (AURC {rc['backends'][a]['aurc']:.3f})")
        pol = rc["backends"][a]["served_policy"]
        ax1.scatter([pol["coverage"]], [pol["risk_answered"]], marker="o", s=52,
                    color=COLOR[a], edgecolor="white", lw=1.3, zorder=4)

    ax1.set_xlabel("Coverage (fraction of studies answered)")
    ax1.set_ylabel("Selective risk (error rate among answered)")
    ax1.set_title("A  Risk–coverage, with the served policy marked", loc="left")
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.set_xlim(0.05, 1.0)

    # -- Panel B: policy minus matched-coverage control, with intervals -- #
    y = np.arange(len(ARMS))
    deltas = [rc["backends"][a]["matched_coverage_control"]["policy_minus_control_risk"]
              for a in ARMS]
    los = [rc["backends"][a]["matched_coverage_control"]["policy_minus_control_risk_ci95"][0]
           for a in ARMS]
    his = [rc["backends"][a]["matched_coverage_control"]["policy_minus_control_risk_ci95"][1]
           for a in ARMS]
    for i, a in enumerate(ARMS):
        ax2.plot([los[i], his[i]], [i, i], color=COLOR[a], lw=2.4, solid_capstyle="round")
        ax2.scatter([deltas[i]], [i], s=64, color=COLOR[a], zorder=3,
                    edgecolor="white", lw=1.2)
    ax2.axvline(0, color="0.2", lw=1.1, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels([SHORT[a] for a in ARMS])
    ax2.set_xlabel("Policy risk − max-softmax risk at matched coverage")
    ax2.set_title("B  The four-criterion policy does not beat its control", loc="left")
    ax2.text(0.001, -0.62, "policy worse →", fontsize=7.5, color="0.3")
    ax2.text(-0.001, -0.62, "← policy better", fontsize=7.5, color="0.3", ha="right")
    ax2.set_ylim(-0.9, len(ARMS) - 0.4)
    ax2.invert_yaxis()

    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig6_risk_coverage.{ext_}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_risk_coverage")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt = _style()
    figure_external(plt)
    figure_risk_coverage(plt)


if __name__ == "__main__":
    main()
