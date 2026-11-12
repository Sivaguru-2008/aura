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
    ax1.legend(loc="upper right", fontsize=7.5)
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


def figure_comparator_and_control(plt) -> None:
    """Figure 1 — the paper's thesis and its positive control, side by side.

    Panel A is the claim: whether a quantum advantage is reported depends on which
    classical model is placed opposite the circuit *and* on how carefully that model
    was fitted. It shows the same four arms on the same split under the published
    heterogeneous optimisers and under one shared optimiser protocol.

    Panel B is the control that stops Panel A from being a tautology. The clinical
    target is linear in the evidence vector, so a null there is predicted rather
    than measured. Panel B varies only the target's structure and shows the same
    comparison detecting a nonlinear advantage where one exists.
    """
    pub_path = ARTIFACTS / "fusion_baselines_n447.json"
    mat_path = ARTIFACTS / "fusion_matched_optimiser.json"
    ctl_path = ARTIFACTS / "nonlinear_control.json"
    for p in (pub_path, mat_path, ctl_path):
        if not p.exists():
            print(f"skip fig1: {p.name} not generated yet")
            return

    pub = json.loads(pub_path.read_text(encoding="utf-8"))
    mat = json.loads(mat_path.read_text(encoding="utf-8"))
    ctl = json.loads(ctl_path.read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # ---- Panel A: NLL by arm, published optimisers vs matched protocol ---- #
    width = 0.36
    xs = np.arange(len(ARMS))
    pub_nll = [pub["backends"][a]["nll"] for a in ARMS]
    mat_nll = [mat["backends"][a]["nll"] for a in ARMS]

    ax1.bar(xs - width / 2, pub_nll, width, label="As published\n(each arm's own optimiser)",
            color=[COLOR[a] for a in ARMS], alpha=0.45, edgecolor="white")
    ax1.bar(xs + width / 2, mat_nll, width, label="Matched optimiser protocol",
            color=[COLOR[a] for a in ARMS], edgecolor="white")

    for x, (p_, m_) in enumerate(zip(pub_nll, mat_nll)):
        ax1.text(x - width / 2, p_ + 0.004, f"{p_:.3f}", ha="center", fontsize=7.5,
                 color="0.35")
        ax1.text(x + width / 2, m_ + 0.004, f"{m_:.3f}", ha="center", fontsize=7.5,
                 weight="bold")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([SHORT[a] for a in ARMS])
    ax1.set_ylabel("Negative log-likelihood (lower is better)")
    ax1.set_title("A  The verdict depends on the comparator\n"
                  "     and on how carefully it was fitted", loc="left")
    lo = min(min(pub_nll), min(mat_nll))
    hi = max(max(pub_nll), max(mat_nll))
    ax1.set_ylim(lo - 0.02, hi + 0.055)

    # The two sentences the panel exists to support, read from the artifacts.
    pub_q = pub["pairwise"]["classical_vs_quantum"]
    mat_q = mat["pairwise"]["classical_vs_quantum"]
    ax1.annotate(
        f"published: VQC beats PoE by {abs(pub_q['delta_nll']):.3f} "
        f"(p = {pub_q['nll_wilcoxon_p']:.3g})\n"
        f"matched:   PoE beats VQC by {abs(mat_q['delta_nll']):.3f} "
        f"(Holm p = {mat_q['nll_wilcoxon_p_holm']:.2g})",
        xy=(0.03, 0.97), xycoords="axes fraction", fontsize=7.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="0.96", ec="0.8", lw=0.6))
    ax1.legend(loc="upper right", fontsize=7.5)
    # The axis is truncated to resolve differences of ~0.01 NLL; say so, because a
    # truncated axis that is not labelled overstates every difference on it.
    ax1.text(0.99, -0.13, "axis truncated", transform=ax1.transAxes,
             ha="right", va="top", fontsize=6.5, color="0.45")

    # ---- Panel B: likelihood advantage over the best linear model ---------- #
    #
    # Plotted on negative log-likelihood, not accuracy, because that is the metric
    # the section's claim is made on and the metric the whole paper treats as
    # primary. Accuracy at n = 447 has a minimum detectable difference of 9.3
    # points and resolves none of these comparisons; plotting it would show four
    # near-equal bars and imply the control had failed when it had not.
    targets = ["T1_clinical_linear", "T2a_conjunctive", "T2b_pairwise"]
    tlabel = {"T1_clinical_linear": "T1  linear target\n(clinical rule)",
              "T2a_conjunctive": "T2a  conjunctive",
              "T2b_pairwise": "T2b  pairwise"}
    plotted = [a for a in ARMS if a != "logistic"]      # logistic is the baseline
    bar_w = 0.24
    base = np.arange(len(targets))

    for i, arm in enumerate(plotted):
        vals = []
        for t in targets:
            b = ctl["targets"][t]["backends"]
            vals.append(b[arm]["nll"] - b["logistic"]["nll"])
        ax2.bar(base + (i - (len(plotted) - 1) / 2) * bar_w, vals, bar_w,
                color=COLOR[arm], edgecolor="white", label=SHORT[arm])
        for x, v in zip(base + (i - (len(plotted) - 1) / 2) * bar_w, vals):
            ax2.text(x, v + (0.006 if v >= 0 else -0.006), f"{v:+.3f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=6.8)

    ax2.axhline(0, color="0.3", lw=1.0)
    # Headroom for the value labels at both ends; without it the negative bars'
    # labels are clipped by the axis, and those are the ones carrying the finding.
    _all = [ctl["targets"][t]["backends"][a]["nll"]
            - ctl["targets"][t]["backends"]["logistic"]["nll"]
            for t in targets for a in plotted]
    ax2.set_ylim(min(_all) - 0.035, max(_all) + 0.035)
    ax2.set_xticks(base)
    ax2.set_xticklabels(
        [tlabel[t] + "\nceiling gap "
         + f"{ctl['targets'][t]['ceilings']['gap']:+.3f}"
         for t in targets], fontsize=8)
    ax2.set_ylabel("Δ NLL vs. converged logistic regression")
    ax2.set_title("B  Same inputs, same arms, same protocol —\n"
                  "     only the target's structure changes", loc="left")

    ax2.annotate("below zero = beats the best linear model",
                 xy=(0.02, 0.04), xycoords="axes fraction", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="0.96", ec="0.8", lw=0.6))
    ax2.legend(loc="upper left", fontsize=7.5, ncol=1)

    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig1_comparator_and_control.{ext_}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_comparator_and_control")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt = _style()
    figure_comparator_and_control(plt)
    figure_external(plt)
    figure_risk_coverage(plt)


if __name__ == "__main__":
    main()
