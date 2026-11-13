import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("E:/AURA/aura-main")
FIGDIR = ROOT / "docs" / "figures"
CONF_FILE = ROOT / "aura" / "artifacts" / "v6_confirmatory.json"
conf = json.loads(CONF_FILE.read_text())

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif"
})

fig = plt.figure(figsize=(7.2, 5.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 0.8], hspace=0.38, wspace=0.28)

# Panel A: C1 and C1b
ax1 = fig.add_subplot(gs[0, :])
labels_a = [
    "C1: VQC vs PoE (matched LR)",
    "C1: VQC vs Gated Head (matched LR)",
    "C1: VQC vs Logistic (matched LR)",
    "C1b: Entangled vs Product circuit (10 seeds)"
]
deltas_a = [0.02049, 0.01964, 0.02095, 0.01037]
cis_a = [
    [0.01858, 0.02240],
    [0.01796, 0.02132],
    [0.01905, 0.02285],
    [0.00934, 0.01138]
]

y_pos_a = np.arange(len(labels_a))[::-1]
ax1.axvspan(-0.010, 0.010, color="#CCCCCC", alpha=0.3, label="Operational equivalence band (+/-0.010 nats)")
ax1.axvline(0, color="black", lw=0.8, ls="--")
ax1.axvline(0.010, color="#B23A48", lw=0.8, ls=":")

for y, d, ci in zip(y_pos_a, deltas_a, cis_a):
    ax1.plot(ci, [y, y], color="#2F5D8A", lw=2)
    ax1.plot(d, y, "s", color="#2F5D8A", ms=6)

ax1.set_yticks(y_pos_a)
ax1.set_yticklabels(labels_a, fontsize=8)
ax1.set_xlim(-0.012, 0.026)
ax1.set_xlabel("Paired Delta-NLL (nats), Quantum - Classical (positive = Quantum worse) ->")
ax1.set_title("A  Analysis C1 & C1b: Confirmatory test under default matched protocol (N = 56,534 patients)", loc="left", fontweight="bold", fontsize=8.5)
ax1.grid(axis="x", alpha=0.25)
ax1.legend(loc="upper left", fontsize=7.5, frameon=False)

# Panel B: C2 Paired differences under 36-config symmetric tuning
ax2 = fig.add_subplot(gs[1, 0])
labels_b = [
    "C2: Tuned VQC vs Tuned PoE",
    "C2: Tuned VQC vs Tuned Gated Head"
]
deltas_b = [-0.00260, -0.00116]
cis95_b = [
    [-0.00359, -0.00160],
    [-0.00192, -0.00041]
]
cis90_b = [
    [-0.00343, -0.00176],
    [-0.00180, -0.00053]
]

y_pos_b = np.array([1, 0])
ax2.axvspan(-0.010, 0.010, color="#CCCCCC", alpha=0.3)
ax2.axvline(0, color="black", lw=0.8, ls="--")
ax2.axvline(-0.010, color="#2F5D8A", lw=0.8, ls=":")

for y, d, ci95, ci90 in zip(y_pos_b, deltas_b, cis95_b, cis90_b):
    ax2.plot(ci95, [y, y], color="#B23A48", lw=1.8, label="95% CI" if y==1 else None)
    ax2.plot(ci90, [y, y], color="#7E2430", lw=3.2, label="90% CI (TOST)" if y==1 else None)
    ax2.plot(d, y, "o", color="#B23A48", ms=6, label="Point estimate" if y==1 else None)

ax2.set_yticks(y_pos_b)
ax2.set_yticklabels(labels_b, fontsize=8)
ax2.set_xlim(-0.012, 0.006)
ax2.set_ylim(-0.6, 1.6)
ax2.set_xlabel("Paired Delta-NLL (nats) <- Favors Quantum | Favors Classical ->")
ax2.set_title("B  Analysis C2: Symmetric 36-config search", loc="left", fontweight="bold", fontsize=8.5)
ax2.grid(axis="x", alpha=0.25)
ax2.legend(loc="upper right", fontsize=7, frameon=False)

# Panel C: TOST sensitivity across margins
ax3 = fig.add_subplot(gs[1, 1])
margins = np.linspace(0.002, 0.015, 50)
n_test = 56534
se_poe = 0.00051
se_gated = 0.000385

p_poe = [max(1 - stats.t.cdf((-0.00260 - (-m)) / se_poe, n_test - 1),
             1 - stats.t.cdf((m - (-0.00260)) / se_poe, n_test - 1)) for m in margins]
p_gated = [max(1 - stats.t.cdf((-0.00116 - (-m)) / se_gated, n_test - 1),
               1 - stats.t.cdf((m - (-0.00116)) / se_gated, n_test - 1)) for m in margins]

ax3.plot(margins, p_poe, "-", color="#B23A48", lw=1.8, label="vs Tuned PoE")
ax3.plot(margins, p_gated, "--", color="#2F5D8A", lw=1.8, label="vs Tuned Gated")
ax3.axhline(0.05, color="black", lw=0.8, ls=":", label="alpha = 0.05 threshold")
ax3.axvline(0.010, color="#555555", lw=0.8, ls="-.", label="Prespecified margin (+/-0.010)")
ax3.set_xlabel("Equivalence Margin delta (nats)")
ax3.set_ylabel("TOST p-value (Equivalence test)")
ax3.set_title("C  Equivalence margin sensitivity", loc="left", fontweight="bold", fontsize=8.5)
ax3.set_yscale("log")
ax3.set_ylim(1e-10, 1.5)
ax3.grid(True, alpha=0.25, which="both")
ax3.legend(loc="lower left", fontsize=7, frameon=False)

fig.savefig(FIGDIR / "fig_v6_confirmatory.png", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_v6_confirmatory.pdf", bbox_inches="tight")
fig.savefig(Path("E:/AURA/fig_v6_confirmatory.png"), bbox_inches="tight")
fig.savefig(Path("E:/AURA/fig_v6_confirmatory.pdf"), bbox_inches="tight")
print("Saved fig_v6_confirmatory.png and .pdf successfully.")
