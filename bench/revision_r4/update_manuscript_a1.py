import sys
from pathlib import Path

p = Path("e:/AURA/aura-main/manuscript.tex")
text = p.read_text(encoding="utf-8")

old_c1_seed = "We emphasize, however, that this $+0.0205$-nat gap represents a single trained checkpoint comparison and is on the order of random initialization seed variance ($\\approx 1.8 \\sigma_{\\text{seed}}$)."
new_c1_seed = "We emphasize that this $+0.0205$-nat single checkpoint gap must be evaluated against random initialization variance. Across 10 random seeds on the 56,534 held-out patients, default VQC achieved a mean test NLL of $1.13056$ nats ($\\sigma_{\\text{seed}} = 0.00448$ nats; range $1.12513$ to $1.13816$) compared to $1.11568 \\pm 0.00010$ nats for matched classical PoE. The resulting mean paired gap over seeds is $\\Delta\\text{NLL} = \\mathbf{+0.01420}$ nats (seed-level 95\\% Student's $t$ CI $[+0.01064, +0.01776]$ nats; $t = 9.01, \\text{df} = 9, p = 8.43 \\times 10^{-6}$), confirming that default VQC is materially and statistically worse than matched classical PoE across random initializations ($p_{\\text{TOST}} = 0.987$). The single-checkpoint $+0.0205$-nat gap corresponds to $4.57 \\sigma_{\\text{seed}}$."

if old_c1_seed in text:
    text = text.replace(old_c1_seed, new_c1_seed)
    print("Replaced old_c1_seed successfully.")
else:
    print("WARNING: old_c1_seed not found.")

old_c2_block = """Crucially, we must distinguish between \\textit{checkpoint-level equivalence} and \\textit{architectural-level equivalence across random initializations}. Across 10 random initialization seeds evaluated on the 56,534 test patients, the classical Product-of-Experts exhibited near-zero initialization variance ($\\sigma_{\\text{seed}} = \\mathbf{0.00001\\text{ nats}}$), and the classical Gated Head exhibited $\\sigma_{\\text{seed}} = \\mathbf{0.00021\\text{ nats}}$. In sharp contrast, the variational quantum circuit exhibited an initialization seed standard deviation of $\\sigma_{\\text{seed}} = \\mathbf{0.01163\\text{ nats}}$ (ranging from 1.1275 to 1.1688 nats in C1b)---50 to 1,000 times larger than classical comparators. Because the VQC seed standard deviation ($\\sigma_{\\text{seed}} = 0.0116$ nats) exceeds the pre-specified equivalence margin ($\\Delta_{\\text{equiv}} = \\pm 0.010$ nats), Two One-Sided Tests (TOST) on a single selected checkpoint pair establishes \\textbf{operational equivalence strictly for the evaluated checkpoints}, but cannot establish architectural equivalence across arbitrary random initializations. Across initializations, VQC performance spans outside the $\\pm 0.010$-nat boundary. Furthermore, finite validation split noise on $n_{\\text{cal}} = 1,600$ patients ($\\text{SE} \\approx \\sigma / \\sqrt{1600} \\approx 0.015$ nats) introduces selection jitter, as directly evidenced by tuned PoE (1.1170) performing 0.0013 nats worse on test than default PoE (1.1157). Consequently, an observed difference of 0.0026 nats cannot be reliably distinguished from training-seed noise and validation-selection jitter, reinforcing the conclusion of operational equivalence."""

new_c2_block = """Crucially, we must distinguish between \\textit{checkpoint-level equivalence} and \\textit{architectural-level equivalence across random initializations}. To rigorously test architectural robustness, we trained 10 independent random initialization seeds of the selected C2 winning configurations and scored each on the 56,534 held-out patients. The classical Product-of-Experts exhibited near-zero initialization variance (mean NLL $1.11698$, $\\sigma_{\\text{seed}} = \\mathbf{0.00010\\text{ nats}}$; range $1.11681$ to $1.11710$), and the classical Gated Head exhibited negligible variance (mean NLL $1.11554$, $\\sigma_{\\text{seed}} = \\mathbf{0.00007\\text{ nats}}$; range $1.11541$ to $1.11563$). In sharp contrast, the tuned variational quantum circuit exhibited an initialization standard deviation of $\\sigma_{\\text{seed}} = \\mathbf{0.00346\\text{ nats}}$ (mean NLL $1.11730$; range $1.11287$ to $1.12465$ nats)---a spread of $0.01178$ nats that exceeds the $\\pm 0.010$-nat equivalence margin. Across the 10 paired seeds vs tuned PoE, the mean paired difference is $\\Delta\\text{NLL} = \\mathbf{-0.00067\\text{ nats}}$ ($\\text{SD} = 0.00308, \\text{SE} = 0.00097\\text{ nats}$), with a seed-level 95\\% Student's $t$ confidence interval of $[-0.00288, +0.00153]$ nats ($t = -0.69, \\text{df} = 9, p = 0.506$). Seed-level TOST against $\\pm 0.010$ nats yields $p_{\\text{TOST}} = 2.57 \\times 10^{-6}$, and a hierarchical bootstrap simultaneously resampling training seeds and test patients ($10,000$ iterations) yields a 95\\% confidence interval of $[-0.00271, +0.00146]$ nats. Versus the tuned Gated Head, the mean paired seed difference is $\\Delta\\text{NLL} = -0.00064$ nats (95\\% Student's $t$ CI $[-0.00339, +0.00211]$ nats; $t = -0.53, p = 0.610$, seed-level $p_{\\text{TOST}} = 1.50 \\times 10^{-5}$).

These seed-level experiments establish three vital conclusions: (1) across the population of training initializations, the mean difference between tuned VQC and tuned classical baselines is under $0.0007$ nats and formally equivalent within $\\pm 0.010$ nats; (2) the single-checkpoint $-0.0026$-nat "win" reported in Table 3 was an initialization-lucky draw (seed 7 yielded $1.11287$, the best of 10 runs), whereas 4 of the 10 seeds (seeds 11, 13, 19, 23) lose to tuned PoE; and (3) finite validation split noise on $n_{\\text{cal}} = 1,600$ patients introduces selection jitter, as directly evidenced by tuned PoE ($1.1170$) scoring $0.0013$ nats worse on test than default PoE ($1.1157$). Consequently, an observed difference of $0.0026$ nats cannot be reliably distinguished from initialization seed noise and validation selection jitter."""

# Normalize newlines for replacement
text_norm = text.replace("\r\n", "\n")
old_c2_norm = old_c2_block.replace("\r\n", "\n")
new_c2_norm = new_c2_block.replace("\r\n", "\n")

if old_c2_norm in text_norm:
    text_norm = text_norm.replace(old_c2_norm, new_c2_norm)
    print("Replaced old_c2_block successfully.")
else:
    print("WARNING: old_c2_block not found.")

p.write_text(text_norm, encoding="utf-8")
print("Wrote updated manuscript.tex")
