from pathlib import Path

tex_path = Path('e:/AURA/aura-main/manuscript.tex')
tex = tex_path.read_text(encoding='utf-8')

# 1. Table 2 Parkinson row
t2_old = r'& $\bullet$ Baseline (Untuned vs Tuned) & \textbf{Symmetric Protocol:} In-Fold PCA $\times$ Subject Split $\times$ Tuned & \textbf{$-$0.3\%} ($-2.1\%$ to +1.5\%) & $p = 0.72$ & Advantage vanishes. \\'
t2_new = r'& $\bullet$ Baseline (Untuned vs Tuned) & \textbf{Tuned on Factorial Split:} In-Fold PCA $\times$ Record Split $\times$ Tuned & \textbf{$-$0.3\%} ($-2.1\%$ to +1.5\%) & $p = 0.72$ & Advantage vanishes. \\'
assert t2_old in tex, 't2_old not found'
tex = tex.replace(t2_old, t2_new)

# 1b. Table 2 caption
t2_cap_old = r"Note that the naive recording-level split on Parkinson's voice phonations yields an artificially narrow interval ($\pm 1.8\%$) because multiple recordings from the same 31 individuals are treated as independent, reproducing the literature's data-leakage flaw."
t2_cap_new = r"Note that the fixed factorial grid evaluated recording-level partitions ($\pm 1.8\%$ CI with 31 subjects), treating multiple voice recordings from the same individual as independent---reproducing the literature's data-leakage flaw. True subject-grouped splitting under repeated nested cross-validation with Nadeau--Bengio correction yields wide uncertainty ($\pm 25.3\%$; Table 1), confirming the finding is statistically inconclusive."
assert t2_cap_old in tex, 't2_cap_old not found'
tex = tex.replace(t2_cap_old, t2_cap_new)

# 2. Section 4 Heading
sec4_old = r'\subsection*{Symmetric hyperparameter tuning confirms practical equivalence}'
sec4_new = r'\subsection*{Symmetric hyperparameter tuning confirms population equivalence but fails class-balanced evaluation}'
assert sec4_old in tex, 'sec4_old not found'
tex = tex.replace(sec4_old, sec4_new)

# 3. Analysis C1 text (PoE 10-seed mean)
c1_init_old = 'Across 10 random seeds on the 56,534 held-out patients, default VQC'
c1_init_new = 'Across 10 random seeds on the 56,534 held-out patients (a post-hoc revision analysis), default VQC'
assert c1_init_old in tex, 'c1_init_old not found'
tex = tex.replace(c1_init_old, c1_init_new)

c1_poe_old = r'compared to $1.11568 \pm 0.00010$ nats for matched classical PoE. The resulting mean paired gap over seeds is $\Delta\text{NLL} = \mathbf{+0.01420}$ nats'
c1_poe_new = r'compared to $1.11636 \pm 0.00121$ nats for matched classical PoE (range $1.11437$ to $1.11849$). The difference of 10-seed means identically equals the mean paired gap over seeds: $1.13056 - 1.11636 = \Delta\text{NLL} = \mathbf{+0.01420}$ nats'
assert c1_poe_old in tex, 'c1_poe_old not found'
tex = tex.replace(c1_poe_old, c1_poe_new)

# 4. Analysis C1b Student-t CI
c1b_ci_old = r'$[0.00491, 0.01583]$ nats'
c1b_ci_new = r'$[0.00526, 0.01544]$ nats'
assert c1b_ci_old in tex, 'c1b_ci_old not found'
tex = tex.replace(c1b_ci_old, c1b_ci_new)

fig3_c1b_old = '0.0049 to 0.0158'
fig3_c1b_new = '0.0053 to 0.0154'
assert fig3_c1b_old in tex, 'fig3_c1b_old not found'
tex = tex.replace(fig3_c1b_old, fig3_c1b_new)

# 5. Analysis C2 seed variance text
c2_intro_old = r'To rigorously test architectural robustness, we trained 10 independent random initialization seeds'
c2_intro_new = r'To rigorously test architectural robustness (a post-hoc revision analysis conducted in response to peer review), we trained 10 independent random initialization seeds'
assert c2_intro_old in tex, 'c2_intro_old not found'
tex = tex.replace(c2_intro_old, c2_intro_new)

c2_poe_gated_old = (
    r"The classical Product-of-Experts exhibited near-zero initialization variance (mean NLL $1.11698$, "
    r"$\sigma_{\text{seed}} = \mathbf{0.00010\text{ nats}}$; range $1.11681$ to $1.11710$), and the classical "
    r"Gated Head exhibited negligible variance (mean NLL $1.11554$, $\sigma_{\text{seed}} = \mathbf{0.00007\text{ nats}}$; "
    r"range $1.11541$ to $1.11563$). In sharp contrast, the tuned variational quantum circuit exhibited an "
    r"initialization standard deviation of $\sigma_{\text{seed}} = \mathbf{0.00346\text{ nats}}$ "
    r"(mean NLL $1.11730$; range $1.11287$ to $1.12465$ nats)---a spread of $0.01178$ nats that exceeds the $\pm 0.010$-nat equivalence margin."
)
c2_poe_gated_new = (
    r"Across the 10 seeds, the classical Product-of-Experts achieved a mean test NLL of $1.11797$ "
    r"($\sigma_{\text{seed}} = \mathbf{0.00130\text{ nats}}$; range $1.11656$ to $1.12019$), and the classical "
    r"Gated Head achieved a mean test NLL of $1.11794$ ($\sigma_{\text{seed}} = \mathbf{0.00374\text{ nats}}$; "
    r"range $1.11391$ to $1.12464$). The tuned variational quantum circuit achieved a mean test NLL of "
    r"$1.11730$ ($\sigma_{\text{seed}} = \mathbf{0.00346\text{ nats}}$; range $1.11287$ to $1.12465$ nats). "
    r"Notice that the difference of 10-seed means identically matches the mean paired difference: "
    r"$1.11730 - 1.11797 = \mathbf{-0.00067\text{ nats}}$ vs tuned PoE ($\text{SD} = 0.00308, \text{SE} = 0.00097\text{ nats}$), "
    r"and every individual paired seed delta falls strictly inside the equivalence margin (range $[-0.00411, +0.00452]$ nats)."
)
assert c2_poe_gated_old in tex, 'c2_poe_gated_old not found'
tex = tex.replace(c2_poe_gated_old, c2_poe_gated_new)

c2_gated_contrast_old = r'Versus the tuned Gated Head, the mean paired seed difference is $\Delta\text{NLL} = -0.00064$ nats'
c2_gated_contrast_new = r'Versus the tuned Gated Head, the mean paired seed difference is identically $1.11730 - 1.11794 = \mathbf{-0.00064\text{ nats}}$'
assert c2_gated_contrast_old in tex, 'c2_gated_contrast_old not found'
tex = tex.replace(c2_gated_contrast_old, c2_gated_contrast_new)

# 6. Subgroup breakdown explanation
subgroup_old = r'\noindent\textbf{Class-Conditional Likelihood Disparities and Macro-Averaging Breakdown:} Evaluating condition-specific subgroups across clinical triage categories revealed that the small prevalence-weighted difference ($\Delta\text{NLL} = -0.00235$ nats, 95\% CI $[-0.00326, -0.00146]$) is the net consequence of large, opposing class-conditional likelihood shifts:'
subgroup_new = r'\noindent\textbf{Class-Conditional Likelihood Disparities and Macro-Averaging Breakdown:} We distinguish the global multi-class cross-entropy difference evaluated on the reference checkpoint across all 56,534 patients ($\Delta\text{NLL} = -0.00260$ nats, 95\% analytic CI $[-0.00359, -0.00160]$) from the condition-stratified subgroup breakdown. In this class-conditional decomposition (a post-hoc revision analysis; \textbf{Table 4}), the prevalence-weighted sum across categories $\sum_i (n_i / N) \delta_i = \mathbf{-0.00235\text{ nats}}$ (95\% patient-level bootstrap CI $[-0.00326, -0.00146]$), which arises from large, opposing class-conditional shifts:'
assert subgroup_old in tex, 'subgroup_old not found'
tex = tex.replace(subgroup_old, subgroup_new)

# 7. Vector scaling explanation
vec_old = r'To determine whether calibration asymmetry reflects an intrinsic quantum geometric advantage or per-class probability threshold shifts, we fitted an independent vector scaling ablation (independent diagonal weights and biases per class fitted on calibration data). Vector scaling closed over 95\% of the calibration gap (test ECE of 0.0287 for VQC vs 0.0281 for PoE and 0.0290 for Gated Head; gap $< 0.0006$) while preserving the exact likelihood ordering (VQC NLL 1.11537 vs PoE 1.11807; $\Delta\text{NLL} = -0.00270$). Under our decision rule, this confirms that the calibration asymmetry is primarily a per-class threshold effect resolvable by vector calibration, not an intrinsic quantum representation property.'

vec_new = r'To explore whether calibration asymmetry reflects an intrinsic quantum representation property or class-specific affine threshold offsets, we conducted an exploratory post-hoc vector scaling ablation (fitting 6 diagonal weights and 6 biases per model on the 1,600 calibration patients). Under vector scaling, overall test loss differences remained practically equivalent ($\Delta\text{NLL} = -0.00270$ nats; VQC 1.11537 vs PoE 1.11807). However, estimating 12 affine parameters per model on a finite calibration set ($n=1,600$) introduces parameter estimation variance and mild overfitting: test ECE increased from $0.008\text{--}0.013$ under scalar temperature scaling to $0.028\text{--}0.029$ across all three architectures (0.0287 for VQC, 0.0281 for PoE, 0.0290 for Gated Head), and PoE test NLL slightly degraded. Because scalar temperature scaling achieves superior calibration fidelity ($ECE \approx 0.008\text{--}0.013$) without calibration overfitting, it remains our primary pre-registered calibration method. Furthermore, vector scaling did not eliminate the underlying class-conditional likelihood trade-offs, demonstrating that the macro-averaging breakdown reflects genuine differences in conditional error rates across rare versus common diagnoses rather than a simple global scaling artifact.'
assert vec_old in tex, 'vec_old not found'
tex = tex.replace(vec_old, vec_new)

# 8. Discussion: sigma_seed and angle wrapping
disc1_old = r'though this single-run gap is on the order of training seed variance ($\sigma_{\text{seed}} = 0.0116$ nats) and calibration error was indistinguishable between calibrated arms. Under symmetric hyperparameter search (C2), it achieved a small likelihood advantage over linear PoE ($-0.0026$ nats) and nonlinear gated heads ($-0.0012$ nats) that fell entirely inside the pre-specified operational equivalence margin ($\pm 0.010$ nats; TOST $p < 10^{-6}$) and within training-seed variance. Crucially, because the quantum initialization seed variance ($\sigma_{\text{seed}} = 0.0116$ nats) is 50- to 1,000-fold larger than classical seed variance ($\le 2 \times 10^{-4}$ nats) and exceeds the equivalence margin ($\Delta_{\text{equiv}} = 0.010$ nats), checkpoint-level equivalence does not establish architectural equivalence across initializations.'

disc1_new = r'which was confirmed across 10 random initialization seeds ($\Delta\text{NLL} = +0.0142$ nats, $p = 8.43 \times 10^{-6}$; $\sigma_{\text{seed}} = 0.0045$ nats; single-checkpoint gap $+0.0205$ nats is $4.57\sigma_{\text{seed}}$), while calibration error was indistinguishable between calibrated arms. Under symmetric hyperparameter search (C2), it achieved a small likelihood advantage over linear PoE ($-0.0026$ nats) and nonlinear gated heads ($-0.0012$ nats) that fell entirely inside the pre-specified operational equivalence margin ($\pm 0.010$ nats; TOST $p < 10^{-6}$). In Analysis C2, 10-seed evaluations confirmed architectural equivalence: the mean paired difference across 10 random seeds was $\Delta\text{NLL} = -0.00067$ nats vs tuned PoE and $-0.00064$ nats vs tuned Gated Head (seed-level $p_{\text{TOST}} \le 1.50 \times 10^{-5}$), with all individual seed paired differences falling strictly within the $\pm 0.010$-nat margin. However, operational equivalence failed under class-balanced macro-averaging ($\Delta\text{NLL}_{\text{macro}} = +0.0389$ nats, $p_{\text{TOST}} > 0.99$), revealing that population-level equivalence is driven by cancellation between majority classes while the quantum circuit degrades on minority acute pathologies.'
assert disc1_old in tex, 'disc1_old not found'
tex = tex.replace(disc1_old, disc1_new)

disc2_old = r'where the quantum circuit degraded severely due to periodic angle-wrapping vulnerabilities under covariate scale drift.'
disc2_new = r'where all models degraded substantially ($\Delta\text{NLL} = +0.131$ to $+0.143$ nats; quantum frozen gap trailing by up to $+0.1967$ nats) due to institutional covariate shift, with empirical scale sweeps refuting periodic angle wrapping as the mechanism.'
assert disc2_old in tex, 'disc2_old not found'
tex = tex.replace(disc2_old, disc2_new)

# 9. Table 3 Caption & 10-seed rows
t3_cap_old = r'The optimal quantum model achieves lower test loss than classical baselines ($\Delta\text{NLL} = -0.0026$ nats vs PoE, $p = 3.07 \times 10^{-7}$), with the 95\% CI falling entirely within the pre-specified operational equivalence margin ($\pm 0.010$ nats; $p_{\text{TOST}} < 10^{-6}$) and within training-seed noise.'
t3_cap_new = r'In Panel B, optimal single checkpoints are reported; across 10 random initialization seeds of these selected winning configurations, tuned VQC achieved mean test NLL $1.11730 \pm 0.00346$ vs $1.11797 \pm 0.00130$ for tuned PoE and $1.11794 \pm 0.00374$ for tuned Gated Head (mean paired difference $-0.00067$ nats vs PoE, $p_{\text{TOST}} = 2.57 \times 10^{-6}$; $-0.00064$ nats vs Gated Head, $p_{\text{TOST}} = 1.50 \times 10^{-5}$), confirming architectural equivalence on the overall population.'
assert t3_cap_old in tex, 't3_cap_old not found'
tex = tex.replace(t3_cap_old, t3_cap_new)

# Add 10-seed row to Panel B
panel_b_end_old = r'\textbf{Optimal Variational Quantum} & 118 & 0.5417 (0.5376, 0.5458) & \textbf{1.1144} & 0.0111 (0.0088, 0.0151) & \textbf{0.7550} & $\mathbf{-0.0026}$ ($-0.0036$, $-0.0016$) [\textbf{\color{royalazure}Practically Equivalent}] \\'
panel_b_end_new = (
    r'\textbf{Optimal Variational Quantum} & 118 & 0.5417 (0.5376, 0.5458) & \textbf{1.1144} & 0.0111 (0.0088, 0.0151) & \textbf{0.7550} & $\mathbf{-0.0026}$ ($-0.0036$, $-0.0016$) [\textbf{\color{royalazure}Practically Equivalent}] \\' + '\n' +
    r'\addlinespace' + '\n' +
    r'\multicolumn{7}{l}{\textit{10-Seed Initialization Means (Post-Hoc Revision Analysis, 56,534 Test Patients):}} \\' + '\n' +
    r'\quad\textit{Tuned Product-of-Experts} & 54 & $0.5425 \pm 0.0015$ & $1.1180 \pm 0.0013$ & $0.0109 \pm 0.0011$ & $0.7533 \pm 0.0009$ & Reference (10-Seed PoE Mean: 1.11797) \\' + '\n' +
    r'\quad\textit{Tuned Gated Head} & 126 & $0.5411 \pm 0.0015$ & $1.1179 \pm 0.0037$ & $0.0111 \pm 0.0024$ & $0.7547 \pm 0.0010$ & $-0.00003$ vs Tuned PoE Mean \\' + '\n' +
    r'\quad\textit{Optimal Variational Quantum} & 118 & $0.5409 \pm 0.0014$ & $1.1173 \pm 0.0035$ & $0.0252 \pm 0.0038$ & $0.7546 \pm 0.0011$ & $\mathbf{-0.00067}$ ($-0.00288, +0.00153$) [$p_{\text{TOST}} < 10^{-5}$] \\'
)
assert panel_b_end_old in tex, 'panel_b_end_old not found'
tex = tex.replace(panel_b_end_old, panel_b_end_new)

# 10. M2 Polynomial LR discussion
m2_old = r'This establishes a vital insight: order-1 classical trigonometric regression outperforms both the tuned quantum circuit (1.1144) and classical PoE (1.1170).'
m2_new = r'Furthermore, a classical polynomial logistic regression ($x, x^2$, Table 3 Panel C) achieved NLL \textbf{1.1092}, outperforming both the order-1 trigonometric surrogate (1.1123) and the tuned quantum circuit (1.1144) by $\Delta\text{NLL} = 0.0051$ ($p < 10^{-4}$). This establishes a vital insight: elementary classical polynomial and trigonometric regressions outperform both the tuned quantum circuit (1.1144) and classical PoE (1.1170).'
assert m2_old in tex, 'm2_old not found'
tex = tex.replace(m2_old, m2_new)

tex_path.write_text(tex, encoding='utf-8')
print('All edits applied successfully to manuscript.tex.')
