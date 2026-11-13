from pathlib import Path

supp_path = Path('e:/AURA/aura-main/supplementary_information.tex')
supp = supp_path.read_text(encoding='utf-8')

old_supp_t3 = r'''\begin{tabular}{lcccc}
\toprule
\textbf{Analysis Arm / Target} & \textbf{Point Estimate} & \textbf{95\% Confidence Interval} & \textbf{Test Statistic / $p$-value} & \textbf{Pre-Registered Status} \\
\midrule
\textbf{C1: Matched Default ($\Delta$NLL vs PoE)} & $+0.0205$ nats & $[+0.01858, +0.02240]$ & $t = 21.034$, $p = 7.65 \times 10^{-98}$ & Quantum Materially Worse \\
\textbf{C1: Uncertainty Sensitivity ($y_{u1}$)} & $+0.0074$ nats & $[+0.00617, +0.00868]$ & $t = 9.871$, $p = 6.12 \times 10^{-23}$ & Quantum Worse \\
\textbf{C1b: Entanglement Penalty ($\Delta$NLL)} & $+0.0104$ nats & $[+0.00934, +0.01138]$ & $p_{\text{boot}} = 0.00128$ & Penalty Replicates \\
\textbf{C2: Tuned Quantum vs PoE ($\Delta$NLL)} & $-0.0026$ nats & $[-0.00359, -0.00160]$ & $p = 3.07 \times 10^{-7}$, $p_{\text{TOST}} < 10^{-6}$ & Practically Equivalent \\
\textbf{C2: Tuned Quantum vs Gated ($\Delta$NLL)} & $-0.0012$ nats & $[-0.00192, -0.00041]$ & $p = 0.00247$, $p_{\text{TOST}} < 10^{-6}$ & Practically Equivalent \\
\textbf{M2: Order-1 Classical Fourier ($\Delta$NLL)} & $-0.0238$ nats & $[-0.0257, -0.0219]$ & $p < 10^{-4}$ & Classical Fourier Wins \\
\textbf{M2: Order-2 Classical Fourier ($\Delta$NLL)} & $-0.0267$ nats & $[-0.0286, -0.0248]$ & $p < 10^{-4}$ & Classical Fourier Wins \\
\bottomrule
\end{tabular}'''

new_supp_t3 = r'''\begin{tabular}{lcccc}
\toprule
\textbf{Analysis Arm / Target} & \textbf{Point Estimate} & \textbf{95\% Confidence Interval} & \textbf{Test Statistic / $p$-value} & \textbf{Status / Verdict} \\
\midrule
\textbf{C1: Matched Default ($\Delta$NLL vs PoE)} & $+0.0205$ nats & $[+0.01858, +0.02240]$ & $t = 21.034$, $p = 7.65 \times 10^{-98}$ & Quantum Materially Worse \\
\textbf{C1: 10-Seed Mean Gap (Default VQC vs PoE)} & $+0.0142$ nats & $[+0.01064, +0.01776]$ & $t = 9.01$, $p = 8.43 \times 10^{-6}$ & Quantum Worse Across Seeds \\
\textbf{C1: Uncertainty Sensitivity ($y_{u1}$)} & $+0.0074$ nats & $[+0.00617, +0.00868]$ & $t = 9.871$, $p = 6.12 \times 10^{-23}$ & Quantum Worse \\
\textbf{C1b: Entanglement Penalty (10-Seed Student-$t$)} & $+0.0104$ nats & $[+0.00526, +0.01544]$ & $t = 4.60$, $p = 0.00128$ & Penalty Replicates Across Seeds \\
\textbf{C2: Tuned Quantum vs PoE (Checkpoint)} & $-0.0026$ nats & $[-0.00359, -0.00160]$ & $p = 3.07 \times 10^{-7}$, $p_{\text{TOST}} < 10^{-6}$ & Practically Equivalent \\
\textbf{C2: 10-Seed Mean Gap (Tuned VQC vs PoE)} & $-0.00067$ nats & $[-0.00288, +0.00153]$ & $t = -0.69$, $p_{\text{TOST}} = 2.57 \times 10^{-6}$ & Architectural Equivalence Confirmed \\
\textbf{C2: 10-Seed Mean Gap (Tuned VQC vs Gated)} & $-0.00064$ nats & $[-0.00339, +0.00211]$ & $t = -0.53$, $p_{\text{TOST}} = 1.50 \times 10^{-5}$ & Architectural Equivalence Confirmed \\
\textbf{A3: Prevalence-Weighted Subgroup Mean} & $-0.00235$ nats & $[-0.00326, -0.00146]$ & Bootstrap 95\% CI & Practically Equivalent \\
\textbf{A3: Class-Balanced Macro-Mean ($\Delta$NLL)} & $+0.03892$ nats & $[+0.03688, +0.04098]$ & $p_{\text{TOST}} > 0.99$ & Equivalence Fails Macro-Balanced \\
\textbf{M2: Order-1 Classical Fourier vs Tuned PoE} & $-0.00465$ nats & $[-0.00573, -0.00357]$ & $p < 10^{-4}$ & Classical Fourier Wins (NLL 1.1123) \\
\textbf{M2: Order-2 Classical Fourier vs Tuned PoE} & $-0.00747$ nats & $[-0.00870, -0.00625]$ & $p < 10^{-4}$ & Classical Fourier Wins (NLL 1.1095) \\
\textbf{M2: Classical Polynomial LR ($x, x^2$)} & $-0.00774$ nats & $[-0.00887, -0.00661]$ & $p < 10^{-4}$ & Classical Polynomial Wins (NLL 1.1092) \\
\bottomrule
\end{tabular}'''

assert old_supp_t3 in supp, 'old_supp_t3 not found'
supp = supp.replace(old_supp_t3, new_supp_t3)
supp_path.write_text(supp, encoding='utf-8')
print('Supplementary Table 3 updated successfully.')
