from pathlib import Path

tex_path = Path('e:/AURA/aura-main/manuscript.tex')
tex = tex_path.read_text(encoding='utf-8')

table4_code = r'''
% Table 4: Full width class-conditional decomposition
\begin{table*}[t]
\centering
\small
\caption*{\textbf{\textsf{Table 4 \textbar\ Class-conditional log-likelihood decomposition across 56,534 held-out patients.}} Post-hoc revision analysis evaluating condition-specific subgroups across clinical triage categories. The small population-level difference ($\Delta\text{NLL} = -0.00235$ nats prevalence-weighted) is the net consequence of large opposing shifts: the quantum circuit achieves lower loss on high-prevalence categories (Normal, Pneumonia), but degrades substantially on minority acute conditions (Malignancy $+0.1771$ nats, Pneumothorax $+0.0549$ nats, Heart Failure $+0.0753$ nats). Under class-balanced macro-averaging, operational equivalence fails ($\Delta\text{NLL}_{\text{macro}} = +0.0389$ nats, $95\%$ bootstrap CI $0.0369$ to $0.0410$, $p_{\text{TOST}} > 0.99$).}
\vspace{-4pt}
\begin{tabular}{l c c c c c c}
\toprule
\textbf{Clinical Triage Diagnosis} & \textbf{Patients ($n$)} & \textbf{Prevalence} & \textbf{VQC NLL} & \textbf{PoE NLL} & \textbf{Gated NLL} & \textbf{$\Delta\text{NLL}$ vs PoE (95\% Boot CI)} \\
\midrule
\textbf{Normal} & 17,692 & 31.29\% & 0.7379 & 0.7577 & 0.7329 & $\mathbf{-0.0198}$ ($-0.0217$, $-0.0180$) \\
\textbf{Pneumonia} & 17,052 & 30.16\% & 1.0263 & 1.1192 & 1.0364 & $\mathbf{-0.0929}$ ($-0.0943$, $-0.0915$) \\
\textbf{Heart Failure} & 16,490 & 29.17\% & 1.0926 & 1.0172 & 1.1060 & $\mathbf{+0.0753}$ ($+0.0742$, $+0.0765$) \\
\textbf{Pneumothorax} & 3,108 & 5.50\% & 2.5705 & 2.5156 & 2.5817 & $\mathbf{+0.0549}$ ($+0.0510$, $+0.0586$) \\
\textbf{Malignancy} & 2,192 & 3.88\% & 2.8990 & 2.7219 & 2.7686 & $\mathbf{+0.1771}$ ($+0.1681$, $+0.1861$) \\
\midrule
\multicolumn{7}{l}{\textbf{Summary Metrics Across All Categories:}} \\
\multicolumn{5}{l}{\quad\textbf{Prevalence-Weighted Mean ($\sum_i \frac{n_i}{N} \delta_i$):}} & \multicolumn{2}{l}{$\mathbf{-0.00235}$ nats (95\% CI $-0.00326$ to $-0.00146$) [\textit{Practically Equivalent}]} \\
\multicolumn{5}{l}{\quad\textbf{Class-Balanced Macro-Mean ($\frac{1}{K}\sum_i \delta_i$):}} & \multicolumn{2}{l}{$\mathbf{+0.03892}$ nats (95\% CI $+0.03688$ to $+0.04098$) [\textbf{\color{alertred}Equivalence Fails}, $p_{\text{TOST}} > 0.99$]} \\
\bottomrule
\end{tabular}
\vspace{6pt}
\end{table*}
'''

# Insert Table 4 right after Figure 3
fig3_marker = r'\end{figure*}'
# Find the second or third figure* (Fig 1 is first, Fig 3 is next or Fig 2)
# Let's locate Figure 3 specifically
fig3_pos = tex.find('fig_v6_confirmatory.pdf')
assert fig3_pos != -1, 'fig_v6_confirmatory.pdf not found'
fig3_end = tex.find(r'\end{figure*}', fig3_pos) + len(r'\end{figure*}')

tex = tex[:fig3_end] + '\n' + table4_code + '\n' + tex[fig3_end:]
tex_path.write_text(tex, encoding='utf-8')
print('Table 4 inserted successfully.')
