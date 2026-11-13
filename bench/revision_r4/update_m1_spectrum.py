from pathlib import Path

tex_path = Path('e:/AURA/aura-main/manuscript.tex')
tex = tex_path.read_text(encoding='utf-8')

old_m1_text = (
    r"In the primary 8-qubit entangled circuit (7 informative clinical channels plus 1 constant baseline "
    r"reference qubit coupled via the CNOT ring), the algebraic decomposition over all 8 physical qubits yields "
    r"$100.0\%$ of non-trivial coefficient energy and $99.99\%$ of logit variance in interaction orders $\ge 2$ "
    r"($0.0\%$ in orders 0--1; $2.2\%$ in order 2; $8.5\%$ in order 3; $42.5\%$ in order 4; $21.0\%$ in order 5; "
    r"$18.3\%$ in order 6; $4.5\%$ in order 7; $3.1\%$ in order 8). Crucially, because the 8th qubit receives a constant "
    r"zero reference input ($x_8 \equiv 0.0$), $\cos(\pi x_8) = 1$ and $\sin(\pi x_8) = 0$; thus Pauli-$Z$ on qubit 8 "
    r"acts as a constant scalar multiplier, and Pauli-$X$ components vanish identically. Consequently, the \textit{effective "
    r"interaction order} across the seven active clinical features is strictly bounded by order 7 (matching the x-axis of "
    r"\textbf{Figure 2a}). In the 10-seed ablation isolating the 7 informative channels (Analysis C1b), the entangled "
    r"CNOT ring circuits similarly concentrated 100.0\% of non-trivial coefficient energy into interaction orders $\ge 2$ "
    r"across all seeds (mean order: 4.3; \textbf{Figure 2a}), whereas unentangled product circuits strictly confine "
    r"interaction order to $\le 1$ by construction (0.0\% cross-channel energy)."
)

new_m1_text = (
    r"In the primary 8-qubit architecture, the 8th qubit receives a constant baseline reference input ($x_8 \equiv 0.0$), "
    r"so $\cos(\pi x_8) = 1$ and $\sin(\pi x_8) = 0$; thus Pauli-$Z$ on qubit 8 acts as a constant scalar multiplier, "
    r"Pauli-$X$ components vanish, and the \textit{effective interaction order} across the seven active clinical features "
    r"is strictly bounded between 0 and 7 (order 8 carries $0.0\%$ effective energy; \textbf{Figure 2a}). Under default "
    r"optimization (Analysis C1), the circuit concentrates $100.0\%$ of its effective coefficient energy and $99.97\%$ "
    r"of logit variance in interaction orders $\ge 2$ ($0.0\%$ in orders 0--1, $2.3\%$ in order 2, $26.8\%$ in order 3, "
    r"$39.7\%$ in order 4, $23.2\%$ in order 5, $4.8\%$ in order 6, and $3.2\%$ in order 7). Under symmetric hyperparameter "
    r"tuning (Analysis C2), the winning VQC shifts slight energy into first-order terms ($4.2\%$ in order 1), but still "
    r"concentrates $95.8\%$ of effective coefficient energy and $85.0\%$ of logit variance in interaction orders $\ge 2$ "
    r"($25.7\%$ in order 2, $21.2\%$ in order 3, $35.8\%$ in order 4, $6.9\%$ in order 5, $5.2\%$ in order 6, and $1.0\%$ in order 7). "
    r"In the 10-seed ablation isolating the 7 clinical channels (Analysis C1b), entangled CNOT ring circuits similarly "
    r"concentrated 100.0\% of coefficient energy into interaction orders $\ge 2$ across all seeds (mean order: 4.3; \textbf{Figure 2a}), "
    r"whereas unentangled product circuits strictly confine interaction order to $\le 1$ by construction (0.0\% cross-channel energy)."
)

assert old_m1_text in tex, 'old_m1_text not found'
tex = tex.replace(old_m1_text, new_m1_text)

tex_path.write_text(tex, encoding='utf-8')
print('M1 effective spectrum text updated successfully.')
