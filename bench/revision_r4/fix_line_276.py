# -*- coding: utf-8 -*-
from pathlib import Path

target = Path("e:/AURA/aura-main/manuscript.tex")
lines = target.read_text(encoding="utf-8").splitlines()

for i, line in enumerate(lines):
    if "Single-Center Institution Effects vs Transportability" in line:
        print(f"Found at line {i+1}")
        lines[i] = r"  \item \textbf{Single-Center Institution Effects vs Transportability:} All 56,534 patients originate from Stanford Hospital. Deduplication guarantees patient-level statistical independence within this institutional distribution, providing extraordinary precision for internal architectural comparison. However, single-center independence does not guarantee multi-center transportability. This boundary is directly demonstrated by our external transfer evaluation from MIMIC-CXR to CheXpert ($n = 4,800$), where domain shift caused the frozen quantum circuit to trail classical baselines severely ($\Delta\text{NLL} = \mathbf{+0.1589}$ nats vs Product-of-Experts, 95\% CI $0.1422$ to $0.1757$; and $\Delta\text{NLL} = \mathbf{+0.1967}$ nats vs Logistic Regression, 95\% CI $0.1785$ to $0.2149$; both $p < 10^{-15}$), 15- to 20-fold beyond our operational equivalence margin."
        break

target.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Successfully fixed line 276!")
