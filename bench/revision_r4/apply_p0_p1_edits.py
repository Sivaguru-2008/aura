import re
from pathlib import Path

tex_path = Path("manuscript.tex")
text = tex_path.read_text(encoding="utf-8")

# 1. Update Algorithm 1
old_alg = """   If massive single confirmatory cohort (N patients):
     Conduct Two One-Sided Tests (TOST) against [-Delta_equiv, +Delta_equiv]:
       t_1 = (mean(d) - (-Delta_equiv)) / SE
       t_2 = (mean(d) - (+Delta_equiv)) / SE
       p_TOST = max(p_1, p_2)."""

new_alg = """   If confirmatory cohort (N patients) or multi-seed evaluation (S seeds):
     Conduct Two One-Sided Tests (TOST) against [-Delta_equiv, +Delta_equiv]:
       t_1 = (mean(d) - (-Delta_equiv)) / SE;  p_1 = 1 - F_t(t_1, df)
       t_2 = (mean(d) - (+Delta_equiv)) / SE;  p_2 = F_t(t_2, df)
       p_TOST = max(p_1, p_2)."""

if old_alg in text:
    text = text.replace(old_alg, new_alg)
    print("Algorithm 1 replaced.")
else:
    print("WARNING: old_alg not found.")

# 2. Update Equations in Methods
old_eqs = r"""Model calibration and discrimination are evaluated via Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE):
\begin{equation}
\text{NLL}(y, \hat{p}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln \hat{p}_{ic}, \quad \text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|
\end{equation}
Under our metric directionality, the paired likelihood difference is defined as:
\begin{equation}
\Delta\text{NLL} = \text{NLL}_{\text{quantum}} - \text{NLL}_{\text{classical}}
\end{equation}
where $\Delta\text{NLL} > 0$ indicates classical superiority (lower loss) and $\Delta\text{NLL} < 0$ indicates quantum superiority.

To test practical statistical equivalence on the confirmatory cohort, Two One-Sided Tests (TOST) \cite{schuirmann1987,lakens2017} evaluate the composite null against the operational margin $\Delta_{\text{equiv}} = \pm 0.010$ nats:
\begin{equation}
t_1 = \frac{\bar{d} - (-\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}, \quad t_2 = \frac{+\Delta_{\text{equiv}} - \bar{d}}{\text{SE}(\bar{d})}, \quad p_{\text{TOST}} = \max\left(1 - \Phi(t_1), 1 - \Phi(t_2)\right)
\end{equation}
Equivalence at significance $\alpha = 0.05$ is established when $p_{\text{TOST}} < 0.05$, which is mathematically equivalent to the 90\% confidence interval falling strictly inside $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$."""

new_eqs = r"""Model calibration and discrimination are evaluated via Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE):
\begin{equation}
\text{NLL}(y, \hat{p}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln \hat{p}_{ic}
\end{equation}
\begin{equation}
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|
\end{equation}
Under our metric directionality, the paired likelihood difference is defined as:
\begin{equation}
\Delta\text{NLL} = \text{NLL}_{\text{quantum}} - \text{NLL}_{\text{classical}}
\end{equation}
where $\Delta\text{NLL} > 0$ indicates classical superiority (lower loss) and $\Delta\text{NLL} < 0$ indicates quantum superiority.

To test practical statistical equivalence on the confirmatory cohort, Two One-Sided Tests (TOST) \cite{schuirmann1987,lakens2017} evaluate the composite null hypothesis $H_0: \mu_d \le -\Delta_{\text{equiv}} \text{ or } \mu_d \ge +\Delta_{\text{equiv}}$ against the alternative $H_1: -\Delta_{\text{equiv}} < \mu_d < +\Delta_{\text{equiv}}$ under the operational margin $\Delta_{\text{equiv}} = \pm 0.010$ nats:
\begin{equation}
t_1 = \frac{\bar{d} - (-\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}, \quad t_2 = \frac{\bar{d} - (+\Delta_{\text{equiv}})}{\text{SE}(\bar{d})}
\end{equation}
\begin{equation}
p_1 = 1 - F_t(t_1; \nu), \quad p_2 = F_t(t_2; \nu), \quad p_{\text{TOST}} = \max(p_1, p_2)
\end{equation}
where $F_t(\cdot; \nu)$ denotes the cumulative distribution function of Student's $t$-distribution with $\nu$ degrees of freedom (equivalently written with $t_2' = \frac{+\Delta_{\text{equiv}} - \bar{d}}{\text{SE}(\bar{d})}$ and $p_2 = 1 - F_t(t_2'; \nu)$). For the massive confirmatory cohort ($N = 56,534$, $\nu = 56,533$), $F_t$ converges asymptotically to the standard normal CDF $\Phi$; for the 10-seed initialization analysis ($\nu = 9$), Student's $t$-distribution with 9 degrees of freedom is strictly evaluated. Equivalence at significance $\alpha = 0.05$ is established when $p_{\text{TOST}} < 0.05$, which is mathematically equivalent to the 90\% confidence interval falling strictly inside $[-\Delta_{\text{equiv}}, +\Delta_{\text{equiv}}]$."""

if old_eqs in text:
    text = text.replace(old_eqs, new_eqs)
    print("Methods equations replaced.")
else:
    print("WARNING: old_eqs not found.")

# 3. Update Architecture in Methods
old_arch = r"Expectation values of Pauli-$Z$ observables $\langle Z_k \rangle(x) \in [-1, 1]$ across all 8 qubits are projected to 6 class logits via linear readout $W_{\text{out}} \langle Z \rangle(x) + b_{\text{out}}$."
new_arch = r"Expectation values of Pauli-$Z$ observables $\langle Z_k \rangle(x) \in [-1, 1]$ across all 8 qubits are projected to 6 class logits via linear readout $W_{\text{out}} \langle Z \rangle(x) + b_{\text{out}}$ (corresponding to the six triage categories: Normal, Pneumonia, Heart Failure, COPD, Malignancy, Pneumothorax). In CheXpert, because automated report labeling lacks the \texttt{hyperinflation} finding, Acuity Class 3 (COPD) has zero positive occurrences in both development and confirmatory test partitions ($y \in \{0, 1, 2, 4, 5\}$). Forward inference and calibration symmetrically compute full 6-way softmax probabilities across all models; sample NLL is evaluated as $-\ln \hat{p}_{i, y_i}$ for observed ground-truth labels $y_i$, while multi-class AUROC is macro-averaged across the 5 observed positive classes to prevent zero-denominator division."

if old_arch in text:
    text = text.replace(old_arch, new_arch)
    print("Architecture text replaced.")
else:
    print("WARNING: old_arch not found.")

tex_path.write_text(text, encoding="utf-8")
print("Done writing manuscript.tex.")
