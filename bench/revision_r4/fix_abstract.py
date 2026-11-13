from pathlib import Path

tex_path = Path('e:/AURA/aura-main/manuscript.tex')
tex = tex_path.read_text(encoding='utf-8')

fixed_abstract = (
    r"\noindent\textbf{\textsf{ABSTRACT}}\quad\textbf{Empirical claims that quantum machine learning (QML) "
    r"outperforms classical models have proliferated in digital health, yet comparison symmetry is rarely audited. "
    r"We evaluated whether reported quantum benefits survive symmetric evaluation requiring matched optimization, "
    r"independent calibration, leak-free partitioning, and resampled inference. In a chest radiograph triage "
    r"prototype across 56,534 held-out patients, default variational quantum circuits (VQC) yielded higher negative "
    r"log-likelihood ($\Delta\text{NLL} = +0.0205$ nats single-run; 10-seed paired $\Delta\text{NLL} = +0.0142$ nats, "
    r"$p = 8.43 \times 10^{-6}$) than matched classical baselines, exceeding random seed variance ($\sigma_{\text{seed}} = 0.0045$ nats). "
    r"Under matched 36-configuration search, optimal VQC test loss fell within operational equivalence on the overall population "
    r"($\Delta\text{NLL} = -0.0026$ nats; 10-seed paired $\Delta\text{NLL} = -0.00067$ nats, $\pm 0.010$ nats margin), "
    r"but equivalence failed under class-balanced macro-averaging ($\Delta\text{NLL}_{\text{macro}} = +0.0389$ nats, "
    r"$p_{\text{TOST}} > 0.99$), while requiring 92.8-fold more training compute. Furthermore, classical polynomial "
    r"($x, x^2$, NLL 1.1092) and trigonometric Fourier surrogates (NLL 1.1123) outperformed the tuned VQC (1.1144). "
    r"Under institutional transfer from MIMIC-CXR to CheXpert, all models degraded ($\Delta\text{NLL} = +0.131$ to $+0.143$ nats, "
    r"with frozen quantum circuits trailing classical baselines by up to $+0.1967$ nats), reflecting general covariate shift "
    r"rather than periodic angle wrapping. In an audit of five reconstructible publications spanning six clinical prediction tasks, "
    r"none of six published claims were verified: one was not reproduced under reported features, one collapsed to chance level (52.1\%) "
    r"after resolving temporal leakage, and four were statistically inconclusive. Across these five audited studies and internal prototype, "
    r"empirical quantum benefits reflected evaluation asymmetries, uncalibrated comparisons, or underpowered cohorts rather than verified "
    r"algorithmic superiority.}\\[6pt]"
)

lines = tex.splitlines()
lines[94] = fixed_abstract
tex = '\n'.join(lines)
tex_path.write_text(tex, encoding='utf-8')
print('Abstract line 95 cleanly restored with full math expressions.')
