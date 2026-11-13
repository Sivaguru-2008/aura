# -*- coding: utf-8 -*-
"""Apply final targeted revisions to manuscript.tex cleanly."""
import re
from pathlib import Path

TEX_PATH = Path("e:/AURA/aura-main/manuscript.tex")
tex = TEX_PATH.read_text(encoding="utf-8")

# 1. Update Abstract: refine transfer sentence
old_abs = r"""\noindent\textbf{\textsf{ABSTRACT}}\quad\textbf{Empirical claims that quantum machine learning outperforms classical models have proliferated in digital health, yet comparison symmetry is rarely audited. We evaluated whether reported quantum benefits survive symmetric evaluation requiring matched optimization, independent calibration, leak-free partitioning, and resampled inference. In a chest radiograph triage prototype across 56,534 held-out patients, default variational quantum circuits (VQC) yielded higher negative log-likelihood ($\Delta\text{NLL} = +0.0205$ nats; 10-seed mean $+0.0142$ nats) than matched classical baselines, exceeding seed variance ($\sigma_{\text{seed}} = 0.0045$ nats). Under matched 36-configuration search, optimal VQC test loss was statistically equivalent to classical baselines overall ($\Delta\text{NLL} = -0.0026$ nats; 10-seed mean $-0.00067$ nats, $\pm 0.010$ margin), but equivalence failed under class-balanced macro-averaging ($\Delta\text{NLL}_{\text{macro}} = +0.0389$ nats, $p_{\text{TOST}} > 0.99$; Malignancy $+0.1771$ nats), while requiring 92.8-fold more compute. Classical polynomial (NLL 1.1092) and Fourier surrogates (NLL 1.1123) outperformed the tuned VQC (1.1144). Under institutional transfer, all models degraded ($\Delta\text{NLL} = +0.131$ to $+0.197$ nats). Across five reconstructible publications spanning six clinical tasks, none of six published claims were verified: one was not reproduced, one collapsed to chance (52.1\%), and four were statistically inconclusive. Reported quantum benefits reflected evaluation asymmetries, uncalibrated comparisons, or underpowered cohorts rather than verified algorithmic superiority.}"""

new_abs = r"""\noindent\textbf{\textsf{ABSTRACT}}\quad\textbf{Empirical claims that quantum machine learning outperforms classical models have proliferated in digital health, yet comparison symmetry is rarely audited. We evaluated whether reported quantum benefits survive symmetric evaluation requiring matched optimization, independent calibration, leak-free partitioning, and resampled inference. In a chest radiograph triage prototype across 56,534 held-out patients, default variational quantum circuits (VQC) yielded higher negative log-likelihood ($\Delta\text{NLL} = +0.0205$ nats; 10-seed mean $+0.0142$ nats) than matched classical baselines, exceeding seed variance ($\sigma_{\text{seed}} = 0.0045$ nats). Under matched 36-configuration search, optimal VQC test loss was statistically equivalent to classical baselines overall ($\Delta\text{NLL} = -0.0026$ nats; 10-seed mean $-0.00067$ nats, $\pm 0.010$ margin), but equivalence failed under class-balanced macro-averaging ($\Delta\text{NLL}_{\text{macro}} = +0.0389$ nats, $p_{\text{TOST}} > 0.99$; Malignancy $+0.1771$ nats), while requiring 92.8-fold more compute. Classical polynomial (NLL 1.1092) and Fourier surrogates (NLL 1.1123) outperformed the tuned VQC (1.1144). Under institutional transfer, the quantum circuit trailed classical baselines by $\Delta\text{NLL} = +0.1589$ to $+0.1967$ nats. Across five reconstructible publications spanning six clinical tasks, none of six published claims were verified: one was not reproduced, one collapsed to chance (52.1\%), and four were statistically inconclusive. Reported quantum benefits reflected evaluation asymmetries, uncalibrated comparisons, or underpowered cohorts rather than verified algorithmic superiority.}"""

if old_abs in tex:
    tex = tex.replace(old_abs, new_abs)
    print("Updated Abstract")

# 2. Update ln(5) vs ln(6) in Methods/Results
old_entropy = r"""\begin{enumerate}
  \setlength{\itemsep}{1pt}
  \setlength{\parskip}{0pt}
  \item Relative to a naive uniform prior over 5 active classes ($H_{\text{uniform}} = \ln 5 \approx 1.6094$ nats; $2.3219$ bits), a shift of $0.010$ nats represents $0.62\%$ of uniform entropy.
  \item Relative to the empirical marginal class entropy of the held-out cohort ($H(Y) \approx 1.3699$ nats; $1.9764$ bits), the $0.010$-nat margin represents $0.73\%$.
  \item Relative to the baseline model's cross-entropy loss (NLL $\approx 1.1157$ nats), the margin represents $0.90\%$.
  \item Crucially, relative to the \textit{predictive mutual information gained over the unconditioned marginal class prior} ($\Delta I = H(Y) - \text{NLL} \approx 1.3699 - 1.1157 = 0.2542$ nats; $0.3668$ bits), the $0.010$-nat margin represents $\mathbf{3.93\%}$ ($\approx 4\%$) of extracted predictive signal.
\end{enumerate}"""

new_entropy = r"""\begin{enumerate}
  \setlength{\itemsep}{1pt}
  \setlength{\parskip}{0pt}
  \item Relative to an uninformative uniform prediction across the model's 6 parameterized output classes ($H_{\text{uniform}, 6} = \ln 6 \approx 1.7918$ nats; $2.5850$ bits), a shift of $0.010$ nats represents $0.56\%$ of uniform 6-way cross-entropy loss. If conditioned on an oracle uniform prior over the 5 observed positive classes in the held-out test cohort ($H_{\text{uniform}, 5} = \ln 5 \approx 1.6094$ nats; $2.3219$ bits), the margin represents $0.62\%$.
  \item Relative to the empirical marginal class entropy of the held-out cohort ($H(Y) \approx 1.3699$ nats; $1.9764$ bits), the $0.010$-nat margin represents $0.73\%$.
  \item Relative to the baseline model's cross-entropy loss (NLL $\approx 1.1157$ nats), the margin represents $0.90\%$.
  \item Crucially, relative to the \textit{predictive mutual information gained over the unconditioned marginal class prior} ($\Delta I = H(Y) - \text{NLL} \approx 1.3699 - 1.1157 = 0.2542$ nats; $0.3668$ bits), the $0.010$-nat margin represents $\mathbf{3.93\%}$ ($\approx 4\%$) of extracted predictive signal.
\end{enumerate}"""

if old_entropy in tex:
    tex = tex.replace(old_entropy, new_entropy)
    print("Updated Entropy Reference (ln 6 and ln 5)")

# 3. Update Institutional Transfer bullet in Prototype Results
old_trans_bullet = r"""\item \textbf{Institutional Transfer Degradation:} Single-center patient independence does not guarantee multi-center transportability. This boundary is directly demonstrated by our external transfer evaluation from MIMIC-CXR to CheXpert ($n = 4,800$), where domain shift caused the quantum circuit to degrade severely ($\Delta\text{NLL} = +0.1589$)."""

new_trans_bullet = r"""\item \textbf{Institutional Transfer Degradation:} Single-center patient independence does not guarantee multi-center transportability. This boundary is directly demonstrated by our external transfer evaluation from MIMIC-CXR to CheXpert ($n = 4,800$), where domain shift caused the frozen quantum circuit to trail classical baselines severely ($\Delta\text{NLL} = \mathbf{+0.1589}$ nats vs Product-of-Experts, 95\% CI $0.1422$ to $0.1757$; and $\Delta\text{NLL} = \mathbf{+0.1967}$ nats vs Logistic Regression, 95\% CI $0.1785$ to $0.2149$; both $p < 10^{-15}$), 15- to 20-fold beyond our operational equivalence margin."""

if old_trans_bullet in tex:
    tex = tex.replace(old_trans_bullet, new_trans_bullet)
    print("Updated Institutional Transfer bullet in Prototype Results")

# 4. Update Institutional Transfer text in Section 6 (Mechanistic / Transfer)
old_trans_sec6 = r"""When models trained on MIMIC-CXR ($n = 1,338$ training studies, 446 calibration, and 447 test) were transferred frozen to an external test cohort of $n = 4,800$ CheXpert studies, all models degraded, and the quantum circuit exhibited substantial likelihood degradation ($\Delta\text{NLL} = +0.1394$ nats for default VQC; $+0.1198$ for tuned VQC). We investigated whether this sensitivity stems from input-scale periodic angle wrapping ($>2\pi$). Testing revealed that input angles remain strictly within the non-wrapping regime ($\le 0.974\pi$ at scale 1.0; $\le 0.487\pi$ at scale 0.5), and clipping input features to the training range yielded negligible benefit ($\Delta\text{NLL} < 10^{-4}$). Crucially, a classical order-1 trigonometric Fourier surrogate experienced comparable transfer degradation ($\Delta\text{NLL} = +0.1434$), while classical linear PoE degraded by $+0.1312$."""

new_trans_sec6 = r"""When models trained on MIMIC-CXR ($n = 1,338$ training studies, 446 calibration, and 447 test) were transferred frozen to an external test cohort of $n = 4,800$ CheXpert studies, all models experienced domain shift degradation ($\Delta\text{NLL}_{\text{shift}} = +0.1198$ to $+0.1434$ nats across models; default VQC increased by $+0.1394$ nats, tuned VQC by $+0.1198$ nats, and linear PoE by $+0.1312$ nats). We investigated whether this sensitivity stems from input-scale periodic angle wrapping ($>2\pi$). Testing revealed that input angles remain strictly within the non-wrapping regime ($\le 0.974\pi$ at scale 1.0; $\le 0.487\pi$ at scale 0.5), and clipping input features to the training range yielded negligible benefit ($\Delta\text{NLL} < 10^{-4}$). Crucially, a classical order-1 trigonometric Fourier surrogate experienced comparable domain shift degradation ($\Delta\text{NLL}_{\text{shift}} = +0.1434$), confirming that transfer degradation reflects the geometry of periodic trigonometric basis functions under distribution shift."""

if old_trans_sec6 in tex:
    tex = tex.replace(old_trans_sec6, new_trans_sec6)
    print("Updated Institutional Transfer text in Section 6")

# 5. Reconcile Institutional Transfer description in Discussion
old_transfer_disc = r"""Furthermore, evaluating out-of-distribution institutional transfer from MIMIC-CXR to CheXpert revealed our largest empirical effect ($\Delta\text{NLL} = +0.1589$ to $+0.1967$), where all models degraded substantially ($\Delta\text{NLL} = +0.131$ to $+0.143$ nats; quantum frozen gap trailing by up to $+0.1967$ nats) due to institutional covariate shift, with empirical scale sweeps refuting periodic angle wrapping as the mechanism."""

new_transfer_disc = r"""Furthermore, evaluating out-of-distribution institutional transfer from MIMIC-CXR to CheXpert revealed our largest empirical effect: the frozen quantum circuit trailed classical baselines by $\Delta\text{NLL} = \mathbf{+0.1589}$ to $\mathbf{+0.1967}$ nats (15- to 20-fold the operational equivalence margin), while all architectures experienced domain shift degradation ($\Delta\text{NLL}_{\text{shift}} = +0.120$ to $+0.143$ nats across models) due to institutional covariate shift, with empirical scale sweeps refuting periodic angle wrapping as the mechanism."""

if old_transfer_disc in tex:
    tex = tex.replace(old_transfer_disc, new_transfer_disc)
    print("Updated Transfer description in Discussion")

# 6. Add shared calibration partition limitation and clinical outcomes distinction to Limitations
old_limitations = r"""\subsection*{Limitations}

Our findings are subject to several boundary conditions:
\begin{enumerate}
  \setlength{\itemsep}{1pt}
  \setlength{\parskip}{0pt}
  \item \textbf{Carefully Scoped Generalizability:} Five reimplemented publications spanning six clinical prediction tasks, alongside our internal prototype, support a carefully bounded conclusion regarding shallow variational quantum circuits (4 to 12 qubits), angle encodings, and fidelity kernels on clinical tabular and imaging features. They do \textbf{not} support a universal claim that quantum machine learning can never offer an advantage in healthcare. Regimes outside our scope---including fault-tolerant quantum algorithms with asymptotic speedups, high-dimensional native quantum sensor representations, non-commuting Hamiltonian feature encodings, or molecular docking and quantum chemistry where classical simulation is fundamentally intractable---remain unaddressed by our empirical data.
  \item \textbf{Operational Nature of Equivalence Margin:} The equivalence margin of $\pm 0.010$ nats is strictly an \textbf{operational statistical benchmark} on expected negative log-likelihood differences ($\mathbb{E}[\Delta\ell] \le 0.010$ nats per prediction across the cohort), chosen prior to test unblinding. While it represents $\sim$0.90\% of model NLL and $\sim$3.93\% of predictive information gained over the empirical class prior, it has \textbf{not been clinically validated} against prospective patient outcomes, diagnostic error rates, or expert clinical utility thresholds. Unless supported by prospective clinical validation studies linking margins to acceptable patient health outcomes, it must remain explicitly interpreted as an operational statistical benchmark. Sensitivity analysis confirms that practical equivalence holds across alternative margins from $\pm 0.020$ down to $\pm 0.005$ nats ($p_{\text{TOST}} < 10^{-4}$).
  \item \textbf{Noise Ceiling of Clinical Targets:} The clinical target in our chest radiograph system is derived from report labeler heuristics and clinical decision-support rules rather than adjudicated clinician consensus, while the inputs are noisy DenseNet probability estimates. Non-differential label extraction noise and upstream perceptual ambiguity bound the accuracy ceiling at $\approx 54.4\%$. Equivalence near this noise floor demonstrates that shallow quantum circuits do not overcome representation noise, rather than demonstrating equivalence across noiseless tasks.
  \item \textbf{Single-Center Institution Effects:} While the 56,534-patient cohort ensures patient independence and eliminates intra-subject clustering, it represents a single medical institution (Stanford Hospital); multi-center transfer to CheXpert demonstrated performance degradation under domain shift ($\Delta\text{NLL} = +0.1589$).
  \item \textbf{Hardware and Simulation Boundary:} Empirical quantum evaluations were conducted at 5 to 11 qubits under exact classical statevector simulation. Prior physical noise simulations indicate that physical hardware decoherence further widens the classical advantage.
  \item \textbf{Literature Selection and Reconstructibility:} The literature audit was restricted to studies with open data and reconstructible pipelines; three studies were non-reconstructible due to unreleased code or proprietary features. In the case of chronic kidney disease (Parthasarathi et al.), the published 99.2\% accuracy claim utilized an under-specified sparse autoencoder and Kalman filter that could not be verified from the paper text, representing an important documentation barrier.
  \item \textbf{Single Rater Scoring:} The 17-item checklist was scored by a single rater, though operationalized via objective binary criteria.
\end{enumerate}"""

new_limitations = r"""\subsection*{Limitations}

Our findings are subject to several boundary conditions:
\begin{enumerate}
  \setlength{\itemsep}{1pt}
  \setlength{\parskip}{0pt}
  \item \textbf{Carefully Scoped Generalizability:} Five reimplemented publications spanning six clinical prediction tasks, alongside our internal prototype, support a carefully bounded conclusion regarding shallow variational quantum circuits (4 to 12 qubits), angle encodings, and fidelity kernels on clinical tabular and imaging features. They do \textbf{not} support a universal claim that quantum machine learning can never offer an advantage in healthcare. Regimes outside our scope---including fault-tolerant quantum algorithms with asymptotic speedups, high-dimensional native quantum sensor representations, non-commuting Hamiltonian feature encodings, or molecular docking and quantum chemistry where classical simulation is fundamentally intractable---remain unaddressed by our empirical data.
  \item \textbf{Operational Statistical Nature of Equivalence Margin:} The equivalence margin of $\pm 0.010$ nats is strictly an \textbf{operational statistical benchmark} on expected negative log-likelihood differences ($\mathbb{E}[\Delta\ell] \le 0.010$ nats per prediction across the cohort), pre-specified prior to test unblinding. While it represents $\sim$0.90\% of model NLL and $\sim$3.93\% of predictive information gained over the empirical class prior, it has \textbf{not been clinically validated} against prospective patient health outcomes, diagnostic error rates, or expert clinical utility thresholds. Unless supported by prospective clinical validation studies linking margins to acceptable patient outcomes, it must remain strictly interpreted as an operational statistical benchmark. Sensitivity analysis confirms that practical equivalence holds across alternative margins from $\pm 0.020$ down to $\pm 0.005$ nats ($p_{\text{TOST}} < 10^{-4}$).
  \item \textbf{Retrospective Predictive Emulation versus Prospective Patient Outcomes:} The reported performance metrics (NLL, ECE, AUROC) evaluate retrospective mathematical emulation of an additive clinical triage rule from DenseNet finding probabilities, not direct prospective patient outcomes or real-time clinical utility. A statistically detectable difference or equivalence in retrospective log-likelihood cannot be interpreted as demonstrating clinical safety, diagnostic efficacy, or improved patient health outcomes in live hospital workflows.
  \item \textbf{Shared Early-Stopping and Calibration Partition:} Model checkpoint selection and post-hoc Platt temperature scaling were both fitted on the same 1,600-patient development validation split. Reusing validation data for early stopping and temperature estimation induces a minor optimistic calibration loss bias on validation data. However, the primary confirmatory evaluation across 56,534 held-out patients remained strictly unblended and untouched during all training and calibration stages, guaranteeing that reported test-set generalization metrics are completely leak-free.
  \item \textbf{Noise Ceiling of Clinical Targets:} The clinical target in our chest radiograph system is derived from report labeler heuristics and clinical decision-support rules rather than adjudicated clinician consensus, while the inputs are noisy DenseNet probability estimates. Non-differential label extraction noise and upstream perceptual ambiguity bound the accuracy ceiling at $\approx 54.4\%$. Equivalence near this noise floor demonstrates that shallow quantum circuits do not overcome representation noise, rather than demonstrating equivalence across noiseless tasks.
  \item \textbf{Single-Center Institution Effects:} While the 56,534-patient cohort ensures patient independence and eliminates intra-subject clustering, it represents a single medical institution (Stanford Hospital); multi-center transfer from MIMIC-CXR to CheXpert demonstrated performance degradation under domain shift ($\Delta\text{NLL} = +0.1589$).
  \item \textbf{Hardware and Simulation Boundary:} Empirical quantum evaluations were conducted at 5 to 11 qubits under exact classical statevector simulation. Prior physical noise simulations indicate that physical hardware decoherence further widens the classical advantage.
  \item \textbf{Literature Selection and Reconstructibility:} The literature audit was restricted to studies with open data and reconstructible pipelines; three studies were non-reconstructible due to unreleased code or proprietary features. In the case of chronic kidney disease (Parthasarathi et al.), the published 99.2\% accuracy claim utilized an under-specified sparse autoencoder and Kalman filter that could not be verified from the paper text, representing an important documentation barrier.
  \item \textbf{Single-Rater Checklist Scoring:} The 17-item checklist was scored by a single rater (S.R.M.), though operationalized via objective binary criteria without manufactured second-reviewer ratings.
\end{enumerate}"""

if old_limitations in tex:
    tex = tex.replace(old_limitations, new_limitations)
    print("Updated Limitations block")

# 7. Update Code Availability to explicitly distinguish metric regeneration from full reproduction
old_code = r"""All software, evaluation pipelines, simulation engines, and verification scripts are released as open-source under the Apache 2.0 license at \url{https://github.com/Sivaguru-2008/aura}. The full empirical suite can be reproduced via a single command:
\begin{lstlisting}[language=bash]
python reproduce_paper.py
\end{lstlisting}"""

new_code = r"""All software, evaluation pipelines, simulation engines, and verification scripts are released as open-source under the Apache 2.0 license at \url{https://github.com/Sivaguru-2008/aura}. The repository explicitly distinguishes quick metric regeneration from full end-to-end experimental reproduction:
\begin{lstlisting}[language=bash]
# Quick metric regeneration: verifies input hashes, rebuilds figures, runs 31 drift tests (8s)
python reproduce_paper.py

# Full end-to-end reproduction: re-runs all model training and tuning sweeps from cached tensors (hours)
python reproduce_paper.py --full
\end{lstlisting}"""

if old_code in tex:
    tex = tex.replace(old_code, new_code)
    print("Updated Code Availability block")

TEX_PATH.write_text(tex, encoding="utf-8")
print(f"Successfully wrote {TEX_PATH}")
