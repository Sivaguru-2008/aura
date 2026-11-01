# Submission Cover Letter: npj Digital Medicine

**To:** The Editor-in-Chief and Editorial Board, *npj Digital Medicine* (Nature Portfolio)  
**From:** Sivaguru R. M.  
**Affiliation:** Independent Researcher, Chennai, India  
**Correspondence:** sivagurumurugan1@gmail.com  
**Manuscript Title:** *Claimed Quantum Benefits in Clinical Machine Learning Do Not Survive Symmetric Evaluation*  
**Article Type:** Original Article  
**Repository & Certified Artifacts:** [https://github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  
**Date:** 25 October 2026  

---

Dear Editors,

Please find enclosed our manuscript, **"Claimed Quantum Benefits in Clinical Machine Learning Do Not Survive Symmetric Evaluation"**, which we submit for consideration as an Original Article in *npj Digital Medicine*.

Over the past five years, the application of quantum machine learning (QML) to clinical medicine and digital health has attracted substantial attention and investment. Dozens of peer-reviewed publications have reported that near-term variational quantum circuits (VQCs) and quantum kernels achieve dramatic performance improvements over classical machine learning algorithms on medical imaging and clinical tabular diagnostic tasks. However, as highlighted in a recent systematic review in this journal (Gupta et al., *npj Digit. Med.*, 2025), comparisons in applied clinical ML are frequently vulnerable to evaluation asymmetries: classical comparators are frequently left uncalibrated, run under default hyperparameters, or subject to subtle data leakage across patient partitions.

In this work, we present an end-to-end, multi-cohort empirical audit investigating whether claimed quantum benefits in digital health survive *symmetric evaluation*—a rigorous methodological standard where every comparator receives matched hyperparameter search budgets, independent probability calibration, leak-free preprocessing, and competitive baseline capacity.

### Key Empirical Findings & Methodological Contributions

1. **Massive Confirmatory Evaluation ($N = 56,534$ Patients):**
   Evaluating a clinical decision-support prototype on an untouched, patient-disjoint confirmatory cohort of 56,534 frontal chest radiographs from CheXpert (zero patient overlap with development: $|S_{\text{dev}} \cap S_{\text{test}}| = 0$):
   - **Default Matched Baseline (Analysis C1):** Under matched default optimization, the quantum variational circuit was materially inferior to classical Product-of-Experts on probability calibration and negative log-likelihood ($\Delta\text{NLL} = +0.0205$ nats, 95% CI: $0.0186$ to $0.0224$; paired $t = 21.03, p = 7.65 \times 10^{-98}$).
   - **Capacity-Matched Entanglement Ablation (Analysis C1b):** Across ten independent prime seeds on clinical feature channels, removing two-qubit CNOT entangling gates improved likelihood in 9 of 10 seeds (mean $\Delta\text{NLL} = +0.0104$, $t = 4.61, p = 0.001269$), demonstrating an inductive bias mismatch with additive clinical diagnostic rules.
   - **Symmetric 36-Trial Hyperparameter Search (Analysis C2):** When both quantum and classical architectures received an identical budget of $k = 36$ hyperparameter configurations, the optimal quantum model achieved a tiny likelihood shift over classical linear PoE ($\Delta\text{NLL} = -0.00260$ nats; 95% CI: $-0.00359$ to $-0.00160$; $p = 3.07 \times 10^{-7}$) and nonlinear Gated Heads ($\Delta\text{NLL} = -0.00116$ nats; 95% CI: $-0.00192$ to $-0.00041$; $p = 0.00247$). Both confidence intervals fell entirely within our pre-registered operational equivalence margin ($\pm 0.010$ nats; Two One-Sided Tests $p_{\text{TOST}} < 10^{-6}$), confirming practical operational equivalence rather than transformative clinical advantage.

2. **Computational Effort and Timing Accounting:**
   We report full wall-clock training accounting across all 36 configurations per arm ($161.4$\,s for classical PoE, $228.1$\,s for classical Gated Head, and $14,969.4$\,s [4.16 hours] for quantum VQC; Intel CPU, float64 statevector simulation). The quantum model consumed $92.8\times$ more training compute than classical PoE. We explicitly distinguish equal configuration counts ($k = 36$) from equal compute budgets; these results represent a search-trial-matched rather than compute-matched comparison.

3. **Field-Wide Literature Benchmark Audit:**
   We independently audited and reimplemented five published clinical QML studies across six prediction tasks under symmetric 25-fold cross-validation with dependency-corrected Nadeau–Bengio inference. **Zero of the six published quantum advantage claims survived symmetric evaluation:**
   - In Chronic Kidney Disease (*PeerJ*, 2025), classical baselines were significantly superior ($\Delta\text{Acc} = -9.65\%$, $p = 0.000915$).
   - In Schizophrenia EEG, claimed quantum superiority ($98.6\%$) was an artifact of random temporal epoch splitting; when evaluated under subject-grouped cross-validation ($N = 84$ subjects), quantum accuracy collapsed to chance ($52.1\%$, $p = 0.3187$).
   - In Parkinson's Disease voice, simply tuning the classical comparator eliminated the reported quantum gap (reducing it from $+28.2\%, p = 0.0105$ to $+0.1\%, p = 0.9876$).
   - In Pima Indian Diabetes, the result was statistically inconclusive ($\Delta\text{AUC} = -5.30\%$, 95% CI: $-11.00\%$ to $+0.39\%$, $p = 0.066610$), precluding claims of superiority under $\alpha = 0.05$.
   - In Heart Disease and Glioma grading, wide corrected confidence intervals demonstrated that comparisons were statistically inconclusive rather than equivalent.

4. **17-Item Symmetric Evaluation Checklist:**
   To establish durable standards for digital health evaluation, we propose a 17-item checklist covering per-arm calibration, capacity matching, split disjointness, leakage audits, multiplicity correction, and cryptographically verified continuous-integration reproducibility.

### Pre-empting Potential Reviewer Enquiries

- **Target Definition:** In our chest radiograph triage prototype, the prediction target is explicitly characterized as an additive decision-support rule emulation derived from DenseNet-121 radiological probabilities, which allowed closed-form derivation of the Bayes-optimal ceiling ($0.9991$) and verified experimental sensitivity via synthetic quadratic positive controls ($\Delta = -0.064$, $p = 2.7 \times 10^{-4}$). In contrast, our brain MRI and literature benchmark evaluations examine hard histopathological gold standards (WHO CNS glioma grades across 369 official BraTS-2020 subjects) and clinical laboratory assays (eGFR, serum creatinine).
- **Qubit Width and Simulation Fidelity:** While 8–10 qubit circuits are classically simulable in milliseconds, over 80% of peer-reviewed digital health QML literature specifically claims empirical quantum advantages within this 4–12 qubit regime. Auditing this exact regime is urgent because translational clinical programs are actively citing these claims. Furthermore, exact classical statevector simulation eliminates physical decoherence and shot noise, demonstrating that the failure of quantum advantage is mathematical and geometric (Huang et al., *Nat. Commun.* 2021; Kübler et al., 2021) rather than an engineering artifact of current NISQ hardware.
- **Reproducibility & Open Science:** Every headline number, test statistic, confidence interval, and figure in the manuscript is backed by raw frozen NumPy arrays and JSON artifacts in the public repository and verified by automated continuous integration test suites (`test_doc_numbers.py`).

### Suggested Reviewers

We suggest the following independent referees who combine machine learning methodology, clinical digital health, and quantum computing expertise:
- **Dr. Maria Schuld** (Xanadu / University of KwaZulu-Natal) — Leading authority on quantum kernel methods and supervised QML benchmarking.
- **Dr. Hsin-Yuan (Robert) Huang** (California Institute of Technology / Google Quantum AI) — Expert on quantum learning theory, geometric difference, and inductive bias.
- **Dr. Gaël Varoquaux** (Inria / Scikit-learn / Parietal) — Pioneer in clinical machine learning evaluation, statistical benchmarking integrity, and imaging methodology.
- **Dr. Matthew Roberts** (Department of Radiology, University of Cambridge) — Lead author of the benchmark medical imaging methodological audit Roberts et al. (*Nat. Mach. Intell.* 2021).

### Declarations

- **Competing Interests:** The author declares no competing financial or non-financial interests.
- **Originality:** This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere.
- **Data & Code Availability:** All code, simulation harnesses, audit pipelines, and raw prediction artifacts are publicly accessible at [https://github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura).

Thank you for your time and consideration of our manuscript.

Sincerely,

**Sivaguru R. M.**  
Independent Researcher, Chennai, India  
Correspondence: sivagurumurugan1@gmail.com  
GitHub: [https://github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  
