# Submission Cover Letter & Author Declaration

**Target Journal:** *npj Digital Medicine* (Nature Portfolio)  
**Article Type:** Article (Original Research / Evidence Audit)  
**Manuscript Title:** Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit  
**Author:** Sivaguru R. M.  
**Affiliation:** Independent researcher  
**Correspondence:** sivagurumurugan1@gmail.com  
**Date of Submission:** October 2026  

---

### To the Editors of *npj Digital Medicine*:

I am pleased to submit our original research article, **“Claimed quantum benefits in clinical machine learning could not be verified under symmetric evaluation: an empirical and literature audit,”** for consideration for publication in *npj Digital Medicine*.

### Scientific Context and Rationale for *npj Digital Medicine*
Clinical artificial intelligence is currently experiencing rapid methodological diversification. Across digital health, novel model architectures—from foundation model adapters and attention-gated heads to hybrid classical-quantum circuits—are routinely reported to deliver superior predictive performance. However, evaluating whether an architectural component offers genuine inductive efficiency versus apparent superiority driven by asymmetric care remains a pervasive challenge. When a novel model is aggressively tuned and calibrated while standard classical baselines are implemented with default settings, evaluated under global pre-split dimensionality reduction, or assessed without paired inferential tests, biomedical researchers and procurement committees can be severely misled.

This paper addresses this challenge by examining quantum machine learning (QML) in digital health—a high-profile paradigm where empirical claims of superiority have proliferated despite theoretical limits at low qubit widths (4 to 12 qubits). Following a recent systematic review in *npj Digital Medicine* (Gupta et al., 2025) which synthesized published metrics and observed negligible differences, our study goes substantially deeper: we directly audit the raw code, data partitioning, hyperparameter spaces, and binary prediction arrays to investigate *why* these discrepancies occur.

### Central Contributions and Key Empirical Findings
We formulate an operational, reproducible **seven-rule symmetric evaluation protocol** and an actionable **17-item audit checklist** designed to eliminate methodological asymmetries in clinical AI evaluations. We apply this protocol across both an internal clinical decision-support prototype and an independent field audit of the published literature:

1. **Internal Pre-Specified Confirmatory Test ($N = 56,534$ Patients):**
   * Using an untouched test cohort of 56,534 independent human patients from CheXpert (one frontal radiograph per patient; verified zero patient overlap with development data), we evaluated a variational quantum circuit against matched classical baselines for medical triage. The upstream DenseNet-121 visual extractor (`best_model.pt`) was trained exclusively on **MIMIC-CXR**; thus, the 56,534 CheXpert patients were completely untouched by *both* the feature extractor and downstream heads.
   * Under default matched optimization (Analysis C1), the quantum circuit checkpoint performed worse on negative log-likelihood than matched classical baselines ($\Delta\text{NLL} = +0.0205$ nats, 95% test-sampling CI 0.0186 to 0.0224). Calibration error (ECE = 0.0076, 95% bootstrap CI 0.0051 to 0.0119) was statistically indistinguishable from the classical Gated Head (0.0085, 95% CI 0.0071 to 0.0130).
   * Removing entanglement from the circuit improved likelihood across random seeds ($\Delta\text{NLL} = +0.0104$ nats, seed-level 95% CI 0.0049 to 0.0158, $p = 0.0013$), an effect traced to inductive bias mismatch on an additive clinical triage rule.
   * When granted symmetric hyperparameter search budgets over 36 configurations (Analysis C2; sampling 27% of the quantum grid vs 60% of the classical grid), the optimal quantum configuration achieved a small likelihood difference over linear Product-of-Experts ($\Delta\text{NLL} = -0.0026$ nats, 95% CI $-0.00359$ to $-0.00160$) and nonlinear Gated Heads ($\Delta\text{NLL} = -0.0012$ nats, 95% CI $-0.00192$ to $-0.00041$). Crucially, this observed difference fell entirely within our pre-specified operational equivalence margin ($\pm 0.010$ nats; Two One-Sided Test $p_{\text{TOST}} < 10^{-6}$) and within training-seed variance ($\sigma_{\text{seed}} = 0.0116$ nats), while requiring 92.8-fold more training compute.
   * Across random initializations, classical models exhibited negligible seed standard deviation (PoE $\sigma_{\text{seed}} \approx 10^{-5}$ nats, Gated Head $\sigma_{\text{seed}} \approx 2 \times 10^{-4}$ nats), whereas VQC exhibited $\sigma_{\text{seed}} = 0.0116$ nats. Because $\sigma_{\text{seed}} > \Delta_{\text{equiv}} = 0.010$ nats, checkpoint-level equivalence does not establish architectural equivalence across initializations.
   * Evaluating condition-specific subgroups on the 56,534 patients revealed substantial trade-offs: while quantum optimization reduced loss on Pneumothorax ($\Delta = -0.0846$, 95% CI $[-0.0859, -0.0833]$), classical models achieved markedly superior calibration on acute conditions like Pneumonia ($\Delta = +0.1446$, 95% CI $[+0.1364, +0.1528]$) and Malignancy ($\Delta = +0.0706$, 95% CI $[+0.0694, +0.0717]$). These opposing shifts cancel out in the aggregate mean, proving that aggregate equivalence can mask acute clinical trade-offs.
   * Under institutional domain shift (MIMIC-CXR $\to$ CheXpert, $n=4,800$), the quantum circuit degraded severely ($\Delta\text{NLL} = +0.1589$ to $+0.1967$), exposing a structural vulnerability: periodic trigonometric angle encoding wraps inputs under out-of-distribution covariate scale drift.

2. **Systematic Field Audit of Published Literature (5 Reimplemented Studies, 6 Tasks):**
   * From 231 screened PubMed records, 8 publications met eligibility criteria, of which 5 publications spanning 6 clinical prediction tasks were reconstructible and reimplemented with full code (3 were non-reconstructible due to unreleased code, proprietary radiomics, or restricted data).
   * Applying our pre-specified decision criterion requiring statistically significant superiority under leak-free nested cross-validation, **none of the six published advantage claims could be verified under symmetric evaluation**:
     * **1 Claim Not Reproduced Under Published Feature Set:** Chronic Kidney Disease (Parthasarathi et al., 2025; the published 99.2% claim used an under-specified autoencoder and Kalman filter that could not be replicated from the text; on the published 13 LASSO features, reimplemented VQC reached only 83.5%, while classical models achieved 97.4% vs 87.7% nested VQC, $\Delta = -9.65\%$, $p = 0.0009$).
     * **4 Claims Statistically Inconclusive:** Parkinson's disease voice ($p = 0.98$), Pima diabetes ($p = 0.067$), Glioma grading ($p = 0.782$ on TCGA; $p = 0.191$ on BraTS 2020), and Heart disease risk ($p = 0.310$ uncorrected; Nadeau–Bengio corrected $p = 0.897$, 95% CI $-18.45\%$ to $+16.25\%$).
     * **1 Claim Collapsed to Chance Level:** Schizophrenia EEG (Aksoy & Cattan, 2024; the reported 100% accuracy reproduced under random epoch splitting, but collapsed to $52.1\%$ chance level when subject-level temporal partition leakage was eliminated).

3. **Mechanistic and Theoretical Clarification:**
   * Through exact Fourier expansion (M1; effective clinical interaction order bounded at 7 due to constant reference qubit $x_8 \equiv 0$), classical trigonometric surrogates (M2; where order-1 classical trigonometric regression achieves NLL 1.1123, outperforming both tuned quantum 1.1144 and classical PoE 1.1170), and geometric kernel distance bounds ($g(K_{\text{RBF}} \parallel K_Q) = 1.2712 \ll 44.72$), we mathematically isolate why shallow quantum circuits provide no inductive advantage over classical models in these tabular and imaging regimes.
   * Through the **Outcome-Conditioned Entanglement Necessity (OCEN)** score test and **CENSA** algorithm (M4), we explain why pruning 50% of candidate entangling gates improves out-of-sample likelihood ($p = 0.0085$), mathematically proving that heuristic dense entanglers inflate parameter variance without reducing clinical prediction bias.

### Open Science, Reproducibility, and Reporting Standards
* **Pre-Specified Protocol:** The confirmatory protocol, primary endpoints, equivalence margin ($\pm 0.010$ nats), and decision rules were pre-specified and deposited in `docs/PREREGISTRATION_v6_2026-10-03.md` (Git commit `61e777b`) on October 3, 2026, prior to test data scoring on October 11, 2026.
* **Reporting Guidelines:** The manuscript is fully compliant with **TRIPOD+AI** (line-by-line item mapping provided in Supplementary Note 10) and **PRISMA 2020** (Supplementary Note 6).
* **One-Command Reproduction:** All experimental code, simulation harnesses, and binary prediction artifacts are openly available at `https://github.com/Sivaguru-2008/aura`. The entire Level B numerical verification suite executes in under 12 seconds via `python reproduce_paper.py`.

---

## Author Declaration & Mandatory Statements

1. **Originality and Prior Publication:**
   The manuscript is original work, has not been published previously, and is not currently under consideration for publication elsewhere.

2. **Ethical Compliance and Data Governance:**
   This study involves secondary analysis of established, fully de-identified clinical and biomedical databases:
   * MIMIC-CXR-JPG (v2.0.0) under PhysioNet Credentialed Health Data License 1.5.0.
   * CheXpert-v1.0-Small under the Stanford University Research Data Agreement.
   * BraTS 2020 via the Synapse repository.
   * UCI Machine Learning Repository cohorts (CKD, Parkinson's Voice, Pima Diabetes, Heart Disease).
   * MSU Adolescent Schizophrenia EEG database.
   * The Cancer Genome Atlas (TCGA) Glioma project via NIH Genomic Data Commons.
   All patient records were de-identified in source repositories, and no direct human participant intervention was performed.

3. **Competing Interests:**
   The author declares no competing financial or non-financial interests.

4. **Data Availability:**
   All public databases are accessible via their respective credentialed repositories. All derived model parameter checkpoints, per-patient prediction arrays, and numerical summary ledgers are deposited in the open GitHub repository.

5. **Code Availability:**
   All source code and reproduction scripts are released openly on GitHub: `https://github.com/Sivaguru-2008/aura`.

Thank you for your consideration of this work. I look forward to your editorial response.

Sincerely,  
**Sivaguru R. M.**  
Independent Researcher, Chennai, India  
Email: sivagurumurugan1@gmail.com
