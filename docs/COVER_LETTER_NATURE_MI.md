# Submission Cover Letter: Nature Machine Intelligence

**To:** The Editors, *Nature Machine Intelligence*  
**From:** Sivaguru R. M.  
**Affiliation:** AURA Research Initiative & Medical Machine Learning Laboratory  
**Correspondence:** sivagurumurugan1@gmail.com  
**Manuscript Title:** *Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system and a field-wide empirical audit*  
**Repository & Certified Artifacts:** [github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  
**Date:** 21 September 2026  

---

Dear Editors,

Please find enclosed our manuscript, *"Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system and a field-wide empirical audit"*, which we submit for consideration as an Article in *Nature Machine Intelligence*.

Over the past five years, quantum machine learning (QML) applied to healthcare has generated immense interest, with scores of peer-reviewed articles claiming that small quantum circuits already outperform classical deep networks and support vector machines on diagnostic imaging and clinical tabular tasks. Yet, as documented in Gupta et al.'s recent systematic review (*npj Digital Medicine*, 2025), over 80% of this literature uses ideal simulation only, comparators are chosen ad-hoc, and nearly none release reproducible code or data. The field currently stands where medical computer vision stood prior to Roberts et al. (*Nat. Mach. Intell.* 2021) and learned PDE solvers prior to McGreivy & Hakim (*Nat. Mach. Intell.* 2024): an active literature unable to distinguish whether no quantum advantage exists or whether current benchmarking practices simply fail to measure it fairly.

In this work, we provide the first comprehensive, end-to-end empirical audit addressing this crisis from both an internal deployed clinical system and a multi-study literature benchmark.

### 1. Key Findings and Methodological Advances

We built AURA, a fully offline clinical decision-support system for chest radiography (MIMIC-CXR, CheXpert) and brain MRI (BraTS 2020) containing a served quantum layer. Having initially observed and published apparent quantum advantages (including a 13.8-fold calibration improvement), we formulated a seven-rule symmetric comparison protocol governing per-arm temperature calibration, capacity matching, optimizer symmetry, and source-level verification. We applied this protocol to every quantum claim in the system:

1. **The Asymmetric Optimization Confound:** A published likelihood advantage of an 8-qubit variational quantum classifier (VQC) over classical Bayesian fusion reversed in sign once all arms were optimized under a single shared protocol. Under matched optimization, all classical and quantum backends became statistically indistinguishable.
2. **Entanglement Costs Likelihood at Matched Capacity:** In a controlled, capacity-matched 10-seed ablation across seven informative evidence channels, entangling two-qubit gates degraded negative log-likelihood in ten of ten seeds (Holm $p = 0.006$).
3. **Equivalence on Histopathological Gold Standards:** In brain MRI glioma grading across all 369 official BraTS-2020 subjects, evaluated via pre-registered 5-fold nested cross-validation with symmetric hyperparameter grids, a 10-qubit fidelity kernel yielded AUROC 0.9422 against 0.9405 for a classical RBF kernel ($\Delta = +0.0017$, 95% CI: −0.0099 to +0.0132, DeLong $p = 0.777$). A pre-registered sensitivity analysis on 111 held-out subjects confirmed this equivalence ($\Delta = -0.0217$, $p = 0.191$).
4. **External Validation Failure:** When transferred frozen to 4,800 external CheXpert studies under pre-registration, the quantum circuit finished last of all four architectures on every metric.
5. **Field-Wide Literature Audit:** Auditing five independent peer-reviewed medical QML studies (spanning Chronic Kidney Disease, Glioma, Parkinson's disease, Diabetes, and Schizophrenia EEG) under symmetric 25-fold cross-validation, **zero of five published claims of quantum advantage survived**. In Chronic Kidney Disease (*PeerJ* 2025), the claimed quantum advantage significantly reversed in favour of classical models ($\Delta = -9.65\%$, $p = 0.000915$). In Parkinson's and EEG studies, reported quantum superiority was traced directly to global data leakage (un-encapsulated PCA and non-disjoint patient splits).

### 2. Pre-empting Potential Reviewer Inquiries

We have specifically designed our study and manuscript to address the three primary concerns rigorous reviewers typically raise:

1. **Prediction Target Grounding (Histopathology vs Proxies):** While our chest radiography fusion experiment utilizes a rule-derived pattern target, this was an intentional methodological asset: knowing the exact deterministic generating rule allowed us to mathematically derive the Bayes-optimal linear ceiling (0.9991) and interaction ceiling (1.0000). This enabled positive controls (§2.3) demonstrating that our apparatus easily detects nonlinear advantages when present ($\Delta = -0.064$, $p = 2.7 \times 10^{-4}$). Furthermore, our brain MRI evaluation (§2.6) evaluates on **hard histopathological WHO CNS tumor grades** across 369 patients, confirmed by surgical biopsy and neuropathology with contrast-enhancement validation ($p = 4.0 \times 10^{-21}$), while our field audit (§2.9) evaluates real clinical biochemical assays (eGFR, serum creatinine) and clinical examinations.
2. **The 8–10 Qubit Scale and Classical Simulability:** Theorists may note that 8–10 qubits are classically simulable in milliseconds. We emphasize that over 80% of peer-reviewed digital health QML literature specifically claims empirical quantum advantage at 4–12 qubits. Auditing this regime is urgent because these exact claims are actively informing translational biomedical programs. Crucially, exact classical statevector simulation is our strongest experimental asset: by removing shot noise, gate errors, and thermal decoherence, we prove that the failure of quantum advantage is not an artifact of noisy NISQ hardware, but a fundamental limitation of inductive bias and geometry (Huang et al., *Nat. Commun.* 2021).
3. **Independent Research Initiative & Continuous Integration Standards:** Conducted independently within the AURA Research Initiative without quantum hardware-vendor sponsorship, this work avoids institutional positive-outcome bias. To ensure unprecedented reproducibility, every metric, interval, and $p$-value in the manuscript is enforced by automated continuous integration test suites (`aura/tests/test_doc_numbers.py`) asserting zero drift against raw hashed JSON artifacts. We also publish a transparent 14-item internal defect register (§S7.3) disclosing the retraction of our own prior mechanisms.

### 3. Suggested Referees

We suggest the following independent experts who possess both the computational learning theory and biomedical imaging background necessary to evaluate this work:
- **Dr. Hsin-Yuan (Robert) Huang** (California Institute of Technology / Google Quantum AI) — Expert in quantum machine learning theory, inductive bias, and empirical benchmarking (`Power of data in quantum machine learning`).
- **Dr. Maria Schuld** (Xanadu / University of KwaZulu-Natal) — Leading authority on quantum kernel methods, supervised QML, and empirical benchmarking pitfalls.
- **Dr. Gaël Varoquaux** (Inria / Scikit-learn / Parietal) — Pioneer in machine learning methodology for medical imaging and statistical benchmarking integrity.
- **Dr. Matthew Roberts** (Department of Radiology, University of Cambridge) — Lead author of the benchmark medical imaging audit Roberts et al. (2021).

We confirm that this manuscript is original, has not been published previously, and is not under consideration elsewhere. Supplementary Information accompanies this submission. All code, checkpoints, pre-registrations, and raw artifacts are immediately available at the public repository.

Thank you for your time and consideration of our work.

Sincerely,

**Sivaguru R. M.**  
AURA Research Initiative & Medical Machine Learning Laboratory  
Correspondence: sivagurumurugan1@gmail.com  
GitHub: [github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  
