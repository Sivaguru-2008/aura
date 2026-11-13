# Submission Cover Letter: Quantum Machine Intelligence

**To:** Prof. Peter Wittek / The Editors-in-Chief, *Quantum Machine Intelligence* (Springer Nature)  
**From:** Sivaguru R. M.  
**Affiliation:** AURA Research Initiative & Medical Machine Learning Laboratory  
**Correspondence:** sivagurumurugan1@gmail.com  
**Manuscript Title:** *Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system and a field-wide empirical audit*  
**Repository & Certified Artifacts:** [github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  
**Date:** 21 September 2026  

---

Dear Editors,

Please find enclosed our manuscript, *"Quantum advantage claims in clinical machine learning do not survive a symmetric comparison: six negative results from a deployed decision-support system and a field-wide empirical audit"*, which we submit for consideration as an Original Research Article in *Quantum Machine Intelligence*.

Over the past five years, the intersection of quantum computing and medical informatics has produced a wave of papers claiming that near-term variational algorithms and quantum kernels achieve dramatic performance advantages over classical deep learning and kernel methods on clinical diagnostic tasks. However, as highlighted in recent methodological syntheses (Gupta et al., *npj Digit. Med.* 2025; Bowles et al., 2024), published comparisons in this domain often suffer from severe asymmetries: uncalibrated classical baselines, heterogeneous optimizer budgets, unequal hyperparameter grids, and data leakage across splits.

In this work, we present a comprehensive, multi-cohort empirical audit conducted across both an internal, deployed clinical decision-support pipeline (AURA) and an external 5-study literature benchmark.

### Core Scientific Contributions:

1. **Protocol for Symmetric Comparison:** We establish an eight-rule protocol enforcing per-arm Platt/temperature calibration (P1), capacity matching (P2), split disjointness (P3), multiplicity correction (P4), artifact-derived reporting with automated continuous integration tests (P6), source-level mechanism audits (P7), and optimizer symmetry (P8).
2. **Dissolution of Apparent Advantages under Matched Optimization:** We demonstrate that an apparent likelihood advantage of an 8-qubit variational circuit over a Bayesian product-of-experts reverses in sign when all models are trained under identical Adam optimizer budgets.
3. **Capacity-Matched Entanglement Ablation:** Across 10 random seeds on seven informative clinical channels, removing the CNOT entangling ring improved negative log-likelihood in 10 of 10 seeds (Holm $p = 0.006$).
4. **Indistinguishability on Official BraTS-2020 (369 Subjects):** Under pre-registered 5-fold nested cross-validation with symmetric hyperparameter grids on all 369 official BraTS-2020 glioma patients, a 10-qubit fidelity kernel yielded AUROC 0.9422 against 0.9405 for an RBF kernel ($\Delta = +0.0017$, DeLong $p = 0.777$). A pre-registered sensitivity analysis on 111 held-out subjects confirmed this null ($\Delta = -0.0217$, $p = 0.191$).
5. **Field-Wide Multi-Study Literature Audit:** Auditing five peer-reviewed medical QML studies under 25-fold symmetric cross-validation (spanning Chronic Kidney Disease, Glioma, Parkinson's disease, Diabetes, and Schizophrenia EEG), **zero of five published claims of quantum advantage survived**. In Chronic Kidney Disease (*PeerJ* 2025), the claimed advantage reversed significantly in favor of classical models ($\Delta = -9.65\%$, $p = 0.000915$). In Parkinson's and EEG studies, reported superiority was traced to global PCA leakage and non-disjoint epoch splitting.
6. **Pre-empting Reviewer Inquiries:**
   - *Target Grounding:* Highlights the hard histopathological gold standard of BraTS-2020 (§2.6) and clinical laboratory assays in the field audit (§2.9), alongside the analytical positive controls on chest radiography (§2.3) that prove apparatus sensitivity.
   - *The 8–10 Qubit Scale:* Explains that while classically simulable, >80% of peer-reviewed digital health QML literature specifically claims advantage at 4–12 qubits; exact statevector evaluation removes shot noise and hardware infidelity, proving that the lack of advantage is mathematical and geometric (Huang et al., Kübler et al.) rather than an artifact of NISQ hardware.
   - *Open Science & CI Drift Tests:* Demonstrates that all numbers in the text are cryptographically verified by automated CI tests (`aura/tests/test_doc_numbers.py`).

We believe this paper will serve as an essential reference for the *Quantum Machine Intelligence* readership, providing a rigorous benchmarking standard that future applied quantum machine learning studies can build upon.

We confirm that this manuscript is original, has not been published previously, and is not under consideration elsewhere. Supplementary Information accompanies this submission. All code and artifacts are publicly accessible at the repository.

Thank you for your consideration.

Sincerely,

**Sivaguru R. M.**  
AURA Research Initiative & Medical Machine Learning Laboratory  
Correspondence: sivagurumurugan1@gmail.com  
GitHub: [github.com/Sivaguru-2008/aura](https://github.com/Sivaguru-2008/aura)  
