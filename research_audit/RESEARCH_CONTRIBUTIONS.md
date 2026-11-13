# RECONSTRUCTION OF SCIENTIFIC CONTRIBUTIONS AND CORE RESEARCH THESIS

**Target Manuscript:** *"Auditing Quantum Machine Learning in Digital Health: One Deployed System and Five Published Claims Under Symmetric Evaluation"*  
**Role:** Senior ML Benchmarking Specialist & Principal Investigator  
**Audit Date:** October 11, 2026

---

## 1. Critical Evaluation of Candidate Scientific Contributions

We critically examined three candidate contributions proposed in the original manuscript drafts:

### Contribution A: The Seven-Rule Symmetric Evaluation Framework
- **The Proposal:** A formal 7-rule protocol (P1: Per-arm calibration; P2: Matched tuning & capacity; P3: Disjoint splits at unit of independence; P4: Inference before ranking; P5: Negative outcome reporting; P6: Numbers derived from artifacts; P7: Source-level mechanism audit).
- **Adversarial Assessment:**
  - Are these rules individually novel? **No.** Rules P1–P4 are established best practices in biostatistics and clinical prediction modeling (TRIPOD, PROBAST). Rule P5 is standard publication ethics. Rule P6 is computational reproducibility.
  - **Where Does Genuine Novelty Lie?** The contribution is the **first systematic operationalization and empirical deployment of these rules specifically targeted at Quantum Machine Learning in Digital Health**, backed by an automated 17-item scoring checklist and continuous integration audit harness. It moves QML evaluation from ad-hoc exploratory benchmarking to regulatory-grade clinical AI standards.

### Contribution B: Empirical Isolation of Failure Mechanisms in QML
- **The Proposal:** Identifying why quantum models appear superior in published literature.
- **Adversarial Assessment:**
  - **Causally Isolated Mechanisms:**
    1. *Asymmetric Post-Hoc Calibration (Causal):* Proven in AURA. Calibrating only the quantum head explains 75% of the reported ECE gap.
    2. *Temporal / Patient Split Leakage (Causal):* Proven via Factorial F3 on EEG and F1 on Parkinson's. Epoch and recording-level splits generate artificial 98–100% accuracy claims that collapse to 52% upon subject grouping.
    3. *Classical Comparator Undertuning (Causal):* Proven on CKD and Pima. Substituting an untuned scikit-learn default baseline with nested cross-validated classical selection reverses the claimed advantage.
  - **Correlational / Non-Causal Hypotheses:**
    - Attributing external transfer degradation strictly to quantum parameter count remains correlational; domain shift and label schema mismatches are substantial confounders.

### Contribution C: External Generalization Under Institutional Distribution Shift
- **The Proposal:** Evaluating frozen quantum circuits against classical baselines across two clinical health systems (MIMIC-CXR $\to$ CheXpert).
- **Adversarial Assessment:**
  - Highly valuable as an empirical case study of clinical domain shift in QML.
  - Must be reported descriptively: all models degrade under transfer, and the quantum circuit does not exhibit superior inductive resilience against real-world dataset shift.

---

## 2. Ranked List of Scientifically Defensible Contributions

1. **Rank 1 (Highest Evidentiary Strength): Direct Causal Dissection of Spurious Quantum Advantages.**
   We isolate and prove three specific methodological mechanisms—asymmetric calibration, split contamination, and comparator undertuning—that manufacture apparent quantum superiority in medical AI literature.
2. **Rank 2 (Methodological Value): The Pre-registered Symmetric Evaluation Protocol & Checklist.**
   A concrete, validated 17-item evaluation instrument that provides clinical AI journal editors, reviewers, and regulatory bodies with unambiguous decision criteria to audit novel component claims.
3. **Rank 3 (Theoretical Foundation): Mechanistic Equivalence and Inductive Bias Limits.**
   Analytic and empirical proof that low-depth angle-encoded variational circuits compute low-order Fourier polynomials whose fidelity kernels reduce to classical closed forms, explaining why quantum components cannot outperform classical nonlinear heads on tabular clinical features.
4. **Rank 4 (Clinical Realism): End-to-End Retrospective External Validation.**
   Demonstrating that frozen quantum decision-support layers provide zero empirical benefit across 56,000+ out-of-sample patients from an independent medical center.

---

## 3. Recommended Primary Research Thesis

> **Primary Scientific Thesis:**  
> *"In clinical digital health applications operating on low-dimensional extracted features ($\le 10$ qubits), published claims of empirical quantum advantage are artifacts of asymmetric evaluation choices—specifically one-sided calibration, comparator undertuning, and partition contamination. Under symmetric optimization, independent calibration, and leak-free splitting, variational quantum circuits provide no measurable benefit over classical nonlinear baselines on both rule-derived triage and clinical prediction tasks."*

This thesis is impervious to adversarial attack because it does not claim quantum algorithms can never work in principle; it rigorously documents that current empirical claims in health AI fail to survive fair comparison.
