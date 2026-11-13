# REPRODUCIBILITY GAPS AND REMEDIATION LEDGER

**Target Manuscript:** *"Auditing Quantum Machine Learning in Digital Health: One Deployed System and Five Published Claims Under Symmetric Evaluation"*  
**Repository State:** `E:/AURA/aura-main`  
**Audit Date:** October 11, 2026

---

## 1. Executive Summary of Reproducibility Status

Across the repository, reproducibility was evaluated against the highest computational and methodological standards. We identified **four technical gaps** and **three methodological vulnerabilities** requiring remediation before external submission:

```
+----------------------------------------------------+-----------------------+---------------------+
| Gap / Vulnerability                                | Severity Level        | Remediation Status  |
+----------------------------------------------------+-----------------------+---------------------+
| 1. Unfinished Confirmatory Pipeline (C2 Quantum)   | Technical / High      | In Progress         |
| 2. Unresolved Placeholders in v6 Manuscript        | Technical / High      | Blocked on C2/Summ  |
| 3. Credentialed Dataset Redistribution Restrictions| Data Governance / Med | Documented / Pinned |
| 4. Positive-Control Claim C Misinterpretation     | Methodological / High | Text Remediation    |
| 5. Primary Task Construct Validity (Rule Emulation)| Methodological / High | Text Remediation    |
| 6. Statistical Equivalence Language on Small n     | Statistical / High    | Corrected in v5.1   |
| 7. Cross-Platform Threading & Simulator Floating Pt| Software Eng / Low    | Verified (simcheck) |
+----------------------------------------------------+-----------------------+---------------------+
```

---

## 2. Technical Gaps Detail

### Gap 1: Incomplete Confirmatory Suite Execution (C2 Quantum & Summarize)
- **Description:** While evidence extraction of all 56,534 untouched CheXpert patients (`chexpert_fresh_evidence.npz`), the C1b entanglement ablation (all 10 seeds), C2 classical (36 configs), and C2 learnable (36 configs) are complete, `C2 quantum` was interrupted after 4 configs due to an external environment reboot.
- **Consequence:** `v6_confirmatory.json` cannot be built, leaving the 24 placeholders in `MANUSCRIPT_v6_npjDM.md` unresolved and causing `aura/tests/test_manuscript_numbers.py::test_no_placeholders_remain` to fail.
- **Action Required:** Resume `python -m aura.ml.evaluation.v6_confirmatory c2 quantum` (resuming from config 4 of 36) and execute `python -m aura.ml.evaluation.v6_confirmatory summarize`.

### Gap 2: Raw Image Redistribution & Dependency Management
- **Description:** Raw medical imaging data (MIMIC-CXR-JPG, CheXpert, BraTS 2020) cannot be redistributed under public licenses due to credentialed research data use agreements (PhysioNet, Stanford, BraTS).
- **Remediation:** The repository uses a two-tiered reproduction architecture:
  1. *Quick Mode (`python reproduce_paper.py`):* Reproduces all statistical analyses, figures, drift tests, and tables directly from cached, cryptographically pinned evidence tensors (`evidence_ledger.json`).
  2. *Full Mode (`python reproduce_paper.py --full`):* Recomputes all feature extractions and raw model predictions directly from the raw imaging directories for authorized researchers.

### Gap 3: Statevector Simulation vs Physical Quantum Hardware
- **Description:** All circuit evaluations in the manuscript run on exact statevector simulators (PyTorch tensor contractions and PennyLane `default.qubit`) at 7, 8, and 10 qubits.
- **Verification:** `v6/simcheck.json` confirms that the PyTorch batched simulator matches PennyLane statevector outputs to within machine precision ($\max |\Delta| < 8 \times 10^{-16}$) across 1,000 random parameter/input vectors.
- **Caveat:** The manuscript must strictly maintain that all findings reflect noise-free simulated quantum states and mathematical properties of the feature maps, without claiming hardware validation.

---

## 3. Historical Defect Register & Resolution Status

The repository maintains a rigorous defect register (Defects D01 through D14):

- **Defect D01 (Asymmetric Calibration):** Quantum arm was temperature scaled while classical arms were uncalibrated. *Resolved by Protocol Rule P1 (symmetric per-arm calibration).*
- **Defect D02 (Optimizer Protocol Bias):** Classical models evaluated with default Adam without grid tuning. *Resolved by Protocol Rule P2 (matched Adam LR grid).*
- **Defect D03 (BraTS 2021 Non-Standard Identifier Leakage):** Erratum in v4; initial cohort used non-standard identifiers. *Resolved by re-embedding all 369 official BraTS 2020 subjects and holding out 111 subjects.*
- **Defect D04 (Shot-Budget Mechanism Artifact):** Proposed finite-shot budget mechanism perturbed analytic expectations rather than executing discrete Poisson/multinomial shots. *Retracted in v5.0 and removed from all claims.*
- **Defect D05 (Equivalence Assertion Overreach on n=447):** Initial drafts asserted equivalence when TOST was non-significant. *Resolved by qualifying language: no difference detected, equivalence requires high-n confirmatory cohort.*
- **Defect D14 (Synthetic Stub Risk):** Potential for test evaluation against synthetic stubs without triggering build failures. *Resolved by SHA-256 byte-digest assertions in CI (`provenance_assertion.py`, `evidence_ledger.json`).*

---

## 4. Remediation Action Plan

1. **Complete Computational Runs:**
   - Execute remaining 32 configs of `c2 quantum`.
   - Run `v6_confirmatory summarize` to generate `v6_confirmatory.json`.
2. **Resolve Manuscript Placeholders:**
   - Populate `MANUSCRIPT_v6_npjDM.md` with final metrics from `v6_confirmatory.json`.
   - Verify `pytest aura/tests/test_manuscript_numbers.py` achieves 100% pass rate.
3. **Harmonize Theoretical Framing:**
   - Eliminate any claims suggesting quantum advantage on positive control target T2b (since classical Gated Head matches Quantum VQC).
   - Characterize the primary clinical task transparently as clinical rule emulation.
