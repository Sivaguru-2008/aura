# CONSOLIDATED REVISION ROADMAP & SCIENTIFIC ACTION PLAN

**Target Manuscript:** *"Claimed quantum benefits in clinical machine learning do not survive symmetric evaluation"* (npj Digital Medicine Submission Track)  
**Lead Auditor:** World-Class Principal Investigator & Interdisciplinary Audit Board  
**Audit Date:** October 11, 2026

---

## 1. Consolidated Revisions Ranked by Scientific Priority

```
+----+-----------------------------------------------------+-------------------+---------------------+
| P  | Scientific Revision Item                            | Dimension         | Implementation Plan |
+----+-----------------------------------------------------+-------------------+---------------------+
| 1  | Clarify Clinical Target Construct (Rule Emulation)  | Medical AI / Rigor| Methods §5, Results |
| 2  | Reframe Positive Control T2b (No Quantum Superiority| QML / Theory      | Results §2.4, Disc. |
| 3  | Reframe "Deployed System" Terminology               | Clinical AI Ethics| Abstract, Intro, Meth|
| 4  | Qualify Equivalence Language on MIMIC-CXR (n=447)   | Biostatistics     | Results §2.3        |
| 5  | Complete C2 Quantum & Summarize on Confirmatory Set | Computation / CI  | v6_confirmatory.py  |
| 6  | Disclose Hierarchical Seed x Study Bootstrap P=0.062| Statistics        | Results §2.5, Supp. |
| 7  | Resolve Manuscript Placeholders & Abstract (<=150w) | Editorial         | MANUSCRIPT_v6_npjDM |
+----+-----------------------------------------------------+-------------------+---------------------+
```

---

## 2. Detailed Revision Specifications

### Priority 1: Clinical Construct Validity Reframing
- **Text Adjustment:** Explicitly state in the Methods and Introduction that AURA's multi-class fusion layer predicts a target derived from a deterministic weighted heuristic applied to chest radiograph findings.
- **Scientific Impact:** Eliminates any reviewer objection that the paper tests quantum models on an arbitrary target; explains *why* linear models (PoE, Logistic) naturally match or exceed nonlinear circuits on this primary task, and why the positive controls (T2a, T2b) are mandatory.

### Priority 2: Positive Control Reframing (Target T2b)
- **Text Adjustment:** Remove any assertion that T2b demonstrates an "advantage for the quantum circuit itself."
- **Revised Phrasing:**  
  *"On quadratic interaction target T2b, the comparison apparatus detects a significant nonlinear advantage over linear models (negative log-likelihood 0.2515 vs 0.3155, Holm $p < 10^{-4}$). However, an ordinary classical nonlinear model—the attention-gated head—attains an indistinguishable negative log-likelihood of 0.2560. The positive control demonstrates that the apparatus detects nonlinear structure when present, but the quantum circuit matches, and does not exceed, standard classical nonlinear alternatives."*

### Priority 3: System Status Nomenclature
- **Text Adjustment:** Replace references to "a deployed clinical decision-support system" with "a prototype clinical decision-support platform" or "a served decision-support service."
- **Scientific Impact:** Adheres strictly to TRIPOD+AI standards; prevents conflating a locally containerized service with an ethically approved hospital clinical trial.

### Priority 4: Statistical Precision on Equivalence
- **Text Adjustment:** Maintain strict distinction between $n=447$ and $n=56,534$:
  - On $n=447$: *"No likelihood difference was detected ($p = 1.0$), but the sample size was statistically underpowered to reject non-equivalence under our pre-registered $\pm 0.010$ NLL margin ($p_{\text{TOST}} = 0.359$)."*
  - On $n=56,534$: Report the formal TOST equivalence outcome with narrow 90% CI ($\approx \pm 0.0015$ nats).

### Priority 5: Complete Computational Confirmatory Run
- Run remaining configurations of C2 Quantum.
- Execute `python -m aura.ml.evaluation.v6_confirmatory summarize`.
- Replace all 24 `⟨…⟩` placeholder tokens in `MANUSCRIPT_v6_npjDM.md`.
- Run `pytest aura/tests/test_manuscript_numbers.py` to confirm 100% verification pass.
