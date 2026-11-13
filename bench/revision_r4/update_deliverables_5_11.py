"""
Script to update DELIVERABLE_05_STATISTICAL_VERIFICATION.md and DELIVERABLE_11_CHANGE_LOG.md
with the final targeted audit details (Manuscript 6).
"""
import shutil
from pathlib import Path

BASE_DIR = Path("E:/AURA")
AUDIT_DIR = BASE_DIR / "audit"
DOCS_DIR = BASE_DIR / "aura-main" / "docs"

# 1. Update Deliverable 5
d5_path = AUDIT_DIR / "DELIVERABLE_05_STATISTICAL_VERIFICATION.md"
d5_text = d5_path.read_text(encoding="utf-8")

# Let's inspect section 2.1 in d5
old_sec2_1 = """### 2.1 Formal Hypothesis Definition
Let $d_i = \\ell_{\\text{quantum}}(x_i, y_i) - \\ell_{\\text{classical}}(x_i, y_i)$ denote the paired difference in Negative Log-Likelihood (NLL) for observation $i$. Let $\\mu_d = \\mathbb{E}[d_i]$ denote the expected mean paired difference. Under our directionality convention:
- $\\mu_d > 0 \\implies$ Classical model achieves lower loss (Classical Superiority).
- $\\mu_d < 0 \\implies$ Quantum model achieves lower loss (Quantum Superiority).

To evaluate operational equivalence rather than mere absence of difference, Schuirmann's Two One-Sided Tests (TOST) procedure evaluates the composite null hypothesis of non-equivalence against the alternative hypothesis of equivalence within a pre-specified margin $\\Delta_{\\text{equiv}} > 0$:
$$H_0: \\mu_d \\le -\\Delta_{\\text{equiv}} \\quad \\text{or} \\quad \\mu_d \\ge +\\Delta_{\\text{equiv}}$$
$$H_1: -\\Delta_{\\text{equiv}} < \\mu_d < +\\Delta_{\\text{equiv}}$$

The pre-specified operational equivalence margin for clinical triage rule emulation was set at:
$$\\Delta_{\\text{equiv}} = \\pm 0.010\\text{ nats}$$
This represents approximately $0.9\\%$ of the baseline cross-entropy loss ($1.1157$ nats), below which changes in likelihood do not alter diagnostic Acuity deciles."""

new_sec2_1 = """### 2.1 Formal Hypothesis Definition and Entropy Reference Verification
Let $d_i = \\ell_{\\text{quantum}}(x_i, y_i) - \\ell_{\\text{classical}}(x_i, y_i)$ denote the paired difference in Negative Log-Likelihood (NLL) for observation $i$. Let $\\mu_d = \\mathbb{E}[d_i]$ denote the expected mean paired difference. Under our directionality convention:
- $\\mu_d > 0 \\implies$ Classical model achieves lower loss (Classical Superiority).
- $\\mu_d < 0 \\implies$ Quantum model achieves lower loss (Quantum Superiority).

To evaluate operational equivalence rather than mere absence of difference, Schuirmann's Two One-Sided Tests (TOST) procedure evaluates the composite null hypothesis of non-equivalence against the alternative hypothesis of equivalence within a pre-specified margin $\\Delta_{\\text{equiv}} > 0$:
$$H_0: \\mu_d \\le -\\Delta_{\\text{equiv}} \\quad \\text{or} \\quad \\mu_d \\ge +\\Delta_{\\text{equiv}}$$
$$H_1: -\\Delta_{\\text{equiv}} < \\mu_d < +\\Delta_{\\text{equiv}}$$

#### Pre-specified Operational Equivalence Margin & Entropy Scale Analysis:
The pre-specified operational equivalence margin for clinical triage rule emulation was set at:
$$\\Delta_{\\text{equiv}} = \\pm 0.010\\text{ nats} \\quad (\\pm 0.0144\\text{ bits})$$

To resolve reviewer queries regarding whether uniform entropy should be normalized to $\\ln(5)$ or $\\ln(6)$ under our 5-active-class / 6-output architecture, we audited all relevant information-theoretic reference baselines:
1. **Uninformative 6-way Output Reference ($H_{\\text{uniform}, 6} = \\ln 6$):** Because all evaluated neural architectures output $C=6$ logits and evaluate complete 6-way softmax probabilities, a maximally uninformative uniform prediction over all parameterized output categories produces cross-entropy loss $H_{\\text{uniform}, 6} = \\ln 6 \\approx 1.7918\\text{ nats}$ ($2.5850\\text{ bits}$). The $\\pm 0.010\\text{ nats}$ margin corresponds to **$0.56\\%$** of this uninformative reference.
2. **Oracle 5-class Uniform Reference ($H_{\\text{uniform}, 5} = \\ln 5$):** If an oracle restricts the uniform baseline strictly to the 5 observed positive diagnostic categories ($y \\in \\{0, 1, 2, 4, 5\\}$), the uniform prediction loss is $H_{\\text{uniform}, 5} = \\ln 5 \\approx 1.6094\\text{ nats}$ ($2.3219\\text{ bits}$). The $\\pm 0.010\\text{ nats}$ margin corresponds to **$0.62\\%$** of this oracle reference.
3. **Empirical Marginal Label Entropy ($H(Y)$):** Evaluating empirical test label proportions across the 5 observed classes yields $H(Y) \\approx 1.3699\\text{ nats}$ ($1.9764\\text{ bits}$). The $\\pm 0.010\\text{ nats}$ margin represents **$0.73\\%$** of the empirical label uncertainty.
4. **Baseline Model Cross-Entropy Loss:** Evaluated models achieve baseline cross-entropy loss of $1.1157\\text{ nats}$. The margin represents **$0.90\\%$** of the model cross-entropy loss.
5. **Predictive Mutual Information Gained:** The predictive mutual information explained by the neural representations over uninformative uniform 6-way prediction is $\\Delta I = \\ln 6 - 1.1157 \\approx 0.6761\\text{ nats}$; relative to oracle uniform prediction over 5 classes, $\\Delta I = \\ln 5 - 1.1157 \\approx 0.4937\\text{ nats}$; and relative to marginal label entropy, the model explains $H(Y) - 1.1157 \\approx 0.2542\\text{ nats}$. The $\\pm 0.010\\text{ nats}$ equivalence bound represents **$3.93\\% \\approx 4.0\\%$** of the total mutual information gained over the unconditioned marginal baseline.

> **Clinical Operational Scope:** The $\\pm 0.010\\text{ nats}$ threshold is an **operational statistical margin** for algorithmic rule emulation equivalence, established empirically because likelihood perturbations below this scale do not shift patients across diagnostic Acuity deciles or emergency triage bins. It is **not** a prospectively validated clinical outcome margin for patient morbidity or mortality."""

assert old_sec2_1 in d5_text, "old_sec2_1 not found in Deliverable 5"
d5_text = d5_text.replace(old_sec2_1, new_sec2_1)

# Now add sections for institutional transfer reconciliation and calibration partition limitations before Section 8
sec7_addition = """
---

## 7. Institutional-Transfer NLL Reconciliation

We conducted a forensic audit to reconcile the reported NLL metrics across different institutional transfer contexts, eliminating any potential conflation between intra-model hospital domain shift and inter-model paired comparison:

### 7.1 Intra-Model Domain Shift Degradation (MIMIC-CXR to CheXpert)
When models trained on MIMIC-CXR are transferred directly without re-calibration to external CheXpert radiographs, all architectures experience significant out-of-distribution domain degradation:
- **Tuned VQC:** $\\text{NLL}_{\\text{MIMIC}} = 1.1144 \\to \\text{NLL}_{\\text{CheXpert}} = 1.2342 \\implies \\mathbf{\\Delta\\text{NLL}_{\\text{shift}} = +0.1198\\text{ nats}}$
- **Product-of-Experts:** $\\text{NLL}_{\\text{MIMIC}} = 1.1170 \\to \\text{NLL}_{\\text{CheXpert}} = 1.2482 \\implies \\mathbf{\\Delta\\text{NLL}_{\\text{shift}} = +0.1312\\text{ nats}}$
- **Default VQC (C1):** $\\text{NLL}_{\\text{MIMIC}} = 1.1375 \\to \\text{NLL}_{\\text{CheXpert}} = 1.2769 \\implies \\mathbf{\\Delta\\text{NLL}_{\\text{shift}} = +0.1394\\text{ nats}}$
- **Order-1 Classical Fourier Surrogate:** $\\mathbf{\\Delta\\text{NLL}_{\\text{shift}} = +0.1434\\text{ nats}}$
- **Summary Range:** Across all evaluated architectures, external institutional domain transfer incurs an intra-model likelihood penalty of **$\\Delta\\text{NLL}_{\\text{shift}} = +0.1198\\text{ to }+0.1434\\text{ nats}$**.

### 7.2 Inter-Model Paired Performance Deficit on External Hospital ($n = 4,800$ Patients)
When comparing frozen model architectures head-to-head on the external hospital cohort:
- **Frozen VQC vs Classical PoE:** $\\mathbf{\\Delta\\text{NLL} = +0.1589\\text{ nats}}$ ($95\\%$ bootstrap CI $[+0.1422, +0.1757]$, $p < 10^{-15}$).
- **Frozen VQC vs Logistic Regression:** $\\mathbf{\\Delta\\text{NLL} = +0.1967\\text{ nats}}$ ($95\\%$ bootstrap CI $[+0.1785, +0.2149]$, $p < 10^{-15}$).
- **Equivalence Violation:** The frozen quantum model trails classical baselines by $+0.1589$ to $+0.1967$ nats on external transfer, which is 15- to 20-fold larger than the operational equivalence margin ($\\pm 0.010$ nats).

### 7.3 Methodological Separation
The manuscript text rigorously separates these two quantities using distinct notation: $\\Delta\\text{NLL}_{\\text{shift}}$ denotes single-model hospital generalization loss, whereas $\\Delta\\text{NLL}$ denotes paired head-to-head architectural deficit.

---

## 8. Calibration Partition & Clinical Prediction Boundaries

### 8.1 Shared 1,600-Patient Partition Limitation
The 1,600-patient development partition was utilized for both hyperparameter early-stopping selection and Platt temperature scaling parameter estimation. While computationally standard for moderate-resource benchmarks, reusing the same validation split can introduce slight optimistic calibration loss bias on validation curves. Crucially, the **56,534-patient confirmatory test cohort** remained completely held-out, untouched, and unexposed throughout training, grid selection, and temperature calibration, ensuring that all primary confirmatory NLL, ECE, and TOST findings are completely unbiased and leak-free.

### 8.2 Retrospective Predictive Emulation vs Prospective Health Outcomes
All evaluated metrics (NLL, ECE, AUROC, AUPRC) quantify retrospective mathematical emulation of clinical triage rules and label likelihoods. They do **not** evaluate prospective clinical deployment, diagnostic workflow throughput, clinician decision agreement, or downstream patient morbidity/mortality. Prospective clinical trials are strictly required before any algorithmic decision support system is deployed in patient care.
"""

# Replace Section 7 & Section 8 header
old_sec7 = """---

## 7. Summary of Statistical Tests and Units of Inference"""

new_sec7_full = sec7_addition + """
---

## 9. Summary of Statistical Tests and Units of Inference"""

d5_text = d5_text.replace(old_sec7, new_sec7_full)

# Replace Section 8 with Section 10
d5_text = d5_text.replace("## 8. Final Statistical Certification", "## 10. Final Statistical Certification")

d5_path.write_text(d5_text, encoding="utf-8")
shutil.copy2(d5_path, DOCS_DIR / "DELIVERABLE_05_STATISTICAL_VERIFICATION.md")
print("Updated DELIVERABLE_05_STATISTICAL_VERIFICATION.md successfully!")

# 2. Update Deliverable 11
d11_path = AUDIT_DIR / "DELIVERABLE_11_CHANGE_LOG.md"
d11_text = d11_path.read_text(encoding="utf-8")

phase16_text = """
### Phase 16: Final Targeted Audit & Consistency Hardening (Manuscript 6)
- **Action:** Executed final targeted audit addressing remaining P0, P1, and P2 review items.
- **P0.1 & P0.2 Uniform Entropy References & Active Class Resolution:**
  * Documented both $\\ln(6) \\approx 1.7918\\text{ nats}$ (uninformative 6-output baseline; margin is $0.56\\%$) and $\\ln(5) \\approx 1.6094\\text{ nats}$ (oracle 5-active-class baseline; margin is $0.62\\%$) in Methods, Results, and Supplementary Information.
  * Verified that predictive information gain over unconditioned marginal label entropy ($H(Y) \\approx 1.3699\\text{ nats}$) is $\\Delta I \\approx 0.2542\\text{ nats}$, with the $\\pm 0.010\\text{ nats}$ margin representing $3.93\\% \\approx 4.0\\%$ of this predictive signal.
- **P0.3 Reconciled Institutional Transfer NLL Ranges:**
  * Eliminated conflation of intra-model domain shift with inter-model paired comparison.
  * Standardized notation: $\\Delta\\text{NLL}_{\\text{shift}} = \\mathbf{+0.1198\\text{ to }+0.1434\\text{ nats}}$ across models (PoE $+0.1312$, VQC default $+0.1394$, VQC tuned $+0.1198$, order-1 Fourier $+0.1434$) for intra-model hospital domain shift from MIMIC-CXR to CheXpert.
  * Preserved paired head-to-head deficit on external CheXpert ($n=4,800$): frozen VQC trailed PoE by $\\mathbf{\\Delta\\text{NLL} = +0.1589\\text{ nats}}$ ($95\\%$ CI $[0.1422, 0.1757]$, $p < 10^{-15}$) and trailed Logistic Regression by $\\mathbf{+0.1967\\text{ nats}}$ ($95\\%$ CI $[0.1785, 0.2149]$, $p < 10^{-15}$).
- **P0.4 ECE Verification & Checkpoint vs Seed-Level Reporting:**
  * Confirmed single tuned checkpoint ECE of $0.0111$ ($95\\%$ CI $[0.0088, 0.0151]$) alongside 10-seed initialization distribution mean ECE of **$0.0252 \\pm 0.0038$** ($2.3\\times$ poorer calibration than PoE at $0.0109 \\pm 0.0011$).
- **P0.5 & P0.6 TOST Implementation and Confidence Intervals:**
  * Re-verified all paired difference standard errors, 90% equivalence CIs, and 95% superiority CIs across all cohorts ($N=447$, $N=1,600$, $N=4,800$, $N=56,534$, and 10 seeds).
- **P1 Clinical Interpretation & Limitations:**
  * Explicitly confirmed $\\pm 0.010\\text{ nats}$ is an operational statistical benchmark for rule emulation, not a prospectively validated clinical outcome margin.
  * Added limitation disclosing shared 1,600-patient partition for early stopping and temperature scaling, while confirming the 56,534-patient confirmatory cohort was strictly untouched.
  * Added prominent disclaimer distinguishing retrospective mathematical likelihood prediction from prospective patient health outcomes.
- **P1 Literature Audit & Checklist:**
  * Confirmed PubMed query, dates, 231 screened $\\to$ 129 health-related $\\to$ 41 claims $\\to$ 8 eligible $\\to$ 5 reconstructible (6 tasks).
  * Retained single-rater scoring as an explicit, disclosed limitation without fabricating synthetic second raters.
- **P1 Reproducibility Instructions:**
  * Formally distinguished single-command quick metric regeneration (`python reproduce_paper.py`, 7 seconds) from full end-to-end retraining (`--full`, multi-hour).
- **P2 npj Digital Medicine Editorial Compliance:**
  * Abstract word count maintained at **199 words** (strictly $\\le 200$ words).
  * Display items capped at **7 items** (4 tables + 3 figures $\\le 8$).
  * Compiled 18-page main text PDF and 17-page supplementary PDF with zero LaTeX errors.
"""

# Append before summary/verification
d11_text = d11_text.strip() + "\n" + phase16_text
d11_path.write_text(d11_text, encoding="utf-8")
shutil.copy2(d11_path, DOCS_DIR / "DELIVERABLE_11_CHANGE_LOG.md")
print("Updated DELIVERABLE_11_CHANGE_LOG.md successfully!")
