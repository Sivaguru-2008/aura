# PENDING_LIST: Outstanding Runs and Lookups for MANUSCRIPT_v5.1.md

Every `[PENDING]` marker in `MANUSCRIPT_v5.1.md` is cataloged below with the exact computation, run, or database lookup required to resolve it. No placeholder may be filled without a verified artifact on disk.

---

### 1. Qubit Distribution from PubMed Screening Log
- **Location:** Introduction (p. 2) and Section 3.2 (p. 11)
- **Placeholder:** `[PENDING: count of qualifying studies in our 231-record PubMed screen using 4–12 qubits, from screening log]`
- **Required Action:** Run `audit_screen_qubits.py` over the 231 screened records in `bench/field_audit/` to extract:
  1. Number of qualifying empirical studies with explicit qubit counts.
  2. Number and percentage of studies operating within the 4–12 qubit regime.
  3. Screening protocol reference.
- **Contingency Rule:** If this tally cannot be verified against the screening log, drop the "urgency" argument entirely and justify the 8–10 qubit regime on noise-free classical simulability alone.

---

### 2. Field Audit Screening Flow Funnel Counts
- **Location:** Section 2.1 (Field-wide audit, p. 3)
- **Placeholder:** `[PENDING: screening flow 231 -> eligible -> reproducible -> 5 with counts from the screening log]`
- **Required Action:** Extract exact breakdown counts from the screening log:
  1. Exclusions at Title/Abstract screening (conceptual proposals, non-clinical simulations, perspectives, non-health data).
  2. Exclusions at Full-Text screening (closed/proprietary data, missing source code/hyperparameters, under-specified circuits, non-deterministic execution).
  3. Yielding the 5 reproducible audited studies.

---

### 3. Pre-registered TOST Equivalence Test for Likelihood Parity
- **Location:** Section 2.3 (old §2.2, p. 5)
- **Placeholder:** `[PENDING: margin to be pre-registered in an addendum before execution, e.g. 0.01 NLL]: [PENDING result]`
- **Required Action:** 
  1. Deposit a pre-registration addendum (`PREREGISTRATION_ADDENDUM.md`) establishing the equivalence margin $\Delta_{\text{equiv}} = \pm 0.010$ NLL prior to analyzing test outcomes.
  2. Execute `tost_equivalence.py` on the paired test likelihood differences between Quantum VQC and Product-of-Experts from `fusion_matched_optimiser_predictions.npz`.
  3. Report two one-sided $t$-statistics, $p_{\text{TOST}}$, and the 90% confidence interval.
- **Contingency Rule:** Until the pre-registration and test results are reported, maintain that no difference was detected; do not claim parity.

---

### 4. Patient-Level Clustered Bootstrap on Test Studies
- **Location:** Section 2.5 (old §2.4, p. 7)
- **Placeholder:** `[PENDING: patient-level bootstrap point delta, SE, and 95% CI]`
- **Required Action:** 
  1. Execute `ablation_patient_bootstrap.py` using clustered resampling over unique patient identifiers across the 447 MIMIC-CXR test studies (1,000 resamples).
  2. Compute bootstrap standard error and 95% percentile confidence interval for $\Delta\text{NLL} = \text{NLL}_{\text{entangled}} - \text{NLL}_{\text{product}}$.

---

### 5. Repeated Patient-Disjoint Re-splits for the Ablation
- **Location:** Section 2.5 (old §2.4, p. 7)
- **Placeholder:** `[PENDING: K] repeated patient-disjoint re-splits, retraining each: [PENDING: mean delta across splits, between-split variance]`
- **Required Action:**
  1. Perform $K = 5$ repeated patient-disjoint stratified splits (70/10/20 train/cal/test).
  2. Retrain both entangled and unentangled backends under the matched Adam protocol on each split.
  3. Calculate empirical between-split variance ($\sigma^2_{\text{data}}$) alongside optimizer initialization variance ($\sigma^2_{\text{init}}$).

---

### 6. Matched-Parameter Data Re-uploading Circuit Ablation
- **Location:** Section 2.5 (old §2.4, p. 7)
- **Placeholder:** `[PENDING: L] layers, parameter count matched between entangled and product arms, ten seeds: [PENDING: re-uploading entangled vs product NLL and delta]`
- **Required Action:**
  1. Execute `reuploading_ablation.py` at 8 qubits (7 informative channels + 1 reference) across $L=3$ layers with alternating data re-uploading blocks.
  2. Strictly match parameter count between entangled and product arms (72 trainable parameters each).
  3. Train across 10 random seeds under the matched Adam protocol.
  4. Report paired $\Delta\text{NLL}$, confidence interval, and Holm $p$-value.

---

### 7. Input-Provenance Cryptographic Assertion Protocol (Defect D14 Resolution)
- **Location:** Section 3.3 (Governance and verification limits, p. 11)
- **Placeholder:** `[PENDING: input-provenance assertion to be implemented and committed before evaluation]`
- **Required Action:**
  1. Commit `provenance_assertion.py` to the repository.
  2. Integrate SHA-256 byte-digest assertions over raw evidence tensors into the CI test suite to guarantee that models are never evaluated against synthetic test stubs without triggering an automated build failure.
