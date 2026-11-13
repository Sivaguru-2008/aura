# ADVERSARIAL PEER REVIEW: THREE INDEPENDENT EXPERT EVALUATIONS

**Simulated Review Context:** Target Journal: *npj Digital Medicine* / *Nature Machine Intelligence*  
**Manuscript Title:** *"Auditing Quantum Machine Learning in Digital Health: One Deployed System and Five Published Claims Under Symmetric Evaluation"*  
**Audit Date:** October 11, 2026

---

## Reviewer 1: Quantum Machine Learning & Theoretical Physics

### Overall Assessment
This is a timely, technically rigorous, and sobering audit of current claims in healthcare QML. The authors examine small variational circuits ($n = 4 \text{ to } 10$ qubits) and show that empirical claims of superiority fail when baselines receive standard optimization and calibration. The theoretical analysis showing that the fidelity kernel reduces to a product of cosines is clean. However, the manuscript must be careful not to conflate the limitations of shallow angle-encoded VQCs with the fundamental capabilities of quantum computation.

### Three Strongest Contributions
1. **Analytic Kernel Reduction:** Proving that the RY($\pi x$) fidelity kernel analytically matches a classical closed form ($O(n)$ classical time) eliminates any illusion of asymptotic quantum advantage for this encoding.
2. **Entanglement Ablation:** Rigorously demonstrating that entanglement imposes an inductive bias penalty on additive clinical tasks across 10 seeds and parameter-matched data re-uploading circuits.
3. **Hardware-Simulated Verification:** Pinned verification confirming that the statevector simulation matches PennyLane to within machine precision ($10^{-15}$).

### Five Major Concerns
1. **Over-generalizing the Entanglement Penalty:** The text risks implying that entanglement is universally detrimental in machine learning. It must be explicitly stated that this penalty arises from inductive bias mismatch with additive targets.
2. **Positive Control T2b Framing:** On quadratic target T2b, the authors claim the circuit achieved an advantage. But the classical Gated Head achieved virtually identical likelihood. This is NOT a quantum advantage; it is standard nonlinear expressivity.
3. **Simulated vs Physical Hardware:** The paper previously retracted a "shot-budget" mechanism. The text must explicitly reiterate that all primary results reflect exact mathematical statevector simulations, not noisy intermediate-scale quantum (NISQ) execution.
4. **Scope Limitation:** The study examines low-depth ($L \le 3$) VQCs and fidelity kernels on $\le 10$ qubits. It does not rule out quantum advantages in fault-tolerant regimes, large-scale kernel methods with provable quantum speedup (e.g. discrete log / group-theoretic kernels), or molecular quantum biology.
5. **Barren Plateau Clarification:** Ensure that the manuscript never attributes poor convergence at 8 qubits to barren plateaus; at $n=8, L=3$, gradient variance does not exponentially concentrate.

### Required Revisions & Claims to Weaken
- **Must Weaken:** Reframe the positive control on T2b: remove any phrase asserting "an advantage for the quantum circuit itself" and state that the circuit matched the classical gated head.
- **Must Clarify:** Explicitly state in the Abstract and Discussion that results apply to $\le 10$ qubits, exact simulation, and low-depth hardware-efficient ansaetze.

---

## Reviewer 2: Statistical Machine Learning & Benchmarking

### Overall Assessment
A devastating and well-executed methodological critique. The factorial decomposition of published quantum claims (especially Parkinson's voice and EEG schizophrenia) exposes elementary benchmarking errors in high-profile QML papers. The statistical apparatus (Nadeau-Bengio correction, clustered patient bootstrap, pre-registered TOST) is top-tier.

### Three Strongest Contributions
1. **Factorial Dissection of Literature Claims:** F1 and F3 prove that pre-split leakage and subject-crossing partitions causally manufactured reported 99–100% claims.
2. **Pre-registration Integrity:** Committing hypotheses and analysis scripts with cryptographic git hashes prior to scoring held-out cohorts.
3. **Asymmetric Calibration Exposure:** Demonstrating that 75% of the published ECE gap in AURA was driven solely by temperature scaling only the quantum head.

### Five Major Concerns
1. **Equivalence Overreach on Small Cohorts:** Early drafts claimed "statistical equivalence" or "parity" on $n=447$ MIMIC studies. The TOST 90% CI on $n=447$ clearly crosses $\pm 0.010$ ($p=0.359$). You cannot assert equivalence on an underpowered sample.
2. **Reverse Asymmetry in Field Audit:** In re-testing published studies, classical models were selected across 16 configurations via nested CV, while the quantum model used the author's published architecture. Could this introduce reverse bias against the quantum model?
3. **Multiple Testing Correction:** Ensure all pairwise contrasts across backends are strictly adjusted via Holm-Bonferroni across the entire family of 6 contrasts.
4. **Causal Attribution in Transfer:** The claim that higher parameter count caused the quantum circuit to degrade on CheXpert transfer is an unverified causal claim based on only four data points.
5. **Hierarchical Seed $\times$ Patient Uncertainty:** The entanglement penalty is significant at the seed level ($p=0.0013$) and study level ($p=0.036$), but marginally non-significant under simultaneous seed $\times$ study hierarchical bootstrap ($p=0.062$). This nuance must be transparently disclosed.

### Required Revisions & Claims to Weaken
- **Must Correct:** Retain strict language regarding $n=447$: "no difference was detected; equivalence was not established." Equivalence assertions must rely solely on the high-$n$ confirmatory cohort.
- **Must Qualify:** Disclose that external degradation on CheXpert is descriptive, not a proven causal consequence of parameter count.

---

## Reviewer 3: Medical AI & Clinical Evaluation

### Overall Assessment
This paper provides an essential service to the digital health community. Clinical AI is currently flooded with hype surrounding quantum computing. By demonstrating that published health QML claims do not survive basic clinical validation standards, this work will serve as an authoritative reference for journal editors and regulatory evaluators. However, the manuscript's description of its own clinical pipeline must be held to the same rigorous standard it demands of others.

### Three Strongest Contributions
1. **Clinical Realism & External Validation:** Testing models across two major institutional imaging datasets (MIMIC-CXR and CheXpert).
2. **Defect Ledger & Scientific Candor:** The willingness to audit and retract the authors' own prior claims sets a benchmark for scientific integrity.
3. **Actionable Checklist:** The 17-item evaluation checklist derived from TRIPOD+AI provides immediate practical utility for clinical peer reviewers.

### Five Major Concerns
1. **Rule Emulation vs Clinical Ground Truth:** The six-class diagnostic target is constructed via `_map_diagnosis` as an additive linear formula of radiographic findings. The fusion layer is emulating a deterministic triage heuristic, NOT clinical diagnosis. This MUST be stated plainly.
2. **"Deployed System" Language:** Describing AURA as a "deployed clinical decision support system" is misleading if it was only served locally via Docker and never integrated into real-world patient care.
3. **CheXpert Label Incompatibility:** COPD is completely unreachable in CheXpert because hyperinflation is not labeled. The impact of this structural label truncation on model ranking must be clearly discussed.
4. **Clinical Utility Metrics:** In medical decision support, Net Benefit (Decision Curve Analysis) and calibration in high-risk probability regions matter more than global NLL.
5. **Data Licensing Transparency:** Reiterate that all models operate strictly under research-only agreements and cannot be used commercially without prospective radiologist re-annotation.

### Required Revisions & Claims to Weaken
- **Must Reframe:** Replace "deployed clinical decision-support system" with "decision-support system prototype" or "research decision-support platform."
- **Must Clarify:** Explicitly explain in the Methods that the multi-class fusion target is an algorithmic rule derived from findings, which naturally favors linear models.
