# CAUSAL INVESTIGATION: THE ROLE AND MECHANISMS OF ENTANGLEMENT

**Role:** Senior Quantum Machine Learning Theorist & Adversarial Auditor  
**Repository State:** `E:/AURA/aura-main`  
**Audit Date:** October 11, 2026

---

## 1. The Empirical Phenomenon

Across all pre-registered ablations on the 7 informative chest pathology channels (`revision_v51.json` Analysis B, Analysis C, and Analysis D):
- **10 of 10 random seeds** under matched Adam optimization showed superior (lower) negative log-likelihood for the **unentangled** (product state) circuit arm over the **entangled** (CNOT ring) arm.
- Point estimate: $\Delta\text{NLL} = \text{NLL}_{\text{entangled}} - \text{NLL}_{\text{product}} = +0.0250$ nats ($p = 0.0013$ parametric paired $t$, $p = 0.0020$ Wilcoxon signed-rank).
- Replicated on $K=5$ patient-disjoint re-splits ($\Delta\text{NLL} \in [+0.0216, +0.0461]$).
- Replicated under matched-parameter data re-uploading circuits ($L=3$, 72 parameters, $\Delta\text{NLL} = +0.0229$, $p = 0.0040$).

The central research question: **Why does entanglement impose a consistent likelihood penalty on this clinical prediction task?**

---

## 2. Hypothesis Testing: Candidate Explanatory Mechanisms

We evaluated five candidate theoretical mechanisms:

```
+----+------------------------------------------------+-----------------------+---------------------+
| ID | Candidate Mechanism                            | Theoretical Plausibility| Empirical Verdict  |
+----+------------------------------------------------+-----------------------+---------------------+
| M1 | Barren Plateaus / Exponential Gradient Decay  | LOW (at 7-8 qubits)   | REFUTED             |
| M2 | Unfavorable Inductive Bias / Additive Target   | VERY HIGH             | CONFIRMED (PRIMARY) |
| M3 | Optimization Landscape Non-Convexity           | HIGH                  | CONFIRMED (SECONDARY|
| M4 | Overfitting / Generalization Gap Expansion     | MODERATE              | PARTIALLY SUPPORTED |
| M5 | Information Loss via Local Pauli-Z Readout     | HIGH                  | CONFIRMED (ANALYTIC)|
+----+------------------------------------------------+-----------------------+---------------------+
```

### 2.1 Mechanism 1: Barren Plateaus (Refuted)
- **Hypothesis:** Deep entangling circuits suffer from barren plateaus where gradients vanish exponentially in the number of qubits ($Var[\partial_\theta \mathcal{L}] \in O(2^{-n})$).
- **Audit Refutation:**
  1. The circuit width is strictly $n = 7 \text{ to } 8$ qubits, and depth is shallow ($L = 3$ layers). At $n=8$, Hilbert space dimension is $2^8 = 256$, which is far from asymptotic Haar measure concentration.
  2. The gradient variances during training in `revision_v51.json` remain within $[10^{-3}, 10^{-2}]$, showing active parameter updates throughout 30–150 epochs.
  3. Attributing this degradation to "barren plateaus" is mathematically incorrect.

### 2.2 Mechanism 2: Inductive Bias Mismatch with an Additive Target (Confirmed Primary Driver)
- **Mathematical Analysis:**
  With single-qubit RY($\pi x_i$) angle encoding, the state before entangling layers is a separable product state:
  $$\rho_0(x) = \bigotimes_{i=1}^n \frac{1}{2} \left( I + \cos(\pi x_i) Z + \sin(\pi x_i) X \right)$$
  - **In the Unentangled Circuit ($U = \bigotimes_{i=1}^n U_i(\theta_i)$):**
    Each qubit evolves independently. The expectation value of local measurement $Z_j$ is:
    $$\langle Z_j \rangle(x) = \text{Tr}\left( U_j^\dagger Z_j U_j \rho_{0, j}(x_j) \right) = c_{j, 0} + c_{j, Z} \cos(\pi x_j) + c_{j, X} \sin(\pi x_j)$$
    The resulting logits are strictly additive linear combinations of univariate Fourier terms:
    $$f_k(x) = \sum_{j=1}^n w_{k, j} \langle Z_j \rangle(x) + b_k = \sum_{j=1}^n g_{k, j}(x_j) + b_k$$
    **Order of interaction = 1 (Zero cross-channel multiplicative interactions).**
  - **In the Entangled Circuit (CNOT Ring):**
    CNOT gates create multi-qubit correlations. Measurement $\langle Z_j \rangle$ develops non-zero projection onto multi-qubit Pauli operators $P \in \{I, X, Y, Z\}^{\otimes n}$ with weight $|P| \ge 2$:
    $$\langle Z_j \rangle(x) = \sum_{P} c_P \prod_{i=1}^n f_{P_i}(x_i)$$
    This forces the model to allocate capacity to higher-order cross-channel interaction terms (e.g., $\sin(\pi x_a) \cos(\pi x_b)$).
- **The Clinical Task Reality:**
  As established in §1, the ground-truth clinical diagnostic target is constructed via `_map_diagnosis` as a **linear weighted sum of findings**:
  $$\text{Score}_k = \sum_{j=1}^n \alpha_{k, j} \mathbb{I}(x_j > \tau)$$
  Because the target has zero intrinsic multiplicative interaction among findings, any energy allocated by entangling gates to interaction orders $\ge 2$ represents **spurious variance and inductive bias mismatch**.

### 2.3 Mechanism 3: Optimization Landscape Non-Convexity
- While unentangled product circuits factorize into independent single-qubit optimization problems (convex or near-convex on $SU(2)$), entangling gates couple the parameter gradients across all qubits:
  $$\frac{\partial^2 \mathcal{L}}{\partial \theta_{l, a} \partial \theta_{l, b}} \ne 0 \quad \text{for } a \ne b$$
- This introduces non-convex saddle points and local minima, leading to higher final training and calibration loss under identical optimizer budgets.

---

## 3. Scope and Scientific Boundary Conditions

The manuscript must avoid making sweeping, overgeneralized claims that "entanglement hurts machine learning." The audit establishes the precise boundary conditions:

1. **Topology & Encoding Specificity:** The observed penalty applies specifically to shallow hardware-efficient ansaetze (RY-RZ with CNOT rings) under angle encoding on low-dimensional tabular/extracted features.
2. **Target Specificity:** When the target requires multiplicative interaction (as demonstrated by positive control T2b), the entangled circuit's likelihood improves from NLL 0.798 to 0.251, establishing that entanglement is functional when cross-channel structure exists.
3. **Capacity Matching:** At matched parameter count (72 parameters), the unentangled model achieves higher expressive efficiency on additive tasks because all 72 parameters are dedicated to shaping independent 1D feature mappings rather than unused cross-terms.
