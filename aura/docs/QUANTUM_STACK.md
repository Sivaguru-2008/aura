# AURA — Quantum Clinical Reasoning Stack (QCRS)
### Quantum-native redesign of the clinical reasoning pipeline

> Companion to `ARCHITECTURE.md` (v1 system) and `PRODUCT_V2.md` (Diagnostic Trajectory OS).
> This document specifies **where quantum computation is conceptually meaningful across the
> reasoning pipeline, as a set of independent services** — and, with equal precision, where
> it is not.
>
> Everything here extends the existing code: the 8-channel evidence encoder
> (`services/fusion/evidence.py`), the VQC and backend abstraction
> (`services/fusion/{quantum,device}.py`), and the head-to-head benchmark harness
> (`ml/evaluation/benchmark.py`).

---

## 0. The Honesty Contract

Quantum claims in healthcare AI are where credibility goes to die. Every component in this
stack therefore carries one of three claim tiers, and the tier appears in its spec header:

| Tier | Meaning | Rule |
|---|---|---|
| **[A] Measured** | Runs today on a simulator, with a classical twin evaluated head-to-head by the benchmark harness on the same held-out data | We show the table, whatever it says |
| **[B] Grounded** | Implemented on a simulator today; the quantum claim is **representational or structural** (a provable memory advantage, a formalism that fits the problem's structure), backed by published literature — never a speed claim | Advantage asserted only in the dimension the literature supports; benchmarked anyway |
| **[C] Vision** | Requires fault-tolerant hardware; a provable speedup exists in theory (amplitude estimation, Grover, quantum walks) with known fine print | Clearly labeled future work; the product must never depend on it |

Three standing laws, inherited from v1 and now stack-wide:

1. **The Benchmark Law.** Every quantum service ships inside the same process as its
   classical fallback, behind a config flag, and `aura_cli bench` compares them on
   accuracy, NLL, ECE, Brier, conformal coverage, and set size. If the classical twin
   wins, the classical twin serves — and we say so on stage. (If a classical surrogate
   can dequantize one of our circuits, that is a *result*, not an embarrassment: we ship
   the surrogate.)
2. **The Quantum-Free Zones.** No quantum — no ML of any kind — in the deterministic
   defensibility path: the **guideline engine, the commitment ledger, report grounding,
   and the audit chain** are exact, versioned, and reproducible by construction. And
   perception stays classical: CNNs won imaging; the v1 decision stands.
3. **The Provenance Law.** Every quantum result is stored with `{backend, device,
   feature_map_version, circuit_version, seed, shots, posterior_std}` pinned into the
   belief snapshot — any historical output is exactly reproducible (extends the v1 audit
   posture).

**Why this problem is quantum-*compatible* at all.** The clinical reasoning core operates
on *compressed, structured, correlation-rich, low-dimensional* state: 8–16 evidence
channels, 6–32 hypotheses, 5–50 candidate actions, 2–5 progression states. That is
exactly the regime today's simulators (and tomorrow's early fault-tolerant machines)
can address — and it is the deliberate v1 design decision ("the compression is the whole
reason quantum is tractable here", `evidence.py`). We never put pixels in a circuit.

---

## 1. Stack overview

```
                     CLASSICAL PIPELINE (v1/v2 — unchanged)
  image → findings → evidence graph → differential → verification → guideline
        → uncertainty → risk → next-best-action → monitoring → report → feedback
              │                                        ▲
              │ evidence vectors, sequences, queries    │ posteriors, rankings,
              ▼                                        │ samples, similarities
 ═══════════════════ QUANTUM CLINICAL REASONING STACK ═══════════════════════
              │
  ┌───────────┴──────────────────────────────────────────────────────────┐
  │  Q0 QUANTUM FEATURE-MAP REGISTRY          (shared representation)    │
  │  trained, versioned evidence→state encodings; one clinical Hilbert   │
  │  space shared by every service below                                 │
  └──┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┘
     ▼          ▼          ▼          ▼          ▼          ▼
  ┌───────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
  │  Q1   │ │  Q2    │ │  Q3     │ │  Q4    │ │  Q5     │ │  Q6      │
  │Evidence│ │Belief  │ │Similarity│ │Diagnostic│ │Trajectory│ │Uncertainty│
  │Fusion │ │Engine  │ │& Cohort │ │Planner │ │Engine   │ │Service   │
  │ VQC   │ │density │ │fidelity │ │ QUBO/  │ │quantum  │ │shot stats│
  │ [A]   │ │matrix +│ │kernels  │ │ QAOA   │ │stochastic│ │+ Born    │
  │exists │ │Kraus   │ │ [A/B]   │ │ [A]    │ │models   │ │sampling  │
  │       │ │ [B]    │ │         │ │        │ │ [B]     │ │ [A/B]    │
  └───────┘ └────────┘ └─────────┘ └────────┘ └─────────┘ └──────────┘
     all behind services/fusion/device.py (default.qubit · default.mixed ·
     lightning · hardware adapters later) — async, cached, never in the
     interactive read path
```

### Pipeline-stage mapping (which stage gets quantum, and which does not)

| Reasoning stage (PRODUCT_V2 §5) | Quantum service | Why / why not |
|---|---|---|
| 1 Image → findings | — | Perception is classical, permanently |
| 2 Findings → evidence | Q0 (encoding) | The evidence vector *is* the circuit input |
| 3 Evidence graph | — (graph is classical) | Storage/relations are classical; Q3 consumes it |
| 4 Differential diagnosis | **Q1** | Higher-order evidence interactions |
| 5 Evidence verification | **Q2** | Coherences expose correlated ambiguity; order-sensitivity diagnostic |
| 6 Guideline matching | — **banned** | Deterministic defensibility zone |
| 7 Uncertainty analysis | **Q6** (+ safety engine) | Shot statistics are native uncertainty |
| 8 Risk prediction | **Q5 + Q6** | Progression models + scenario sampling |
| 9 Next best action | **Q4** | Test-panel selection is natively QUBO |
| 10 Treatment planning support | — | Guideline-anchored, deterministic |
| 11 Longitudinal monitoring | **Q2 + Q5** | Belief updates over time; progression |
| 12 Report | — **banned** | Grounded composition, exact |
| 13 Feedback learning | Q0 (retraining) | Feature maps retrained on feedback |
| 14 Fleet aggregation | — | Analytics, classical |

---

## 2. Q0 — Quantum Feature-Map Registry `[A]`
*(the honest reframing of "Quantum Knowledge Embedding")*

- **Why this layer exists.** Every service below needs to encode clinical evidence into a
  quantum state. If each invents its own encoding, their outputs live in incomparable
  spaces and nothing composes: fusion posteriors, kernel similarities, and belief states
  must share one representation to be mutually consistent. Q0 makes "the quantum
  representation of clinical evidence" a single, trained, versioned, signed asset — the
  registry every circuit imports. Theoretical basis: parameterized quantum models are
  formally kernel methods determined by their feature map (Schuld 2021), so *the feature
  map is the model* — it deserves first-class governance. Data re-uploading maps
  (Pérez-Salinas et al. 2020) make few-qubit encodings universal function approximators.
- **Approach.** Quantum embedding: today the v1 product encoding `RY(π·xᵢ)` per channel
  (`device.py`); adds trainable re-uploading layers whose parameters are fit by
  **kernel-target alignment** (Hubregtsen et al. 2022) against feedback labels — the v1
  learning loop now retrains the *representation*, not just calibration.
- **Inputs → outputs.** In: evidence vectors (8 channels today; delta/velocity/staleness
  channels from v2 belief service extend to ~12–16). Out: versioned encoder artifacts
  (circuit template + trained angles) consumed by Q1–Q6; alignment metrics per version.
- **Classical integration.** Registered in the existing `models` service alongside
  classical model versions; every belief snapshot pins `feature_map_version` (Provenance
  Law).
- **Hackathon demo.** Show two registry versions — untrained vs feedback-aligned — and the
  retrieval precision jump in Q3's benchmark. One chart, honest.
- **FTQC outlook.** Larger maps (50–200 channels for multimodal evidence: labs, genomics,
  note embeddings) evaluated faithfully — today's 8–16 channel ceiling is a simulator
  artifact, not a design one.
- **Status & benchmark.** The static map runs today (it is v1's encoder); trained
  re-uploading is a small addition. Benchmark: kernel-target alignment score + downstream
  Q1/Q3 metrics per registry version.

---

## 3. Q1 — Quantum Evidence Fusion `[A — exists, measured]`

- **Why quantum at this stage.** Diagnosis is not a weighted sum of evidence. *Opacity +
  effusion + fever-prior* jointly shift the posterior differently than any pairwise
  combination — clinical evidence interacts at high order. The entangling ansatz (CNOT
  ring, `device.py`) represents order-k interactions among all 8 channels with
  **O(n·layers) parameters in a 2⁸-dimensional space**, where an explicit classical
  interaction model needs exponentially many terms or a feature-engineered approximation.
  This is a compactness/inductive-bias argument, not a speed claim — and it is testable,
  so we test it.
- **Approach.** Variational quantum circuit — correct choice because we need a *trained,
  differentiable* map from evidence to posterior (parameter-shift gradients, hybrid
  training with the classical linear head, `quantum.py`). A kernel would give similarity,
  not a posterior; optimization doesn't apply.
- **Inputs → outputs.** In: 8-channel evidence vector in [0,1] (`evidence.py`). Out:
  calibrated joint posterior over 6 diagnoses + per-diagnosis `posterior_std` from
  finite-shot variance `(1−⟨Z⟩²)/n_shots` Monte-Carlo-propagated through the head
  (`quantum.py` lines 53–63).
- **Classical integration.** Drop-in peer of `ClassicalFusion` (Bayesian
  product-of-experts) behind one engine interface; the safety engine consumes either's
  logits identically; config flag switches backends.
- **Hackathon demo.** Already live: `aura_cli bench` prints accuracy / NLL / ECE / Brier /
  conformal coverage / set size, quantum vs classical, held-out. We show the table
  *as measured* — near-parity at 8 features is the honest expected result, and saying so
  is the credibility move.
- **FTQC outlook.** More channels without parameter explosion (multimodal evidence) and
  deeper ansätze evaluated noiselessly. Countervailing honesty: barren plateaus (McClean
  et al. 2018) and classical surrogates may cap useful depth — which is why the Benchmark
  Law exists.
- **Status.** Production path of the demo today, analytic simulator serving, shots
  reported as uncertainty.

---

## 4. Q2 — Quantum Belief Engine `[B — grounded; the flagship]`
*(the quantum-native form of v2's `belief` service)*

- **Why quantum at this stage.** A probability vector is a lossy belief state. When a
  chest film is compatible with *pneumonia-or-heart-failure jointly* — the evidence
  supports the pair without discriminating within it — a vector `{pna: .4, chf: .4}`
  cannot distinguish that **correlated ambiguity** from two independently uncertain
  hypotheses. A **density matrix ρ can**: diagonal = the classical posterior;
  off-diagonal coherences = which hypothesis *pairs* remain entangled by the current
  evidence. Sequential evidence arrival becomes a **quantum channel update** (Kraus
  operators), which is the mathematically native formalism for "measurement updates a
  mixed state." Two clinically meaningful quantities fall out for free:
  1. **Ambiguity decomposition**: von Neumann entropy S(ρ) ≤ Shannon entropy of the
     diagonal H(diag ρ); the gap **H − S measures structured, pairwise ambiguity** —
     "the uncertainty is specifically pneumonia↔CHF, resolve it with BNP" — feeding Q4
     a *target*, not just a magnitude.
  2. **Order-sensitivity diagnostic**: non-commuting evidence updates make belief
     order-dependent — exactly the anchoring/order effects documented in human clinical
     judgment (quantum-cognition literature: Busemeyer & Bruza 2012; Pothos & Busemeyer).
     Classical Bayes is order-invariant; clinicians are not. Q2 replays the evidence
     stream in multiple orders and reports the posterior spread as an **anchoring-risk
     flag** on the case: "this workup's conclusion is fragile to the order in which the
     evidence was seen."
  The claim tier is deliberately [B]: this is a **representational fit** argument backed
  by a real literature — never a speedup claim.
- **Approach.** Quantum embedding + parameterized quantum channels: hypotheses as basis
  states of ⌈log₂ 6⌉ = 3 qubits (8-dim space, 6 used); ρ is 8×8 on `default.mixed`;
  per-evidence-channel Kraus operators realized as parameterized unitaries on
  system+ancilla with ancilla trace-out, trained by maximum likelihood on longitudinal
  sequences + calibration penalty. VQC is the parameterization; the *concept* is the
  channel formalism.
- **Inputs → outputs.** In: prior ρ (site prevalence on the diagonal), evidence events in
  arrival order (Q0-encoded). Out: `BeliefState` (v2 contract) extended with
  `{coherence_matrix |ρᵢⱼ|, von_neumann_entropy, diag_entropy, ambiguity_gap,
  order_sensitivity, purity}`.
- **Classical integration.** Implements the v2 `belief` service interface; the classical
  recursive Bayes filter (fusion-as-update-operator) is the first-class fallback and the
  baseline in every benchmark; the premature-closure and conflict flags of v2 consume
  Q2's outputs (coverage vs. confidence; per-source KL).
- **Hackathon demo.** The **coherence heatmap**: a 6×6 hypothesis-pair grid that lights up
  where ambiguity is concentrated, updating live as evidence arrives in the case cockpit —
  then the *order-replay* toggle showing the same evidence in two orders yielding
  different sequential beliefs, flagged. Runs comfortably on `default.mixed` (3–4 qubits).
  Benchmark vs classical filter: end-of-sequence accuracy, log-loss, ECE on synthetic
  longitudinal cases.
- **FTQC outlook.** Hypothesis lattices at SNOMED scale (thousands of basis states,
  10–12 system qubits + ancillae) with amplitude estimation for marginals — today's 3-qubit
  hypothesis space is a demo constraint, not a formalism constraint. Speculative; labeled
  [C] in the roadmap table.
- **Status.** New service; smallest circuits in the stack; the most novel honest claim.

---

## 5. Q3 — Quantum Similarity & Cohort Retrieval `[A/B]`
*(merges "Quantum Similarity Search", "Quantum Cohort Retrieval", and the retrieval half
of "Quantum Clinical Memory" — one representation, three query shapes)*

- **Why quantum at this stage.** "Who else looked like this, and how did it end?" matters
  most exactly where classical ML is weakest: **rare presentations, tiny n** — kernel
  regime, not deep-net regime. Fidelity kernels `k(x,x′) = |⟨φ(x)|φ(x′)⟩|²` give a
  genuinely different geometry from RBF/cosine — similarity in the *entangled interaction
  space* of Q0's feature map, so two cases match on the joint pattern
  (nodule × smoker × prior-growth), not channel-by-channel. Literature honesty: feature
  maps exist whose kernels are believed classically hard (Havlíček et al., Nature 2019),
  and a rigorous quantum kernel advantage is *proven* — for a contrived discrete-log
  problem (Liu, Arunachalam & Temme, Nat. Phys. 2021). **No one has shown advantage on
  clinical data; we claim only a different, testable inductive bias.**
- **Approach.** Quantum kernel (correct tool: we need a Gram matrix, not a trained
  posterior). Computed today as exact statevector overlaps on the simulator; SWAP-test /
  compute-uncompute on hardware later. Trajectory variant: the same kernel over v2
  trajectory summary vectors (deltas, velocities, staleness). Cohort mode (P1): select k
  patients maximizing match+diversity — a small QUBO sharing Q4's solver.
- **Inputs → outputs.** In: query evidence/trajectory vector + candidate set. Out: ranked
  neighbors with kernel scores + outcome labels where known (feeds the v2 "similar
  trajectories" panel), with similarity provenance.
- **Classical integration.** **Two-stage retrieval**: the existing memory engine's cosine
  ANN recalls top-50 (fast, scalable); the quantum kernel re-ranks the shortlist —
  O(50) circuit evaluations per query, honest about where each tool is strong. Gram rows
  cached; index refreshed nightly.
- **Hackathon demo.** Retrieval bench on seeded cases: precision@k for cosine vs RBF vs
  fidelity kernel, one table (Benchmark Law). Plus the cockpit panel: "3 similar
  trajectories, 2 resolved benign, 1 malignant at 9 months."
- **FTQC outlook.** Amplitude-amplified search gives provable O(√N) over unstructured
  candidate sets `[C]` — with the known fine print: it presumes efficient state access
  (QRAM), and quadratic speedups are easily eaten by error-correction overheads (Babbush
  et al., PRX Quantum 2021). The nearer-term real gain: bigger trained feature maps
  evaluated faithfully.
- **Risk we monitor.** Exponential kernel concentration at high qubit counts (Thanasilp
  et al.) — at 8–16 qubits with trained maps we are in the workable regime; the benchmark
  catches degradation.

---

## 6. Q4 — Quantum Diagnostic Planner `[A — verifiable at demo scale]`

- **Why quantum at this stage.** v1's recommender ranks *single* next tests by expected
  information gain. Real workups order **panels**: sets of tests under budget and time
  constraints, where tests interact — CT and lateral view partially *resolve the same
  channels* (redundant); BNP resolves what imaging cannot (complementary). Greedy
  EIG ranking cannot see redundancy; it will happily recommend two tests that answer the
  same question. Panel selection with pairwise interactions is natively **QUBO**:
  maximize `Σ EIG_a·z_a − λ·Σ R_ab·z_a·z_b − μ·Σ cost_a·z_a` (+ budget penalty),
  `z ∈ {0,1}ⁿ` — the canonical near-term quantum optimization target.
- **Approach.** Quantum optimization — QAOA (Farhi et al. 2014), depth p=1–3, one qubit
  per candidate action, on `default.qubit`; the redundancy matrix `R_ab` is computed
  exactly through the existing fusion posterior (joint-EIG of pairs = 4 outcome
  evaluations via `recommend/engine.py` machinery).
- **Inputs → outputs.** In: current belief, candidate catalog (5 actions today), budget.
  Out: recommended *panel* + expected joint information gain + the redundancy it avoided
  (rendered as rationale: "CT chosen over CT+lateral: 82% of the lateral's information is
  contained in the CT").
- **Classical integration.** Three solvers in one service, benchmark-lawed: exhaustive
  enumeration (2⁵ = 32 subsets today — the *provably optimal* reference), simulated
  annealing (the classical heuristic), QAOA. Greedy v1 EIG remains as the sanity floor.
- **Hackathon demo.** The most honest optimization demo possible: construct the case where
  greedy picks the redundant pair and the QUBO solvers find the complementary panel —
  then show **QAOA matching the brute-force verified optimum**. No advantage claim;
  a correctness demonstration on a problem whose *formulation* is the contribution.
- **FTQC outlook.** At 30–50 candidate actions (multi-modality panels, trial-matching
  constraints) enumeration dies (2³⁰⁺) and classical heuristics compete without
  certificates; error-corrected QAOA / Grover-enhanced branch-and-bound is the target
  regime `[C]`. Honesty: classical heuristics are excellent; near-term QAOA advantage is
  unproven and we say so.
- **Status.** New service; smallest engineering lift in the stack (5-qubit circuits,
  existing EIG machinery).

---

## 7. Q5 — Quantum Trajectory Engine `[B — the one provable advantage]`

- **Why quantum at this stage.** Disease progression (stable → progressing → invasive,
  observed at checkup intervals) is a stochastic process. Here quantum has one of its few
  **provable representational advantages relevant to modeling**: quantum models can
  simulate stochastic processes using **strictly less memory than the provably minimal
  classical models** (ε-machines) — Gu, Wiesner, Rieper & Vedral, *Nature Communications*
  2012, with the memory-compression line extended by Mahoney/Aghamohammadi/Crutchfield.
  The advantage is in *statistical memory dimension*, *not runtime* — and that is exactly
  what we claim, nothing more. Product relevance: compact progression models with honest
  generative uncertainty feed the v2 risk curves ("what does waiting four weeks cost?").
- **Approach.** Quantum stochastic-process model (a unitary on memory+output qubits,
  measured each timestep — a quantum HMM), parameters trained variationally to match
  sequence statistics. Chosen over a VQC-classifier because the object we need *is a
  generative process model*, and over kernels because we need rollouts, not similarity.
- **Inputs → outputs.** In: longitudinal evidence sequences per loop (from v2
  `trajectory`). Out: progression forecasts `P(state at t+Δ | belief)` with sampling
  uncertainty (via Q6); memory-dimension diagnostics; rollout ensembles for risk curves.
- **Classical integration.** Classical HMM with matched state count is the resident
  baseline (Benchmark Law: predictive log-likelihood, plus statistical-memory comparison);
  outputs feed `trajectory`'s risk curves, which keep their literature-derived hazard
  templates as the fallback and the guideline window as the floor (v2 §6.4 failure
  doctrine).
- **Hackathon demo.** Train on synthetic nodule-growth sequences; show a **1-qubit quantum
  memory reproducing the statistics that require a 3-state classical machine**, next to
  the log-likelihood table. Small, verifiable, and backed by the one citation in this
  document that contains the word "provable."
- **FTQC outlook.** Long-horizon, multi-morbidity progression processes with larger memory
  gaps; amplitude estimation for rollout statistics with quadratically fewer samples
  (Montanaro 2015) `[C]`.
- **Status.** New; 2–3 qubit circuits; the scientifically strongest talking point in the
  stack when precision matters (judges with physics backgrounds).

---

## 8. Q6 — Quantum Uncertainty Service `[A/B]`

- **Why quantum at this stage.** Uncertainty is not an add-on in quantum mechanics — it is
  the output format. Three product uses:
  1. **Shot statistics as calibrated spread** `[A — exists]`: finite-shot variance of
     every readout propagates to `posterior_std` (`quantum.py`); v1 already reports it
     and the safety engine already consumes it.
  2. **Scenario sampling** `[B]`: measurement of the fused state *is* sampling from the
     modeled joint distribution — Born-rule sampling generates coherent joint outcomes of
     unresolved evidence (respecting the correlations Q1/Q2 learned), driving the risk-curve
     Monte Carlo. Classically you would fit and validate a *separate* generative model;
     here inference and generation are the same trained object. The claim is
     **consistency and parsimony, not sampling hardness** — our distributions are not
     candidates for sampling-supremacy arguments and we won't pretend they are.
  3. **Adaptive shot budgeting** `[A]`: allocate measurement shots where decisions are
     sensitive — near abstention thresholds and conformal-set boundaries — mirroring how
     real hardware time would be spent. On simulators this is a cost model; on hardware
     it becomes literal scheduling.
- **Approach.** Born-machine sampling from Q1/Q2 circuits (shots mode of the same QNodes);
  mixed-state entropy accounting shared with Q2. No new model families — this service
  *operates* the others' circuits in sampling mode.
- **Inputs → outputs.** In: any stack circuit + shot budget + sensitivity targets. Out:
  scenario ensembles, per-decision confidence intervals, shot-allocation plans,
  `posterior_std` fields.
- **Classical integration.** Feeds the existing safety engine (abstention thresholds,
  conformal machinery stay classical and distribution-free — conformal guarantees are
  *classical statistics* and remain so); risk curves consume scenario ensembles.
- **Hackathon demo.** One slider: shots 64 → 4096, watch `posterior_std` tighten and an
  abstention flip to a confident call — "this is what buying more quantum measurement
  literally buys you." Honest, visual, thirty seconds.
- **FTQC outlook.** Quantum amplitude estimation gives a **provable quadratic reduction in
  sample complexity for estimating expectations** (Montanaro 2015) — directly applicable
  to EIG estimation (an expectation) in Q4 and rollout statistics in Q5 `[C]`. This is
  the most defensible future-speedup line in the entire stack.

---

## 9. Killed, merged, and banned (the discipline section)

| Candidate | Verdict | Reason |
|---|---|---|
| **Quantum Clinical Memory** (storing beliefs *in* quantum states) | ❌ **killed** | There is no persistent quantum memory; decoherence makes "storage" the single thing quantum computers cannot do. Storage is classical (Postgres + object store + audit chain), full stop. The *retrieval* intelligence lives in Q3; the name was a category error and saying so is worth more than the service |
| **Quantum Belief Update** (as separate from fusion) | ✅ **merged into Q2** | One belief formalism (density matrix + channels), one service; fusion (Q1) remains its single-encounter special case |
| **Quantum Cohort Retrieval** | ✅ **merged into Q3** | Same kernel, same registry, different query shape; a separate service would duplicate the Gram machinery |
| **Quantum Knowledge Embedding** (knowledge-graph embeddings in Hilbert space) | ⚠️ **reframed as Q0** + deferred | QKG-embedding literature is thin and classical KG embeddings are strong; the *useful* kernel of the idea — one governed, trained clinical feature map — is Q0. Quantum walks on ontology graphs (provable hitting-time speedups) go to the FTQC table `[C]` |
| **Quantum image processing** | ❌ stays dead (v1 decision) | CNNs won perception; amplitude-encoding megapixel images is a data-loading fantasy |
| Quantum in **guideline / ledger / report / audit** | 🚫 **banned** | The defensibility path must be deterministic and exactly reproducible; probabilistic hardware/simulators have no business there |

---

## 10. Hackathon demo plan — "the Quantum Bench"

Extends the existing `aura_cli bench` and the dashboard with one new tab. Total new demo
surface: one CLI command, one page, five artifacts — all Benchmark-Law compliant.

| Beat | What judges see | Services | Time |
|---|---|---|---|
| 1 | Existing fusion table: quantum vs classical, six metrics, held-out — presented as measured, parity acknowledged | Q1 | 30 s |
| 2 | **Coherence heatmap** live in the cockpit: ambiguity concentrated on pneumonia↔CHF; BNP recommended *because that pair* is where the entropy lives | Q2→Q4 | 45 s |
| 3 | **Order-replay**: same evidence, two arrival orders, different sequential beliefs — anchoring made visible, flagged | Q2 | 30 s |
| 4 | Panel selection: greedy picks the redundant pair; QAOA + annealer + brute force agree on the complementary panel — optimum *verified by enumeration* | Q4 | 45 s |
| 5 | 1-qubit progression model matching 3-state classical statistics (the provable-memory-advantage story, one table) | Q5 | 30 s |
| 6 | Shots slider: 64 → 4096, `posterior_std` tightens, an abstention resolves | Q6 | 20 s |
| 7 | Closing line: *"Six quantum services, every one shipped next to its classical twin, every claim on a benchmark table. This is what quantum-native looks like when you're honest about it."* | — | 10 s |

Engineering budget (all simulators, all local): Q1 8 qubits statevector (exists) ·
Q2 3–4 qubits `default.mixed` · Q3 8-qubit overlaps on 50-candidate shortlists, Gram rows
cached · Q4 5-qubit QAOA p≤3 · Q5 2–3 qubits · Q6 shots mode of existing QNodes. Nothing
quantum in the interactive read path — all results computed async on evidence events and
cached into the case bundle (v2 latency budgets unchanged: reads < 300 ms).

---

## 11. Fault-tolerant horizon — what real hardware would change

Labeled `[C]` throughout; the product must remain whole if none of it arrives.

| Capability (provable basis) | Stack impact | Fine print we acknowledge |
|---|---|---|
| **Amplitude estimation** — quadratic sample-complexity reduction for expectations (Montanaro 2015) | EIG estimation in Q4 over large outcome trees; rollout statistics in Q5; conformal calibration statistics | Needs deep coherent circuits ⇒ error correction; crossover point unknown |
| **Amplitude amplification / Grover** — O(√N) unstructured search | Q3 retrieval over million-case federated cohorts | Requires QRAM/oracle access; data loading can negate the speedup; quadratic gains are fragile against EC overhead (Babbush et al. 2021) |
| **Quantum walks** — quadratic hitting-time speedups on graphs | Reasoning over SNOMED/ontology graphs (the deferred knowledge-graph idea) | Query-model results; practical embedding unclear |
| **Faithful large feature maps** — 50–200 channel encodings, deep trained ansätze | Q0/Q1/Q3 on full multimodal evidence (labs, genomics, notes) | Barren plateaus and kernel concentration may cap useful scale; benchmarks decide |
| **Large mixed-state belief spaces** | Q2 at thousand-hypothesis lattices | Speculative; formalism ready, hardware isn't |
| Quantum linear algebra (HHL lineage) | *Not banked on.* | The fine print (sparsity, conditioning, state prep, readout — Aaronson 2015) excludes our use cases today; listed to show we considered and declined it |

---

## 12. Scientific risk register

| Risk | Where it bites | Mitigation |
|---|---|---|
| **Barren plateaus** (McClean et al. 2018) | Q0/Q1/Q2 training at depth | Shallow circuits (≤4 layers), small n, layerwise training; benchmarks catch trainability collapse |
| **Exponential kernel concentration** (Thanasilp et al.) | Q3 at high qubit counts | Stay ≤16 qubits, trained (not random) maps, bandwidth tuning; precision@k monitored per registry version |
| **Dequantization / classical surrogates** | Q1/Q3 claims | The Benchmark Law converts this from embarrassment to procedure: surrogate wins ⇒ surrogate ships |
| **Data-loading bottleneck** | Everything, on hardware | Designed around it: evidence vectors are 8–16 dims *by construction* — the v1 compression decision is the whole reason this stack is credible |
| **NISQ noise** (when hardware adapters land) | All services | Analytic simulation today; error mitigation (ZNE) at P2; hardware runs are research artifacts, not serving paths, until benchmarks say otherwise |
| **Hype backlash** | The company | This document. Tiered claims, twin fallbacks, published benches — the honesty *is* the differentiation |

---

## 13. Summary — what is quantum-native here, honestly

- **Today, measured `[A]`:** VQC evidence fusion with shot-noise uncertainty (running);
  fidelity-kernel re-ranking; QAOA panel selection verified against brute force; adaptive
  shot budgeting. All with classical twins on one benchmark table.
- **Today, conceptually grounded `[B]`:** density-matrix belief with coherence-targeted
  ambiguity and order-sensitivity flags (quantum-cognition lineage); quantum stochastic
  progression models (provable memory advantage — Gu et al. 2012); Born-rule scenario
  sampling.
- **Vision, labeled `[C]`:** amplitude-estimation speedups for EIG and rollouts; Grover
  retrieval at federated scale; quantum walks on ontologies; multimodal feature maps.
- **Never:** quantum perception, quantum "storage", quantum anywhere in the deterministic
  defensibility path.

The stack's thesis in one sentence: **quantum computation belongs where clinical reasoning
is small, structured, correlation-rich, and probabilistic — belief, similarity, planning,
progression, and uncertainty — and it earns each seat on a benchmark table, next to a
classical twin, or it doesn't ship.**

---

*References (claim-supporting): Havlíček et al., Nature 567 (2019) · Liu, Arunachalam &
Temme, Nat. Phys. 17 (2021) · Gu, Wiesner, Rieper & Vedral, Nat. Commun. 3, 762 (2012) ·
Pérez-Salinas et al., Quantum 4, 226 (2020) · Schuld, arXiv:2101.11020 (2021) · Hubregtsen
et al., Phys. Rev. A 106 (2022) · Farhi, Goldstone & Gutmann, arXiv:1411.4028 (2014) ·
Montanaro, Proc. R. Soc. A 471 (2015) · McClean et al., Nat. Commun. 9 (2018) · Thanasilp
et al., Nat. Commun. (2024) · Babbush et al., PRX Quantum 2 (2021) · Aaronson, Nat. Phys.
11 (2015) · Busemeyer & Bruza, Quantum Models of Cognition and Decision (CUP, 2012).*
