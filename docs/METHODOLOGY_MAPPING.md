# AURA — Problem → Quantum Method

One page. What the clinical problem is, why a variational quantum circuit is the
method chosen for it, and what justifies each design decision. Every claim here points
at a committed artifact you can open.

---

## 1. The problem, stated precisely

A chest radiograph produces **seven finding probabilities** (opacity, consolidation,
effusion, cardiomegaly, nodule, hyperinflation, pneumothorax) from a DenseNet-121.
Structured priors contribute an **eighth** channel (prior risk). The task is to map
that 8-dimensional evidence vector to a **posterior over six diagnoses** — and, because
this is a clinical system, to know when it cannot.

```
  vision (DenseNet-121)  ->  x ∈ [0,1]^8  ->  FUSION  ->  p(diagnosis | evidence) ∈ Δ^5
                                                  |
                                                  +-> abstain when unresolvable
```

The hard part is not the mapping. It is that **evidence channels interact**. Opacity
plus consolidation means something different from either alone; cardiomegaly with
effusion suggests failure, while cardiomegaly with pneumothorax suggests something else
entirely. A model that treats the channels as independent throws that away.

## 2. Why a variational quantum circuit

An 8-qubit register spans a 2⁸ = 256-dimensional Hilbert space. Encoding one evidence
channel per qubit and entangling them represents joint evidence configurations
directly, rather than as an explicit feature-cross the modeller has to enumerate.

That is the *hypothesis*. AURA tests it rather than asserting it, and reports what the
test says — including when the answer is no (§6).

**The honest scope claim:** at 8 qubits this problem is classically simulable, so no
speed-up or capacity advantage is claimed and none is possible. What the quantum
formulation genuinely provides is a different resource model — precision costs
measurements — which §5 turns into a clinical capability that has no classical
counterpart.

## 3. The mapping, decision by decision

| Design decision | Choice | What justifies it |
|---|---|---|
| **Qubit count** | 8 | Set by the **input**, not by expressivity: the encoding is one evidence channel per qubit, so 8 channels ⇒ 8 qubits. `docs/DESIGN_SPACE.md` sweeps {4,6,8} and shows what dropping channels costs. |
| **Encoding** | `RY(π·xᵢ)` on qubit *i* | Findings are already probabilities in [0,1]. `RY(π·x)` maps that interval onto a full meridian of the Bloch sphere — x=0 → \|0⟩, x=1 → \|1⟩, x=0.5 → equator — so a calibrated probability becomes a calibrated amplitude with no rescaling and no learned input layer to distort it. |
| **Ansatz** | 3 layers × per-qubit (RY, RZ) | Hardware-efficient: the two-axis rotation pair spans arbitrary single-qubit unitaries, and depth stays shallow enough to keep the local-cost barren-plateau argument (§4) applicable. Layer count swept in `DESIGN_SPACE.md`. |
| **Entangler** | CNOT ring | The topology that couples every channel to its neighbours with `n` two-qubit gates instead of `n(n−1)/2`. Two-qubit gates dominate the hardware error budget, so topology is a cost decision as much as a capacity one — `linear`, `full` and `none` are all implemented and measured. |
| **Readout** | `⟨Zᵢ⟩` → linear head → softmax | A *local* observable, which is what keeps gradient variance polynomial rather than exponential (§4). The linear head keeps the quantum part a feature map and the diagnostic decision inspectable. |
| **Parameters** | 102 trainable | Small on purpose. The training set is ~1,300 patient-disjoint studies; a larger circuit would overfit before it expressed anything the data supports. |

### The mapping is measured, not just argued

If the entangled representation is doing nothing, learned pairwise coupling would be
uniform. It is not. From `artifacts/quantum_study.json` (`q4_evidence_coupling`), the
mean absolute differential coupling between channels:

| Channel pair | Coupling |
|---|---:|
| opacity ↔ nodule | **0.539** |
| consolidation ↔ cardiomegaly | **0.436** |
| effusion ↔ nodule | 0.398 |
| cardiomegaly ↔ pneumothorax | 0.334 |
| … | |
| opacity ↔ cardiomegaly | **0.033** |

A 16× spread between the strongest and weakest pair. The circuit learned that some
evidence channels inform each other and others do not — which is the structure the
mapping was chosen to capture.

## 4. Barren plateaus — three levers, named

Gradient variance in a random deep circuit vanishes as 2⁻ⁿ, which would make training
impossible. Three properties of this design attack that directly:

1. **Width is capped by the input.** `n = 8` regardless of image resolution or model
   size, because the evidence vector is 8-dimensional. Capping *n* is the only lever
   that touches the exponent itself.
2. **The cost is local.** Readout is single-qubit `⟨Zᵢ⟩`. Cerezo et al. (2021) show
   local cost functions on shallow (O(log n)) circuits have gradient variance vanishing
   only *polynomially*.
3. **Data re-uploading is available** (`make_reuploading_qnode`) to raise expressivity
   without deepening the trainable block (Pérez-Salinas et al. 2020). **Status:
   implemented, not served** — the shipped circuit uses single angle-encoding, and this
   is documented as an extension point rather than presented as active.

These are mitigations with citations, not a theorem. Training converges in ~44s on CPU
at the shipped size, which is the practical evidence.

## 5. What the quantum formulation actually buys — measurement as an instrument

This is the load-bearing part of the mapping, and it does not depend on beating a
classical model on accuracy.

On a quantum device you never obtain an expectation value, only an estimate whose
precision you purchase:

```
    Var[⟨Z⟩] = (1 − ⟨Z⟩²) / n_shots
```

So "how many measurements is this patient worth?" is a real question with a real
answer, and answering it *sequentially* splits an unresolved case into two states that
call for **opposite clinical actions**:

* **measurement-limited** — the analytic margin is genuinely non-zero; this run had
  not bought enough precision. → *Run the circuit longer.* The system reports how much.
* **model-limited** — the top two diagnoses are tied at **infinite** measurement
  precision. No budget resolves it. → *Escalate to a human.*

Measured over 447 patient-disjoint studies (`artifacts/quantum_study.json`,
`q3_measurement_budget`): 86.4% commit, median **128 shots** against a fixed 512-shot
budget, and of 61 abstentions **1 was measurement-limited and 60 were model-limited**.

A classical softmax reading 0.55 means "unsure" and cannot tell you which kind of
unsure it is, because classical inference has no measurement budget to vary. This runs
on the serving path (`gateway/pipeline.py`) and surfaces in the console.

## 6. What the mapping does *not* buy

Reported here rather than in a footnote, because a benchmark you only publish when it
flatters you is not a benchmark.

* **Accuracy.** On the real MIMIC-CXR evidence distribution the 8-qubit VQC does not
  beat a fairly-calibrated classical product-of-experts (0.6377 vs 0.6957, n=69 — a
  four-case difference that is *not* statistically resolvable; McNemar p ≥ 0.125 under
  every pairing). See `docs/BENCHMARKS.md`.
* **Entanglement, specifically.** The ablation holds qubits, layers, parameter count,
  encoding, readout, optimiser and seed fixed and removes only the CNOT ring. Over 447
  studies with 2,000 bootstrap resamples: accuracy Δ −0.029 (CI [−0.058, 0.000], ns),
  **NLL Δ +0.056 (CI [+0.022, +0.090] — significantly worse)**, ECE Δ −0.027
  (CI [−0.068, +0.018], ns).

  The 48-cell design sweep reproduces this independently across four topologies and
  four depths (`docs/DESIGN_SPACE.md`). Mean accuracy at 8 qubits:

  | topology | mean accuracy | mean NLL | mean 2q gates |
  |:---|---:|---:|---:|
  | `none` (product state) | **0.5637** | **1.2193** | **0** |
  | `full` (all-to-all) | 0.5487 | 1.2403 | 70.0 |
  | `ring` (served) | 0.5386 | 1.2956 | 20.0 |
  | `linear` | 0.5308 | 1.2994 | 17.5 |

  The product-state control beats every entangling topology, including all-to-all at 70
  two-qubit gates. Two independent experiments, same direction: **on this 8-channel
  evidence vector the CNOT ring is not earning its place.** The served configuration is
  published unchanged rather than re-tuned to the winner, because a single grid search on
  this split is not grounds to alter a served clinical model — but the honest statement
  is that if the ring is kept, it is kept for representational reasons, not measured ones.
* **The quantum kernel.** QKL loses to a classical RBF on glioma grading by ΔAUROC
  −0.016 (p = 0.002, significant). Its ECE *looks* better (0.028 vs 0.041) but the
  paired subject-level bootstrap puts that at Δ −0.0009 (CI [−0.029, +0.015],
  p = 0.88) — **not significant**, and not claimed.

## 7. Hardware, and where the error comes from

The mapping was executed on a real QPU: **`ibm_marrakesh`** (156 qubits), job
`d9js49rjf64c739haeg0`. The top-1 diagnosis survived device noise. The same circuit
built independently in Qiskit and PennyLane agrees to ~1e-15, so the mapping is
implementation-independent.

The error budget is attributed rather than quoted (`artifacts/noise_rung.json`):

| Rung | mean \|Δ⟨Z⟩\| vs analytic | Top-1 preserved |
|---|---:|:---:|
| ideal simulator, shot noise only (4096 shots) | 0.0074 | ✅ |
| + `FakeMarrakesh` device-noise model | 0.0617 | ✅ |
| real `ibm_marrakesh` hardware | 0.1858 | ✅ |

**Sampling accounts for 12% of the simulated error; decoherence accounts for 88%.**
That settles which lever matters: buying more shots is nearly useless, and shortening
the circuit is the fix. Hence the transpiler work — optimization level 3 rather than 1,
measured on the real coupling map at **60 two-qubit gates instead of 75 (−20%)** and
depth 146 instead of 185 (`artifacts/transpile_study.json`).

Real hardware is 3× worse than the static noise model, which is expected and worth
stating: a calibration snapshot does not reproduce drift or crosstalk.

---

## Artifact index

| Claim | Artifact |
|---|---|
| Circuit definition, all four entangler topologies | `aura/services/fusion/device.py` |
| Evidence coupling, entanglement ablation, shot budget | `aura/artifacts/quantum_study.json` |
| Design-space sweep (qubits × layers × topology) | `aura/artifacts/design_sweep.json`, `docs/DESIGN_SPACE.md` |
| Transpilation before/after on the real coupling map | `aura/artifacts/transpile_study.json` |
| Noise attribution (sampling vs decoherence) | `aura/artifacts/noise_rung.json` |
| Real QPU execution | `aura/artifacts/ibm_hardware_run.json` |
| Backend comparison + significance | `aura/artifacts/benchmark.json`, `docs/BENCHMARKS.md` |
| Quantum kernel result | `aura/artifacts/brain/qkl_training_report.json` |
| Measurement budget on the serving path | `aura/gateway/pipeline.py`, `aura/services/fusion/qmba.py` |
