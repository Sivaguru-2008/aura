# AURA — Grading Readiness Against the Stated Rubric

**Rubric:** methodology mapping 10 · approach 5 · feasibility 5 · quantum utilization 10 ·
quantum optimization 10 · quantum implementation 10 — **50 total, 30 of it (60%) quantum.**

**Assessed:** 2026‑07‑30, against the working tree at `HEAD = 5eb1b70` + the remediation in
`CODEBASE_REVIEW.md`. Every number below is read from a committed artifact, not from prose.

> This is my prediction of how a technical judge applying that rubric would score the
> project as it stands, and what moves each number. Judges vary; the *reasoning* is the
> useful part, not the digits.

---

## 1. Predicted scorecard

| Criterion | Weight | At assessment | **After the work below** | What changed |
|---|---:|---:|---:|---|
| Methodology mapping | 10 | 7 | **9** | `docs/METHODOLOGY_MAPPING.md` — one page, every decision justified, artifact-indexed |
| Approach | 5 | 4 | **5** | Narrative reordered: capability first, negative results kept in full |
| Feasibility | 5 | **5** | **5** | Unchanged — already the strongest category |
| **Quantum utilization** | 10 | 5 | **9** | QMBA is on the serving path, in the API and in the console |
| **Quantum optimization** | 10 | 5 | **8** | 48-cell design sweep, transpiler L1→L3 (−20% 2q gates), noise attribution |
| **Quantum implementation** | 10 | 8 | **9** | Noise-model rung added; 4 entangler topologies implemented and measured |
| **Total** | **50** | **≈34** | **≈45** | |

> **Status: executed.** Everything in §5's plan has been implemented and measured except the
> queue-bound hardware item (§7), which needs the operator's IBM quota. Section 7 below records
> what each change actually produced — including one proposed "win" that testing killed.

**The headline:** implementation is the strongest pillar and needs almost nothing.
Utilization and optimization are together worth **20 points** and are where the project
currently under-scores — not because the quantum work is weak, but because the best of it
is sitting in `ml/evaluation/` instead of in the serving path, and because the narrative
foregrounds "classical wins" over "here is what quantum uniquely bought us."

---

## 2. What is actually there (verified)

### Served quantum surface

| Component | Status | Evidence |
|---|---|---|
| **Fusion VQC** — 8 qubits, 3 layers, 102 trainable params, `RY(πx_i)` encoding, ring CNOT, `⟨Z_i⟩` → linear head → softmax | ✅ **served** (`fusion_backend = "quantum"`) | `artifacts/fusion_quantum.npz`, `services/fusion/device.py` |
| **QKL** — quantum-kernel head on brain studies, glioma grade | ✅ **served** (`neuro_qkl_enabled = true`) | `artifacts/brain/qkl_classifier.npz` |
| **IBM Quantum provider** — Runtime EstimatorV2/SamplerV2, TREX + ZNE, preset transpilation, graceful fallback | ✅ wired, opt-in | `services/quantum/ibm.py` |
| **Braket provider** — IonQ / Rigetti / IQM | ✅ wired, opt-in | `services/quantum/braket.py` |

### Built, evidenced — but **not on the serving path**

| Component | Status | Why it matters |
|---|---|---|
| **QMBA** — sequential shot-budgeted abstention | ❌ imported only by `ml/evaluation/quantum_demo.py`, `quantum_study.py`, one test | **This is the single best quantum-utilization argument in the project** (§3) |
| **Data re-uploading ansatz** | ❌ `STATUS — EXPERIMENTAL, NOT WIRED` | Barren-plateau mitigation, written and reasoned, never used |
| **JointProjection** (high-dim embedding path) | ❌ `NOT WIRED INTO THE SERVING PATH` | The designed partner for re-uploading |
| **QAE** (amplitude estimation) | ❌ `qae_enabled = False` | — |
| **QBN** (quantum Bayesian net) | ❌ `reasoner_backend = "classical"` | — |

That table is the score. A judge grading *utilization* counts what runs, and five of the
nine quantum components are dark.

---

## 3. The one change that moves the most points

**Wire QMBA into the serving path and lead the narrative with it.**

QMBA is the only thing in the project that does something a classical model **cannot do at
all** — and that is precisely what "quantum utilization" is asking about. From
`artifacts/quantum_study.json` (447 patient-disjoint studies):

```
commit rate                86.4%
median shots spent         128        (vs a fixed 512-shot budget → 4× fewer)
abstentions                13.7%  →   1 measurement-limited,  60 model-limited
median predicted shots to resolve a measurement-limited case:  8,433
```

The clinical content of that split is the argument:

* **Measurement-limited** — the answer exists, this run just had not bought enough
  precision. *Instruction: run the circuit longer.* The module says how much longer.
* **Model-limited** — the top two diagnoses are tied at infinite measurement precision.
  *Instruction: this case needs a human.* More compute is wasted effort.

No classical model can separate those two, because classical inference has no measurement
budget to vary. A softmax confidence of 0.55 means "unsure" and cannot tell you which kind
of unsure it is. That is a **clinical capability that exists because the model is quantum**
— and it is currently unreachable from the running system.

Wiring it also earns points in the other two quantum categories: it is a genuine
**optimization** (4× median shot reduction on real hardware economics) and its integration
is **implementation** work.

**Effort:** the module is complete, tested and evidenced. This is an integration job in
`gateway/pipeline.py` plus a panel in the console, not new research.

---

## 4. Criterion-by-criterion

### 4.1 Quantum implementation — 8/10 (strongest; leave mostly alone)

What a judge will find and credit:

* **Ran on real hardware.** `ibm_marrakesh` (156 qubits), job `d9js49rjf64c739haeg0`,
  retrievable. Top-1 diagnosis survived device noise (`pneumothorax_dx` both analytic and
  hardware), mean `|Δ⟨Z⟩|` vs analytic **0.186**, per-qubit shot std ~0.013.
* **Cross-framework verification.** The same circuit built independently in Qiskit and
  PennyLane agrees to **~1e-15** (`local_check.z_qiskit` vs `z_pennylane`). That is the
  kind of check that separates "I called a library" from "I know what the circuit does."
* **Error mitigation is real, not claimed.** `ERROR_MITIGATION = {none:0, readout:1 (TREX),
  zne:2}` mapped onto EstimatorV2 resilience levels; transpiled depth and two-qubit gate
  count are recorded per run.
* **Everything degrades.** Missing `qiskit-ibm-runtime`, absent credentials, no operational
  device, or an over-long queue all raise `ProviderUnavailable` → local simulator with
  `fell_back=True`. Nothing silently pretends.

Remaining 2 points:

1. **n=1 on hardware.** One case (`test_index 144`) was executed on a QPU. A judge reading
   "ran on real hardware" will ask how many. Even 20–30 cases would turn an anecdote into a
   measurement — and would let you report *top-1 agreement rate under device noise*, which
   is a far stronger sentence than one matching case.
2. **No noise-model rung.** There is ideal simulator and there is hardware, with nothing
   between. A `FakeMarrakesh`/noise-model run isolates *how much* of the 0.186 gap is
   decoherence vs sampling, and it is a 20-line addition.

### 4.2 Quantum optimization — 5/10 (biggest gap, 10 points at stake)

"Optimization" in a quantum rubric means: did you *engineer* the circuit, or accept the
first one that worked? Today the evidence for engineering is thin, and what exists is
mostly unwired.

**Present:**
* Barren-plateau reasoning with three named levers and citations (Cerezo 2021 local cost,
  Pérez-Salinas 2020 re-uploading), documented in `device.py`.
* A rigorous **entanglement ablation** — same qubits, layers, parameter count, encoding,
  readout, optimiser and seed; only the CNOT ring removed.
* QMBA's 4× median shot reduction.

**Missing — and each is a straightforward, high-yield study:**

| Gap | Why it costs points | Cost to fix |
|---|---|---|
| **No qubit-count sweep.** Why 8? (Because there are 8 evidence channels — defensible, never demonstrated.) | Judges read an unjustified hyperparameter as an unconsidered one | ~1 hr, the harness exists |
| **No layer-depth sweep.** Why 3 layers / 102 params? | Same | ~1 hr |
| **No entangler-topology study.** `ring` vs `none` exists; `linear`, `all-to-all`, `circular+reverse` do not | Topology *is* the optimization knob on hardware-efficient ansätze | ~2 hrs |
| **`optimization_level=1`** in `generate_preset_pass_manager` | Level 3 is the standard for a "we optimized" claim; you already record depth + 2q-gate count, so the before/after is free | 5 min + a rerun |
| **QMBA unwired** | The one real optimization is not in the product | integration |
| **Re-uploading ansatz unwired** | Written, reasoned, cited — and dead | integration |

A single table — *qubits × layers × entangler → accuracy, NLL, ECE, depth, 2q-gate count,
train time* — would move this from 5 to 8 on its own. You have the training harness, the
ablation machinery, and the bootstrap code already.

### 4.3 Quantum utilization — 5/10 (biggest risk, and it is a framing problem as much as a code problem)

Here is the uncomfortable part, stated plainly.

The project's most prominent quantum claim is currently **a negative result about itself**:

* Fusion, n=69: classical PoE 0.6957 vs quantum VQC 0.6377 — *(not significant; McNemar
  p ≥ 0.125 under every pairing — see `docs/BENCHMARKS.md`)*
* Entanglement ablation, n=447, 2000 bootstrap resamples:
  accuracy Δ **−0.029** CI [−0.058, 0.000] (ns) · NLL Δ **+0.056** CI [+0.022, +0.090]
  **(significantly worse)** · ECE Δ **−0.027** CI [−0.068, +0.018] (ns)
* QKL, n=669: quantum AUROC 0.706 vs classical RBF 0.721, Δ **−0.016**, p = **0.002**
  **(significantly worse)**

**That honesty is the project's best quality and I am not suggesting you soften a word of
it.** It is also, right now, the first thing a judge scoring *utilization* reads — and the
natural inference from "entanglement significantly worsens NLL" is "the quantum part is
decorative." That inference is wrong, but the documents currently invite it.

The fix is **ordering, not spin**. Two true statements, in the right order:

> **Lead:** AURA uses quantum measurement as a clinical instrument. Because a quantum model's
> precision is bought with shots, AURA can ask "is this case unresolved because we haven't
> measured enough, or because the model genuinely cannot separate these two diagnoses?" —
> and answer it per patient (447 studies: 1 measurement-limited, 60 model-limited). No
> classical model can make that distinction. That capability, not an accuracy delta, is what
> the quantum layer is for.
>
> **Then:** On raw accuracy the 8-qubit VQC does *not* beat a fairly-calibrated classical
> product-of-experts, and we publish the ablation showing the CNOT ring does not earn its
> place on an 8-channel evidence vector. We report that because a benchmark you only publish
> when it flatters you is not a benchmark.

That is the same evidence with the load-bearing claim first. It costs nothing in integrity
and it is the difference between "quantum is decorative" and "quantum is doing a job
classical cannot."

**One genuinely positive quantum finding is currently buried and should be surfaced:** the
QKL head is *better calibrated* than its classical counterpart — **ECE 0.0277 vs 0.0408**,
a 32% reduction — while losing 0.016 AUROC. In a system whose entire thesis is calibrated
doubt, "the quantum kernel trades a little discrimination for materially better calibration"
is directly on-message and is sitting unmentioned in
`artifacts/brain/qkl_training_report.json`. Check whether that ECE gap is significant under
the bootstrap you already run; if it is, it belongs in the README.

### 4.4 Methodology mapping — 7/10

The mapping is real and defensible: 8 clinical evidence channels → 8 qubits, one channel per
qubit, `RY(π·x_i)` angle encoding, higher-order evidence interactions represented in a 2⁸
Hilbert space, `⟨Z_i⟩` readout into a linear diagnostic head. `q4_evidence_coupling` even
*measures* the learned pairwise coupling (e.g. nodule↔opacity 0.539, cardiomegaly↔consolidation
0.436), which is exactly the artifact that justifies the mapping empirically.

What costs the 3 points: **there is no single document that states the mapping and lets a
judge grade it quickly.** It is distributed across `device.py` docstrings, two 40–67 KB
"bible" files, and a JSON artifact. Add one page — problem → why a VQC → encoding choice →
ansatz choice → readout choice → what each is justified by — and cite the coupling matrix as
the evidence. This is the cheapest 2–3 points on the board.

### 4.5 Approach — 4/5 · Feasibility — 5/5

Feasibility is the quietly outstanding category and needs nothing: runs fully offline on CPU,
every optional dependency degrades to a documented fallback, Docker + compose + CI, a preflight
that refuses to boot on a missing served artifact, and a real QPU execution path that falls
back cleanly. That is a 5.

Approach loses its point to the same ordering problem as §4.3.

---

## 5. Prioritized plan

| # | Action | Points | Effort | Category |
|---|---|---:|---|---|
| 1 | **Wire QMBA into the serving path** + a console panel showing shots spent / measurement- vs model-limited | **+3–4** | ~half day | utilization, optimization |
| 2 | **Reorder the quantum narrative** — capability first, negative benchmark second (README, WHAT_IS_AURA, BENCHMARKS) | **+2–3** | ~2 hrs | utilization, approach |
| 3 | **One-page methodology-mapping doc**, citing the coupling matrix | **+2–3** | ~2 hrs | methodology |
| 4 | **Design-space sweep table** — qubits × layers × entangler → acc/NLL/ECE/depth/2q-gates | **+2–3** | ~half day | optimization |
| 5 | **Surface the QKL calibration win** (ECE 0.0277 vs 0.0408), with significance | **+1** | ~1 hr | utilization |
| 6 | **`optimization_level=3`** + publish before/after depth and 2q-gate count | **+1** | ~15 min | optimization |
| 7 | **Run 20–30 cases on hardware**, report top-1 agreement rate under noise | **+1** | queue-bound | implementation |
| 8 | Noise-model rung between ideal sim and QPU | +0.5 | ~1 hr | implementation |

Items **1–3 are the whole game**: roughly a day of work for an estimated **+7 to +10 points**,
all of it either integration of code that already exists or re-ordering of claims that are
already true.

---

## 6a. What the work actually produced

Executed 2026-07-30/31. Every number below is from a committed artifact.

### QMBA is on the serving path

`gateway/pipeline.py` runs the measurement budget on every quantum-backend study; the
result rides on `CaseBundle.measurement`, is returned by `GET /v1/cases/{id}`, and renders
in a new console panel. Cost: **27 ms/study** (0.42× the fusion step it annotates). It is
reporting, never a gate — a QMBA failure degrades to `None` and cannot fail a study — and
on a classical backend the field is `None` rather than a fabricated budget. Live behaviour
on ten studies: 128 shots on hopeless cases, 1,920 on hard-but-resolvable ones, median 384
against a fixed 512. 9 new tests pin it.

### Circuit cost is now engineered and measured

* **Two new entangler topologies** (`linear`, `full`) alongside `ring`/`none`, with an
  exact `two_qubit_gate_count()`, so topology is a measured trade rather than a constant.
* **48-cell design sweep** (`docs/DESIGN_SPACE.md`) over qubits × layers × topology.
* **Transpiler L1 → L3**, measured on the real `ibm_marrakesh` coupling map:
  **75 → 60 two-qubit gates (−20%)**, depth 185 → 146 (−21%). Both levels are still run so
  the improvement is a number in the artifact rather than an assertion.
* **Noise attribution** (`artifacts/noise_rung.json`): at 4096 shots, sampling contributes
  mean \|Δ⟨Z⟩\| 0.0074 and device noise 0.0544 — **88% is decoherence.** That is the
  measurement proving gate count is the right lever, and it is why the transpiler work
  matters. Real hardware sits at 0.1858, ~3× the static noise model; top-1 survives all
  three rungs.

### Two findings that went against us — reported anyway

**The QKL calibration "win" is not real.** §4.3 of this document recommended surfacing
QKL's better ECE (0.0276 vs 0.0408, a 32% point-estimate gap) if it survived testing. It
does not. The paired subject-level bootstrap gives Δ **−0.0009**, CI [−0.029, +0.015],
**p = 0.88** — with 11 test subjects the point estimate is noise. It is now computed on
every run, printed next to the AUROC delta, and stored in the artifact, specifically so the
flattering half cannot be quoted alone. **It is not claimed anywhere.**

**Entanglement loses across the whole grid.** The sweep independently reproduces the
existing ablation: at 8 qubits the product-state control (0 two-qubit gates) beats every
entangling topology, including all-to-all at 70 gates. The served `ring` configuration is
*not* the best cell. It is published that way rather than silently re-tuned, because one
grid search on one split is not grounds to change a served clinical model — but the honest
statement is now on the record.

Both are better outcomes than a manufactured win. A judge who sees a project publish two
results against its own thesis has a reason to believe the rest of the numbers.

### Still open

* **Hardware n=1 → n≥20** (§4.1) needs the operator's IBM quota and queue time. Not run:
  spending someone's QPU allocation is their call, not mine. The harness supports it
  (`ml/evaluation/run_ibm_hardware.py`, with a `--job-id` retrieve mode).

---

## 6. What not to do

**Do not manufacture a quantum win.** The temptation, with 30 of 50 points on quantum, is to
find a split or a metric where the VQC comes out ahead and lead with it. Everything in this
repository — the drift guards, the n=69 disclosure, the published ablation, the abstaining
QKL subtype head — exists because someone decided not to do that. It is the project's most
unusual quality and the fastest thing to lose.

A judge who catches one inflated number discounts every other number on the page. A judge who
sees a negative result published cleanly, next to a quantum capability that classical cannot
replicate, has been given a reason to trust the whole thing. The second is worth more, and it
is already true — it just needs to be said first.
