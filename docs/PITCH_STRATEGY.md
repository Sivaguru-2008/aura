# AURA — Pitch Strategy & Judge Defence

**Rubric:** methodology mapping 10 · approach 5 · feasibility 5 · quantum utilization 10 ·
quantum optimization 10 · quantum implementation 10 → **50 total, 30 of it quantum.**

**Snapshot:** 2026-07-31. Every number below was pulled from the committed artifacts
immediately before writing; the check command is in §7. If you present after a retrain,
re-run it — the QKL figures already moved once during this work.

---

## 1. The thesis

> **Most quantum-health projects claim an advantage they cannot demonstrate. We built the
> one thing a quantum model can do that a classical model structurally cannot — and we
> publish every result that does not favour us.**

That sentence wins this rubric, because the rubric asks about *utilization, optimization
and implementation* — how well you used the quantum resource — **not** "did quantum win."
A team claiming a fake speed-up scores 0 on credibility the moment a judge probes. A team
that shows a real capability plus honest negatives is unattackable, because there is
nothing left to expose.

**The capability, in one breath:** quantum precision is *bought* with measurements
(`Var[⟨Z⟩] = (1 − ⟨Z⟩²)/n_shots`). Sequencing that budget splits "unsure" into two states
with **opposite clinical instructions** — *run the circuit longer* vs *escalate to a
human*. A classical softmax reading 0.55 cannot tell you which. Over 447 patient-disjoint
studies: **1 measurement-limited, 60 model-limited.**

**Say this, verbatim, when asked what's quantum about it:**
> "We don't claim quantum accuracy. We claim a resource model classical inference doesn't
> have. Measurement is a dial, so 'I need more computation' and 'I need a doctor' become
> two different answers instead of one number. That runs on every study we serve."

---

## 2. Rubric-by-rubric play

### Methodology mapping — 10 pts · **artifact: `docs/METHODOLOGY_MAPPING.md`**

Open that file on screen. It is a single page: problem → why a VQC → every design decision
with its justification → what it does not buy → artifact index.

The three sentences that land it:
- **Qubit count is set by the input, not chosen.** 7 findings + 1 prior = 8 channels ⇒ 8
  qubits, one channel per qubit. Nothing arbitrary to defend.
- **Encoding is principled.** Findings are already probabilities in [0,1]; `RY(π·x)` maps
  that onto a full Bloch meridian (0→|0⟩, 1→|1⟩, 0.5→equator), so a *calibrated
  probability becomes a calibrated amplitude* with no rescaling.
- **The mapping is measured, not argued.** If the entangled register did nothing, learned
  pairwise coupling would be flat. It is not: opacity↔nodule 0.539 vs opacity↔cardiomegaly
  0.033 — a **16× spread**, shown as a live heatmap at `/quantum` §07.

### Quantum utilization — 10 pts

Lead with **QMBA on the serving path** (`gateway/pipeline.py`), not with the VQC.

| Measured | Value |
|---|---|
| Commit rate | 86.4% |
| Median shots | 128 (vs a fixed 512 budget) |
| Measurement-limited abstentions | **1** |
| Model-limited abstentions | **60** |
| Median predicted shots to resolve | 8,433 |

Then show it live on an uploaded case: the Assess page renders shots spent, the ±1σ
shot-noise band, and the verdict. **This is the differentiator. Spend your time here.**

### Quantum optimization — 10 pts

The chain is the point: *measure where the error is → fix that specific thing → prove it.*

1. **Noise attribution** (`artifacts/noise_rung.json`) — three rungs on one circuit:
   sampling contributes mean |Δ⟨Z⟩| **0.0074**; adding the device noise model takes it to
   0.0617. **88% of the error is decoherence, not sampling.** So more shots is the wrong
   lever and a shorter circuit is the right one.
2. **Therefore** transpiler level 1 → 3, measured on the real `ibm_marrakesh` coupling map:
   **75 → 60 two-qubit gates (−20%)**, depth 185 → 146. Level 1 is still run on every
   hardware call so the improvement is a number in the artifact, not a claim.
3. **Design sweep** — 48 cells (qubits × layers × 4 topologies) with gate count as the cost
   axis, because two-qubit gates dominate both the error budget and the depth.

Say: *"We didn't guess which lever mattered. We measured that decoherence dominates at 88%,
which told us to cut two-qubit gates, and we cut them by 20% on the real coupling map."*

### Quantum implementation — 10 pts

- **Ran on a real QPU:** `ibm_marrakesh` (156 qubits), job **`d9js49rjf64c739haeg0`**,
  retrievable. Mean |Δ⟨Z⟩| 0.1858 vs analytic, **top-1 diagnosis survived**.
- **Cross-framework verification:** the same circuit built independently in Qiskit and
  PennyLane agrees to ~1e-15 — the implementation is not framework-dependent.
- **Error mitigation wired:** TREX readout twirling, ZNE resilience levels.
- **Backend-agnostic provider layer** with graceful degradation: IBM → Braket → local
  simulator, `fell_back=True` recorded. Four live endpoints under `/v1/quantum/`.
- **Barren plateaus addressed with three named levers** and citations (width capped by
  input; local `⟨Z⟩` cost — Cerezo 2021; data re-uploading available — Pérez-Salinas 2020).

### Approach — 5 pts

The product is **calibrated doubt**, not a diagnosis. Conformal prediction with a 90%
coverage guarantee, temperature scaling, OOD energy scoring, and an explicit abstention
policy whose thresholds are pinned to the served config by a test.

### Feasibility — 5 pts

Runs **entirely offline on CPU** on one box — no external identity provider, no network
call, no GPU required. Docker + compose, `deploy/preflight.py` refuses to boot on a missing
served artifact, DICOM intake, FHIR/HL7 export, path-gated auth. **755 tests, 0 failures.**

---

## 3. The 7-minute demo

| Time | Screen | Line |
|---|---|---|
| 0:00 | `/app`, upload a chest X-ray | "Nothing is pre-baked. This is inference, now." |
| 1:00 | **Read** page — Grad-CAM overlay, evidence graph | "It reports observations, never conclusions." |
| 2:00 | **Assess** — differential + conformal set | "The product is the *set*, with a 90% guarantee." |
| 3:00 | **Assess** — measurement budget panel | **The money moment.** "Shots bought this decision. Had it not resolved, the system would tell you whether more measurement helps or whether you need a human." |
| 4:00 | `/quantum` §02 | "1 measurement-limited, 60 model-limited, over 447 studies." |
| 5:00 | `/quantum` §03 | "Real QPU, job ID retrievable. 88% of error is decoherence — which is why we cut gates." |
| 5:45 | `/quantum` §05 | **"Here is everything that didn't work."** |
| 6:30 | `/quantum` §06–07 | 48-cell sweep + coupling heatmap. "All of it regenerated, none transcribed." |

**Do not skip §05.** Volunteering the negatives is what converts a sceptical judge. If you
present them first, no one can "catch" you with them.

---

## 4. Judge attacks, and exactly how to answer

Ordered by likelihood × damage. The rebuttals are short on purpose — **answer, evidence,
stop.** Do not over-explain; that reads as defensiveness.

### A1 · "Your quantum model loses to classical. Why is this a quantum project?" ★★★★★
> "It does, and we publish it: 0.6957 vs 0.6377. But accuracy was never the claim. The
> claim is a resource model — measurement as a dial — which produces a clinical capability
> classical inference structurally cannot have. That's §02 of our evidence page, and it's
> on the serving path, not in a notebook."

**Never** get defensive here. Concede instantly, pivot to QMBA. This attack is *expected*
and conceding it is what makes everything else believable.

### A2 · "8 qubits is classically simulable. There is no advantage available." ★★★★★
> "Correct, and we say so in our own docs — no speed-up is claimed and none is possible at
> this width. What's available is the resource model, and that's what we built."

### A3 · "Your own sweep says the *product state* wins. Why entangle at all?" ★★★★★
**The hardest question. Know the exact numbers.** Best cell is 8q/1L/`none` at 0.5749 with
**0 two-qubit gates**; four of the top five cells use no entanglement.
> "That's our finding, and we published it rather than burying it. Two things follow.
> First, it independently reproduces our entanglement ablation, which bootstraps NLL as
> **significantly worse** with the ring (Δ +0.056, CI [+0.022, +0.090]). Second, we did
> *not* re-tune the served model to the winner — one grid search on one split is not
> grounds to change a served clinical model. So the honest statement is: on an 8-channel
> evidence vector, the CNOT ring is not earning its 24 gates, and we say so."

If pushed *"then remove it"*:
> "That's the right next step and it's the recommendation in our own docs. What we won't do
> is ship a config change we validated on a single split an hour before a demo."

### A4 · "Your sweep says 0.5749, your benchmark says 0.6377. Which is it?" ★★★★
**Pre-empt this — it looks like a contradiction and isn't.**
> "Different experiments. The sweep is 48 models trained for 20 epochs on a 900-sample
> budget, scored on a 447-study test split. The headline benchmark is the fully-trained
> served model on a 69-case held-out split. Compare cells *within* the sweep, never across."

### A5 · "n=69 is far too small to conclude anything." ★★★★
> "Agreed, which is why we publish the interval, not just the point estimate. 48 vs 44
> correct. Jeffreys CIs [0.581, 0.795] and [0.520, 0.744]. McNemar cannot reach
> significance under *any* pairing of the discordant pairs — best case p = 0.125. Our own
> docs state the backends are not distinguishable at this sample size."

That answer usually ends the line of questioning, because it's more rigorous than the
question.

### A6 · "Did you really run on hardware, or just simulate?" ★★★★
> "Real. `ibm_marrakesh`, 156 qubits, job `d9js49rjf64c739haeg0` — retrievable. Top-1
> survived device noise. And we built the same circuit independently in Qiskit and
> PennyLane; they agree to 1e-15."

### A7 · "One case on hardware isn't a benchmark." ★★★★
> "Correct, n=1, and we don't present it as a benchmark. It demonstrates the pipeline
> executes end-to-end on a QPU and that top-1 survives device noise. The statistical work
> is on the simulator, where we can afford the sample size. Scaling that run is queue time,
> not engineering."

**Do not overclaim here.** This is the single easiest place to get caught inflating.

### A8 · "QAE and QBN are dead code padding your repo." ★★★
> "Both are wired into their engine, and both are inert because no trained artifact ships —
> our evidence page says exactly that. The QBN specifically returns `None` when untrained,
> because its constructor default is six unfitted constants and serving those as a quantum
> inference would be dressing up hardcoded numbers as a learned model."

That last clause usually earns credit rather than losing it.

### A9 · "Is the quantum part even load-bearing, or is it decoration on a CNN?" ★★★
> "Vision is a classical DenseNet-121 — we're explicit about that boundary. The quantum
> layer is the *fusion* stage: it takes the 8-channel evidence vector to a posterior over
> six diagnoses, and it's the served backend. If you switch it off you lose the measurement
> budget and the entanglement telemetry entirely."

### A10 · "Why should I believe any of these numbers?" ★★★
> "Because none of them are typed by hand. Every figure on `/quantum` is fetched from
> `/v1/quantum/evidence`, which reads the artifact each generating script wrote. If an
> artifact is missing, the page says 'not measured' — it never shows a default. And a test
> fails the build if the published tables drift from `benchmark.json`."

Then show `test_doc_numbers.py`. Its docstring says *"the fix is to re-copy the numbers out
of benchmark.json, never to loosen this test."*

### A11 · "Your QKL quantum kernel loses too." ★★
> "It's now statistically indistinguishable: ΔAUROC −0.0051, p = 0.27. Earlier it was
> significantly worse and we published that too. Note the ECE went the other way and we
> tested that as well — Δ p = 0.998, also not significant. We report the null result on the
> metric that would have flattered us."

### A12 · "What's your actual clinical validation?" ★★
Answer honestly: MIMIC-CXR labels from a **rule-based labeler**, validated against 66
hand-read reports (macro F1 0.89, κ 0.86), cross-checked against torchxrayvision — with
**nodule agreement below chance and flagged unreliable** in the model card. No prospective
clinical trial. Say that plainly.

---

## 5. Landmines — never say these

| ✗ Never say | ✓ Say instead |
|---|---|
| "quantum advantage" / "quantum speed-up" | "a different resource model" |
| "our quantum model outperforms classical" | "they're statistically indistinguishable at n=69" |
| "entanglement improves our results" | "the ablation says the ring isn't earning its gates" |
| "validated on patients" | "validated against a rule-based labeler + 66 hand-read reports" |
| "we benchmarked against nnU-Net" | "we cite published nnU-Net values; we did not run it" |
| "production-ready" / "clinical-grade" | "runs offline on one box; not a cleared device" |
| "99% accurate" (any inflated figure) | quote the artifact |

**The brain Dice trap:** AURA's Dice is **pooled 2-D per-slice**; BraTS literature values are
**per-case 3-D**. Pooled-2-D flatters. If a judge compares them, correct it *yourself* —
`docs/benchmark_report.md` carries the warning, so pointing at it converts a catch into a
credibility win.

---

## 6. If you have time before the demo

Highest value first:

1. **Re-run `train_qkl` and the sweep if anything was retrained.** The QKL numbers already
   moved once (ΔAUROC −0.0156 p=0.002 → −0.0051 p=0.27). Present stale numbers and A10
   collapses.
2. **More hardware cases.** n=1 → n=20 turns A7 from a concession into a result. Queue time,
   not engineering.
3. **Train the QBN, or delete it.** Inert-but-wired is defensible; trained is better.
4. **Widen the fusion split beyond n=69.** Nothing short of a ~15-point gap is detectable
   at that size, so the benchmark can only ever say "comparable."

---

## 7. Pre-demo check (run this the morning of)

```bash
E:\AURA\venv\Scripts\python.exe -m deploy.preflight --skip-brain
```

Expect: every served artifact PASS, and a WARN on authentication (correct for a local demo).
Then start the server and open `/app` and `/quantum`:

```bash
E:\AURA\venv\Scripts\python.exe -m uvicorn aura.gateway.app:app --host 127.0.0.1 --port 8000 --app-dir E:\AURA\aura-main
```

Hard-refresh once (`Ctrl+Shift+R`) so no stale JS is cached, upload a chest X-ray, and
confirm the measurement-budget panel renders on **Assess**. If it doesn't, the fusion
backend fell back to classical — check `preflight` output before presenting.

---

## 8. The closing line

> "We could have claimed a quantum advantage. The measurements didn't support one, so we
> published the measurements instead — including the ablation that says our own entangler
> isn't earning its gates. What we built is a system that knows the difference between
> needing more computation and needing a doctor. That distinction doesn't exist classically,
> it runs on every study, and every number behind it regenerates from an artifact you can
> open."
