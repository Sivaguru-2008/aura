# Deck patch — put the measured quantum results on stage

**Why this exists.** `AURA_QuantAThan_2026.pptx` was built **19 Jul**. The quantum evidence
pack was measured **23–28 Jul**. So the deck is honest in *framing* but silent on *outcome*:

- Slide 18 promises the benchmark law — *"If the classical twin wins, the classical twin ships
  — and we say so on stage."* **The deck never says which one won.** The first judge to ask
  "so which ships?" is asking a question no slide answers.
- The two results that are genuinely yours and genuinely quantum — the **measurement-budget
  saving** and the **real QPU run** — are **not in the deck at all**.

At a quantum hackathon, the current deck's strongest quantum claim is *representational*
(slide 7: "a richer similarity geometry… not 'quantum is faster'"). That is a good claim, but
it is a *conjecture-backed* one. You now have two **measured** ones. Put them on stage.

Every number below is quoted from a shipped artifact — the source file is named on each line.
Nothing here is new analysis; it is transcription.

---

## Patch 0 — a live contradiction to resolve *before* you present

**Slide 18 says:** *"If the classical twin wins, the classical twin ships."*
**`aura/pyproject.toml:44` says:** `fusion_backend = "quantum"   # SERVED backend`
**`benchmark.json` says:** the classical twin won on accuracy, ECE and AUROC.

So the deck states a law the shipped system does not follow. A judge who opens
`pyproject.toml` — and at a quantum event, one will — can ask *"your slide says classical
ships; your config serves quantum. Which is true?"* That question is only dangerous if it
surprises you.

**The design itself is fine and defensible.** `FusionEngine._resolve()` serves the VQC, and
the Wasserstein conflict guard is armed **only** when quantum serves, deferring to the PoE
twin when the two posteriors diverge. `backend_calibration.json` carries a **separate**
temperature, conformal q̂ and OOD statistics for each backend (quantum T = 0.9057, classical
T = 0.479 — a 1.89× spread), so serving quantum is *not* miscalibrated. What is wrong is only
the sentence on slide 18.

**Two ways out — pick one and be consistent everywhere:**

| | Change | Consequence |
|---|---|---|
| **A. Fix the words** *(recommended)* | Patch B below; keep `fusion_backend = "quantum"` | Nothing about the system changes. You describe what you actually built: quantum serves, classical is the calibrated reference and the fallback. Keeps the quantum layer in the serving path — which matters at this event. |
| **B. Fix the config** | Set `fusion_backend = "classical"` | The slide-18 law becomes literally true, but the VQC leaves the serving path *and the conflict guard disarms with it* (`_guard_enabled` requires `backend == "quantum"`). Re-run the demo end-to-end before committing to this — do **not** change it on the morning of. |

I did not change the config: which backend serves is a product decision and it moves live
behaviour. Option A needs only the wording patch below.

---

## Patch A — NEW slide, inserted immediately after slide 7 (Quantum Core)

**Title:** `QUANTUM — THE SCORECARD`
**Kicker:** `03 ·` style to match; **Purpose line:** *"Purpose — state the outcome of the
benchmark law, not just the law."*
**Layout:** 2×2 cards, same Panel `#F6FAFA` / Border `#C6E4E1` treatment as slide 13.
**Timing:** 0:45. **Build:** cards 1→2→3→4 on separate clicks.

### Card 1 — `THE FAIR FIGHT` (border: Warn red `#C05B52`)
> Each backend temperature-scaled on **its own** calibration split. n = 69 held-out.
>
> | | acc | ECE | AUROC |
> |---|---|---|---|
> | classical PoE | **0.6957** | 0.2194 | **0.7875** |
> | quantum VQC | 0.6377 | 0.2381 | 0.7696 |
>
> **Verdict: on accuracy, the classical twin wins — and AURA reports it.**
> *The VQC still serves, with the classical twin as its calibrated reference and the
> conflict-guard fallback. Both are calibrated on their own logits — see Patch 0.*

`aura/artifacts/benchmark.json` → `metrics_full`

### Card 2 — `ENTANGLEMENT: A NEGATIVE RESULT WE KEPT` (border: Warn red `#C05B52`)
> Entangled ring vs. product state — same trainer, same seed, same data,
> **same 102 parameters**. n = 447 patient-disjoint, 2 000 bootstrap.
>
> **ΔNLL = +0.056, 95% CI [0.022, 0.090] — excludes zero.**
> Entanglement is **significantly worse** on NLL.
> Δaccuracy and ΔECE: not significant.
>
> *The CNOT ring is not earning its place on an 8-channel evidence vector.*

`aura/artifacts/quantum_study.json` → `q1_q2_ablation.entanglement_effect`

### Card 3 — `QMBA — 64× FEWER MEASUREMENTS` (border: Neon teal `#2DD4BF` — this is a win)
> Sequential-margin measurement, 447 studies.
> Median **128 shots** spent against the **8 192-shot** per-circuit device ceiling → **64×**
> at the median. 86.4 % commit / 13.6 % abstain.
>
> And it **types** the abstention:
> **1 measurement-limited** (more shots would resolve it) vs **60 model-limited**
> (tied at infinite precision → needs a human).
>
> **No classical fusion model can draw that distinction — it has no measurement budget to vary.**

`aura/artifacts/quantum_study.json` → `q3_measurement_budget`;
ceiling from `aura/services/fusion/qmba.py:DEFAULT_MAX_SHOTS`

### Card 4 — `IT RAN ON REAL HARDWARE` (border: Neon teal `#2DD4BF`)
> **IBM `ibm_marrakesh`**, 156 qubits · job `d9js49rjf64c739haeg0`
> 8 qubits · 3 layers · **logical** circuit depth 31 · 24 two-qubit gates
> *(as-built depth from `local_check`, **not** the transpiled-to-device depth — if a judge
> asks for the routed depth on a 156-qubit heavy-hex lattice, say you didn't record it.)*
>
> Device noise attenuated every ⟨Z⟩ (mean abs error **0.186** vs analytic) —
> **and the top-1 diagnosis survived anyway**: `pneumothorax_dx`,
> agreeing with the simulator **and** with the true label.

`aura/artifacts/ibm_hardware_run.json`

### Footer caveat — **put this on the slide, in Muted `#5B6672`, 10 pt**
> Cards 2–3 are statevector simulation (`default.qubit`). Card 4 is one case
> (test index 144) on a real QPU. We do not blend the two.

> **Why the caveat is not a weakness:** the quantum jury will look for exactly this conflation.
> Pre-empting it is the same move that wins Q23 in the battle card.

---

## Patch B — slide 18 (appendix, benchmark law) — **wording must change, see Patch 0**

The slide currently promises: *"If the classical twin wins, the classical twin ships — and we
say so on stage."* The classical twin won on accuracy, and the VQC still serves. As written,
the slide states a law the system does not follow.

Replace that line with the truthful version:

> **The classical twin won on accuracy — and we say so on stage.**
> The VQC serves; the PoE twin is its calibrated reference and the conflict-guard fallback.
> Each backend is calibrated on **its own** logits — `backend_calibration.json`.

---

## Patch C — speaker notes

**Slide 7 (Quantum Core) — replace the last sentence of the note.**
Currently ends: *"…every quantum service ships beside a classical twin and must beat it
head-to-head — next slides show how."*

Replace with:
> "…every quantum service ships beside a classical twin and must beat it head-to-head. Next
> slide: it didn't, and I'll show you the scorecard — including the two places where the
> quantum layer *does* pay for itself."

**New slide 7b — note (0:45):**
> "Here is the scorecard, and I'm going to start with the one that goes against me. Fair fight,
> each backend calibrated on its own split: the classical twin wins on accuracy and AUROC. So
> the classical twin is what AURA serves. I promised on the previous slide that I'd say that
> out loud, so there it is. Second: we ablated entanglement — same parameter count, same seed —
> and the entangled ring is significantly *worse* on NLL. We kept that result and shipped it.
> Now the two that go my way. QMBA: by measuring adaptively instead of at a fixed budget, the
> median study resolves in 128 shots against an 8 192-shot device ceiling — sixty-four times
> fewer measurements. And it tells you *which kind* of uncertainty you hit: one case was
> measurement-limited, sixty were model-limited — genuinely tied, send them to a human. A
> classical model cannot make that distinction, because it has no measurement budget to vary.
> That is my actual quantum advantage claim, and it's measured, not conjectured. Finally: this
> circuit ran on ibm_marrakesh, 156 qubits, job ID on the slide. Noise flattened every
> expectation value — and the top-1 diagnosis survived anyway."
>
> **JUDGES SHOULD FEEL:** this team reports losses and wins in the same voice, and the win is
> the kind you only find by building the thing.

**Slide 16 (closing) — the note currently ends** *"…claimed honestly and benchmarked
ruthlessly."* Keep it. It now cashes a cheque the deck actually wrote.

---

## Patch D — playbook timing

The new slide costs 0:45. Slides 14–15 (Roadmap + Impact) are already marked *"cut both if
running long"* in the playbook — cutting one covers it.

**Revised 5-minute spine:** 2 → 3 → 5 → 7 → **7b** → 10 → 11 → 16.
The scorecard belongs in the 5-minute cut. At a quantum event it is the slide that decides
whether they believe the rest.

---

## What NOT to change

- **Don't soften slide 7.** The representational claim (Havlíček / Liu) is correctly hedged and
  the notes already say "not 'quantum is faster'". It is the right setup for the scorecard.
- **Don't claim quantum advantage on accuracy** anywhere. You don't have it, the artifact says
  so, and the battle card's "five things to never say" already rules it out.
- **Don't merge cards 3 and 4.** Simulator and hardware are different evidentiary classes.
