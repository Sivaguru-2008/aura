# AURA — Presenter Playbook
### RIT Quant-a-than 2026 · Sivaguru R.M

The deck: `AURA_QuantAThan_2026.pptx` (19 slides: 16 main + 3 appendix).
Every slide's speaker notes contain the **script**, **timing**, **what judges should feel**, and **animation cues**. This document is everything that doesn't fit in notes.

---

## 1. Design system (if you edit slides, stay inside this)

| Token | Value | Use |
|---|---|---|
| Ink | `#0F172A` | titles, primary text on white |
| Muted | `#5B6672` | secondary text, captions |
| Deep teal | `#0E7490` | kickers, emphasis text, equations |
| Mid teal | `#0D9488` | arrows, node borders |
| Neon teal | `#2DD4BF` | bars, glows, dark-slide accents |
| Bright neon | `#5EEAD4` | dark-slide headline accent |
| Pale teal | `#ECF8F7` | highlighted cards |
| Panel | `#F6FAFA` | standard cards |
| Border | `#C6E4E1` | card outlines |
| Dark bg | `#05090E` | cinematic slides 2 & 16 |
| Warn red | `#C05B52` | "the gap" annotations only |
| Gold | `#B98A2F` | "partial" marks in comparison table only |

**Fonts:** Arial (all UI text — ships everywhere, zero substitution risk), Cambria (equations only — its math glyphs render correctly on any Windows/Office machine), Courier New (CLI commands).
If you have IBM Plex Sans / SF Pro installed on the presentation machine, you may swap Arial titles for them — but verify on the *actual* venue machine first.

**Rules the deck follows:** one idea per slide · no bullet paragraphs · every slide has a diagram · citations at the bottom of every technical slide · dark slides only for open/close (the "sandwich").

---

## 2. Timing plan (10:00 total, leaves buffer in a 12-min slot)

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | Opening cinematic | 0:25 | 0:40 |
| 3 | The trust gap | 0:50 | 1:30 |
| 4 | The reasoning gap | 0:40 | 2:10 |
| 5 | AURA pipeline | 1:00 | 3:10 |
| 6 | Architecture | 0:40 | 3:50 |
| 7 | **Quantum core** | 1:30 | 5:20 |
| 8 | Reasoning graph | 0:45 | 6:05 |
| 9 | Explainability | 0:45 | 6:50 |
| 10 | Uncertainty | 1:00 | 7:50 |
| 11 | Evidence planner | 0:50 | 8:40 |
| 12 | Clinical loop | 0:35 | 9:15 |
| 13 | Positioning table | 0:45 | 10:00 |
| 14–15 | Roadmap + Impact | 0:35 + 0:35 | *(cut both if running long — say one sentence each)* |
| 16 | Closing | 0:25 | — |

**If given only 5 minutes:** slides 2 → 3 → 5 → 7 → 10 → 11 → 16. That's the spine.

---

## 3. Animation & transition spec

Transitions: the template's **Fade** is already set on all slides — do not add anything else. Within slides (add in PowerPoint > Animations; all durations ≤ 0.5 s, no bounce/spin ever):

| Slide | Build |
|---|---|
| 2 | headline line 2 fades in on click |
| 3 | three stat cards wipe left→right (one click), red gap box on second click |
| 4 | pipeline draws left→right; red box + ✕ items on click |
| 5 | row 1 wipes in → down arrow → row 2 right-to-left (mirrors the snake) |
| 6 | layers stack bottom-up; quantum layer last with a subtle pulse |
| 7 | left pipeline steps appear downward; right cards on separate clicks |
| 8 | evidence nodes → edges draw → posterior bars grow (0.4 s) |
| 10 | equation first; three cards; conformal card last |
| 11 | equation → bars grow staggered → recommendation chip pops |
| 12 | loop arrows animate clockwise once, then stop |
| 13 | baseline columns together; AURA column + outline sweep in last |
| 16 | three lines on three clicks, 0.6 s fades — the only theatrical slide |

---

## 4. Judge Q&A — anticipated questions & strong answers

### Quantum (professors / IBM-aligned researchers)

**Q: "Isn't 8 qubits trivially classically simulable? Why call it quantum at all?"**
A: Completely — a 256-dim statevector simulates in microseconds, and we say so. The point at this scale is *representational structure*, not supremacy: the entangling feature map builds pairwise evidence interactions into phase structure, and the kernel is a state fidelity — a similarity measure with different geometry than any polynomial classical kernel. Our architecture is device-portable: the same circuit spec runs on `default.qubit` today and on hardware backends when scale demands it. And critically, we benchmark against a classical twin — if the classical twin wins, we ship the classical twin. We're building the *discipline* for the day the crossover happens.

**Q: "Havlíček's hardness is a conjecture. Liu et al.'s speedup is for a contrived DLP problem. Aren't you overclaiming?"**
A: That's exactly why slide 7 says "conjectured" and "for a related family," and why our claim tier system (docs/QUANTUM_STACK.md) labels this [B]-Grounded: a representational claim backed by literature, never a speed claim, benchmarked anyway. If a classical surrogate dequantizes our circuit — that's a result, and we'd ship the surrogate.

**Q: "Why angle encoding and not amplitude encoding?"**
A: Amplitude encoding packs 2ⁿ features into n qubits but needs deep state-preparation circuits and destroys per-feature interpretability. Angle encoding is NISQ-shallow, keeps one clinical channel per qubit (auditable — a hospital requirement), and lets the entangling layers, not the encoding, carry the interaction structure.

**Q: "Barren plateaus?"**
A: Real concern for deep VQCs. We're at 8 qubits with shallow entangling layers — well below the regime where gradient variance vanishes (Cerezo et al. 2021). Parameter-shift gradients are exact, and we monitor gradient norms during training as a standing check.

**Q: "Why put quantum in the reasoning layer and not the imaging layer?"**
A: Pixels are high-dimensional and CNNs already won imaging — encoding a 10⁶-pixel image into a circuit is neither feasible nor useful. Clinical *reasoning* state is the opposite regime: 8–16 evidence channels, 6–32 hypotheses — compressed, structured, correlation-rich, low-dimensional. That is exactly what today's simulators and early fault-tolerant machines can address. We never put pixels in a circuit.

### AI / ML (AI researchers)

**Q: "How is the multi-agent Bayesian part different from just an ensemble?"**
A: An ensemble averages predictions of one task. Our agents own different *epistemic roles* — vision evidence, prior/context, precedent similarity, calibration — and exchange messages over a typed belief graph, so the posterior is a product of separately auditable likelihoods. You can ask "which evidence moved the belief?" — you cannot ask that of an averaged softmax.

**Q: "Conformal prediction guarantees are marginal, not conditional. A per-patient '90%' can be misleading."**
A: Correct — the guarantee is marginal coverage. That's why we present sets alongside decomposed uncertainty rather than as a per-patient probability, and why our roadmap includes group-balanced (Mondrian) conformal calibration by pathology class, which restores class-conditional validity where it clinically matters.

**Q: "What data did you train on? Synthetic data is a red flag."**
A: For the hackathon build, a synthetic generator with controlled ground truth — chosen deliberately so we can *measure* calibration and coverage against known truth, which real datasets don't give you cleanly. The vision encoder is replaceable by contract; the validation roadmap is retrospective evaluation on CheXpert/MIMIC-CXR cohorts.

**Q: "How do you know your ECE numbers mean anything?"**
A: They come from `aura_cli bench` — a fixed harness, same held-out split for quantum and classical paths, six metrics (accuracy, NLL, ECE, Brier, coverage, set size), seeds pinned. Every number is reproducible from the provenance record.

### Clinical / regulatory (professors, founders)

**Q: "Radiologists won't read all this. Doesn't more output mean more burden?"**
A: The default view is one line: differential, confidence class, and the recommended next step. The four artifacts are on-demand drill-down. And uncertainty-triaged worklists *save* attention — confident-normal studies deprioritized, uncertain-critical fast-tracked.

**Q: "Regulatory path?"**
A: AURA is positioned as clinical decision *support* — the radiologist signs every report, and the abstention policy means the system withholds rather than guesses. The audit ledger (every claim → evidence artifact → provenance record) is designed with FDA's Good Machine Learning Practice principles in mind. Adaptive components (nightly recalibration) fit the Predetermined Change Control Plan framework — the update path is itself specified.

**Q: "Who pays? What's the business?"**
A: Radiology backlog is the wedge: uncertainty-triaged worklists sell to hospital groups on throughput, not on diagnosis. India's 1:100k radiologist ratio makes tele-radiology triage the beachhead market; the EIG planner cuts unnecessary CT/MRI spend, which payers notice.

**Q: "What's actually running vs. vision?"**
A: Appendix A2 draws the line explicitly. Running today, offline: VQC fusion + classical twin, conformal + temperature calibration, ensembles/MC-dropout, energy-score OOD, occlusion saliency, EIG recommender, grounded reports, dashboard, audit ledger. Vision (labeled): hardware kernels, federated learning, digital twins, ICU streaming.

### The trap question

**Q: "Would this have worked without quantum?"**
A: The uncertainty engine, planner, and reports — yes, and they run classically today; that honesty is load-bearing for us. What the quantum layer adds is a similarity geometry over evidence that classical kernels don't reproduce, and a Bayesian-update formalism (density matrices, measurement update) that *matches the structure* of sequential evidence fusion. We keep it because it earns its place on the benchmark — and the day it doesn't, the architecture drops it without a rewrite. That's what "quantum is not marketing" means operationally.

---

## 5. Demo insurance

- `cd aura && py -m aura_cli train && py -m aura_cli bench && py -m aura_cli serve` → dashboard at `http://localhost:8000`. Rehearse once on the venue machine; it's offline-capable by design.
- Pre-record a 60-second screen capture of the dashboard as backup.
- If asked live: run `bench` — the quantum-vs-classical table printing in front of judges is the single most credible artifact you have.

## 6. Pre-flight checklist

- [ ] Replace institution on slide 1 if you're not presenting under RIT.
- [ ] Open the deck on the venue machine — check slides 7, 10, 17 (equation glyphs ⟨⟩, ρ, ⊗, q̂) render correctly.
- [ ] Add the PowerPoint animations from §3 (10 minutes of work) — they're specified but not baked in, so the deck degrades gracefully if you skip them.
- [ ] Presenter view on; notes font enlarged.
- [ ] Water. Slide 7 is a 90-second monologue.
