# AURA — Build Status Report
### What is actually built, what is partial, and what is not built yet

*Generated from a direct read of the codebase (not the pitch). Companion to
[`aura/docs/ARCHITECTURE.md`](aura/docs/ARCHITECTURE.md) and
[`aura/docs/QUANTUM_STACK.md`](aura/docs/QUANTUM_STACK.md).*

**How to read the status column**

| Mark | Meaning |
|---|---|
| ✅ | **Built & running** today, offline, on CPU — exercised by the live console |
| 🟡 | **Partial / proxy** — works, but is a simpler stand-in than the name suggests (read the caveat) |
| ⚪ | **Designed, not built** — specified in the docs, no code yet |
| 🔭 | **Vision** — needs fault-tolerant hardware or real-world scale; product must never depend on it |

The honest one-liner: **the classical reasoning pipeline end-to-end is real and running; one of six quantum services (evidence fusion) is real; the other five quantum services are designed but not implemented; everything is trained on synthetic data with no tests yet.**

---

## 1. Snapshot

| Subsystem | Status | What's real | File |
|---|---|---|---|
| FastAPI gateway + `/v1` API | ✅ | health, cases, case detail, feedback, sign, simulate, similar, models, admin | [`aura/gateway/app.py`](aura/gateway/app.py) |
| Pipeline orchestration (7 engines, event bus) | ✅ | in-process, event-emitting, contract-typed | [`aura/gateway/pipeline.py`](aura/gateway/pipeline.py) |
| Vision → findings | 🟡 | per-finding **logistic regressions over hand-crafted features** (not a CNN) | [`aura/services/vision/engine.py`](aura/services/vision/engine.py) |
| Evidence encoding (8 channels) | ✅ | vision + priors → [0,1]⁸ vector | [`aura/services/fusion/evidence.py`](aura/services/fusion/evidence.py) |
| **Quantum fusion (Q1)** | ✅ | 8-qubit PennyLane VQC + linear head; shot-noise `posterior_std` | [`aura/services/fusion/quantum.py`](aura/services/fusion/quantum.py) |
| Classical fusion twin | ✅ | Bayesian product-of-experts (the honest baseline) | [`aura/services/fusion/classical.py`](aura/services/fusion/classical.py) |
| Backend switch + auto-fallback | ✅ | config flag; falls back to classical if quantum artifact missing | [`aura/services/fusion/engine.py`](aura/services/fusion/engine.py) |
| Safety: temperature scaling | ✅ | 1-D NLL fit | [`aura/services/safety/calibration.py`](aura/services/safety/calibration.py) |
| Safety: conformal set | ✅ | split-conformal, **marginal** 90% coverage | same |
| Safety: epistemic uncertainty | 🟡 | **MC input-perturbation proxy**, not a real deep ensemble | [`aura/services/safety/engine.py`](aura/services/safety/engine.py) |
| Safety: aleatoric / OOD / abstention | ✅ | entropy / energy z-score / 4-reason abstention policy | same |
| Explainability: saliency | ✅ | model-agnostic occlusion saliency | [`aura/services/explain/engine.py`](aura/services/explain/engine.py) |
| Explainability: attribution + counterfactuals | 🟡 | **leave-one-out** (the "Shapley-style" claim is an approximation) | same |
| Missing-evidence recommender | 🟡 | **greedy single-action** EIG over a 5-item catalog (not panel/QUBO) | [`aura/services/recommend/engine.py`](aura/services/recommend/engine.py) |
| Grounded report | ✅ | structured template + evidence-grounding map (no LLM) | [`aura/services/report/engine.py`](aura/services/report/engine.py) |
| Case memory / similarity | 🟡 | **in-memory cosine** over embeddings, rebuilt each boot | [`aura/services/memory/engine.py`](aura/services/memory/engine.py) |
| Benchmark harness (quantum vs classical) | ✅ | 6 metrics, held-out, writes `benchmark.json` | [`aura/ml/evaluation/benchmark.py`](aura/ml/evaluation/benchmark.py) |
| Persistence + audit ledger | ✅ | SQLite: cases, feedback, append-only `audit_log` | [`aura/gateway/storage.py`](aura/gateway/storage.py) |
| Model registry | 🟡 | version listing only (minimal) | [`aura/services/models/registry.py`](aura/services/models/registry.py) |
| Web experience (landing + console) | ✅ | zero-dependency SPA | [`aura/apps/web/`](aura/apps/web) |
| Portable pitch prototype | ✅ | single self-contained HTML, offline | [`presentation/AURA_Prototype.html`](presentation/AURA_Prototype.html) |
| CLI (`train` / `bench` / `serve` / `demo`) | ✅ | one entry point | [`aura/aura_cli.py`](aura/aura_cli.py) |
| Training data | 🟡 | **synthetic generator** with controlled ground truth (no real scans) | [`aura/ml/data.py`](aura/ml/data.py) |
| Quantum services **Q2–Q6** | ⚪ | belief / similarity-kernel / QAOA-planner / trajectory / uncertainty — **designed, no code** | [`aura/docs/QUANTUM_STACK.md`](aura/docs/QUANTUM_STACK.md) |
| Automated tests / CI | ⚪ | **none** (pytest is a dependency but there are no test files) | — |

---

## 2. What is genuinely built (the real system)

A study flows through the full pipeline end-to-end, entirely offline on CPU, and every value shown in the console is live model output:

1. **Vision** scores 7 findings and emits an evidence embedding.
2. **Evidence encoding** turns findings + patient priors into an 8-channel vector.
3. **Quantum fusion** (8-qubit VQC) produces a posterior over 6 diagnoses and a finite-shot uncertainty; a **classical product-of-experts twin** runs beside it.
4. **Safety** applies temperature scaling, builds a conformal set with a marginal coverage guarantee, estimates epistemic/aleatoric/OOD, and **abstains** when out of envelope (4 explicit reasons).
5. **Explainability** produces occlusion saliency over the image and per-channel attribution + counterfactuals.
6. **Recommender** ranks the next best test by expected information gain per cost·risk.
7. **Report** writes a grounded findings/impression/recommendation, each sentence traced to evidence.
8. **Feedback + sign-off** are recorded to a SQLite **audit ledger**; the worklist is triaged by uncertainty.

Plus: a benchmark harness that prints the quantum-vs-classical table on the same held-out split, a live "simulate a study" endpoint, a cinematic web console, and a portable single-file demo. **This is a real, coherent, working product** — the rest of this report is about being precise on where it stops.

---

## 3. Partial / proxy — things that are real but simpler than they sound

These work in the demo but a sharp judge will (rightly) probe them. Say the honest version first.

- **Vision is a feature model, not a CNN.** Per-finding logistic regressions over hand-crafted anatomical features, with a heuristic fallback. It is *model-agnostic by contract* (which is what keeps occlusion saliency valid), so a real CNN drops in behind the same interface — but that CNN is not written yet. `torch` is optional and unused in the demo path.
- **Epistemic uncertainty is a proxy, not a deep ensemble.** It Monte-Carlo perturbs the *evidence vector* and measures posterior spread (`engine.py` comment says so explicitly). A true deep ensemble / MC-dropout over model weights is not implemented.
- **Attribution is leave-one-out, not true Shapley.** Each channel is zeroed and the probability delta is measured — a first-order approximation the README calls "Shapley-style."
- **The recommender is greedy and single-action.** It ranks individual tests; it cannot see redundancy between tests (the QUBO/QAOA "panel" planner, Q4, is designed but not built).
- **Conformal coverage is marginal, not conditional.** 90% coverage on average, not per-class — group-balanced (Mondrian) conformal is roadmap.
- **Memory is in-memory cosine.** Similarity works within a running session (index rebuilt on boot); there is no persistent/scalable ANN index, and the quantum fidelity kernel (Q3) is not built.
- **The "learning loop" records but does not retrain.** Feedback verdicts/corrections and abstention rates are stored and surfaced; there is **no automated nightly recalibration** — retraining is a manual `aura_cli train`.
- **All numbers are on synthetic data.** The generator has controlled ground truth (deliberately, so calibration/coverage can be *measured* against known truth), but there is no real-scan validation.

---

## 4. Not built — designed but no code

Specified in [`aura/docs/QUANTUM_STACK.md`](aura/docs/QUANTUM_STACK.md) and `PRODUCT_V2.md`, with zero implementation today:

- **Q0 — trained quantum feature-map registry** ⚪ — only the *static* `RY(π·x)` encoding exists; trainable data-re-uploading maps are not implemented.
- **Q2 — quantum belief engine** ⚪ — density-matrix beliefs, coherence heatmap, order-sensitivity/anchoring flag. The flagship idea; no code.
- **Q3 — quantum similarity kernels** ⚪ — fidelity-kernel re-ranking of neighbours. Memory is cosine only.
- **Q4 — QAOA diagnostic planner** ⚪ — panel selection as QUBO; the current recommender is greedy.
- **Q5 — quantum trajectory engine** ⚪ — disease-progression / longitudinal modelling. No trajectory service exists.
- **Q6 — quantum uncertainty service** ⚪ — scenario (Born) sampling and adaptive shot budgeting; only shot-variance `posterior_std` exists.

Product/infra not built:

- **No automated tests and no CI.** Highest-value engineering gap.
- **No DICOM/PACS integration** (positioned as roadmap).
- **No real authentication** — header-based principal stub; RBAC/OIDC seam is marked but empty.
- **No retrospective validation** on CheXpert / MIMIC-CXR (roadmap).
- **No multimodal evidence** — labs, genomics, notes; fixed at 8 imaging-derived channels.

## 5. Vision — needs hardware or real-world scale 🔭

Correctly labelled `[C]` in the quantum stack and never depended on: amplitude-estimation speedups (EIG, rollouts), Grover retrieval at federated scale, quantum walks on ontologies, large multimodal feature maps, federated learning, hardware kernels, digital twins, ICU real-time streaming.

---

## 6. The benchmark, honestly

From [`aura/artifacts/benchmark.json`](aura/artifacts/benchmark.json) (n=100 held-out, **synthetic**):

| metric | quantum | classical | better |
|---|---|---|---|
| accuracy | **0.96** | 0.93 | higher |
| NLL | **0.093** | 0.488 | lower |
| ECE (calibration) | **0.020** | 0.276 | lower |
| Brier | **0.060** | 0.204 | lower |
| conformal coverage | 0.92 | 0.92 | ≈0.90 |
| conformal set size | **0.94** | 0.98 | lower |

Reproducible via `py -m aura_cli bench`. **Caveat to state out loud:** this is synthetic data, and at 8 qubits the quantum circuit is trivially classically simulable — the claim is *representational structure and calibration discipline*, not speedup. The calibration gap (ECE 0.020 vs 0.276) is the real, defensible headline.

---

## 7. Gaps a judge will probe — and the honest answer

| Likely question | Straight answer |
|---|---|
| "Is this a real CNN?" | No — feature-based detectors behind a swappable interface; a CNN is a drop-in, not yet written. |
| "Deep-ensemble epistemic uncertainty?" | It's an input-perturbation proxy; a true ensemble is future work. |
| "Real Shapley values?" | Leave-one-out attribution — an approximation, labelled as such. |
| "Trained on real scans?" | No — synthetic with known ground truth, chosen to *measure* calibration; validation on public cohorts is roadmap. |
| "Where are the tests?" | None yet — the top engineering priority. |
| "Six quantum services?" | One (fusion) runs and is benchmarked; five are designed and labelled as such — that honesty is deliberate. |

---

## 8. Suggested priorities (if continuing)

1. **Add a test suite + CI** — even a dozen contract/pipeline tests; biggest credibility-per-hour.
2. **Wire the feedback → recalibration loop** so the "learning" claim is literally true.
3. **Build Q2 (belief coherence) or Q4 (QAOA panel)** — the two most demo-able quantum services with the smallest circuits.
4. **Swap in a real CNN** behind the vision interface and re-run the benchmark.
5. **Retrospective validation** on a public CXR cohort to replace the synthetic-data caveat.
6. **Real auth + DICOM/PACS seam** for any pilot conversation.

---

*Legend: ✅ built · 🟡 partial/proxy · ⚪ designed, not built · 🔭 vision. Every ✅/🟡 row above was verified against the file it links to.*
