# AURA v2 — The Diagnostic Trajectory Operating System
### Product & System Redesign (founding team, full session)

> Status: **STRATEGY — extends `ARCHITECTURE.md` v1. Nothing in the running P0 demo is
> discarded; every v1 engine becomes a kernel service of v2.**
>
> v1 tagline: *"The copilot that knows what it doesn't know."*
> v2 tagline: **"The OS that never loses a patient's thread."**

---

## 0. The Big Question, answered first

**If every hospital in the world already had image classification AI, why would they still buy AURA?**

Because classification AI makes the real problem *worse*, not better.

A classifier answers one question in three seconds: *what is in this image?* A diagnosis is
not an event — it is a **process** that runs for three weeks to three years, crosses six
clinicians, four IT systems, two institutions, and a dozen handoffs. The classifier fires at
minute zero of that process and then disappears. Every finding it detects **opens a loop**:
a follow-up to schedule, a differential to resolve, a test to order, a result to reconcile.
Perception AI is an industrial-scale *loop-opening machine* installed in hospitals that have
no system for *closing* loops.

The measurable result, from the diagnostic-safety literature (order-of-magnitude figures,
US, presented as literature ranges, not our claims):

- ~**795,000 Americans die or are permanently disabled every year from diagnostic error**
  (BMJ Quality & Safety, 2023, Newman-Toker et al.) — the largest patient-safety problem in
  medicine.
- Diagnostic error is the **single largest category of paid malpractice claims** (~$4–5B/yr).
- **30–70% of imaging follow-up recommendations are never completed** (incidental findings
  studies); every lost lung nodule is simultaneously a potential six-figure claim and
  $20–50k of downstream care the hospital never delivers.
- Most of these failures are **not perception failures**. The finding was seen. It was in
  the report. The *process* failed: nobody owned the next step, the context died at a
  handoff, the differential closed prematurely, the disconfirming evidence was never
  gathered.

So the answer to the big question, in one line:

> **Classifiers are peripherals. AURA is the operating system.** Hospitals that buy more
> perception AI get more findings, more alerts, and more open loops — and still have no
> system of record for what the care team believes, how sure they are, what is missing,
> and who owes the next action. AURA is that system. Every classifier sold makes AURA
> more necessary, not less.

---

## 1. New Product Vision

### 1.1 The category

**AURA is the system of record for clinical reasoning.** EHRs record what was *done*
(orders, notes, billing). PACS records what was *seen* (pixels). **Nothing records what was
*thought***: the differential, the calibrated confidence, why this test was ordered, what
would change the plan, and what must not be forgotten. Clinical reasoning today lives in
clinicians' working memory and dies at every shift change, service transfer, discharge, and
referral — it is re-serialized into free-text prose and lossily re-derived by the next
reader, or not re-derived at all.

AURA v2 externalizes that working memory as a first-class, auditable, longitudinal data
structure — and then *operates* on it, the way an OS kernel operates on processes:

| OS concept | AURA concept |
|---|---|
| Process | **Diagnostic loop** — one open clinical question about one patient ("is this nodule malignant?"), with a lifecycle: `open → active → resolved / escalated`, never silently killed |
| Process state | **Belief state** — calibrated posterior over hypotheses, updated on every new piece of evidence, with epistemic/aleatoric uncertainty split |
| Memory | **Evidence graph** — typed evidence nodes (present *and absent*) with provenance, spanning encounters |
| Scheduler | **Risk-ranked loop scheduler** — attention goes to the loop with the highest (harm velocity × staleness × uncertainty), not FIFO |
| Interrupts | New evidence arrives → belief update → the right loop wakes and the right clinician is paged |
| Watchdog / supervisor | **Commitment ledger** — every accepted recommendation becomes an owned, time-boxed obligation that cannot expire silently |
| Device drivers | DICOM / FHIR / HL7 adapters — **and third-party perception AI vendors**, whose outputs enter as typed evidence events |
| Syscall API | The `/v1` contract other systems call to open loops, post evidence, and query belief |
| Userland apps | Radiology cockpit (the existing dashboard), tumor-board packet builder, results manager, fleet console |

This is not a metaphor for the pitch deck; it is the actual service decomposition (§4, §6).

### 1.2 What changes from v1

v1 built **Clinical Epistemic Intelligence for a single study**: one image in, one
calibrated, explained, abstention-capable epistemic assessment out. That is the correct
kernel — and it stays byte-for-byte. v2 makes three moves:

1. **Time.** The unit of work changes from *case = one study* to *loop = one question over
   time*. The fusion engine becomes the update operator of a temporal belief filter.
2. **Accountability.** Recommendations stop being suggestions and become tracked
   commitments with owners, deadlines, and escalation. This is the piece that prevents the
   lawsuit and earns the CFO's signature.
3. **Population.** The hospital gets one screen showing every open diagnostic loop ranked
   by risk of silent failure — air traffic control for diagnosis.

### 1.3 North-star metric (revised)

v1: minutes of radiologist time saved per case.
v2: **percentage of diagnostic loops closed within their guideline window, at
equal-or-better diagnostic accuracy, with zero silent failures.** This is a number a
hospital board, a malpractice carrier, and a CMS quality officer all understand — and no
perception-AI vendor can claim it, because they don't model loops at all.

---

## 2. The Real Problem We Solve

### 2.1 The cognitive bottleneck

We studied the post-diagnosis workflow of every persona in the brief. The bottleneck is the
same everywhere, and it is cognitive, not computational: **collective working memory.**

- **Radiologist:** reads 100 studies/day; comparing priors and tracking her own "recommend
  follow-up" sentences is manual, so under load it is skipped. Her recommendation exits her
  head into prose and is never seen again.
- **Oncologist / tumor board:** a fellow spends 2–4 hours per patient re-assembling the
  story (imaging over time + path + labs + treatments) into a packet; the board decides in
  15 minutes based on whatever subset got assembled.
- **ER physician at 3am:** decides admit/discharge before the formal read, with no view of
  what is known vs. unknown; over-orders "to be safe" because uncertainty is invisible.
- **ICU / neurology:** the question is never "classify this image" — it is *"what changed,
  how fast, and is the trajectory bending toward harm?"* Answered today by scrolling.
- **Resident:** inherits a 400-page chart with zero explicit reasoning state; re-derives
  the differential from scratch — the handoff is a lossy decompression.
- **Administrator / CMO:** cannot answer "how many follow-up recommendations issued in this
  hospital in the last 90 days were actually completed?" Most track incidental findings in
  Excel, or not at all. Lung-cancer-screening programs are *legally required* to maintain
  tracking registries — and run them by hand.
- **Insurance review:** clinicians spend ~14 hrs/week justifying decisions; the evidence
  dossier a denial appeal needs is exactly the evidence graph AURA already maintains.
- **Clinical trials / research:** eligibility screening is manual chart re-derivation of
  the same reasoning state, a third time.

Same failure, ten costumes: **the state of the diagnostic process exists only in human
working memory, so it is either expensively re-derived at every step, or dangerously not.**

### 2.2 The named failure modes we target

The diagnostic-safety literature has names for the ways this kills people. Each one is a
design requirement (and §5 maps one pipeline stage to each):

| Failure mode | What happens | AURA countermeasure |
|---|---|---|
| **Lost follow-up** ("failure to close the loop") | Recommendation written, never executed | Commitment ledger + watchdog (§6.2) |
| **Premature closure** | First plausible diagnosis accepted; search stops | Belief engine flags high confidence + low evidence coverage (§6.1) |
| **Anchoring / confirmation bias** | Disconfirming evidence discounted | Evidence-verification stage surfaces contradicting nodes explicitly (§5, stage 5) |
| **Handoff amnesia** | Reasoning dies at shift change | Belief state is the handoff artifact — resumable, not re-derivable |
| **Unavailable priors** | Prior studies exist but aren't compared | Trajectory engine auto-computes deltas (§6.4) |
| **Alert fatigue** | Everything urgent = nothing urgent | Risk-ranked scheduler + abstention-first philosophy; escalation budgets (§13.5) |
| **Habit ordering / over-ordering** | Next test chosen by reflex, not information value | EIG-ranked next-best-action (exists, v1) |
| **Unwarranted variation** | Care departs from published pathway invisibly | Guideline engine with deviation flags (§6.3) |

### 2.3 Why this problem grows for a decade

1. Imaging volume grows 3–5%/yr; radiologist supply is roughly flat. Less time per study →
   more reliance on process, which doesn't exist.
2. **Perception AI increases findings per scan.** Every deployed classifier raises the
   incidental-finding rate — more open loops into a system with no loop closure. Our
   problem is the *derivative* of our competitors' success.
3. Aging, multimorbid populations → longer, more fragmented diagnostic journeys across
   more institutions.
4. Regulatory tailwind: the National Academies named diagnostic error "the blind spot" of
   patient safety; AHRQ funds diagnostic-safety centers; Leapfrog added diagnostic
   excellence standards; CMS is moving toward diagnostic-safety measures. Hospitals will be
   *scored* on exactly the number AURA manages.

---

## 3. Why Existing AI Fails

| Category (examples) | What it does | Why it doesn't solve this |
|---|---|---|
| **Perception AI** (Aidoc, Viz.ai, Lunit, Annalise) | Flags findings on single studies, fast triage alerts | Fires at minute zero, owns nothing afterward. Adds loops, closes none. Single-condition alert pipes, no belief state, no memory across studies |
| **Reporting AI** (Rad AI, Nuance DAX) | Dictation → prose faster | Accelerates the lossy serialization. The reasoning still dies in free text |
| **Chatbots / RAG on guidelines** | Answers questions when asked | Stateless and unaccountable. Doesn't know your patient exists unless you retype the case; owns no obligation; no calibration |
| **EHR modules** (Epic results routing) | Rule-based result inbox tickets | Records actions, not beliefs. No posterior, no uncertainty, no information-value ranking, no cross-study reasoning. Tickets without reasoning become spam |
| **Point results-management tools** | Regex the word "follow-up" out of reports, make a ticket | No belief state → cannot rank by actual risk; no EIG → cannot say what to do; no calibration → cries wolf; single-department scope |
| **v1 AURA (our own)** | Calibrated, explained, abstention-capable single-study epistemics | Right kernel, wrong lifespan: the epistemic state evaporates when the case is signed. v2 keeps it alive until the question is *answered* |

The structural gap all of them share: **no persistent, calibrated, accountable model of
what the care team believes and owes.** That model is AURA's core primitive, and everything
else in this document is machinery around it.

---

## 4. New System Architecture

### 4.1 Kernel view

```
                          USERLAND (apps — existing visuals, reframed)
   ┌──────────────┬──────────────────┬──────────────────┬─────────────────┐
   │ Radiology    │ Trajectory /     │ Loop Census      │ Fleet Console   │
   │ Cockpit      │ Tumor-Board View │ ("air traffic")  │ (C-suite)       │
   └──────┬───────┴────────┬─────────┴────────┬─────────┴───────┬─────────┘
          │            HTTPS / WS  (gateway: auth, RBAC, audit) │
   ═══════╪════════════════╪══════════════════╪═════════════════╪═════════
          ▼                ▼   AURA KERNEL    ▼                 ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  BELIEF STATE ENGINE (§6.1)      COMMITMENT LEDGER + WATCHDOG (§6.2)│
   │  per-loop temporal posterior      append-only, hash-chained,        │
   │  update op = Fusion Engine        owned obligations, escalation     │
   │                                                                     │
   │  LOOP SCHEDULER: priority = harm-velocity × staleness × uncertainty │
   └───────┬─────────────────────────────────────────────────────┬───────┘
           │            internal event bus (exists: common/eventbus;      │
           │            Redis Streams in production)                      │
   ┌───────┴──────────────────────────────────────────────────────┴──────┐
   │ KERNEL SERVICES (v1 engines, unchanged)   NEW SERVICES              │
   │  • vision      • safety      • explain     • guideline  (§6.3)      │
   │  • fusion (quantum/classical)• recommend   • trajectory (§6.4)      │
   │  • report      • memory      • models      • fleet      (§6.5)      │
   └───────┬──────────────────────────────────────────────────────┬──────┘
           ▼                    DRIVERS                            ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ DICOM / DICOMweb   ·  HL7 FHIR (Task, DiagnosticReport)  ·  OAuth2   │
   │ THIRD-PARTY AI DRIVER API: any vendor's classifier posts findings    │
   │ as typed evidence events → opens/updates loops (they feed us)        │
   └──────────────────────────────────────────────────────────────────────┘
        Postgres (loops, beliefs, ledger)  ·  Object store  ·  Vector DB
```

### 4.2 What stays, what's new

**Unchanged:** vision, fusion (quantum + classical + benchmark), safety (calibration,
conformal, OOD, abstention), explain (saliency, Shapley, counterfactuals), recommend (EIG),
report (grounded composer + hallucination gate), memory (similarity), models, gateway,
schemas, dashboard visuals, event bus. The v1 pipeline (`gateway/pipeline.py`) becomes the
**loop-bootstrap path**: it runs when a study arrives and its outputs seed a loop instead
of terminating in a signed case.

**New:** `belief`, `ledger`, `guideline`, `trajectory` (absorbs and extends `memory`),
`fleet`. Five services, each individually justified in §6 — and a list of services we
explicitly *killed* in §6.6, because discipline is the difference between an OS and a
feature pile.

### 4.3 Core data model change

```
v1:  Study ──► CaseBundle (terminal)
v2:  Study ──► evidence events ──► Loop(s)
       Loop = {patient, question, hypotheses[], belief_history[],
               evidence_graph, commitments[], state}
       One study may open several loops (pneumonia workup AND incidental nodule).
       A loop outlives studies, encounters, and clinicians. It closes only by
       resolution, explicit override-with-reason, or escalation — never by timeout.
```

`CaseBundle` remains as the render contract for one study within a loop, so the existing
dashboard keeps working unmodified while gaining loop context.

---

## 5. New AI Architecture — the reasoning pipeline

v1 pipeline: `image → findings → evidence → posterior → safety → explain → recommend →
report`. v2 extends it to fourteen stages. **Design rule: every stage must name the
documented diagnostic failure mode it kills. If it can't, it gets cut.**

| # | Stage | What it does | Why it exists (failure mode killed) | Status |
|---|---|---|---|---|
| 1 | **Image → Findings** | Commodity perception (ours or any vendor's via driver API) | Missed findings. Deliberately replaceable — the moat is downstream | ✅ v1 `vision` |
| 2 | **Findings → Clinical Evidence** | Normalize every datum (finding, lab, prior, clinician note) into typed `EvidenceItem` with provenance + timestamp | Unusable/unauditable AI output; vendor lock-in. Everything becomes a citable evidence node | ✅ v1 `fusion/evidence.py`, generalized |
| 3 | **Evidence Graph** | Assemble nodes + relations; **absence is first-class** (`ABSENT_EVIDENCE` already in schema) | Unknown unknowns — you cannot reason about what you haven't noticed you lack | ✅ v1 schema, persisted per-loop in v2 |
| 4 | **Differential Diagnosis** | Calibrated posterior over hypotheses incl. NORMAL (quantum/classical fusion) | Single overconfident answer; premature closure starts here | ✅ v1 `fusion` |
| 5 | **Evidence Verification** | Cross-source consistency: per-source posteriors compared (KL divergence); disconfirming evidence surfaced *by name*; conflict score on the graph | **Anchoring & confirmation bias** — the UI literally shows "what contradicts this" | 🆕 in `belief` |
| 6 | **Guideline Matching** | Deterministic, versioned decision tables (Fleischner, Lung-RADS, ACR-AC) fire on belief + evidence; deviations flagged | Unwarranted variation; converts "AI opinion" into "published-standard obligation" — the defensibility jump | 🆕 `guideline` |
| 7 | **Uncertainty Analysis** | Temperature-scaled calibration, conformal sets with coverage guarantee, epistemic/aleatoric split, OOD, abstention | Silent failure — the v1 crown jewel, unchanged | ✅ v1 `safety` |
| 8 | **Risk Prediction** | Time-to-harm curves per hypothesis: *P(harm) vs. weeks of delay*, literature-anchored hazard templates modulated by priors | Flat worklists / alert fatigue. Converts uncertainty into **urgency** — probability alone can't rank a queue | 🆕 in `trajectory` |
| 9 | **Next Best Action** | EIG per cost/risk over candidate acquisitions, now including *"wait + recheck at T"* as a first-class action | Habit ordering, over-ordering | ✅ v1 `recommend`, extended |
| 10 | **Treatment Planning Support** | Guideline-anchored *options* with evidence status per option; auto-assembled tumor-board packet. **Not a treatment recommender** | The 2–4 hr/patient packet-assembly burden; also our regulatory line (§13.8) | 🆕 thin layer over guideline+report |
| 11 | **Longitudinal Monitoring** | Loop stays open; each new evidence event triggers belief update, delta report, commitment checks | **Lost follow-up** — the lawsuit stage. Watchdog semantics: no loop dies silently | 🆕 `belief`+`ledger`+`trajectory` |
| 12 | **Clinician Report** | Grounded report (every sentence → evidence nodes) now emitted in two forms: clinical report + **defense dossier** (prior-auth / med-legal appendix: guideline citation, posterior, evidence chain) | Documentation double-work; unjustifiable denials | ✅ v1 `report`, second output format |
| 13 | **Feedback Learning** | Verdicts → calibration refresh (v1); **closed loops → outcome labels** for risk curves and trajectory models | Static models; also the data moat: nobody else captures process→outcome pairs | ✅ v1 + 🆕 |
| 14 | **Fleet Aggregation** | Population roll-up: open loops, leakage $, drift, delay concentration | The CMO can't manage what nothing measures | 🆕 `fleet` |

**Quantum's role, stated honestly (criterion 6).** The fusion circuit is now the
*measurement-update operator* of a temporal Bayesian filter (stage 4, called at stage 11 on
every update). The evidence vector gains temporal channels (delta magnitude, change
velocity, time-since-evidence), so the entangling ansatz is modeling exactly the
higher-order interactions that matter clinically — *growth × smoking history × age* shifts
malignancy posterior non-additively. Second use: **quantum fidelity-kernel similarity over
evidence trajectories** (stage 8/§6.4) for rare-presentation retrieval in low-data regimes,
where quantum kernels are most plausibly interesting. Both ship beside classical
equivalents behind the existing `device.py` abstraction, benchmarked head-to-head by the
existing bench harness. We tell judges the truth: on 8 features the classical fallback is
near parity today; the quantum path is a benchmarked research edge, not the reason
hospitals buy. That honesty is a feature (§14.5).

---

## 6. New Backend Services (full specs)

### 6.1 `belief` — Belief State Engine (the kernel core)

- **Purpose.** Maintain, per diagnostic loop, the calibrated posterior over hypotheses as a
  *temporal* object: updated on every evidence event, never recomputed from scratch,
  auditable at any historical instant. The stateful successor to one-shot fusion.
- **Inputs.** Evidence events (`EvidenceItem` + provenance + timestamp); prior belief
  snapshot; fusion engine (update operator); safety engine (calibration per update).
- **Outputs.** `BeliefState {loop_id, seq, posterior, posterior_std, entropy,
  epistemic, aleatoric, top, delta_from_prev (KL), conflict_score, coverage_score,
  premature_closure_flag, model_versions, created_at}`.
- **Internal algorithms.** Recursive Bayesian update (predict = identity or slow decay
  toward prior; update = fusion over temporally-extended evidence vector). Conflict
  detection: KL divergence between single-source posteriors. **Premature-closure guard:**
  flag when top-probability rises above threshold while evidence *coverage* (fraction of
  EIG-relevant channels resolved) stays low — confidence without coverage is the bias
  signature.
- **ML models.** Existing fusion (quantum VQC / classical product-of-experts) as update
  operator; existing calibration. No new model families.
- **Quantum models.** The VQC consumes the extended vector (v1's 8 channels + delta,
  velocity, staleness); shot-variance → `posterior_std`, as in v1.
- **DB schema.**
  ```sql
  loops(id PK, patient_id FK, question TEXT, hypothesis_set JSONB,
        state ENUM(open,active,resolved,escalated,overridden), opened_by,
        opened_at, closed_at, close_reason)
  evidence_events(id PK, loop_id FK, item JSONB, source_service, source_kind
        ENUM(vision,lab,prior,clinician,external_ai), study_id NULL, occurred_at,
        ingested_at)                       -- append-only
  belief_snapshots(id PK, loop_id FK, seq INT, posterior JSONB, posterior_std JSONB,
        entropy REAL, epistemic REAL, aleatoric REAL, conflict REAL, coverage REAL,
        flags JSONB, model_versions JSONB, created_at)   -- append-only
  ```
- **REST endpoints.** `POST /v1/loops` · `POST /v1/loops/{id}/evidence` (202, event-sourced)
  · `GET /v1/loops/{id}/belief` · `GET /v1/loops/{id}/belief/history` ·
  `GET /v1/loops/{id}/conflicts`.
- **Caching.** Latest snapshot materialized in Redis keyed by `loop_id`, invalidated on
  evidence event; history reads from Postgres.
- **Failure handling.** Ingestion never blocks on computation (ingest-then-update). Update
  failure ⇒ belief stays at last snapshot with a visible `stale` flag; scheduler treats
  staleness as risk, not as silence.
- **Latency targets.** Evidence ingest ack < 100 ms; belief update < 2 s p95 (async);
  belief read < 50 ms (cached).
- **Security.** Org/role-scoped RBAC at gateway; snapshots immutable; every snapshot pins
  model versions for full historical reproducibility (extends v1 audit posture).
- **Scalability.** Per-loop updates are sequential; cross-loop embarrassingly parallel;
  event-sourced log → horizontal consumers.
- **Roadmap.** Hierarchical hypothesis spaces (SNOMED-mapped); multi-loop interaction
  (comorbidity coupling); federated site priors.

### 6.2 `ledger` — Commitment Ledger + Watchdog (the monetizable core)

- **Purpose.** Convert accepted recommendations into owned, time-boxed, escalating
  obligations. Append-only and hash-chained: the tamper-evident record that the process was
  followed — or the early warning when it isn't. This is the service that prevents the
  lawsuit, and the reason turning AURA off becomes unthinkable (§1.3, §10).
- **Inputs.** Accepted `Recommendation`s (v1 feedback flow); guideline matches
  (auto-propose action + window); clinician actions; external events (order placed, study
  arrived — FHIR `Task`/`ServiceRequest` later, simulated now).
- **Outputs.** `Commitment` records; risk-ranked watchdog queue; daily digest per owner;
  escalation events.
- **Internal algorithms.** Deterministic state machine
  `open → satisfied | overridden(reason required) | escalated(chain: owner → supervisor →
  service line → quality office)`. **No silent terminal state exists.** Queue priority =
  `harm_velocity(trajectory §6.4) × overdue_ratio × belief_severity`. Deliberately **no ML
  in the core** — determinism is what makes it audit- and court-grade.
- **DB schema.**
  ```sql
  commitments(id PK, loop_id FK, action, source_recommendation_id,
        guideline_match_id NULL, owner_id FK users, due_at, window JSONB,
        escalation_policy JSONB, state ENUM(open,satisfied,overridden,escalated),
        satisfied_by_event_id NULL, override_reason NULL, override_by NULL,
        hash_prev, hash_self, created_at)          -- append-only, hash-chained
  escalations(id PK, commitment_id FK, level INT, notified_id, delivered BOOL,
        acknowledged BOOL, created_at)
  ```
- **REST endpoints.** `POST /v1/commitments` · `GET /v1/commitments?state=&overdue=&owner=`
  · `POST /v1/commitments/{id}/satisfy` · `POST /v1/commitments/{id}/override` (reason
  mandatory, role-gated) · `GET /v1/watchdog/queue`.
- **Caching.** Watchdog queue materialized every scheduler tick (30–60 s); all reads from
  the materialized view.
- **Failure handling.** **The no-fail service.** Postgres-backed; watchdog emits an
  external heartbeat — if silent > 5 min, an out-of-band alarm fires (the watchdog is
  itself watched; dead-man-switch semantics). Escalation delivery is at-least-once with a
  human-visible undelivered queue — we never assume the page arrived.
- **Latency.** Writes < 50 ms; queue read < 200 ms.
- **Security.** DB grants: INSERT/SELECT only (no UPDATE/DELETE — matches v1 `audit_log`
  posture); hash chain verified nightly; overrides require role + reason and are
  fleet-visible (§6.5) to expose rubber-stamping.
- **Scalability.** Row-level, trivially horizontal; millions of commitments are a small
  table.
- **Roadmap.** FHIR `Task` write-back into EHR worklists; patient-facing reminders;
  payer-visible closure attestations (denials defense).

### 6.3 `guideline` — Guideline Engine

- **Purpose.** Deterministically map (belief, evidence, priors) → the applicable published
  pathway, with exact version pinning and deviation detection. This is the step that turns
  an AI posterior into a *published-standard obligation* — the difference between "the
  model thinks" and "Fleischner 2017 requires," which is what clinicians will act on,
  payers will accept, and lawyers can defend.
- **Inputs.** Belief snapshot, evidence graph, structured priors.
- **Outputs.** `GuidelineMatch {guideline_id, version, rule_path (full predicate trace),
  recommended_action, window_days, strength, citation}`; `Deviation` flags when care
  departs from a matched pathway without an override.
- **Internal algorithms.** Guidelines authored as declarative, versioned, signed YAML
  decision tables, compiled to evaluators. Every match carries its complete predicate path
  — explainable by construction. **No LLM anywhere in the decision path** (an LLM may
  draft rule encodings offline; a clinician reviewer signs them before activation).
  Explicit `no_match` is a first-class output — the engine never guesses.
- **ML/Quantum models.** None. Determinism is the point.
- **DB schema.**
  ```sql
  guidelines(id PK, name, version, source, effective_date, content_hash, signed_by,
        content JSONB, status ENUM(draft,active,retired))
  guideline_matches(id PK, loop_id FK, guideline_id FK, rule_path JSONB,
        action, window_days, strength, created_at)
  deviations(id PK, loop_id FK, match_id FK, kind, detail JSONB, resolved BOOL)
  ```
- **REST endpoints.** `GET /v1/guidelines` · `POST /v1/loops/{id}/guideline-match` ·
  `GET /v1/loops/{id}/deviations`.
- **Caching.** Matches memoized on `(belief_hash, evidence_hash, guideline_version)`.
- **Failure handling.** Rules are local versioned files — near-zero failure surface;
  version pinning means any historical match is exactly reproducible.
- **Latency.** < 50 ms.
- **Security.** Content signed; activation is a privileged, audited action; per-hospital
  pathway overlays are separate signed layers, never mutations of the source guideline.
- **Scalability.** Stateless; scale horizontally.
- **Roadmap.** Guideline marketplace — specialty societies publish signed rule packs;
  hospital-local overlays; automatic prior-auth dossier composition (with §5 stage 12).
- **P0 scope (hackathon-real).** Fleischner 2017 pulmonary-nodule criteria + Lung-RADS —
  small, fully encodable, canonical, and exactly what the demo story needs.

### 6.4 `trajectory` — Trajectory Engine (absorbs `memory`)

- **Purpose.** The longitudinal spine: patient timeline, study-over-study deltas,
  similar-trajectory retrieval, and time-to-harm risk curves. Answers "what changed?",
  "what happens if we wait?", and "who else looked like this, and how did it end?"
- **Inputs.** Belief history, evidence events, study metadata; (P1) registered image pairs.
- **Outputs.** `TimelineView`; `DeltaReport` (evidence-vector diff + belief KL between
  studies); `SimilarTrajectories` (k neighbors + outcomes where known, with similarity
  provenance); `RiskCurve {hypothesis, P(harm) vs weeks_of_delay, basis}`.
- **Internal algorithms.** Delta: channel-wise evidence diff + belief KL. Similarity:
  cosine ANN over evidence embeddings (v1 `memory`, kept) **and** a fidelity-kernel
  similarity `k(x,x′)=|⟨φ(x)|φ(x′)⟩|²` over trajectory feature vectors — the v1 §10.2 plan,
  now with a product reason and a benchmark against the classical kernel. Risk curves:
  parametric hazard templates per hypothesis (e.g., nodule volume-doubling-time bands),
  **literature-derived and labeled as such** until site outcomes exist; modulated by priors.
- **ML/Quantum.** Embeddings from v1 vision; quantum kernel via existing `device.py`
  abstraction with classical fallback; hazard templates are curves, not black boxes.
- **DB schema.**
  ```sql
  deltas(id PK, loop_id FK, from_study, to_study, payload JSONB, created_at)
  risk_curves(id PK, loop_id FK, hypothesis, curve JSONB, basis
        ENUM(literature,site_learned), model_version, created_at)
  -- timeline is a view over evidence_events + belief_snapshots + commitments
  ```
- **REST endpoints.** `GET /v1/patients/{id}/timeline` (v1 endpoint, extended) ·
  `GET /v1/loops/{id}/delta` · `GET /v1/loops/{id}/similar` ·
  `GET /v1/loops/{id}/risk-curve`.
- **Caching.** Timeline cached per patient, invalidated on new event; ANN index updated
  incrementally, rebuilt nightly.
- **Failure handling.** Similarity degrades gracefully (fewer neighbors, banner); risk
  curve failure falls back to the guideline window (a curve is nice; the deadline is law).
- **Latency.** Timeline < 300 ms; similarity < 500 ms; risk curve < 100 ms.
- **Security.** Neighbor results are de-identified exemplars; org-scoped; outcome labels
  visible only per RBAC.
- **Scalability.** Vector index (Qdrant in prod) scales independently; deltas are cheap.
- **Roadmap.** SimpleITK image registration for pixel-level deltas; **site-learned hazard
  models trained on the ledger's closed loops — every closed loop is a labeled
  process→outcome pair, the dataset nobody else is collecting** (the compounding moat).

### 6.5 `fleet` — Hospital Intelligence

- **Purpose.** The population layer: every open loop, overdue commitment, abstention/OOD
  hot spot, calibration drift, and delay concentration by service line — plus the leakage
  report in dollars. The C-suite and quality-office product; also where override abuse
  becomes visible.
- **Inputs.** Ledger, belief snapshots, safety metrics (extends v1 `/v1/admin/safety`).
- **Outputs.** Risk-ranked open-loop census; leakage report (lost follow-ups × downstream
  revenue + claim exposure ranges); calibration/drift dashboards; board-report export.
- **Internal algorithms.** Cohort aggregation; queueing statistics; survival analysis on
  loop closure times (Kaplan–Meier of "time to loop closed" per service line — a genuinely
  novel hospital KPI).
- **ML/Quantum.** None; read-only analytics over kernel tables.
- **DB schema.** Materialized views over `loops`, `commitments`, `belief_snapshots`,
  plus `fleet_reports(id, period, payload JSONB, generated_at)`.
- **REST endpoints.** `GET /v1/fleet/loops` · `GET /v1/fleet/leakage` ·
  `GET /v1/fleet/calibration` · `GET /v1/fleet/export`.
- **Caching.** Nightly materialization + on-demand refresh; all reads cached.
- **Failure/latency.** Read-only, non-critical path; < 500 ms cached; failure = stale
  dashboard, clearly stamped.
- **Security.** Admin/CMO roles only; aggregates de-identified by default; row-level
  drill-down audited.
- **Roadmap.** Cross-site de-identified benchmarking; payer attestation exports;
  malpractice-carrier risk reports (the distribution channel, §10).

### 6.6 Services we designed and then killed (brutality section)

| Candidate | Verdict | Why |
|---|---|---|
| **Treatment Simulation Engine** | ❌ killed | Simulating treatment outcomes without interventional data is fantasy dressed as software, and it moves us from CDS-exempt decision support into autonomous-therapy territory — regulatory suicide for zero demo value. Treatment support = guideline-anchored options + evidence status, nothing more (§5 stage 10) |
| **Consensus Engine** (aggregate multiple doctors' opinions algorithmically) | ❌ killed as a service | Tumor-board mode is a *workflow* over belief + evidence + packet composer. An algorithm that arbitrates between attending physicians is a solution looking for a problem and a liability magnet. UI feature, not engine |
| **Standalone Outcome Prediction Engine** | ❌ killed | Mortality/outcome prediction without site cohorts is a college project with a ROC curve. Folded into `trajectory` as literature-anchored hazard *templates*, honestly labeled, upgraded only when ledger outcomes exist |
| **Disease Progression generative model** ("watch the tumor grow") | ❌ killed | Demo candy. Clinically vacuous, validation-impossible at our stage |
| **Conversational layer / clinical chatbot** | ❌ killed | Stateless Q&A is the opposite of our thesis; the brief bans it and the brief is right |
| **Quantum image processing** | ❌ stays killed (v1 decision) | Classical CNNs won perception. Quantum stays where structure is small and correlation-rich: fusion and kernels |

---

## 7. New Frontend Workflow

Existing visual language, components, and motion stay exactly as they are. The change is
**semantic**: every panel is retitled as the clinical question it answers, and three new
surfaces reuse the existing component vocabulary (cards, sparklines, ranked lists, badges).

### 7.1 Surface A — Loop Census ("air traffic control")
*The worklist, evolved.* Every open diagnostic loop in the hospital, ranked by
`harm_velocity × overdue × uncertainty`, not FIFO.

| Panel | Clinical question it answers |
|---|---|
| Risk-ranked loop queue | **"Which patient is silently falling through, right now?"** |
| Watchdog strip (overdue commitments) | "What did we promise and not do?" |
| Abstention lane (v1 abstained cases) | "Where does the AI itself say 'I don't know'?" |

### 7.2 Surface B — Case Cockpit (the existing dashboard, reframed)
| Existing panel | Reframed question |
|---|---|
| Posterior + conformal set | "What do we believe, and how sure are we — with a guarantee?" |
| Evidence attribution (Shapley) | "What evidence **supports** this?" |
| Counterfactuals | "What diagnosis would disappear if this finding were absent?" |
| 🆕 Conflict panel (stage 5) | "What evidence **contradicts** this?" |
| EIG recommendations | "What additional test reduces uncertainty the most, per dollar and per risk?" |
| 🆕 Guideline chip | "Which published rule applies — chapter and verse?" |
| 🆕 Risk-of-delay curve | "What does waiting four weeks cost this patient?" |
| Saliency overlay | "Where in the image is this coming from?" |
| Report draft + grounding | "What will we sign, and can every sentence cite its evidence?" |

### 7.3 Surface C — Trajectory View (patient timeline)
| Panel | Question |
|---|---|
| Belief sparklines per hypothesis over time | "How has our thinking moved — and what moved it?" |
| Study-over-study delta | "**What changed since the previous scan?**" |
| Commitments as milestones on the timeline | "What was promised, done, and outstanding?" |
| Similar trajectories + outcomes | "**What similar patients existed, and what happened to them?**" |
| One-click tumor-board packet | "Can the whole story be in the room in 15 seconds?" |

### 7.4 Surface D — Fleet Console (extends existing admin/safety view)
| Panel | Question |
|---|---|
| Open-loop census + closure survival curve | "Is this hospital closing its loops on time?" |
| Leakage report ($) | "What are lost follow-ups costing us?" |
| Calibration & drift (v1) | "Can we still trust the models this month?" |
| Override ledger | "Who is rubber-stamping, and why?" |

**Design law (unchanged from v1, now enforced product-wide): no panel may display data
that does not answer a named clinical question.**

---

## 8. Demo Flow (~4½ minutes, seven beats)

Built entirely on existing machinery plus the new services; the only demo affordance is a
time-jump control on the simulate endpoint.

1. **[0:00] Study arrives.** Upload CXR via existing simulate flow. Cockpit shows the v1
   pipeline live: differential with conformal set, supporting *and contradicting* evidence,
   saliency, EIG next step. One line: "Everything you've seen so far, others also do —
   watch what happens next."
2. **[1:00] One study, two questions.** Vision also flags a 9 mm incidental nodule. AURA
   opens a **second loop** automatically — the event every current system drops on the
   floor. Guideline chip fires: *Fleischner 2017 → CT follow-up in 3–6 months.* A
   commitment is created: owner, due window, escalation chain visible.
3. **[1:45] The silent failure, made loud.** Time-jump six months. No CT was ever ordered.
   The watchdog escalates; on the Loop Census the patient climbs the risk queue as the
   risk-of-delay curve rises. Line to judges: **"In most hospitals, this moment is
   invisible. It surfaces two years later as a stage-IV diagnosis and a lawsuit. AURA just
   made it a Tuesday-morning to-do item."**
4. **[2:30] Evidence arrives, belief moves.** The CT arrives (simulate). Belief update
   runs live: malignancy posterior 0.18 → 0.62, entropy drops, the quantum-vs-classical
   fusion benchmark badge shows both backends agreeing (honesty on stage).
5. **[3:15] The story assembles itself.** Trajectory view: belief sparkline over six
   months, the delta report, three similar historical trajectories with outcomes.
   One click → grounded tumor-board packet, every sentence citing its evidence node.
6. **[3:45] The C-suite screen.** Fleet console on seeded data: *"Last 90 days: 214 open
   loops, 37 overdue, $1.9 M downstream revenue at risk, 2 loops escalated to quality."*
7. **[4:15] Close.** "Perception AI opens loops. **AURA closes them.** That's the
   operating system, and that's why hospitals that already own classifiers need it more,
   not less."

---

## 9. Judge Pitch (90 seconds, verbatim)

> "Every AI company in this room can find a nodule on a chest X-ray. Here's the number
> none of them touch: about eight hundred thousand Americans a year are killed or
> permanently disabled by diagnostic error — and in most of those cases, *the finding was
> seen*. It was in the report. Then the process failed: the follow-up was never done, the
> context died at a handoff, the differential closed too early. Detection AI actually makes
> this worse — more findings means more open loops, into hospitals that have no system for
> closing loops.
>
> AURA is that system: an operating system for diagnosis. Every open clinical question
> becomes a tracked process with a calibrated belief state that updates as evidence
> arrives, a published guideline attached — chapter and verse — and a named owner with a
> deadline that cannot expire silently. Our watchdog means a forgotten six-month CT
> follow-up stops being a lawsuit in two years and becomes a to-do item on Tuesday.
>
> Under the hood: conformal prediction with coverage guarantees, an AI that's allowed to
> say 'I don't know', expected-information-gain test selection, and a quantum evidence
> fusion core benchmarked live against its classical fallback — we'll show you the
> benchmark, not hand-wave it.
>
> EHRs record what was done. PACS records what was seen. AURA records what was *thought* —
> and makes sure it gets acted on. We're not another radiology AI. We're the system of
> record for clinical reasoning."

---

## 10. Investor Pitch

- **Problem.** Diagnostic process failure: ~795 k serious harms/yr US; largest malpractice
  category (~$4–5 B/yr paid); 30–70 % of imaging follow-ups never completed; each lost
  follow-up is both claim exposure and $20–50 k lost downstream revenue. Perception AI
  (a crowded, commoditizing market) *increases* open loops and closes none.
- **Product.** The system of record for diagnostic loops: belief states + commitment
  ledger + guideline engine + trajectory intelligence. Category: **Clinical Intelligence
  OS.** Not a better classifier — the layer classifiers plug into.
- **Wedge (land).** *Shadow-mode results management.* Point AURA at 90 days of existing
  radiology reports — text only, two-week deployment, no PACS surgery: it extracts every
  follow-up recommendation and shows which ones never happened, priced in dollars and
  risk. Every hospital has this skeleton in its closet; the audit *is* the sales demo.
  Beachhead segment: lung-cancer-screening programs, which are already legally required to
  run tracking registries and do it in Excel today — mandated behavior, zero behavior
  change required.
- **Expand.** Radiology cockpit → oncology/tumor board → ED/ICU handoffs → hospital-wide
  loop census → payer products (auto-composed prior-auth and denial-defense dossiers from
  the evidence graph).
- **Platform (the endgame).** The driver API turns every perception-AI vendor into a
  supplier: their findings are evidence events into our loops. Competitors become
  peripherals.
- **Business model.** SaaS per site + per-monitored-study platform fee. Comparable
  single-use-case perception tools command $50–200 k/yr/site; a system of record with
  CFO-legible ROI supports $150–300 k/yr mid-size hospital. US ~6,000 hospitals ⇒ ~$1–2 B
  core US TAM before payer products, international (single-payer systems with backlog
  crises are ideal), and platform fees.
- **Channel.** Malpractice carriers (they already discount premiums for risk-reduction
  programs — a loop-closure ledger is their dream product) and chief quality/medical
  officers, alongside classic health-system sales.
- **Moat.** (1) The ledger: switching off the system of record for open obligations is a
  liability event — retention is structural. (2) The data: every closed loop is a labeled
  process→outcome pair; nobody else is collecting reasoning-trajectory outcomes at scale.
  (3) Guideline packs + hospital overlays accumulate as a signed content network.
  (4) Regulatory posture as asset: built to the CDS-exemption spec (basis-transparent),
  audit-grade by construction.
- **Why now.** Perception AI commoditized (input supply exists) + radiologist shortage +
  diagnostic-safety regulation arriving (Leapfrog, AHRQ, CMS direction) + hospitals under
  margin pressure looking for leakage recovery that pays for itself.

---

## 11. Technical Roadmap (from today's repo)

Ordered so the demo strengthens at every step; v1 code is never rewritten.

| Step | Build | On top of |
|---|---|---|
| 1 | `loops` + `evidence_events` + `belief_snapshots` tables; wrap `Pipeline.run()` so a study seeds/updates loops (CaseBundle unchanged as render contract) | `gateway/pipeline.py`, `storage.py` |
| 2 | `belief` service: temporal evidence vector (add delta/velocity/staleness channels), recursive update calling existing `FusionEngine`; conflict + premature-closure flags | `services/fusion` |
| 3 | `guideline` service with Fleischner + Lung-RADS as signed YAML decision tables; guideline chip in cockpit | new, deterministic |
| 4 | `ledger` service: commitments + watchdog + escalation + hash chain; time-jump control on simulate for the demo | `gateway/seed.py`, feedback flow |
| 5 | `trajectory`: absorb `memory`, add delta report, risk-curve templates, similar-trajectory panel; belief sparklines | `services/memory`, existing UI components |
| 6 | Loop Census + Fleet Console views; leakage report over seeded data | existing SPA vocabulary, `/v1/admin/safety` |
| 7 | Quantum trajectory kernel behind `device.py` + extend existing bench to kernel-vs-kernel | `services/fusion/{device,quantum}.py`, `ml/evaluation` |
| 8 | Report v2: defense-dossier output format from existing grounding map | `services/report` |
| 9 | Driver API: `POST /v1/loops/{id}/evidence` hardened for external AI vendors (auth scopes, provenance required) | gateway |

Latency budget end-to-end (demo hardware): study → loop seeded < 5 s; evidence → updated
belief < 2 s p95; all UI reads < 300 ms cached. Failure doctrine: engines may degrade,
**the ledger may not**; every degraded state is visibly stamped, never silent — the
product's own epistemic honesty applied to itself.

## 12. 5-Year Startup Roadmap

| Year | Milestones | Proof point |
|---|---|---|
| **1** | Hackathon → 2–3 academic design partners; shadow-mode backlog audits; results-manager GA (text-only integration); retrospective validation on public longitudinal data (e.g. MIMIC-CXR) | "We found N dangling follow-ups at your site" reports; first peer-reviewed shadow study |
| **2** | Belief+ledger GA; 10–20 sites; SMART-on-FHIR + FHIR Task write-back; SOC 2 Type II; malpractice-carrier channel pilot; Series A on loop-closure metrics | Published before/after: loop-closure rate ↑ at pilot sites |
| **3** | Guideline marketplace v1 (2–3 specialty societies); payer dossier product; 50+ sites; FDA Q-Sub for any triage-adjacent claims; site-learned hazard models from ledger outcomes | First payer contract; first society-signed guideline pack |
| **4** | Driver/syscall platform GA — third-party AI vendors certified as evidence sources; federated calibration across sites; 150+ sites; international (NHS/EU backlog programs) | Competitor perception vendors integrate as suppliers |
| **5** | System-of-record status: hospitals run diagnostic operations through the loop census; outcomes-based contracts (paid on closure rates); the OS claim is earned, not asserted | Diagnostic loop-closure rate appears in hospital board packs and quality ratings |

---

## 13. Honest Weaknesses (no varnish)

1. **EHR gravity.** Epic owns the clinician's screen and could build a shallow version of
   results tracking (fragments exist). Our defenses — reasoning depth, vendor-neutral
   driver API, calibration/abstention machinery Epic won't build, speed — are real but not
   guaranteed. Anyone who tells you Epic risk is "handled" is lying; we mitigate by living
   inside SMART-on-FHIR and starting where Epic is weakest (radiology results management,
   screening registries).
2. **Quantum lift is unproven.** On 8-feature synthetic data, classical product-of-experts
   is near parity with the VQC. We say this on stage, ship the classical fallback as
   first-class, and treat quantum as a benchmarked research edge (trajectory kernels are
   the honest test). If the lift never materializes, the product loses zero clinical value
   — and we will have said so from day one.
3. **Synthetic data ≠ clinical validity.** Today's pipeline runs on generated studies.
   Clinical validity is exactly zero until retrospective studies on real longitudinal data
   and prospective shadow deployment. This is the single biggest gap between demo and
   product, and no amount of architecture hides it.
4. **The audit-trail paradox.** A ledger of missed follow-ups is discoverable in
   litigation; some hospital general counsels will hate it. Counterpoints: documented
   process adherence is the strongest defense there is, carriers reward tracked-loop
   programs, and retention is configurable within regulation — but expect this objection
   in every enterprise sale.
5. **Watchdog ≠ new pager hell.** An escalation system can recreate the alert fatigue we
   exist to kill. Design guards: risk-ranked digests instead of per-event pages,
   escalation budgets per clinician per day (an SLO we publish), org-tunable thresholds,
   abstention-first philosophy. This must be *measured*, not assumed.
6. **Behavior change risk.** Commitments require owners who accept ownership. If overrides
   become rubber stamps, the value collapses. Mitigations: mandatory override reasons,
   fleet-visible override analytics, and a beachhead (lung screening registries) where
   tracking is already legally mandated so the behavior exists.
7. **LLM containment.** The report LLM stays where v1 put it: phrasing only, behind the
   grounding validator. No LLM in belief, guideline, or ledger decision paths. Any drift
   from this is a product-integrity failure.
8. **Regulatory line discipline.** We are decision support engineered to the CDS-exemption
   spec (clinician can independently review the basis of every output — our explainability
   is that spec, by construction). Triage/prioritization claims may still cross into SaMD;
   we budget for a Q-Sub in year 3 and never market autonomous diagnosis. Marketing
   discipline is a genuine operational risk in this category.

---

## 14. What Would Make AURA Impossible To Ignore

1. **The 90-day backlog audit.** Text-only shadow mode over a hospital's existing reports:
   *"You issued 1,412 follow-up recommendations last quarter. 214 are open. 37 are past
   their guideline window. Here they are, ranked by risk of harm."* Zero integration risk,
   two-week deployment, and the output is simultaneously the sales demo, the ROI case, and
   a patient-safety intervention. No perception-AI vendor can produce this artifact.
2. **The "lawsuit prevented" demo beat.** Watching a forgotten Fleischner follow-up
   escalate to the top of the queue — live — is the moment judges and buyers *feel* the
   product. Everyone in the room has a family story about a late diagnosis.
3. **One publishable number.** *Diagnostic loop-closure rate, before vs. after.* A metric
   hospital boards, carriers, and regulators all read — and that only a system which
   models loops can even compute. Owning the metric owns the category.
4. **Competitors as suppliers.** The day the first perception vendor certifies against the
   driver API, the "why buy AURA if we have classifiers" question inverts permanently:
   classifiers need somewhere to put their findings.
5. **Institutionalized honesty.** Conformal guarantees, first-class abstention, a live
   quantum-vs-classical benchmark instead of quantum hand-waving, literature-labeled risk
   curves, and a self-monitoring watchdog. In a market burned by AI overclaiming, *the
   system that knows what it doesn't know* — and proves it — is the one clinicians will
   let live inside their reasoning.

---

*— The AURA founding team: CMO (radiology), Mayo clinical-AI research, MIT CSAIL,
DeepMind Health, NVIDIA quantum AI, Stanford CDS, YC founder, staff architect,
principal ML engineer, Apple product design.*
