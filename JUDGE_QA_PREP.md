# AURA — Judge Q&A Battle Card

**Prep doc for the tough questions.** Every answer below is backed by something in the repo you
can open on the spot. The golden rule: **AURA's credibility IS its honesty.** Never oversell.
When a number is unflattering, say it first — that's the whole brand ("the model that knows what
it doesn't know"). A judge who catches you hiding something sinks you; a judge who sees you
disclose it before they ask trusts everything else.

**Files to have open:** `README.md`, `aura/ml/training/train_fusion.py`,
`aura/ml/vision_cxr/model.py`, `aura/artifacts/benchmark.json`,
`aura/artifacts/labeler_validation.json`, `aura/artifacts/kappa_crosscheck.json`,
`aura/artifacts/quantum_study.json`, `aura/artifacts/ibm_hardware_run.json`.

---

## Category A — "Is any of this actually yours?" (novelty)

### Q1. "You just used DenseNet-121 off the shelf. What did you actually build?"
**The trap:** getting defensive and claiming you built the CNN — they'll open `model.py` and see `torchvision.models.densenet121` and you're done.

**Answer:** "Correct — the CNN backbone is DenseNet-121 from torchvision, and that's deliberate. Every serious CXR system (CheXNet included) uses a standard validated backbone; hand-rolling a CNN would be a red flag, not a plus. The backbone is ~15% of the system. What we built is the other 85%: the four-way evidence-fusion layer including an 8-qubit quantum circuit, the entire calibrated-doubt safety stack — Mondrian conformal prediction + online Adaptive Conformal Inference + energy-score OOD + abstention — the information-gain recommender, the evidence-grounded reporting loop, and a from-scratch multi-task brain-MRI network. None of those are pretrained; all trained by us."

**Evidence:** exactly one pretrained model in the whole repo (`DenseNet121_Weights.DEFAULT` at [model.py:69](aura/ml/vision_cxr/model.py)); everything in `train_fusion.py` is random-init.

### Q2. "So the only pretrained thing is ImageNet DenseNet — and even that isn't medical?"
**Answer:** "Right. ImageNet is cats and cars — zero medical knowledge. We take only its generic edge/texture filters and **fine-tune the whole network on MIMIC-CXR**, replacing the head entirely and rebuilding the input conv for grayscale via a luminance-weighted init. So even the 'borrowed' part is retrained on our task and our data."

**Evidence:** `luminance_init_conv0` + `nn.Linear(in, 7)` head swap in [model.py](aura/ml/vision_cxr/model.py).

### Q3. "What is the single most novel thing here?"
**Answer:** "That AURA's *product is calibrated doubt*, not a label — and it's wired end-to-end into the served answer. Concretely: a **working conformal coverage guarantee that self-corrects online** (ACI), plus an **abstention path** so the model can refuse. Most medical AI ships the classifier; almost nobody ships the disciplined uncertainty machinery around it, offline."

---

## Category B — Quantum (the highest-risk area)

### Q4. "Is the quantum part real or a buzzword?"
**The trap:** overclaiming quantum advantage. Don't.

**Answer:** "It's real, and we're honest about what it does. It's an 8-qubit variational circuit in PennyLane: each evidence channel is angle-encoded `RY(π·xᵢ)` on its own qubit, entangling layers capture higher-order interactions, and we read `⟨Zᵢ⟩` into a linear head — 102 trainable parameters, trained by backprop on real MIMIC evidence. It's correctly *placed* at evidence fusion, not imaging. And we benchmark it head-to-head against a classical twin on the same distribution."

**Evidence:** `quantum_study.json` circuit block; `train_quantum()` in [train_fusion.py:77](aura/ml/training/train_fusion.py).

### Q5. "Does the quantum model actually beat classical? Show me."
**The trap:** claiming it wins. It doesn't, on the fair test. Say so proudly.

**Answer:** "No — and that's our most important integrity point. On real held-out MIMIC-CXR evidence, with **each backend temperature-scaled on its own split** (a fair fight), classical wins: accuracy 0.696 vs 0.638, ECE 0.219 vs 0.238. The earlier 'quantum wins' result was an artifact of applying the quantum temperature to classical logits — we found and fixed that. So we lead with calibration discipline, not an unverifiable quantum claim. The VQC is real, competitive, and correctly placed — not magic."

**Evidence:** `benchmark.json` (`quantum.accuracy 0.6377` vs `classical.accuracy 0.6957`); regenerate live with `py -m aura.aura_cli bench` then `python ../audit_all.py`.

### Q5b. "If classical wins, why does your config serve quantum?" *(they can read `pyproject.toml`)*
**The trap:** being caught not knowing your own serving config, or pretending the benchmark decided it. Own the decision.

**Answer:** "Correct — `fusion_backend = "quantum"`, and that's deliberate, not an oversight. The benchmark decides what we *claim*, not what we *serve*. Three things make serving the VQC safe. One: each backend is calibrated on its own logits — quantum T = 0.9057, classical T = 0.479, a 1.89× spread — so the served posterior isn't borrowing another model's temperature. That exact bug is what produced our earlier fake 'quantum wins' result; we found it and fixed it. Two: the Wasserstein conflict guard is armed *only* when quantum serves, and it defers to the classical twin whenever the two posteriors diverge — so the classical model is a live safety net, not a benchmark trophy. Three: on this held-out split the gap is ~6 accuracy points at n=69, and the classical head's advantage is real but not so large that keeping the research substrate in the serving path is reckless — especially with the guard behind it. If I were shipping this to a hospital tomorrow I'd serve classical and say so; for a research prototype I want the quantum path exercised by real traffic."

**Evidence:** `aura/pyproject.toml:44`; `backend_calibration.json` (`temperature_spread: 1.89`); `_guard_enabled` in [engine.py](aura/services/fusion/engine.py) requires `backend == "quantum"`.

### Q6. "Then why include quantum at all if it loses?"
**Answer:** "Four reasons. One, scientific honesty demands the comparison exist. Two, it's a genuine research result — we ran an **entanglement ablation** at matched parameter count and found entanglement *hurts* on NLL: Δnll +0.056, 95% bootstrap CI [0.022, 0.090], which excludes zero. Accuracy and ECE moved too, but their intervals **include** zero, so I'll only claim the NLL result. A clean negative result worth publishing.

Three — and this is the one I'd actually defend as a quantum advantage — **QMBA**, our measurement-budget allocator. Because a quantum model has a measurement budget, we can spend shots adaptively: the median study resolves in **128 shots against the 8 192-shot per-circuit device ceiling — 64× fewer measurements**. Better, it tells us *which kind* of uncertainty we hit: across 447 studies, 1 abstention was measurement-limited — more shots would fix it — and 60 were model-limited, genuinely tied at infinite precision, so they need a human. **A classical fusion model cannot make that distinction, because it has no measurement budget to vary.** That's a capability, not a speedup, and it's measured.

Four, we ran the fusion circuit on **real IBM quantum hardware** (`ibm_marrakesh`, 156 qubits, job `d9js49rjf64c739haeg0`) — device noise attenuated every expectation value, mean absolute error 0.186 versus analytic, and the top-1 diagnosis survived anyway."

**Evidence:** `quantum_study.json` → `q1_q2_ablation.entanglement_effect.bootstrap` (read the CIs, not the point estimates) and `q3_measurement_budget`; shot ceiling at [qmba.py:67](aura/services/fusion/qmba.py); `ibm_hardware_run.json` (real QPU job).

**Say the boundary out loud:** the ablation and QMBA numbers are statevector simulation; the hardware run is one case on a real QPU. Don't let the two blur — the quantum jury is listening for exactly that.

### Q7. "Isn't 102 parameters / 8 qubits a toy?"
**Answer:** "Yes, intentionally — it fuses an 8-dimensional evidence vector, not raw pixels. Quantum isn't placed where it can't scale (imaging). It's placed at low-dimensional evidence fusion where an 8-qubit circuit is exactly sized, runs on a laptop simulator in seconds, and can run on today's real QPUs. We chose the one spot in the pipeline where NISQ-era quantum is honestly applicable."

---

## Category C — Calibration & the statistical claims

### Q8. "You keep saying 'calibrated.' Prove it."
**Answer:** "Expected Calibration Error is measured, not asserted. Served vision ECE is ~0.031 after per-finding Platt scaling. We use temperature scaling on a held-out calibration split, then **class-conditional (Mondrian) conformal prediction sets** for a coverage guarantee, and **online Adaptive Conformal Inference** that adjusts the threshold from confirmed outcomes so coverage holds even under drift."

**Evidence:** `RETRAIN_V3_RESULTS.txt` (ECE 0.0312); `benchmark.json` conformal_coverage ~0.91–0.93; fitting code in [train_fusion.py](aura/ml/training/train_fusion.py) (`fit_temperature`, `fit_conformal`, `mondrian_qhats`).

### Q9. "Conformal coverage is 0.9275 but you targeted 0.90 — and n=69. Is that meaningful?"
**The trap:** pretending small-n results are rock solid.

**Answer:** "Honest limitation: the fusion held-out set is small (n=69 in that benchmark; the fuller quantum study uses a patient-disjoint 1338/446/447 split). Coverage lands near target but I wouldn't claim tight guarantees at that n — the point is the *machinery* is correct and reproducible, and ACI keeps tightening it as real cases arrive. We report the n every time rather than hiding it."

**Evidence:** `benchmark.json` `n_eval: 69`; `quantum_study.json` `data` block shows `patient_disjoint: true`, 1338/446/447.

---

## Category D — The circularity / label-integrity attack (they WILL ask this)

### Q10. "Your labels come from the radiology reports, and you test against those same labels. Isn't your AUROC circular?"
**The trap:** this is the single most sophisticated attack. Have the answer ready cold.

**Answer:** "We pre-empted exactly this. Two independent defenses. **One:** we validated our rule-based v2 labeler against a **hand-read gold standard of 66 reports** — agreement κ≈0.86, F1≈0.89 — so the labels are faithful, not noise we're fitting. **Two:** we cross-checked our vision model against a **separately-trained public model** (torchxrayvision `densenet121-res224-mimic_ch`) on 700 images. Because its training labels are independent of ours, the rank-correlation and cross-AUROC agreement is label-free evidence we learned real anatomy, not our own label artifacts."

**Evidence:** `labeler_validation.json` (66-report gold, purpose field literally states the circularity pre-emption); `kappa_crosscheck.json` (700 images, independent xrv weights).

### Q11. "Who made your gold standard? A radiologist?"
**The trap:** overclaiming clinical validation.

**Answer:** "No — and we state that in the artifact itself. The 66 gold reports were read by an independent LLM reader applying CheXpert rules, which resolves negation/uncertainty a regex can't, but it is **not a board-certified radiologist**. It's a single annotator with a documented ceiling, and we name the next step: a radiologist over-read. We disclose the Wilson CIs on rare-finding cells too. This is honest labeler validation, not a claim of clinical ground truth."

**Evidence:** `labeler_validation.json` `gold_provenance.annotator` and `limits` fields — read them verbatim if pushed.

### Q12. "Which finding does your labeler fail on?"
**Answer:** "Nodule. Our torchxrayvision cross-check fails to corroborate nodule, and we disclose that openly in the model card and UI rather than burying it. Naming your own weakest finding is how we earn trust on the strong ones."

---

## Category E — Clinical / safety / "would you deploy this?"

### Q13. "Would you put this in a hospital tomorrow?"
**Answer:** "No, and any team that says yes is lying. AURA is a decision-support *copilot*, not an autonomous diagnostician — it's explicitly designed to hand control back to the clinician, including abstaining. It needs prospective clinical validation and regulatory clearance (FDA SaMD path) before deployment. What it *is* today is a working, calibrated, auditable research system that demonstrates the safety architecture such a product would need."

### Q14. "What stops it from confidently misdiagnosing?"
**Answer:** "Four gates in series. Temperature scaling stops overconfident logits; the Wasserstein conflict guard falls back to the trusted classical posterior when the two fusion backends disagree; energy-score OOD detection flags inputs unlike training data; and an explicit abstention policy lets it say 'I don't know.' Feed it a non-chest image and the chest gate refuses it rather than inventing a lung finding — that was a real bug we found and fixed."

**Evidence:** conflict guard + OOD + abstention in `services/safety/` and `services/fusion/`; the "brain MRI → Pneumonia" fix in the audit history.

### Q15. "Sensitivity for pneumonia looks low (0.375). That's dangerous."
**The trap:** hiding weak per-class numbers.

**Answer:** "Agreed it's low, and it's visible in our own metrics — support is only 8 cases there, so it's a wide interval, but I won't wave it away. This is exactly why the system *abstains and recommends the next test* instead of committing: on low-confidence, low-evidence calls it escalates rather than ruling out. The safety value isn't perfect sensitivity — it's never being silently, confidently wrong."

**Evidence:** `benchmark.json` `metrics_full.quantum.per_class.pneumonia` (sensitivity 0.375, support 8).

---

## Category F — Data & generalization

### Q16. "You only trained on MIMIC-CXR. Will it generalize to other hospitals?"
**Answer:** "Unknown, and we don't claim otherwise — single-source training is a real external-validity limit. Two things mitigate it: our online ACI loop is *built* to re-calibrate under distribution shift from confirmed outcomes, and OOD detection flags inputs that look unlike MIMIC so it degrades to abstention rather than to confident error. Cross-institution validation is named future work, not a solved claim."

### Q17. "How big is your dataset? Isn't it small?"
**Answer:** "The vision model trains on MIMIC-CXR at full 224×224; the fusion study uses a patient-disjoint 1338/446/447 split. It's a research-scale corpus, not a registration-scale one. We compensate with class-capping so rare dangerous findings (pneumothorax, malignancy) aren't drowned out, and with `pos_weight` imbalance correction — but we're upfront that scale is a limitation, not a strength."

**Evidence:** `quantum_study.json` `data`; per-class cap logic in [train_fusion.py:150](aura/ml/training/train_fusion.py).

---

## Category G — Multimodal / brain

### Q18. "You claim brain MRI too — is that real or a slide?"
**Answer:** "Real and separate. It's a from-scratch multi-task network — residual encoder + U-Net-style decoder with deep supervision, nnU-Net-inspired — trained on BraTS with segmentation masks, five heads. It's FLAIR-driven; T1-only Dice collapses to ~0.02, which tells us it genuinely uses the modality rather than guessing. A modality router detects study type and dispatches chest→Thorax, brain→NeuroMind — adding CT or retina is one engine class, not a rewrite. That's the 'Medical AI OS' claim."

**Evidence:** `backend/vision/brain/model/` (encoder/decoder/heads); `backend/README.md` router section.

---

## Category H — Engineering / reproducibility

### Q19. "How do I know these numbers aren't hand-typed into a README?"
**Answer:** "Regenerate them live. `py -m aura.aura_cli bench` rewrites `benchmark.json`; `python audit_all.py` runs DeLong / McNemar / bootstrap significance tests. The whole thing trains from scratch in ~30 seconds on a laptop CPU and passes 366 automated test functions across 32 test files. Nothing here is asserted; it's all reproducible."

**Evidence:** `py -m aura.aura_cli bench`, `audit_all.py`; 366 `def test_` across `aura/tests/`.

### Q20. "Why offline? Isn't cloud easier?"
**Answer:** "Because PHI. AURA runs fully offline on local CPU — no cloud, no API keys, no patient data leaving the box. In healthcare that's not a nice-to-have, it's the difference between deployable and not. It also means our reproducibility claims are real: the judge can run the entire system air-gapped."

---

## Category I — Business / "so what"

### Q21. "Who actually needs this? Radiologists are already good."
**Answer:** "The value isn't replacing the radiologist — it's the triage/overflow and second-opinion layer where the scarce resource is expert time. AURA's differentiator is that it tells you *when to trust it and when not to*, ranks the next test by information gain per cost, and does it offline. A confident black box is unusable in medicine; a calibrated one that abstains is a force multiplier."

### Q22. "What's the moat? Anyone can fine-tune DenseNet."
**Answer:** "Exactly — the backbone is commodity, so the moat isn't there. It's the integrated calibrated-doubt system: conformal + ACI + OOD + abstention + EIG recommender + evidence-grounded reporting + the feedback loop, all wired into one served answer and reproducible offline. That *combination* is what doesn't exist off the shelf, and it's the part that's genuinely hard to copy."

---

## Category J — Adversarial gotchas

### Q23. "Show me something in the repo that proves you're honest, not marketing."
**Answer:** "Open `quantum_study.json` — it's a **negative result** we chose to keep: entanglement makes our model *worse*, stated with significance tests. Or the doc-honesty pass where we scrubbed fabricated 'tested on ibm_kyoto/osaka' claims and expected-metrics-presented-as-measured out of our own bibles. We delete our own overclaims. That's the tell."

### Q24. "What's the weakest part of AURA — you pick."
**The trap:** saying "nothing." Have a real answer.

**Answer (pick ONE, then rank the rest):** "The single weakest part is **external validity** — we trained and tested on MIMIC-CXR alone, so I cannot show you cross-hospital, cross-scanner generalization. That's the one I'd attack if I were you.

Behind it, ranked: **two**, our 'gold standard' is 66 reports read by a single non-radiologist annotator — that validates our *labeler* (κ≈0.86), it is not clinical ground truth, and we say so in the artifact itself. **Three**, the fusion benchmark's held-out set is small (n=69), so the calibration numbers are directional, not tight guarantees.

Two things I want you to notice about that list. First — I just handed you our three sharpest attacks unprompted, and every one is already written into the artifacts (`labeler_validation.json` states the circularity risk and the annotator ceiling; `benchmark.json` prints its own n=69). We don't hide weaknesses; we ship them. Second — these are *data and scale* limits, not *architecture* limits. AURA is built so that exactly these gaps degrade to **abstention**, not to confident error: an out-of-distribution scanner trips OOD detection, a low-n low-evidence call abstains and recommends the next test. So our weakest point is 'unproven on more data' — which is a funding-and-time problem — not 'silently wrong,' which is the one that actually kills people."

### Q25. "If I had one reason to fail you, what would it be — and why shouldn't I?"
**Answer:** "The one reason: it's a research prototype, not a validated clinical product, and quantum doesn't beat classical. Why you shouldn't: we never claimed either. We claimed a rigorously honest, fully reproducible, calibrated-uncertainty system that solves the actual unsolved problem in medical AI — trustworthy doubt — and every number is regenerable in front of you in under a minute. Judge it for what it says it is, and it delivers exactly that."

---

## The five things to never say
1. ❌ "Quantum beats classical." → ✅ "Quantum is real, correctly placed, and competitive; classical wins the fair calibration test."
2. ❌ "We built our own CNN." → ✅ "DenseNet-121 backbone, fine-tuned; the novelty is the fusion/safety/reasoning stack."
3. ❌ "It's ready for hospitals." → ✅ "It's a calibrated copilot; deployment needs prospective validation + regulatory clearance."
4. ❌ "Our accuracy proves it works." → ✅ "Our *calibration and abstention* are the point; here are the numbers with their n and CIs."
5. ❌ "There are no weaknesses." → ✅ "Single-source data, non-radiologist gold, small held-out n — all disclosed in the artifacts."

**Closing line for the pitch:** *"Every other team will show you a model that answers. We built the one that knows when it shouldn't — and you can regenerate every number we've shown you, offline, in sixty seconds."*
