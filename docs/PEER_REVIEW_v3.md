# Publication-level review — *Evaluating quantum machine learning in a clinical decision-support system* (preprint v3, 6 Aug 2026)

Reviewer posture: top-tier venue (npj Digital Medicine / Medical Image Analysis / IEEE TMI / NeurIPS D&B). Harsh, constructive, no protection of feelings.

> **Status note (added after the review was delivered).** Several items here have since been actioned, and the outcomes are recorded in `docs/RESEARCH_PAPER_v4.md` Appendix A.1. In summary: **CRITICAL-1** is resolved by a seven-qubit re-run (§8.3.1) which confirmed the confound — the accuracy and calibration findings did not survive it. **W35** is resolved by a corrected Murphy decomposition (§8.3.2); the original implementation was wrong in the way this review predicted the argument was. **W39/W40** are actioned but produced a weaker result, because the converged checkpoint covers 55 subjects rather than 185. **W2, W16 (partly), W31, W38, W57, W58, W62** are fixed. **CRITICAL-2** is partly addressed: `docs/PROTOCOL.md` now exists and is committed, but it postdates the experiments it governs and no pre-registration exists. **CRITICAL-3** (the abstention provenance), **W19/CRITICAL-4** (target definition), **W25** (trivial baselines) and the external validation of Phase 3 E1 remain outstanding. This review is otherwise left as written, as the record of what prompted those changes.

**Method note.** This review is not impressionistic. Every claim in the manuscript that maps to a released artifact was re-read from `aura/artifacts/` and recomputed where cheap. Findings marked **[VERIFIED]** were confirmed against the artifact named. Findings marked **[VERIFIED — NEW]** are defects the manuscript does not disclose and, as far as I can tell, the author does not know about.

---

## 0. Overall assessment in one paragraph

This is a better paper than most of what it critiques. The thesis — that quantum advantage claims in medical ML are artifacts of asymmetric comparison, demonstrated from inside a system built to find one — is genuinely valuable, timely, and correctly aimed at [1]. The self-retraction in §10.6 is the most credible thing in the manuscript and would, alone, distinguish it. **But the paper does not currently meet the evidentiary standard it demands of others.** Its own protocol (P6, artifact-derived reporting) is violated in at least four places; its single surviving positive claim (§9.1, abstention) rests on an artifact from an undocumented split with calibration parameters that contradict the protocol; its strongest negative result (§8.3, entanglement) is confounded by an evidence channel that is identically zero in all 2,231 studies and is never disclosed; and the pre-specification claim that carries contribution C2 has **no verifiable provenance whatsoever** — the paper is not under version control. A reviewer with repository access will find all of this. A reviewer without it will find enough.

**Current publication readiness: 52/100** for the venues named. **After the Must-Fix list: 74/100.** **After Must-Fix + Strongly Recommended: 86/100.**

---

# PHASE 1 — Deep critical review, section by section

Severity scale: **S1** = rejection-causing on its own. **S2** = major revision, would block acceptance until fixed. **S3** = minor revision. **S4** = polish.

---

## 1.1 The three findings that dominate everything else

I am pulling these out of section order because they change how the rest of the review reads.

---

### ★ CRITICAL-1. The eighth evidence channel is identically zero. **[VERIFIED — NEW]** — S1

**Evidence.** `aura/artifacts/quantum_study_evidence.npz`, all three splits:

```
Xtr  nonzero fraction per channel: [1.0, 1.0, 1.0, 0.972, 1.0, 1.0, 1.0, 0.0]
Xcal nonzero fraction per channel: [1.0, 1.0, 1.0, 0.973, 1.0, 1.0, 1.0, 0.0]
Xte  nonzero fraction per channel: [1.0, 1.0, 1.0, 0.975, 1.0, 1.0, 1.0, 0.0]
channels = [opacity, consolidation, effusion, cardiomegaly, nodule, hyperinflation, pneumothorax, prior_risk]
```

`prior_risk` is 0.0 in **2,231 of 2,231 studies**. Under the stated encoding `RY(π·xᵢ)`, qubit 8 receives `RY(0)` = identity. It carries no data. It is a constant register that the trainable rotations and the entangler act on.

**Why it is a problem.** Three separate load-bearing claims are affected.

1. §4 and §8.4 state that eight qubits is "determined by the input dimension rather than chosen for expressivity," and that the qubit axis is an evidence ablation because "each evidence channel occupies one qubit." The true input dimension is **seven**. The eighth qubit is dead weight, and every cell in the 48-cell sweep at 8 qubits is really a 7-channel model with a spare register.
2. **The entanglement ablation in §8.3 is confounded.** In the ring topology, the constant qubit is CNOT-coupled into the ring on both sides. Entangling a data qubit with a constant register cannot add information, but it *can* consume expressive capacity and distort the reachable function class — which is precisely the effect §8.3 measures and attributes to entanglement in general. The product-state arm is immune: with no entangler, the dead qubit's contribution factorises into the linear head's bias and costs nothing. **You have built an asymmetry into your own capacity-matched ablation.** That the two arms have equal *parameter* counts does not make them equal in *effective* parameter counts once one arm wires a constant into its correlation structure.
3. This is an alternative explanation for the paper's headline negative result that the paper does not consider, cannot currently exclude, and which a hostile reviewer will raise as the first objection.

**How serious.** S1. This is the finding most likely to sink the paper at review, because it attacks the result the abstract leads with ("worse on NLL in ten of ten seeds") and because the fix is obvious enough that a reviewer will not accept its absence.

**Exactly how to fix.**
- Disclose it, in §4 (evidence vector definition), §5.1 (data), and Table 8's caption. State the nonzero fractions above.
- Re-run the ten-seed ablation at **7 qubits with `prior_risk` dropped**. This is ~15 minutes of compute (`train_seconds` ≈ 70/run × 20 runs). Report both. If the ΔNLL sign and magnitude survive, §8.3 becomes materially *stronger* than it is now — you will have shown the effect is not an artifact of a degenerate register, which no comparable paper does. If it does not survive, you have found a sixth negative result about your own experiment, which given this paper's thesis is publishable rather than fatal.
- Re-run the 8-qubit-row cells of the design sweep the same way, or restrict §8.4's claims to what the 7-channel data support.
- Explain *why* `prior_risk` is zero. Is the channel unimplemented? Always-absent in MIMIC-CXR? A serving-path default? The reader needs to know whether this is a bug, a placeholder, or a property of the corpus.

**Expected reviewer comment.** *"The authors report that eight qubits is fixed by the input dimensionality, but inspection of the released evidence tensor shows the eighth channel (`prior_risk`) is identically zero in all 2,231 studies. The entangled ansatz therefore couples data qubits to a constant register while the product ansatz does not. Since the entire Section 8.3 result rests on the claim that only topology differs between arms, this must be resolved before the result can be interpreted. I would want to see the ablation repeated on the seven informative channels."*

**Probability this alone causes rejection: 55%** if unaddressed at a top venue; **~25%** at a mid-tier venue. Near zero if fixed.

---

### ★ CRITICAL-2. Contribution C2 (pre-specification) has no verifiable provenance. **[VERIFIED — NEW]** — S1

**Evidence.** `git log -- docs/RESEARCH_PAPER.md` returns **nothing**. The manuscript is untracked. A repo-wide grep for the protocol text finds it in exactly one file: the paper itself. There is no protocol document, no pre-registration, no OSF/AsPredicted deposit, no commit hash, no timestamp.

**Why it is a problem.** The paper's central methodological contribution is that the comparison protocol was *fixed in advance* (§1.3, §6 preamble, C2, and the abstract's "a comparison protocol fixed in advance"). This is the claim that converts a set of null results into a credible test. It is asserted seven times and evidenced zero times. Worse, the paper explicitly concedes that P7 was added *after* it invalidated a result (§6, P7) — which demonstrates the author understands the distinction and therefore makes the silence about P1–P6's timing conspicuous rather than innocent. A paper whose thesis is that unverifiable claims should not be believed cannot make its own central claim unverifiable.

**How serious.** S1 for a venue that takes pre-registration seriously; S2 elsewhere. It is also the *cheapest* S1 in this review to fix.

**Exactly how to fix.**
1. Put the repository under version control including `docs/`, and commit the protocol as a standalone file (`PROTOCOL.md`) *now*, dated, before any further experiments. Cite the commit hash and date in §6.
2. State plainly in §6 which rules were fixed before which experiments, with dates, and which were not. If P1–P6 predate the 447-study runs, the artifact timestamps (`design_sweep.json` `generated: 2026-07-30T18:35:53Z`, etc.) can corroborate that; use them.
3. If any rule was written after the experiment it governs — say so, as you already do for P7. Honest partial pre-specification is worth far more than an unevidenced claim of full pre-specification.
4. For the external-validation experiments recommended in Phase 3, deposit a timestamped pre-registration. That converts C2 from an assertion into a demonstration.

**Expected reviewer comment.** *"The authors repeatedly state that the protocol was fixed before the comparisons. No evidence is offered: there is no pre-registration, and the manuscript is not version-controlled. Given that the paper's argument is precisely that unverifiable methodological claims should not be trusted, I am unwilling to accept this one on the authors' word. At minimum, provide commit-level or registry-level evidence of the ordering."*

**Probability this alone causes rejection: 30%.** But it caps the paper's ceiling: without it, C2 cannot be credited, and C2 is half the paper's novelty.

---

### ★ CRITICAL-3. The only surviving positive claim comes from an artifact that violates the paper's own protocol. **[VERIFIED — NEW]** — S1

The abstract's closing sentence, contribution C6, §9.1/Table 13, §10.5, and the Conclusion all rest on: *"abstaining on 8% of cases raised accuracy among committed cases by 7 percentage points."*

**Evidence.** Table 13 is read from `aura/artifacts/fusion_ablation.json`, which contains:

| | quantum | classical | learnable |
|---|---:|---:|---:|
| calibration_temperature | 0.7583 | **0.2906** | 0.7000 |
| model_accuracy | 0.6272 | 0.6627 | 0.6036 |
| model_ece | 0.1661 | 0.1632 | 0.1668 |
| agent_abstain_rate | 0.0828 | 0.0059 | 0.0296 |
| agent_accuracy_at_commit | 0.6968 | 0.7024 | 0.6951 |

None of these match the primary n = 447 split (temperatures 0.9057 / 0.4790 / 0.8867; accuracies 0.5324 / 0.5414 / 0.5526; ECE 0.0896 / 0.1266 / 0.0902), and none match the n = 69 benchmark (temperatures 0.8784 / 0.3579; accuracies 0.6377 / 0.6957). The artifact records **no `n`, no split identifier, and no data source.**

Meanwhile `quantum_study.json` → `q3_measurement_budget` reports a *different* abstention analysis, explicitly on the 447-study split: abstain rate **0.1365**, accuracy at commit **0.5725**. The paper cites that artifact in §10.6 and the other in §9.1 without ever noting that they describe the same mechanism on different data with different results.

**Why it is a problem.**
- The paper's rule P1 requires per-arm calibration on the 446-study calibration split. Table 13's temperatures are not those temperatures. So either §9.1 predates P1's enforcement or it uses a different calibration set — and the paper says neither.
- The paper's rule P3 requires disjoint splits at the correct unit. §9.1's unit is unstated.
- The paper's rule P4 requires a test with every ordering claim. §9.1 has no test, no CI, no n.
- This is the paper's *only* positive finding. It is held to a visibly lower evidentiary standard than every negative finding. That inversion is exactly the asymmetry-of-attention failure the paper diagnoses in others (§10.3) — occurring in the paper's own results section.

**How serious.** S1. A reviewer who checks will conclude the author applies the protocol to claims they want to refute and not to the claim they want to keep.

**Exactly how to fix.** Recompute §9.1 end-to-end on the 447-study patient-disjoint split under P1, and report it with the machinery Phase 3 / Experiment E3 specifies (risk–coverage curves, AURC, matched-coverage comparison, and a max-softmax rejector control). Until then, scope the claim to what the artifact supports and label its provenance explicitly. Do not carry it in the abstract in its present form.

**Expected reviewer comment.** *"Section 9.1 is the paper's only positive result and the only one not held to the stated protocol. The supporting artifact records calibration temperatures inconsistent with Table 4, does not state its sample size or split, and reports an abstention rate (8.3%) that disagrees with the 447-study measurement-budget artifact (13.7%). Please recompute on the primary split under P1 and add the significance test P4 requires."*

**Probability this alone causes rejection: 35%.**

---

### ★ CRITICAL-4. The fusion target is a deterministic function of the fusion inputs. **[VERIFIED — NEW]** — S1

**Evidence.** `aura/mimic/labeling.py`, `_map_diagnosis()` (lines 157–193). The six diagnosis labels are not an independent clinical outcome. They are computed by a hand-written weighted score over the *same* CheXpert-style report concepts that produce the seven evidence channels:

```
pneumothorax      → PNEUMOTHORAX   += 1.0
nodule            → MALIGNANCY     += 1.0
edema 0.7 + cardiomegaly 0.5 + effusion 0.2   → HEART_FAILURE
pneumonia 0.9 + consolidation 0.6 + opacity 0.3 → PNEUMONIA
hyperinflation    → COPD           += 0.9
(nothing positive) → NORMAL
ties broken by a fixed acuity priority: pneumothorax > malignancy > heart failure > pneumonia > COPD > normal
```

**Why this is the most consequential finding in the review.** Four things follow, and they are not all bad for the paper.

1. **The primary task is label-recovery, not diagnosis.** The fusion model predicts a deterministic weighted-sum-then-argmax of the concepts whose *predicted probabilities* are its input. The only thing standing between input and target is the vision model's error. This must be stated in §5.1; it reframes every accuracy in the paper.

2. **It explains why the linear head wins, and the explanation strengthens the paper.** The target's Bayes-optimal decision rule is (approximately) an argmax over a linear score of the seven concepts, subject to a priority tie-break. A linear map on the evidence vector is therefore *the correctly specified model class*. The product-of-experts and the variational circuit are both mis-specified relative to a target that is linear-then-argmax by construction. **This is a mechanistic explanation for the null that the paper currently lacks**: there is no nonlinear inter-channel structure in the target for entanglement to exploit, so the theoretical prediction of [11–14] is not merely consistent with the result — it is *derivable* for this task. §8.3's finding becomes a confirmed prediction rather than an observation. This is the single largest available upgrade to the paper's scientific contribution and it requires no new experiments.

3. **It bounds the paper's clinical relevance and must be disclosed.** `MALIGNANCY` is assigned whenever a report mentions a nodule. A nodule mention is not a malignancy diagnosis. Likewise `HEART_FAILURE` from edema + cardiomegaly + effusion is a radiographic pattern, not a clinical diagnosis, and no laboratory, echocardiographic, or outcome data enter. The paper must say plainly that these are *radiographic-pattern proxies derived by rule*, not clinical diagnoses.

4. **It closes an open loop the paper leaves dangling.** §8.2 reports that malignancy has essentially no discriminative signal in any arm (per-class AUROC 0.6415 / 0.5696 / 0.5951). §9.3 reports that nodule is the one finding that fails independent corroboration (AUROC 0.439 against independent labels, κ = −0.031). Given the derivation above, `MALIGNANCY` **is** the nodule concept. These are the same failure, reported in two sections, never connected. Connecting them is a genuine analytic result: the diagnosis-level failure is traced to a specific image-model failure, which is exactly what C5 claims the two-axis design is for.

**How serious.** S1 as an omission; a significant *opportunity* once addressed.

**Exactly how to fix.**
- Add §5.1.2 stating the derivation in full, with the weights, the tie-break, and the patient-level aggregation rule.
- State that §9.3's circularity defence validates the seven *finding* labels and does **not** validate the diagnosis map's weights, which are unvalidated clinical heuristics.
- Add the linear-specification argument (point 2) to §8.2 and §8.3 — it converts an empirical null into an explained one.
- Add the malignancy↔nodule chain (point 4) to §8.2 and §9.3.
- Rename the target throughout from "diagnosis" to "report-derived diagnostic category" or similar, and say why.

**Expected reviewer comment.** *"On inspection of the released labeler, the six-class target is a fixed weighted function of the same concepts that generate the evidence vector. The fusion task is therefore recovery of a known deterministic map, not diagnosis. This must be stated, and it materially changes the interpretation of Section 8 — in particular, it suggests the target is linearly separable in the given features by construction, which would predict exactly the observed result."*

**Probability this alone causes rejection: 40%** if undisclosed. Addressed properly, it *raises* the paper's contribution.

---

## 1.2 Title, Abstract, Keywords

**W1 [S2] — The abstract asserts a comparative conclusion from a non-significant test.** "an ordinary gated linear head attained lower negative log-likelihood than the quantum circuit (p = 0.052)" is used to overturn a significant result (p = 0.006). This is the difference-of-significances fallacy (Gelman & Stern 2006, *The difference between "significant" and "not significant" is not itself statistically significant*, Am. Stat. 60(4):328–331 — **currently uncited, add it**). What the data support is: *quantum and the learnable head are not distinguishable; both beat the product-of-experts.* That is still sufficient for your thesis — the point is that comparator choice determines whether you report a win — but the sentence as written overclaims in the same grammatical form the paper criticises.
- **Fix.** Rewrite to: "a gated linear head was not distinguishable from the quantum circuit (Δ = 0.019, p = 0.052) while beating the product-of-experts decisively (p < 10⁻⁵); the quantum arm's apparent advantage is therefore a statement about which classical model was placed opposite it, not about the quantum model."
- **Expected comment.** *"The authors commit the error they document. p = 0.052 does not establish that the linear head is superior."*
- **P(rejection): 10%,** but it is the single most quotable weakness and will appear in every review.

**W2 [S2] — `p = 7.3 × 10⁻⁶` is untraceable.** **[VERIFIED]** The cited artifact `fusion_headline_n447.json` records `"nll_wilcoxon_p": 0.0` for `classical_vs_learnable`. The value 7.3 × 10⁻⁶ appears nowhere in the repository. Under the paper's own P6 ("every published figure is read from the artifact that produced it") this is a violation, and it is in the abstract.
- **Fix.** Emit full-precision p-values in the artifact (`float(p)` without rounding), or report `p < 1 × 10⁻⁵` and say the artifact stores it below printing precision. The former is better.
- **P(rejection): 5%** alone; contributes to a pattern.

**W3 [S3] — The abstract is 520 words and reads as a results dump.** Six numeric clauses in one sentence (the entanglement sentence runs 95 words). npj DM caps structured abstracts at 250–300 words; MedIA and TMI want ~250.
- **Fix.** Cut to ~280 words. Keep: the null, the comparator-selection mechanism, the entanglement result with its direction split, and the retraction. Move the 13.8× calibration story, the 48-cell sweep, and the kernel result to a single sentence each.

**W4 [S3] — "None of the five quantum claims survived" is stated before the reader knows what the five claims are.** Enumerate them in one clause.

**W5 [S4] — Keywords omit the two terms indexers will search on:** *quantum kernel*, *variational quantum circuit*, *entanglement ablation*, *reproducibility*.

---

## 1.3 §1 Introduction

**W6 [S2] — §1.3 "We tested this proposition from a position that favours a false positive."** This is asserted, not demonstrated, and it is doing heavy rhetorical work (it is the reason the reader should trust the nulls). The evidence *for* it — that the author built the quantum layer, intended an advantage, and published the pre-audit claims — is scattered across §8.1 and §10.6 and never assembled.
- **Fix.** Add three sentences making the adversarial-to-self design explicit and citable: what was believed at the outset, what was published before the audit (the 13.8× claim, the QMBA claim), and what the audit changed. You have the receipts; use them here.

**W7 [S2] — The failure mode is asserted as "not misconduct but asymmetric care," but the paper never distinguishes the two empirically.** In §8.1 the classical arm was left uncalibrated. That is consistent with asymmetric care, but also with a bug. The paper's own §7 shows the served calibration artifact carries the quantum fit — i.e., there was a *production* reason the classical arm was mis-scored. The charitable reading may be right, but it should be argued, not assumed.

**W8 [S3] — §1.2's theory review is well-chosen but concludes too strongly.** "a reported advantage is more plausibly explained by an asymmetry in the comparison than by a physical effect" is a Bayesian claim with no prior stated. Soften to: "the theoretical results above give no mechanism that would predict an advantage at this scale, which shifts the burden onto the comparison design."

**W9 [S3] — §1.4's disclaimer paragraph ("We make no clinical claim…") is buried at the end of the introduction and repeated in §12.3 and the Disclaimer.** Three statements of the same thing. Keep one in §1.4, cross-reference thereafter.

**W10 [S3] — Missing citation.** The introduction's argument about researcher degrees of freedom would be strengthened by Simmons, Nelson & Simonsohn (2011) on false-positive psychology and by Gelman & Loken's garden-of-forking-paths, neither cited. Also missing: Varoquaux & Cheplygina (2022), *Machine learning for medical imaging: methodological failures and recommendations*, npj Digital Medicine 5:48 — this is the single most on-point missing citation in the paper and reviewers at npj DM will notice its absence immediately.

---

## 1.4 §2 Related work

**W11 [S2] — §2.4 and §2.5 are literature dumps with no synthesis.** Twenty-two citations across two subsections, almost all in the form "X et al. [n] show Y." There is no statement of what the *gap* is in the calibration/selective-prediction literature, because there isn't one — the paper correctly disclaims novelty there (§2.4 last line). If you claim no novelty, cut these two subsections to a third of their length and move the detail to §9's positive-control framing.

**W12 [S2] — Missing related work on negative-results QML replication.** The paper positions itself as the first to do an in-system pre-specified quantum-vs-classical comparison (§2.6). That claim needs a defended search. At minimum engage with: Bowles, Ahmed & Schuld (2024), *Better than classical? The subtle art of benchmarking quantum machine learning models* (arXiv:2403.07059) — this is the closest prior art to your entire methodology and its absence is a serious omission; and Schuld & Killoran (2022), *Is quantum advantage the right goal for quantum machine learning?* (PRX Quantum 3:030101), which makes your §1.2 argument at the field level.
- **Expected comment.** *"Bowles et al. (2024) perform systematically what this paper does for one system, and reach compatible conclusions. Its omission makes the novelty claim in §2.6 hard to assess."*
- **P(rejection): 15%** at a QML-literate venue. This is the most damaging missing citation in the paper.

**W13 [S3] — References 24, 25, 27, 29 have no author lists.** The editorial note at the head of the reference list admits three could not be verified. **Do not submit a manuscript containing unverified references.** Either verify them or delete them — they support background sentences only, as the note itself says.

**W14 [S3] — Reference 25 (arXiv:2604.18837), 26 (arXiv:2607.01197), 29 (2026), 30 (arXiv:2604.07896) are dated 2026.** If any of these are not yet peer-reviewed, mark them as preprints in-text where they support a claim, not just in the list.

**W15 [S4] — §2.3 is four sentences and does not justify its own subsection.** Merge into §2.1 or into §10.6.

---

## 1.5 §3 Research gap

**W16 [S2] — Table 1 column 3 overstates several entries.**
- G3 "Execution on `ibm_marrakesh`" — **[VERIFIED]** this is *one case, one job, retrieved after the fact* (`ibm_hardware_run.json`: `"retrieved_after_the_fact": true`, `test_index: 144`). Presented in a table row that answers "81.7% used ideal simulation only," it implies a hardware evaluation. It is a single-instance existence proof. Say so in the cell.
- G4 "Three-rung attribution: sampling 0.0074, device 0.0544, hardware 0.1858" — **[VERIFIED]** all three numbers come from the *same single evidence vector*. n = 1. See W31.
- G6 "Artifact release with SHA-256 manifests… drift test" — **[VERIFIED]** the drift test (`aura/tests/test_doc_numbers.py`) covers `README.md`, `WHAT_IS_AURA.md`, `BENCHMARKS.md`, `KNOWN_LIMITATIONS.md`, `benchmark_report.md`. It **does not cover `RESEARCH_PAPER.md`**. The paper's P6 claim, applied to the paper, is false.
- G2 "one training function, one seed, one split" — for §8.3 this is now ten seeds; the cell is stale.

**Fix.** Rewrite each cell to the literal scope of what was done. This table is the paper's contract with the reader and it is currently written in marketing voice.

**W17 [S2] — G7/G8/G9 are claimed as gaps "we identified rather than inherited," and G8 (statistical inference) is described as "Not reported as present" in [1].** That is a claim about what a review did not report, used as evidence that a field does not do it. Absence of a reported dimension in a systematic review is not evidence of absence in the field. Rephrase to "not assessed as a dimension in [1]."

**W18 [S3] — C5 claims the two-axis design "localises a genuine model failure (nodule detection) to the image model rather than the labels."** This is the best-supported contribution in the paper and it is buried as the fifth of six. Consider promoting it — it is more transferable than C3 or C4.

---

## 1.6 §4 System under evaluation

**W19 [S1] — The prediction target is never defined.** See **CRITICAL-4**, which resolves what the target is (a rule-derived weighted score over report concepts, `aura/mimic/labeling.py::_map_diagnosis`) and explains why the omission matters more than a missing definition normally would. In addition to the fixes listed there: state inter-label exclusivity (the accuracy metric implies single-label — confirm), state how `pneumothorax_dx` differs from the `pneumothorax` evidence channel (they derive from the same concept), and give the class prevalences.

**W20 [S2] — Table 2's parameter counts are asserted without derivation, and the "matched capacity" claim (P2) turns on them.** 102 = 8 qubits × 3 layers × 3 rotations? + linear head 8×6 + 6 = 54? Show the arithmetic. And note that "matched trainable parameters" is not matched capacity in any function-class sense — VC dimension, effective dimension (Abbas et al. 2021, *The power of quantum neural networks*, Nat. Comput. Sci. 1:403), or Fisher-information rank would be the defensible matching. At minimum, acknowledge that parameter counting is a proxy and cite the effective-dimension literature.

**W21 [S2] — The learnable head is described as "Linear map on the raw evidence vector" (Table 2) but as a "learnable **gated** head" in Tables 6 and 7 and §8.2.** Which is it? If gated, it is not linear, and the §8.2 argument "the ordering places the *linear* head first" is wrong. This inconsistency appears five times.

**W22 [S3] — The ensemble backend appears in Table 2 with "—" parameters and is then never evaluated.** Either evaluate it in §8.2 (you have the artifacts: `fusion_ensemble.npz`) or remove it from Table 2. A backend listed and not tested invites the question of whether it was tested and not reported — which, in this paper, is a costly question to invite.

**W23 [S3] — The implementation note about entangler topology not being recoverable from the parameter vector is excellent and is in the wrong place.** It belongs in the reproducibility section or as a boxed "reproducibility hazard." As written it interrupts the system description.

---

## 1.7 §5 Data and setup

**W24 [S2] — Table 3 is missing the fusion training/calibration/test *class* distributions**, which I had to recover from the artifact: test = {normal 102, pneumonia 98, heart_failure 106, copd 67, malignancy 59, pneumothorax_dx 15}. **[VERIFIED]** Add them. They are needed to interpret every accuracy in the paper, and they immediately reveal W25.

**W25 [S1] — There is no trivial baseline anywhere in the paper.** **[VERIFIED — computed from released labels]** Majority-class accuracy on the 447-study test split is 106/447 = **0.2371**. The three backends score 0.5324–0.5526. The paper never reports the majority-class rate, a stratified-random rate, a logistic regression on the eight evidence channels, or an argmax-of-strongest-channel rule.
- **Why it matters.** This is a paper *about baseline selection*. Omitting the trivial baselines from a paper arguing that baseline choice determines conclusions is the most avoidable weakness in the manuscript. It also weakens the paper's own argument: showing that a plain multinomial logistic regression on 8 features matches an 8-qubit VQC is a far more damaging result for QML than showing that a "learnable gated head" does, because no reader can claim the logistic regression was tuned.
- **Fix.** Add a fourth and fifth row to Table 6: majority class, and multinomial logistic regression (sklearn defaults, no tuning), both calibrated under P1 on the same 446-study split. This costs minutes — the evidence tensors are released. Add them to Table 7's pairwise tests.
- **Expected comment.** *"What is chance performance on this task? No trivial baseline is reported."*
- **P(rejection): 25%** alone; near-certain major-revision request.

**W26 [S2] — §5.2 reports the environment but not the training configuration.** Epochs, optimiser, learning rate, batch size, early-stopping criterion, and loss function are given for *no* model in the paper. §8.3 mentions "30 epochs" and §8.4 "20 epochs" in passing; §8.2's arms have no stated epoch count and the artifact does not record one. This is a reproducibility failure in a paper whose sixth contribution is reproducibility.

**W27 [S2] — Three different values are reported for the same model.** **[VERIFIED]** The 8-qubit, 3-layer, ring VQC at seed 7 on the 447-study split appears as:

| Source | Accuracy | NLL | Temperature | Epochs |
|---|---:|---:|---:|---|
| Table 6 (`fusion_headline_n447.json`) | 0.5324 | 1.2853 | 0.9057 | not recorded |
| Table 8, seed 7 (`entanglement_multiseed.json`) | 0.5414 | 1.2682 | 1.1342 | 30 |
| Table 9 (`design_sweep.json`) | 0.5481 | 1.2794 | — | 20 |

And the classical product-of-experts appears as 0.5414 / 1.3026 / T = 0.479 in Table 6 but 0.5145 / 1.3230 / T = 0.7338 in `quantum_study.json`. The paper never reconciles these. A reader will read it as instability; a reviewer with the repo will read it as undocumented configuration drift.
- **Fix.** One configuration table stating, per experiment, the epochs/optimiser/seed/artifact, plus one sentence explaining that Tables 6, 8 and 9 differ in training length and are not intended to be read across. Record epochs in every artifact.

**W28 [S2] — §5.3 lists tests but does not state the multiplicity policy, because there isn't one.** §8.2 runs six tests (3 pairs × 2 metrics), §8.3 runs six more (3 metrics × {t, Wilcoxon}) and repeats them at a second epoch count, §8.5 runs two. No Bonferroni, Holm, or FDR anywhere.
- **Consequence — computed, and it favours the paper.** I applied Holm within each table. Table 8 (Wilcoxon, 3 metrics): adjusted p = 0.006 / 0.0078 / 0.0195 — **all three survive**, including the ECE result the paper reports as real. Table 8 (t, 3 metrics): 0.0027 / 0.0046 / 0.0070 — all survive. Table 7 (Wilcoxon, 3 pairs): 3 × 10⁻⁵ / 0.012 / 0.052 — conclusions unchanged. Table 7 (McNemar, 3 pairs): 1.0 / 1.0 / 0.528 — unchanged. **Every conclusion in §8.2 and §8.3 is robust to Holm at the table level.** Only under a pooled 14-test family does the ECE result lose significance (0.0195 × 14 = 0.27), and pooling across tables is not the natural family here.
- **Fix.** This is therefore a cheap win, not a threat: declare the family per table, apply Holm, report adjusted p alongside raw, and state that conclusions are unchanged. Pre-specify which comparisons are primary and which are exploratory. A reviewer who asks for multiplicity control and receives "we applied Holm; nothing moved" is disarmed.

**W29 [S2] — ECE is reported 20+ times with no bin count, no binning scheme, and no confidence interval.** ECE is bin-count-dependent and biased upward at small n; comparing ECE across models at unstated binning is not interpretable, and n = 447 with 15-shot classes makes the bias non-negligible. The `benchmark.json` reliability curve implies 10 equal-width bins, but the paper never says.
- **Fix.** State the scheme. Report adaptive-binning ECE and a debiased estimator alongside. Add bootstrap CIs on every ECE difference you interpret (Table 6, Table 8, Table 11). Prefer Brier or log-score decompositions where you are making a calibration-vs-refinement argument (see W35).

**W30 [S3] — "All experiments used seed 7"** contradicts §8.3, which uses ten seeds. Reword.

---

## 1.8 §6 Protocol

**W31 [S2] — P6 as stated is false for this manuscript.** **[VERIFIED]** "a test fails the build when documentation and artifacts diverge" — the test does not cover the paper. And the paper contains at least four numbers that do not appear in the artifacts it cites (W2's p-value; Table 13's provenance; Table 17's paths; §11's supports). Either extend `test_doc_numbers.py` to parse `RESEARCH_PAPER.md` (this is a half-day of work and would be a genuinely novel reproducibility contribution worth its own paragraph), or restate P6 to describe what is actually enforced.

**W32 [S2] — P2 as stated ("identical trainable parameter counts") is not what §8.2 does.** Table 6 compares a 54-parameter PoE, a 102-parameter VQC, and a 54-parameter head. Capacity is *not* matched in the paper's primary comparison; it is matched only in §8.3's ablation. The text of P2 should distinguish "matched within an ablation" from "diverse across the comparator set," and §8.2 should state that the quantum arm has ~2× the parameters of both classical arms and still does not win — which is a *stronger* result for your thesis than matched capacity would be. You are underselling your own evidence by mis-stating the rule.

**W33 [S3] — P5 ("publication of negative outcomes") cannot be verified by a reader**, since only the author knows what runs existed. Support it: state how many training runs, sweep cells, and comparisons were executed in total, and confirm the count matches what is reported.

**W34 [S3] — P7's retroactivity is admirably disclosed but is stated twice (§6 and §10.6) in near-identical words.** Keep §6's.

---

## 1.9 §7 and §8 Results

**W35 [S1] — §8.3's central reconciliation argument is asserted, never computed.** The paper's most quotable claim is: "Negative log-likelihood… decomposes into a calibration term and a refinement term. Entanglement improves the first and degrades the second, and because the net movement in negative log-likelihood is unfavourable in every seed, the loss of refinement exceeds the calibration gain."
- The decomposition is **never performed**. It is inferred from the sign of ΔNLL together with the sign of ΔECE. That inference is not valid in general: ECE is not the calibration term of the log score, ΔNLL < 0 with ΔECE < 0 is possible under several configurations, and the "therefore refinement must have degraded by more" step is a subtraction of two quantities measured on different scales.
- **Fix.** Compute the Murphy / DeGroot–Fienberg decomposition (reliability–resolution–uncertainty) of the Brier score, and the corresponding calibration–sharpness decomposition of the log score, per seed. Report both terms in Table 8. **The predictions are already saved** (`fusion_*.npz`); this is an afternoon of work and it converts the paper's headline mechanistic claim from rhetoric into measurement. Cite Murphy (1973) and Bröcker (2009).
- **Expected comment.** *"The calibration/refinement argument is the crux of Section 8.3 and is not measured. Please report the decomposition."*
- **P(rejection): 20%;** near-certain major revision.

**W36 [S2] — The ten-seed interval conditions on a single test set, and the paper does not say so.** **[VERIFIED]** `entanglement_multiseed.json` states it explicitly in its own `purpose` field: *"The published single-seed bootstrap quantifies test-set sampling variability; this quantifies training variability."* Table 8 reports only the across-seed interval and describes it as "95% CI of mean," which a reader will take as total uncertainty. It is not — it excludes sampling error on the 447 studies entirely.
- **Fix.** Either (a) run a nested bootstrap resampling seeds × studies and report the combined interval, or (b) state in Table 8's caption that intervals are conditional on the fixed test split and report the single-seed test bootstrap alongside (you have it: `quantum_study.json` `q1_q2_ablation.bootstrap`, which gives NLL CI [0.0222, 0.0902] — notably ~3× wider than Table 8's [0.0372, 0.0607]). Option (a) is better and is a few hours' work.
- Note that (b) is also *honest about a tension*: the single-seed test-set bootstrap for accuracy has CI [−0.0582, 0.0000], which **includes zero**, whereas Table 8 reports the accuracy effect as significant (p = 0.0023). Both are true of different variance components; the paper reports only the one that is significant. Given this paper's thesis, that must be surfaced, not buried.

**W37 [S1] — Table 10 selects the five best cells *on the test set*.** **[VERIFIED]** `design_sweep.json` evaluates all 48 cells on `test: 447` — the same split used for §8.2, §8.3, §7 and §10.6. Ranking 48 cells by test NLL and reporting the top five is selection on the evaluation set; the resulting ordering is optimistically biased and the "four of five best use no entanglement" claim inherits that bias.
- **Fix.** Rank the sweep on the 446-study calibration split and report test metrics only for the selected cells; or keep the current table and label it explicitly as test-selected with the bias stated. The §8.3 ablation is the load-bearing entanglement evidence anyway — §8.4 should be framed as corroborative and its selection procedure made honest.
- Also disclose that the entire paper reuses one 447-study test split across ≥60 model evaluations, and discuss adaptive-overfitting risk.
- **P(rejection): 20%.**

**W38 [S2] — §8.5's Table 11 is mislabelled.** **[VERIFIED]** The caption says "37 held-out subjects," but `qkl_training_report.json` records `test_slices_used: 2000`, `n: 2000`, `n_positive: 1000`. The metrics are computed over **2,000 slices drawn from 37 subjects, balanced to exactly 50% by construction**. The bootstrap is correctly subject-clustered, which is good, but:
  - the table's n is wrong;
  - AUPRC 0.595 against a 0.50 baseline (balanced) is barely above chance and the paper reports it without the baseline;
  - the balancing is undisclosed and makes accuracy/ECE non-transferable to the true grade prevalence.

**W39 [S1] — §8.5 uses first-epoch embeddings.** **[VERIFIED — NEW]** `qkl_training_report.json` → `embedding_source: ".../brain/embeddings/epoch_001.npz"`. The directory contains `epoch_000` through `epoch_009`. The quantum-kernel comparison — and the conclusion that "glioma grade is poorly separated by these embeddings irrespective of the kernel" — is run on a checkpoint from the second epoch of training.
- **Why it matters.** A null between two kernels on an undertrained representation is a much weaker result than a null on a converged one, because both kernels may be operating in a regime where no kernel could separate. The paper draws a substantive clinical conclusion ("we regard this as the more clinically relevant observation") from it.
- **Fix.** Disclose. Then re-run on `epoch_009` (or whichever checkpoint the deployed system serves) and report both. If the null holds at convergence it is worth far more.
- **P(rejection): 20%.**

**W40 [S2] — Neither kernel is tuned, and the paper cites the literature saying this invalidates the comparison.** **[VERIFIED]** `qkl_training_report.json` records `C: 1.0` and no γ. The RBF has no bandwidth search; the IQP map has no bandwidth/scaling/repetition search. The paper cites Shaydulin & Wild [9] ("kernel bandwidth alone can dominate the observed comparison") and Egginger et al. [10] and then runs both arms at library defaults.
- Equal non-tuning is symmetric but uninformative: it tests two arbitrary points, not two model classes. Reviewers from the QML community will reject §8.5 on this ground alone.
- **Fix.** Nested cross-validation over C ∈ {0.01…100} and γ ∈ {scale, auto, grid} for RBF, and over bandwidth/feature scaling and IQP repetitions for the quantum kernel, selection on the 37 calibration subjects, evaluation on the 37 test subjects. Report the tuned comparison as primary.
- **P(rejection): 25%** at a QML venue.

**W41 [S2] — §8.5's handling of the superseded result is honest but incomplete.** The paper says the earlier 669-slice run gave ΔAUROC = −0.016, p = 0.002, and the current gives −0.0051, p = 0.27, and correctly refuses to select the more striking one. Good. But it does not say *why* the numbers changed — different slice count, different embeddings, different resampling unit? Without that, a reader cannot tell whether the current result supersedes the old one or is a different experiment. State the diff.

**W42 [S2] — §8.2 has no per-class analysis despite noting that two classes have no signal.** "Two classes show essentially no discriminative signal in any arm: malignancy and pneumothorax as a diagnosis." **[VERIFIED]** — per-class AUROC for malignancy is 0.6415 / 0.5696 / 0.5951 (classical/quantum/learnable) and pneumothorax_dx is 0.8353 / 0.8696 / 0.8352 at n = 15. The claim that pneumothorax_dx has no signal is **wrong** — its AUROC is among the highest; it is *rare* (15 cases), not undiscriminated. Fix the sentence and report the per-class table (it is in the artifact).

**W43 [S3] — The MDE of 9.3 accuracy points is reported without its assumptions.** McNemar power depends on the discordance rate. State the assumed discordant proportion, the test used for the calculation, α, and whether it is two-sided.

**W44 [S3] — §8.1's Table 5 is on n = 100** and is never connected to the 447-study or 69-study splits. What is this cohort? It appears nowhere in Table 3.

**W45 [S3] — §7's "exceeds the margin claimed by most published quantum-advantage results in this domain"** is stated twice more (§10.3, §13) and is never supported by a distribution of published margins. Either tabulate the margins from the 16 studies in [1] — which would be a genuinely useful contribution and is a day's work — or delete the claim.

**W46 [S4] — §8.6 "Four experiments, four methods, two corpora, and three splits"** — §8 contains five subsections and five claims. Count consistently.

---

## 1.10 §9 Positive controls

**W47 [S2] — §9.1 has no matched-coverage control and no trivial rejector baseline.** Beyond CRITICAL-3's provenance problem: quantum abstains at 8.28% for +7.0 points, learnable at 2.96% for +9.2, classical at 0.59% for +4.0. These are three points on three different risk–coverage curves and are not comparable. And the paper's claim — "The abstention machinery rather than the fusion model accounts for the improvement" — is unsupported, because there is no comparison against the one-line alternative: thresholding max calibrated probability.
- **Fix.** Full risk–coverage curves for all three backends, AURC and E-AURC (Geifman, Uziel & El-Yaniv 2019), comparison at *matched coverage*, and a max-softmax rejector control. If the four-criterion policy does not beat max-softmax at matched coverage, say so — it would be a sixth negative result and entirely in keeping with the paper.

**W48 [S2] — Table 14 reports AUROC, AUPRC, sensitivity, specificity — and omits PPV and F1, which are much worse.** **[VERIFIED]** `evaluation_retrain_v2/metrics.json` records macro precision **0.2638** and macro F1 **0.3679**. At the stated prevalences (pneumothorax 63/2099 = 3.0%), PPV is the clinically decisive quantity for a screening aid and the paper does not report it. In a paper about selective metric reporting, this will be noticed.
- **Fix.** Add PPV, NPV, and F1 columns. Add the AUPRC baseline (= prevalence) so the reader can see that pneumothorax AUPRC 0.114 sits against a 0.030 floor and nodule 0.165 against 0.045. The per-finding PPVs are in the artifact and are stark: pneumothorax **0.096**, nodule **0.119**, consolidation **0.126**, hyperinflation 0.267, cardiomegaly 0.315, opacity 0.365, effusion 0.560. Macro NPV is 0.957, so the honest framing is a rule-out aid, not a rule-in one — which is a defensible and publishable position, but only if stated.

**W49 [S2] — §9.2's comparison to literature AUROC (0.83–0.89) is not like-for-like and the paper says so, then makes the comparison anyway** in the text and in Figure 4. Either drop the comparison or restrict it to the two findings where label definitions genuinely align, with the definitional differences tabulated.

**W50 [S2] — §9.3's reference standard is a single non-radiologist annotator with no second reader.** The paper discloses this well (better than most). But κ = 0.861 is then used as a *ceiling* on the AUROC claim in §12.2 without an interval. With 66 reports and ≤6 positives in rare-finding cells, the interval on per-finding κ is enormous.
- **Fix.** Report per-finding κ with bootstrap CIs and the positive count per cell. State that rare-finding κ is not estimable at this n. Consider that the 21 keyword-selected supplements make the pooled κ a mixture over two sampling frames — pooling κ across a random stratum and an enriched stratum without weighting is not a valid estimate of labeler agreement on the corpus. This is a real statistical error, not just an imprecision.

**W51 [S3] — §9.3's cross-model check uses `torchxrayvision densenet121-res224-mimic_ch`, which was trained on MIMIC-CXR.** The paper says "whose training labels are independent of ours." The *labels* may be independently derived, but the *images* are the same corpus and plausibly overlapping patients. Independence of labels is not independence of the check. State whether patient overlap was excluded; if not, this bounds the corroboration.

**W52 [S2] — §9.4's sequence-ablation values (Dice 0.58 / 0.52 / 0.28 / 0.02 / 0.00) have no n, no split, no CI, and no explanation of why the four-sequence value (0.58) is 0.33 below the pooled whole-tumour figure (0.915).** "These five values share a scoring basis with one another but not with the pooled figures above" is not an explanation. This is the result that justifies a *safety behaviour* (refusing incomplete studies), so it is the one result in §9 that carries clinical weight, and it is the least documented number in the paper.

**W53 [S3] — §9.4's honesty about Table 16 is a highlight of the paper.** Keep it, and consider adding the per-case 3D metric for your own model so the comparison can actually be made. You have the volumes; per-case 3D Dice is a re-scoring, not a retrain. This converts a paragraph of caveat into a real number and would meaningfully strengthen the paper.

**W54 [S2] — §9.5's noise attribution is n = 1.** **[VERIFIED]** `noise_rung.json` contains a single `evidence_vector` (the same one as the hardware run, `test_index: 144`). The three-rung ladder, the 0.0074/0.0544 split, and the 88.1% figure all derive from one circuit evaluation with no repeats and no interval.
- Additionally, "88.1% attributable to decoherence" is computed as (0.0617 − 0.0074)/0.0617 — a **difference of mean absolute errors**. MAE is not additive across independent error sources and this is not a variance decomposition. Two error channels do not partition an MAE.
- **Fix.** Run the ladder over ≥100 test evidence vectors with ≥10 repeats each; report mean ± CI. Perform the decomposition on **variances** (or on squared error), where additivity is at least approximately defensible, and rename the quantity accordingly. This is cheap (13 s per noisy rung × 100 = ~20 min) and it is the contribution the paper claims is "least represented elsewhere" (C3) — it should not rest on n = 1.
- **P(rejection): 20%.**

**W55 [S3] — C3's "we act on it, reducing two-qubit gate count by 20% through transpilation."** **[VERIFIED]** `transpile_study.json` shows the reduction is entirely from setting Qiskit `optimization_level` 1 → 3 (75 → 60 two-qubit gates; levels 2 and 3 are identical). Describing a compiler flag as an engineering intervention is precisely the kind of inflation this paper polices. Reword to: "raising the transpiler optimisation level from 1 to 3 removes 20% of two-qubit gates at no cost, which the served path now does."

**W56 [S4] — Depth is reported as 146 in the paper and Table, and 147 in `noise_rung.json`.** Trivial, but P6.

---

## 1.11 §10 Discussion, §11 Threats, §12 Limitations

**W57 [S1] — §11 "Five of six diagnosis classes have single-digit support" is factually wrong for the primary analysis.** **[VERIFIED]** On the n = 447 split the supports are 102 / 98 / 106 / 67 / 59 / 15. **None is single-digit.** The statement is true only of the superseded n = 69 benchmark (44/8/7/4/4/2) and is stale text carried from an earlier version.

**W58 [S1] — §11 "the comparison in Section 8.2 is powered only to approximately 15 accuracy points" contradicts §8.2 and the artifact.** **[VERIFIED]** `fusion_headline_n447.json` records `minimum_detectable_accuracy_difference_80pct_power: 0.0934`. §8.2 correctly says 9.3 points. §11 says 15. §12.1 also says 15, correctly scoped to the n = 69 benchmark, but §11 is not so scoped.
- W57 and W58 together mean the paper's *threats to validity* section describes a different experiment than the one reported. This is the most damaging class of error a reviewer can find, because the threats section is where a reader looks to calibrate their trust.
- **P(rejection for W57+W58 together): 25%.**

**W59 [S2] — §12.4 discloses two documentation-drift defects and misses at least three more.** **[VERIFIED]** In addition to the README kernel value and `benchmark_report.md`:
  - `quantum_study.json` → `q3_measurement_budget.note` still asserts the **retracted** claim verbatim: *"No classical fusion model can make this distinction, because classical inference has no measurement budget to vary."*
  - `quantum_study.json` → `honesty.hardware` still reads *"simulated; no quantum hardware was used"*, contradicting §9.5.
  - Table 17's paths are all wrong (see W62).
  A §12.4 that applies P6 to the repository and misses the artifact backing the retraction is self-undermining.

**W60 [S2] — §10.6 is excellent and under-argued in one respect.** The retraction concludes the typing "reduces to a threshold on the analytic margin." That is right. But the paper does not test the stronger and more interesting question: *does the shot-based framing produce a better-ordered abstention set than the margin threshold it reduces to?* If the answer is no, say it; if yes, the surviving contribution in §10.6's last paragraph ("a contribution to the interpretability of a threshold") becomes measurable rather than asserted. One rank-correlation between predicted-shots-to-resolve and margin would settle it.

**W61 [S3] — §12.5 lists CheXpert and BraTS-2021 as "available locally and were not used."** **[VERIFIED]** — both are on disk (`E:\AURA\train.csv` is CheXpert-v1.0-small with 223k rows; `E:\AURA\BraTS2021\` exists). Stating that the single most-demanded experiment is *available and not done* invites the obvious question. Either do it (see Phase 3, E1) or explain the specific obstacle.

**W62 [S2] — Table 17's artifact paths are all wrong.** **[VERIFIED]** The table says `artifacts/…`; the files are at `aura/artifacts/…`. There is no `artifacts/` directory at the repository root. The claim-to-artifact map — the single most important reproducibility table in the paper — does not resolve.

**W63 [S3] — §12 does not list the limitations this review found**: no trivial baseline, undefined diagnosis labels, single test split reused across all analyses, ECE binning unstated, no multiplicity control, first-epoch embeddings, untuned kernels, dead evidence channel, n = 1 noise attribution, n = 1 hardware run.

**W64 [S3] — No fairness or subgroup analysis.** The paper cites Yang et al. [61] on the limits of fair medical imaging AI and performs no subgroup analysis, despite MIMIC-CXR carrying sex, age, race and insurance metadata and CheXpert carrying sex and age. At npj DM or Lancet DH this is not optional.

**W65 [S4] — §10.4's "one non-technical recommendation" is a good sentence in a weak paragraph.** Sharpen: the recommendation is *regenerate, do not inspect*. Say that in five words and stop.

---

## 1.12 Ethics, availability, and general writing

**W66 [S2] — Ethics section does not address the hand-annotation of 66 reports.** A named individual read free-text MIMIC-CXR reports under the PhysioNet DUA. That is permitted, but the manuscript should state (a) that the annotator was credentialed, (b) that no report text is redistributed in the released artifacts, and (c) whether `labeler_validation.json` contains report excerpts. *I did not verify (c) exhaustively and the author must.* If any report text is embedded in a released artifact, that is a DUA breach and must be fixed before release.

**W67 [S2] — "Derived artifacts that do not contain patient data… are released with the code."** The evidence tensors `quantum_study_evidence.npz` contain per-study model outputs and labels for 2,231 MIMIC-CXR studies. These are derived data, arguably permitted, but the paper should state the reasoning and confirm no study identifiers are included.

**W68 [S3] — No clinical author.** For npj DM / Lancet DH, a single-author submission with no clinician co-author, no radiologist over-read, and a self-produced reference standard will attract an editorial desk-reject risk independent of scientific merit. This is worth naming plainly: **recruit a radiologist co-author.** It is the highest-value non-experimental action available.

**W69 [S3] — Reference list editorial note must be removed and its content resolved, not carried.**

**W70 [S4] — Register/voice.** The prose is unusually good — better than most accepted papers — but it is essayistic in places where journals expect declaration ("We note the availability of that reading explicitly, because…"). Some of this is the paper's charm and should survive; some (§8.3's final two paragraphs, §8.5's last three sentences, §13's opening) reads as argument with an imagined critic and will be trimmed by a copy editor anyway.

---

# PHASE 2 — Publication gap analysis

Ranked by how much each missing component costs at the named venues. "Cost" = probability mass of rejection attributable to its absence.

| Rank | Missing component | npj DM | MedIA | TMI | NeurIPS | MICCAI | Sci Rep | Cost | Feasible now? |
|---:|---|:--:|:--:|:--:|:--:|:--:|:--:|---|---|
| 1 | **External validation** (second cohort) | Fatal | Fatal | Fatal | Major | Major | Major | Highest | **Yes — CheXpert + BraTS-2021 on disk** |
| 2 | **Defined prediction target** (W19) | Fatal | Fatal | Fatal | Fatal | Fatal | Fatal | Highest | Yes — documentation |
| 3 | **Trivial + untuned baselines** (W25) | Fatal | Fatal | Major | Fatal | Major | Major | Highest | Yes — minutes |
| 4 | **Pre-specification evidence** (CRITICAL-2) | Major | Major | Major | Major | Minor | Minor | High | Yes — hours |
| 5 | **Dead-channel confound resolution** (CRITICAL-1) | Major | Fatal | Fatal | Fatal | Major | Major | High | Yes — ~1 h compute |
| 6 | **Statistical power / multiplicity** (W28, W43) | Fatal | Major | Major | Major | Major | Minor | High | Yes |
| 7 | **Reader study / clinician comparison** | Fatal | Major | Major | Minor | Minor | Minor | High | No (needs collaborators) |
| 8 | **Subgroup / fairness analysis** (W64) | Fatal | Major | Major | Minor | Minor | Minor | High | **Yes — metadata present** |
| 9 | **Calibration methodology rigour** (W29, W35) | Major | Major | Major | Major | Minor | Minor | Medium | Yes |
| 10 | **Kernel hyperparameter search** (W40) | Minor | Major | Major | Fatal | Major | Major | Medium | Yes |
| 11 | **Hardware evaluation beyond n = 1** (W54, W16) | Minor | Major | Major | Major | Minor | Medium | Medium | Yes — IBM open plan |
| 12 | **Prospective evaluation** | Fatal | Minor | Minor | — | — | — | Medium | No |
| 13 | **Dataset diversity** (2 corpora, 1 country) | Fatal | Major | Major | Minor | Minor | Minor | Medium | Partial |
| 14 | **Per-case 3D segmentation metrics** (W53) | Major | Fatal | Fatal | Minor | Fatal | Major | Medium | Yes — re-scoring only |
| 15 | **Ablation completeness** (ensemble untested, W22) | Minor | Major | Major | Major | Major | Minor | Low | Yes |
| 16 | **Open-science: version control, DOI, container** | Major | Major | Minor | Major | Minor | Minor | Low | Yes |
| 17 | **Novelty framing vs. Bowles et al. 2024** (W12) | Minor | Major | Major | Fatal | Major | Minor | Medium | Yes — citation + positioning |
| 18 | **Deployment/impact evidence** (latency, workflow) | Major | Minor | Minor | — | — | Minor | Low | Partial |
| 19 | **Clinical co-authorship** (W68) | Fatal (desk) | Major | Minor | — | — | Minor | High | Depends on network |

**Reading of this table.** Rows 1, 2, 3, 8 are the difference between "interesting preprint" and "npj DM submission." Rows 2, 3, 5, 6 are the difference between "rejected at MedIA/TMI" and "major revision at MedIA/TMI." Rows 2, 3, 5, 10, 17 are the difference between "desk reject at NeurIPS" and "borderline at NeurIPS D&B."

**What the paper already has that most accepted papers do not:**
- A documented self-retraction (§10.6). Genuinely rare and genuinely valuable.
- A worked demonstration of the confound with before/after numbers (§8.1).
- Two-axis label-provenance validation localising a failure to a specific component (§9.3). This is publishable on its own.
- An explicit refusal to use the more favourable of two runs (§8.5).
- Honest destruction of its own favourable segmentation comparison (§9.4).

These are strengths, and the revision should make them more prominent, not less.

---

# PHASE 3 — Missing experiments

Ordered by value per unit of effort. Everything listed is performable with data and hardware already available; nothing here requires new data collection.

---

### E1. External validation on CheXpert *(MANDATORY)*

- **Objective.** Test whether the §8.2–8.4 conclusions (backend indistinguishability; entanglement penalty) hold on a second cohort from a different institution. This is the single highest-value experiment available.
- **Dataset.** CheXpert-v1.0-small, present at `E:\AURA\train.csv` / `valid.csv` (223,415 studies, Stanford). Different centre, different scanner mix, different labeler version, shared label ontology.
- **Protocol.** Freeze all MIMIC-trained models. Extract the 7 informative evidence channels with the frozen DenseNet on CheXpert frontals. Refit *only* per-arm temperature on a CheXpert calibration split (P1). Evaluate all backends on a patient-disjoint CheXpert test split. Repeat the ten-seed entanglement ablation with CheXpert-derived evidence. **Pre-register before running.**
- **Sample size.** Target ≥ 2,000 test studies; at n = 2,000 the McNemar MDE falls from 9.3 to ≈ 4.4 accuracy points, which converts your accuracy nulls from "undetectable" to "bounded."
- **Metrics.** Accuracy, NLL, ECE (stated binning), Brier with decomposition, macro AUROC, per-class AUROC, conformal coverage and set size.
- **Tests.** McNemar (Holm-corrected over pairs), Wilcoxon on per-study NLL, DeLong on AUROC, subject-clustered bootstrap on all deltas.
- **Figures.** Forest plot of Δ(metric) per backend pair, MIMIC vs CheXpert side by side. Reliability diagrams per backend per cohort.
- **Tables.** CheXpert replicate of Tables 6, 7, 8.
- **Quality uplift.** **+18 points.** Converts the paper from single-centre to externally validated and directly answers the [60] objection the paper itself raises. Without this, npj DM and Lancet DH are closed.

---

### E2. Seven-channel re-run of the entanglement ablation *(MANDATORY — resolves CRITICAL-1)*

- **Objective.** Determine whether the entanglement penalty is an effect of the CNOT ring or an artifact of coupling data qubits to the constant `prior_risk` register.
- **Dataset.** Existing `quantum_study_evidence.npz`, `prior_risk` column dropped → 7 channels, 7 qubits.
- **Protocol.** Identical ten-seed protocol (seeds 7…41, 30 epochs, same training function). Two arms: ring vs product at 7 qubits. Also re-run 8-qubit arms for direct comparison.
- **Sample size.** 10 seeds × 2 topologies × 2 widths = 40 runs ≈ 50 min at the recorded ~70 s/run.
- **Metrics/tests.** As Table 8, plus Holm correction; plus the Brier decomposition from E4.
- **Figures.** Extend Figure 1 to a 2×3 panel: {7-qubit, 8-qubit} × {ΔNLL, Δacc, ΔECE}.
- **Quality uplift.** **+10 points**, and removes the most likely rejection cause. If the effect survives at 7 qubits it is the strongest result in the paper.

---

### E3. Risk–coverage analysis of selective prediction *(MANDATORY — resolves CRITICAL-3)*

- **Objective.** Establish, on the primary split under P1, (a) how much abstention buys, (b) whether the four-criterion policy beats a max-softmax threshold at matched coverage, and (c) whether backend choice matters once coverage is matched.
- **Dataset.** 447-study patient-disjoint test split, all three backends, P1 calibration.
- **Protocol.** Sweep coverage from 100% to 50%. Four rejectors: max calibrated probability; conformal set size; the paper's four-criterion policy; and an oracle rejector as an upper bound.
- **Metrics.** Risk–coverage curve, AURC, E-AURC, selective accuracy at coverage ∈ {0.95, 0.90, 0.80}, plus coverage-matched pairwise deltas.
- **Tests.** Bootstrap CI on AURC differences (2,000 resamples, study-level); paired test at each matched coverage.
- **Figures.** One risk–coverage figure with four curves × three backends. This should replace Table 13.
- **Quality uplift.** **+8 points.** Turns the paper's only positive claim into its best-evidenced one, and is likely to yield a sixth honest negative result (policy ≈ max-softmax).

---

### E4. Proper-score decomposition of the entanglement effect *(MANDATORY — resolves W35)*

- **Objective.** Measure, rather than infer, the calibration-vs-refinement trade the paper's headline argument depends on.
- **Dataset.** Saved per-seed predictions from E2 / existing multiseed runs.
- **Protocol.** Per seed and arm, compute the Murphy decomposition of the Brier score (reliability, resolution, uncertainty) and the calibration–sharpness decomposition of the log score. Report Δreliability and Δresolution alongside ΔNLL.
- **Metrics/tests.** One-sample t and Wilcoxon across seeds on each decomposition term; Holm across terms.
- **Tables.** Two extra columns in Table 8.
- **Quality uplift.** **+6 points.** Converts §8.3's crux from rhetoric to measurement. Cheapest high-value experiment in this list.

---

### E5. Trivial and untuned classical baselines *(MANDATORY — resolves W25)*

- **Objective.** Bound the task from below and strengthen the comparator-selection argument.
- **Protocol.** Add to Table 6: majority class (0.2371, already computed); multinomial logistic regression on the 7 informative channels at sklearn defaults; gradient-boosted trees at defaults; and a single-channel argmax rule. All calibrated under P1.
- **Metrics/tests.** As Table 6/7 with Holm correction.
- **Quality uplift.** **+7 points.** A logistic regression matching an 8-qubit VQC is a more damaging and more quotable result than anything currently in §8.2.

---

### E6. Tuned quantum-vs-classical kernel comparison on converged embeddings *(MANDATORY — resolves W39, W40)*

- **Objective.** Test whether the kernel null survives (a) a trained representation and (b) symmetric hyperparameter optimisation.
- **Dataset.** `brain/embeddings/epoch_009.npz` (or the served checkpoint) — present on disk. 111/37/37 subject split retained.
- **Protocol.** Nested CV: outer = subject-disjoint test; inner = grid over C for both arms, γ for RBF, bandwidth/scaling and IQP repetitions for the quantum map. Report at the true grade prevalence *and* balanced, with the AUPRC baseline stated.
- **Sample size.** Report subject-level n explicitly; retain subject-clustered bootstrap (2,000 resamples).
- **Metrics/tests.** AUROC (DeLong + clustered bootstrap), AUPRC vs baseline, ECE with stated binning, Brier.
- **Quality uplift.** **+6 points.** Currently §8.5 is the weakest of the five results; this makes it the most methodologically defensible.

---

### E7. Noise attribution at n ≥ 100 with a variance decomposition *(STRONGLY RECOMMENDED — resolves W54)*

- **Objective.** Give C3 a sample size.
- **Protocol.** Run the three-rung ladder over 100 test evidence vectors × 10 repeats at 4,096 shots. Decompose squared error (not MAE). Report mean ± 95% CI per rung and the sampling/decoherence variance shares.
- **Figures.** Replace Figure 3's left panel with a distribution (violin or box) per rung.
- **Quality uplift.** **+5 points.** Turns the paper's self-described most-distinctive methodological element from an anecdote into a measurement.

---

### E8. Hardware evaluation at n ≥ 50 cases *(STRONGLY RECOMMENDED — resolves W16/G3)*

- **Objective.** Report top-1 agreement between hardware and analytic posterior with an interval, rather than one case.
- **Protocol.** 50 stratified test cases on `ibm_marrakesh` at the served transpilation level. Report top-1 agreement rate (Jeffreys CI), mean |Δ⟨Z⟩|, mean posterior KL from analytic, and per-class breakdown.
- **Quality uplift.** **+5 points**, and it is the one row of Table 1 currently carried by n = 1.

---

### E9. Subgroup analysis on sex, age band, and race *(STRONGLY RECOMMENDED — resolves W64)*

- **Objective.** Report differential performance, as the venues the paper targets require.
- **Dataset.** MIMIC-CXR patient metadata; replicate on CheXpert (sex, age).
- **Metrics.** Per-subgroup AUROC and calibration for the vision model and the fusion backends; equalised-odds gap; calibration gap.
- **Quality uplift.** **+5 points** at npj DM / Lancet DH; **+2** elsewhere. Mandatory if targeting a clinical journal.

---

### E10. Per-case 3D segmentation re-scoring *(STRONGLY RECOMMENDED — resolves W53)*

- **Objective.** Make Table 16 an actual comparison instead of a caveat.
- **Protocol.** Reassemble per-volume predictions, compute per-case 3D Dice and HD95, average across cases, per BraTS convention.
- **Quality uplift.** **+4 points**, and removes a table the paper currently has to apologise for.

---

### E11. Reader study *(OPTIONAL — required only for npj DM / Lancet DH)*

- **Objective.** Clinical utility of abstention: does a radiologist-plus-AURA arm beat radiologist alone at matched reading time?
- **Protocol.** 3 readers × 200 cases, crossover with washout, MRMC ROC analysis (Obuchowski–Rockette / Dorfman–Berbaum–Metz).
- **Sample size.** ~200 cases × 3 readers for adequate MRMC power at ΔAUC ≈ 0.05.
- **Quality uplift.** **+15 points at npj DM specifically**; near zero at NeurIPS/MICCAI. Not feasible without clinical collaborators — treat as the gate for clinical-journal submission, not as a revision item.

---

### E12. Margin-vs-shots rank correlation *(OPTIONAL — resolves W60)*

- **Objective.** Test the surviving claim in §10.6 (shot count is a more actionable expression of the margin).
- **Protocol.** Spearman ρ between predicted-shots-to-resolve and analytic margin over the 447 studies; if ρ ≈ −1, the surviving contribution is purely presentational and should be scoped to that.
- **Quality uplift.** **+2 points**, but high honesty value given the retraction context.

---

# PHASE 4 — Reviewer simulation

---

## Reviewer 1 — Medical imaging / clinical AI (senior, npj DM editorial board)

**Summary.** The authors build a chest-radiograph and brain-MRI decision-support system, embed quantum components, and report that five quantum claims fail under a symmetric comparison protocol. The paper is well written and unusually candid, including a retraction of the authors' own proposed mechanism.

**Strengths.**
1. The self-retraction in §10.6 is exemplary and rare.
2. §9.3's two-axis label-provenance design, which localises the nodule failure to the image model rather than the labeler, is a genuine methodological contribution independent of the quantum content.
3. §9.4's refusal to claim a favourable segmentation comparison is the right call and well argued.
4. Complete artifact release with a claim-to-artifact map.

**Weaknesses.**
1. **The prediction target of the primary experiment is never defined**, and on inspecting the released labeler it turns out to be a hand-weighted rule over the same report concepts that generate the model's input features. `MALIGNANCY` is assigned on a nodule mention. These are radiographic-pattern proxies, not clinical diagnoses, and the paper presents them as diagnoses throughout. §9.3 validates the *finding* labels; it does not validate this map.
2. **No external validation**, on a paper that cites [60] on exactly this point and states in §12.5 that the required cohort is already on the authors' disk.
3. **No trivial baseline.** What is chance? What does logistic regression on eight numbers achieve?
4. **No subgroup analysis**, despite citing [61].
5. §11 describes an experiment that is not the one reported (single-digit class support; 15-point power). This is a serious lapse in the section a reader relies on most.
6. Single author, no clinician, self-produced reference standard by a non-radiologist.
7. Table 14 omits PPV and F1 at prevalences where PPV is the decisive quantity.

**Major comments.** Define the target. Externally validate. Add trivial baselines. Fix §11. Add PPV. Recruit a clinical co-author.

**Minor comments.** Abstract too long; Table 17 paths do not resolve; reference list contains an editorial note and unverified entries.

**Questions.** (i) How are the six diagnoses defined and validated? (ii) Is the label single-exclusive? (iii) Why was CheXpert not used? (iv) What is the abstention analysis' sample size?

**Decision.** **Reject** (npj DM); resubmission encouraged after external validation and clinical co-authorship.
**Confidence.** 4/5.

---

## Reviewer 2 — Quantum machine learning (theory + benchmarking)

**Summary.** An applied QML null-results paper with a pre-specified protocol, an entanglement ablation, a kernel comparison, and hardware execution. Conclusions align with the theoretical literature.

**Strengths.**
1. Correct framing against [11–14]; the authors understand why an advantage was not expected.
2. The comparator-selection demonstration in §8.2 is the paper's best idea and is well executed.
3. Hardware execution and noise attribution are more than most applied QML papers do.
4. Reporting the calibration result that *favours* entanglement, then explaining why it does not carry, is good practice.

**Weaknesses.**
1. **The ablation is confounded.** The released evidence tensor has an identically-zero eighth channel. The entangled ansatz couples data qubits to a constant register; the product ansatz does not. Parameter-count matching does not control for this. The headline result must be repeated at seven qubits.
2. **Bowles, Ahmed & Schuld (2024) is not cited.** That paper does systematically what this one does for a single system. The novelty claim in §2.6 cannot be assessed without engaging it. Schuld & Killoran (PRX Quantum 2022) is also missing.
3. **Neither kernel is tuned.** The authors cite Shaydulin & Wild on bandwidth dominance and then run both arms at `C = 1.0` with no bandwidth search. §8.5 is uninformative as it stands.
4. **The kernel embeddings are from training epoch 1** of ten available checkpoints. Undisclosed.
5. **Model selection on the test set** in §8.4: 48 cells evaluated on the same 447 studies used for every other result, then the top five reported.
6. No multiplicity correction anywhere. (Applying Holm within each table leaves every conclusion intact — so declare the policy and report the adjusted values; the omission costs the authors nothing but currently looks like an oversight.)
7. "Matched capacity" via parameter count is a weak notion; effective dimension (Abbas et al. 2021) would be defensible.
8. Noise attribution is n = 1 and subtracts MAEs, which is not a decomposition.
9. Hardware result is n = 1, retrieved after the fact.

**Major comments.** Fix the dead channel. Cite and position against Bowles et al. Tune both kernels. Use the final embeddings. Correct for multiplicity. Rank the sweep off-test.

**Minor comments.** Derive the parameter counts; state the epoch counts; the "20% gate reduction" is a compiler flag.

**Questions.** (i) Why is `prior_risk` always zero? (ii) Does the entanglement effect survive at 7 qubits? (iii) What are the training hyperparameters?

**Decision.** **Major revision.** The core contribution is sound and the field needs this paper; the specific execution has correctable defects.
**Confidence.** 5/5.

---

## Reviewer 3 — Statistics / ML methodology and reproducibility

**Summary.** A negative-results paper resting on a pre-specified comparison protocol, with claims of artifact-derived reporting.

**Strengths.**
1. The paper reports effect sizes, intervals, and minimum detectable effects rather than bare p-values — better than most submissions.
2. §8.5's refusal to report the more favourable of two runs is exactly right.
3. Clustering the bootstrap at the correct unit (subjects, not slices) is done correctly and explained well.

**Weaknesses.**
1. **The pre-specification claim has no evidence.** No pre-registration, no version control on the manuscript, no protocol artifact, no timestamps. The paper's central contribution is unverifiable by its own standard.
2. **No multiplicity control** across ~14 hypothesis tests. (I checked: Holm within each table changes nothing. Report it.)
3. **Difference-of-significances fallacy** in the abstract and §8.2 (p = 0.006 vs p = 0.052). Gelman & Stern (2006) uncited.
4. **The ten-seed interval conditions on one test set** and is presented as total uncertainty; the artifact's own `purpose` field says so. The single-seed test bootstrap gives an accuracy CI that includes zero while Table 8 reports significance — both are reported in the repository, only the significant one in the paper.
5. **ECE reported ~20 times with no bin count and no CI**, and interpreted comparatively.
6. **§8.3's calibration/refinement argument is never computed.**
7. **One test split reused across ≥60 evaluations**, including a 48-cell selection.
8. **§11 misstates the reported experiment** (class support, power).
9. **A p-value in the abstract does not appear in the cited artifact.**
10. **The drift test invoked by P6 does not cover the manuscript.**
11. §9.3 pools κ across a uniform-random stratum and a keyword-enriched stratum without weighting — that is not a valid corpus-level agreement estimate.

**Major comments.** Provide pre-specification evidence. Apply Holm. Fix the fallacy. Report combined uncertainty. State ECE binning. Compute the decomposition. Fix §11. Fix the traceability failures.

**Minor comments.** State the MDE assumptions; the n = 100 cohort in Table 5 is undefined.

**Questions.** (i) When were P1–P6 written, and what is the evidence? (ii) How many runs were executed in total (P5)? (iii) What binning is used for ECE?

**Decision.** **Major revision**, bordering reject. Every defect is correctable; there are many of them, and they are concentrated in the paper's own claimed contribution.
**Confidence.** 5/5.

---

## Associate Editor

**Assessment.** Three reviewers converge: the paper is worth publishing and is not yet publishable in this form. R2 and R3 recommend major revision; R1 rejects on clinical-venue grounds specific to npj DM (no external validation, no clinician, undefined target).

The unusual feature of this submission is that its defects are of exactly the type it documents in others. Reviewers 2 and 3 have both found instances where the authors' protocol is not applied to the authors' own favourable claim, or where a result is reported at the variance component that makes it significant. That is not fatal — the paper's whole argument is that this happens to careful people — but it means the revision must be visibly thorough, and the authors should address it directly rather than quietly.

I would encourage resubmission after: defining the target variable; external validation; resolving the constant-channel confound; adding trivial baselines; providing pre-specification evidence; and applying multiplicity correction. I would also strongly encourage a clinical co-author and a statistical co-author. As a single-author submission the paper carries no independent check, which is a structural weakness given its subject matter.

**Recommendation.** **Reject and resubmit** at this venue; **major revision** if submitted to a methods-oriented venue.

---

## Acceptance probability estimates

**As submitted (v3):**

| Venue | P(accept) | Binding constraint |
|---|---:|---|
| Nature Digital Medicine / npj Digital Medicine | **2%** | No external validation, no clinician, undefined target, no subgroup analysis |
| Lancet Digital Health | **1%** | Same, plus no prospective component |
| Medical Image Analysis | **6%** | Confounded ablation, no external validation, test-set selection |
| IEEE TMI | **5%** | Same; TMI also expects methodological novelty |
| Nature Scientific Reports | **22%** | Survives on soundness; §11 errors and traceability failures are the risk |
| NeurIPS (main track) | **5%** | No methodological novelty; QML reviewers will find the dead channel |
| NeurIPS Datasets & Benchmarks | **14%** | Better fit; still needs Bowles positioning and multiplicity fixes |
| MICCAI | **9%** | MICCAI rewards methods; negative results are a hard sell |
| CVPR Workshops (medical/quantum) | **65%** | Low bar; paper is well above workshop average |
| IEEE conference (general) | **80%** | — |
| IIT / institutional conference | **93%** | — |

**After the Must-Fix list (Phase 7, Tier 1) only:**

| Venue | P(accept) |
|---|---:|
| npj Digital Medicine | 8% |
| Medical Image Analysis | 22% |
| IEEE TMI | 20% |
| Scientific Reports | 55% |
| NeurIPS D&B | 30% |
| MICCAI | 20% |
| **TMLR** | **50%** |
| **Quantum Machine Intelligence** | **55%** |
| **IEEE Trans. Quantum Engineering** | **50%** |

**After Must-Fix + Strongly Recommended (E1–E10, clinical co-author):**

| Venue | P(accept) |
|---|---:|
| npj Digital Medicine | 30% (55% with E11 reader study) |
| Medical Image Analysis | 45% |
| IEEE TMI | 40% |
| Scientific Reports | 80% |
| NeurIPS D&B | 45% |
| MICCAI | 35% |
| TMLR | 75% |
| Quantum Machine Intelligence | 80% |

**Venue recommendation.** The paper's natural homes, given what it is, are **TMLR** (no novelty requirement, explicitly welcomes negative results, open review suits the reproducibility argument), **Quantum Machine Intelligence**, and **IEEE Transactions on Quantum Engineering**. Target npj DM only after E1, E9, E11 and a clinical co-author. Do not target MICCAI or NeurIPS main — the paper's contribution is methodological hygiene, which those venues systematically undervalue.

---

# PHASE 5 — Scientific improvements (no fabricated results)

Ordered by novelty gained per unit of honest effort.

**S1. Make the protocol the artifact, not the prose.** Extract §6 into a machine-checkable `PROTOCOL.md` plus a test harness that verifies, for any quantum-vs-classical comparison, that: both arms were calibrated on their own split; parameter counts are recorded; splits are disjoint at a declared unit; every ordering claim has an attached test; and every published number resolves to an artifact. Release it as a standalone tool. **This is the paper's most transferable contribution and it currently exists only as seven paragraphs.** A protocol others can *run* is a different order of contribution from a protocol others can read.

**S2. Extend the drift test to the manuscript (W31).** A build that fails when a paper's numbers diverge from its artifacts is, as far as I know, not standard practice anywhere. Implement it, describe it in one paragraph, and it becomes a named contribution rather than an unenforced rule.

**S3. Quantify the confound headroom against the literature (W45).** The paper claims three times that the 1.89 temperature spread "exceeds the margin claimed by most published quantum-advantage results in this domain." Extract the reported margins from the 16 studies synthesised in [1], tabulate them, and show the comparison. This converts the paper's most rhetorically effective claim into a measurement, and it is a day's work with no new compute.

**S4. Reframe §9.3 as a standalone contribution.** The two-axis design — labeler-vs-gold and model-vs-independent-model — that *localises* a failure to a component is generalisable well beyond this system and beyond QML. Give it a name, a schematic figure, and a paragraph on when each axis is diagnostic. It may be more citable than the quantum content.

**S5. Turn the retraction into a method.** §10.6 currently narrates one instance. Generalise it: what class of claims is unfalsifiable from metrics? (Capability claims, expressivity claims, "no classical analogue" claims.) What is the audit procedure? Give three criteria and apply them retrospectively to the claims in [1]'s 16 synthesised studies — how many are capability claims that metrics cannot falsify? That is a striking, honest, entirely feasible analysis.

**S6. Strengthen the negative result by making the classical arm indefensibly weak-looking-but-strong.** Add logistic regression at sklearn defaults (E5). "An 8-qubit variational circuit with 102 parameters does not beat `LogisticRegression()` on seven numbers" is a sentence that will be quoted. The current "learnable gated head" comparator invites the response that it was tuned.

**S7. Report the negative result about your own abstention policy, if E3 finds one.** If the four-criterion policy does not beat max-softmax at matched coverage, that is a sixth negative result, it is about a classical component, and it demonstrates that the protocol was applied symmetrically to the author's own favoured mechanism. That would answer the AE's structural criticism more effectively than any rebuttal.

**S8. Make the clinical scoping constructive rather than defensive.** The paper's clinical content is currently a list of things it is not. Replace with one concrete statement of what a deployment would require: the external cohort, the reader study design, the subgroup thresholds, and the abstention rate a department would tolerate. Two paragraphs, no new experiments, and it converts §12.3 from apology into roadmap.

**S9. Add the mechanistic explanation for the entanglement penalty — it is now available for free.** CRITICAL-4 shows the target is an argmax over a fixed weighted sum of the seven concepts. The Bayes-optimal rule for this task is therefore linear in the evidence vector up to a tie-break, which *predicts* that (a) a linear head is the correctly specified model, (b) a correlation-modelling ansatz spends capacity on inter-channel structure the target does not contain, and (c) entanglement should hurt refinement while leaving calibration roughly intact — which is precisely the pattern in Table 8. Stating this converts §8.3 from an empirical observation into a confirmed prediction, and converts the paper's relationship to [11–14] from "consistent with" to "derivable in this instance." Corroborate cheaply with the conditional mutual information between evidence channels given the label, measurable from the released training tensor. **This is the highest-value paragraph the author can add to this paper.**

**S10. Do not add a new quantum method.** The temptation after a null is to propose a fix. Resist it. The paper's value is that it did not.

---

# PHASE 6 — Writing improvements

**Redundant text (cut).**
- The clinical disclaimer appears in §1.4, §12.3, and the Disclaimer. Keep §1.4 + Disclaimer.
- "The temperature spread across backends is 1.89, which exceeds most published quantum-advantage margins" appears in the abstract, §7, §10.3, and §13. Keep §7 and one reference back.
- Gupta et al. [1]'s statistics are recited in the abstract, §1.1, §3.1's table, §10.2, and §13. Keep §1.1 and Table 1; cross-reference elsewhere.
- §2.4 and §2.5 are ~700 words supporting a "we claim no novelty" statement. Cut by two-thirds.
- P7's retroactivity stated in §6 and §10.6.
- §12.5 and §11's external-validity paragraph overlap.

**Weak arguments.**
- §8.2's conclusion from p = 0.052 (W1).
- §8.3's calibration/refinement reconciliation, asserted not computed (W35).
- §9.1's "the abstention machinery rather than the fusion model accounts for the improvement" — no control (W47).
- §1.2's "more plausibly explained by" — no prior stated (W8).
- §7's "exceeds the margin claimed by most published results" — unsupported (W45).

**Poor transitions.**
- §7 → §8.1 is abrupt; §7 measures headroom and §8.1 reports a historical audit. One linking sentence.
- §8.5 → §8.6: the summary arrives without warning that §8.5 was the last result.
- §9.4 → §9.5: hardware follows segmentation with no bridge; §9.5's own first line apologises for the placement, which is a sign it is in the wrong section. Move §9.5 to its own §10 or fold into §8 as an infrastructure subsection.

**Overly long paragraphs.**
- Abstract Results (~350 words in one block) → three paragraphs or a structured list.
- §8.2's "It does not survive the introduction of a second classical baseline" paragraph → split at the ordering claim.
- §10.6's three-consequences paragraph → numbered list (it already reads as one).
- §12.3 → split generalisation from modality-router limitations.

**Passive voice worth converting.** §5.3 entirely ("Paired accuracy comparisons used…" is fine; "were assessed by bootstrap resampling" → "we resampled"). §6's P1–P7 are written in a mixture of imperative and passive; make them uniformly imperative — they are rules.

**Claims without evidence (must be cited, measured, or cut).**
- "exceeds the margin claimed by most published quantum-advantage results" (§7, §10.3, §13).
- "We tested this proposition from a position that favours a false positive" (§1.3).
- "the protocol… fixed before the comparisons were run" (throughout) — CRITICAL-2.
- "This is the mechanism described by Wolfrath et al. [73] and by the weak-baseline literature" (§8.2) — those papers describe weak baselines, not this specific calibration asymmetry; the mapping needs one sentence of argument.
- "8 qubits is determined by the input dimension" (§8.4) — false (CRITICAL-1).
- "the abstention machinery rather than the fusion model accounts for the improvement" (§9.1).

**Where figures would help.**
- **A risk–coverage figure** replacing Table 13 (currently the paper's weakest evidence presented as its strongest claim).
- **A forest plot** of all five negative results on one axis with intervals — this would be the paper's headline figure and it does not exist.
- **Reliability diagrams** per backend (you have the data in `benchmark.json`'s reliability curves).
- **A schematic of the two-axis validation design** in §9.3.
- **A CONSORT-style flow diagram** for data: MIMIC-CXR studies → filtered → split → evidence extraction → fusion. Clinical reviewers expect this and its absence is conspicuous.
- **A timeline figure** for the protocol: when each rule was fixed relative to each experiment. This would make CRITICAL-2 into a strength.

**Where equations are needed.**
- The fusion model. The paper defines no model formally. Give the PoE posterior, the VQC readout (`p = softmax(W⟨Z⟩ + b)`), and the learnable head in three lines.
- Temperature scaling and the per-arm fitting objective.
- ECE with its binning, explicitly.
- The Brier/log-score decomposition (E4) once computed.
- The abstention rule as a predicate, not prose.
- The IQP feature map.

Currently the paper contains **one** equation (the shot variance, in the section that retracts the claim it supports). For a methods paper in these venues that is far too few.

**Sections to merge.**
- §2.3 into §2.1 or §10.6.
- §11 (Threats) into §12 (Limitations) — they overlap substantially and §11's errors partly arise from maintaining two lists of the same concerns.
- §8.6 into §10.1.

**Sections to split.**
- §5.1 → data + **label derivation** (W19).
- §9.5 → move out of "positive controls"; it is infrastructure, not a control.
- §12.4 → its own short "reproducibility defects found in this project" subsection; it is a contribution, not a limitation.

**Structural recommendation.** Move §7 (confound size) to become §8.1, and move current §8.1 (the historical audit) to §7. The audit motivates the protocol; the headroom measurement is a result. They are currently in the reverse of their logical order, which is why §7 reads as an interruption.

---

# PHASE 7 — Prioritised roadmap

## Tier 1 — MUST FIX BEFORE SUBMISSION (any venue)

| # | Item | Ref | Effort |
|---:|---|---|---|
| 1 | Define the six-diagnosis target in full (weights, tie-break, aggregation); state that it is a rule-derived radiographic-pattern proxy, not a clinical diagnosis; state that §9.3 does not validate the diagnosis map | CRITICAL-4, W19 | 3 h writing |
| 1b | Add the linear-specification argument and the malignancy↔nodule chain — the two free scientific upgrades that follow from CRITICAL-4 | CRITICAL-4 | 3 h writing |
| 2 | Disclose the constant `prior_risk` channel; re-run the ten-seed ablation at 7 qubits | CRITICAL-1, E2 | 1 h compute |
| 3 | Recompute §9.1 on the primary split under P1 with risk–coverage analysis and a max-softmax control | CRITICAL-3, E3 | 1 day |
| 4 | Provide pre-specification evidence: version-control the repo, commit `PROTOCOL.md`, cite hashes and dates, state honestly what was fixed when | CRITICAL-2 | 3 h |
| 5 | Add trivial + untuned baselines (majority 0.2371, logistic regression) to Tables 6–7 | W25, E5 | 2 h |
| 6 | Fix §11: class supports and power figure are wrong for the reported experiment | W57, W58 | 15 min |
| 7 | Apply Holm correction across each test family; report adjusted p | W28 | 2 h |
| 8 | Fix the difference-of-significances framing in the abstract and §8.2; cite Gelman & Stern | W1 | 1 h |
| 9 | Compute the Brier/log-score decomposition for §8.3 | W35, E4 | 4 h |
| 10 | State ECE binning; add bootstrap CIs to every interpreted ECE difference | W29 | 3 h |
| 11 | Disclose that §8.5 uses epoch-1 embeddings and 2,000 balanced slices; fix Table 11's stated n | W38, W39 | 30 min |
| 12 | Tune both kernels symmetrically, or restrict §8.5's conclusion to "two untuned kernels" | W40, E6 | 1 day |
| 13 | Fix Table 17's paths; make the p-value in the abstract traceable; extend or restate P6 | W2, W31, W62 | 3 h |
| 14 | Disclose test-split reuse; rank the §8.4 sweep off-test or label the selection bias | W37 | 2 h |
| 15 | State the ten-seed interval's conditioning; report both variance components | W36 | 2 h |
| 16 | Add PPV/F1 to Table 14; add AUPRC baselines | W48 | 30 min |
| 17 | Cite and position against Bowles et al. (2024) and Schuld & Killoran (2022); add Varoquaux & Cheplygina (2022), Gelman & Stern (2006) | W12, W10 | 3 h |
| 18 | Remove the reference-list editorial note; verify or delete refs 24, 25, 27, 29 | W13, W69 | 2 h |
| 19 | Correct C3's transpilation framing; correct §8.2's pneumothorax "no signal" claim; correct the linear/gated head inconsistency | W55, W42, W21 | 1 h |
| 20 | Fix §12.4 to include the three additional drift defects, including the retracted claim still asserted in `quantum_study.json` | W59 | 1 h |
| 21 | Add the training-configuration table (epochs, optimiser, LR) and reconcile the three quantum triples | W26, W27 | 3 h |
| 22 | Confirm no report text is redistributed in released artifacts; state DUA compliance for the hand-annotation | W66, W67 | 1 h |

**Tier 1 total: roughly 3–4 focused working days plus ~2 hours of compute.**

## Tier 2 — STRONGLY RECOMMENDED (needed for MedIA / TMI / npj DM)

| # | Item | Ref |
|---:|---|---|
| 23 | **External validation on CheXpert** (pre-registered) | E1 |
| 24 | Noise attribution at n ≥ 100 with a variance (not MAE) decomposition | E7 |
| 25 | Hardware evaluation at n ≥ 50 cases with an interval | E8 |
| 26 | Subgroup analysis (sex, age, race) | E9 |
| 27 | Per-case 3D segmentation re-scoring so Table 16 becomes a real comparison | E10 |
| 28 | Recruit a clinical co-author and a statistical co-author | W68 |
| 29 | Extract the protocol into a runnable tool; extend the drift test to the manuscript | S1, S2 |
| 30 | Add the forest plot, risk–coverage figure, reliability diagrams, and CONSORT-style flow diagram | Phase 6 |
| 31 | Add formal model definitions and equations | Phase 6 |
| 32 | Document sequence-ablation methodology (n, split, CI) | W52 |
| 33 | Fix the pooled-κ stratification error in §9.3 | W50 |
| 34 | Restructure: merge §11 into §12, move §9.5 out of positive controls, swap §7 and §8.1 | Phase 6 |

## Tier 3 — OPTIONAL (raises quality, not required)

- Tabulate published quantum-advantage margins from [1]'s 16 studies (S3).
- Conditional mutual information between evidence channels to explain the entanglement penalty (S9).
- Effective-dimension capacity matching alongside parameter matching (W20).
- Margin-vs-shots rank correlation (E12).
- Evaluate or remove the ensemble backend (W22).
- Generalise the capability-audit procedure and apply it retrospectively to [1]'s corpus (S5).

## Tier 4 — FUTURE WORK (separate papers)

- Reader study / MRMC analysis (E11) — gates clinical-journal submission.
- Prospective silent-mode deployment with abstention monitoring.
- Shot-allocation loop executed on hardware (the experiment §10.6 identifies).
- Multi-site external validation beyond two public corpora.
- The protocol tool as a standalone methods paper.

---

## Readiness scores

| | Score | Justification |
|---|---:|---|
| **Current (v3)** | **52 / 100** | Strong thesis, excellent candour, good writing. Undermined by an undefined target variable, a confounded headline ablation, an unevidenced pre-specification claim, a protocol-violating positive result, stale statistics in the threats section, and multiple traceability failures. |
| **After Tier 1** | **74 / 100** | Internally consistent, statistically defensible, fully traceable, with the confound resolved. Publishable at TMLR / QMI / IEEE TQE / Scientific Reports. Still single-centre and single-author. |
| **After Tier 1 + Tier 2** | **86 / 100** | Externally validated, subgroup-analysed, hardware-evaluated at n > 1, clinically co-authored. Competitive at MedIA and IEEE TMI; viable at npj DM. |
| **After adding E11 (reader study)** | **91 / 100** | Clinical-journal grade. |

**Expected publication tier after Tier 1 + Tier 2: strong acceptance at TMLR / Quantum Machine Intelligence / IEEE TQE; genuine competitiveness at Medical Image Analysis and IEEE TMI; a real but not favourite chance at npj Digital Medicine.**

---

## Closing note to the author

The uncomfortable finding of this review is structural. Every one of the three critical defects is an instance of the failure mode the paper itself describes: the constant channel was not noticed because attention was on the ansatz, not the input; the abstention result was not held to P1 because it was the claim you wanted to keep; the pre-specification claim was not evidenced because you knew it was true. None of this is misconduct. It is the same asymmetry of attention you document in §10.3, operating on you.

That is not a reason to soften the paper. It is a reason to say so in it. A paper that documents an asymmetric-care failure mode, and then reports the three instances of it that a review found in the paper's own execution, is a substantially better paper than one that does not — and it is the only version of this manuscript that is fully consistent with its own argument.
