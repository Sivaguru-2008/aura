# Pre-registration addendum — revision analyses for manuscript v5.1

**Deposited:** 2026-10-02, committed to this repository *before* any of the analyses below
were executed. The commit timestamp is the registration record.
**Supersedes:** the undeposited draft `E:\AURA\PREREGISTRATION_ADDENDUM.md` (never committed;
see §6 for what changed and why).
**Executes in:** `aura/ml/evaluation/revision_v51.py` → `aura/artifacts/revision_v51.json`.

## 0. What is and is not blind — read this first

A pre-registration is only as strong as what the authors had not yet seen. Stated per analysis:

| Analysis | Already known when this was written | Status |
|---|---|---|
| A. TOST, quantum vs PoE (matched optimiser) | **Point estimate Δ = +0.0033 NLL (PoE better) and Wilcoxon p = 0.66 are published** in `fusion_matched_optimiser.json`. The standard error was not computed. | **Not blind.** Confirmatory in form only. |
| B. Study-level bootstrap of the 7-channel ablation | **Per-seed ΔNLL for all 10 seeds is published** (`entanglement_multiseed_7q.json`, mean +0.025). Per-study NLLs were never saved. | **Not blind.** A re-analysis of known data. |
| C. K = 5 patient-disjoint re-splits | Nothing — never run. | Blind. |
| D. Data re-uploading ablation | Nothing — circuit never built. | Blind. |

Analyses A and B are therefore reported as **re-analyses with the decision rule fixed in advance**.
They are not independent confirmation. C and D are prospective.

## 1. Cohort facts that shape the resampling unit

`build_real_evidence_dataset` keeps **one study per patient** (the most recent), so the
2,231-study evidence cache (`quantum_study_evidence.npz`: 1,338 / 446 / 447) has one row per
patient. A patient-clustered bootstrap is therefore *identical* to a study-level bootstrap here,
and any random permutation of rows is a patient-disjoint split. Patient identifiers are not
stored in the cache and the MIMIC-CXR source folder is no longer on disk, so this is asserted
from the builder's code, not re-derived from IDs. Weak corroboration: the cache has zero
duplicate evidence rows.

## 2. Analysis A — equivalence test (TOST), quantum VQC vs product-of-experts

* **Data:** `fusion_matched_optimiser_predictions.npz`, per-study NLL, n = 447, paired.
* **Statistic:** d_i = NLL_quantum,i − NLL_PoE,i. Paired t-based TOST.
* **Margin:** ±0.010 nats, taken unchanged from the v5.1 draft text where it appeared as an
  example. **It has no clinical anchoring**; we do not claim it does. We report the 90% CI so a
  reader can apply any margin they prefer.
* **Decision:** equivalence iff max(p_lower, p_upper) < 0.05 (⇔ 90% CI inside ±0.010).
  Otherwise the manuscript says only "no difference detected" — never "parity".
* **Sensitivity (pre-declared):** the same TOST on a 10,000-resample percentile bootstrap of
  mean d (non-parametric, because per-study NLL differences are heavy-tailed).

## 3. Analysis B — between-study uncertainty for the 7-channel entanglement ablation

* **Re-run, unchanged:** `quantum_study_7q.py`'s protocol — 7 qubits, 3 layers, RY(πx)
  encoding, `train_quantum` (Adam lr 0.05, batch 50, 30 epochs), seeds
  {7, 11, 13, 17, 19, 23, 29, 31, 37, 41}, per-arm temperature on the calibration split —
  now also **saving per-study test probabilities**.
* **Reproduction gate:** every per-seed ΔNLL must match `entanglement_multiseed_7q.json` to
  4 dp. If any seed fails to reproduce, analysis B is reported as failed and not substituted.
* **Primary:** per-study ΔNLL averaged over the 10 seeds, then a 10,000-resample bootstrap
  over the 447 studies (= patients, §1). Report mean, bootstrap SE, 95% percentile CI, and the
  two-sided bootstrap p (2 × the smaller tail mass at 0).
* **Secondary:** two-stage hierarchical bootstrap (resample seeds, then studies), which carries
  both initialisation and sampling variance. 10,000 resamples.
* **Decision:** the claim "entanglement worsens NLL" is retained in the abstract iff the
  primary 95% CI excludes 0. If it does not, the claim is downgraded to "directionally worse
  in all seeds; not distinguishable from zero over patients".

## 4. Analysis C — repeated patient-disjoint re-splits

* **K = 5** re-splits of the 2,231 pooled rows, permutation seeds {101, 102, 103, 104, 105},
  **60/20/20** (train/cal/test) — the ratio of the primary split. (The draft's 70/10/20 would
  have changed two things at once.)
* Each split: both arms retrained from scratch at training seed 7 under the analysis-B
  protocol; per-arm temperature on that split's calibration set.
* **Report:** per-split ΔNLL, mean, between-split SD (σ_data), alongside the seed SD from
  analysis B (σ_init). With K = 5 we report the sign count and a one-sample t-test but treat
  it as descriptive; K = 5 cannot carry a significance claim on its own.

## 5. Analysis D — data re-uploading ablation

* **Qubits:** **7** (the 7 informative channels), *not* 8. The draft's "8 qubits with a
  constant reference channel" re-introduces the dead-register coupling that the 7-channel
  analysis was built to remove.
* **Circuit, per layer l = 1..L, per qubit i:**
  `RY(w_{l,i} · π x_i)` (trainable encoding scale, initialised to 1) → `RY(θ_{l,i,0})` →
  `RZ(θ_{l,i,1})` → entangler. Readout ⟨Z_i⟩ → linear head to 6 classes (as in the shipped VQC).
* **L = 3.** Entangled arm: CNOT ring. Product arm: no two-qubit gates.
* **Parameter count (both arms, identical by construction):** circuit 3 × 7 × 3 = 63
  (the draft's "72" omitted the encoding weights and assumed 8 qubits), head 6 × 7 + 6 = 48,
  **total 111**.
* **Training:** identical to analysis B (Adam lr 0.05, batch 50, 30 epochs, no early stopping),
  same 10 seeds, same cached split. The only change from analysis B is the ansatz; matching the
  optimiser to B is deliberate so that a B-vs-D difference is attributable to re-uploading.
* **Report:** per-seed ΔNLL (entangled − product), mean, 95% t-CI, Wilcoxon p, sign count;
  plus the analysis-B study-level bootstrap applied to D's per-study predictions; plus each
  arm's NLL against the non-re-uploading arms of B (does re-uploading help either arm at all?).
* **Decision:** if the D bootstrap CI excludes 0 in the same direction as B, the entanglement
  penalty is reported as surviving a higher-expressivity encoding. If it includes 0 or reverses,
  the manuscript states that the B result is specific to the RY(πx) encoding family.

## 6. Changes from the undeposited draft, and why

| Draft | This addendum | Reason |
|---|---|---|
| "Pre-registered prior to data ingestion" | §0 table | A and B were not blind; saying so is the point |
| 0.010 margin "≈ <0.25% posterior shift" | no clinical claim | the 0.25% figure was not derived from anything |
| Re-uploading on 8 qubits, 72 params | 7 qubits, 111 params | constant-channel confound; miscounted parameters |
| lr 0.01 + early stopping, seeds 1–10 | analysis-B protocol and seeds | so B vs D isolates the ansatz |
| 70/10/20 re-splits | 60/20/20 | match the primary split |
| Provenance ledger (Protocol 4) | out of scope here | an engineering change, not an analysis; tracked separately |
