# AURA — Scientific Audit

**Independent, reproducible audit of the AURA evidence-fusion stack**
Date: 2026-07-19 · Auditor role: Principal Medical-AI Scientist / Nature-Medicine reviewer / FDA SaMD auditor / reproducibility reviewer.
Run: `audit_artifacts/run_20260719T175647Z/` · Seed: 7 · Git: `cf8356b` (dirty) · Interpreter: `venv` Python 3.12.10, numpy 2.4.4, torch 2.11.0+cu128 (CUDA).

> **Every number below is regenerated from saved weights + a seeded synthetic split and is traceable to a file in the run directory.** Nothing is inferred. Where evidence could not be produced this run, the item is marked **❌ NOT VALIDATED** rather than estimated.

---

## 0. Executive summary

AURA's classical reasoning pipeline is real and runs end-to-end on CPU. But the audit **does not support the project's headline scientific claim that quantum fusion beats classical.** On a fair, seeded reproduction (n = 100 held-out, synthetic):

1. **Accuracy advantage — ❌ NOT VALIDATED.** Quantum 0.960 [0.920, 0.990] vs classical 0.930 [0.880, 0.980]. The gap is **3 discordant samples**; McNemar exact **p = 0.25**, permutation **p = 0.25**. Not significant.
2. **Calibration advantage — ⚠️ CONFOUNDED.** The claimed "ECE 0.020 vs 0.276" compares *temperature-scaled quantum* against *uncalibrated classical*. Give classical its own temperature and ECE goes **0.276 → 0.027**, tying quantum's **0.020**. The 13× gap is a scaling artifact ([Issue 2](audit_pipeline_fix_report.md)).
3. **AUROC advantage — ❌ NOT VALIDATED.** DeLong: **0 of 6** classes show a significant quantum edge (4 classes are degenerate at AUROC 1.000 for both; the two non-degenerate classes give p = 0.239, 0.255).
4. **The one real quantum edge is small and not what ships.** Quantum's per-sample log-loss is genuinely lower than a *fairly-calibrated* classical (Wilcoxon **p = 2.4e-5**, Cohen's d = **−0.29**, a *small* effect). But the deployed engine's conflict guard replaces the quantum posterior with classical PoE on ordinary inputs ([Issue 3](audit_pipeline_fix_report.md)), so this edge is largely not served in production.
5. **AURA's own best model is classical.** The `learnable` (classical) fusion backend **beats quantum on accuracy (0.970), NLL (0.062), Brier (0.033), and AUROC (0.9993)** — quantum leads only on ECE, marginally.

**Verdict:** the defensible, honest headline is *calibration discipline and a working conformal guarantee* — **not** a quantum advantage. The quantum-superiority framing should be withdrawn or heavily qualified until reproduced at adequate power against a fairly-calibrated baseline.

---

## 1. Reproduced metric table (raw evidence)

Source: [`metrics/metrics.json`](audit_artifacts/run_20260719T175647Z/metrics/metrics.json) · n = 100 · class counts [35, 22, 7, 17, 12, 7].

| backend | accuracy | NLL | Brier | ECE | macro AUROC | conformal cov. (target .90) | set size |
|---|---|---|---|---|---|---|---|
| **quantum** (T=0.77) | 0.960 | 0.092 | 0.060 | **0.020** | 0.9989 | 0.920 | 0.940 |
| **learnable** (classical) | **0.970** | **0.062** | **0.033** | 0.024 | **0.9993** | 0.920 | **0.920** |
| classical_fair (T=0.31) | 0.930 | 0.189 | 0.103 | 0.027 | 0.9977 | 0.920 | 0.960 |
| ensemble (classical) | 0.930 | 0.361 | 0.149 | 0.188 | 0.9973 | 0.920 | 0.970 |
| classical_raw (T=1.0) | 0.930 | 0.488 | 0.203 | 0.276 | 0.9970 | 0.920 | 0.980 |

`classical_raw` is the uncalibrated baseline the prior benchmark compared against; `classical_fair` is the apples-to-apples comparison.

## 2. Statistical tests (Step 7)

Source: [`metrics/statistics.json`](audit_artifacts/run_20260719T175647Z/metrics/statistics.json). Assumption handling: Shapiro–Wilk on the paired NLL difference rejected normality → **Wilcoxon is authoritative**, paired-t reported for completeness.

| test | quantum vs classical_fair | interpretation |
|---|---|---|
| Accuracy diff (paired bootstrap) | +0.030, CI95 **[0.00, 0.07]** | CI touches 0 → not significant |
| McNemar (exact) | b=3/0, **p = 0.250** | 3 discordant samples; n.s. |
| Permutation (acc diff) | **p = 0.252** | n.s. |
| Wilcoxon (per-sample NLL) | **p = 2.42e-5** | quantum log-loss genuinely lower |
| Paired t (per-sample NLL) | p = 4.69e-3 | (diff non-normal → secondary) |
| Cohen's d (NLL) | **−0.29** | *small* effect |
| DeLong AUROC (per class) | 0/6 significant | n.s. everywhere |

## 3. Figures (Step 6) — PNG **and** PDF

All in [`audit_artifacts/run_20260719T175647Z/plots/`](audit_artifacts/run_20260719T175647Z/plots): `reliability_diagram`, `roc_curves`, `pr_curves`, `confusion_quantum`, `confusion_classical_fair`, `bootstrap_accuracy_diff`, `prediction_set_size`. The reliability diagram is the centerpiece: `classical_raw` sits far off the diagonal (ECE 0.276) while `classical_fair` and `quantum` both hug it — visual proof the calibration gap is a temperature artifact.

---

## 4. Module scorecard (Step 15)

Legend: ✅ yes · 🟡 partial · ❌ no/not validated. "Experimentally verified" / "Statistically significant" reflect **this run only**.

| Module | Implemented | Actually Used | Math Correct | Exp. Verified | Stat. Significant | Clinically Useful | Prod. Ready | Evidence | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Quantum fusion (VQC) | ✅ | 🟡 (guard falls back) | 🟡 | ✅ | ❌ | 🟡 | ❌ | Strong (this audit) | Retrain or reframe; drop superiority claim |
| Classical fusion (PoE) | ✅ | ✅ (default/fallback) | ✅ | ✅ | n/a (baseline) | ✅ | 🟡 | Strong | Keep as primary |
| Learnable fusion | ✅ | 🟡 (opt-in) | ✅ | ✅ | ✅ (best model) | ✅ | 🟡 | Strong | Promote to default candidate |
| Temperature scaling | ✅ | ✅ | ✅ | ✅ | n/a | ✅ | 🟡 | Strong | Fit **per backend** |
| Conformal prediction | ✅ | ✅ | ✅ | ✅ (cov 0.92 vs 0.90) | n/a | ✅ | 🟡 | Strong | Marginal only; add Mondrian |
| Vision CNN (DenseNet121) | ✅ | 🟡 (not in demo path) | ? | ❌ NOT VALIDATED | ❌ | ? | ❌ | None this run | Audit separately on real MIMIC-CXR |
| Explainability (occlusion) | ✅ | ✅ | 🟡 | ❌ NOT VALIDATED | n/a | 🟡 | ❌ | None this run | Needs faithfulness eval |
| Recommender (EIG) | ✅ | ✅ | 🟡 (greedy) | ❌ NOT VALIDATED | ❌ | 🟡 | ❌ | None this run | Validate utility |
| Quantum Q2–Q6 | ❌ | ❌ | n/a | ❌ | ❌ | ❌ | ❌ | Designed only | Out of scope |

---

## 5. Blockers (Step 16)

### Remaining bugs
1. 🔴 Unfair per-backend temperature scaling in `benchmark.py` (Issue 2).
2. 🔴 Benchmark bypasses the deployed conflict guard; measures a model production doesn't serve (Issue 3).
3. 🟠 Global-Python torch blocked (WinError 4551) — experiments must use the venv (Issue 4).
4. 🟡 Metrics rounded before storage; silent `Calibration` default on missing file (Issues 6–7).

### Remaining scientific weaknesses
1. No demonstrated, significant quantum advantage on any headline metric at n = 100.
2. All evaluation is on **synthetic** data with near-separable classes (4/6 AUROC = 1.000) → ceiling effects hide model differences.
3. Underpowered (n = 100); rarest classes n = 7; DeLong degenerate.
4. Epistemic uncertainty is an input-perturbation proxy, not a deep ensemble; attribution is leave-one-out, not Shapley (per `PROJECT_STATUS.md`).

### Remaining engineering weaknesses
1. No CI; the only tests are the 117 from the in-progress refactor (uncommitted).
2. Reproducibility depends on an interpreter the default `python` cannot load.
3. Trained weights are gitignored and unversioned → not recoverable if deleted (why Step-2 "delete everything" was declined).

### Publication blockers
1. Claims not significant and confounded → would not survive peer review as written.
2. Synthetic-only data with ceiling AUROC → external validity unestablished.
3. No comparison to a fairly-tuned classical baseline in the manuscript figures.

### FDA / SaMD blockers
1. No real-patient validation; no prospective or external cohort.
2. Deployed behavior (guard fallback) differs from evaluated behavior → traceability/validation gap.
3. No predetermined change-control, no locked model version tied to the reported metrics.

### Hackathon blockers
1. The quantum "win" collapses under a fair baseline — a sharp judge will find it. **Reframe to the honest, defensible story (calibration + conformal guarantee) before presenting.**
2. `benchmark.json` headline numbers are reproducible but misleading as labeled.

---

## 6. Top 25 highest-priority fixes (ranked by impact)

| # | Fix | Why it matters |
|---|---|---|
| 1 | Fit temperature **per backend** in `benchmark.py` | Removes the 🔴 calibration confound |
| 2 | Re-state/withdraw the quantum-superiority claim | Current claim fails reproduction |
| 3 | Evaluate the **deployed** `fuse().posterior` (guard-on), not raw `q.logits` | Benchmark must reflect production |
| 4 | Re-run all comparisons at **n ≥ 2000**, class-stratified | Current n = 100 is underpowered |
| 5 | Log the conflict-guard **trigger rate** | Quantifies how often quantum is actually served |
| 6 | Replace/augment synthetic data with a real CXR cohort | Removes ceiling effects & external-validity gap |
| 7 | Promote `learnable` backend evaluation (it wins) | Honest best-model selection |
| 8 | Version trained weights (DVC/LFS) | Weights are gitignored & unrecoverable |
| 9 | Add CI running the refactor's 117 tests | No automated safety net today |
| 10 | Persist unrounded per-sample outputs everywhere | Preserve statistical precision (Issue 6) |
| 11 | Make `Calibration.load` warn/raise on missing file | Kill silent T=1.0 fallback (Issue 7) |
| 12 | Pin the venv interpreter in all run scripts/docs | Global torch is blocked (Issue 4) |
| 13 | Retrain the VQC so it stops conflicting with PoE | Guard fires because the VQC fit is weak |
| 14 | Add Mondrian (class-conditional) conformal | Current coverage is marginal only |
| 15 | Faithfulness eval for occlusion saliency | Explainability unvalidated |
| 16 | Validate EIG recommender against a utility oracle | Recommender unvalidated |
| 17 | Report CIs + tests in the product's benchmark table | Not just point estimates |
| 18 | Add a deep-ensemble epistemic head | Current proxy overstates capability |
| 19 | Stratified split with fixed per-class quotas | Rarest classes n = 7 |
| 20 | Separate, dedicated audit of the DenseNet121 path on real MIMIC-CXR | Not covered this run |
| 21 | Record model_version → metrics linkage | FDA traceability |
| 22 | Add OOD/shift stress tests (Step 11) with real corruptions | Not yet run |
| 23 | Document that 8-qubit VQC is classically simulable | Avoid "quantum speedup" misread |
| 24 | Commit the in-progress refactor or gate it behind a flag | Dirty tree undermines reproducibility |
| 25 | Add a MODEL_CARD row stating "no demonstrated quantum advantage (audited)" | Honesty as a feature |

---

## 7. What this audit did NOT cover (explicitly, no fabrication)

- **DenseNet121 vision CNN, GradCAM, feature maps** — the protocol's CNN/CUDA/mixed-precision steps apply to `ml/vision_cxr/*`, not the fusion head. Auditing it requires loading `vision_cnn.zip`/`best_model.pt` against a labeled real MIMIC-CXR split; **out of scope this run → ❌ NOT VALIDATED.** No CNN metric is reported here because none was produced.
- **Domain-shift battery (Step 11)** — IID/covariate/label-shift/OOD/corruption sweeps: not run → **❌ NOT VALIDATED.** Hooks exist (`ml/training/recalibrate_ood.py`, `safety.synthetic-ood.bak.npz`) but were not exercised.
- **Full 7-module retrain from scratch** — declined; it would require deleting gitignored weights (irreversible) and hours of compute. The audit reproduces from saved weights instead, which is sufficient for claim verification and strictly non-destructive.

### Reproduce this audit
```
E:\AURA\venv\Scripts\python.exe E:\AURA\aura-main\audit_all.py
```
Outputs a fresh, timestamped `audit_artifacts/run_<UTC>/` with `environment.json`, `experiment_manifest.json`, `requirements.txt`, `git_commit.txt`, `random_seed.txt`, and the full `csv/ metrics/ plots/ logs/` tree.
