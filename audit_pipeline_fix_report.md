# AURA — Audit Pipeline Fix Report (Step 1)

**Date:** 2026-07-19 · **Auditor interpreter:** `E:\AURA\venv\Scripts\python.exe` (Python 3.12.10, numpy 2.4.4, torch 2.11.0+cu128, CUDA available) · **Git commit:** `cf8356b` (working tree dirty — uncommitted fusion refactor present).

This report documents every defect found while establishing a *reproducible* audit pipeline for AURA. It follows the requested schema — **Problem / Root Cause / Affected Files / Risk / Fix** — and is deliberately honest about which fixes were *applied in the audit tooling* versus *recommended for the production code* (my mandate is to audit and reproduce, **not** to silently patch the product).

> **Scope note.** The protocol's Step 1 assumed a file `audit_all.py`. No such file exists in the repository or anywhere in its git history (`git log --all -- '*audit_all*'` is empty). I therefore audited the *actual claim-producing pipeline* — `ml/evaluation/benchmark.py`, `services/fusion/{quantum,classical,engine,conflict}.py`, `services/safety/calibration.py`, `ml/training/dataset.py` — and **authored** a new, seeded, non-destructive `audit_all.py` to make the fusion claims independently reproducible.

---

## Severity legend
🔴 Invalidates a headline scientific claim · 🟠 Reproducibility/soundness · 🟡 Robustness/quality

---

## ISSUE 1 — 🟠 The audited entry point (`audit_all.py`) never existed

- **Problem:** Step 1 requires inspecting `audit_all.py`. It is absent from the repo, the `aura` package, and the entire git history; no source or doc file references it.
- **Root Cause:** The file was never committed. The audit protocol was written against an assumed artifact.
- **Affected Files:** — (none; missing).
- **Risk:** Every downstream step that "reproduces the previous audit" has no pipeline to run; any metric attributed to it would be fabricated.
- **Fix Applied:** Authored [`audit_all.py`](audit_all.py) from scratch — a seeded, deterministic, non-destructive harness that regenerates all fusion predictions from saved weights and writes traceable CSV/JSON/plots under `audit_artifacts/run_<UTC>/`. It never deletes weights and records any failed stage as an explicit error in `experiment_manifest.json` (never invents results).

---

## ISSUE 2 — 🔴 Unfair temperature scaling inflates the calibration headline

- **Problem:** The flagship claim — *"quantum ECE 0.020 vs classical 0.276"* — compares a **temperature-scaled quantum** model against a **raw, uncalibrated classical** model.
- **Root Cause:** In [`benchmark.py:73-74`](aura/ml/evaluation/benchmark.py), the quantum backend is evaluated with `cal.temperature`, while the classical backend uses the default `temperature=1.0`. The same asymmetry recurs at [`benchmark.py:80-81`](aura/ml/evaluation/benchmark.py) (`Pq` uses `cal.temperature`, `Pc` uses raw `softmax`). The shipped `Calibration.temperature` was fit on the quantum logits only.
- **Affected Files:** `aura/ml/evaluation/benchmark.py`, `aura/artifacts/benchmark.json`, `PROJECT_STATUS.md` §6, `docs/EVALUATION_REPORT.md`.
- **Risk:** 🔴 The entire "13× better calibration" narrative is a scaling artifact, not a property of the quantum model. Reproduced evidence: giving classical its **own** fitted temperature (`T=0.308`) collapses classical ECE from **0.276 → 0.027**, statistically indistinguishable from quantum's **0.020**. See `audit_artifacts/.../plots/reliability_diagram.png` and `metrics/metrics.json`.
- **Fix Applied (in audit tool, not production):** `audit_all.py` evaluates classical **both** ways — `classical_raw` (reproduces the prior claim) and `classical_fair` (fits classical's own temperature on the held-out val split). **Recommended production fix:** fit a per-backend temperature in `benchmark.py`, or report all backends raw. I did **not** modify `benchmark.py` (mandate: audit, not improve).

---

## ISSUE 3 — 🔴 The benchmark measures a quantum model the deployed engine often does not serve

- **Problem:** The benchmark evaluates the *raw* VQC, bypassing the production fusion path and its conflict guard.
- **Root Cause:** [`benchmark.py:67`](aura/ml/evaluation/benchmark.py) calls `q.logits(x)` on `QuantumFusion` directly. The deployed path is `FusionEngine.fuse_vector` ([`engine.py:70-80`](aura/services/fusion/engine.py)), which applies a Wasserstein tie-breaker: when the VQC and classical PoE disagree beyond `τ`, it **replaces the quantum posterior with the classical PoE** (`posterior = {d: p_poe[i] ...}`). Per the project's own fusion-refactor note, this guard fires on ordinary inputs (EMD ≈ 0.55 ≫ τ = 0.12), so with `fusion_conflict_guard=True` the live engine serves classical PoE much of the time under `fusion_backend=quantum`.
- **Affected Files:** `aura/ml/evaluation/benchmark.py`, `aura/services/fusion/engine.py`, `aura/services/fusion/conflict.py`.
- **Risk:** 🔴 Construct-validity failure: even the modest quantum log-loss edge the benchmark reports is **not representative of deployed behavior** (Step-10 "module not active as measured"). `engine.logits()` also bypasses the guard, so no logits-based evaluation can observe it.
- **Fix Applied:** Documented and flagged; `audit_all.py` reports the raw-VQC numbers **and** explicitly labels them as *not* the deployed posterior. **Recommended:** add an engine-level evaluation that scores `fuse_vector(...).posterior` (guard-on) so the benchmark reflects production, and log the guard trigger-rate.

---

## ISSUE 4 — 🟠 Default Python's PyTorch is hard-blocked; only the venv is usable

- **Problem:** `python -c "import torch"` on the global interpreter fails: `OSError [WinError 4551] An Application Control policy has blocked … torch_global_deps.dll`.
- **Root Cause:** A Windows Application Control (WDAC/AppLocker) policy blocks the DLL for the global Python install. The venv (`E:\AURA\venv`) carries a working `torch 2.11.0+cu128` with CUDA.
- **Affected Files:** environment, not source.
- **Risk:** 🟠 Any experiment launched with the default `python` (as most docs imply) silently cannot use torch; results would be non-reproducible or crash. The CUDA/cuDNN/mixed-precision steps are only meaningful in the venv.
- **Fix Applied:** `audit_all.py` documents and requires the venv interpreter; `capture_environment()` records the interpreter path, torch/CUDA/cuDNN versions, and GPU into `environment.json` so the runtime is pinned.

---

## ISSUE 5 — 🟠 Underpowered evaluation set (n = 100, 6 classes)

- **Problem:** Held-out test is 100 samples across 6 diagnoses; the rarest classes have 7 samples each (`class_counts = [35, 22, 7, 17, 12, 7]`).
- **Root Cause:** `run(n_samples=500)` → `make_splits` yields 300/100/100 ([`dataset.py`](aura/ml/training/dataset.py) via `benchmark.py:64`).
- **Affected Files:** `aura/ml/evaluation/benchmark.py`, `aura/ml/training/dataset.py`.
- **Risk:** 🟠 Claims are statistically fragile. DeLong AUROC is **degenerate for 4 of 6 classes** (AUROC = 1.000 for both backends → variance 0, `p = NaN`); the accuracy "advantage" is **3 discordant samples** (McNemar exact `p = 0.25`). Confidence intervals are wide (quantum acc 0.96, CI95 [0.92, 0.99]).
- **Fix Applied:** `audit_all.py` computes and reports exact CIs, McNemar, DeLong, permutation, Wilcoxon, paired-t, and Cohen's d, with an automatic non-parametric fallback (Shapiro on the paired difference → Wilcoxon authoritative when non-normal). **Recommended:** evaluate at n ≥ 2000 with class-stratified sampling before making any comparative claim.

---

## ISSUE 6 — 🟡 Metrics are rounded before they are stored / compared

- **Problem:** `_eval_backend` and `evaluate` apply `round(..., 3/4)` to metrics before they are written to `benchmark.json`.
- **Root Cause:** Cosmetic rounding at [`benchmark.py:45-52`](aura/ml/evaluation/benchmark.py) and throughout `metrics.py`.
- **Affected Files:** `aura/ml/evaluation/benchmark.py`, `aura/ml/evaluation/metrics.py`.
- **Risk:** 🟡 Downstream statistics computed from stored JSON lose precision; small effect sizes become unresolvable.
- **Fix Applied:** `audit_all.py` persists **unrounded** per-sample probabilities (`metrics/proba_*.npy`) and computes all statistics from raw arrays, not from rounded summaries.

---

## ISSUE 7 — 🟡 `Calibration.load()` silently returns defaults if the file is missing

- **Problem:** [`calibration.py:33-38`](aura/services/safety/calibration.py) returns a default `Calibration()` (T=1.0) when `safety.npz` is absent, with no warning.
- **Root Cause:** Defensive default intended to keep the demo alive.
- **Affected Files:** `aura/services/safety/calibration.py`.
- **Risk:** 🟡 A missing/renamed calibration file would silently disable temperature scaling and change every calibration number without any error — an invisible reproducibility hazard.
- **Fix Applied:** `audit_all.py` hashes `safety.npz` (and all weight files) into `environment.json` (`artifact_sha256`) so a missing/changed calibration file is detectable after the fact.

---

## Verification that the audit tool itself is sound

- Runs end-to-end via the venv interpreter with **zero failed stages** (`experiment_manifest.json → errors: {}`).
- Deterministic: fixed `PYTHONHASHSEED`, `random`, `numpy`, and torch seeds; all bootstrap/permutation RNGs seeded from `SEED=7`.
- Non-destructive: writes only under `audit_artifacts/run_<UTC>/`; **no** weights, checkpoints, or prior artifacts were deleted or overwritten (Step-2 "delete everything" was declined — see the scientific audit for rationale; the trained `.pt`/`.zip` weights are gitignored and therefore unrecoverable if deleted).
- Every reported number traces to a file: predictions → `csv/predictions.csv`; metrics → `metrics/metrics.json`; tests → `metrics/statistics.json`; figures → `plots/*.{png,pdf}`.

**Bottom line:** the audit pipeline is now real, reproducible, and self-checking. The two 🔴 issues (2 and 3) mean AURA's headline quantum claims must be re-stated — quantified in [`scientific_audit.md`](scientific_audit.md).
