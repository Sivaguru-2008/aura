# AURA — Codebase Review

> ## ✅ Remediation complete — 2026‑07‑30, 21:40 IST
>
> Every finding in this document has been fixed except one, which is deliberately
> deferred with reasons (§6.3, `console.js`). See **§10 — Remediation log** at the end
> for what changed, and **§11** for the two items that need your decision.
>
> | | Before | After |
> |---|---|---|
> | Test suite | 669 passed / **33 failed** | **738 passed / 0 failed** (exit 0) |
> | Unauthenticated PHI reads | every `GET`, incl. FHIR/HL7 export | **gated** — 22 new authz tests |
> | Tracked repo weight | 1339.5 MB | **216.0 MB** (−1.12 GB) |
> | Tests writing served state | cases, outcomes, **conformal q̂** | **isolated**, guard fails the run |
> | Fabricated competitor numbers | 4 tables + 1 CSV | **cited or removed**, pinned by tests |
>
> Findings are left in their original tense below so the reasoning stays legible;
> each priority-table row now carries its resolution.

**Reviewer:** Claude Opus 5 (Claude Code)
**Snapshot:** 2026‑07‑30, 17:27 IST · `HEAD = 5eb1b70` · working tree dirty (48 modified, 36 untracked)
**Scope:** the whole repository at `E:\AURA\aura-main` — 380 first‑party Python files (~58 kLOC), 47 test modules
(~10 kLOC), the FastAPI gateway, the modality‑router backend, the web console, docs, CI and the Docker stack.
**Method:** every claim below was executed, not read. Test counts come from the same command CI runs
(`pytest -q -m "not slow"` from `aura/`). Where I assert a behaviour is broken, a reproduction is included.

> ### ⚠️ Read this first: the tree moved under me
> A concurrent session was editing this working tree throughout the review. I watched the suite go
> **32 collection errors → 40 failed → 33 → 31 → 13** in ninety minutes as fixes landed. Several defects I had
> fully reproduced (§8) were repaired mid‑review. Everything in §1–§7 was re‑verified at the 17:27 snapshot and
> is current; §8 is the record of what got fixed while I watched, kept because the *pattern* it shows is itself a
> finding. If you are reading this hours later, re‑run the commands — the numbers, not the prose, are the contract.

---

## 1. Verdict

| Dimension | Grade (at review) | After remediation |
|---|---|---|
| Engineering discipline | **A−** | **A** — guards now cover attribution, artifacts, imports, auth |
| Scientific honesty | **A−** | **A** — every published number is measured or cited |
| Test health | **B** | **A−** — 738/738 green, isolated, 0 dead async tests |
| Security posture | **C** | **B+** — reads gated, config surfaced, preflight warns |
| Repo hygiene | **D** | **B** — 216 MB tracked; LFS/release-asset migration still open |
| Structural coherence | **B−** | **A−** — single import root, enforced; collision removed |

**Bottom line.** This is a genuinely impressive system with an unusual virtue: it argues against its own marketing.
The docs state that classical fusion beats the quantum backend, that the entanglement ablation was a negative
result, that the quantum kernel loses to an RBF, that a label axis is untrainable and the head abstains. That
posture is worth more than the quantum results themselves, and it is the thing to protect.

The problems are not in the science. They are in the **operational shell around it**: an unauthenticated read
surface, a gigabyte of dead binaries in git history, a missing test dependency that keeps CI red, and one
statistical claim the sample size does not support. All are days of work, not months.

---

## 2. Test health — 689 passed, 13 failed, one root cause

```bash
cd aura && python -m pytest -q -m "not slow" --durations=10
```

```
13 failed, 689 passed, 1 skipped, 4 deselected in 454s
```

Failures by module — and they collapse to a single cause:

| Module | Failed | Cause |
|---|---|---|
| `tests/test_agent_board.py` | 9 | `async def` test, no async plugin installed |
| `tests/test_copilot.py` | 4 | same |

```
Failed: async def functions are not natively supported.
You need to install a suitable plugin for your async framework, for example:
  - pytest-asyncio
```

### Finding 2.1 — `pytest-asyncio` is not a declared dependency ⇒ CI is red and 13 tests never execute · **High**

`aura/requirements.txt` pins `pytest>=8.0` and nothing else test‑related. `requirements-docker.txt` — the file
CI installs — has no async plugin either. So:

- All 13 `@pytest.mark.asyncio` tests **fail** in CI, on every push, on every branch (`branches: ["**"]`).
- They also provide **zero coverage**. `test_agent_board.py` is the multi‑agent consensus engine — arbitration,
  timeout fallback, exception fallback, the "requires review" path. `test_copilot.py` covers the Ollama client's
  connection‑error, timeout and HTTP‑error handling. These are exactly the failure paths you want tested, and
  none of them has ever run.
- `pyproject.toml`'s `[tool.pytest.ini_options] markers` list registers `slow` but not `asyncio`, so pytest also
  emits 13 `PytestUnknownMarkWarning`s — the signal that the plugin is absent was there the whole time.

**Fix.** Add `pytest-asyncio>=0.23` to `aura/requirements.txt` *and* `requirements-docker.txt`, then set
`asyncio_mode = "auto"` (or register the marker) in `[tool.pytest.ini_options]`. Expect some of the 13 to fail
for real once they actually run — that is the point.

### Finding 2.2 — the suite is order‑dependent · **Medium**

`tests/test_architecture_8.py` passed 100/100 standalone while failing 18 in a full‑suite run started minutes
earlier. Part of that was the concurrent fixes landing mid‑run, but I also saw `test_audit_repairs.py` and
`test_quantum_measurement.py` pass in isolation and fail in‑suite. Tests that share the real
`aura/artifacts/` directory and the real SQLite store will do this. `git status` showing 24 modified files under
`aura/artifacts/` — including `safety.npz`, `calibration.json` and `registry.json` — is the same symptom from the
other direction: **the suite mutates served calibration state.**

This matters more here than in an ordinary codebase. Memory of a prior audit records an operating‑point
degeneracy (pneumothorax never firing, effusion always firing) that turned out to be a *test‑clobbered n=16
calibration fit* overwriting a validated n=2099 one. The mechanism that caused that is still live.

**Fix.** Point artifact writes at `tmp_path` in every test that writes; add a session‑scoped fixture that fails
the run if `aura/artifacts/` is dirty at exit. Then run `pytest -p randomly` in CI to keep it honest.

---

## 3. Security — the read surface is wide open

### Finding 3.1 — authentication covers only mutating methods; every `GET` is public · **Critical**

`aura/gateway/app.py:94‑108`:

```python
@app.middleware("http")
async def audit_mw(request: Request, call_next):
    if request.method in ("POST", "PUT", "DELETE"):
        from .security import enforce
        try:
            enforce(request)
```

`enforce()` is the *only* call site of authentication, authorization and rate limiting, and it is gated on the
method. Consequences, with `auth_token` set to a strong value and rate limiting on:

| Endpoint | Method | Auth | Exposes |
|---|---|---|---|
| `/v1/cases` | GET | **none** | the entire worklist |
| `/v1/cases/{id}` | GET | **none** | full case bundle: findings, posterior, report text |
| `/v1/cases/{id}/export/fhir` | GET | **none** | FHIR DiagnosticReport + Observations |
| `/v1/cases/{id}/export/hl7` | GET | **none** | HL7 ORU^R01 message |
| `/v1/cases/{id}/chat` | GET | **none** | clinician chat transcript |
| `/v1/admin/safety` | GET | **none** | admin safety/policy state |
| `/v1/cases/{id}/neuroview` | GET | **none** | volumetric brain render payload |

The FHIR and HL7 exporters are the sharp end: they emit structured, standards‑conformant clinical records
designed to be ingested by other systems, and they are reachable by anyone who can route a packet to the port,
with no token and no rate limit. Case IDs are enumerable via the unauthenticated `/v1/cases`. Rate limiting is
also method‑gated, so the read surface cannot even be slowed down.

`/v1/admin/safety` being an unauthenticated `GET` is independently wrong regardless of PHI.

**Fix.** Move `enforce(request)` out of the method conditional and gate on path instead — allowlist
`/v1/health`, `/`, `/app`, `/history`, `/static/*` and require a principal for everything under `/v1/cases`,
`/v1/studies`, `/v1/admin`. Keep the audit write on mutations only if you want, but the *gate* must run on reads.
This is a five‑line change and it is the highest‑value fix in this document.

### Finding 3.2 — the documented Docker path deploys with auth off · **High**

`common/config.py` supports `auth_token`, `auth_header` and `rate_limit_rpm` (lines 189‑191, wired at 281‑283).
Neither `.env.example` nor `docker-compose.yml` mentions any of them. `.env.example` is meticulous elsewhere — it
has a dedicated "NOT set here" section warning about `AURA_FUSION_BACKEND` and `AURA_ALLOW_FALLBACK_VISION` — so
the omission reads as oversight, not intent. An operator following `docs/DOCKER_DEPLOYMENT.md` exactly gets a
server with authentication and rate limiting both inert.

**Fix.** Add a `# --- access control ---` block to `.env.example` with `AURA_AUTH_TOKEN=` and
`AURA_RATE_LIMIT_RPM=60`, pass both through in `docker-compose.yml`, and make `deploy/preflight.py --strict`
*warn* when `auth_token` is empty. A preflight that already refuses to boot on a missing artifact should have an
opinion about an unauthenticated clinical endpoint.

### Finding 3.3 — rate limiting is per‑process · **Low**

`RateLimiter` (`gateway/security.py:48`) is an in‑memory token bucket. The docstring is honest about it
("no Redis needed"), but `docs/BENCHMARKS.md` §3 pitches "multiple server replicas on a single GPU" — at which
point the effective limit becomes `rpm × replicas`. Note the interaction in the deployment doc.

---

## 4. Scientific claims — strong posture, two gaps

The honesty layer is real and I want to be specific about that, because it is unusual:

- `docs/BENCHMARKS.md` §1 leads with **"Classical Superiority"** and states plainly that the 8‑qubit VQC does
  not yield a quantum advantage on a classically‑simulable task.
- The headline table's sample size (`n_eval = 69`) is disclosed in `README.md:44`, `WHAT_IS_AURA.md:202` **and**
  `docs/BENCHMARKS.md` — three places, unprompted.
- `tests/test_doc_numbers.py` mechanically pins those tables to `artifacts/benchmark.json` and its own docstring
  says *"the fix is to re‑copy the numbers out of `benchmark.json`, never to loosen this test."* It passes.
- `docs/KNOWN_LIMITATIONS.md` volunteers that a PNG‑exported head CT will misroute to the brain engine, and that
  the brain model is invalid for postoperative, radiation‑necrosis and non‑glial cases.

That is a system that has been argued with. Now the two gaps.

### Finding 4.1 — "Classical Superiority" is not supported at n=69 · **High**

The claim rests on 0.6957 vs 0.6377 accuracy. At n=69 that is **48 correct vs 44 correct — a four‑case
difference.**

```
classical  48/69 = 0.6957   95% CI [0.581, 0.795]
quantum    44/69 = 0.6377   95% CI [0.520, 0.744]
learnable  43/69 = 0.6232   95% CI [0.506, 0.731]
```

The intervals overlap across almost their entire range. A paired McNemar test bounds it even more tightly — even
in the **most favourable possible pairing**, where all four discordant pairs break for the classical model:

```
McNemar (classical‑only right=4, quantum‑only right=0): p = 0.125
```

There is no assignment of the discordant pairs that reaches significance. The three backends are statistically
indistinguishable on this split, and per‑class support makes it starker: `malignancy` n=4 (sensitivity 0.000),
`copd` n=4, `heart_failure` n=7, `pneumonia` n=8. Five of six classes have single‑digit support.

This is the one place the honesty layer is inconsistent with itself. The QKL work reports bootstrap CIs and a
p‑value (ΔAUROC −0.016, p=0.002) and correctly calls that result significant. The fusion table — the more
prominent claim, the one in the README — has neither.

**Fix.** Keep the direction, drop the certainty. Add a CI column to the table and replace the "Classical
Superiority" heading with something like *"Classical PoE leads by 4/69 cases; the backends are not
statistically distinguishable at this sample size (McNemar p ≥ 0.125). The classical model is served as the
fair‑accuracy reference on that basis, not on a demonstrated gap."* That sentence is stronger than the current
one, because it survives scrutiny.

### Finding 4.2 — hardcoded competitor numbers presented as measurements · **High**

`ml/evaluation/run_pipeline.py:258‑280` generates `docs/benchmark_report.md`. AURA's own rows are f‑string
interpolations from measured artifacts. The competitor rows are literals:

```python
| **AURA Brain (ResU-Net)** | **{eval_summary["brain_model"]["metrics"]["dice_mean"]:.3f}** | ... |
| **nnUNet** | 0.871 | 0.920 | 0.852 | 0.841 | 185.0 s | 8.52 s | ... |
| **SwinUNETR** | 0.858 | 0.910 | 0.838 | 0.825 | 240.0 s | 12.10 s | ... |
| **MONAI Baseline** | 0.835 | 0.895 | 0.812 | 0.798 | 12.5 s | 0.65 s | ... |
```

nnU‑Net, SwinUNETR and MONAI were never run. The surrounding prose says *"This report compares the AURA
architecture against industry‑standard baseline models"*, the rows carry no citation and no "literature value"
qualifier, and the same literals are re‑emitted into `comparison_table.csv` (lines 310‑312) where the framing is
lost entirely. The `Classical Chest Baseline` row (0.7650 / 0.2850 / 0.2850) and AURA's own `GPU Latency` of
`0.42 s` are hardcoded on the same footing.

There is a second aggravating factor: the comparison is not like‑for‑like. AURA's brain Dice is a pooled 2D
per‑slice figure; the BraTS numbers those baselines are known for are per‑case 3D. Pooled‑2D flatters.

`docs/benchmark_report.md` is also **outside** the `test_doc_numbers.py` guard, which covers only `README.md`,
`WHAT_IS_AURA.md` and `docs/BENCHMARKS.md`.

**Fix.** Either measure them or label them. Labelling is cheap and honest: retitle the section *"AURA vs.
published literature values"*, cite each number to its paper, add a row note that AURA's Dice is pooled‑2D and
the baselines are per‑case‑3D, and stop writing unattributed literals into a CSV. Then extend
`test_doc_numbers.py` to cover `benchmark_report.md`. Given how well the rest of the repo handles this exact
problem, this table is conspicuous — and it is the first thing an informed reviewer will attack.

### Finding 4.3 — `KNOWN_LIMITATIONS.md` publishes superseded thresholds · **Medium**

The doc states abstention fires when epistemic uncertainty exceeds `0.45`, the OOD z‑score exceeds `3.0`, and
conformal set size exceeds `3`. Served values:

```
epistemic_threshold       = 0.15
ood_threshold             = 2.5
abstention_conformal_size = 4
active_policy             = 'community_conservative'
```

The code knows. `common/config.py:161` reads: *"the previous values (0.45, 3) were tuned for the overconfident
synthetic fusion model … and abstained on ~91% of real films once fusion was honestly calibrated."* The
recalibration happened; the limitations doc was not updated with it. So the one document a cautious reader
consults to find the safety envelope describes an envelope that was deliberately replaced.

**Fix.** Regenerate those three numbers from `get_settings()` and extend the drift guard to cover safety‑policy
thresholds, not just benchmark metrics. Same pattern as `test_doc_numbers.py`, same justification.

---

## 5. Repository hygiene — 1.34 GB tracked, ~720 MB provably dead

```
1339.5 MB across 851 tracked files   (no Git LFS — .gitattributes marks *.pt binary, nothing more)
```

### Finding 5.1 — dead and training‑only binaries are tracked at full size · **High**

| Path | Tracked size | Referenced anywhere? |
|---|---|---|
| `aura/artifacts/brain/v1_broken_quality_head/` | **400.1 MB** | **No** — and the name says it is broken |
| `aura/artifacts/_smoke/` | **161.0 MB** | **No** — smoke‑test scratch output |
| `aura/artifacts/_smoke2/` | **161.0 MB** | **No** |
| `aura/artifacts/retrain_v2/` | 161.1 MB | v2 is the served model; `optimizer.pt` within is not |
| `aura/artifacts/retrain_v3/` | 161.1 MB | v3 is not served |
| `*optimizer.pt` (5 copies) | **266.8 MB** | resume‑training only, never at inference |

A grep across all `.py`, `.json` and `.md` for `_smoke`, `_smoke2` and `v1_broken_quality_head` returns **zero
hits outside the artifact tree itself**. That is ~720 MB of git‑tracked payload that no code path, test, doc or
config reads — plus five 84 MB `last_checkpoint.pt` files that are byte‑identical siblings.

Because it is in history, every `git clone` pays for it forever, and `git gc` cannot help. This also quietly
contradicts a CI assumption: `.github/workflows/ci.yml` justifies `--skip-brain` with *"the BraTS checkpoint is
gitignored, so a fresh clone never has it"* — true of BraTS, but a fresh clone still pulls a gigabyte of chest
and brain‑v1 checkpoints.

**Fix.** `git rm --cached` the three dead trees and all `optimizer.pt`, add them to `.gitignore`, and move the
genuinely‑served weights to Git LFS or the release‑asset flow that `deploy/fetch_models.py` and
`AURA_MODELS_URL` already implement — the mechanism exists and is used for BraTS; extend it to the rest. History
rewrite is optional and can wait; stopping the growth cannot.

### Finding 5.2 — `aura/change/` is a 28 kLOC untracked duplicate of the source tree · **Medium**

171 Python files, 28,304 lines, 4.1 MB, **0 tracked**, not in `.gitignore`, imported by nothing. The layout is a
doubled‑up copy — `change/common/common/`, `change/gateway/gateway/`, `change/ml/ml/` — which is the signature of
a refactor tool run into the wrong destination. It contains a second `run_pipeline.py` carrying the same
fabricated baseline table from §4.2, so a future grep‑and‑fix will silently miss one copy.

It also sits in `git status` permanently as `?? aura/change/`, which is how the 36 untracked entries that
actually matter (new tests, `services/quantum/`, `services/copilot/`, `bench/`) get lost in the noise.

**Fix.** Delete it. If any of it is wanted, it is recoverable from the tree it was copied from. If deletion feels
premature, move it outside the repo root today and `.gitignore` the path.

### Finding 5.3 — 48 modified / 36 untracked files, with new subsystems uncommitted · **Medium**

Uncommitted work includes whole subsystems: `services/quantum/`, `services/copilot/`, `services/agent/specialists/`,
`gateway/adapters/`, `gateway/api/`, `common/storage/`, `bench/`, `ml/training/train_qkl.py`, and 13 new test
modules. Meanwhile 24 files under `aura/artifacts/` are modified as a side effect of running the suite (§2.2),
which means the interesting diff is buried under regenerated binaries.

**Fix.** Land the source in reviewable commits; fix §2.2 so artifacts stop appearing in `git status` at all.

---

## 6. Structure — the dual‑root layout is justified, but has three traps

`aura/backend/README.md` §1 makes a good, explicit case for keeping `gateway/`, `services/`, `schemas/` and `ml/`
in place rather than physically relocating them under `backend/`: the chest stack is audited and calibrated, it
imports itself by top‑level name in hundreds of places, and a wholesale move would be a large untestable diff
through validated clinical code. It even names the correct sequencing for a future consolidation (move the chest
stack behind `ThoraxEngine` first, then relocate, with router tests as the net). **I agree with this decision**
and would not reopen it. But it leaves three sharp edges.

### Finding 6.1 — `aura/common/config.py` and `aura/common/config/` collide · **Medium**

```
aura/common/config.py            ← the module every caller imports
aura/common/config/
└── safety_policy.yaml           ← bare directory, no __init__.py
```

Today this resolves to the module (`is_pkg: False`) because a regular module outranks a namespace package. The
moment anyone adds `aura/common/config/__init__.py` — or a tool does it for them while packaging — the directory
becomes a regular package, wins the lookup, and **every `from aura.common.config import get_settings` in the
codebase breaks at once.** A one‑file mistake takes down the whole system, and the traceback will point at
imports rather than at the new file.

Compounding it, `config.py:97‑98` probes two locations for the YAML:

```python
_SAFETY_POLICY_PATH     = ROOT / "common" / "config" / "safety_policy.yaml"   # exists
_SAFETY_POLICY_PATH_ALT = ROOT / "config"  / "safety_policy.yaml"             # does not exist
```

and `_load_safety_policy()` falls back to hardcoded defaults on miss. So a path regression degrades silently to
built‑in thresholds instead of failing — in the module that decides when the system abstains from a clinical claim.

**Fix.** Rename the directory to `aura/common/policy/` (or move the YAML to `aura/config/safety_policy.yaml` and
delete the dead alternate probe). Then make a missing policy file **loud** — log at WARNING with the resolved
path, and have `deploy/preflight.py` assert the file exists and that the loaded thresholds match the served
`Settings`. Silent fallback is the wrong failure mode for an abstention policy.

### Finding 6.2 — two import roots must both be on `sys.path` · **Medium**

Modules mix `from aura.services.x import …` with bare `from services.x import …` / `from knowledge…`, so the
package only imports cleanly with **both** `aura-main/` and `aura-main/aura/` on the path. Demonstrated:

```
# aura-main/ only
FAIL aura.services.safety.controller -> ModuleNotFoundError: No module named 'knowledge'
FAIL aura.gateway.app               -> ModuleNotFoundError: No module named 'knowledge'

# both roots
OK   aura.gateway.app
```

pytest papers over this via `pythonpath = ["."]` plus rootdir insertion, so the suite passes and the breakage only
appears when something imports the package directly — a script, a notebook, a `python -c`, a WSGI/ASGI loader
configured slightly differently. The 175 KB `IMPORT_REPAIR_REPORT.md` in the repo root is evidence this has
already cost real time once. The `backend/` README explicitly accepts this debt; the mitigation is to stop it
growing.

**Fix.** Don't do the big rewrite. Add a lint gate that fails on new bare‑root imports (`grep -rE
"^from (services|gateway|schemas|knowledge|common|ml)\." --include=*.py aura/`, allowlist the current set,
shrink the allowlist over time), and document the two‑root requirement in `docs/ARCHITECTURE.md`.

### Finding 6.3 — `console.js` is 2,232 lines · **Low**

`aura/apps/web/js/console.js` is the single largest first‑party file in the repo and carries the whole clinical
console: worklist, case detail, differential, Grad‑CAM overlay, NeuroView, chat, report, export. It is also
modified in the working tree. Given that memory records a past bug where *"stale cached JS left buttons rendered
by fresh HTML with no handlers bound"* — the reason `Cache-Control: no-cache` exists in `app.py:113` — this file
is the highest‑risk unreviewed surface in the UI.

**Fix.** Split by panel along the seams that already exist, at whatever pace the UI work allows. Not urgent.

---

## 7. Things that are right, and should not be traded away

Reviews that list only defects mislead. These are load‑bearing and worth defending:

1. **`tests/test_doc_numbers.py`.** A mechanical guard that pins published tables to the artifact that produced
   them, with a docstring forbidding its own loosening. It caught a real drift (accuracy 0.667→0.6377). This
   pattern should be extended (§4.2, §4.3), never relaxed.
2. **Config commentary that records *why*.** `common/config.py:159‑170` explains the abstention recalibration,
   including the rejected values and the 91 %-abstention failure it fixed. `.env.example`'s "NOT set here"
   section explains that an empty `AURA_FUSION_BACKEND` would be read as a real backend name and break fusion.
   This is the kind of comment that survives a year.
3. **`AURA_ALLOW_FALLBACK_VISION` defaulted to `0` everywhere** — CI env, compose, `.env.example` — each with the
   same warning: at `1`, a container whose DenseNet weights failed to load serves fabricated findings and still
   reports healthy. Correctly identified as the most dangerous switch in the system, and pinned in three places.
4. **Label provenance is surfaced, not buried.** `services/report/clinical_report.py:255‑280` reads the gold
   hand‑read validation and the torchxrayvision cross‑check out of artifacts and states them in the report —
   including that nodule agreement is below chance and flagged unreliable. When the artifacts are missing it
   falls back to an honest generic sentence rather than a confident one.
5. **`deploy/preflight.py` as a boot gate.** `AURA_PREFLIGHT=strict` refusing to start on a missing served
   artifact is the right default. It should also have an opinion about auth (§3.2).
6. **The `backend/` non‑relocation decision.** Documented, reasoned, with a correct migration order for later.
7. **Real hardware provenance.** The fusion VQC actually ran on `ibm_marrakesh` with a retrievable job ID, and
   the docs were scrubbed of fabricated "tested on ibm_kyoto/osaka" claims. Ungated, untrained components
   (QAE, QBN) are marked as such rather than implied working.

---

## 8. Repaired during this review — kept as a process finding

These were reproduced, then fixed by the concurrent session before I finished writing. They are recorded because
the *class* of defect recurs, and because two of them were serving‑path breakage that no test caught at the time.

**8.1 — `schemas/contracts.py` was momentarily empty.** 32 of 47 test modules failed to collect
(`cannot import name 'StructuredPriors' / 'MultimodalContext' / 'AbstentionReason' …`). The file is now 514 lines.
A truncated core contracts module took out two thirds of the suite in one stroke.

**8.2 — the gateway could not boot.** `services/safety/__init__.py` eagerly imported `readiness`, which imported
a `coverage_ratio` that did not exist in `knowledge/guidelines/templates.py`. Verified against the real server:

```
uvicorn aura.gateway.app:app
  File "aura/gateway/app.py", line 23, in <module>      from .pipeline import Pipeline
  File "aura/gateway/pipeline.py", line 31, in <module>  from aura.services.safety import SafetyEngine, …
  File "aura/services/safety/readiness.py", line 22      from aura.knowledge.guidelines.templates import coverage_ratio
ImportError: cannot import name 'coverage_ratio'
```

Same for `get_safety_policy` / `SafetyPolicyThresholds`, imported by `safety/controller.py:12` and
`safety/readiness.py:20` from a `common/config.py` that did not define them. Both now exist.

**8.3 — clinician feedback contaminated the conformal calibration.** The sharpest of the four, and worth keeping
in full because the mechanism is subtle. `POST /v1/cases/{id}/feedback` called `_record_conformal_outcome`, and
the diagnosis it fed in defaulted to **the model's own top prediction**:

```python
diagnosis = payload.get("diagnosis", b.safety.top.value if b.safety else "")
...
if get_settings().aci_enabled and b.safety is not None:
    aci_info = _record_conformal_outcome(b, diagnosis)     # ← ACI update on unverified feedback
```

Three defects in four lines:

1. **Self‑confirmation.** With `diagnosis` absent from the payload, Adaptive Conformal Inference was scored
   against the model's own output — `covered=True` by construction — so q̂ shrank monotonically and the 90 %
   coverage guarantee decayed toward vacuity. Reproduced: `qhat 0.9000 → 0.8980, covered=True`.
2. **Unverified input on the verified path.** `POST /outcome` validates `source ∈ {pcr, expert_consensus,
   biopsy, pathology, clinical_course}`, validates the diagnosis enum, and rejects duplicates with 409.
   `/feedback` did none of that and reached the same ACI update.
3. **Ledger poisoning ⇒ HTTP 500 and a lockout.** `record_outcome()` also writes an `OutcomeRow`, and
   `outcomes.case_id` is UNIQUE. So a *second* "accept" on the same case raised an unhandled
   `sqlite3.IntegrityError: UNIQUE constraint failed: outcomes.case_id` — a 500 on the most common clinician
   action — and the first one already made `has_outcome()` true, so the authoritative `/outcome` submission for
   that case was permanently blocked with 409.

`tests/test_outcome_decoupling.py` states the intent exactly (*"Clinician feedback (accept / edit / reject) must
NOT update the ACI threshold"*) and was failing. Now fixed: feedback returns no `conformal` key, writes no
outcome row, and a subsequent biopsy‑sourced outcome runs the ACI update correctly. **Two follow‑ups remain:**
`storage.py:302‑309`'s docstring still says *"called whenever a clinician confirms a case's diagnosis (see the
feedback endpoint)"*, which is now false and will mislead the next person to wire a caller; and the endpoint
should catch `IntegrityError` and return 409 rather than relying on a pre‑check.

**8.4 — `bench/` and `ModelRegistry`'s checkpoint API did not exist.** 18 failures in `test_architecture_8.py`
(`No module named 'bench'`, `'ModelRegistry' object has no attribute 'register_checkpoint' / 'verify_checkpoint'`,
`cannot import name 'sha256_bytes'`). `bench/` now exists (untracked) and the registry has
`register_checkpoint`, `verify_checkpoint`, `verify_all`, `sha256_file`, `sha256_bytes`.

**The process finding.** In all four cases, tests encoding the correct intent existed and were red, and in two of
them the *server could not start*. What was missing was a gate that made red block anything. CI runs on
`branches: ["**"]` and would have caught every one — but CI has been red on an unrelated missing dependency
(§2.1), so a red build carries no information. **Fix §2.1 first**; it is what makes every other guard in this
repo mean something.

---

## 9. Priority order

| # | Finding | Sev | Status |
|---|---|---|---|
| 1 | §2.1 `pytest-asyncio` missing ⇒ CI red, 13 tests dead | High | ✅ **Fixed** — pinned + `asyncio_mode`; surfaced a live NaN bug (§10.1) |
| 2 | §3.1 all `GET`s unauthenticated, incl. FHIR/HL7 export | **Critical** | ✅ **Fixed** — gate moved to path‑based; 22 regression tests |
| 3 | §3.2 documented Docker path ships auth off | High | ✅ **Fixed** — `.env.example`, compose, preflight warning |
| 4 | §4.2 hardcoded competitor numbers as measurements | High | ✅ **Fixed** — cited, non‑equivalence disclosed, pinned by tests |
| 5 | §4.1 "Classical Superiority" unsupported at n=69 | High | ✅ **Fixed** — CIs + McNemar computed, claim restated |
| 6 | §5.1 ~720 MB dead tracked binaries | High | ✅ **Fixed** — 1339.5 → 216.0 MB tracked |
| 7 | §4.3 `KNOWN_LIMITATIONS.md` superseded thresholds | Med | ✅ **Fixed** — synced + pinned to `get_settings()` |
| 8 | §2.2 order‑dependent tests mutate served calibration | Med | ✅ **Fixed** — DB isolated; session guard fails the run |
| 9 | §6.1 `config.py` / `config/` collision + silent fallback | Med | ✅ **Fixed** — moved to `common/policy/`; fallback now loud |
| 10 | §5.2 delete `aura/change/` (28 kLOC dead duplicate) | Med | ✅ **Fixed** — archived outside the repo, gitignored |
| 11 | §6.2 lint gate on new bare‑root imports | Med | ✅ **Fixed** — and the debt was already zero; now enforced |
| 12 | §8.3 stale `record_outcome` docstring; catch `IntegrityError` | Med | ✅ **Fixed** — duplicate returns 409, not 500 |
| 13 | §5.3 land uncommitted subsystems | Med | ⏸ **Your call** — see §11 |
| 14 | §3.3 note per‑replica rate limiting | Low | ✅ **Fixed** — plus per‑replica ACI‑state caveat |
| 15 | §6.3 split `console.js` | Low | ⏸ **Deliberately deferred** — see §11 |

---

## 10. Remediation log

Everything below was executed and verified; the suite ends at **738 passed, 0 failed,
exit 0** (was 669 passed / 33 failed).

### 10.1 A real bug the dead tests were hiding

Installing `pytest-asyncio` did not just turn 13 failures green — it made them *run*
for the first time, and they immediately exposed a NaN. `ConsensusEngine._weighted_average`
computed each agent's mean peer‑agreement over `[o for o in agreement[name] if o != name]`.
For a **single agent** that list is empty, `np.mean([])` is `nan`, and the two `or 1.0`
guards downstream did not catch it **because `nan` is truthy**:

```
peer list        : []
avg_agreement    : nan
total_w (or 1.0) : nan   <-- guard does not fire
final posterior  : {'normal': nan, ...}
```

A single‑agent consensus therefore returned an all‑NaN probability distribution. The
existing test asserted only `consensus_entropy` and `confidence`, never the posterior,
so it passed. Fixed by giving a peerless agent weight 1.0 (matching `_consensus_entropy`,
which already returns 0.0 entropy for `< 2` verdicts) and re‑guarding on
`np.isfinite(total) and total > 0` instead of truthiness. The test now asserts the
posterior is finite, uniform and sums to 1.

### 10.2 Security

`enforce()` moved out of the method conditional into a path‑based gate with an explicit
`PUBLIC_PATHS` / `PUBLIC_PREFIXES` allowlist, so a new endpoint is protected by default
and forgetting to register it fails **closed**. Verified under both modes:

| | auth off (demo default) | auth on |
|---|---|---|
| `/`, `/app`, `/history`, `/static/*` | 200 | **200** (dashboard still renders) |
| `/v1/health` | 200 | **200** (probes still work) |
| `/v1/cases`, `/v1/cases/{id}`, `/export/fhir`, `/export/hl7`, `/v1/admin/safety` | 200 | **401** |
| `/v1/cases` + token, no `x-aura-user` | — | **403** (calls stay attributable) |

`aura/tests/test_gateway_authz.py` (22 tests) pins this, including a structural test that
enumerates the live app and fails if any `/v1` route other than `/v1/health` is public.

### 10.3 A latent startup crash, found on the way

`pick()` in `common/config.py` cast **any set** env var, so `AURA_RATE_LIMIT_RPM=` in a
`.env` reached `int("")` and killed the process at boot — and `AURA_FUSION_BACKEND=` was
read as a real backend name, which is why `.env.example` carried a warning never to list
it. A present‑but‑empty `AURA_*` var is now treated as unset. Both footguns are gone and
the warning has been rewritten accordingly.

### 10.4 Honesty layer

- `benchmark_report.md` is regenerated with every competitor row cited (Isensee 2021,
  Hatamizadeh 2021, MONAI tutorial) and marked **not measured here**; a `[!WARNING]`
  block states that AURA's Dice is pooled 2‑D over 7,531 slices while the cited values
  are per‑case 3‑D. The invented `0.42 s` GPU latency is removed rather than estimated.
- `comparison_table.csv` gained `Provenance` and `Scoring` columns, because a CSV strips
  the prose that qualifies a table — which is how the literature values got mistaken for
  an experiment in the first place.
- The fusion rows are now read from `benchmark.json` instead of hardcoded, and
  `_fusion_rows()` **computes** the Jeffreys intervals and the best‑case McNemar p‑value,
  so the caveat weakens on its own if the split ever grows.
- `docs/BENCHMARKS.md` §1 replaces "Classical Superiority" with the measured position:
  48/69 vs 44/69, CIs `[0.581, 0.795]` and `[0.520, 0.744]`, McNemar p = 0.125 best case,
  and an explicit statement that classical is served for interpretability and cost, **not**
  a demonstrated accuracy advantage.
- `docs/DEPLOYMENT.md` documented a `AURA_SEC_AUTH_ENABLED` variable that **does not
  exist anywhere in the codebase**, and a `0.45` low‑confidence default that is really
  `0.3`. Both corrected.
- Four new guards in `test_doc_numbers.py` pin attribution, the 2‑D/3‑D disclosure, the
  presence of a significance statement, and the safety thresholds.

### 10.5 Test isolation

`aura/conftest.py` (new) redirects `AURA_DB_PATH` to a throwaway database before
`common/config` is imported. The live worklist had accumulated `CASE-TEST-1`, `R1`, `R2`,
`V1` — and, more seriously, a `conformal_state` row, meaning **test runs were writing the
served adaptive‑conformal threshold q̂**. Those rows were purged (a backup of the polluted
DB is in the scratchpad) and a full run now leaves the live DB byte‑identical.

A `pytest_sessionfinish` hook hashes every git‑tracked file under `aura/artifacts/` before
and after the session and **fails the run** if any changed. Verified by probe: a test that
appends one byte to `registry.json` passes, and the session still exits 1 with a named
diagnostic. This is the exact mechanism that let an n=16 calibration fit silently clobber
a validated n=2099 one.

### 10.6 Hygiene

Untracked (files kept on disk, now gitignored): `brain/v1_broken_quality_head/` 400 MB,
`_smoke/` + `_smoke2/` 322 MB, five `optimizer.pt` 267 MB, five `last_checkpoint.pt`
402 MB. All verified unreferenced; the served `best_model.pt` and every artifact in
`preflight.py`'s required list stay tracked and preflight still passes.
`aura/change/` moved to `E:\AURA\_archived_change_tree_20260730\`.

### 10.7 Import roots — better than reported

§6.2 recommended an allowlist that shrinks over time. It turned out the repair was already
complete: **zero** bare‑root first‑party imports remain, and the package imports cleanly
with only the repo root on `sys.path`. So `test_import_hygiene.py` enforces the stronger
property directly — an AST scan plus a subprocess import with a clean path.

---

## 11. Two items that need your decision

**§5.3 — Committing.** I have not committed anything. The tree now holds 49 modified,
89 staged deletions (the untracking) and 41 untracked paths, spanning both my changes and
the concurrent session's in‑flight subsystems (`services/quantum/`, `services/copilot/`,
`bench/`, `gateway/adapters/`, 15 new test modules). Splitting that into reviewable
commits is a judgement call about what belongs together, and it is your repo and your
branch — tell me how you want it sliced and I will do it.

**§6.3 — `console.js`.** Deliberately not split, and I want to be explicit about why
rather than quietly skip it. The file is a single IIFE — `window.CONSOLE = (() => { … })()`
— with 52 closure‑scoped functions sharing mutable state and a `return { boot }`, loaded
as a **classic** script (`<script src="/static/js/console.js?v=7">`), depending on a global
`FX`. Splitting it means either converting to ES modules, which makes execution deferred —
precisely the timing class that caused the earlier "buttons rendered by fresh HTML with no
handlers bound" bug that `Cache-Control: no-cache` exists to work around — or hoisting the
shared closure state to globals, which is worse than the encapsulation it has now. Neither
is a safe mechanical transform, the benefit is maintainability only with zero functional
gain, and the file currently carries uncommitted edits from the other session that I would
be tangling with. It is the one item where doing it now costs more than it returns. Happy
to do it as a focused piece of work with the UI exercised panel by panel, if you want it.

---

## Appendix — how to reproduce

```bash
# Test suite, exactly as CI runs it (from aura/)
cd aura && python -m pytest -q -m "not slow" --durations=10

# Import smoke across both roots
PYTHONPATH="E:\AURA\aura-main;E:\AURA\aura-main\aura" python -c "import aura.gateway.app; print('ok')"

# Served settings vs. the docs (§4.3)
python -c "from aura.common.config import get_settings as g; s=g(); print(s.epistemic_threshold, s.ood_threshold, s.abstention_conformal_size)"

# Tracked repo weight and the dead trees (§5.1)
git ls-files -z | while IFS= read -r -d '' f; do [ -f "$f" ] && stat -c%s "$f"; done | awk '{s+=$1} END {printf "%.1f MB / %d files\n", s/1048576, NR}'

# Confirm the dead artifact trees are referenced nowhere (§5.1)
grep -rn "_smoke\|v1_broken_quality_head" --include=*.py --include=*.json --include=*.md . | grep -v "\.venv\|change/"
```

Environment used: `E:\AURA\venv` (Python 3.12.10). The global `py -3.14` interpreter reproduces the same import
behaviour. Statistics in §4.1 computed with `scipy.stats` (Jeffreys interval for the binomial CIs, exact binomial
for McNemar).
