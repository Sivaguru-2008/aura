# AURA — Repository-Wide Import & Package-Structure Repair

**Date:** 2026-07-29
**Trigger:** `ModuleNotFoundError: No module named 'common'` running `python -m aura.aura_cli serve`
**Result:** ✅ `python -m aura.aura_cli serve` starts and serves 27 routes; `compileall` clean; 256/257 modules import; 398/398 tests pass.

---

## 1. Root cause

This was **not** a missing `common` package. `aura/common/config.py` was present the whole time.

The repository was refactored: the project used to live at `E:\AURA\aura\aura\` and be imported with
**`aura/` itself on `sys.path`**, so its subdirectories were *top-level* packages — `common`, `services`,
`gateway`, `ml`, `mimic`, `schemas`, `backend`. It was later moved to `E:\AURA\aura-main\aura\` and is now
invoked as `python -m aura.aura_cli`, which makes **`aura` the package root**. Every one of the ~1000
intra-project imports was left addressing the *old* root.

`aura/aura_cli.py` asked for `common.config`. Under the new entrypoint the only importable name is
`aura.common.config`. Hence the error — and it was the first of ~1000 identical failures, not a one-off.

### Evidence the refactor happened (three independent confirmations)

| # | Evidence | What it proves |
|---|---|---|
| 1 | `Dockerfile` line 67 set `PYTHONPATH=/app/aura` | The container deliberately put `aura/` on the path so `common`, `gateway`, … resolved as top-level. |
| 2 | `aura/aura_cli.py` docstring: *"Run from the aura/ directory."* and `py -m aura_cli serve` | The documented entrypoint was `aura_cli`, not `aura.aura_cli`. |
| 3 | A **stale editable install** in `E:\AURA\venv`: `aura-0.1.0.dist-info`, whose finder mapped `common`, `services`, `gateway`, `mimic`, `ml`, `schemas`, `aura_cli` → `E:\AURA\aura\aura\…` — **a directory that no longer exists** | Direct proof of both the old flat layout *and* the move. |

**No package was missing, renamed, or deleted. Nothing needs manual restoration.**

---

## 2. What was changed

| Category | Count |
|---|---|
| Import statements rewritten (automated AST codemod) | **1 025** across **211 files** |
| — of those, rewritten to intra-package **relative** imports | 398 |
| — of those, rewritten to absolute `aura.*` | 627 |
| Dynamic / string-literal module references repaired by hand | **12** |
| Stale Sphinx cross-references in docstrings | **74** in 43 files |
| `__init__.py` files created | **5** |
| Deployment / CI files corrected | **6** |
| Launchers + documentation corrected (`-m aura_cli` → `-m aura.aura_cli`) | **12 files** |
| Conflicting package roots removed | **1** |
| Fake files, stubs, placeholders, or commented-out imports created | **0** |
| `sys.path` manipulation or `PYTHONPATH` added | **0** |

### 2.1 Import rewrite convention

Applied by an AST-driven codemod (module token only — formatting, comments, `# noqa`, and multi-line
parenthesised lists preserved):

- **Relative** when the source file and the target share at least `aura.<subpackage>` *and* the hop is
  one or two levels (`.sibling`, `..parent`). This is the "prefer relative inside packages" rule.
- **Absolute `aura.<...>`** for everything else — cross-subpackage imports, and any file outside
  `aura/` (root scripts, `deploy/`). Deep relative chains (`....x`) were deliberately avoided: they are
  a known refactoring hazard and read far worse than the absolute form.

```diff
- from backend.foundation.mri.metadata import MRIMetadata     # same subpackage
+ from .metadata import MRIMetadata

- from backend.foundation.mri.config import LoaderConfig      # parent subpackage
+ from ..config import LoaderConfig

- from schemas.contracts import StudyInput                    # cross-subpackage
+ from aura.schemas.contracts import StudyInput
```

Every distinct mapping is in **Appendix A**; every individual statement is in **Appendix B**.

### 2.2 Dynamic and string-literal references (invisible to AST rewriting)

These do not appear in the import graph, so they were found by targeted search and fixed individually.
Four of them were silently breaking tests until repaired.

| File | Was | Now |
|---|---|---|
| `aura/aura_cli.py:92` | `uvicorn.run("gateway.app:app", …)` | `uvicorn.run("aura.gateway.app:app", …)` |
| `aura/backend/vision/brain/__init__.py` | `_LAZY` mapped to absolute `backend.vision.brain.*` | mapped to `.ingest`, `.dataset`, … resolved against `__name__` — now rename-proof |
| `aura/backend/vision/brain/ingest.py` | `__import__("backend.vision.brain.types", fromlist=[…])` ×2 | real module-level imports of `BRAIN_VISION_VERSION` / `FOUNDATION_VERSION` |
| `aura/backend/vision/brain/ingest.py:566` | `implementation="backend.vision.brain.ingest"` | `"aura.backend.vision.brain.ingest"` (provenance label written into cache manifests) |
| `aura/backend/vision/brain/cli.py:237` | `__import__("backend.vision.brain.types", …)` | `from .types import BRAIN_VISION_VERSION` |
| `aura/tests/test_audit_repairs.py:163` | `__import__("schemas.contracts", fromlist=["ReasoningStep"])` | `ReasoningStep` imported normally at the top |
| `aura/tests/test_mri_foundation.py` ×3 | `patch("backend.foundation.mri.standardize.…")` | `patch("aura.backend.foundation.mri.standardize.…")` |
| `aura/tests/test_mri_intake_manager.py:184` | `patch("backend.engines.neuro.multisequence._check_order")` | `patch("aura.backend.engines.neuro.multisequence._check_order")` |
| `aura/tests/test_startup_regression.py:17` | `mock.patch("ml.vision_cxr.inference.VisionModel")` | `mock.patch("aura.ml.vision_cxr.inference.VisionModel")` |
| `audit_all.py:406-407` | `import_module("services.fusion.ensemble" / ".learnable")` | `aura.services.fusion.…` |

> `logging.getLogger("mimic.cleaning")` and its ~15 siblings were **left alone deliberately** — those are
> logger channel names, not module references, and renaming them would silently break any log-filtering
> configuration keyed to them.

### 2.3 Package initialization

| File | Why |
|---|---|
| `aura/__init__.py` | **The core fix.** `aura` was an *implicit namespace package* — it worked by accident and gave no import root. Now an explicit package. Deliberately imports nothing: the subpackages pull in torch/cv2/the quantum stack, and paying that to reach `aura.common.config` would slow every CLI start and the container health check. |
| `aura/tests/__init__.py` | `aura/tests/test_mri_intake_manager.py` imports fixtures from `test_mri_foundation`; without this the two modules collided in pytest's rootdir import mode. |
| `aura/services/enterprise/__init__.py` | Had real modules (`fhir`, `hl7`, `dicom_listener`) but no `__init__`. Kept import-free so a deployment needing only FHIR doesn't fail because the DICOM stack is absent. |
| `aura/backend/services/reasoning/__init__.py` | Had `progression.py`, `tracking.py` but no `__init__`. |
| `deploy/__init__.py` | Lets the deploy scripts run as `python -m deploy.preflight` from the repo root, which is what makes `import aura` resolve **without** a `PYTHONPATH` entry. |

`aura/demo_data/` was **intentionally not** made a package: `prepare_demo_data.py` is a standalone
generator, nothing imports it, and it has no first-party imports.

### 2.4 Conflicting package root removed

`E:\AURA\venv` carried a broken editable install (`aura 0.1.0`) whose meta-path finder claimed the
top-level names `aura_cli`, `common`, `gateway`, `mimic`, `ml`, `schemas`, `services` and pointed them at
the deleted `E:\AURA\aura\aura\`. Uninstalled (`pip uninstall aura`). It could resolve nothing — its
target directory does not exist — and it shadowed exactly the names this repair depends on.

### 2.5 Deployment and CI kept consistent

The re-rooting invalidates every deployment invocation, so these were corrected in the same pass:

| File | Change |
|---|---|
| `Dockerfile` | Removed `PYTHONPATH=/app/aura`; `WORKDIR /app` + `-m` is now sufficient. Build-time preflight → `python -m deploy.preflight`. |
| `deploy/entrypoint.sh` | ASGI target `gateway.app:app` → `aura.gateway.app:app`; `--app-dir /app/aura` → `/app`; `cd /app/aura && python -m aura_cli` → `cd /app && python -m aura.aura_cli`; preflight → `-m deploy.preflight`. |
| `.github/workflows/ci.yml` | `python deploy/preflight.py` → `python -m deploy.preflight`; same for `smoke_test`. |
| `.github/workflows/docker.yml` | `python deploy/smoke_test.py` → `python -m deploy.smoke_test`. |
| `aura/run.sh` | `cd` to the repo root instead of `aura/`; `-m aura_cli` → `-m aura.aura_cli`. |
| `deploy/preflight.py` | Diagnostic message now names the real ASGI target. |

### 2.6 Launchers and documentation

The re-rooting changes the documented command, so every place that told a user how to start AURA was
corrected — otherwise the repo would ship instructions that reproduce the original error.

| File | Change |
|---|---|
| `aura/run.sh`, `aura/run.bat` | **Functional.** `cd` to the repo root instead of `aura/`; `-m aura_cli` → `-m aura.aura_cli`; requirements path → `aura/requirements.txt`. |
| `aura/aura_cli.py` docstring | Usage block and *"Run from the aura/ directory"* → run from the repository root. |
| `aura/apps/web/js/console.js` | The GATEWAY OFFLINE banner told users to run the old command. |
| `README.md`, `WHAT_IS_AURA.md` | `cd aura/aura` → `cd aura`; install path and `-m` target. |
| `docs/BENCHMARKS.md`, `docs/TRAINING_GUIDE.md`, `docs/VALIDATION.md`, `docs/DEPLOYMENT.md`, `docs/deployment_readiness.md`, `JUDGE_QA_PREP.md` | `cd` target + `-m` target. |
| `aura/tests/test_doc_numbers.py`, `presentation/build_deck.py`, `presentation/build_pitch.py` | Command strings shown in skip messages and generated slides. |

### 2.7 One latent bug found and fixed

`aura/aura_cli.py` re-execs into `E:\AURA\venv` on Windows when torch is missing. It did:

```python
subprocess.call([venv_python] + sys.argv)      # sys.argv[0] is a FILE PATH
```

Under `-m`, `sys.argv[0]` is the full path to `aura_cli.py`, so the child ran the file **by path** — which
puts `aura/` on `sys.path`, not the repo root. Post-refactor the child would have died with the exact
`ModuleNotFoundError` this redirect exists to avoid. Now:

```python
subprocess.call([venv_python, "-m", "aura.aura_cli", *sys.argv[1:]], cwd=str(repo_root))
```

The hardcoded venv path is pre-existing; it now honours `AURA_VENV_PYTHON` if set.

---

## 3. Missing modules

**No first-party source file is missing.** Nothing was reconstructed, stubbed, or invented.

One module fails to import, for a reason unrelated to structure:

| Module | Failure | Assessment |
|---|---|---|
| `aura.services.enterprise.dicom_listener` | `ModuleNotFoundError: No module named 'pynetdicom'` | **Environment gap, not a code defect.** `pynetdicom>=2.0` *is* declared in `requirements-docker.txt:24`; it is simply not installed in `E:\AURA\venv`. The gateway already degrades gracefully: `[gateway] WARNING: mock DICOM listener failed to start`. Install with `pip install pynetdicom` if the mock PACS listener is wanted locally. |

Third-party packages imported by the repo but absent from **both** local venvs — all in optional
research/analysis paths, none on the serve path, none blocking startup:

`psutil` (`ml/evaluation/perf_benchmark.py`), `shap` (`mimic/explain.py`, `services/explain/engine.py`),
`SimpleITK` (`backend/foundation/mri/standardize.py`), `seaborn` (`generate_assets.py`),
`catboost` / `lightgbm` / `xgboost` (`mimic/training.py`), `validators`.

> `presentation/_scripts/**` was **excluded from this repair by design.** It is vendored deck-building
> tooling with its own `sys.path` root and its own local `helpers` / `office` / `validators` packages; its
> unresolved `pptx` / `lxml` / `defusedxml` imports are not part of the `aura` package and rewriting them
> would break it.

---

## 4. Files that were moved

**None were moved during this repair.** The only move is the historical one that caused the problem, and
it predates this work:

```
E:\AURA\aura\aura\          →   E:\AURA\aura-main\aura\
  (imports NOT updated)          (imports updated by this repair)
```

Recovered from the stale editable install's `direct_url.json`: `file:///E:/AURA/aura/aura`.

---

## 5. Files requiring manual restoration

**None.** Every broken reference resolved to a module that exists in the repository.

---

## 6. Validation

All commands run from `E:\AURA\aura-main`.

| Check | Command | Result |
|---|---|---|
| Static import scan | AST sweep of all 249 project `.py` files | **0 stale first-party imports** (was 1 775 names / 1 025 statements) |
| Syntax | `python -m compileall .` | **exit 0**, no errors |
| Real imports | every module in `aura/` + `deploy/` imported individually | **256 OK / 1 failed** (the `pynetdicom` env gap above) |
| Package integrity | every dir containing `.py` has `__init__.py` where required | **47 packages**, 0 missing |
| Circular imports | implied by the full-import pass succeeding | **none** |
| Test suite | `python -m pytest aura/tests -m "not slow"` | **398 passed**, exit 0 |
| **Entrypoint** | **`python -m aura.aura_cli serve`** | **✅ starts, ready in ~11 s** |

Startup log:

```
[aura.backend.registry] registered engine engine='thorax'    status='available' modalities=['chest_xray']
[aura.backend.registry] registered engine engine='neuromind' status='available' modalities=['brain_mri']
[aura.backend.router.detector] detector ready signatures=['ChestRadiographSignature', 'BrainMRISignature',
                                                          'HeadCTSignature', 'DicomModalitySignature']
[aura.backend.bootstrap] modality router installed engines=['thorax', 'neuromind']
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

```
GET /v1/health → 200
{"status":"ok","backend":"quantum","trained":true,"cases":53}
27 routes registered
```

---

## 7. Final package tree

47 packages, all with explicit `__init__.py`, all rooted at `aura`.

```
aura-main/                          repo root — the import root
├── aura/                           ← now a real package (was an implicit namespace package)
│   ├── __init__.py                 NEW
│   ├── aura_cli.py                 entrypoint: python -m aura.aura_cli
│   ├── run_ablation.py
│   ├── backend/                    Medical-AI-OS routing layer
│   │   ├── api/
│   │   ├── core/{router,shared,upload}/
│   │   ├── engines/{base,neuro,thorax}/
│   │   ├── foundation/mri/io/
│   │   ├── models/
│   │   ├── services/reasoning/     __init__.py NEW
│   │   └── vision/brain/{io,model}/
│   ├── common/                     config, eventbus, mathx, anatomy
│   ├── gateway/                    FastAPI app — ASGI target aura.gateway.app:app
│   ├── knowledge/guidelines/
│   ├── mimic/                      MIMIC-CXR corpus pipeline
│   ├── ml/{evaluation,explain_demo,training,vision_cxr}/
│   ├── schemas/                    clinical + contracts
│   ├── services/                   intelligence engines
│   │   ├── agent/ explain/ fusion/ inference/ memory/ models/
│   │   ├── reasoning/ recommend/ report/ safety/ vision/
│   │   └── enterprise/             __init__.py NEW
│   ├── tests/                      __init__.py NEW
│   ├── apps/web/                   static console (not Python)
│   ├── artifacts/                  trained models + evaluation output (not Python)
│   └── demo_data/                  standalone script — intentionally not a package
├── deploy/                         __init__.py NEW — run as python -m deploy.preflight
├── Dockerfile  docker-compose.yml  .github/workflows/
└── audit_all.py  demo.py  generate_assets.py  run_failure_demo.py  verify_exports.py
```

---

## 8. Follow-ups (not blocking, not done)

1. **No packaging metadata.** There is no `pyproject.toml`, so `pip install -e .` is impossible and the
   layout is enforced only by convention. Adding one would make the `aura` root explicit and let CI
   install rather than rely on cwd. Not added here — it changes the Docker build contract and is a
   separate decision.
2. **`aura/requirements.txt` is incomplete** relative to what the code imports (`pynetdicom`, `psutil`,
   `shap`, `SimpleITK`, `timm`, `torchxrayvision`, `qiskit` appear only in `requirements-docker.txt` or
   nowhere). Worth reconciling the two files.
3. **Two divergent virtualenvs.** `aura-main/.venv` is Python 3.14 and lacks h5py/qiskit/timm/
   torchxrayvision; `E:\AURA\venv` is 3.12 and has them. The CLI hard-redirects to the latter. Retiring
   the stale 3.14 `.venv` would remove a foot-gun.
4. **`aura/pytest-tmp/`** is an unreadable leftover pytest temp directory in the source tree; safe to delete.

---

*Appendices A and B below list every broken module reference and every individual repaired import.*
## Appendix A — every broken module reference and its repair

`.`/`..` targets are intra-package relative imports; the resolved absolute
module is the same in every case. Count = number of import statements rewritten.

| # | Broken module (old root) | Repaired to | Count |
|---|---|---|---|
| 1 | `backend.api.routes` | `.routes` | 1 |
| 2 | `backend.api.routes` | `.api.routes` | 1 |
| 3 | `backend.api.routes` | `aura.backend.api.routes` | 1 |
| 4 | `backend.bootstrap` | `aura.backend.bootstrap` | 2 |
| 5 | `backend.core.router.detector` | `.detector` | 2 |
| 6 | `backend.core.router.detector` | `aura.backend.core.router.detector` | 1 |
| 7 | `backend.core.router.features` | `.features` | 3 |
| 8 | `backend.core.router.router` | `aura.backend.core.router.router` | 4 |
| 9 | `backend.core.router.router` | `.router` | 1 |
| 10 | `backend.core.router.router` | `..core.router.router` | 1 |
| 11 | `backend.core.router.signatures` | `.signatures` | 2 |
| 12 | `backend.core.shared.errors` | `aura.backend.core.shared.errors` | 11 |
| 13 | `backend.core.shared.errors` | `..core.shared.errors` | 7 |
| 14 | `backend.core.shared.errors` | `..shared.errors` | 3 |
| 15 | `backend.core.shared.errors` | `.errors` | 1 |
| 16 | `backend.core.shared.logging` | `aura.backend.core.shared.logging` | 30 |
| 17 | `backend.core.shared.logging` | `..shared.logging` | 5 |
| 18 | `backend.core.shared.logging` | `..core.shared.logging` | 2 |
| 19 | `backend.core.shared.logging` | `.core.shared.logging` | 1 |
| 20 | `backend.core.shared.logging` | `.logging` | 1 |
| 21 | `backend.core.shared.types` | `aura.backend.core.shared.types` | 7 |
| 22 | `backend.core.shared.types` | `..shared.types` | 6 |
| 23 | `backend.core.shared.types` | `..core.shared.types` | 2 |
| 24 | `backend.core.shared.types` | `.types` | 1 |
| 25 | `backend.core.upload` | `..core.upload` | 3 |
| 26 | `backend.core.upload.intake` | `aura.backend.core.upload.intake` | 6 |
| 27 | `backend.core.upload.intake` | `.intake` | 1 |
| 28 | `backend.engines.base.contract` | `.contract` | 2 |
| 29 | `backend.engines.base.contract` | `..base.contract` | 2 |
| 30 | `backend.engines.base.contract` | `aura.backend.engines.base.contract` | 1 |
| 31 | `backend.engines.base.registry` | `aura.backend.engines.base.registry` | 2 |
| 32 | `backend.engines.base.registry` | `..base.registry` | 2 |
| 33 | `backend.engines.base.registry` | `.engines.base.registry` | 1 |
| 34 | `backend.engines.base.registry` | `.registry` | 1 |
| 35 | `backend.engines.base.registry` | `..engines.base.registry` | 1 |
| 36 | `backend.engines.neuro.bundle` | `.bundle` | 2 |
| 37 | `backend.engines.neuro.bundle` | `aura.backend.engines.neuro.bundle` | 1 |
| 38 | `backend.engines.neuro.calibration` | `.calibration` | 1 |
| 39 | `backend.engines.neuro.engine` | `aura.backend.engines.neuro.engine` | 6 |
| 40 | `backend.engines.neuro.engine` | `.engines.neuro.engine` | 1 |
| 41 | `backend.engines.neuro.engine` | `.engine` | 1 |
| 42 | `backend.engines.neuro.multisequence` | `.multisequence` | 3 |
| 43 | `backend.engines.neuro.multisequence` | `aura.backend.engines.neuro.multisequence` | 3 |
| 44 | `backend.engines.neuro.neuroinsight` | `.neuroinsight` | 1 |
| 45 | `backend.engines.neuro.neuroinsight` | `aura.backend.engines.neuro.neuroinsight` | 1 |
| 46 | `backend.engines.neuro.neuroview` | `.neuroview` | 1 |
| 47 | `backend.engines.neuro.neuroview` | `aura.backend.engines.neuro.neuroview` | 1 |
| 48 | `backend.engines.neuro.qkl` | `.qkl` | 1 |
| 49 | `backend.engines.neuro.qkl` | `aura.backend.engines.neuro.qkl` | 1 |
| 50 | `backend.engines.neuro.sequence_features` | `.sequence_features` | 1 |
| 51 | `backend.engines.thorax.engine` | `.engines.thorax.engine` | 1 |
| 52 | `backend.engines.thorax.engine` | `.engine` | 1 |
| 53 | `backend.foundation.mri` | `aura.backend.foundation.mri` | 3 |
| 54 | `backend.foundation.mri.config` | `.config` | 5 |
| 55 | `backend.foundation.mri.config` | `..config` | 2 |
| 56 | `backend.foundation.mri.config` | `aura.backend.foundation.mri.config` | 1 |
| 57 | `backend.foundation.mri.errors` | `aura.backend.foundation.mri.errors` | 7 |
| 58 | `backend.foundation.mri.errors` | `.errors` | 6 |
| 59 | `backend.foundation.mri.errors` | `..errors` | 5 |
| 60 | `backend.foundation.mri.errors` | `..foundation.mri.errors` | 1 |
| 61 | `backend.foundation.mri.geometry` | `aura.backend.foundation.mri.geometry` | 4 |
| 62 | `backend.foundation.mri.geometry` | `.geometry` | 4 |
| 63 | `backend.foundation.mri.geometry` | `..geometry` | 4 |
| 64 | `backend.foundation.mri.intake_manager` | `aura.backend.foundation.mri.intake_manager` | 2 |
| 65 | `backend.foundation.mri.intake_manager` | `..foundation.mri.intake_manager` | 1 |
| 66 | `backend.foundation.mri.io.base` | `.base` | 5 |
| 67 | `backend.foundation.mri.io.base` | `.io.base` | 4 |
| 68 | `backend.foundation.mri.io.base` | `aura.backend.foundation.mri.io.base` | 3 |
| 69 | `backend.foundation.mri.io.dicom_reader` | `aura.backend.foundation.mri.io.dicom_reader` | 1 |
| 70 | `backend.foundation.mri.io.dicom_reader` | `.io.dicom_reader` | 1 |
| 71 | `backend.foundation.mri.io.discovery` | `.io.discovery` | 1 |
| 72 | `backend.foundation.mri.io.nifti_reader` | `aura.backend.foundation.mri.io.nifti_reader` | 4 |
| 73 | `backend.foundation.mri.io.nifti_reader` | `.io.nifti_reader` | 2 |
| 74 | `backend.foundation.mri.io.nrrd_reader` | `aura.backend.foundation.mri.io.nrrd_reader` | 2 |
| 75 | `backend.foundation.mri.io.nrrd_reader` | `.io.nrrd_reader` | 1 |
| 76 | `backend.foundation.mri.loader` | `.loader` | 2 |
| 77 | `backend.foundation.mri.masking` | `.masking` | 6 |
| 78 | `backend.foundation.mri.masking` | `aura.backend.foundation.mri.masking` | 1 |
| 79 | `backend.foundation.mri.metadata` | `.metadata` | 7 |
| 80 | `backend.foundation.mri.metadata` | `..metadata` | 1 |
| 81 | `backend.foundation.mri.metadata` | `aura.backend.foundation.mri.metadata` | 1 |
| 82 | `backend.foundation.mri.pipeline` | `.pipeline` | 1 |
| 83 | `backend.foundation.mri.quality` | `.quality` | 3 |
| 84 | `backend.foundation.mri.quality` | `aura.backend.foundation.mri.quality` | 1 |
| 85 | `backend.foundation.mri.registration` | `.registration` | 3 |
| 86 | `backend.foundation.mri.sequence` | `.sequence` | 4 |
| 87 | `backend.foundation.mri.standardize` | `.standardize` | 2 |
| 88 | `backend.foundation.mri.standardize` | `aura.backend.foundation.mri.standardize` | 1 |
| 89 | `backend.foundation.mri.study` | `.study` | 2 |
| 90 | `backend.foundation.mri.study` | `aura.backend.foundation.mri.study` | 2 |
| 91 | `backend.foundation.mri.types` | `.types` | 13 |
| 92 | `backend.foundation.mri.types` | `aura.backend.foundation.mri.types` | 7 |
| 93 | `backend.foundation.mri.types` | `..types` | 5 |
| 94 | `backend.foundation.mri.volume` | `.volume` | 6 |
| 95 | `backend.foundation.mri.volume` | `aura.backend.foundation.mri.volume` | 1 |
| 96 | `backend.models.routing` | `aura.backend.models.routing` | 5 |
| 97 | `backend.models.routing` | `..models.routing` | 2 |
| 98 | `backend.models.routing` | `.routing` | 1 |
| 99 | `backend.services.dispatch` | `aura.backend.services.dispatch` | 2 |
| 100 | `backend.services.dispatch` | `..services.dispatch` | 1 |
| 101 | `backend.services.dispatch` | `.services.dispatch` | 1 |
| 102 | `backend.services.dispatch` | `.dispatch` | 1 |
| 103 | `backend.services.reasoning.progression` | `..services.reasoning.progression` | 1 |
| 104 | `backend.services.reasoning.progression` | `aura.backend.services.reasoning.progression` | 1 |
| 105 | `backend.services.reasoning.tracking` | `..services.reasoning.tracking` | 1 |
| 106 | `backend.services.reasoning.tracking` | `aura.backend.services.reasoning.tracking` | 1 |
| 107 | `backend.vision.brain.augment` | `.augment` | 1 |
| 108 | `backend.vision.brain.augment` | `aura.backend.vision.brain.augment` | 1 |
| 109 | `backend.vision.brain.checkpoint` | `.checkpoint` | 3 |
| 110 | `backend.vision.brain.checkpoint` | `aura.backend.vision.brain.checkpoint` | 2 |
| 111 | `backend.vision.brain.cli` | `aura.backend.vision.brain.cli` | 2 |
| 112 | `backend.vision.brain.config` | `.config` | 11 |
| 113 | `backend.vision.brain.config` | `..config` | 1 |
| 114 | `backend.vision.brain.config` | `aura.backend.vision.brain.config` | 1 |
| 115 | `backend.vision.brain.dataset` | `.dataset` | 5 |
| 116 | `backend.vision.brain.dataset` | `aura.backend.vision.brain.dataset` | 1 |
| 117 | `backend.vision.brain.degradations` | `.degradations` | 2 |
| 118 | `backend.vision.brain.degradations` | `..degradations` | 1 |
| 119 | `backend.vision.brain.degradations` | `aura.backend.vision.brain.degradations` | 1 |
| 120 | `backend.vision.brain.embeddings` | `.embeddings` | 2 |
| 121 | `backend.vision.brain.embeddings` | `aura.backend.vision.brain.embeddings` | 1 |
| 122 | `backend.vision.brain.errors` | `.errors` | 7 |
| 123 | `backend.vision.brain.errors` | `..errors` | 2 |
| 124 | `backend.vision.brain.errors` | `aura.backend.vision.brain.errors` | 1 |
| 125 | `backend.vision.brain.inference` | `aura.backend.vision.brain.inference` | 4 |
| 126 | `backend.vision.brain.inference` | `.inference` | 1 |
| 127 | `backend.vision.brain.ingest` | `.ingest` | 5 |
| 128 | `backend.vision.brain.ingest` | `aura.backend.vision.brain.ingest` | 2 |
| 129 | `backend.vision.brain.io.brats_h5` | `.brats_h5` | 1 |
| 130 | `backend.vision.brain.io.brats_h5` | `aura.backend.vision.brain.io.brats_h5` | 1 |
| 131 | `backend.vision.brain.losses` | `.losses` | 3 |
| 132 | `backend.vision.brain.losses` | `aura.backend.vision.brain.losses` | 1 |
| 133 | `backend.vision.brain.metrics` | `.metrics` | 2 |
| 134 | `backend.vision.brain.metrics` | `aura.backend.vision.brain.metrics` | 1 |
| 135 | `backend.vision.brain.model` | `.model` | 2 |
| 136 | `backend.vision.brain.model` | `.` | 2 |
| 137 | `backend.vision.brain.model` | `aura.backend.vision.brain.model` | 1 |
| 138 | `backend.vision.brain.model.blocks` | `.blocks` | 4 |
| 139 | `backend.vision.brain.model.decoder` | `.decoder` | 1 |
| 140 | `backend.vision.brain.model.encoder` | `.encoder` | 1 |
| 141 | `backend.vision.brain.model.heads` | `.heads` | 2 |
| 142 | `backend.vision.brain.model.network` | `.model.network` | 5 |
| 143 | `backend.vision.brain.model.network` | `.network` | 1 |
| 144 | `backend.vision.brain.model.registry` | `.registry` | 4 |
| 145 | `backend.vision.brain.output` | `aura.backend.vision.brain.output` | 8 |
| 146 | `backend.vision.brain.output` | `.output` | 3 |
| 147 | `backend.vision.brain.sampling` | `.sampling` | 2 |
| 148 | `backend.vision.brain.sampling` | `aura.backend.vision.brain.sampling` | 1 |
| 149 | `backend.vision.brain.train` | `.train` | 3 |
| 150 | `backend.vision.brain.train` | `aura.backend.vision.brain.train` | 3 |
| 151 | `backend.vision.brain.types` | `.types` | 14 |
| 152 | `backend.vision.brain.types` | `aura.backend.vision.brain.types` | 9 |
| 153 | `backend.vision.brain.types` | `..types` | 3 |
| 154 | `backend.vision.brain.validate` | `.validate` | 2 |
| 155 | `common` | `aura.common` | 1 |
| 156 | `common.anatomy` | `aura.common.anatomy` | 4 |
| 157 | `common.config` | `aura.common.config` | 63 |
| 158 | `common.eventbus` | `aura.common.eventbus` | 1 |
| 159 | `common.mathx` | `aura.common.mathx` | 29 |
| 160 | `gateway.app` | `aura.gateway.app` | 6 |
| 161 | `gateway.pipeline` | `aura.gateway.pipeline` | 10 |
| 162 | `gateway.pipeline` | `.pipeline` | 2 |
| 163 | `gateway.security` | `aura.gateway.security` | 9 |
| 164 | `gateway.security` | `.security` | 3 |
| 165 | `gateway.seed` | `.seed` | 1 |
| 166 | `gateway.storage` | `aura.gateway.storage` | 6 |
| 167 | `gateway.storage` | `.storage` | 3 |
| 168 | `knowledge.guidelines.templates` | `aura.knowledge.guidelines.templates` | 2 |
| 169 | `mimic.cleaning` | `.cleaning` | 2 |
| 170 | `mimic.cleaning` | `aura.mimic.cleaning` | 1 |
| 171 | `mimic.config` | `aura.mimic.config` | 15 |
| 172 | `mimic.config` | `.config` | 9 |
| 173 | `mimic.evaluation` | `aura.mimic.evaluation` | 2 |
| 174 | `mimic.explain` | `aura.mimic.explain` | 1 |
| 175 | `mimic.features` | `.features` | 2 |
| 176 | `mimic.features` | `aura.mimic.features` | 2 |
| 177 | `mimic.labeling` | `.labeling` | 3 |
| 178 | `mimic.labeling` | `aura.mimic.labeling` | 2 |
| 179 | `mimic.labeling_v2` | `aura.mimic.labeling_v2` | 1 |
| 180 | `mimic.loaders` | `aura.mimic.loaders` | 5 |
| 181 | `mimic.loaders` | `.loaders` | 4 |
| 182 | `mimic.parsing` | `aura.mimic.parsing` | 4 |
| 183 | `mimic.parsing` | `.parsing` | 2 |
| 184 | `mimic.patient` | `aura.mimic.patient` | 4 |
| 185 | `mimic.patient` | `.patient` | 3 |
| 186 | `mimic.performance` | `aura.mimic.performance` | 1 |
| 187 | `mimic.seed` | `aura.mimic.seed` | 3 |
| 188 | `mimic.splits` | `.splits` | 1 |
| 189 | `mimic.splits` | `aura.mimic.splits` | 1 |
| 190 | `mimic.tasks` | `aura.mimic.tasks` | 4 |
| 191 | `mimic.tasks` | `.tasks` | 1 |
| 192 | `mimic.timeline` | `.timeline` | 1 |
| 193 | `mimic.timeline` | `aura.mimic.timeline` | 1 |
| 194 | `mimic.training` | `aura.mimic.training` | 2 |
| 195 | `mimic.training` | `.training` | 1 |
| 196 | `mimic.uncertainty` | `aura.mimic.uncertainty` | 1 |
| 197 | `ml.data` | `aura.ml.data` | 5 |
| 198 | `ml.data` | `..data` | 5 |
| 199 | `ml.evaluation` | `aura.ml.evaluation` | 5 |
| 200 | `ml.evaluation.clinical_eval` | `aura.ml.evaluation.clinical_eval` | 2 |
| 201 | `ml.evaluation.clinical_eval` | `.clinical_eval` | 1 |
| 202 | `ml.evaluation.metrics` | `aura.ml.evaluation.metrics` | 2 |
| 203 | `ml.evaluation.metrics` | `.metrics` | 1 |
| 204 | `ml.evaluation.perf_benchmark` | `.perf_benchmark` | 1 |
| 205 | `ml.evaluation.perf_benchmark` | `aura.ml.evaluation.perf_benchmark` | 1 |
| 206 | `ml.evaluation.vision_calibration` | `aura.ml.evaluation.vision_calibration` | 2 |
| 207 | `ml.training` | `aura.ml.training` | 1 |
| 208 | `ml.training.cxr_dataset` | `.cxr_dataset` | 1 |
| 209 | `ml.training.dataset` | `aura.ml.training.dataset` | 3 |
| 210 | `ml.training.dataset` | `..training.dataset` | 3 |
| 211 | `ml.training.dataset` | `.dataset` | 2 |
| 212 | `ml.training.train_cnn` | `aura.ml.training.train_cnn` | 1 |
| 213 | `ml.training.train_fusion` | `..training.train_fusion` | 1 |
| 214 | `ml.vision_cxr.checkpoint` | `.checkpoint` | 1 |
| 215 | `ml.vision_cxr.config` | `..vision_cxr.config` | 1 |
| 216 | `ml.vision_cxr.config` | `.config` | 1 |
| 217 | `ml.vision_cxr.dataset` | `..vision_cxr.dataset` | 6 |
| 218 | `ml.vision_cxr.dataset` | `.dataset` | 1 |
| 219 | `ml.vision_cxr.inference` | `..vision_cxr.inference` | 4 |
| 220 | `ml.vision_cxr.inference` | `aura.ml.vision_cxr.inference` | 2 |
| 221 | `ml.vision_cxr.losses` | `.losses` | 2 |
| 222 | `ml.vision_cxr.metrics` | `.metrics` | 1 |
| 223 | `ml.vision_cxr.model` | `.model` | 2 |
| 224 | `ml.vision_cxr.model` | `..vision_cxr.model` | 1 |
| 225 | `ml.vision_cxr.model` | `aura.ml.vision_cxr.model` | 1 |
| 226 | `ml.vision_cxr.utils` | `.utils` | 1 |
| 227 | `ml.vision_cxr.validate` | `.validate` | 1 |
| 228 | `schemas.clinical` | `aura.schemas.clinical` | 84 |
| 229 | `schemas.clinical` | `.clinical` | 2 |
| 230 | `schemas.contracts` | `aura.schemas.contracts` | 38 |
| 231 | `schemas.contracts` | `.contracts` | 1 |
| 232 | `services.agent.active_diagnosis` | `aura.services.agent.active_diagnosis` | 4 |
| 233 | `services.agent.active_diagnosis` | `.active_diagnosis` | 1 |
| 234 | `services.agent.discussion` | `aura.services.agent.discussion` | 1 |
| 235 | `services.enterprise.dicom_listener` | `aura.services.enterprise.dicom_listener` | 1 |
| 236 | `services.enterprise.fhir` | `aura.services.enterprise.fhir` | 1 |
| 237 | `services.enterprise.hl7` | `aura.services.enterprise.hl7` | 1 |
| 238 | `services.explain` | `aura.services.explain` | 5 |
| 239 | `services.explain` | `..explain` | 2 |
| 240 | `services.explain` | `.` | 1 |
| 241 | `services.explain.engine` | `.engine` | 1 |
| 242 | `services.explain.methods` | `.methods` | 1 |
| 243 | `services.explain.scorecam` | `.scorecam` | 1 |
| 244 | `services.fusion` | `aura.services.fusion` | 6 |
| 245 | `services.fusion.classical` | `aura.services.fusion.classical` | 4 |
| 246 | `services.fusion.classical` | `.classical` | 1 |
| 247 | `services.fusion.conflict` | `.conflict` | 1 |
| 248 | `services.fusion.device` | `aura.services.fusion.device` | 3 |
| 249 | `services.fusion.device` | `.device` | 2 |
| 250 | `services.fusion.engine` | `aura.services.fusion.engine` | 2 |
| 251 | `services.fusion.engine` | `.engine` | 1 |
| 252 | `services.fusion.ensemble` | `aura.services.fusion.ensemble` | 1 |
| 253 | `services.fusion.ensemble` | `..fusion.ensemble` | 1 |
| 254 | `services.fusion.evidence` | `aura.services.fusion.evidence` | 7 |
| 255 | `services.fusion.evidence` | `..fusion.evidence` | 5 |
| 256 | `services.fusion.evidence` | `.evidence` | 3 |
| 257 | `services.fusion.learnable` | `aura.services.fusion.learnable` | 4 |
| 258 | `services.fusion.learnable` | `.learnable` | 1 |
| 259 | `services.fusion.multimodal` | `aura.services.fusion.multimodal` | 1 |
| 260 | `services.fusion.qae` | `.qae` | 1 |
| 261 | `services.fusion.qae` | `aura.services.fusion.qae` | 1 |
| 262 | `services.fusion.qmba` | `aura.services.fusion.qmba` | 3 |
| 263 | `services.fusion.qmeasure` | `aura.services.fusion.qmeasure` | 2 |
| 264 | `services.fusion.qmeasure` | `.qmeasure` | 1 |
| 265 | `services.fusion.quantum` | `aura.services.fusion.quantum` | 8 |
| 266 | `services.fusion.quantum` | `.quantum` | 1 |
| 267 | `services.inference.audit_log` | `aura.services.inference.audit_log` | 2 |
| 268 | `services.inference.audit_log` | `.audit_log` | 1 |
| 269 | `services.inference.predict` | `aura.services.inference.predict` | 3 |
| 270 | `services.inference.predict` | `.predict` | 1 |
| 271 | `services.memory` | `aura.services.memory` | 1 |
| 272 | `services.memory.engine` | `.engine` | 1 |
| 273 | `services.models` | `aura.services.models` | 1 |
| 274 | `services.models.registry` | `.registry` | 1 |
| 275 | `services.reasoning` | `aura.services.reasoning` | 1 |
| 276 | `services.reasoning.engine` | `.engine` | 1 |
| 277 | `services.reasoning.qbn` | `.qbn` | 1 |
| 278 | `services.reasoning.qbn` | `aura.services.reasoning.qbn` | 1 |
| 279 | `services.recommend` | `aura.services.recommend` | 2 |
| 280 | `services.recommend.causal` | `.causal` | 1 |
| 281 | `services.recommend.engine` | `..recommend.engine` | 2 |
| 282 | `services.recommend.engine` | `aura.services.recommend.engine` | 1 |
| 283 | `services.recommend.engine` | `.engine` | 1 |
| 284 | `services.report` | `aura.services.report` | 3 |
| 285 | `services.report.clinical_report` | `aura.services.report.clinical_report` | 2 |
| 286 | `services.report.clinical_report` | `..report.clinical_report` | 1 |
| 287 | `services.report.engine` | `.engine` | 1 |
| 288 | `services.safety` | `aura.services.safety` | 8 |
| 289 | `services.safety.aci` | `aura.services.safety.aci` | 1 |
| 290 | `services.safety.calibration` | `aura.services.safety.calibration` | 9 |
| 291 | `services.safety.calibration` | `.calibration` | 1 |
| 292 | `services.safety.controller` | `aura.services.safety.controller` | 2 |
| 293 | `services.safety.controller` | `.controller` | 1 |
| 294 | `services.safety.engine` | `.engine` | 1 |
| 295 | `services.safety.readiness` | `.readiness` | 1 |
| 296 | `services.safety.readiness` | `aura.services.safety.readiness` | 1 |
| 297 | `services.safety.uncertainty` | `aura.services.safety.uncertainty` | 9 |
| 298 | `services.safety.uncertainty` | `.uncertainty` | 1 |
| 299 | `services.vision` | `aura.services.vision` | 3 |
| 300 | `services.vision.cnn` | `aura.services.vision.cnn` | 1 |
| 301 | `services.vision.cnn` | `.cnn` | 1 |
| 302 | `services.vision.engine` | `aura.services.vision.engine` | 3 |
| 303 | `services.vision.engine` | `.engine` | 1 |
| 304 | `services.vision.features` | `aura.services.vision.features` | 2 |
| 305 | `services.vision.features` | `.features` | 1 |
| 306 | `services.vision.io` | `aura.services.vision.io` | 7 |
| 307 | `services.vision.io` | `.io` | 2 |
| 308 | `services.vision.io` | `..vision.io` | 1 |
| 309 | `services.vision.xray_gate` | `aura.services.vision.xray_gate` | 7 |
| 310 | `services.vision.xray_gate` | `..vision.xray_gate` | 1 |
| 311 | `tests.test_mri_foundation` | `.test_mri_foundation` | 1 |

## Appendix B — every repaired import, by file (209 files, 1021 statements)

<details><summary><code>audit_all.py</code> — 10 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from common.config import ARTIFACTS, get_settings
+ from aura.common.config import ARTIFACTS, get_settings
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
- from services.fusion.classical import ClassicalFusion
+ from aura.services.fusion.classical import ClassicalFusion
- from services.safety.calibration import (
+ from aura.services.safety.calibration import (
- from ml.training.dataset import build_evidence_dataset, make_splits
+ from aura.ml.training.dataset import build_evidence_dataset, make_splits
- from ml.evaluation.metrics import evaluate
+ from aura.ml.evaluation.metrics import evaluate
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.safety.calibration import fit_conformal
+ from aura.services.safety.calibration import fit_conformal
```
</details>

<details><summary><code>aura/aura_cli.py</code> — 16 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from ml.training import train_vision, train_fusion
+ from aura.ml.training import train_vision, train_fusion
- from ml.training.train_cnn import run, TrainConfig
+ from aura.ml.training.train_cnn import run, TrainConfig
- from ml.evaluation import benchmark
+ from aura.ml.evaluation import benchmark
- from services.inference.predict import predict_image
+ from aura.services.inference.predict import predict_image
- from ml.evaluation import clinical_eval
+ from aura.ml.evaluation import clinical_eval
- from ml.evaluation import vision_calibration
+ from aura.ml.evaluation import vision_calibration
- from services.inference.predict import predict_image
+ from aura.services.inference.predict import predict_image
- from ml.evaluation import perf_benchmark
+ from aura.ml.evaluation import perf_benchmark
- from ml.evaluation import vision_calibration
+ from aura.ml.evaluation import vision_calibration
- from services.vision.engine import VisionEngine
+ from aura.services.vision.engine import VisionEngine
- from services.fusion.engine import FusionEngine
+ from aura.services.fusion.engine import FusionEngine
- from services.fusion.evidence import encode
+ from aura.services.fusion.evidence import encode
- from services.agent.active_diagnosis import ActiveDiagnosisAgent
+ from aura.services.agent.active_diagnosis import ActiveDiagnosisAgent
- from schemas.contracts import StructuredPriors
+ from aura.schemas.contracts import StructuredPriors
- from services.report.clinical_report import render_text
+ from aura.services.report.clinical_report import render_text
```
</details>

<details><summary><code>aura/backend/api/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.api.routes import build_router
+ from .routes import build_router
```
</details>

<details><summary><code>aura/backend/api/routes.py</code> — 16 statement(s)</summary>

```diff
- from backend.core.shared.errors import AuraBackendError
+ from ..core.shared.errors import AuraBackendError
- from backend.core.shared.logging import get_logger, new_correlation_id, use_correlation_id
+ from ..core.shared.logging import get_logger, new_correlation_id, use_correlation_id
- from backend.models.routing import AnalysisEnvelope, RoutingMetadata
+ from ..models.routing import AnalysisEnvelope, RoutingMetadata
- from backend.services.dispatch import DispatchService
+ from ..services.dispatch import DispatchService
- from backend.core.upload import UploadIntake
+ from ..core.upload import UploadIntake
- from backend.core.shared.errors import UploadRejected
+ from ..core.shared.errors import UploadRejected
- from backend.core.upload import UploadIntake
+ from ..core.upload import UploadIntake
- from backend.core.shared.errors import UploadRejected
+ from ..core.shared.errors import UploadRejected
- from backend.core.upload import UploadIntake
+ from ..core.upload import UploadIntake
- from backend.foundation.mri.intake_manager import MRIIntakeManager
+ from ..foundation.mri.intake_manager import MRIIntakeManager
- from backend.core.shared.errors import UploadRejected
+ from ..core.shared.errors import UploadRejected
- from backend.foundation.mri.errors import StudyValidationError
+ from ..foundation.mri.errors import StudyValidationError
- from gateway.app import store
+ from aura.gateway.app import store
- from backend.services.reasoning.progression import LongitudinalAnalyzer
+ from ..services.reasoning.progression import LongitudinalAnalyzer
- from gateway.app import store
+ from aura.gateway.app import store
- from backend.services.reasoning.tracking import TumorTracker
+ from ..services.reasoning.tracking import TumorTracker
```
</details>

<details><summary><code>aura/backend/bootstrap.py</code> — 6 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from .core.shared.logging import get_logger
- from backend.engines.base.registry import EngineRegistry, default_registry
+ from .engines.base.registry import EngineRegistry, default_registry
- from backend.services.dispatch import DispatchService
+ from .services.dispatch import DispatchService
- from backend.engines.neuro.engine import register_neuromind_engine
+ from .engines.neuro.engine import register_neuromind_engine
- from backend.engines.thorax.engine import register_thorax_engine
+ from .engines.thorax.engine import register_thorax_engine
- from backend.api.routes import build_router
+ from .api.routes import build_router
```
</details>

<details><summary><code>aura/backend/core/router/__init__.py</code> — 4 statement(s)</summary>

```diff
- from backend.core.router.detector import (
+ from .detector import (
- from backend.core.router.features import ImageFingerprint, fingerprint
+ from .features import ImageFingerprint, fingerprint
- from backend.core.router.router import ModalityRouter
+ from .router import ModalityRouter
- from backend.core.router.signatures import (
+ from .signatures import (
```
</details>

<details><summary><code>aura/backend/core/router/detector.py</code> — 5 statement(s)</summary>

```diff
- from backend.core.router.features import ImageFingerprint, fingerprint
+ from .features import ImageFingerprint, fingerprint
- from backend.core.router.signatures import (
+ from .signatures import (
- from backend.core.shared.logging import get_logger
+ from ..shared.logging import get_logger
- from backend.core.shared.types import ImageAsset, ImagingModality
+ from ..shared.types import ImageAsset, ImagingModality
- from backend.core.shared.types import MODALITY_LABELS
+ from ..shared.types import MODALITY_LABELS
```
</details>

<details><summary><code>aura/backend/core/router/features.py</code> — 1 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from ..shared.logging import get_logger
```
</details>

<details><summary><code>aura/backend/core/router/router.py</code> — 8 statement(s)</summary>

```diff
- from backend.core.router.detector import (
+ from .detector import (
- from backend.core.shared.errors import ModalityUndetermined
+ from ..shared.errors import ModalityUndetermined
- from backend.core.shared.logging import correlation_id, get_logger
+ from ..shared.logging import correlation_id, get_logger
- from backend.core.shared.types import MODALITY_LABELS, ImageAsset, ImagingModality
+ from ..shared.types import MODALITY_LABELS, ImageAsset, ImagingModality
- from backend.engines.base.registry import EngineRegistry, default_registry
+ from aura.backend.engines.base.registry import EngineRegistry, default_registry
- from backend.models.routing import ModalityCandidate, RoutingMetadata
+ from aura.backend.models.routing import ModalityCandidate, RoutingMetadata
- from backend.core.shared.errors import UnsupportedModality
+ from ..shared.errors import UnsupportedModality
- from backend.core.shared.types import ImagingModality
+ from ..shared.types import ImagingModality
```
</details>

<details><summary><code>aura/backend/core/router/signatures.py</code> — 5 statement(s)</summary>

```diff
- from backend.core.router.features import ImageFingerprint
+ from .features import ImageFingerprint
- from backend.core.shared.logging import get_logger
+ from ..shared.logging import get_logger
- from backend.core.shared.types import MODALITY_LABELS, ImagingModality
+ from ..shared.types import MODALITY_LABELS, ImagingModality
- from services.vision.xray_gate import validate_cxr
+ from aura.services.vision.xray_gate import validate_cxr
- from services.vision.xray_gate import HEAD_COMMIT_SCORE, head_geometry_from_path
+ from aura.services.vision.xray_gate import HEAD_COMMIT_SCORE, head_geometry_from_path
```
</details>

<details><summary><code>aura/backend/core/shared/__init__.py</code> — 3 statement(s)</summary>

```diff
- from backend.core.shared.errors import (
+ from .errors import (
- from backend.core.shared.logging import (
+ from .logging import (
- from backend.core.shared.types import (
+ from .types import (
```
</details>

<details><summary><code>aura/backend/core/shared/types.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import Modality
+ from aura.schemas.clinical import Modality
```
</details>

<details><summary><code>aura/backend/core/upload/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.core.upload.intake import UploadIntake, stage_bytes
+ from .intake import UploadIntake, stage_bytes
```
</details>

<details><summary><code>aura/backend/core/upload/intake.py</code> — 6 statement(s)</summary>

```diff
- from backend.core.shared.errors import UploadRejected
+ from ..shared.errors import UploadRejected
- from backend.core.shared.logging import get_logger
+ from ..shared.logging import get_logger
- from backend.core.shared.types import ImageAsset
+ from ..shared.types import ImageAsset
- from common.config import get_settings
+ from aura.common.config import get_settings
- from gateway.security import read_capped, validate_upload_name, validate_mri_content
+ from aura.gateway.security import read_capped, validate_upload_name, validate_mri_content
- from gateway.security import read_capped, validate_upload_name, validate_mri_content
+ from aura.gateway.security import read_capped, validate_upload_name, validate_mri_content
```
</details>

<details><summary><code>aura/backend/engines/base/__init__.py</code> — 2 statement(s)</summary>

```diff
- from backend.engines.base.contract import (
+ from .contract import (
- from backend.engines.base.registry import (
+ from .registry import (
```
</details>

<details><summary><code>aura/backend/engines/base/contract.py</code> — 4 statement(s)</summary>

```diff
- from backend.core.shared.errors import (
+ from aura.backend.core.shared.errors import (
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
+ from aura.backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
- from backend.models.routing import EngineOutcome, ResultStatus
+ from aura.backend.models.routing import EngineOutcome, ResultStatus
```
</details>

<details><summary><code>aura/backend/engines/base/registry.py</code> — 4 statement(s)</summary>

```diff
- from backend.core.shared.errors import EngineNotAvailable
+ from aura.backend.core.shared.errors import EngineNotAvailable
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.core.shared.types import EngineStatus, ImagingModality
+ from aura.backend.core.shared.types import EngineStatus, ImagingModality
- from backend.engines.base.contract import AnalysisEngine, EngineDescriptor
+ from .contract import AnalysisEngine, EngineDescriptor
```
</details>

<details><summary><code>aura/backend/engines/neuro/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.engines.neuro.engine import NeuroMindEngine
+ from .engine import NeuroMindEngine
```
</details>

<details><summary><code>aura/backend/engines/neuro/bundle.py</code> — 5 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.output import BrainVisionOutput
+ from aura.backend.vision.brain.output import BrainVisionOutput
- from backend.vision.brain.types import CompositeRegion, TumorRegion
+ from aura.backend.vision.brain.types import CompositeRegion, TumorRegion
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
- from schemas.contracts import (
+ from aura.schemas.contracts import (
```
</details>

<details><summary><code>aura/backend/engines/neuro/calibration.py</code> — 2 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/backend/engines/neuro/engine.py</code> — 30 statement(s)</summary>

```diff
- from backend.core.shared.errors import EngineExecutionError, UnreadableImage
+ from aura.backend.core.shared.errors import EngineExecutionError, UnreadableImage
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
+ from aura.backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
- from backend.engines.base.contract import (
+ from ..base.contract import (
- from backend.engines.base.registry import EngineRegistry, default_registry
+ from ..base.registry import EngineRegistry, default_registry
- from backend.engines.neuro.multisequence import (
+ from .multisequence import (
- from schemas.contracts import StructuredPriors
+ from aura.schemas.contracts import StructuredPriors
- from backend.engines.neuro.qkl import QKLClassifier
+ from .qkl import QKLClassifier
- from backend.foundation.mri import FoundationConfig, MRIFoundationPipeline
+ from aura.backend.foundation.mri import FoundationConfig, MRIFoundationPipeline
- from backend.vision.brain.inference import BrainVisionEngine
+ from aura.backend.vision.brain.inference import BrainVisionEngine
- from backend.engines.neuro.calibration import load_calibrator
+ from .calibration import load_calibrator
- from backend.foundation.mri.io.dicom_reader import DicomSeriesReader
+ from aura.backend.foundation.mri.io.dicom_reader import DicomSeriesReader
- from backend.foundation.mri.io.nifti_reader import NiftiReader
+ from aura.backend.foundation.mri.io.nifti_reader import NiftiReader
- from backend.foundation.mri.io.nrrd_reader import NrrdReader
+ from aura.backend.foundation.mri.io.nrrd_reader import NrrdReader
- from backend.core.router.router import ModalityRouter
+ from aura.backend.core.router.router import ModalityRouter
- from backend.core.shared.types import ImagingModality
+ from aura.backend.core.shared.types import ImagingModality
- from backend.vision.brain.types import DEFAULT_MODALITIES
+ from aura.backend.vision.brain.types import DEFAULT_MODALITIES
- from backend.foundation.mri.intake_manager import MRIIntakeManager
+ from aura.backend.foundation.mri.intake_manager import MRIIntakeManager
- from backend.foundation.mri.errors import StudyValidationError
+ from aura.backend.foundation.mri.errors import StudyValidationError
- from backend.engines.neuro.multisequence import looks_multisequence
+ from .multisequence import looks_multisequence
- from backend.foundation.mri.errors import MRIFoundationError
+ from aura.backend.foundation.mri.errors import MRIFoundationError
- from backend.core.shared.errors import UnreadableImage
+ from aura.backend.core.shared.errors import UnreadableImage
- from backend.engines.neuro.bundle import build_case_bundle
+ from .bundle import build_case_bundle
- from backend.foundation.mri.errors import StudyValidationError
+ from aura.backend.foundation.mri.errors import StudyValidationError
- from backend.foundation.mri.types import SequenceType
+ from aura.backend.foundation.mri.types import SequenceType
- from common.config import get_settings
+ from aura.common.config import get_settings
- from backend.engines.neuro.neuroview import build_neuroview_payload
+ from .neuroview import build_neuroview_payload
- from schemas.contracts import AbstentionReason
+ from aura.schemas.contracts import AbstentionReason
- from common.config import get_settings
+ from aura.common.config import get_settings
- from backend.engines.neuro.bundle import _representative_index
+ from .bundle import _representative_index
```
</details>

<details><summary><code>aura/backend/engines/neuro/multisequence.py</code> — 3 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.engines.neuro.sequence_features import FEATURE_DIM, sequence_features
+ from .sequence_features import FEATURE_DIM, sequence_features
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/backend/engines/neuro/neuroinsight.py</code> — 1 statement(s)</summary>

```diff
- from backend.foundation.mri.geometry import voxel_to_world
+ from aura.backend.foundation.mri.geometry import voxel_to_world
```
</details>

<details><summary><code>aura/backend/engines/neuro/neuroview.py</code> — 3 statement(s)</summary>

```diff
- from backend.engines.neuro.multisequence import MultiSequenceStudy
+ from .multisequence import MultiSequenceStudy
- from backend.vision.brain.types import ModalitySpec, TumorRegion
+ from aura.backend.vision.brain.types import ModalitySpec, TumorRegion
- from backend.engines.neuro.neuroinsight import compute_neuroinsight
+ from .neuroinsight import compute_neuroinsight
```
</details>

<details><summary><code>aura/backend/engines/neuro/qkl.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
```
</details>

<details><summary><code>aura/backend/engines/thorax/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.engines.thorax.engine import ThoraxEngine, register_thorax_engine
+ from .engine import ThoraxEngine, register_thorax_engine
```
</details>

<details><summary><code>aura/backend/engines/thorax/engine.py</code> — 9 statement(s)</summary>

```diff
- from backend.core.shared.errors import EngineExecutionError, UnreadableImage
+ from aura.backend.core.shared.errors import EngineExecutionError, UnreadableImage
- from backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
+ from aura.backend.core.shared.types import EngineStatus, ImageAsset, ImagingModality
- from backend.engines.base.contract import (
+ from ..base.contract import (
- from backend.engines.base.registry import EngineRegistry, default_registry
+ from ..base.registry import EngineRegistry, default_registry
- from backend.core.router.router import ModalityRouter
+ from aura.backend.core.router.router import ModalityRouter
- from backend.core.shared.types import ImagingModality
+ from aura.backend.core.shared.types import ImagingModality
- from services.vision.xray_gate import validate_cxr
+ from aura.services.vision.xray_gate import validate_cxr
- from services.vision.io import study_from_cxr
+ from aura.services.vision.io import study_from_cxr
- from services.inference.audit_log import log_inference
+ from aura.services.inference.audit_log import log_inference
```
</details>

<details><summary><code>aura/backend/foundation/mri/__init__.py</code> — 14 statement(s)</summary>

```diff
- from backend.foundation.mri.config import (
+ from .config import (
- from backend.foundation.mri.errors import (
+ from .errors import (
- from backend.foundation.mri.geometry import VoxelGeometry
+ from .geometry import VoxelGeometry
- from backend.foundation.mri.loader import LoadedStudy, MRIStudyLoader, load_study
+ from .loader import LoadedStudy, MRIStudyLoader, load_study
- from backend.foundation.mri.masking import BrainMaskSlot
+ from .masking import BrainMaskSlot
- from backend.foundation.mri.metadata import MetadataExtractor, MRIMetadata
+ from .metadata import MetadataExtractor, MRIMetadata
- from backend.foundation.mri.pipeline import (
+ from .pipeline import (
- from backend.foundation.mri.quality import (
+ from .quality import (
- from backend.foundation.mri.registration import RegistrationPlan, RegistrationPreparer
+ from .registration import RegistrationPlan, RegistrationPreparer
- from backend.foundation.mri.sequence import (
+ from .sequence import (
- from backend.foundation.mri.standardize import (
+ from .standardize import (
- from backend.foundation.mri.study import (
+ from .study import (
- from backend.foundation.mri.types import (
+ from .types import (
- from backend.foundation.mri.volume import MRIVolume, VolumeBuilder
+ from .volume import MRIVolume, VolumeBuilder
```
</details>

<details><summary><code>aura/backend/foundation/mri/config.py</code> — 1 statement(s)</summary>

```diff
- from backend.foundation.mri.types import NormalizationMethod
+ from .types import NormalizationMethod
```
</details>

<details><summary><code>aura/backend/foundation/mri/errors.py</code> — 1 statement(s)</summary>

```diff
- from backend.core.shared.errors import AuraBackendError
+ from aura.backend.core.shared.errors import AuraBackendError
```
</details>

<details><summary><code>aura/backend/foundation/mri/geometry.py</code> — 1 statement(s)</summary>

```diff
- from backend.foundation.mri.types import AnatomicalPlane
+ from .types import AnatomicalPlane
```
</details>

<details><summary><code>aura/backend/foundation/mri/intake_manager.py</code> — 7 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.errors import StudyValidationError
+ from .errors import StudyValidationError
- from backend.foundation.mri.io.nifti_reader import NiftiReader
+ from .io.nifti_reader import NiftiReader
- from backend.foundation.mri.metadata import MetadataExtractor
+ from .metadata import MetadataExtractor
- from backend.foundation.mri.sequence import RuleBasedSequenceDetector
+ from .sequence import RuleBasedSequenceDetector
- from backend.foundation.mri.types import FileFormat, SequenceType
+ from .types import FileFormat, SequenceType
- from backend.engines.neuro.multisequence import MultiSequenceStudy, looks_multisequence, load_multisequence
+ from aura.backend.engines.neuro.multisequence import MultiSequenceStudy, looks_multisequence, load_multisequence
```
</details>

<details><summary><code>aura/backend/foundation/mri/io/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.foundation.mri.io.base import (
+ from .base import (
```
</details>

<details><summary><code>aura/backend/foundation/mri/io/base.py</code> — 2 statement(s)</summary>

```diff
- from backend.foundation.mri.geometry import VoxelGeometry
+ from ..geometry import VoxelGeometry
- from backend.foundation.mri.types import FileFormat
+ from ..types import FileFormat
```
</details>

<details><summary><code>aura/backend/foundation/mri/io/dicom_reader.py</code> — 8 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.config import LoaderConfig
+ from ..config import LoaderConfig
- from backend.foundation.mri.errors import CorruptStudy, StudyValidationError
+ from ..errors import CorruptStudy, StudyValidationError
- from backend.foundation.mri.geometry import VoxelGeometry, affine_from_dicom
+ from ..geometry import VoxelGeometry, affine_from_dicom
- from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity, SliceIssue
+ from .base import RawSeries, SeriesIntegrity, SliceIssue
- from backend.foundation.mri.metadata import dicom_header_subset
+ from ..metadata import dicom_header_subset
- from backend.foundation.mri.types import FileFormat
+ from ..types import FileFormat
- from backend.foundation.mri.errors import StudyValidationError
+ from ..errors import StudyValidationError
```
</details>

<details><summary><code>aura/backend/foundation/mri/io/discovery.py</code> — 5 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.config import LoaderConfig
+ from ..config import LoaderConfig
- from backend.foundation.mri.errors import StudyNotFound
+ from ..errors import StudyNotFound
- from backend.foundation.mri.io.base import StudyReader
+ from .base import StudyReader
- from backend.foundation.mri.types import FileFormat
+ from ..types import FileFormat
```
</details>

<details><summary><code>aura/backend/foundation/mri/io/nifti_reader.py</code> — 5 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.errors import CorruptStudy, UnsupportedStudyFormat
+ from ..errors import CorruptStudy, UnsupportedStudyFormat
- from backend.foundation.mri.geometry import VoxelGeometry, affine_from_quaternion
+ from ..geometry import VoxelGeometry, affine_from_quaternion
- from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
+ from .base import RawSeries, SeriesIntegrity
- from backend.foundation.mri.types import FileFormat
+ from ..types import FileFormat
```
</details>

<details><summary><code>aura/backend/foundation/mri/io/nrrd_reader.py</code> — 5 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.errors import CorruptStudy, UnsupportedStudyFormat
+ from ..errors import CorruptStudy, UnsupportedStudyFormat
- from backend.foundation.mri.geometry import LPS_TO_RAS, VoxelGeometry
+ from ..geometry import LPS_TO_RAS, VoxelGeometry
- from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
+ from .base import RawSeries, SeriesIntegrity
- from backend.foundation.mri.types import FileFormat
+ from ..types import FileFormat
```
</details>

<details><summary><code>aura/backend/foundation/mri/loader.py</code> — 9 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.config import LoaderConfig
+ from .config import LoaderConfig
- from backend.foundation.mri.errors import (
+ from .errors import (
- from backend.foundation.mri.io.base import RawSeries, StudyReader
+ from .io.base import RawSeries, StudyReader
- from backend.foundation.mri.io.dicom_reader import DicomSeriesReader
+ from .io.dicom_reader import DicomSeriesReader
- from backend.foundation.mri.io.discovery import Discovery, discover
+ from .io.discovery import Discovery, discover
- from backend.foundation.mri.io.nifti_reader import NiftiReader
+ from .io.nifti_reader import NiftiReader
- from backend.foundation.mri.io.nrrd_reader import NrrdReader
+ from .io.nrrd_reader import NrrdReader
- from backend.foundation.mri.types import FileFormat
+ from .types import FileFormat
```
</details>

<details><summary><code>aura/backend/foundation/mri/masking.py</code> — 2 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.types import MaskProvenance
+ from .types import MaskProvenance
```
</details>

<details><summary><code>aura/backend/foundation/mri/metadata.py</code> — 2 statement(s)</summary>

```diff
- from backend.foundation.mri.geometry import VoxelGeometry
+ from .geometry import VoxelGeometry
- from backend.foundation.mri.types import (
+ from .types import (
```
</details>

<details><summary><code>aura/backend/foundation/mri/pipeline.py</code> — 13 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.config import FoundationConfig
+ from .config import FoundationConfig
- from backend.foundation.mri.errors import (
+ from .errors import (
- from backend.foundation.mri.io.base import RawSeries
+ from .io.base import RawSeries
- from backend.foundation.mri.loader import LoadedStudy, MRIStudyLoader
+ from .loader import LoadedStudy, MRIStudyLoader
- from backend.foundation.mri.metadata import MetadataExtractor, MRIMetadata
+ from .metadata import MetadataExtractor, MRIMetadata
- from backend.foundation.mri.quality import MRIQualityInspector, QualityReport
+ from .quality import MRIQualityInspector, QualityReport
- from backend.foundation.mri.registration import RegistrationPreparer
+ from .registration import RegistrationPreparer
- from backend.foundation.mri.sequence import RuleBasedSequenceDetector, SequenceDetector
+ from .sequence import RuleBasedSequenceDetector, SequenceDetector
- from backend.foundation.mri.standardize import (
+ from .standardize import (
- from backend.foundation.mri.study import (
+ from .study import (
- from backend.foundation.mri.types import FileFormat, QualityVerdict, StepStatus
+ from .types import FileFormat, QualityVerdict, StepStatus
- from backend.foundation.mri.volume import MRIVolume, VolumeBuilder
+ from .volume import MRIVolume, VolumeBuilder
```
</details>

<details><summary><code>aura/backend/foundation/mri/quality.py</code> — 7 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.config import QualityThresholds
+ from .config import QualityThresholds
- from backend.foundation.mri.io.base import SeriesIntegrity
+ from .io.base import SeriesIntegrity
- from backend.foundation.mri.masking import estimate_foreground_mask
+ from .masking import estimate_foreground_mask
- from backend.foundation.mri.metadata import MRIMetadata
+ from .metadata import MRIMetadata
- from backend.foundation.mri.types import CheckStatus, QualityVerdict
+ from .types import CheckStatus, QualityVerdict
- from backend.foundation.mri.volume import MRIVolume
+ from .volume import MRIVolume
```
</details>

<details><summary><code>aura/backend/foundation/mri/registration.py</code> — 2 statement(s)</summary>

```diff
- from backend.foundation.mri.masking import BrainMaskSlot
+ from .masking import BrainMaskSlot
- from backend.foundation.mri.volume import MRIVolume
+ from .volume import MRIVolume
```
</details>

<details><summary><code>aura/backend/foundation/mri/sequence.py</code> — 3 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.metadata import MRIMetadata, sequence_evidence_text
+ from .metadata import MRIMetadata, sequence_evidence_text
- from backend.foundation.mri.types import SEQUENCE_LABELS, SequenceType
+ from .types import SEQUENCE_LABELS, SequenceType
```
</details>

<details><summary><code>aura/backend/foundation/mri/standardize.py</code> — 10 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.config import StandardizationConfig
+ from .config import StandardizationConfig
- from backend.foundation.mri.errors import StageFailed, StageUnavailable
+ from .errors import StageFailed, StageUnavailable
- from backend.foundation.mri.geometry import to_canonical
+ from .geometry import to_canonical
- from backend.foundation.mri.masking import BrainMaskSlot, estimate_foreground_mask
+ from .masking import BrainMaskSlot, estimate_foreground_mask
- from backend.foundation.mri.metadata import MRIMetadata
+ from .metadata import MRIMetadata
- from backend.foundation.mri.types import MaskProvenance, NormalizationMethod
+ from .types import MaskProvenance, NormalizationMethod
- from backend.foundation.mri.volume import MRIVolume
+ from .volume import MRIVolume
- from backend.foundation.mri.masking import otsu_threshold, BrainMaskSlot
+ from .masking import otsu_threshold, BrainMaskSlot
- from backend.foundation.mri.types import MaskProvenance
+ from .types import MaskProvenance
```
</details>

<details><summary><code>aura/backend/foundation/mri/study.py</code> — 7 statement(s)</summary>

```diff
- from backend.foundation.mri.masking import BrainMaskSlot
+ from .masking import BrainMaskSlot
- from backend.foundation.mri.metadata import MRIMetadata
+ from .metadata import MRIMetadata
- from backend.foundation.mri.quality import QualityReport
+ from .quality import QualityReport
- from backend.foundation.mri.registration import RegistrationPlan
+ from .registration import RegistrationPlan
- from backend.foundation.mri.sequence import SequenceAssignment
+ from .sequence import SequenceAssignment
- from backend.foundation.mri.types import (
+ from .types import (
- from backend.foundation.mri.volume import MRIVolume
+ from .volume import MRIVolume
```
</details>

<details><summary><code>aura/backend/foundation/mri/volume.py</code> — 4 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.errors import StudyValidationError
+ from .errors import StudyValidationError
- from backend.foundation.mri.geometry import VoxelGeometry
+ from .geometry import VoxelGeometry
- from backend.foundation.mri.io.base import RawSeries
+ from .io.base import RawSeries
```
</details>

<details><summary><code>aura/backend/models/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.models.routing import (
+ from .routing import (
```
</details>

<details><summary><code>aura/backend/services/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.services.dispatch import DispatchService
+ from .dispatch import DispatchService
```
</details>

<details><summary><code>aura/backend/services/dispatch.py</code> — 10 statement(s)</summary>

```diff
- from backend.core.router.router import ModalityRouter
+ from ..core.router.router import ModalityRouter
- from backend.core.shared.errors import AuraBackendError, EngineNotAvailable
+ from ..core.shared.errors import AuraBackendError, EngineNotAvailable
- from backend.core.shared.logging import get_logger, use_correlation_id
+ from ..core.shared.logging import get_logger, use_correlation_id
- from backend.core.shared.types import ImageAsset
+ from ..core.shared.types import ImageAsset
- from backend.engines.base.registry import EngineRegistry, default_registry
+ from ..engines.base.registry import EngineRegistry, default_registry
- from backend.models.routing import (
+ from ..models.routing import (
- from backend.core.shared.errors import UnsupportedModality, ModalityConflict
+ from ..core.shared.errors import UnsupportedModality, ModalityConflict
- from gateway.app import store
+ from aura.gateway.app import store
- from backend.core.shared.errors import ModalityConflict
+ from ..core.shared.errors import ModalityConflict
- from backend.core.shared.types import MODALITY_LABELS, ImagingModality
+ from ..core.shared.types import MODALITY_LABELS, ImagingModality
```
</details>

<details><summary><code>aura/backend/services/reasoning/progression.py</code> — 2 statement(s)</summary>

```diff
- from common.config import get_settings
+ from aura.common.config import get_settings
- from schemas.contracts import CaseBundle
+ from aura.schemas.contracts import CaseBundle
```
</details>

<details><summary><code>aura/backend/vision/brain/__init__.py</code> — 9 statement(s)</summary>

```diff
- from backend.vision.brain.config import (
+ from .config import (
- from backend.vision.brain.errors import (
+ from .errors import (
- from backend.vision.brain.types import (
+ from .types import (
- from backend.vision.brain.dataset import BrainSliceDataset
+ from .dataset import BrainSliceDataset
- from backend.vision.brain.ingest import BrainCorpusIngestor, CacheManifest
+ from .ingest import BrainCorpusIngestor, CacheManifest
- from backend.vision.brain.inference import BrainVisionEngine
+ from .inference import BrainVisionEngine
- from backend.vision.brain.model import BrainVisionNetwork
+ from .model import BrainVisionNetwork
- from backend.vision.brain.output import BrainVisionOutput
+ from .output import BrainVisionOutput
- from backend.vision.brain.train import BrainVisionTrainer
+ from .train import BrainVisionTrainer
```
</details>

<details><summary><code>aura/backend/vision/brain/augment.py</code> — 1 statement(s)</summary>

```diff
- from backend.vision.brain.config import AugmentationConfig
+ from .config import AugmentationConfig
```
</details>

<details><summary><code>aura/backend/vision/brain/checkpoint.py</code> — 4 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.config import BrainVisionConfig, PathsConfig
+ from .config import BrainVisionConfig, PathsConfig
- from backend.vision.brain.errors import CheckpointError
+ from .errors import CheckpointError
- from backend.vision.brain.types import BRAIN_VISION_VERSION
+ from .types import BRAIN_VISION_VERSION
```
</details>

<details><summary><code>aura/backend/vision/brain/cli.py</code> — 14 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.config import (
+ from .config import (
- from backend.vision.brain.types import SplitName
+ from .types import SplitName
- from backend.vision.brain.ingest import BrainCorpusIngestor
+ from .ingest import BrainCorpusIngestor
- from backend.vision.brain.train import BrainVisionTrainer
+ from .train import BrainVisionTrainer
- from backend.vision.brain.checkpoint import load_network_checkpoint
+ from .checkpoint import load_network_checkpoint
- from backend.vision.brain.dataset import build_datasets
+ from .dataset import build_datasets
- from backend.vision.brain.losses import MultiTaskLoss
+ from .losses import MultiTaskLoss
- from backend.vision.brain.model.network import build_network
+ from .model.network import build_network
- from backend.vision.brain.validate import BrainValidator
+ from .validate import BrainValidator
- from backend.vision.brain.ingest import BrainCorpusIngestor
+ from .ingest import BrainCorpusIngestor
- from backend.vision.brain.train import BrainVisionTrainer
+ from .train import BrainVisionTrainer
- from backend.vision.brain.model import available_encoders, declared_architectures
+ from .model import available_encoders, declared_architectures
- from backend.vision.brain.ingest import load_manifest
+ from .ingest import load_manifest
```
</details>

<details><summary><code>aura/backend/vision/brain/config.py</code> — 3 statement(s)</summary>

```diff
- from backend.vision.brain.errors import ConfigurationError
+ from .errors import ConfigurationError
- from backend.vision.brain.types import (
+ from .types import (
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/backend/vision/brain/dataset.py</code> — 8 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.augment import SliceAugmenter
+ from .augment import SliceAugmenter
- from backend.vision.brain.config import BrainVisionConfig
+ from .config import BrainVisionConfig
- from backend.vision.brain.degradations import Degradation, DegradationSimulator
+ from .degradations import Degradation, DegradationSimulator
- from backend.vision.brain.errors import CacheUnavailable
+ from .errors import CacheUnavailable
- from backend.vision.brain.ingest import CacheManifest, load_manifest, load_slice_index
+ from .ingest import CacheManifest, load_manifest, load_slice_index
- from backend.vision.brain.sampling import SliceTable
+ from .sampling import SliceTable
- from backend.vision.brain.types import (
+ from .types import (
```
</details>

<details><summary><code>aura/backend/vision/brain/degradations.py</code> — 1 statement(s)</summary>

```diff
- from backend.vision.brain.config import DegradationConfig
+ from .config import DegradationConfig
```
</details>

<details><summary><code>aura/backend/vision/brain/embeddings.py</code> — 3 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.errors import BrainVisionError
+ from .errors import BrainVisionError
- from backend.vision.brain.types import BRAIN_VISION_VERSION, EmbeddingSpec
+ from .types import BRAIN_VISION_VERSION, EmbeddingSpec
```
</details>

<details><summary><code>aura/backend/vision/brain/inference.py</code> — 9 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.checkpoint import CheckpointMeta, load_network_checkpoint
+ from .checkpoint import CheckpointMeta, load_network_checkpoint
- from backend.vision.brain.config import BrainVisionConfig, ModelConfig
+ from .config import BrainVisionConfig, ModelConfig
- from backend.vision.brain.dataset import decode_size, fit_to_grid, normalize_slice
+ from .dataset import decode_size, fit_to_grid, normalize_slice
- from backend.vision.brain.errors import ModelNotTrained
+ from .errors import ModelNotTrained
- from backend.vision.brain.model.network import BrainVisionNetwork, build_network
+ from .model.network import BrainVisionNetwork, build_network
- from backend.vision.brain.output import (
+ from .output import (
- from backend.vision.brain.types import BRAIN_VISION_VERSION, ModalitySpec
+ from .types import BRAIN_VISION_VERSION, ModalitySpec
- from backend.foundation.mri.types import SequenceType
+ from aura.backend.foundation.mri.types import SequenceType
```
</details>

<details><summary><code>aura/backend/vision/brain/ingest.py</code> — 10 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri import (
+ from aura.backend.foundation.mri import (
- from backend.foundation.mri.geometry import to_canonical
+ from aura.backend.foundation.mri.geometry import to_canonical
- from backend.foundation.mri.study import ProcessingHistory, StepTimer, step
+ from aura.backend.foundation.mri.study import ProcessingHistory, StepTimer, step
- from backend.foundation.mri.types import NormalizationMethod, StepStatus
+ from aura.backend.foundation.mri.types import (
- from backend.vision.brain.config import BrainVisionConfig
+ FOUNDATION_VERSION,
- from backend.vision.brain.errors import CacheUnavailable, CorpusIntegrityError
+ NormalizationMethod,
- from backend.vision.brain.io.brats_h5 import (
+ StepStatus,
- from backend.vision.brain.types import (
+ from .types import (
- from backend.foundation.mri.io.base import RawSeries
+ from aura.backend.foundation.mri.io.base import RawSeries
```
</details>

<details><summary><code>aura/backend/vision/brain/io/__init__.py</code> — 1 statement(s)</summary>

```diff
- from backend.vision.brain.io.brats_h5 import (
+ from .brats_h5 import (
```
</details>

<details><summary><code>aura/backend/vision/brain/io/brats_h5.py</code> — 6 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.foundation.mri.geometry import VoxelGeometry
+ from aura.backend.foundation.mri.geometry import VoxelGeometry
- from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
+ from aura.backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
- from backend.foundation.mri.types import FileFormat
+ from aura.backend.foundation.mri.types import FileFormat
- from backend.vision.brain.errors import CorpusIntegrityError, CorpusNotFound
+ from ..errors import CorpusIntegrityError, CorpusNotFound
- from backend.vision.brain.types import (
+ from ..types import (
```
</details>

<details><summary><code>aura/backend/vision/brain/losses.py</code> — 3 statement(s)</summary>

```diff
- from backend.vision.brain.config import LossConfig
+ from .config import LossConfig
- from backend.vision.brain.model.network import NetworkOutput, downsample_label
+ from .model.network import NetworkOutput, downsample_label
- from backend.vision.brain.types import HeadName, TumorRegion
+ from .types import HeadName, TumorRegion
```
</details>

<details><summary><code>aura/backend/vision/brain/metrics.py</code> — 1 statement(s)</summary>

```diff
- from backend.vision.brain.types import (
+ from .types import (
```
</details>

<details><summary><code>aura/backend/vision/brain/model/__init__.py</code> — 6 statement(s)</summary>

```diff
- from backend.vision.brain.model.blocks import (
+ from .blocks import (
- from backend.vision.brain.model.decoder import UNetDecoder2D
+ from .decoder import UNetDecoder2D
- from backend.vision.brain.model.encoder import ModalityStem, ResidualUNetEncoder2D
+ from .encoder import ModalityStem, ResidualUNetEncoder2D
- from backend.vision.brain.model.heads import (
+ from .heads import (
- from backend.vision.brain.model.network import (
+ from .network import (
- from backend.vision.brain.model.registry import (
+ from .registry import (
```
</details>

<details><summary><code>aura/backend/vision/brain/model/decoder.py</code> — 2 statement(s)</summary>

```diff
- from backend.vision.brain.model.blocks import UpsampleBlock
+ from .blocks import UpsampleBlock
- from backend.vision.brain.model.registry import register_decoder
+ from .registry import register_decoder
```
</details>

<details><summary><code>aura/backend/vision/brain/model/encoder.py</code> — 3 statement(s)</summary>

```diff
- from backend.vision.brain.model.blocks import ConvNormAct, ResidualStage
+ from .blocks import ConvNormAct, ResidualStage
- from backend.vision.brain.model.registry import register_encoder
+ from .registry import register_encoder
- from backend.vision.brain.types import DEFAULT_MODALITIES, ModalitySpec
+ from ..types import DEFAULT_MODALITIES, ModalitySpec
```
</details>

<details><summary><code>aura/backend/vision/brain/model/heads.py</code> — 1 statement(s)</summary>

```diff
- from backend.vision.brain.model.blocks import make_activation
+ from .blocks import make_activation
```
</details>

<details><summary><code>aura/backend/vision/brain/model/network.py</code> — 8 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.config import ModelConfig
+ from ..config import ModelConfig
- from backend.vision.brain.degradations import ARTIFACT_CLASSES, ARTIFACT_ORDER
+ from ..degradations import ARTIFACT_CLASSES, ARTIFACT_ORDER
- from backend.vision.brain.model import decoder as _register_decoders  # noqa: F401
+ from . import decoder as _register_decoders  # noqa: F401
- from backend.vision.brain.model import encoder as _register_encoders  # noqa: F401
+ from . import encoder as _register_encoders  # noqa: F401
- from backend.vision.brain.model.heads import (
+ from .heads import (
- from backend.vision.brain.model.registry import build_decoder, build_encoder
+ from .registry import build_decoder, build_encoder
- from backend.vision.brain.types import (
+ from ..types import (
```
</details>

<details><summary><code>aura/backend/vision/brain/model/registry.py</code> — 2 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.errors import ArchitectureUnavailable
+ from ..errors import ArchitectureUnavailable
```
</details>

<details><summary><code>aura/backend/vision/brain/output.py</code> — 1 statement(s)</summary>

```diff
- from backend.vision.brain.types import (
+ from .types import (
```
</details>

<details><summary><code>aura/backend/vision/brain/sampling.py</code> — 3 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.config import CurriculumConfig, SamplingConfig
+ from .config import CurriculumConfig, SamplingConfig
- from backend.vision.brain.types import CurriculumStage
+ from .types import CurriculumStage
```
</details>

<details><summary><code>aura/backend/vision/brain/train.py</code> — 13 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.checkpoint import (
+ from .checkpoint import (
- from backend.vision.brain.config import BrainVisionConfig
+ from .config import BrainVisionConfig
- from backend.vision.brain.dataset import build_datasets
+ from .dataset import build_datasets
- from backend.vision.brain.embeddings import EmbeddingStore
+ from .embeddings import EmbeddingStore
- from backend.vision.brain.errors import ConfigurationError
+ from .errors import ConfigurationError
- from backend.vision.brain.losses import MultiTaskLoss
+ from .losses import MultiTaskLoss
- from backend.vision.brain.metrics import LossMeter
+ from .metrics import LossMeter
- from backend.vision.brain.model.network import BrainVisionNetwork, build_network
+ from .model.network import BrainVisionNetwork, build_network
- from backend.vision.brain.sampling import AdaptiveSliceSampler
+ from .sampling import AdaptiveSliceSampler
- from backend.vision.brain.types import BRAIN_VISION_VERSION, SplitName
+ from .types import BRAIN_VISION_VERSION, SplitName
- from backend.vision.brain.validate import BrainValidator, ValidationReport
+ from .validate import BrainValidator, ValidationReport
- from backend.vision.brain.output import QUALITY_VALIDITY_THRESHOLD
+ from .output import QUALITY_VALIDITY_THRESHOLD
```
</details>

<details><summary><code>aura/backend/vision/brain/types.py</code> — 1 statement(s)</summary>

```diff
- from backend.foundation.mri.types import SequenceType
+ from aura.backend.foundation.mri.types import SequenceType
```
</details>

<details><summary><code>aura/backend/vision/brain/validate.py</code> — 9 statement(s)</summary>

```diff
- from backend.core.shared.logging import get_logger
+ from aura.backend.core.shared.logging import get_logger
- from backend.vision.brain.config import BrainVisionConfig
+ from .config import BrainVisionConfig
- from backend.vision.brain.dataset import TARGET_REGIONS, decode_size
+ from .dataset import TARGET_REGIONS, decode_size
- from backend.vision.brain.degradations import ARTIFACT_CLASSES, ARTIFACT_ORDER
+ from .degradations import ARTIFACT_CLASSES, ARTIFACT_ORDER
- from backend.vision.brain.embeddings import EmbeddingBatch, EmbeddingStore
+ from .embeddings import EmbeddingBatch, EmbeddingStore
- from backend.vision.brain.losses import MultiTaskLoss
+ from .losses import MultiTaskLoss
- from backend.vision.brain.metrics import (
+ from .metrics import (
- from backend.vision.brain.model.network import BrainVisionNetwork
+ from .model.network import BrainVisionNetwork
- from backend.vision.brain.types import HeadName
+ from .types import HeadName
```
</details>

<details><summary><code>aura/gateway/app.py</code> — 37 statement(s)</summary>

```diff
- from common.config import DB_PATH, ensure_dirs, get_settings
+ from aura.common.config import DB_PATH, ensure_dirs, get_settings
- from ml.data import IMG, make_multimodal, make_sample
+ from aura.ml.data import IMG, make_multimodal, make_sample
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
- from schemas.contracts import StudyInput, StructuredPriors
+ from aura.schemas.contracts import StudyInput, StructuredPriors
- from services.models import ModelRegistry
+ from aura.services.models import ModelRegistry
- from gateway.pipeline import Pipeline
+ from .pipeline import Pipeline
- from gateway.seed import seed
+ from .seed import seed
- from gateway.storage import Store
+ from .storage import Store
- from services.enterprise.fhir import export_fhir_diagnostic_report, export_fhir_observations
+ from aura.services.enterprise.fhir import export_fhir_diagnostic_report, export_fhir_observations
- from services.enterprise.hl7 import export_hl7_oru_r01
+ from aura.services.enterprise.hl7 import export_hl7_oru_r01
- from services.agent.discussion import SpecialistDiscussionEngine
+ from aura.services.agent.discussion import SpecialistDiscussionEngine
- from backend.bootstrap import install_router
+ from aura.backend.bootstrap import install_router
- from services.enterprise.dicom_listener import DicomListener
+ from aura.services.enterprise.dicom_listener import DicomListener
- from gateway.security import enforce
+ from .security import enforce
- from schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
+ from aura.schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
- from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
+ from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
- from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
+ from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
- from common.config import finding_present_threshold
+ from aura.common.config import finding_present_threshold
- from schemas.contracts import CaseState
+ from aura.schemas.contracts import CaseState
- from gateway.storage import compute_provenance_hash
+ from .storage import compute_provenance_hash
- from schemas.contracts import ClinicalContext
+ from aura.schemas.contracts import ClinicalContext
- from backend.engines.neuro.bundle import _findings_text, _impression_text, _recommendation_text, _confidence_text, _recommendations, _priority
+ from aura.backend.engines.neuro.bundle import _findings_text, _impression_text, _recommendation_text, _confidence_text, _recommendations, _priority
- from services.fusion.evidence import encode, to_evidence_items
+ from aura.services.fusion.evidence import encode, to_evidence_items
- from schemas.contracts import StudyInput
+ from aura.schemas.contracts import StudyInput
- from gateway.security import validate_upload_name, read_capped, validate_mri_content
+ from .security import validate_upload_name, read_capped, validate_mri_content
- from services.vision.xray_gate import validate_cxr
+ from aura.services.vision.xray_gate import validate_cxr
- from services.vision.io import study_from_cxr
+ from aura.services.vision.io import study_from_cxr
- from services.safety import ClinicalSafetyException
+ from aura.services.safety import ClinicalSafetyException
- from services.inference.audit_log import log_inference
+ from aura.services.inference.audit_log import log_inference
- from gateway.security import validate_upload_name, read_capped, validate_mri_content
+ from .security import validate_upload_name, read_capped, validate_mri_content
- from services.vision.xray_gate import validate_cxr
+ from aura.services.vision.xray_gate import validate_cxr
- from services.vision.io import study_from_cxr
+ from aura.services.vision.io import study_from_cxr
- from services.fusion.evidence import encode
+ from aura.services.fusion.evidence import encode
- from services.agent.active_diagnosis import ActiveDiagnosisAgent
+ from aura.services.agent.active_diagnosis import ActiveDiagnosisAgent
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
+ from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/gateway/pipeline.py</code> — 17 statement(s)</summary>

```diff
- from common import eventbus as ev
+ from aura.common import eventbus as ev
- from common.config import get_settings
+ from aura.common.config import get_settings
- from common.eventbus import EventBus
+ from aura.common.eventbus import EventBus
- from common.mathx import entropy, softmax
+ from aura.common.mathx import entropy, softmax
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
- from schemas.contracts import (
+ from aura.schemas.contracts import (
- from services.explain import ExplainEngine
+ from aura.services.explain import ExplainEngine
- from services.fusion import FusionEngine
+ from aura.services.fusion import FusionEngine
- from services.fusion.evidence import encode, to_evidence_items
+ from aura.services.fusion.evidence import encode, to_evidence_items
- from services.memory import MemoryEngine
+ from aura.services.memory import MemoryEngine
- from services.reasoning import ClinicalReasoner
+ from aura.services.reasoning import ClinicalReasoner
- from services.recommend import RecommendEngine
+ from aura.services.recommend import RecommendEngine
- from services.report import ReportEngine
+ from aura.services.report import ReportEngine
- from services.safety import SafetyEngine, ClinicalSafetyController, ClinicalDecisionReadinessEngine
+ from aura.services.safety import SafetyEngine, ClinicalSafetyController, ClinicalDecisionReadinessEngine
- from services.vision import VisionEngine
+ from aura.services.vision import VisionEngine
- from schemas.clinical import Finding
+ from aura.schemas.clinical import Finding
- from services.safety.controller import ClinicalSafetyException
+ from aura.services.safety.controller import ClinicalSafetyException
```
</details>

<details><summary><code>aura/gateway/security.py</code> — 1 statement(s)</summary>

```diff
- from common.config import get_settings
+ from aura.common.config import get_settings
```
</details>

<details><summary><code>aura/gateway/seed.py</code> — 4 statement(s)</summary>

```diff
- from ml.data import make_dataset, make_multimodal, make_ood_sample, IMG
+ from aura.ml.data import make_dataset, make_multimodal, make_ood_sample, IMG
- from schemas.contracts import StudyInput, StructuredPriors
+ from aura.schemas.contracts import StudyInput, StructuredPriors
- from gateway.pipeline import Pipeline
+ from .pipeline import Pipeline
- from gateway.storage import Store
+ from .storage import Store
```
</details>

<details><summary><code>aura/gateway/storage.py</code> — 5 statement(s)</summary>

```diff
- from schemas.contracts import CaseBundle
+ from aura.schemas.contracts import CaseBundle
- from schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
+ from aura.schemas.clinical import DIAGNOSIS_LABELS, FINDING_LABELS
- from schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
+ from aura.schemas.clinical import DIAGNOSIS_LABELS, Diagnosis
- from common.config import get_settings
+ from aura.common.config import get_settings
- from services.safety.aci import AdaptiveConformalInference, ACIState
+ from aura.services.safety.aci import AdaptiveConformalInference, ACIState
```
</details>

<details><summary><code>aura/knowledge/guidelines/templates.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import Finding, Diagnosis
+ from aura.schemas.clinical import Finding, Diagnosis
```
</details>

<details><summary><code>aura/mimic/cleaning.py</code> — 3 statement(s)</summary>

```diff
- from mimic.labeling import ReportLabel, label_patient_reports
+ from .labeling import ReportLabel, label_patient_reports
- from mimic.loaders import MimicCxrLoader, PatientRecord
+ from .loaders import MimicCxrLoader, PatientRecord
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
```
</details>

<details><summary><code>aura/mimic/evaluation.py</code> — 2 statement(s)</summary>

```diff
- from ml.evaluation.metrics import evaluate as evaluate_diagnosis_core
+ from aura.ml.evaluation.metrics import evaluate as evaluate_diagnosis_core
- from services.safety.uncertainty import brier_score
+ from aura.services.safety.uncertainty import brier_score
```
</details>

<details><summary><code>aura/mimic/features.py</code> — 4 statement(s)</summary>

```diff
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.patient import Patient, iter_patients
+ from .patient import Patient, iter_patients
- from schemas.clinical import DIAGNOSES, FINDINGS, Diagnosis, Finding
+ from aura.schemas.clinical import DIAGNOSES, FINDINGS, Diagnosis, Finding
- from services.vision.features import FEATURE_NAMES as IMG_FEATURE_NAMES, extract_features
+ from aura.services.vision.features import FEATURE_NAMES as IMG_FEATURE_NAMES, extract_features
```
</details>

<details><summary><code>aura/mimic/labeling.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Finding
+ from aura.schemas.clinical import CHEST_DIAGNOSES, Diagnosis, Finding
```
</details>

<details><summary><code>aura/mimic/labeling_v2.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import CHEST_FINDINGS, Finding
+ from aura.schemas.clinical import CHEST_FINDINGS, Finding
```
</details>

<details><summary><code>aura/mimic/loaders.py</code> — 2 statement(s)</summary>

```diff
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.parsing import safe_str_list
+ from .parsing import safe_str_list
```
</details>

<details><summary><code>aura/mimic/patient.py</code> — 8 statement(s)</summary>

```diff
- from mimic.cleaning import CleanedPatient, clean_record
+ from .cleaning import CleanedPatient, clean_record
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.labeling import ReportLabel
+ from .labeling import ReportLabel
- from mimic.loaders import MimicCxrLoader, PatientRecord
+ from .loaders import MimicCxrLoader, PatientRecord
- from mimic.timeline import PatientTimeline, StudyEvent, build_timeline
+ from .timeline import PatientTimeline, StudyEvent, build_timeline
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
- from schemas.contracts import MultimodalContext, StructuredPriors, StudyInput
+ from aura.schemas.contracts import MultimodalContext, StructuredPriors, StudyInput
- from services.vision.io import study_from_cxr  # lazy: pulls numpy/PIL
+ from aura.services.vision.io import study_from_cxr  # lazy: pulls numpy/PIL
```
</details>

<details><summary><code>aura/mimic/performance.py</code> — 3 statement(s)</summary>

```diff
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.features import FeatureRow, feature_names, patient_feature_row
+ from .features import FeatureRow, feature_names, patient_feature_row
- from mimic.patient import Patient, iter_patients
+ from .patient import Patient, iter_patients
```
</details>

<details><summary><code>aura/mimic/seed.py</code> — 4 statement(s)</summary>

```diff
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from gateway.storage import Store
+ from aura.gateway.storage import Store
- from mimic.config import get_mimic_paths
+ from .config import get_mimic_paths
- from mimic.patient import iter_patients
+ from .patient import iter_patients
```
</details>

<details><summary><code>aura/mimic/splits.py</code> — 4 statement(s)</summary>

```diff
- from mimic.cleaning import DataCleaner
+ from .cleaning import DataCleaner
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.loaders import MimicCxrLoader
+ from .loaders import MimicCxrLoader
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
```
</details>

<details><summary><code>aura/mimic/tasks.py</code> — 4 statement(s)</summary>

```diff
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.features import FeatureEngineer, feature_names
+ from .features import FeatureEngineer, feature_names
- from mimic.splits import DatasetBuilder
+ from .splits import DatasetBuilder
- from schemas.clinical import DIAGNOSES, FINDINGS
+ from aura.schemas.clinical import DIAGNOSES, FINDINGS
```
</details>

<details><summary><code>aura/mimic/timeline.py</code> — 3 statement(s)</summary>

```diff
- from mimic.labeling import ReportLabel, label_report
+ from .labeling import ReportLabel, label_report
- from mimic.loaders import PatientRecord
+ from .loaders import PatientRecord
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
```
</details>

<details><summary><code>aura/mimic/training.py</code> — 2 statement(s)</summary>

```diff
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.tasks import TaskDataset, TaskDatasetBuilder
+ from .tasks import TaskDataset, TaskDatasetBuilder
```
</details>

<details><summary><code>aura/mimic/uncertainty.py</code> — 4 statement(s)</summary>

```diff
- from services.safety.calibration import (
+ from aura.services.safety.calibration import (
- from services.safety.uncertainty import (
+ from aura.services.safety.uncertainty import (
- from mimic.training import GBMTrainer
+ from .training import GBMTrainer
- from services.safety.uncertainty import enable_dropout
+ from aura.services.safety.uncertainty import enable_dropout
```
</details>

<details><summary><code>aura/mimic/verify.py</code> — 2 statement(s)</summary>

```diff
- from mimic.config import MimicPaths, get_mimic_paths
+ from .config import MimicPaths, get_mimic_paths
- from mimic.parsing import safe_list as _safe_list
+ from .parsing import safe_list as _safe_list
```
</details>

<details><summary><code>aura/ml/data.py</code> — 4 statement(s)</summary>

```diff
- from schemas.clinical import CHEST_FINDINGS, Diagnosis, Finding
+ from aura.schemas.clinical import CHEST_FINDINGS, Diagnosis, Finding
- from schemas.contracts import StructuredPriors
+ from aura.schemas.contracts import StructuredPriors
- from common.anatomy import IMG, REGIONS, _px  # noqa: F401  (re-export)
+ from aura.common.anatomy import IMG, REGIONS, _px  # noqa: F401  (re-export)
- from schemas.contracts import (
+ from aura.schemas.contracts import (
```
</details>

<details><summary><code>aura/ml/evaluation/benchmark.py</code> — 11 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, get_settings
+ from aura.common.config import ARTIFACTS, get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.classical import ClassicalFusion
+ from aura.services.fusion.classical import ClassicalFusion
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
- from services.safety.calibration import (
+ from aura.services.safety.calibration import (
- from ml.training.dataset import build_evidence_dataset, make_splits
+ from ..training.dataset import build_evidence_dataset, make_splits
- from ml.training.dataset import real_evidence_splits
+ from ..training.dataset import real_evidence_splits
- from ml.evaluation.metrics import evaluate, print_report
+ from .metrics import evaluate, print_report
- from services.fusion.ensemble import DeepEnsemble
+ from aura.services.fusion.ensemble import DeepEnsemble
- from services.fusion.learnable import LearnableFusion
+ from aura.services.fusion.learnable import LearnableFusion
```
</details>

<details><summary><code>aura/ml/evaluation/clinical_eval.py</code> — 6 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
- from ml.vision_cxr.config import TrainConfig
+ from ..vision_cxr.config import TrainConfig
- from ml.vision_cxr.dataset import load_mimic_samples
+ from ..vision_cxr.dataset import load_mimic_samples
- from ml.vision_cxr.dataset import ChestXrayDataset, get_transforms
+ from ..vision_cxr.dataset import ChestXrayDataset, get_transforms
- from ml.vision_cxr.inference import VisionModel
+ from ..vision_cxr.inference import VisionModel
```
</details>

<details><summary><code>aura/ml/evaluation/metrics.py</code> — 2 statement(s)</summary>

```diff
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.safety.uncertainty import brier_score, reliability_curve
+ from aura.services.safety.uncertainty import brier_score, reliability_curve
```
</details>

<details><summary><code>aura/ml/evaluation/perf_benchmark.py</code> — 3 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
- from ml.vision_cxr.model import DenseNet121CXR
+ from ..vision_cxr.model import DenseNet121CXR
```
</details>

<details><summary><code>aura/ml/evaluation/quantum_demo.py</code> — 5 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.qmba import QuantumMeasurementBudget
+ from aura.services.fusion.qmba import QuantumMeasurementBudget
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
```
</details>

<details><summary><code>aura/ml/evaluation/quantum_study.py</code> — 11 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, ensure_dirs, get_settings
+ from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.device import make_qnode
+ from aura.services.fusion.device import make_qnode
- from services.fusion.qmba import QuantumMeasurementBudget
+ from aura.services.fusion.qmba import QuantumMeasurementBudget
- from services.fusion.qmeasure import measure_entanglement
+ from aura.services.fusion.qmeasure import measure_entanglement
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
- from services.safety.calibration import (
+ from aura.services.safety.calibration import (
- from ml.training.dataset import build_evidence_dataset, make_splits, \
+ from ..training.dataset import build_evidence_dataset, make_splits, \
- from ml.training.train_fusion import train_classical, train_quantum
+ from ..training.train_fusion import train_classical, train_quantum
- from services.fusion.evidence import EVIDENCE_CHANNELS
+ from aura.services.fusion.evidence import EVIDENCE_CHANNELS
```
</details>

<details><summary><code>aura/ml/evaluation/run_ibm_hardware.py</code> — 4 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
```
</details>

<details><summary><code>aura/ml/evaluation/run_pipeline.py</code> — 1 statement(s)</summary>

```diff
- from ml.evaluation.perf_benchmark import run as run_perf
+ from .perf_benchmark import run as run_perf
```
</details>

<details><summary><code>aura/ml/evaluation/vision_calibration.py</code> — 7 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
- from ml.evaluation.clinical_eval import binary_ece, load_validation
+ from .clinical_eval import binary_ece, load_validation
- from ml.vision_cxr.dataset import ChestXrayDataset, get_transforms
+ from ..vision_cxr.dataset import ChestXrayDataset, get_transforms
- from ml.vision_cxr.inference import VisionModel
+ from ..vision_cxr.inference import VisionModel
- from ml.vision_cxr.dataset import ChestXrayDataset, get_transforms
+ from ..vision_cxr.dataset import ChestXrayDataset, get_transforms
- from services.safety.uncertainty import enable_dropout
+ from aura.services.safety.uncertainty import enable_dropout
```
</details>

<details><summary><code>aura/ml/explain_demo/calibration_audit.py</code> — 5 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from ml.vision_cxr.dataset import load_mimic_samples
+ from ..vision_cxr.dataset import load_mimic_samples
- from ml.vision_cxr.inference import VisionModel
+ from ..vision_cxr.inference import VisionModel
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
```
</details>

<details><summary><code>aura/ml/explain_demo/gradcam_validation.py</code> — 8 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from ml.vision_cxr.dataset import load_mimic_samples
+ from ..vision_cxr.dataset import load_mimic_samples
- from ml.vision_cxr.inference import VisionModel
+ from ..vision_cxr.inference import VisionModel
- from schemas.clinical import Finding, FINDINGS
+ from aura.schemas.clinical import Finding, FINDINGS
- from services.explain import methods as M
+ from aura.services.explain import methods as M
- from services.explain import overlays as OV
+ from aura.services.explain import overlays as OV
- from services.vision.engine import _FINDING_REGION
+ from aura.services.vision.engine import _FINDING_REGION
```
</details>

<details><summary><code>aura/ml/explain_demo/ood_gate_demo.py</code> — 4 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.parsing import safe_str_list
+ from aura.mimic.parsing import safe_str_list
- from services.vision.xray_gate import validate_cxr
+ from aura.services.vision.xray_gate import validate_cxr
```
</details>

<details><summary><code>aura/ml/training/cxr_dataset.py</code> — 3 statement(s)</summary>

```diff
- from schemas.clinical import FINDINGS, Finding
+ from aura.schemas.clinical import FINDINGS, Finding
- from services.vision.io import load_cxr
+ from aura.services.vision.io import load_cxr
- from ml.data import make_dataset
+ from ..data import make_dataset
```
</details>

<details><summary><code>aura/ml/training/dataset.py</code> — 6 statement(s)</summary>

```diff
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.evidence import encode
+ from aura.services.fusion.evidence import encode
- from services.vision import VisionEngine
+ from aura.services.vision import VisionEngine
- from ml.data import Sample, make_dataset
+ from ..data import Sample, make_dataset
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.patient import iter_patients
+ from aura.mimic.patient import iter_patients
```
</details>

<details><summary><code>aura/ml/training/prepare_mimic_manifest.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import FINDINGS, Finding
+ from aura.schemas.clinical import FINDINGS, Finding
```
</details>

<details><summary><code>aura/ml/training/recalibrate_backend.py</code> — 8 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, ensure_dirs, get_settings
+ from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.classical import ClassicalFusion
+ from aura.services.fusion.classical import ClassicalFusion
- from services.fusion.learnable import LearnableFusion
+ from aura.services.fusion.learnable import LearnableFusion
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
- from services.safety.calibration import (
+ from aura.services.safety.calibration import (
- from ml.training.dataset import real_evidence_splits
+ from .dataset import real_evidence_splits
```
</details>

<details><summary><code>aura/ml/training/recalibrate_ood.py</code> — 11 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, get_settings
+ from aura.common.config import ARTIFACTS, get_settings
- from common.mathx import energy_score
+ from aura.common.mathx import energy_score
- from ml.data import make_sample
+ from ..data import make_sample
- from mimic.patient import iter_patients
+ from aura.mimic.patient import iter_patients
- from services.fusion import FusionEngine
+ from aura.services.fusion import FusionEngine
- from services.fusion.evidence import encode
+ from aura.services.fusion.evidence import encode
- from services.safety.calibration import Calibration
+ from aura.services.safety.calibration import Calibration
- from services.vision import VisionEngine
+ from aura.services.vision import VisionEngine
- from ml.data import IMG
+ from ..data import IMG
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from schemas.contracts import StudyInput
+ from aura.schemas.contracts import StudyInput
```
</details>

<details><summary><code>aura/ml/training/train_cnn.py</code> — 4 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, ensure_dirs
+ from aura.common.config import ARTIFACTS, ensure_dirs
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
- from services.vision.cnn import TIMM_ARCHES, select_device
+ from aura.services.vision.cnn import TIMM_ARCHES, select_device
- from ml.training.cxr_dataset import (
+ from .cxr_dataset import (
```
</details>

<details><summary><code>aura/ml/training/train_fusion.py</code> — 8 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, ensure_dirs, get_settings
+ from aura.common.config import ARTIFACTS, ensure_dirs, get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.device import make_qnode
+ from aura.services.fusion.device import make_qnode
- from services.safety.calibration import (
+ from aura.services.safety.calibration import (
- from ml.training.dataset import (
+ from .dataset import (
- from services.fusion.learnable import train_learnable
+ from aura.services.fusion.learnable import train_learnable
- from services.safety.uncertainty import mondrian_qhats
+ from aura.services.safety.uncertainty import mondrian_qhats
```
</details>

<details><summary><code>aura/ml/training/train_vision.py</code> — 5 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, ensure_dirs
+ from aura.common.config import ARTIFACTS, ensure_dirs
- from common.mathx import sigmoid
+ from aura.common.mathx import sigmoid
- from schemas.clinical import CHEST_FINDINGS, Finding
+ from aura.schemas.clinical import CHEST_FINDINGS, Finding
- from services.vision.features import FEATURE_NAMES, extract_features
+ from aura.services.vision.features import FEATURE_NAMES, extract_features
- from ml.data import Sample, make_dataset
+ from ..data import Sample, make_dataset
```
</details>

<details><summary><code>aura/ml/vision_cxr/config.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
```
</details>

<details><summary><code>aura/ml/vision_cxr/dataset.py</code> — 4 statement(s)</summary>

```diff
- from schemas.clinical import FINDINGS, Finding
+ from aura.schemas.clinical import FINDINGS, Finding
- from mimic.parsing import safe_str_list
+ from aura.mimic.parsing import safe_str_list
- from mimic.labeling import label_report
+ from aura.mimic.labeling import label_report
- from mimic.labeling_v2 import label_v2
+ from aura.mimic.labeling_v2 import label_v2
```
</details>

<details><summary><code>aura/ml/vision_cxr/inference.py</code> — 3 statement(s)</summary>

```diff
- from ml.vision_cxr.model import DenseNet121CXR
+ from .model import DenseNet121CXR
- from schemas.clinical import FINDINGS, Finding
+ from aura.schemas.clinical import FINDINGS, Finding
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/ml/vision_cxr/train.py</code> — 12 statement(s)</summary>

```diff
- from ml.vision_cxr.config import TrainConfig
+ from .config import TrainConfig
- from ml.vision_cxr.dataset import build_loaders
+ from .dataset import build_loaders
- from ml.vision_cxr.model import DenseNet121CXR
+ from .model import DenseNet121CXR
- from ml.vision_cxr.losses import MultiLabelLoss, RegularizedMultiLabelLoss
+ from .losses import MultiLabelLoss, RegularizedMultiLabelLoss
- from ml.vision_cxr.validate import evaluate_model
+ from .validate import evaluate_model
- from ml.vision_cxr.checkpoint import save_model_checkpoint, save_best_model, load_model_checkpoint
+ from .checkpoint import save_model_checkpoint, save_best_model, load_model_checkpoint
- from ml.vision_cxr.utils import set_seed, HistoryLogger, plot_training_history
+ from .utils import set_seed, HistoryLogger, plot_training_history
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from schemas.contracts import StructuredPriors
+ from aura.schemas.contracts import StructuredPriors
- from mimic.loaders import MimicCxrLoader
+ from aura.mimic.loaders import MimicCxrLoader
- from services.vision.io import load_cxr, study_from_cxr
+ from aura.services.vision.io import load_cxr, study_from_cxr
- from common.config import get_settings
+ from aura.common.config import get_settings
```
</details>

<details><summary><code>aura/ml/vision_cxr/validate.py</code> — 2 statement(s)</summary>

```diff
- from ml.vision_cxr.metrics import compute_multilabel_metrics
+ from .metrics import compute_multilabel_metrics
- from ml.vision_cxr.losses import RegularizedMultiLabelLoss
+ from .losses import RegularizedMultiLabelLoss
```
</details>

<details><summary><code>aura/run_ablation.py</code> — 10 statement(s)</summary>

```diff
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.quantum import QuantumFusion
+ from aura.services.fusion.quantum import QuantumFusion
- from services.fusion.classical import ClassicalFusion
+ from aura.services.fusion.classical import ClassicalFusion
- from services.fusion.learnable import LearnableFusion
+ from aura.services.fusion.learnable import LearnableFusion
- from services.agent.active_diagnosis import ActiveDiagnosisAgent
+ from aura.services.agent.active_diagnosis import ActiveDiagnosisAgent
- from services.recommend.engine import RecommendEngine
+ from aura.services.recommend.engine import RecommendEngine
- from services.safety.calibration import fit_temperature, expected_calibration_error
+ from aura.services.safety.calibration import fit_temperature, expected_calibration_error
- from ml.training.dataset import build_real_evidence_dataset
+ from aura.ml.training.dataset import build_real_evidence_dataset
- from ml.training.dataset import make_splits, build_evidence_dataset
+ from aura.ml.training.dataset import make_splits, build_evidence_dataset
```
</details>

<details><summary><code>aura/schemas/__init__.py</code> — 2 statement(s)</summary>

```diff
- from schemas.clinical import (
+ from .clinical import (
- from schemas.contracts import (
+ from .contracts import (
```
</details>

<details><summary><code>aura/schemas/contracts.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import Diagnosis, Finding, Modality
+ from .clinical import Diagnosis, Finding, Modality
```
</details>

<details><summary><code>aura/services/agent/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.agent.active_diagnosis import ActiveDiagnosisAgent, DiagnosisTrajectory
+ from .active_diagnosis import ActiveDiagnosisAgent, DiagnosisTrajectory
```
</details>

<details><summary><code>aura/services/agent/active_diagnosis.py</code> — 4 statement(s)</summary>

```diff
- from common.mathx import entropy, softmax
+ from aura.common.mathx import entropy, softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from services.fusion.evidence import EVIDENCE_CHANNELS
+ from ..fusion.evidence import EVIDENCE_CHANNELS
- from services.recommend.engine import CATALOG, RecommendEngine
+ from ..recommend.engine import CATALOG, RecommendEngine
```
</details>

<details><summary><code>aura/services/agent/discussion.py</code> — 2 statement(s)</summary>

```diff
- from schemas.contracts import CaseBundle
+ from aura.schemas.contracts import CaseBundle
- from schemas.clinical import Diagnosis, Finding, DIAGNOSIS_LABELS, FINDING_LABELS
+ from aura.schemas.clinical import Diagnosis, Finding, DIAGNOSIS_LABELS, FINDING_LABELS
```
</details>

<details><summary><code>aura/services/enterprise/dicom_listener.py</code> — 1 statement(s)</summary>

```diff
- from backend.core.upload.intake import stage_bytes
+ from aura.backend.core.upload.intake import stage_bytes
```
</details>

<details><summary><code>aura/services/enterprise/fhir.py</code> — 2 statement(s)</summary>

```diff
- from schemas.contracts import CaseBundle
+ from aura.schemas.contracts import CaseBundle
- from schemas.clinical import DIAGNOSIS_LABELS
+ from aura.schemas.clinical import DIAGNOSIS_LABELS
```
</details>

<details><summary><code>aura/services/enterprise/hl7.py</code> — 2 statement(s)</summary>

```diff
- from schemas.contracts import CaseBundle
+ from aura.schemas.contracts import CaseBundle
- from schemas.clinical import DIAGNOSIS_LABELS
+ from aura.schemas.clinical import DIAGNOSIS_LABELS
```
</details>

<details><summary><code>aura/services/explain/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.explain.engine import ExplainEngine
+ from .engine import ExplainEngine
```
</details>

<details><summary><code>aura/services/explain/engine.py</code> — 6 statement(s)</summary>

```diff
- from common.anatomy import IMG, resize_to as _resize_to
+ from aura.common.anatomy import IMG, resize_to as _resize_to
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
- from schemas.contracts import Explanation
+ from aura.schemas.contracts import Explanation
- from services.fusion.evidence import EVIDENCE_CHANNELS
+ from ..fusion.evidence import EVIDENCE_CHANNELS
- from services.explain import methods as M
+ from . import methods as M
```
</details>

<details><summary><code>aura/services/explain/methods.py</code> — 3 statement(s)</summary>

```diff
- from schemas.clinical import Finding
+ from aura.schemas.clinical import Finding
- from common.anatomy import resize_to
+ from aura.common.anatomy import resize_to
- from services.explain.scorecam import score_cam
+ from .scorecam import score_cam
```
</details>

<details><summary><code>aura/services/explain/scorecam.py</code> — 2 statement(s)</summary>

```diff
- from schemas.clinical import Finding
+ from aura.schemas.clinical import Finding
- from services.explain.methods import _resize01, _target_index
+ from .methods import _resize01, _target_index
```
</details>

<details><summary><code>aura/services/fusion/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.fusion.engine import FusionEngine
+ from .engine import FusionEngine
```
</details>

<details><summary><code>aura/services/fusion/classical.py</code> — 3 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
```
</details>

<details><summary><code>aura/services/fusion/conflict.py</code> — 1 statement(s)</summary>

```diff
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
```
</details>

<details><summary><code>aura/services/fusion/engine.py</code> — 12 statement(s)</summary>

```diff
- from common.config import get_settings
+ from aura.common.config import get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
- from schemas.contracts import FusionResult, StructuredPriors, VisionResult
+ from aura.schemas.contracts import FusionResult, StructuredPriors, VisionResult
- from services.fusion.classical import ClassicalFusion
+ from .classical import ClassicalFusion
- from services.fusion.conflict import WassersteinTieBreaker
+ from .conflict import WassersteinTieBreaker
- from services.fusion.evidence import encode
+ from .evidence import encode
- from services.fusion.learnable import LearnableFusion
+ from .learnable import LearnableFusion
- from services.fusion.quantum import QuantumFusion
+ from .quantum import QuantumFusion
- from services.fusion.qae import QuantumAutoencoder
+ from .qae import QuantumAutoencoder
- from services.fusion.qmeasure import measure_entanglement
+ from .qmeasure import measure_entanglement
- from services.fusion.evidence import prior_risk_score
+ from .evidence import prior_risk_score
```
</details>

<details><summary><code>aura/services/fusion/ensemble.py</code> — 3 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
```
</details>

<details><summary><code>aura/services/fusion/evidence.py</code> — 2 statement(s)</summary>

```diff
- from schemas.clinical import Finding
+ from aura.schemas.clinical import Finding
- from schemas.contracts import EvidenceItem, EvidenceKind, StructuredPriors, VisionResult
+ from aura.schemas.contracts import EvidenceItem, EvidenceKind, StructuredPriors, VisionResult
```
</details>

<details><summary><code>aura/services/fusion/learnable.py</code> — 3 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import sigmoid, softmax
+ from aura.common.mathx import sigmoid, softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
```
</details>

<details><summary><code>aura/services/fusion/multimodal.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import softmax
+ from aura.common.mathx import softmax
```
</details>

<details><summary><code>aura/services/fusion/projection.py</code> — 1 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/services/fusion/qae.py</code> — 1 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/services/fusion/qmba.py</code> — 2 statement(s)</summary>

```diff
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES
+ from aura.schemas.clinical import DIAGNOSES
```
</details>

<details><summary><code>aura/services/fusion/qmeasure.py</code> — 1 statement(s)</summary>

```diff
- from services.fusion.evidence import EVIDENCE_CHANNELS
+ from .evidence import EVIDENCE_CHANNELS
```
</details>

<details><summary><code>aura/services/fusion/quantum.py</code> — 5 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, get_settings
+ from aura.common.config import ARTIFACTS, get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
- from services.fusion.device import make_qnode
+ from .device import make_qnode
- from services.fusion.device import make_probs_qnode
+ from .device import make_probs_qnode
```
</details>

<details><summary><code>aura/services/inference/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.inference.predict import predict_image
+ from .predict import predict_image
```
</details>

<details><summary><code>aura/services/inference/audit_log.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, finding_present_threshold
+ from aura.common.config import ARTIFACTS, finding_present_threshold
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
```
</details>

<details><summary><code>aura/services/inference/predict.py</code> — 10 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import FINDING_LABELS, Finding
+ from aura.schemas.clinical import FINDING_LABELS, Finding
- from schemas.contracts import StructuredPriors, StudyInput
+ from aura.schemas.contracts import StructuredPriors, StudyInput
- from services.report.clinical_report import build_clinical_report, save_report
+ from ..report.clinical_report import build_clinical_report, save_report
- from services.vision.io import load_cxr, study_from_cxr
+ from ..vision.io import load_cxr, study_from_cxr
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from services.inference.audit_log import log_inference
+ from .audit_log import log_inference
- from common.config import finding_present_threshold
+ from aura.common.config import finding_present_threshold
- from services.explain import methods as M
+ from ..explain import methods as M
- from services.explain import overlays as O
+ from ..explain import overlays as O
```
</details>

<details><summary><code>aura/services/memory/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.memory.engine import MemoryEngine
+ from .engine import MemoryEngine
```
</details>

<details><summary><code>aura/services/models/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.models.registry import ModelRegistry
+ from .registry import ModelRegistry
```
</details>

<details><summary><code>aura/services/models/registry.py</code> — 1 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/services/reasoning/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.reasoning.engine import ClinicalReasoner
+ from .engine import ClinicalReasoner
```
</details>

<details><summary><code>aura/services/reasoning/engine.py</code> — 6 statement(s)</summary>

```diff
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES, Diagnosis, Finding
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis, Finding
- from schemas.contracts import (
+ from aura.schemas.contracts import (
- from knowledge.guidelines.templates import GUIDELINE_TEMPLATES
+ from aura.knowledge.guidelines.templates import GUIDELINE_TEMPLATES
- from common.config import get_settings
+ from aura.common.config import get_settings
- from services.reasoning.qbn import QuantumBayesianNetwork
+ from .qbn import QuantumBayesianNetwork
```
</details>

<details><summary><code>aura/services/reasoning/qbn.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
```
</details>

<details><summary><code>aura/services/recommend/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.recommend.engine import RecommendEngine
+ from .engine import RecommendEngine
```
</details>

<details><summary><code>aura/services/recommend/engine.py</code> — 5 statement(s)</summary>

```diff
- from common.mathx import entropy, softmax
+ from aura.common.mathx import entropy, softmax
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
- from schemas.contracts import Recommendation
+ from aura.schemas.contracts import Recommendation
- from services.fusion.evidence import EVIDENCE_CHANNELS
+ from ..fusion.evidence import EVIDENCE_CHANNELS
- from services.recommend.causal import CausalDependencyGraph, JointEIGSelector
+ from .causal import CausalDependencyGraph, JointEIGSelector
```
</details>

<details><summary><code>aura/services/report/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.report.engine import ReportEngine
+ from .engine import ReportEngine
```
</details>

<details><summary><code>aura/services/report/clinical_report.py</code> — 6 statement(s)</summary>

```diff
- from schemas.clinical import (
+ from aura.schemas.clinical import (
- from schemas.contracts import CaseBundle
+ from aura.schemas.contracts import CaseBundle
- from services.fusion.evidence import EVIDENCE_CHANNELS
+ from ..fusion.evidence import EVIDENCE_CHANNELS
- from common.config import finding_present_threshold
+ from aura.common.config import finding_present_threshold
- from schemas.contracts import EvidenceKind
+ from aura.schemas.contracts import EvidenceKind
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
```
</details>

<details><summary><code>aura/services/report/engine.py</code> — 3 statement(s)</summary>

```diff
- from schemas.clinical import (
+ from aura.schemas.clinical import (
- from schemas.contracts import (
+ from aura.schemas.contracts import (
- from common.config import finding_present_threshold
+ from aura.common.config import finding_present_threshold
```
</details>

<details><summary><code>aura/services/safety/__init__.py</code> — 3 statement(s)</summary>

```diff
- from services.safety.engine import SafetyEngine
+ from .engine import SafetyEngine
- from services.safety.controller import ClinicalSafetyController, ClinicalSafetyException, SafetyControllerOutput
+ from .controller import ClinicalSafetyController, ClinicalSafetyException, SafetyControllerOutput
- from services.safety.readiness import ClinicalDecisionReadinessEngine, DecisionReadinessProfile
+ from .readiness import ClinicalDecisionReadinessEngine, DecisionReadinessProfile
```
</details>

<details><summary><code>aura/services/safety/calibration.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import energy_score, softmax
+ from aura.common.mathx import energy_score, softmax
```
</details>

<details><summary><code>aura/services/safety/controller.py</code> — 3 statement(s)</summary>

```diff
- from common.config import get_settings
+ from aura.common.config import get_settings
- from common.mathx import energy_score
+ from aura.common.mathx import energy_score
- from backend.core.shared.errors import AuraBackendError
+ from aura.backend.core.shared.errors import AuraBackendError
```
</details>

<details><summary><code>aura/services/safety/engine.py</code> — 7 statement(s)</summary>

```diff
- from common.config import ARTIFACTS, get_settings
+ from aura.common.config import ARTIFACTS, get_settings
- from common.mathx import energy_score, entropy, softmax
+ from aura.common.mathx import energy_score, entropy, softmax
- from schemas.clinical import DIAGNOSES, Diagnosis
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis
- from schemas.contracts import AbstentionReason, Prediction, SafetyAssessment
+ from aura.schemas.contracts import AbstentionReason, Prediction, SafetyAssessment
- from services.fusion.ensemble import DeepEnsemble
+ from ..fusion.ensemble import DeepEnsemble
- from services.safety.calibration import Calibration
+ from .calibration import Calibration
- from services.safety.uncertainty import ensemble_decomposition, mondrian_set
+ from .uncertainty import ensemble_decomposition, mondrian_set
```
</details>

<details><summary><code>aura/services/safety/readiness.py</code> — 7 statement(s)</summary>

```diff
- from common.config import get_settings
+ from aura.common.config import get_settings
- from common.mathx import softmax
+ from aura.common.mathx import softmax
- from schemas.clinical import DIAGNOSES, Diagnosis, Finding
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis, Finding
- from knowledge.guidelines.templates import GUIDELINE_TEMPLATES
+ from aura.knowledge.guidelines.templates import GUIDELINE_TEMPLATES
- from services.recommend.engine import RecommendEngine, CATALOG, _COST_W, _RISK_W
+ from ..recommend.engine import RecommendEngine, CATALOG, _COST_W, _RISK_W
- from services.vision.xray_gate import _structural_score
+ from ..vision.xray_gate import _structural_score
- from services.fusion.evidence import encode
+ from ..fusion.evidence import encode
```
</details>

<details><summary><code>aura/services/safety/uncertainty.py</code> — 1 statement(s)</summary>

```diff
- from common.mathx import entropy
+ from aura.common.mathx import entropy
```
</details>

<details><summary><code>aura/services/vision/__init__.py</code> — 1 statement(s)</summary>

```diff
- from services.vision.engine import VisionEngine
+ from .engine import VisionEngine
```
</details>

<details><summary><code>aura/services/vision/cnn.py</code> — 2 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from schemas.clinical import Finding
+ from aura.schemas.clinical import Finding
```
</details>

<details><summary><code>aura/services/vision/engine.py</code> — 11 statement(s)</summary>

```diff
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from common.mathx import sigmoid
+ from aura.common.mathx import sigmoid
- from schemas.clinical import CHEST_FINDINGS, Finding
+ from aura.schemas.clinical import CHEST_FINDINGS, Finding
- from schemas.contracts import FindingScore, VisionResult
+ from aura.schemas.contracts import FindingScore, VisionResult
- from services.vision.features import FEATURE_NAMES, extract_features, feature_vector
+ from .features import FEATURE_NAMES, extract_features, feature_vector
- from ml.vision_cxr.model import DenseNet121CXR
+ from aura.ml.vision_cxr.model import DenseNet121CXR
- from schemas.clinical import FINDINGS
+ from aura.schemas.clinical import FINDINGS
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from ml.vision_cxr.inference import VisionModel
+ from aura.ml.vision_cxr.inference import VisionModel
- from common.config import get_settings
+ from aura.common.config import get_settings
- from services.vision.cnn import get_backbone
+ from .cnn import get_backbone
```
</details>

<details><summary><code>aura/services/vision/features.py</code> — 1 statement(s)</summary>

```diff
- from common.anatomy import IMG, REGIONS, _px, resize_to
+ from aura.common.anatomy import IMG, REGIONS, _px, resize_to
```
</details>

<details><summary><code>aura/services/vision/io.py</code> — 2 statement(s)</summary>

```diff
- from schemas.clinical import Modality
+ from aura.schemas.clinical import Modality
- from schemas.contracts import StructuredPriors, StudyInput
+ from aura.schemas.contracts import StructuredPriors, StudyInput
```
</details>

<details><summary><code>aura/services/vision/xray_gate.py</code> — 2 statement(s)</summary>

```diff
- from services.vision.io import load_dicom
+ from .io import load_dicom
- from services.vision.io import load_dicom
+ from .io import load_dicom
```
</details>

<details><summary><code>aura/tests/test_audit_repairs.py</code> — 21 statement(s)</summary>

```diff
- from schemas.clinical import DIAGNOSES, Diagnosis, Finding
+ from aura.schemas.clinical import DIAGNOSES, Diagnosis, Finding
- from schemas.contracts import (
+ from aura.schemas.contracts import (
- from services.safety import SafetyEngine
+ from aura.services.safety import SafetyEngine
- from services.fusion import FusionEngine
+ from aura.services.fusion import FusionEngine
- from services.safety import SafetyEngine
+ from aura.services.safety import SafetyEngine
- from services.fusion import FusionEngine
+ from aura.services.fusion import FusionEngine
- from services.safety.uncertainty import min_calibration_count
+ from aura.services.safety.uncertainty import min_calibration_count
- from services.safety.uncertainty import mondrian_qhats
+ from aura.services.safety.uncertainty import mondrian_qhats
- from services.safety.uncertainty import _quantile_hi
+ from aura.services.safety.uncertainty import _quantile_hi
- from services.safety import SafetyEngine
+ from aura.services.safety import SafetyEngine
- from services.fusion import FusionEngine
+ from aura.services.fusion import FusionEngine
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from gateway.storage import Store
+ from aura.gateway.storage import Store
- from services.report import ReportEngine
+ from aura.services.report import ReportEngine
- from services.report import ReportEngine
+ from aura.services.report import ReportEngine
- from services.recommend import RecommendEngine
+ from aura.services.recommend import RecommendEngine
- from services.fusion import FusionEngine
+ from aura.services.fusion import FusionEngine
- from gateway.security import read_capped
+ from aura.gateway.security import read_capped
- from gateway.security import validate_upload_name
+ from aura.gateway.security import validate_upload_name
- from gateway.security import RateLimiter
+ from aura.gateway.security import RateLimiter
- from services.vision.io import DEFAULT_GRID, _resize_grid
+ from aura.services.vision.io import DEFAULT_GRID, _resize_grid
```
</details>

<details><summary><code>aura/tests/test_backend_router.py</code> — 13 statement(s)</summary>

```diff
- from backend.core.router.detector import SignatureModalityDetector
+ from aura.backend.core.router.detector import SignatureModalityDetector
- from backend.core.router.router import ModalityRouter
+ from aura.backend.core.router.router import ModalityRouter
- from backend.core.shared.types import EngineStatus, ImagingModality
+ from aura.backend.core.shared.types import EngineStatus, ImagingModality
- from backend.core.upload.intake import stage_bytes
+ from aura.backend.core.upload.intake import stage_bytes
- from backend.engines.base.contract import (
+ from aura.backend.engines.base.contract import (
- from backend.engines.base.registry import EngineRegistry
+ from aura.backend.engines.base.registry import EngineRegistry
- from backend.models.routing import ResultStatus
+ from aura.backend.models.routing import ResultStatus
- from backend.services.dispatch import DispatchService
+ from aura.backend.services.dispatch import DispatchService
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
- from services.vision.xray_gate import validate_cxr
+ from aura.services.vision.xray_gate import validate_cxr
- from backend.engines.neuro.engine import NeuroMindEngine
+ from aura.backend.engines.neuro.engine import NeuroMindEngine
- from backend.core.shared.errors import EngineNotAvailable
+ from aura.backend.core.shared.errors import EngineNotAvailable
- from backend.api.routes import build_router
+ from aura.backend.api.routes import build_router
```
</details>

<details><summary><code>aura/tests/test_brain_vision.py</code> — 33 statement(s)</summary>

```diff
- from backend.vision.brain.augment import SliceAugmenter, affine, flip, rot90  # noqa: E402
+ from aura.backend.vision.brain.augment import SliceAugmenter, affine, flip, rot90  # noqa: E402
- from backend.vision.brain.config import (  # noqa: E402
+ from aura.backend.vision.brain.config import (  # noqa: E402
- from backend.vision.brain.dataset import (  # noqa: E402
+ from aura.backend.vision.brain.dataset import (  # noqa: E402
- from backend.vision.brain.degradations import DegradationSimulator  # noqa: E402
+ from aura.backend.vision.brain.degradations import DegradationSimulator  # noqa: E402
- from backend.vision.brain.embeddings import (  # noqa: E402
+ from aura.backend.vision.brain.embeddings import (  # noqa: E402
- from backend.vision.brain.errors import (  # noqa: E402
+ from aura.backend.vision.brain.errors import (  # noqa: E402
- from backend.vision.brain.ingest import (  # noqa: E402
+ from aura.backend.vision.brain.ingest import (  # noqa: E402
- from backend.vision.brain.io.brats_h5 import (  # noqa: E402
+ from aura.backend.vision.brain.io.brats_h5 import (  # noqa: E402
- from backend.vision.brain.losses import (  # noqa: E402
+ from aura.backend.vision.brain.losses import (  # noqa: E402
- from backend.vision.brain.metrics import (  # noqa: E402
+ from aura.backend.vision.brain.metrics import (  # noqa: E402
- from backend.vision.brain.model import (  # noqa: E402
+ from aura.backend.vision.brain.model import (  # noqa: E402
- from backend.vision.brain.output import BrainVisionOutput, build_regions  # noqa: E402
+ from aura.backend.vision.brain.output import BrainVisionOutput, build_regions  # noqa: E402
- from backend.vision.brain.sampling import (  # noqa: E402
+ from aura.backend.vision.brain.sampling import (  # noqa: E402
- from backend.vision.brain.types import (  # noqa: E402
+ from aura.backend.vision.brain.types import (  # noqa: E402
- from backend.vision.brain.types import DEFAULT_MODALITIES
+ from aura.backend.vision.brain.types import DEFAULT_MODALITIES
- from backend.vision.brain.output import (
+ from aura.backend.vision.brain.output import (
- from backend.vision.brain.types import EmbeddingSpec
+ from aura.backend.vision.brain.types import EmbeddingSpec
- from backend.vision.brain.output import build_regions as build
+ from aura.backend.vision.brain.output import build_regions as build
- from backend.vision.brain.checkpoint import CheckpointMeta
+ from aura.backend.vision.brain.checkpoint import CheckpointMeta
- from backend.vision.brain.inference import BrainVisionEngine
+ from aura.backend.vision.brain.inference import BrainVisionEngine
- from backend.vision.brain.output import QUALITY_VALIDITY_THRESHOLD
+ from aura.backend.vision.brain.output import QUALITY_VALIDITY_THRESHOLD
- from backend.vision.brain.output import (
+ from aura.backend.vision.brain.output import (
- from backend.vision.brain.types import EmbeddingSpec
+ from aura.backend.vision.brain.types import EmbeddingSpec
- from backend.vision.brain.types import EmbeddingSpec
+ from aura.backend.vision.brain.types import EmbeddingSpec
- from backend.vision.brain.cli import build_parser, config_from_args
+ from aura.backend.vision.brain.cli import build_parser, config_from_args
- from backend.vision.brain.cli import command_info
+ from aura.backend.vision.brain.cli import command_info
- from backend.vision.brain.inference import BrainVisionEngine
+ from aura.backend.vision.brain.inference import BrainVisionEngine
- from backend.vision.brain.train import BrainVisionTrainer
+ from aura.backend.vision.brain.train import BrainVisionTrainer
- from backend.vision.brain.inference import BrainVisionEngine
+ from aura.backend.vision.brain.inference import BrainVisionEngine
- from backend.vision.brain.ingest import BrainCorpusIngestor
+ from aura.backend.vision.brain.ingest import BrainCorpusIngestor
- from backend.vision.brain.train import BrainVisionTrainer
+ from aura.backend.vision.brain.train import BrainVisionTrainer
- from backend.vision.brain.train import BrainVisionTrainer
+ from aura.backend.vision.brain.train import BrainVisionTrainer
- from backend.vision.brain.checkpoint import (
+ from aura.backend.vision.brain.checkpoint import (
```
</details>

<details><summary><code>aura/tests/test_clinical_honesty.py</code> — 7 statement(s)</summary>

```diff
- from gateway.app import app
+ from aura.gateway.app import app
- from backend.core.shared.errors import UnsupportedModality, ModalityConflict
+ from aura.backend.core.shared.errors import UnsupportedModality, ModalityConflict
- from schemas.contracts import CaseBundle, CaseState, AbstentionReason
+ from aura.schemas.contracts import CaseBundle, CaseState, AbstentionReason
- from backend.services.reasoning.progression import LongitudinalAnalyzer
+ from aura.backend.services.reasoning.progression import LongitudinalAnalyzer
- from backend.services.reasoning.tracking import TumorTracker
+ from aura.backend.services.reasoning.tracking import TumorTracker
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
- from schemas.contracts import CaseState, SafetyAssessment, VisionResult
+ from aura.schemas.contracts import CaseState, SafetyAssessment, VisionResult
```
</details>

<details><summary><code>aura/tests/test_mimic_cleaning.py</code> — 4 statement(s)</summary>

```diff
- from mimic.cleaning import CleanedPatient, _clean_text, _dedup, clean_record
+ from aura.mimic.cleaning import CleanedPatient, _clean_text, _dedup, clean_record
- from mimic.labeling import label_report, label_patient_reports
+ from aura.mimic.labeling import label_report, label_patient_reports
- from mimic.loaders import PatientRecord
+ from aura.mimic.loaders import PatientRecord
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
```
</details>

<details><summary><code>aura/tests/test_mimic_end_to_end.py</code> — 7 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.tasks import TaskDatasetBuilder
+ from aura.mimic.tasks import TaskDatasetBuilder
- from mimic.training import GBMTrainer
+ from aura.mimic.training import GBMTrainer
- from mimic.evaluation import evaluate_multiclass
+ from aura.mimic.evaluation import evaluate_multiclass
- from mimic.tasks import TaskDatasetBuilder
+ from aura.mimic.tasks import TaskDatasetBuilder
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from mimic.patient import iter_patients
+ from aura.mimic.patient import iter_patients
```
</details>

<details><summary><code>aura/tests/test_mimic_evaluation.py</code> — 1 statement(s)</summary>

```diff
- from mimic.evaluation import (
+ from aura.mimic.evaluation import (
```
</details>

<details><summary><code>aura/tests/test_mimic_explain.py</code> — 1 statement(s)</summary>

```diff
- from mimic.explain import (
+ from aura.mimic.explain import (
```
</details>

<details><summary><code>aura/tests/test_mimic_features.py</code> — 2 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.features import (
+ from aura.mimic.features import (
```
</details>

<details><summary><code>aura/tests/test_mimic_integration.py</code> — 9 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- import gateway.app as app
+ import aura.gateway.app as app
- from mimic.seed import seed_mimic
+ from aura.mimic.seed import seed_mimic
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from gateway.storage import Store
+ from aura.gateway.storage import Store
- from mimic.seed import seed_mimic
+ from aura.mimic.seed import seed_mimic
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from gateway.storage import Store
+ from aura.gateway.storage import Store
- from mimic.seed import seed_mimic
+ from aura.mimic.seed import seed_mimic
```
</details>

<details><summary><code>aura/tests/test_mimic_loaders.py</code> — 3 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.loaders import MimicCxrLoader, PatientRecord, SchemaError
+ from aura.mimic.loaders import MimicCxrLoader, PatientRecord, SchemaError
- from mimic.parsing import safe_list, safe_str_list
+ from aura.mimic.parsing import safe_list, safe_str_list
```
</details>

<details><summary><code>aura/tests/test_mimic_patient.py</code> — 5 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.loaders import MimicCxrLoader, PatientRecord
+ from aura.mimic.loaders import MimicCxrLoader, PatientRecord
- from mimic.patient import Patient, build_patient, iter_patients
+ from aura.mimic.patient import Patient, build_patient, iter_patients
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
- from schemas.contracts import StructuredPriors, StudyInput
+ from aura.schemas.contracts import StructuredPriors, StudyInput
```
</details>

<details><summary><code>aura/tests/test_mimic_performance.py</code> — 3 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.performance import MemmapCache, ParallelFeatureEngineer, gpu_standardize
+ from aura.mimic.performance import MemmapCache, ParallelFeatureEngineer, gpu_standardize
- from mimic.features import feature_names
+ from aura.mimic.features import feature_names
```
</details>

<details><summary><code>aura/tests/test_mimic_splits.py</code> — 2 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.splits import (
+ from aura.mimic.splits import (
```
</details>

<details><summary><code>aura/tests/test_mimic_tasks.py</code> — 2 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.tasks import (
+ from aura.mimic.tasks import (
```
</details>

<details><summary><code>aura/tests/test_mimic_timeline.py</code> — 4 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.loaders import MimicCxrLoader, PatientRecord
+ from aura.mimic.loaders import MimicCxrLoader, PatientRecord
- from mimic.timeline import StudyEvent, build_timeline, _study_of
+ from aura.mimic.timeline import StudyEvent, build_timeline, _study_of
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
```
</details>

<details><summary><code>aura/tests/test_mimic_training.py</code> — 2 statement(s)</summary>

```diff
- from mimic.tasks import TaskDataset, TaskSpec
+ from aura.mimic.tasks import TaskDataset, TaskSpec
- from mimic.training import (
+ from aura.mimic.training import (
```
</details>

<details><summary><code>aura/tests/test_mimic_uncertainty.py</code> — 1 statement(s)</summary>

```diff
- from mimic.uncertainty import (
+ from aura.mimic.uncertainty import (
```
</details>

<details><summary><code>aura/tests/test_mri_foundation.py</code> — 23 statement(s)</summary>

```diff
- from backend.foundation.mri import (
+ from aura.backend.foundation.mri import (
- from backend.foundation.mri.config import StandardizationConfig
+ from aura.backend.foundation.mri.config import StandardizationConfig
- from backend.foundation.mri.errors import (
+ from aura.backend.foundation.mri.errors import (
- from backend.foundation.mri.geometry import (
+ from aura.backend.foundation.mri.geometry import (
- from backend.foundation.mri.io.nifti_reader import NiftiReader
+ from aura.backend.foundation.mri.io.nifti_reader import NiftiReader
- from backend.foundation.mri.io.nrrd_reader import NrrdReader
+ from aura.backend.foundation.mri.io.nrrd_reader import NrrdReader
- from backend.foundation.mri.masking import BrainMaskSlot, estimate_foreground_mask
+ from aura.backend.foundation.mri.masking import BrainMaskSlot, estimate_foreground_mask
- from backend.foundation.mri.metadata import (
+ from aura.backend.foundation.mri.metadata import (
- from backend.foundation.mri.quality import _check
+ from aura.backend.foundation.mri.quality import _check
- from backend.foundation.mri.standardize import (
+ from aura.backend.foundation.mri.standardize import (
- from backend.foundation.mri.types import (
+ from aura.backend.foundation.mri.types import (
- from backend.foundation.mri.volume import MRIVolume
+ from aura.backend.foundation.mri.volume import MRIVolume
- from backend.foundation.mri.io.nifti_reader import _NIFTI1_DTYPE
+ from aura.backend.foundation.mri.io.nifti_reader import _NIFTI1_DTYPE
- from backend.foundation.mri.io.nifti_reader import _NIFTI2_DTYPE
+ from aura.backend.foundation.mri.io.nifti_reader import _NIFTI2_DTYPE
- from backend.foundation.mri.io.base import RawSeries
+ from aura.backend.foundation.mri.io.base import RawSeries
- from backend.foundation.mri.errors import StageUnavailable
+ from aura.backend.foundation.mri.errors import StageUnavailable
- from backend.foundation.mri.errors import StageUnavailable
+ from aura.backend.foundation.mri.errors import StageUnavailable
- from backend.core.upload.intake import stage_bytes
+ from aura.backend.core.upload.intake import stage_bytes
- from backend.engines.neuro.engine import NeuroMindEngine
+ from aura.backend.engines.neuro.engine import NeuroMindEngine
- from backend.foundation.mri.study import FoundationStudy
+ from aura.backend.foundation.mri.study import FoundationStudy
- from backend.engines.neuro.engine import NeuroMindEngine
+ from aura.backend.engines.neuro.engine import NeuroMindEngine
- from backend.engines.neuro.engine import NeuroMindEngine
+ from aura.backend.engines.neuro.engine import NeuroMindEngine
- from backend.engines.neuro.engine import NeuroMindEngine
+ from aura.backend.engines.neuro.engine import NeuroMindEngine
```
</details>

<details><summary><code>aura/tests/test_mri_intake_manager.py</code> — 9 statement(s)</summary>

```diff
- from backend.foundation.mri.intake_manager import MRIIntakeManager
+ from aura.backend.foundation.mri.intake_manager import MRIIntakeManager
- from backend.foundation.mri.errors import StudyValidationError
+ from aura.backend.foundation.mri.errors import StudyValidationError
- from backend.foundation.mri.types import SequenceType
+ from aura.backend.foundation.mri.types import SequenceType
- from backend.engines.neuro.multisequence import MultiSequenceStudy
+ from aura.backend.engines.neuro.multisequence import MultiSequenceStudy
- from tests.test_mri_foundation import write_nifti1, ras_affine, head_phantom
+ from .test_mri_foundation import write_nifti1, ras_affine, head_phantom
- from gateway.security import validate_mri_content
+ from aura.gateway.security import validate_mri_content
- from gateway.security import validate_mri_content
+ from aura.gateway.security import validate_mri_content
- from gateway.security import validate_mri_content
+ from aura.gateway.security import validate_mri_content
- from gateway.security import validate_mri_content
+ from aura.gateway.security import validate_mri_content
```
</details>

<details><summary><code>aura/tests/test_neuroinsight.py</code> — 2 statement(s)</summary>

```diff
- from backend.engines.neuro.neuroinsight import (
+ from aura.backend.engines.neuro.neuroinsight import (
- from backend.vision.brain.output import (
+ from aura.backend.vision.brain.output import (
```
</details>

<details><summary><code>aura/tests/test_neuroview.py</code> — 4 statement(s)</summary>

```diff
- from backend.engines.neuro.multisequence import MultiSequenceStudy
+ from aura.backend.engines.neuro.multisequence import MultiSequenceStudy
- from backend.engines.neuro.neuroview import build_neuroview_payload
+ from aura.backend.engines.neuro.neuroview import build_neuroview_payload
- from backend.vision.brain.output import (
+ from aura.backend.vision.brain.output import (
- from backend.vision.brain.types import DEFAULT_MODALITIES
+ from aura.backend.vision.brain.types import DEFAULT_MODALITIES
```
</details>

<details><summary><code>aura/tests/test_production_agent.py</code> — 5 statement(s)</summary>

```diff
- from gateway.app import app
+ from aura.gateway.app import app
- from ml.data import make_sample
+ from aura.ml.data import make_sample
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
- from services.fusion.engine import FusionEngine
+ from aura.services.fusion.engine import FusionEngine
- from services.agent.active_diagnosis import ActiveDiagnosisAgent
+ from aura.services.agent.active_diagnosis import ActiveDiagnosisAgent
```
</details>

<details><summary><code>aura/tests/test_production_eval.py</code> — 7 statement(s)</summary>

```diff
- from mimic.config import get_mimic_paths
+ from aura.mimic.config import get_mimic_paths
- from mimic.parsing import safe_str_list
+ from aura.mimic.parsing import safe_str_list
- from ml.evaluation.clinical_eval import binary_ece
+ from aura.ml.evaluation.clinical_eval import binary_ece
- from ml.evaluation.vision_calibration import conformal_evaluation
+ from aura.ml.evaluation.vision_calibration import conformal_evaluation
- from ml.evaluation.clinical_eval import evaluate_validation
+ from aura.ml.evaluation.clinical_eval import evaluate_validation
- from ml.evaluation.vision_calibration import run_calibration, SERVING_CAL_PATH
+ from aura.ml.evaluation.vision_calibration import run_calibration, SERVING_CAL_PATH
- from ml.evaluation.perf_benchmark import run
+ from aura.ml.evaluation.perf_benchmark import run
```
</details>

<details><summary><code>aura/tests/test_production_explain.py</code> — 5 statement(s)</summary>

```diff
- from schemas.clinical import Finding
+ from aura.schemas.clinical import Finding
- from services.explain import overlays as O
+ from aura.services.explain import overlays as O
- from common.config import ARTIFACTS
+ from aura.common.config import ARTIFACTS
- from ml.vision_cxr.inference import VisionModel
+ from aura.ml.vision_cxr.inference import VisionModel
- from services.explain import methods as M
+ from aura.services.explain import methods as M
```
</details>

<details><summary><code>aura/tests/test_production_predict.py</code> — 4 statement(s)</summary>

```diff
- from services.inference.predict import predict_image
+ from aura.services.inference.predict import predict_image
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from ml.data import make_sample
+ from aura.ml.data import make_sample
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
```
</details>

<details><summary><code>aura/tests/test_production_report.py</code> — 5 statement(s)</summary>

```diff
- from ml.data import IMG, make_multimodal, make_sample
+ from aura.ml.data import IMG, make_multimodal, make_sample
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
- from schemas.contracts import StudyInput
+ from aura.schemas.contracts import StudyInput
- from services.report.clinical_report import (
+ from aura.services.report.clinical_report import (
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
```
</details>

<details><summary><code>aura/tests/test_quantum_integration.py</code> — 5 statement(s)</summary>

```diff
- from backend.engines.neuro.qkl import QKLClassifier
+ from aura.backend.engines.neuro.qkl import QKLClassifier
- from schemas.clinical import Diagnosis
+ from aura.schemas.clinical import Diagnosis
- from services.fusion.multimodal import UnifiedFusionEngine
+ from aura.services.fusion.multimodal import UnifiedFusionEngine
- from services.fusion.qae import QuantumAutoencoder
+ from aura.services.fusion.qae import QuantumAutoencoder
- from services.reasoning.qbn import QuantumBayesianNetwork
+ from aura.services.reasoning.qbn import QuantumBayesianNetwork
```
</details>

<details><summary><code>aura/tests/test_quantum_measurement.py</code> — 8 statement(s)</summary>

```diff
- from services.fusion.device import make_probs_qnode, make_qnode          # noqa: E402
+ from aura.services.fusion.device import make_probs_qnode, make_qnode          # noqa: E402
- from services.fusion.qmba import (                                       # noqa: E402
+ from aura.services.fusion.qmba import (                                       # noqa: E402
- from services.fusion.qmeasure import (                                   # noqa: E402
+ from aura.services.fusion.qmeasure import (                                   # noqa: E402
- from services.fusion.quantum import QuantumFusion                        # noqa: E402
+ from aura.services.fusion.quantum import QuantumFusion                        # noqa: E402
- from schemas.clinical import DIAGNOSES                                   # noqa: E402
+ from aura.schemas.clinical import DIAGNOSES                                   # noqa: E402
- from services.safety import calibration as calibration_module
+ from aura.services.safety import calibration as calibration_module
- from services.safety import calibration as calibration_module
+ from aura.services.safety import calibration as calibration_module
- from services.safety import calibration as calibration_module
+ from aura.services.safety import calibration as calibration_module
```
</details>

<details><summary><code>aura/tests/test_safety_architecture.py</code> — 5 statement(s)</summary>

```diff
- from schemas.clinical import Diagnosis, Finding
+ from aura.schemas.clinical import Diagnosis, Finding
- from schemas.contracts import CaseBundle, CaseState
+ from aura.schemas.contracts import CaseBundle, CaseState
- from services.safety.controller import ClinicalSafetyController, ClinicalSafetyException
+ from aura.services.safety.controller import ClinicalSafetyController, ClinicalSafetyException
- from services.safety.readiness import ClinicalDecisionReadinessEngine
+ from aura.services.safety.readiness import ClinicalDecisionReadinessEngine
- from gateway.storage import compute_provenance_hash
+ from aura.gateway.storage import compute_provenance_hash
```
</details>

<details><summary><code>aura/tests/test_startup_regression.py</code> — 1 statement(s)</summary>

```diff
- from services.vision.engine import VisionEngine
+ from aura.services.vision.engine import VisionEngine
```
</details>

<details><summary><code>demo.py</code> — 8 statement(s)</summary>

```diff
- from backend.core.upload.intake import stage_bytes
+ from aura.backend.core.upload.intake import stage_bytes
- from backend.services.dispatch import DispatchService
+ from aura.backend.services.dispatch import DispatchService
- from backend.models.routing import ResultStatus
+ from aura.backend.models.routing import ResultStatus
- from backend.core.shared.errors import UnsupportedModality, ModalityConflict
+ from aura.backend.core.shared.errors import UnsupportedModality, ModalityConflict
- from backend.bootstrap import install_router
+ from aura.backend.bootstrap import install_router
- from gateway.pipeline import Pipeline
+ from aura.gateway.pipeline import Pipeline
- from gateway.storage import Store
+ from aura.gateway.storage import Store
- from common.config import DB_PATH
+ from aura.common.config import DB_PATH
```
</details>

<details><summary><code>run_failure_demo.py</code> — 6 statement(s)</summary>

```diff
- from backend.core.upload.intake import stage_bytes
+ from aura.backend.core.upload.intake import stage_bytes
- from backend.core.router.router import ModalityRouter
+ from aura.backend.core.router.router import ModalityRouter
- from backend.engines.neuro.engine import NeuroMindEngine
+ from aura.backend.engines.neuro.engine import NeuroMindEngine
- from backend.core.shared.errors import UnsupportedModality, ModalityConflict, AuraBackendError
+ from aura.backend.core.shared.errors import UnsupportedModality, ModalityConflict, AuraBackendError
- from backend.models.routing import ResultStatus
+ from aura.backend.models.routing import ResultStatus
- from backend.core.upload.intake import stage_bytes
+ from aura.backend.core.upload.intake import stage_bytes
```
</details>
