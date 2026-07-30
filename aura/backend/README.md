# AURA Intelligent Modality Router

The routing layer that turns AURA from a chest-X-ray product into a modular Medical AI
Operating System. An upload arrives, the router works out what kind of study it is, and
dispatches it to the engine that serves that modality.

Today: **Chest X-ray → AURA Thorax** (live) and **Brain MRI → AURA NeuroMind** (live —
BraTS2020 segmentation network, epoch 24, composite Dice 0.875, with a Platt-calibrated
presence head). NeuroMind requires a *volumetric* MR study with all four sequences and
does not classify tumour subtype; see §8. Adding CT, mammography, retina, or ultrasound
is an additive change: one engine class and one registry entry.

---

## 1. Where it lives, and why it isn't a rewrite

The requested layout is implemented as a `backend/` package inside the existing source
root (`aura/`), sitting alongside the untouched `gateway/`, `services/`, `schemas/`, and
`ml/` packages:

```
aura/
├── backend/                  ← NEW: the routing architecture
│   ├── core/
│   │   ├── router/           modality detection + engine selection
│   │   ├── upload/           intake: allowlist, size cap, staging, cleanup
│   │   └── shared/           logging, error taxonomy, primitive types
│   ├── engines/
│   │   ├── base/             abstract contract + plug-in registry
│   │   ├── thorax/           adapter over the existing pipeline (nothing rewritten)
│   │   └── neuro/            NeuroMind — trained brain MRI engine
│   ├── services/             orchestration (intake → route → dispatch → envelope)
│   ├── api/                  FastAPI routes mounted onto the existing gateway
│   ├── models/               wire contracts (pydantic)
│   └── bootstrap.py          the single integration point
│
├── gateway/                  UNCHANGED except 8 lines in the lifespan hook
├── services/                 UNCHANGED — the chest-X-ray intelligence
├── schemas/                  UNCHANGED
└── ml/                       UNCHANGED
```

**Why not physically move `gateway/` → `backend/api/` and `services/` → `backend/services/`?**
Because requirement 4 (*"do NOT change the existing Thorax engine"*) and a wholesale
directory move are mutually exclusive. The chest-X-ray stack is an audited, calibrated
system whose modules import each other by top-level name (`from services.vision import …`)
in several hundred places, and whose artifact paths are resolved relative to the package
root. Relocating it would mean touching every one of those imports — a large, untestable
diff through validated clinical code, delivering no behavioural benefit. The structure the
request asks for is what matters, and it is what `backend/` provides; the legacy packages
keep their paths and their audit trail. If a physical consolidation is wanted later, the
right order is: move the chest stack *behind* the `ThoraxEngine` adapter first (it is
already the only thing that touches it), then relocate, with the router's tests as the
safety net.

### The complete integration diff

Everything outside `backend/` amounts to one block in `gateway/app.py`'s lifespan:

```python
from backend.bootstrap import install_router
state["dispatch"] = install_router(
    app, pipeline, store, on_case_created=session_case_ids.append
)
```

wrapped in a `try/except` so a router failure can never stop the chest-X-ray service from
starting. Plus one line in `pyproject.toml` adding `backend` to `testpaths`. Nothing else.

---

## 2. Request flow

```
POST /v1/studies/analyze
        │
        ▼
  core/upload/intake.py ──── allowlist + size cap (reuses gateway.security)
        │                    stage to temp · SHA-256 · guaranteed cleanup
        ▼
  core/router/features.py ── decode ONCE → DICOM tags + pixel geometry
        │
        ▼
  core/router/signatures.py  every signature scores the same fingerprint
        │                    Chest · BrainMRI · HeadCT · DicomModality
        ▼
  core/router/detector.py ── rank · commit thresholds · review flag
        │
        ▼
  core/router/router.py ──── registry lookup → RoutingMetadata
        │                    {modality, selected_engine, confidence, supported, reason}
        ▼
  services/dispatch.py ───── resolve engine (lazy construct) → engine.run(asset)
        │
        ▼
  engines/<engine>/ ──────── validate_input → preprocess → analyze → generate_report
        │
        ▼
  AnalysisEnvelope {request_id, routing, result, timings_ms}
```

One correlation id threads the whole path and appears on every log line, so `grep <id>`
reconstructs a complete routing decision.

---

## 3. The engine contract

Every engine implements four methods (`engines/base/contract.py`):

| Method | Returns | Contract |
|---|---|---|
| `validate_input(asset)` | `ValidationOutcome` | The engine's *own* opinion on whether it should look at this image. Must not raise. |
| `preprocess(asset)` | `PreparedStudy` | Bytes → model-ready study. Where fidelity is won or lost. |
| `analyze(prepared)` | `AnalysisResult` | The only stage allowed to be expensive or stochastic. `async`. |
| `generate_report(result)` | `EngineReport` | **Pure function of the result** — no model calls, no new evidence. That constraint is what makes a report grounded. |

`AnalysisEngine.run()` is a template method that calls all four in order and applies one
uniform error policy, so no engine reimplements it. Engines inherit it untouched.

`validate_input` deliberately re-checks what the router already decided. The router decides
*where* an image goes; the engine decides whether *this model* is willing to look at it.
That means swapping the detector — for a learned classifier, say — can never expose the
chest model to a knee film.

### Adding an engine

```python
@register_engine
class RetinaEngine(AnalysisEngine):
    descriptor = EngineDescriptor(
        engine_id="retina", display_name="AURA Retina", version="1.0.0",
        modalities=(ImagingModality.RETINAL_FUNDUS,),
        status=EngineStatus.AVAILABLE,
    )
    def validate_input(self, asset): ...
    def preprocess(self, asset): ...
    async def analyze(self, prepared): ...
    def generate_report(self, result): ...
```

No router, API, or service change. Out-of-tree engines need no repo change at all — they
advertise an entry point and the registry discovers them at startup:

```toml
[project.entry-points."aura.engines"]
retina = "aura_retina.engine:RetinaEngine"
```

Registration stores a **factory**, not an instance: engines load model weights in their
constructor, so a registered-but-never-used engine costs nothing. A factory that raises is
contained — that engine is marked `unavailable` and every other route keeps working.

---

## 4. How modality detection actually works, and what it can't do

**All thresholds below were measured, not assumed.** The corpora: 600 random real
MIMIC-CXR films for the chest side, and the real MR/CR/CT/US DICOMs bundled with pydicom
for the rest.

### Two evidence channels, with precedence

**DICOM header** — `Modality` and `BodyPartExamined` are written by the acquisition
device. When present they are near-definitional and no pixel heuristic overrules them.

**Pixel geometry** — for PNG/JPEG exports, which carry no header.

### Measured feature distributions (n=600 real MIMIC-CXR films)

| Feature | min | p1 | p5 | p50 | p95 |
|---|---|---|---|---|---|
| `edge_max` (brightest border strip) | 0.007 | 0.116 | 0.290 | 0.722 | 0.933 |
| `foreground_bbox_fraction` | 0.573 | 0.672 | 0.768 | 0.977 | 1.000 |
| `background_fraction` | 0.016 | 0.042 | 0.080 | 0.123 | 0.327 |
| `corner_dark_fraction` | — | 0.055 | 0.134 | **0.501** | 0.834 |

Two findings from that measurement shaped the design:

**1. Dark corners are not a head-imaging signal.** Half of all real chest films already
have corners below 0.10 from collimation (median `corner_dark_fraction` = 0.50). The
intuitive "brain MRI has black corners" rule would have misfired constantly. It was
discarded on the evidence.

**2. Axial head CT and axial head MRI are pixel-wise indistinguishable.** Both put a
compact convex mass in signal-free air. Measured on the real head-CT DICOM:
`background_fraction` 0.62, `bbox_fraction` 0.71, `edge_max` 0.002 — the same signature a
brain MRI produces. Pixels can identify a *cross-sectional head study*; they cannot tell
you which scanner made it. That is physics, not a shortcut, and it is why the pixel-only
brain-MRI path is capped and flagged.

### Confidence is an ordinal score, not a probability

The `confidence` field is an **ordinal decision score in [0,1], not a calibrated
posterior**. Calling it a probability would require a labelled multi-modality corpus and a
fitted classifier; AURA has one very large chest corpus and a handful of other-modality
DICOMs. Every score therefore carries a `calibrated` flag:

| Path | Score | `calibrated` | Basis |
|---|---|---|---|
| DICOM header names chest radiograph | 0.97 | ✅ | Acquisition tags |
| DICOM header names MR + head/brain | 0.97 | ✅ | Acquisition tags |
| Header (no region) + pixel confirmation | 0.90–0.93 | ✅ | Tags + measured geometry |
| Pixel-only chest (PNG/JPEG) | 0.88 | ✅ | Validated CXR gate, 261k-film corpus |
| **Pixel-only brain MRI (PNG/JPEG)** | **≤ 0.70** | ❌ | **See below** |
| MR header with no region, no head geometry | 0.45 | ✅ | Below commit threshold — declines |

**The uncalibrated path, stated plainly:** no brain-MRI image corpus was available in this
environment to fit or validate a pixel-only MR separator. That path is capped at 0.70,
marked `calibrated: false`, sets `requires_review: true`, and its `reason` names the MR/CT
ambiguity explicitly. It is the known limitation to close when NeuroMind becomes a real
engine; the fix is a learned modality classifier dropped in behind the `ModalityDetector`
protocol, which requires no other change.

### Precedence enforced by the numbers, not by branching

The uncalibrated cap (0.70) sits deliberately below the chest signature's pixel-only score
(0.88). So when a chest film happens to also satisfy the head geometry — measured at
**0.67% of real MIMIC films** — plain argmax routes it to Thorax, because the validated
signature outscores the provisional one by construction. Keeping precedence in the
confidence scale rather than in `if` statements means a new signature automatically obeys
the same rule.

### Measured routing behaviour

`tests/test_backend_router.py::test_batch_real_films_route_to_thorax_and_never_to_neuromind`
re-measures this on every run:

```
routed 200 real chest films: {'thorax': 199, 'none': 1}  →  99.5% to Thorax
                                                             0 misrouted to NeuroMind
```

The one `none` is a film the pre-existing CXR gate itself rejects — unchanged gate
behaviour, not a routing regression.

### The router declines rather than guesses

Two distinct refusals, mirroring the safety engine's abstention posture:

* **`modality_undetermined`** (422) — nothing scored above the 0.55 commit threshold, or
  the top two are within 0.10 of each other. Genuine ambiguity is not resolved by guessing.
* **`supported: false`** (200) — the study *was* identified (head CT, ultrasound,
  mammogram) but no engine claims it. The response names what it is and lists what AURA
  does serve. This is why `HeadCTSignature` exists at all: without it, an axial head CT
  would match the brain geometry test and be misrouted to NeuroMind.

---

## 5. API

| Endpoint | Purpose |
|---|---|
| `POST /v1/studies/route` | Identify modality + engine. No analysis, no model load. Returns every scored candidate with its evidence — the endpoint to reach for when diagnosing a rejected upload. |
| `POST /v1/studies/analyze` | Route and analyse. Modality-agnostic upload. |
| `GET /v1/engines` | Registered engines, served modalities, detector configuration. |
| `POST /v1/studies/upload` | **Legacy chest-X-ray endpoint — unchanged.** |

`200` from `/analyze` does not by itself mean a clinical result exists. Check
`result.status`:

* `completed` — analysed; `payload` and `report` are real
* `not_implemented` — routed correctly to an engine still being built
* `unsupported` — modality identified, no engine serves it
* `failed` — the engine rejected or could not analyse the study

Unsupported and not-implemented return `200` on purpose: the routing metadata is the
valuable part of those responses, and an HTTP error body would bury it.

### Example — chest radiograph

```json
{
  "modality": "chest_xray",
  "selected_engine": "thorax",
  "confidence": 0.88,
  "supported": true,
  "reason": "chest anatomy confirmed by the validated chest-radiograph intake gate; dispatching to AURA Thorax",
  "calibrated": true,
  "requires_review": false,
  "engine_status": "available",
  "candidates": [ ... every modality scored, with evidence ... ]
}
```

### Example — brain MRI (placeholder engine)

```json
{
  "routing": {
    "modality": "brain_mri",
    "selected_engine": "neuromind",
    "confidence": 0.97,
    "supported": true,
    "reason": "DICOM header declares an MR study of the head/brain (matched 'HEAD'); dispatching to AURA NeuroMind"
  },
  "result": {
    "status": "not_implemented",
    "engine": "neuromind",
    "message": "AURA NeuroMind is not yet implemented. The study was correctly identified as a brain MRI and routed to this engine, but no brain MRI model is available, so no clinical interpretation was produced.",
    "payload": {"routing_verified": true, "planned_capabilities": [...]},
    "report": null,
    "case_id": null
  }
}
```

Note `report: null` and `case_id: null`. The placeholder returns **no** clinical payload.
Returning an empty finding list or a stub result would have been less code and by far the
most dangerous thing it could do — a result that *looks* clinical but is grounded in no
model is precisely what the rest of AURA's safety machinery exists to prevent.

---

## 6. Compatibility with the existing workflow

Verified end-to-end against the real gateway with real model weights:

| Check | Result |
|---|---|
| Gateway starts with router installed | ✅ `/v1/health` ok, fusion backend `classical` |
| `POST /v1/studies/upload` (legacy) | ✅ unchanged — `CASE-UPLOAD-25`, 4.03 s |
| `POST /v1/studies/analyze` (new) | ✅ routed to Thorax — `CASE-UPLOAD-26`, real diagnosis + 5-section grounded report |
| Routed case via `GET /v1/cases/{id}` | ✅ identical shape to a legacy case |
| Console worklist `GET /v1/studies` | ✅ holds both cases |
| Brain MRI DICOM | ✅ → NeuroMind (routing verified at the time of this table, when the engine was a placeholder; re-verify against the trained engine) |
| Undecodable upload | ✅ 422 `modality_undetermined` with all candidates |
| Full existing test suite | ✅ 151 passed, 1 skipped, 0 failed |

A case created through the router is indistinguishable from a legacy one — same
`CASE-UPLOAD-{n}` scheme, same persistence, same audit rows, same inference log — so the
console, history portal, and report signing all keep working untouched.

---

## 7. Operations

**Logging.** `AURA_LOG_LEVEL` (default `INFO`), `AURA_LOG_JSON=1` for one JSON object per
line. Every line carries the request correlation id.

**Error taxonomy** (`core/shared/errors.py`). Clients switch on the stable `error` code;
internal exception text never leaves the server, per the existing audit rules (§10.9,
§11.5).

| Code | HTTP | Meaning |
|---|---|---|
| `upload_rejected` | 415 / 413 | Disallowed type or oversized |
| `unreadable_image` | 422 | Decoders could not read it |
| `modality_undetermined` | 422 | Nothing scored high enough to commit |
| `unsupported_modality` | 422 | Identified, but no engine serves it |
| `engine_unavailable` | 503 | Engine registered but could not be constructed |
| `engine_not_implemented` | 501 | Declared placeholder (rendered as a 200 result) |
| `engine_failed` | 500 | Engine ran and failed |

**Failure containment.** A broken signature is excluded from one decision; a broken engine
is marked `unavailable`; a broken plug-in is skipped at discovery; a broken router does not
stop the gateway from starting. No single new component can take the chest-X-ray service
down.

**Cleanup.** Staged uploads are removed on every path including errors — the context
manager is the only supported entry point precisely so this cannot be forgotten.

---

## 8. Known limitations

1. **Pixel-only brain MRI detection is uncalibrated.** No brain-MRI corpus was available
   *when the detector was built*; BraTS2020 is now present, so this is closable by fitting
   a learned classifier behind the `ModalityDetector` protocol — it has not been done.
   Capped at 0.70, flagged `calibrated: false`, `requires_review: true`. Note the practical
   consequence has shrunk: a pixel-only PNG/JPEG that routes here is then **refused by
   NeuroMind itself**, which requires a volumetric four-sequence study, so an uncalibrated
   pixel score can no longer produce a clinical result — only a slower refusal.
2. **Pixel-only head CT vs brain MRI cannot be separated.** Physics, not implementation.
   Head CT is caught on the header channel; a *headerless* head-CT PNG would score as
   brain MRI at capped confidence with the ambiguity named in its `reason`.
3. **`confidence` is ordinal, not a calibrated posterior.** See §4.
4. **One engine per modality.** Collisions are resolved first-registered-wins and logged.
   Multi-engine fan-out (e.g. two opinions on one chest film) is not supported yet.
5. **NeuroMind is trained on gliomas only.** BraTS2020 contains no meningioma, metastasis,
   abscess or demyelinating lesion, so subtype is not classified and the subtype surface
   abstains. The QKL head is trained on glioma *grade* (HGG/LGG) — test AUROC 0.706 — which
   is the only tumour label axis in AURA's corpus.
6. **Brain MRI is the only MR route.** An MR study of a named non-head region (knee, spine,
   abdomen) is scored `REJECT` by `BrainMRISignature` and answered as unsupported. That is
   correct behaviour, not a gap to patch by loosening the signature.

---

## 9. Tests

```bash
python -m pytest tests/test_backend_router.py -m "not slow"   # fast: contract, routing, API
python -m pytest tests/test_backend_router.py -m slow -s      # batch validation, real films
```

Tests that need real images **skip** when the corpus is absent rather than substituting
synthetic phantoms. A phantom proves nothing about a detector whose entire job is telling
real acquisitions apart.
