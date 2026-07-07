# AURA — Clinical Intelligence Copilot

> *Adaptive Uncertainty-aware Reasoning Assistant* — "the copilot that knows what it doesn't know."

AURA is a clinical **epistemic** intelligence platform. Unlike disease classifiers, its primary
outputs are **calibrated uncertainty, missing-evidence gaps, and the next best diagnostic step**.
A diagnosis probability is one field among many — not the product.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full product & system design.

## What runs here (P0 hackathon cut)

An end-to-end, **fully offline** pipeline over chest-X-ray-style studies:

```
study → vision engine → quantum evidence fusion → safety/uncertainty
      → explainability → missing-evidence recommender → grounded report
      → doctor dashboard → clinician feedback → learning loop
```

Every engine is an independently replaceable package under `services/`, all speaking the
shared Pydantic contracts in `packages/schemas`. The gateway orchestrates them over an
in-process async event bus (swappable for Redis Streams in production).

### Highlights that are *actually real* (not stubbed)

- **Quantum evidence fusion** — a variational quantum circuit (PennyLane statevector) that
  angle-encodes an 8-feature evidence vector and uses entangling layers to model higher-order
  evidence interactions. Ships beside a classical Bayesian product-of-experts fusion; a
  built-in benchmark compares the two head-to-head.
- **Confidence & safety** — deep-ensemble + MC-dropout epistemic variance, temperature
  scaling, **conformal prediction sets** with a coverage guarantee, energy-score OOD
  detection, and an explicit **abstention** policy (no silent failures).
- **Missing-evidence engine** — ranks candidate next tests by **expected information gain**
  (entropy reduction) per cost/risk, computed through the fusion posterior.
- **Explainability** — model-agnostic **occlusion saliency** over the image plus Shapley-style
  attribution over evidence nodes and counterfactual sensitivity.

## Quick start

```bash
cd aura
py -m pip install -r requirements.txt          # Windows: py   |  else: python
py -m aura_cli train                           # fit fusion models + calibration (seconds)
py -m aura_cli bench                            # quantum vs classical fusion benchmark
py -m aura_cli serve                            # start gateway + dashboard on :8000
```

Then open http://localhost:8000 — the doctor dashboard. Seed cases are generated on first run.

`run.bat` (Windows) / `run.sh` do all of the above in one shot.

## Layout

```
packages/schemas   shared Pydantic contracts (single source of truth)
services/           vision · fusion · safety · explain · recommend · report · memory · models
gateway/            FastAPI app: /v1 API, pipeline orchestration, SQLite persistence, audit
apps/web/           doctor dashboard (static SPA)
ml/                 synthetic data generator, training, quantum-vs-classical evaluation
docs/               architecture, ADRs
```

Not medical advice. Research / decision-support only; a clinician is always in the loop.
