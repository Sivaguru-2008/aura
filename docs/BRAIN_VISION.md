# AURA NeuroMind — Brain Vision Engine

The first perception module of NeuroMind. It is **not** a segmentation tool that happens
to expose some features — it is a representation learner whose segmentation output is
one of five things it produces, and the other four exist because the modules that come
after it need a *description of a brain*, not a mask.

```python
from backend.vision.brain import BrainVisionEngine

engine = BrainVisionEngine.load()
result = engine.analyze_study(foundation_study)      # -> BrainVisionOutput

result.segmentation        # (Z, H, W) uint8 dense class labels
result.confidence          # (Z, H, W) float32, max softmax of the assigned class
result.tumor_probability   # from the presence head, not from thresholding the mask
result.regions             # per-region voxels, volume in mm3, probability, confidence
result.embedding           # (Z, 128) L2-normalised latent — the reusable asset
result.study_embedding     # one vector for the study
result.features            # coarse decoder maps + pooled bottleneck
result.quality             # predicted score AND the foundation layer's, kept apart
result.to_dict()           # JSON-safe; describes the arrays, never serialises them
```

---

## 1. Where it lives

```
aura/backend/
├── core/                       UNCHANGED — the Intelligent Router
├── engines/
│   ├── thorax/                 UNCHANGED — the chest-X-ray engine
│   └── neuro/engine.py         UNCHANGED — still a documented placeholder
├── foundation/mri/             UNCHANGED — the MRI Foundation Layer
└── vision/                     ← NEW
    └── brain/
        ├── types.py            vocabulary: regions, heads, stages, modality specs
        ├── errors.py           error taxonomy, same shape as the foundation layer's
        ├── config.py           injectable frozen configuration; the only path builder
        ├── io/brats_h5.py      BraTS2020 HDF5 corpus reader
        ├── ingest.py           BraTS -> Foundation Layer -> cached studies
        ├── dataset.py          slices + multi-task targets, recomputed post-crop
        ├── augment.py          geometric/intensity; image and label always together
        ├── degradations.py     synthetic MR artefacts for the quality head
        ├── sampling.py         curriculum + region focus + hard-example mining
        ├── model/
        │   ├── registry.py     the 3D-U-Net / SwinUNETR / nnU-Net extension seam
        │   ├── blocks.py       residual conv units
        │   ├── encoder.py      shared encoder — the artefact meant to outlive this net
        │   ├── decoder.py      skip-connected upsampling, multi-scale outputs
        │   ├── heads.py        the five prediction heads
        │   └── network.py      assembly + internal NetworkOutput
        ├── losses.py           multi-task objective with deep supervision
        ├── metrics.py          Dice/IoU/HD95/PPV/sens/spec + embedding probes
        ├── embeddings.py       the embedding store — write once, reuse forever
        ├── checkpoint.py       the six artefacts
        ├── output.py           BrainVisionOutput — the tensor/numpy boundary
        ├── validate.py         the validation cycle
        ├── train.py            the training loop
        ├── inference.py        serving
        └── cli.py              ingest | train | evaluate | smoke | info
```

**Everything this module writes lives under `aura/artifacts/brain/`.** No Thorax file,
directory, or code path is touched. `PathsConfig` is the only place a path is
constructed, which makes that a structural property rather than a convention.

### The one file outside the module that changed

`FileFormat` in `backend/foundation/mri/types.py` gained an `HDF5` member, and
`FORMAT_PREFERENCE` in `loader.py` gained it in last position. Research corpora are
redistributed as HDF5 and a study that flowed through the foundation layer should be
able to say what it was read from rather than be recorded as `UNKNOWN`. Placing it last
means no existing discovery decision changes. The foundation layer's 109 tests and the
router's tests pass unchanged.

---

## 2. Data flow

```
BraTS2020 HDF5 corpus     57 195 files, 369 subjects
    │
    ├─ BratsH5Reader              4 modality volumes + label, background restored to 0
    │
    ├─ MRIFoundationPipeline      per sequence: reorient to RAS, resample check,
    │  (run_series)               metadata, sequence ID, 7 quality checks, head mask
    │
    ├─ FoundationStudy            cached as JSON per subject, with full history
    │
    ├─ shared crop + slice select union brain mask; label rides the same transform
    │
    ├─ cache/volumes/*.img.npy    (Z, C, H, W) float16 memmap  ─┐
    │  cache/volumes/*.seg.npy    (Z, H, W)    uint8 memmap     ├─ ~11 GB, built once
    │  cache/slice_index.npz      per-slice areas, quality      ─┘
    │
    ├─ BrainSliceDataset          normalise, degrade, augment, fit, targets
    ├─ AdaptiveSliceSampler       curriculum + region focus + hard mining
    │
    ├─ BrainVisionNetwork         1 encoder, 1 decoder, 5 heads
    │
    └─ BrainVisionOutput          mask, confidence, size, embedding, metadata
       + embeddings/latest.npz    stored so nothing downstream recomputes them
```

---

## 3. The five heads

| # | Head | Output | Supervision |
|---|------|--------|-------------|
| 1 | Segmentation | 4-class logits at 4 decoder scales | expert consensus label, deeply supervised |
| 2 | Presence | 4 multi-label logits (WT, NCR/NET, ED, ET) | derived from the final label |
| 3 | Size | 4 scaled log-areas, softplus | derived from the final label |
| 4 | Quality | score in [0, 1] + artefact class | **synthetic degradations of known severity** |
| 5 | Embedding | 128-d L2-normalised | supervised contrastive + VICReg anti-collapse |

Heads 2, 3 and 5 read the **pooled encoder bottleneck**, not decoder features. That is
deliberate: it means an embedding can be computed without running the decoder, which is
what makes it cheap enough for a longitudinal module to compute over forty studies.

### Why the quality head is trained on synthetic artefacts

The foundation layer's quality score is nearly constant across BraTS — every subject was
preprocessed identically by the challenge organisers. A head regressed on that learns
the mean and reports it confidently. So a configurable share of every batch is degraded
by a known amount with a physically motivated artefact (Rician noise, a smooth
multiplicative bias field, k-space phase-encode ghosting, a k-space spike, blur) and the
head predicts the severity it can see. The validation report includes `predicted_std`,
`pearson_r`, and `severity_correlation` precisely so a constant predictor cannot pass as
a working head — and in v1 it did not pass, which is how the three defects below were
found.

### Why the quality head does not look like the others

It is the only head that reads the **image** rather than the encoder, and the only one
that reads **nothing** from the encoder. Three measured reasons, each of which cost v1 a
working head:

1. **Instance normalisation deletes the signal.** Every encoder stage is `Conv ->
   InstanceNorm -> LeakyReLU`, and instance norm removes each channel's per-sample mean
   and variance — exactly where "this image is noisy" lives. Whatever survives is then
   flattened by global *average* pooling. So the head has its own `QualityBranch`: three
   strided convolutions with **no normalisation anywhere**, read out with mean **and
   standard-deviation** pooling. The std term is the point; average pooling cannot
   express texture energy.
2. **Severity across five artefacts is not one regression.** Blur lowers high-frequency
   energy and noise raises it. Measured on texture statistics of the normalised slice:
   pooled severity is recoverable at r=0.53, but *within* an artefact it is r=0.97 for
   noise and r=0.77 for blur, and the artefact type is 68% recoverable on its own. So
   the head also classifies the artefact, which gives it a representation where severity
   is conditional rather than an averaged contradiction. The effect is visible in
   training: blur severity was unlearned (r=-0.03) until the artefact classifier began
   separating types around epoch 9, after which it reached r=-0.60.
3. **Concatenating the bottleneck makes it collapse.** The obvious design feeds the
   320-d pooled vector in alongside the 128-d texture vector. Measured, the head then
   settles on predicting the mean and the artefact accuracy pins to exactly the
   clean-class prevalence — the bottleneck is larger, segmentation-shaped, and still
   moving early in training. The branch trained alone reaches r=0.65. So the bottleneck
   is left out, which also makes this head independently replaceable.

One correction worth recording, because it was my own reasoning error: the loss weight
was raised from 0.20 to 2.0 on the theory that the head was training too slowly. AdamW is
approximately scale-invariant to loss magnitude, so that change was largely inert — 10x
the weight moved the severity correlation from -0.036 to -0.068. Decoupling from the
bottleneck was the fix.

---

## 4. The three adaptive mechanisms

**Curriculum** — area thresholds are percentiles of the measured corpus distribution
(p75 = 2 288 px, p25 = 499 px over a 3 000-slice sample), not round numbers. Stages walk
large → medium → small → the full distribution *including tumour-free anatomy*. The
negative fraction is never zero, because a network that sees only tumours for five
epochs learns to find one in a normal brain.

**Region focus** — 43% of cached slices carry a label. `tumor_fraction` (default 0.70)
fixes the share of each epoch drawn from the positive pool. Deliberately not 1.0.

**Hard-example mining** — the difficulty signal comes from *training* observations, not
from validation. Validation subjects are held out and say nothing about which training
slice is hard, and a separate difficulty pass would cost a second forward pass over
34 000 slices per epoch. The trainer already computes a per-sample foreground Dice as
part of the loss, so the signal is free and continuous. Difficulty is an EMA, and weights
are clamped to `[0.25x, 6x]` of the mean — the floor keeps solved samples in the
distribution, the ceiling stops a handful of mislabelled slices owning the epoch.

---

## 5. Three properties that had to be made structural

**The label goes wherever the image goes.** Every geometric operation — reorientation at
ingest, cropping, flips, rotations, affine warps, grid fitting — takes `(image, label)`
and returns `(image, label)`, with `order=0` for the label. There is no API in this
module that transforms one without the other. A model trained against a label rotated
three degrees away from its image converges perfectly happily and is worthless, and no
metric in this package would notice.

**No raw tensor leaves the module.** `NetworkOutput` holds tensors and is internal;
`BrainVisionOutput` holds numpy and is public. `to_dict()` describes arrays — shape,
dtype, per-class voxel counts, confidence statistics — and never serialises them.

**Measured and assumed are kept apart.** The corpus's modality channel order was
*derived by measurement* and is re-verified on every subject at ingest, with a
configurable agreement floor below which the cache refuses to build. The corpus's
laterality was **not** verified — the HDF5 redistribution carries no affine — so a 1 mm
LPS grid is declared from the BraTS convention and every study, checkpoint, and result
carries the caveat that left/right must not be reported.

---

## 6. Checkpoints

| File | Contents | Why it exists separately |
|------|----------|--------------------------|
| `best_brain_model.pt` | full network at the best validated epoch | serving |
| `latest_brain_model.pt` | full network at the last epoch | resume, inspection |
| `brain_encoder.pt` | encoder alone + declared pyramid contract | **transfer learning** |
| `brain_decoder.pt` | decoder + segmentation head | swap a decoder without retraining |
| `brain_embedding_head.pt` | projector alone | recompute embeddings from any encoder |
| `training_state.pt` | optimiser, scaler, EMA, sampler difficulty, RNG | exact resume |

Every checkpoint carries its architecture description, configuration, metrics, and the
corpus caveats. `BrainVisionEngine.load` builds the network from the checkpoint's own
architecture record rather than from the caller's configuration, and a mismatch raises
`CheckpointError` naming the cause rather than 200 lines of shape errors.

Validation runs on the EMA weights when EMA is enabled and the best checkpoint is
selected on them, so `load_network_checkpoint` prefers the EMA copy — serving raw weights
after selecting on EMA ones would deploy a model that was never the one measured.

---

## 7. Extending it

The training pipeline never names a model class. It asks
`backend.vision.brain.model.registry` for an encoder and a decoder by string, and the
losses, heads, metrics, and checkpoints are all expressed against two protocols:

```python
@register_encoder("unet3d")
def build_unet3d(**kwargs) -> EncoderBackbone:
    ...   # feature_channels, strides, embedding_channels; forward -> [finest .. bottleneck]
```

`unet3d`, `swin_unetr`, `nnunet`, and `vit` are registered as **declarations**: asking
for one raises `ArchitectureUnavailable` naming what would be required. That is the same
posture the foundation layer takes with N4 and skull stripping — a roadmap entry that
raises is honest, while an alias that quietly returns the 2D network produces a model
card that lies.

Adding a modality (PET, CT, a diffusion map) is a `ModalitySpec` in `ModelConfig`. The
encoder's input stem is **per modality** rather than a shared 4-channel convolution
precisely so that a new sequence is a new stem and the rest of the encoder transfers
untouched — and so a *missing* sequence can be dropped from the average rather than
zero-filled, which a convolution reads as "uniformly dark tissue".

---

## 8. Running it

```bash
python -m backend.vision.brain.cli ingest --corpus <path-to-BraTS-h5-dir>
```

```bash
python -m backend.vision.brain.cli train --epochs 30 --batch-size 16 --num-workers 4
```

```bash
python -m backend.vision.brain.cli evaluate --split test --full
```

```bash
python -m backend.vision.brain.cli info
```

```bash
python -m backend.vision.brain.cli smoke --corpus <path-to-BraTS-h5-dir>
```

`AURA_BRATS_ROOT` supplies the corpus path when `--corpus` is omitted. `smoke` runs the
whole pipeline over six subjects with a three-stage network in about a minute — it
exists so "is the pipeline wired correctly" and "is the model any good" are separable
questions.

Measured on this deployment (RTX 5050 Laptop, 8 GB):

| stage | cost |
|-------|------|
| ingest, 369 subjects | 42 min, 49 581 cached slices, ~11 GB |
| training step, batch 16 at 192x192 | 194 ms, 1.9 GB peak GPU |
| epoch, 8 000 samples | ~97 s |
| full validation with HD95, 7 407 slices | ~60 s |

---

## 9. Measured results (v2 checkpoint)

Trained the full 30-epoch budget; best epoch 24. **Held-out test split: 56 subjects,
7 531 slices, no subject shared with training or model selection.**

### Segmentation

| region | Dice (pooled) | Dice (per-slice) | IoU | sens | spec | prec | HD95 |
|---|---|---|---|---|---|---|---|
| whole tumour | **0.9150** | 0.8799 | 0.8433 | 0.9309 | 0.9979 | 0.8996 | 7.08 px |
| tumour core | **0.8456** | 0.8861 | 0.7325 | 0.8322 | 0.9990 | 0.8595 | 6.15 px |
| enhancing tumour | **0.8349** | 0.8967 | 0.7165 | 0.8371 | 0.9994 | 0.8327 | 4.53 px |
| NCR/NET | 0.7228 | 0.8193 | 0.5659 | 0.6981 | 0.9991 | 0.7492 | 8.67 px |
| oedema | 0.8227 | 0.8384 | 0.6988 | 0.8538 | 0.9972 | 0.7939 | 8.12 px |

Composite mean 0.8652, class mean 0.7934. Inference 319 slices/s (3.1 ms/slice), 883 MB
peak.

Two things to hold on to when comparing against published BraTS numbers. These are **2D
per-slice** scores, not 3D per-case. And pooled Dice weighs a 3 000-pixel tumour like a
3-pixel focus while the per-slice mean does the opposite — both are given for that
reason, and the enhancing region is the one where they diverge most (0.835 vs 0.897).

### Per-head verdicts, on the test split

Each head has a metric a degenerate predictor cannot fake. The model card carries the
verdict for **both** splits, and where they disagree the test one stands — the
validation split is what the best epoch was selected on, so a marginal result there is
exactly the one likely to be selection noise.

| head | metric | test result | verdict |
|---|---|---|---|
| segmentation | composite Dice | 0.8652 | pass |
| presence | AUROC, worst region | 0.9774 | pass |
| size | Pearson r, worst region | 0.8874 | pass |
| quality | correlation with known severity | **−0.4234** | **pass** |
| embedding (morphology) | k-NN purity, 16 classes | 0.7479, not collapsed | pass |
| embedding (grade transfer) | cross-subject k-NN on held-out grade | 0.7798 vs 0.7892 baseline | **fail** |

### The quality head, v1 → v2

v1's head was a near-constant predictor. v2's works:

| | v1 | v2 |
|---|---|---|
| severity correlation | −0.071 | **−0.423** |
| predicted σ (target σ 0.308) | 0.015 | **0.217** |
| clean vs degraded separation | 0.002 | **0.239** |
| artefact type accuracy (chance 0.167) | — | **0.665** |
| overall r vs quality target | 0.079 | **0.635** |

Per artefact: blur −0.707, ghosting −0.584, k-space spike −0.562, Rician noise −0.555,
bias field −0.289. The bias field is the weak one and that is expected rather than
disappointing — per-slice z-scoring removes most of a smooth multiplicative field by
construction, and a texture probe on the normalised slice caps it at r≈0.23.

The three defects behind v1 are in §3. The blur result is the clearest evidence that the
artefact classifier earned its place: blur severity sat at r≈−0.03 for nine epochs and
only began to move once the classifier started separating artefact types, ending at
−0.707.

### Grade transfer is a negative result

0.7798 against a 0.7892 majority baseline. It was +0.015 over baseline on validation and
−0.009 on test, which is noise either side of chance. **The embedding does not carry
tumour grade.** It demonstrably clusters the morphology it *was* trained on — purity
0.7479 over 16 classes, mean pairwise cosine 0.477, effective rank 5.7 of 128 dimensions
— and that is what it is validated for and the limit of what it is validated for.

### Segmentation: v1 vs v2

| region | v1 | v2 | delta |
|---|---|---|---|
| whole tumour | 0.9156 | 0.9150 | −0.0006 |
| tumour core | 0.8517 | 0.8456 | −0.0061 |
| enhancing | 0.8405 | 0.8349 | −0.0056 |

A −0.004 composite drop. **This is one run each, so it should be read as run-to-run
variance rather than as a regression caused by the head change** — the quality head is
fully decoupled in v2 and contributes no gradient to the shared encoder at all. Settling
it would take several seeds, which has not been done.

---

## 10. What this module does not claim

* It is trained on **glioma only**. It has never seen a metastasis, a meningioma, an
  abscess, a demyelinating lesion, or a healthy brain, and its output on any of those is
  undefined rather than negative.
* Every BraTS subject is pre-skull-stripped, N4-corrected, and atlas-registered. A
  clinical study that has not been through the same preparation is out of distribution.
* Training and validation are on **2D axial slices**. Volumetric consistency between
  adjacent slices is neither enforced by the model nor measured by the reported metrics.
* **Laterality is not verified** and must not be reported.
* The tumour-grade figure in the embedding metrics is a *representation-quality probe*,
  not a grading model, and on the test split it does not beat the majority baseline.
  Grade is never a training target — that is what makes it an honest probe, and what
  makes the negative result meaningful.
* `backend/engines/neuro/engine.py` still returns `not_implemented`. Wiring this engine
  into the router is a separate, deliberate step: it means a clinical-looking payload
  starts reaching API callers, and that decision deserves its own review.
