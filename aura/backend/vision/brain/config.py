"""Configuration for the Brain Vision Engine.

Frozen dataclasses, constructed by the caller and injected — the same posture as the
MRI Foundation Layer, for the same reason: no module-level mutable state, no path baked
into a default, no ``os.environ`` read at import time. Two trainings with different
curricula can run in one process without disturbing each other, which is what makes an
ablation possible at all.

Every knob the specification asked to be configurable is here, and every threshold that
encodes a judgement carries the measurement that produced it. The tumour-area
thresholds in :class:`CurriculumConfig` in particular are not round numbers picked to
look reasonable — they are percentiles measured over a 3 000-slice sample of the
corpus, and the comment says which.

Paths
-----
Brain artefacts live under ``artifacts/brain/`` and nothing in this module ever writes
outside it. That is the whole of the "store Brain artefacts separately from Thorax
artefacts" requirement, made structural: :class:`PathsConfig` is the only place a path
is constructed, and every other module asks it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backend.vision.brain.errors import ConfigurationError
from backend.vision.brain.types import (
    DEFAULT_MODALITIES,
    CurriculumStage,
    HeadName,
    ModalitySpec,
)


def _artifacts_root() -> Path:
    """Locate ``aura/artifacts``, preferring the project's own answer.

    Imported lazily and defensively: ``common.config`` belongs to the chest stack, and
    this module must remain importable in a checkout where that package has moved. The
    fallback walks up from this file, which gives the same answer for the same tree.
    """
    try:
        from common.config import ARTIFACTS

        return Path(ARTIFACTS)
    except Exception:                                    # pragma: no cover - layout dep
        return Path(__file__).resolve().parents[3] / "artifacts"


def _default_corpus_root() -> Path | None:
    """Where the BraTS corpus is, if an operator said.

    ``AURA_BRATS_ROOT`` is read here rather than at import time, and only as a
    *default* the caller may override. A training script that hardcodes a dataset path
    stops working on the next machine; one that has no default at all is tedious to
    run. An environment variable consulted at construction is the compromise.
    """
    value = os.environ.get("AURA_BRATS_ROOT")
    return Path(value) if value else None


@dataclass(frozen=True)
class PathsConfig:
    """Every path this module reads or writes. The only place a path is built."""

    #: Root of the BraTS2020 HDF5 corpus — the directory holding ``volume_*_slice_*.h5``
    #: together with ``meta_data.csv`` and ``name_mapping.csv``.
    corpus_root: Path | None = field(default_factory=_default_corpus_root)
    #: Everything this module writes lives under here. Never ``artifacts/`` itself:
    #: the Thorax stack owns that directory and its filenames.
    artifacts_root: Path = field(
        default_factory=lambda: _artifacts_root() / "brain")

    # -- derived locations ---------------------------------------------------- #
    @property
    def cache_dir(self) -> Path:
        """Standardised volumes and labels, written once by ingest."""
        return self.artifacts_root / "cache"

    @property
    def manifest_path(self) -> Path:
        """Per-slice index: geometry, tumour areas, quality, split, provenance."""
        return self.cache_dir / "manifest.json"

    @property
    def studies_dir(self) -> Path:
        """Per-subject :class:`FoundationStudy` descriptions, as JSON."""
        return self.cache_dir / "studies"

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifacts_root / "checkpoints"

    @property
    def embedding_dir(self) -> Path:
        """Exported latent representations, keyed by sample id."""
        return self.artifacts_root / "embeddings"

    @property
    def report_dir(self) -> Path:
        """Validation reports, history, and the model card."""
        return self.artifacts_root / "reports"

    @property
    def tensorboard_dir(self) -> Path:
        return self.artifacts_root / "tensorboard"

    # -- the six checkpoints the specification names -------------------------- #
    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / "best_brain_model.pt"

    @property
    def latest_model_path(self) -> Path:
        return self.checkpoint_dir / "latest_brain_model.pt"

    @property
    def encoder_path(self) -> Path:
        return self.checkpoint_dir / "brain_encoder.pt"

    @property
    def decoder_path(self) -> Path:
        return self.checkpoint_dir / "brain_decoder.pt"

    @property
    def embedding_head_path(self) -> Path:
        return self.checkpoint_dir / "brain_embedding_head.pt"

    @property
    def training_state_path(self) -> Path:
        return self.checkpoint_dir / "training_state.pt"

    @property
    def history_path(self) -> Path:
        return self.report_dir / "history.jsonl"

    @property
    def model_card_path(self) -> Path:
        return self.report_dir / "model_card.json"

    def ensure(self) -> None:
        """Create every output directory. Never touches ``corpus_root``."""
        for directory in (self.artifacts_root, self.cache_dir, self.studies_dir,
                          self.checkpoint_dir, self.embedding_dir, self.report_dir,
                          self.tensorboard_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class IngestConfig:
    """How the corpus is turned into a cache of standardised studies."""

    #: Fraction of a slice that must be brain before the slice is cached at all. The
    #: corpus stores 155 slices per subject of which roughly 80 contain a head; the
    #: remainder are empty and, because the corpus z-scores each slice independently,
    #: an empty slice's handful of nonzero voxels is amplified by a factor of ~60. They
    #: are excluded here rather than filtered at every later stage.
    min_brain_fraction: float = 0.01
    #: Margin in voxels around the brain bounding box when cropping the cached volume.
    crop_margin_voxels: int = 4
    #: Crop the cached volume to the brain bounding box. Halves the cache and removes
    #: air the network would otherwise spend capacity on. The crop box is computed once
    #: over the union of all sequences so voxel correspondence — and therefore label
    #: alignment — is preserved exactly.
    crop_to_brain: bool = True
    #: Store cached voxels as float16. The corpus's own values are z-scores in roughly
    #: [-1, 8]; float16 resolves those to ~3 decimal places, well below the noise floor,
    #: and halves 30 GB of cache to 15.
    store_float16: bool = True
    #: Verify the modality channel order on every subject that has an enhancing region,
    #: by testing that the post-contrast channel is the most hyperintense one there.
    #: See :mod:`backend.vision.brain.io.brats_h5` for why this is not optional.
    verify_channel_assignment: bool = True
    #: Fraction of verified subjects that must agree before ingest refuses to continue.
    min_channel_agreement: float = 0.90
    #: Cap on subjects ingested. ``None`` means all of them; a small number is how the
    #: whole pipeline is smoke-tested in a minute.
    max_subjects: int | None = None
    #: Re-ingest subjects that are already cached.
    overwrite: bool = False

    #: How many cached volumes one process may hold memory-mapped at once. A training
    #: split is 258 subjects at ~40 MB, and a uniform sampler touches nearly all of them
    #: in every worker over one epoch — an unbounded cache ends the epoch with ~10 GB of
    #: mapped views per worker, which on Windows exhausts the system commit limit and
    #: kills the run with an unrelated-looking shared-memory error. 24 keeps a worker
    #: under a gigabyte; reopening a map is a file handle, not a read.
    max_open_volumes: int = 24


@dataclass(frozen=True)
class SplitConfig:
    """Train / validation / test partition.

    Split **by subject**. A BraTS subject contributes ~80 cached slices that are
    near-duplicates of their neighbours; splitting by slice would put slice 71 of a
    tumour in training and slice 72 in validation, and the resulting Dice would measure
    interpolation, not generalisation.
    """

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    #: The remainder. Named explicitly so a config that does not sum to 1 is a config
    #: error rather than a silently shrunken test set.
    test_fraction: float = 0.15
    seed: int = 7
    #: Keep the HGG/LGG ratio equal across splits. The corpus is 293/76, so an
    #: unstratified draw can leave the test set with a handful of low-grade subjects
    #: and make the grade probe meaningless.
    stratify_by_grade: bool = True

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"split fractions must sum to 1.0; got {total:.4f}",
                detail={"train": self.train_fraction, "val": self.val_fraction,
                        "test": self.test_fraction})


@dataclass(frozen=True)
class SamplingConfig:
    """Region-focused sampling and hard-example mining.

    Uniform sampling over cached slices would spend 57% of every epoch on slices with
    no tumour in them (measured over a 3 000-slice sample: 43.1% of slices carry a
    label). ``tumor_fraction`` is the share of each epoch's draws that are guaranteed to
    contain pathology; the rest are drawn from the tumour-free pool so the network still
    sees normal anatomy and does not learn that every brain has a tumour.
    """

    enabled: bool = True
    #: Share of each epoch drawn from tumour-bearing slices. 0.70 against a natural
    #: 0.43 is a moderate oversample — enough to matter, far from the degenerate 1.0
    #: that would make the presence head useless.
    tumor_fraction: float = 0.70
    #: Draws per epoch. Decouples "epoch" from "one pass over 45 000 slices" so the
    #: validation cadence, the curriculum, and hard-example refresh are all on a
    #: schedule the operator picks.
    samples_per_epoch: int = 8000

    # -- hard example mining -------------------------------------------------- #
    #: Raise the sampling probability of slices the model currently segments badly.
    hard_mining: bool = True
    #: Exponential-moving-average factor for a sample's observed difficulty. Low, so a
    #: single unlucky batch does not brand a sample as hard for the rest of training.
    difficulty_ema: float = 0.30
    #: Difficulty is raised to this power before becoming a weight. 1.0 is linear; above
    #: 2 the tail dominates and the easy majority stops being seen at all.
    hard_mining_power: float = 1.5
    #: Bounds on the multiplier any one sample's weight may reach. The floor keeps
    #: solved samples in the distribution (a model that never revisits them forgets
    #: them); the ceiling stops a handful of mislabelled slices from owning the epoch.
    min_weight: float = 0.25
    max_weight: float = 6.0
    #: Fraction of a sample's weight that comes from difficulty rather than from the
    #: base region-focused prior. At 1.0 the prior is discarded.
    hard_mining_strength: float = 0.60


@dataclass(frozen=True)
class CurriculumConfig:
    """Progressive exposure, from unmissable tumours to the real distribution.

    Area thresholds are percentiles of the total tumour area of *positive* slices,
    measured over a 3 000-slice random sample of the corpus:

    ===========  ========  ==============
    percentile   pixels    % of a slice
    ===========  ========  ==============
    p25            499       0.87
    p50           1270       2.20
    p75           2288       3.97
    ===========  ========  ==============

    ``large`` is the p75 tail, ``medium`` reaches down to p25, ``small`` admits every
    positive slice, and ``full`` restores the complete distribution including
    tumour-free anatomy.

    The full ingested corpus later confirmed the sample: 49 581 slices, 49.2% positive,
    p25/p50/p75 = 487/1265/2269 px. The thresholds below are the sampled values and were
    left as they were rather than nudged to match — a 2% difference does not change
    which slices a stage admits, and rewriting a threshold to agree with a later
    measurement makes it look better-founded than it is.
    """

    enabled: bool = True
    #: Epochs spent in each stage, in order. The sum may be less than the epoch budget:
    #: training continues in the final stage until the budget or early stopping ends it.
    schedule: tuple[tuple[CurriculumStage, int], ...] = (
        (CurriculumStage.LARGE, 2),
        (CurriculumStage.MEDIUM, 3),
        (CurriculumStage.SMALL, 3),
        (CurriculumStage.FULL, 100),
    )
    #: Minimum total tumour area, in pixels, admitted at each stage.
    stage_min_area: dict[CurriculumStage, int] = field(default_factory=lambda: {
        CurriculumStage.LARGE: 2288,     # p75
        CurriculumStage.MEDIUM: 499,     # p25
        CurriculumStage.SMALL: 1,        # every positive slice
        CurriculumStage.FULL: 0,
    })
    #: Share of tumour-free slices admitted at each stage. Never zero before ``FULL``:
    #: a network that has seen only tumours for five epochs learns to find one in
    #: healthy tissue, and unlearning that costs more than avoiding it.
    stage_negative_fraction: dict[CurriculumStage, float] = field(
        default_factory=lambda: {
            CurriculumStage.LARGE: 0.15,
            CurriculumStage.MEDIUM: 0.20,
            CurriculumStage.SMALL: 0.25,
            CurriculumStage.FULL: 0.30,
        })

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        """Which stage epoch ``epoch`` (0-based) belongs to."""
        if not self.enabled:
            return CurriculumStage.FULL
        cursor = 0
        for stage, span in self.schedule:
            cursor += span
            if epoch < cursor:
                return stage
        return self.schedule[-1][0] if self.schedule else CurriculumStage.FULL


@dataclass(frozen=True)
class AugmentationConfig:
    """Geometric and intensity augmentation.

    Left-right flipping is **off by default** and the reason is clinical, not
    statistical: laterality is a reportable finding, and a network trained with mirrored
    brains has been told explicitly that left and right are interchangeable. The corpus
    also carries no verified laterality (see the ingest record), so a flip here would
    compound an unverified assumption rather than test one.
    """

    enabled: bool = True
    flip_lr: bool = False
    flip_ap: bool = True
    rot90: bool = True
    #: Small in-plane rotation/scale/shift, applied identically to image and label.
    affine: bool = True
    max_rotation_deg: float = 15.0
    max_scale: float = 0.15
    max_shift: float = 0.08
    #: Per-channel multiplicative and additive intensity jitter, after normalisation.
    intensity_scale: float = 0.15
    intensity_shift: float = 0.10
    gamma: float = 0.20
    #: Probability any one augmentation is applied to a given sample.
    probability: float = 0.50


@dataclass(frozen=True)
class DegradationConfig:
    """Synthetic MR artefacts, used to supervise the image-quality head.

    Why this exists at all: the foundation layer's quality score is nearly constant
    across BraTS, because every subject was preprocessed identically by the challenge
    organisers. A head regressed on a constant learns the constant and reports a
    confident number that means nothing. So quality supervision is *manufactured*: a
    fraction of every batch is degraded by a known amount with a physically motivated
    artefact, and the head predicts the severity it can see. The label is exact because
    we chose it.

    Each artefact is a real MR failure mode, simulated in the domain it actually occurs
    in — ghosting and spikes in k-space, bias as a smooth multiplicative field, noise as
    Rician rather than Gaussian.
    """

    enabled: bool = True
    #: Share of training samples that receive a synthetic degradation.
    probability: float = 0.35
    #: Artefacts in the pool. Removing one is how its contribution gets ablated.
    artifacts: tuple[str, ...] = ("rician_noise", "bias_field", "motion_ghosting",
                                  "k_space_spike", "blur")
    #: Severity is drawn uniformly from this range; the quality target is ``1 -
    #: severity`` for a degraded sample.
    severity_range: tuple[float, float] = (0.15, 0.95)
    #: Apply degradation to validation samples too, on a fixed seed, so the quality
    #: head is measured against known severities rather than against a constant.
    validate_on_degraded: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. Names resolve through the registry, so this is the extension seam."""

    encoder: str = "residual_unet2d"
    decoder: str = "unet2d"
    #: Channels at each encoder stage. Five stages take a 192x192 input down to 12x12.
    stage_channels: tuple[int, ...] = (32, 64, 128, 256, 320)
    #: Residual blocks per stage.
    blocks_per_stage: tuple[int, ...] = (1, 2, 2, 2, 2)
    norm: str = "instance"                 # instance | batch | group
    activation: str = "leaky_relu"
    dropout: float = 0.0
    #: Number of decoder levels that emit a supervised prediction, counting from the
    #: finest. 1 disables deep supervision.
    deep_supervision_levels: int = 4
    #: Latent embedding width. 128 is small enough to store for every validation sample
    #: without thought and wide enough to carry 22 morphology classes apart.
    embedding_dim: int = 128
    #: Width of the hidden layer in the embedding projector.
    embedding_hidden: int = 256
    #: Channel width of the quality head's un-normalised texture branch. Small on
    #: purpose: image quality is a texture statistic, not a semantic property.
    quality_texture_channels: int = 64
    #: Heads to build. Dropping one removes its parameters, its loss, and its metrics.
    heads: tuple[HeadName, ...] = (HeadName.SEGMENTATION, HeadName.PRESENCE,
                                   HeadName.SIZE, HeadName.QUALITY, HeadName.EMBEDDING)
    #: Input channels.
    modalities: tuple[ModalitySpec, ...] = DEFAULT_MODALITIES
    #: In-plane size the network trains at. Cached slices are cropped or padded to it.
    input_size: tuple[int, int] = (192, 192)


@dataclass(frozen=True)
class LossConfig:
    """Weighted multi-task objective. Every component is separately switchable."""

    # -- segmentation --------------------------------------------------------- #
    dice_weight: float = 1.0
    cross_entropy_weight: float = 1.0
    focal_weight: float = 0.0
    boundary_weight: float = 0.0
    focal_gamma: float = 2.0
    #: Per-class weight in the cross-entropy. Background is down-weighted because it is
    #: ~97% of every slice; the enhancing class is up-weighted because it is the
    #: smallest, the hardest, and the one that changes management.
    class_weights: tuple[float, ...] = (0.2, 1.0, 1.0, 1.5)
    #: Smoothing term in the Dice denominator. Also the value of a Dice on an
    #: all-empty class, which is why it is not zero.
    dice_smooth: float = 1.0
    #: Exclude the background channel from the Dice term. Standard for BraTS: including
    #: it makes the metric ~0.97 regardless of whether the tumour was found.
    dice_ignore_background: bool = True

    # -- deep supervision ----------------------------------------------------- #
    #: Relative weight of each supervised decoder level, finest first. Halving per level
    #: is the nnU-Net convention; the coarse levels shape features without dominating.
    deep_supervision_weights: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

    # -- other heads ---------------------------------------------------------- #
    segmentation_weight: float = 1.0
    presence_weight: float = 0.30
    size_weight: float = 0.20
    #: Deliberately larger than the other auxiliary heads, and the reason is arithmetic
    #: rather than importance. A head's own parameters receive gradient only from its own
    #: loss terms, so what governs how fast it trains is ``weight x learning_rate``, not
    #: its share of the total. Quality MSE sits around 0.05-0.08 while the segmentation
    #: and artefact terms are 1.9-2.3, and at 0.20 the quality objective trained roughly
    #: 17x slower than a standalone probe that fitted the same target to r=0.65 in 2 500
    #: steps — so the v1 head never left its initialisation. 2.0 restores the effective
    #: step size that worked, and the 5:1 ratio to the artefact term below is the one the
    #: standalone experiment used.
    quality_weight: float = 2.0
    #: Auxiliary artefact-type classification inside the quality head. It exists to give
    #: the head a representation in which severity is conditional on artefact type;
    #: measured recoverability is r=0.97 for noise and r=0.23 for a bias field, so a
    #: single pooled severity regression is asking one output to be two things.
    quality_artifact_weight: float = 0.40
    embedding_weight: float = 0.15

    # -- embedding objective -------------------------------------------------- #
    #: Temperature of the supervised-contrastive term. 0.1 is the value the SupCon
    #: paper reports for its best ImageNet results and is a reasonable prior here.
    supcon_temperature: float = 0.10
    #: Hinge target for the per-dimension standard deviation of the embedding batch.
    #: The anti-collapse term: without it a contrastive objective on a shared encoder
    #: can satisfy itself by mapping everything to a point.
    variance_target: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.04

    def __post_init__(self) -> None:
        seg_terms = (self.dice_weight + self.cross_entropy_weight
                     + self.focal_weight + self.boundary_weight)
        if seg_terms <= 0:
            raise ConfigurationError(
                "every segmentation loss component is zero; the segmentation head "
                "would receive no gradient",
                detail={"dice": self.dice_weight, "ce": self.cross_entropy_weight,
                        "focal": self.focal_weight, "boundary": self.boundary_weight})


@dataclass(frozen=True)
class OptimConfig:
    """Optimisation and the stability machinery around it."""

    epochs: int = 20
    batch_size: int = 16
    #: Optimiser steps are taken every ``grad_accum`` batches, so the effective batch is
    #: ``batch_size * grad_accum``. The knob that lets an 8 GB card train at a batch
    #: size the schedule was tuned for.
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.99)
    #: Linear warmup, in epochs. Fractional values are honoured — 0.5 warms over half
    #: an epoch, which is usually enough at this batch size.
    warmup_epochs: float = 1.0
    grad_clip: float = 12.0
    amp: bool = True
    #: Exponential moving average of the weights. Costs one extra copy of the model and
    #: reliably buys a little Dice; validated *and* checkpointed separately so the claim
    #: can be checked rather than assumed.
    ema: bool = True
    ema_decay: float = 0.999
    #: Stop when the monitored metric has not improved for this many validations.
    early_stopping_patience: int = 6
    #: Metric that decides "best". Mean Dice over the three BraTS composite regions —
    #: the number the field reports — rather than over the primary classes.
    monitor: str = "composite_dice_mean"
    monitor_mode: str = "max"
    num_workers: int = 4
    seed: int = 7
    #: Resume from ``training_state.pt`` when one exists.
    auto_resume: bool = True
    #: Compile the model with ``torch.compile``. Off by default: on Windows the
    #: Inductor backend needs a C++ toolchain that is not part of this deployment, and
    #: a training run that dies at step 1 for a 5% speedup is a bad trade.
    compile: bool = False


@dataclass(frozen=True)
class ValidationConfig:
    """What is measured, how often, and what is exported."""

    every_n_epochs: int = 1
    #: Cap on validation batches. ``None`` uses the whole split.
    max_batches: int | None = None
    batch_size: int = 24
    #: Hausdorff is O(n log n) per class per slice via a distance transform. Real cost,
    #: real value — it is the metric that catches a Dice-happy model producing scattered
    #: false-positive islands.
    compute_hausdorff: bool = True
    #: 95th percentile rather than the maximum. The maximum is decided by a single
    #: outlier voxel and is close to useless as a comparison between epochs.
    hausdorff_percentile: float = 95.0
    #: Export a latent embedding for every validation sample, every validation cycle.
    export_embeddings: bool = True
    #: Cap on stored embeddings per cycle. 128 floats per sample is 512 bytes; 20 000
    #: samples is 10 MB, which is cheap enough that the cap exists only for pathological
    #: configurations.
    embedding_limit: int = 20000
    #: Measure inference latency and peak GPU memory during validation.
    measure_performance: bool = True
    #: Evaluate the embedding space against the held-out tumour grade. The grade is
    #: never a training target, which is what makes this an honest probe.
    probe_grade: bool = True


@dataclass(frozen=True)
class BrainVisionConfig:
    """Complete configuration for one Brain Vision training or inference run."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    #: ``cuda`` / ``cpu`` / ``None`` to detect at build time.
    device: str | None = None
    #: Free-text label for this run, stamped into checkpoints and reports.
    run_name: str = "brain_vision_v1"

    def with_overrides(self, **kwargs: Any) -> "BrainVisionConfig":
        """Return a copy with top-level fields replaced. Frozen in, frozen out."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description, for the model card and the run record."""
        return {
            "run_name": self.run_name,
            "device": self.device,
            "paths": {"corpus_root": str(self.paths.corpus_root)
                                     if self.paths.corpus_root else None,
                      "artifacts_root": str(self.paths.artifacts_root)},
            "ingest": _asdict(self.ingest),
            "split": _asdict(self.split),
            "sampling": _asdict(self.sampling),
            "curriculum": {
                "enabled": self.curriculum.enabled,
                "schedule": [[s.value, n] for s, n in self.curriculum.schedule],
                "stage_min_area": {s.value: v
                                   for s, v in self.curriculum.stage_min_area.items()},
                "stage_negative_fraction": {
                    s.value: v
                    for s, v in self.curriculum.stage_negative_fraction.items()},
            },
            "augmentation": _asdict(self.augmentation),
            "degradation": _asdict(self.degradation),
            "model": {**_asdict(self.model),
                      "heads": [h.value for h in self.model.heads],
                      "modalities": [m.to_dict() for m in self.model.modalities]},
            "loss": _asdict(self.loss),
            "optim": _asdict(self.optim),
            "validation": _asdict(self.validation),
        }


def _asdict(obj: Any) -> dict[str, Any]:
    """``dataclasses.asdict`` with tuples flattened to lists and paths to strings."""
    from dataclasses import fields, is_dataclass

    def convert(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (tuple, list)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {(k.value if hasattr(k, "value") else str(k)): convert(v)
                    for k, v in value.items()}
        if hasattr(value, "value") and isinstance(value, __import__("enum").Enum):
            return value.value
        return value

    return convert(obj)                                   # type: ignore[return-value]


def smoke_config(corpus_root: Path | str | None = None,
                 artifacts_root: Path | str | None = None) -> BrainVisionConfig:
    """A configuration small enough to run the whole pipeline end to end in a minute.

    Used by the tests and by ``cli.py --smoke``. It exists so that "does the pipeline
    work" and "is the model any good" are separable questions — the first should never
    require an afternoon of GPU time to answer.
    """
    paths = PathsConfig(
        corpus_root=Path(corpus_root) if corpus_root else _default_corpus_root(),
        artifacts_root=(Path(artifacts_root) if artifacts_root
                        else _artifacts_root() / "brain_smoke"),
    )
    return BrainVisionConfig(
        paths=paths,
        run_name="brain_vision_smoke",
        ingest=IngestConfig(max_subjects=6),
        model=ModelConfig(stage_channels=(8, 16, 32), blocks_per_stage=(1, 1, 1),
                          deep_supervision_levels=2, embedding_dim=32,
                          embedding_hidden=64, input_size=(96, 96)),
        optim=OptimConfig(epochs=2, batch_size=4, num_workers=0, amp=False,
                          warmup_epochs=0.0, early_stopping_patience=2),
        sampling=SamplingConfig(samples_per_epoch=32),
        curriculum=CurriculumConfig(schedule=((CurriculumStage.LARGE, 1),
                                              (CurriculumStage.FULL, 100))),
        validation=ValidationConfig(batch_size=4, max_batches=2, embedding_limit=64),
    )


__all__ = [
    "AugmentationConfig", "BrainVisionConfig", "CurriculumConfig", "DegradationConfig",
    "IngestConfig", "LossConfig", "ModelConfig", "OptimConfig", "PathsConfig",
    "SamplingConfig", "SplitConfig", "ValidationConfig", "smoke_config",
]
