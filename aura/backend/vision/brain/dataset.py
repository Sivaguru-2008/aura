"""The Brain Dataset: cached foundation studies to model-ready training samples.

One sample is one axial slice with every supervision signal the multi-task network
needs, all derived from the *same* final label — after cropping and augmentation, never
before. That ordering is the point of :meth:`BrainSliceDataset.__getitem__`: a random
crop can remove a small enhancing focus from the visible field, and a presence target
computed from the pre-crop label would then be teaching the network to report a tumour
it cannot see. Every target is recomputed from what the network actually receives.

Per-sample pipeline::

    memmap read (C,H,W float16) + label (H,W uint8)
        -> brain mask               from the cached background-zero intensities
        -> synthetic degradation    training only; supplies the quality target
        -> per-slice normalisation  z-score over brain voxels, per channel
        -> augmentation             image and label together, always
        -> fit to the network grid  pad or crop, image and label together
        -> targets                  recomputed from the final label

Normalisation is per slice, over brain voxels, per channel — not per volume. Two
measured reasons. The corpus arrives per-slice z-scored, so a volume-level statistic is
partly meaningless anyway (the per-slice scale factor is unrecoverable; see
:mod:`backend.vision.brain.io.brats_h5`), and the residual inter-slice scale spread is
a factor of 1.4-1.8 within a volume, which per-slice normalisation removes outright.
Over brain voxels rather than the whole frame because the frame is mostly air: including
it makes the statistics a function of how much of the head is in this slice.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from backend.core.shared.logging import get_logger
from backend.vision.brain.augment import SliceAugmenter
from backend.vision.brain.config import BrainVisionConfig
from backend.vision.brain.degradations import Degradation, DegradationSimulator
from backend.vision.brain.errors import CacheUnavailable
from backend.vision.brain.ingest import CacheManifest, load_manifest, load_slice_index
from backend.vision.brain.sampling import SliceTable
from backend.vision.brain.types import (
    CompositeRegion,
    FOREGROUND_REGIONS,
    SplitName,
    TumorGrade,
)

log = get_logger("vision.brain.dataset")

#: Order of the presence / size head outputs. Whole tumour first because it is the
#: number a triage decision keys off; the three primary classes follow.
TARGET_REGIONS: tuple[str, ...] = ("whole_tumor", "ncr_net", "edema", "enhancing")

#: Grade encoding for the held-out embedding probe. Never a training target.
_GRADE_CODES: dict[str, int] = {TumorGrade.HGG.value: 0, TumorGrade.LGG.value: 1,
                                TumorGrade.UNKNOWN.value: 2}


@dataclass(frozen=True)
class MorphologyLabeller:
    """Assigns each slice a tumour-morphology class for the contrastive objective.

    The class is ``(which subregions are present) x (how large the tumour is)``: eight
    presence patterns over three subregions, times three size buckets, plus one class
    for tumour-free. Twenty-five labels in principle, fewer in practice because several
    presence patterns are anatomically rare.

    Why this and not something learned: the embedding is supposed to cluster *similar
    tumour morphologies* and separate different ones, and a composition of subregions
    with a size is the most direct description of morphology available from the label
    without inventing a taxonomy. It is computed from the segmentation, so it costs
    nothing, and it is deliberately *not* the tumour grade — grade is held out so the
    embedding can be probed against something it never optimised for.

    Size boundaries come from the training split only. Fitting them on all slices would
    let the validation set's area distribution influence a training-time label, which is
    a small leak but a real one.
    """

    small_max: float
    medium_max: float

    @classmethod
    def fit(cls, areas: np.ndarray) -> "MorphologyLabeller":
        positive = areas[areas > 0]
        if positive.size < 3:
            return cls(small_max=1.0, medium_max=2.0)
        return cls(small_max=float(np.percentile(positive, 33.3)),
                   medium_max=float(np.percentile(positive, 66.7)))

    @property
    def num_classes(self) -> int:
        return 1 + 7 * 3            # tumour-free, plus 7 non-empty patterns x 3 sizes

    def __call__(self, areas: Sequence[float]) -> int:
        """``areas`` is ``(ncr_net, edema, enhancing)`` in pixels."""
        pattern = sum(1 << i for i, a in enumerate(areas) if a > 0)
        if pattern == 0:
            return 0
        total = float(sum(areas))
        bucket = 0 if total <= self.small_max else 1 if total <= self.medium_max else 2
        return 1 + (pattern - 1) * 3 + bucket

    def to_dict(self) -> dict[str, Any]:
        return {"num_classes": self.num_classes,
                "size_boundaries_px": [round(self.small_max, 1),
                                       round(self.medium_max, 1)],
                "definition": "(subregion presence pattern) x (size tertile)"}


class BrainSliceDataset(Dataset):
    """Axial brain-MRI slices with multi-task supervision, backed by the ingest cache.

    Memory-mapped: a subject's cached volume is opened on first access in whichever
    worker process needs it and never read whole. The map cache is per-instance and
    populated lazily precisely so that ``num_workers > 0`` works — a memmap handle
    created in the parent process and inherited through a fork or a spawn is a handle to
    a file the child did not open.
    """

    def __init__(self, config: BrainVisionConfig, table: SliceTable,
                 indices: np.ndarray, manifest: CacheManifest, *,
                 split: SplitName, morphology: MorphologyLabeller,
                 train: bool) -> None:
        self.config = config
        self.table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.manifest = manifest
        self.split = split
        self.morphology = morphology
        self.train = bool(train)

        self._volumes_dir = config.paths.cache_dir / "volumes"
        self._subject_ids = [record.subject_id for record in manifest.subjects]
        #: LRU of open memory maps, per process. Bounded — see :meth:`_memmap`.
        self._maps: "OrderedDict[int, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self.max_open_volumes = int(config.ingest.max_open_volumes)

        self.augmenter = SliceAugmenter(config.augmentation) if train else None
        degrade = (config.degradation.enabled
                   and (train or config.degradation.validate_on_degraded))
        self.degrader = DegradationSimulator(config.degradation) if degrade else None
        self._base_seed = int(config.optim.seed) + (0 if train else 9973)
        self._epoch = 0
        #: Draws served by *this* process. See :meth:`_rng_for`.
        self._draws = 0

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, Any]:
        import torch

        index = int(self.indices[position]) if position < self.indices.size else 0
        rng = self._rng_for(index)

        image, label = self._read(index)
        brain = image.max(axis=0) > 0

        base_quality = float(self.table.quality_score[index])
        degradation = Degradation.none(base_quality)
        if self.degrader is not None:
            forced = self._forced_artifact(index) if not self.train else None
            image, degradation = self.degrader(image, rng, base_quality=base_quality,
                                               force=forced)

        image = normalize_slice(image, brain)
        if self.augmenter is not None:
            image, label = self.augmenter(image, label, rng)
        image, label = fit_to_grid(image, label, self.config.model.input_size,
                                   rng=rng if self.train else None)

        image = np.ascontiguousarray(image, dtype=np.float32)
        label = np.ascontiguousarray(label, dtype=np.int64)
        areas = _class_areas(label)
        pixels = float(label.size)

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "presence": torch.from_numpy(_presence_target(areas)),
            "size": torch.from_numpy(_size_target(areas, pixels)),
            "quality": torch.tensor([degradation.target_quality], dtype=torch.float32),
            # Which artefact was applied (or "clean"). An auxiliary target: severity
            # pooled across five artefacts that move texture in opposite directions is
            # not one regression, and the type is what makes it conditional.
            "artifact": torch.tensor(degradation.index, dtype=torch.long),
            "morphology": torch.tensor(self.morphology(areas), dtype=torch.long),
            # Carried for the held-out probe and for per-subject metric grouping.
            # Never read by a loss — see the grade note in types.TumorGrade.
            "grade": torch.tensor(self._grade_code(index), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "subject_index": torch.tensor(int(self.table.subject_index[index]),
                                          dtype=torch.long),
            # Slice position within the subject's cached volume. Carried so an exported
            # embedding can be traced back to an anatomical location, which is what a
            # longitudinal or planning module needs from it.
            "cache_z": torch.tensor(int(self.table.cache_z[index]), dtype=torch.long),
            "severity": torch.tensor(degradation.severity, dtype=torch.float32),
            "degraded": torch.tensor(float(degradation.severity > 0.0),
                                     dtype=torch.float32),
        }

    # ------------------------------------------------------------------ #
    def _read(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        subject = int(self.table.subject_index[index])
        z = int(self.table.cache_z[index])
        images, labels = self._memmap(subject)
        return (np.asarray(images[z], dtype=np.float32),
                np.asarray(labels[z], dtype=np.uint8))

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open memory maps before this object is pickled to a worker.

        ``np.memmap`` does not pickle as a map — numpy reduces it to a plain array, so
        a populated cache would send whole 40 MB volumes to every worker process and
        turn a lazy read into an eager broadcast. The maps are per-process state and are
        reopened on first use in whichever process needs them.
        """
        state = dict(self.__dict__)
        state["_maps"] = OrderedDict()
        state["_draws"] = 0
        return state

    def _memmap(self, subject: int) -> tuple[np.ndarray, np.ndarray]:
        """Open (or reuse) a subject's cached volumes, keeping the cache bounded.

        The bound is not an optimisation. A training split holds 258 subjects at ~40 MB
        each, and a sampler that draws uniformly across them will, over one epoch, touch
        nearly all of them in every worker process. An unbounded cache therefore ends
        the epoch with ~10 GB of mapped views open *per worker*, and on Windows —
        where dataloader workers are separate processes and torch's collate allocates
        pagefile-backed shared segments to hand batches back — that exhausts the system
        commit limit and the run dies with ``error code 1455`` somewhere unrelated, in
        the middle of a validation pass.

        An LRU of :attr:`max_open_volumes` caps it. Reopening a memmap is a file handle
        and a page-table entry, not a read, so the eviction cost is negligible against
        the failure it prevents.
        """
        cached = self._maps.get(subject)
        if cached is not None:
            self._maps.move_to_end(subject)
            return cached
        while len(self._maps) >= self.max_open_volumes:
            self._maps.popitem(last=False)
        subject_id = self._subject_ids[subject]
        image_path = self._volumes_dir / f"{subject_id}.img.npy"
        label_path = self._volumes_dir / f"{subject_id}.seg.npy"
        if not image_path.exists() or not label_path.exists():
            raise CacheUnavailable(
                f"cached volumes for {subject_id} are missing",
                detail={"image": str(image_path), "label": str(label_path)})
        pair = (np.load(image_path, mmap_mode="r"), np.load(label_path, mmap_mode="r"))
        self._maps[subject] = pair
        return pair

    def _rng_for(self, index: int) -> np.random.Generator:
        """A generator seeded per sample, per worker, and per draw.

        **Validation** is seeded from the sample index alone. That is what makes the
        degraded-validation numbers comparable between epochs: the quality head is
        measured against the same artefacts at the same severities every cycle, so a
        change in the score is a change in the model.

        **Training** must differ every time the same slice is drawn — a slice the hard-
        example miner picks four times in an epoch should not receive the same rotation
        four times. The obvious source of variation, an epoch counter, does not work:
        with ``persistent_workers=True`` the trainer's ``set_epoch`` runs in the parent
        process and never reaches the children, so every worker would stay on epoch 0
        for the whole run. A per-worker draw counter does work, because it lives in the
        process actually doing the drawing.
        """
        if not self.train:
            return np.random.default_rng([self._base_seed, index])
        self._draws += 1
        return np.random.default_rng(
            [self._base_seed, index, self._worker_salt(), self._draws, self._epoch])

    @staticmethod
    def _worker_salt() -> int:
        """Distinguish the streams of concurrent dataloader workers."""
        import torch.utils.data

        info = torch.utils.data.get_worker_info()
        return int(info.id) + 1 if info is not None else 0

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch. Called by the trainer at each epoch boundary.

        It contributes to the augmentation seed in the single-process case and is
        carried on the object for the run record. It is deliberately *not* the only
        source of per-draw variation — see :meth:`_rng_for` for why it cannot be.
        """
        self._epoch = int(epoch)

    def _forced_artifact(self, index: int) -> str | None:
        """Deterministic artefact assignment for validation.

        A fixed rotation through the artefact pool rather than a random draw, so every
        validation cycle scores the quality head on the same mix and the number moves
        only when the model does. Two thirds of validation slices are left clean, which
        is roughly the balance the head sees in training.
        """
        pool = self.degrader.artifacts if self.degrader else ()
        if not pool or index % 3 != 0:
            return None
        return pool[(index // 3) % len(pool)]

    def _grade_code(self, index: int) -> int:
        if self.table.subject_grade.size == 0:
            return _GRADE_CODES[TumorGrade.UNKNOWN.value]
        grade = str(self.table.subject_grade[int(self.table.subject_index[index])])
        return _GRADE_CODES.get(grade, _GRADE_CODES[TumorGrade.UNKNOWN.value])

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        area = self.table.area_total[self.indices]
        return {
            "split": self.split.value,
            "slices": int(self.indices.size),
            "subjects": int(np.unique(self.table.subject_index[self.indices]).size),
            "positive_slices": int((area > 0).sum()),
            "positive_fraction": (round(float((area > 0).mean()), 4)
                                  if self.indices.size else 0.0),
            "augmentation": self.augmenter.describe() if self.augmenter else None,
            "degradation": ({"enabled": True, "artifacts": list(self.degrader.artifacts),
                             "probability": self.config.degradation.probability}
                            if self.degrader else {"enabled": False}),
            "morphology": self.morphology.to_dict(),
            "input_size": list(self.config.model.input_size),
        }


# --------------------------------------------------------------------------- #
# Sample construction helpers
# --------------------------------------------------------------------------- #
def normalize_slice(image: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Z-score each channel over brain voxels; force background to exactly zero.

    A channel with no brain voxels, or with no variation across them, is returned as
    zeros rather than divided by an epsilon: an empty slice normalised by a near-zero
    standard deviation becomes amplified noise, which is precisely the pathology this
    corpus already exhibits and which the ingest filters out.
    """
    result = np.zeros_like(image, dtype=np.float32)
    if not brain.any():
        return result
    for channel in range(image.shape[0]):
        values = image[channel][brain]
        std = float(values.std())
        if std < 1e-6:
            continue
        result[channel][brain] = (values - float(values.mean())) / std
    return result


def fit_to_grid(image: np.ndarray, label: np.ndarray, size: Sequence[int],
                rng: np.random.Generator | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Pad or crop ``(C, H, W)`` and ``(H, W)`` to exactly ``size``.

    Padding is symmetric and zero-valued, which after normalisation is the background
    value, so a padded region is indistinguishable from air rather than from a bright
    artefact. Cropping is random when ``rng`` is supplied (training) and centred
    otherwise (validation, inference), because a validation number that moves with the
    crop position is not measuring the model.
    """
    target_h, target_w = int(size[0]), int(size[1])
    height, width = label.shape

    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h or pad_w:
        before_h, before_w = pad_h // 2, pad_w // 2
        image = np.pad(image, ((0, 0), (before_h, pad_h - before_h),
                               (before_w, pad_w - before_w)))
        label = np.pad(label, ((before_h, pad_h - before_h),
                               (before_w, pad_w - before_w)))
        height, width = label.shape

    offset_h = _crop_offset(height, target_h, rng)
    offset_w = _crop_offset(width, target_w, rng)
    return (image[:, offset_h:offset_h + target_h, offset_w:offset_w + target_w],
            label[offset_h:offset_h + target_h, offset_w:offset_w + target_w])


def _crop_offset(extent: int, target: int, rng: np.random.Generator | None) -> int:
    slack = extent - target
    if slack <= 0:
        return 0
    return int(rng.integers(0, slack + 1)) if rng is not None else slack // 2


def _class_areas(label: np.ndarray) -> tuple[float, float, float]:
    return tuple(float((label == region.value).sum())    # type: ignore[return-value]
                 for region in FOREGROUND_REGIONS)


def _presence_target(areas: Sequence[float]) -> np.ndarray:
    """``[whole tumour, NCR/NET, oedema, enhancing]`` as 0/1 floats."""
    whole = 1.0 if sum(areas) > 0 else 0.0
    return np.asarray([whole] + [1.0 if a > 0 else 0.0 for a in areas],
                      dtype=np.float32)


def _size_target(areas: Sequence[float], pixels: float) -> np.ndarray:
    """Log-area, scaled to roughly [0, 1] by the largest area a slice could hold.

    Log rather than linear because tumour areas span three orders of magnitude and a
    linear target makes the loss indifferent to everything below a few hundred pixels —
    which is most of the clinically interesting range. Scaled rather than raw so the
    size head's loss is on the same numerical footing as the other heads' and the
    configured weights mean what they say.
    """
    scale = float(np.log1p(pixels)) or 1.0
    values = [float(np.log1p(sum(areas))) / scale]
    values += [float(np.log1p(a)) / scale for a in areas]
    return np.asarray(values, dtype=np.float32)


def decode_size(prediction: np.ndarray, pixels: float) -> np.ndarray:
    """Invert :func:`_size_target` — scaled log-area back to pixels."""
    scale = float(np.log1p(pixels)) or 1.0
    return np.expm1(np.clip(prediction, 0.0, 1.5) * scale)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_slice_table(config: BrainVisionConfig,
                      manifest: CacheManifest | None = None) -> tuple[SliceTable,
                                                                     CacheManifest]:
    """Load the cache's slice index and attach each subject's split and grade."""
    manifest = manifest or load_manifest(config)
    columns = load_slice_index(config)
    splits = np.asarray([record.split.value for record in manifest.subjects])
    grades = np.asarray([record.grade.value for record in manifest.subjects])
    table = SliceTable(
        subject_index=columns["subject_index"].astype(np.int64),
        cache_z=columns["cache_z"].astype(np.int64),
        source_slice=columns["source_slice"].astype(np.int64),
        brain_voxels=columns["brain_voxels"].astype(np.int64),
        area_ncr_net=columns["area_ncr_net"].astype(np.int64),
        area_edema=columns["area_edema"].astype(np.int64),
        area_enhancing=columns["area_enhancing"].astype(np.int64),
        quality_score=columns["quality_score"].astype(np.float32),
        subject_split=splits,
        subject_grade=grades,
    )
    return table, manifest


def build_datasets(config: BrainVisionConfig
                   ) -> tuple[dict[SplitName, BrainSliceDataset], SliceTable,
                              CacheManifest, MorphologyLabeller]:
    """Build the train/validation/test datasets from the ingest cache.

    The morphology labeller is fitted on the training split alone and shared with the
    others, so validation samples are labelled by a rule they did not contribute to.
    """
    table, manifest = build_slice_table(config)
    train_indices = table.split_indices(SplitName.TRAIN.value)
    morphology = MorphologyLabeller.fit(table.area_total[train_indices])

    datasets: dict[SplitName, BrainSliceDataset] = {}
    for split in SplitName:
        indices = table.split_indices(split.value)
        datasets[split] = BrainSliceDataset(
            config, table, indices, manifest, split=split, morphology=morphology,
            train=(split is SplitName.TRAIN))
        log.info("dataset built", extra={"context": datasets[split].describe()})
    return datasets, table, manifest, morphology


def composite_presence(label: np.ndarray, region: CompositeRegion) -> np.ndarray:
    """Boolean mask of a BraTS composite region, from a dense class label."""
    mask = np.zeros(label.shape, dtype=bool)
    for member in region.members:
        mask |= label == member.value
    return mask


__all__ = [
    "BrainSliceDataset", "MorphologyLabeller", "TARGET_REGIONS", "build_datasets",
    "build_slice_table", "composite_presence", "decode_size", "fit_to_grid",
    "normalize_slice",
]
