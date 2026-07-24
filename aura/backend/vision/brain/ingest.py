"""BraTS corpus -> MRI Foundation Layer -> cached, model-ready brain studies.

This is the ``BraTS Dataset -> MRI Foundation Layer -> Study -> Brain Dataset`` leg of
the pipeline, run once and cached. The foundation layer is not bypassed and not
re-implemented: every subject is read into :class:`RawSeries` objects and handed to
:meth:`~backend.foundation.mri.pipeline.MRIFoundationPipeline.run_series`, which
reorients it to canonical RAS, checks its resolution and field of view, measures seven
quality checks, estimates a foreground mask, and records every stage it ran. What comes
out is a real :class:`~backend.foundation.mri.study.FoundationStudy` per subject, kept
alongside the voxels as JSON.

Why cache at all
----------------
Running the foundation layer inside the training loop would mean re-reading 155 HDF5
files and recomputing an Otsu mask over 8.9 million voxels for every subject, every
epoch. The cache turns a ~40-minute one-off into a memory-mapped array the sampler can
index in microseconds, and — more importantly — it makes the preprocessing an artefact
that can be inspected, versioned, and diffed rather than a side effect of a training
run nobody can reproduce.

Three decisions in here are worth stating, because each one is a place where the
obvious choice would corrupt the labels.

**The crop box is shared across sequences.** The foundation layer's own cropping stage
is switched off (``crop_to_mask=False``) and the crop is applied here instead, from the
*union* of the four sequences' foreground masks. Cropping each sequence by its own mask
would shift them relative to one another by a voxel or two, and relative to the label by
the same amount — a misalignment that no metric in this package would flag because every
metric compares the prediction to the label, and both would have moved.

**The label follows the image through the same transform.** ``CanonicalOrientation`` is
a permutation and some flips; applying it to the image and not the label produces a
model trained on mirrored ground truth, which converges perfectly well to something
useless. :func:`_align_label` puts the label through
:func:`~backend.foundation.mri.geometry.to_canonical` with the same source affine, and
then asserts the shapes agree.

**Intensity normalisation is deliberately not run here.** The corpus arrives per-slice
z-scored and the reader restores background to exactly zero (see
:mod:`backend.vision.brain.io.brats_h5`). That zero is what the dataset's per-slice
brain-voxel normalisation keys off, and a volume-level z-score applied first would
destroy it while changing nothing the network could benefit from. The stage is recorded
as ``skipped`` with that reason in every subject's processing history rather than
silently omitted.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.foundation.mri import (
    FoundationConfig,
    FoundationStudy,
    MRIFoundationPipeline,
    StandardizationConfig,
    StandardizedSeries,
)
from backend.foundation.mri.geometry import to_canonical
from backend.foundation.mri.study import ProcessingHistory, StepTimer, step
from backend.foundation.mri.types import NormalizationMethod, StepStatus
from backend.vision.brain.config import BrainVisionConfig
from backend.vision.brain.errors import CacheUnavailable, CorpusIntegrityError
from backend.vision.brain.io.brats_h5 import (
    BratsCorpusIndex,
    BratsH5Reader,
    BratsSubject,
    ChannelVerification,
    SubjectVolumes,
)
from backend.vision.brain.types import (
    CACHE_VERSION,
    CompositeRegion,
    FOREGROUND_REGIONS,
    SplitName,
    TumorGrade,
    TumorRegion,
)

log = get_logger("vision.brain.ingest")

#: Filenames inside the cache directory.
_VOLUMES_SUBDIR = "volumes"
_SLICE_INDEX_NAME = "slice_index.npz"

#: Columns of the slice index. Parallel arrays rather than a list of dicts: 45 000
#: slices as JSON objects is a 12 MB file that takes a second to parse at the start of
#: every training; as six numpy arrays it is 900 KB and loads instantly.
_SLICE_COLUMNS: tuple[str, ...] = (
    "subject_index", "cache_z", "source_slice", "brain_voxels",
    "area_ncr_net", "area_edema", "area_enhancing", "quality_score")


@dataclass
class SubjectRecord:
    """Everything the cache knows about one ingested subject."""

    subject_id: str
    volume_id: int
    grade: TumorGrade
    split: SplitName
    #: Cached volume shape, ``(Z, C, H, W)``.
    shape: tuple[int, int, int, int]
    #: Crop box applied to the source grid, as ``[[i0, i1], [j0, j1], [k0, k1]]``.
    crop_box: tuple[tuple[int, int], ...]
    source_shape: tuple[int, int, int]
    slices_cached: int
    slices_available: int
    tumor_voxels: int
    quality_score: float
    quality_verdict: str
    sequences: tuple[str, ...]
    channel_check: dict[str, Any]
    age_years: float | None = None
    survival_days: int | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id, "volume_id": self.volume_id,
            "grade": self.grade.value, "split": self.split.value,
            "shape": list(self.shape),
            "crop_box": [list(b) for b in self.crop_box],
            "source_shape": list(self.source_shape),
            "slices_cached": self.slices_cached,
            "slices_available": self.slices_available,
            "tumor_voxels": self.tumor_voxels,
            "quality_score": round(self.quality_score, 4),
            "quality_verdict": self.quality_verdict,
            "sequences": list(self.sequences),
            "channel_check": dict(self.channel_check),
            "age_years": self.age_years, "survival_days": self.survival_days,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectRecord":
        return cls(
            subject_id=data["subject_id"], volume_id=int(data["volume_id"]),
            grade=TumorGrade(data["grade"]), split=SplitName(data["split"]),
            shape=tuple(int(v) for v in data["shape"]),        # type: ignore[arg-type]
            crop_box=tuple(tuple(int(v) for v in b)            # type: ignore[arg-type]
                           for b in data["crop_box"]),
            source_shape=tuple(int(v) for v in data["source_shape"]),  # type: ignore
            slices_cached=int(data["slices_cached"]),
            slices_available=int(data["slices_available"]),
            tumor_voxels=int(data["tumor_voxels"]),
            quality_score=float(data["quality_score"]),
            quality_verdict=str(data["quality_verdict"]),
            sequences=tuple(data.get("sequences", ())),
            channel_check=dict(data.get("channel_check", {})),
            age_years=data.get("age_years"), survival_days=data.get("survival_days"),
            warnings=tuple(data.get("warnings", ())))


@dataclass
class CacheManifest:
    """The cache's own description of itself.

    Loaded before anything reads a voxel. It carries the corpus root, the ingest
    configuration, the channel-verification result, and the subject/split table, so a
    cache found on disk months later can be matched against the code that wrote it and
    the corpus it came from.
    """

    cache_version: str
    brain_vision_version: str
    foundation_version: str
    created_at: str
    corpus_root: str
    modalities: list[dict[str, Any]]
    ingest_config: dict[str, Any]
    split_config: dict[str, Any]
    subjects: list[SubjectRecord] = field(default_factory=list)
    channel_verification: dict[str, Any] = field(default_factory=dict)
    #: Corpus-level caveats that must reach the model card.
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_version": self.cache_version,
            "brain_vision_version": self.brain_vision_version,
            "foundation_version": self.foundation_version,
            "created_at": self.created_at,
            "corpus_root": self.corpus_root,
            "modalities": list(self.modalities),
            "ingest_config": dict(self.ingest_config),
            "split_config": dict(self.split_config),
            "channel_verification": dict(self.channel_verification),
            "caveats": list(self.caveats),
            "subject_count": len(self.subjects),
            "subjects": [s.to_dict() for s in self.subjects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheManifest":
        return cls(
            cache_version=data["cache_version"],
            brain_vision_version=data.get("brain_vision_version", "unknown"),
            foundation_version=data.get("foundation_version", "unknown"),
            created_at=data.get("created_at", ""),
            corpus_root=data.get("corpus_root", ""),
            modalities=list(data.get("modalities", [])),
            ingest_config=dict(data.get("ingest_config", {})),
            split_config=dict(data.get("split_config", {})),
            subjects=[SubjectRecord.from_dict(s) for s in data.get("subjects", [])],
            channel_verification=dict(data.get("channel_verification", {})),
            caveats=list(data.get("caveats", [])))

    def by_split(self, split: SplitName) -> list[SubjectRecord]:
        return [s for s in self.subjects if s.split is split]


class BrainCorpusIngestor:
    """Turns a BraTS corpus into a cache of standardised, foundation-checked studies."""

    def __init__(self, config: BrainVisionConfig,
                 *, pipeline: MRIFoundationPipeline | None = None,
                 reader: BratsH5Reader | None = None) -> None:
        self.config = config
        self.reader = reader or BratsH5Reader(config.model.modalities)
        self.pipeline = pipeline or MRIFoundationPipeline(self._foundation_config())

    # ------------------------------------------------------------------ #
    @staticmethod
    def _foundation_config() -> FoundationConfig:
        """Foundation configuration for corpus ingest, with every choice justified.

        Each stage that is switched off is switched off for a stated reason, and the
        foundation layer records it as ``skipped`` — materially different from
        ``unavailable`` (no backend) and from silence.
        """
        return FoundationConfig(
            standardization=StandardizationConfig(
                target_orientation="RAS",
                target_spacing_mm=(1.0, 1.0, 1.0),
                # BraTS is distributed on a 1 mm isotropic grid, so this is a no-op
                # that we still run: it is the check that the grid really is what the
                # convention says, and a no_op in the history is evidence of that.
                resample_order=1,
                # See the module docstring. The dataset normalises per slice over brain
                # voxels; a volume z-score here would destroy the background-zero
                # property that normalisation depends on.
                normalization=NormalizationMethod.NONE,
                # The challenge organisers already ran N4 on this corpus.
                bias_correction=False,
                # And already skull-stripped it. The foreground mask below therefore
                # coincides with a brain mask for this corpus — but the foundation layer
                # cannot verify that from the volume alone and correctly keeps labelling
                # it FOREGROUND_HEURISTIC.
                skull_strip=False,
                foreground_mask=True,
                # Cropping happens here, from the union of all four sequences' masks,
                # so the sequences and the label stay on one grid.
                crop_to_mask=False,
            ),
            reject_on_quality=False,
        )

    # ------------------------------------------------------------------ #
    def run(self, subjects: Sequence[BratsSubject] | None = None) -> CacheManifest:
        """Ingest the corpus. Returns the manifest it wrote."""
        paths = self.config.paths
        if paths.corpus_root is None:
            raise CorpusIntegrityError(
                "no BraTS corpus root is configured",
                detail={"hint": "set AURA_BRATS_ROOT or PathsConfig(corpus_root=...)"})
        paths.ensure()
        (paths.cache_dir / _VOLUMES_SUBDIR).mkdir(parents=True, exist_ok=True)

        index = BratsCorpusIndex(paths.corpus_root)
        pool = list(subjects if subjects is not None else index.subjects())
        if self.config.ingest.max_subjects is not None:
            pool = pool[:self.config.ingest.max_subjects]
        if not pool:
            raise CorpusIntegrityError("the corpus contains no subjects",
                                       detail={"root": str(paths.corpus_root)})

        splits = assign_splits(pool, self.config.split)
        previous = self._previous_records()
        records: list[SubjectRecord] = []
        slice_rows: list[list[float]] = []
        verifications: list[ChannelVerification] = []
        reused = 0
        started = time.perf_counter()

        for position, subject in enumerate(pool):
            if not self.config.ingest.overwrite:
                carried = self._reuse(subject, previous, len(records))
                if carried is not None:
                    record, rows, verification = carried
                    records.append(record)
                    slice_rows.extend(rows)
                    verifications.append(verification)
                    reused += 1
                    continue
            try:
                record, rows, verification = self._ingest_subject(
                    subject, splits[subject.subject_id], len(records))
            except CorpusIntegrityError as exc:
                log.error("subject skipped", extra={"context": {
                    "subject": subject.subject_id, "reason": exc.reason}})
                continue
            records.append(record)
            slice_rows.extend(rows)
            verifications.append(verification)
            if (position + 1) % 25 == 0 or position + 1 == len(pool):
                elapsed = time.perf_counter() - started
                log.info("ingest progress", extra={"context": {
                    "done": position + 1, "total": len(pool),
                    "slices": len(slice_rows),
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(elapsed / (position + 1) * (len(pool) - position - 1),
                                   1)}})

        summary = self._verification_summary(verifications)
        if (self.config.ingest.verify_channel_assignment
                and summary["evaluated"] > 0
                and summary["agreement"] < self.config.ingest.min_channel_agreement):
            raise CorpusIntegrityError(
                "the modality channel assignment failed verification on "
                f"{1 - summary['agreement']:.1%} of testable subjects; the cache was "
                "not completed because a wrong channel order trains a model on the "
                "wrong contrasts without any downstream symptom",
                detail=summary)

        manifest = CacheManifest(
            cache_version=CACHE_VERSION,
            brain_vision_version=__import__(
                "backend.vision.brain.types", fromlist=["BRAIN_VISION_VERSION"]
            ).BRAIN_VISION_VERSION,
            foundation_version=__import__(
                "backend.foundation.mri.types", fromlist=["FOUNDATION_VERSION"]
            ).FOUNDATION_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            corpus_root=str(paths.corpus_root),
            modalities=[m.to_dict() for m in self.config.model.modalities],
            ingest_config=self.config.to_dict()["ingest"],
            split_config=self.config.to_dict()["split"],
            subjects=records,
            channel_verification=summary,
            caveats=_CORPUS_CAVEATS,
        )
        self._write_manifest(manifest, slice_rows)
        log.info("ingest complete", extra={"context": {
            "subjects": len(records), "reused_from_cache": reused,
            "slices": len(slice_rows),
            "seconds": round(time.perf_counter() - started, 1),
            "cache": str(paths.cache_dir)}})
        return manifest

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    def _previous_records(self) -> dict[str, SubjectRecord]:
        """Subject records from an earlier ingest, if one is on disk.

        Ingest takes 40 minutes over the full corpus; an interruption at minute 35
        should not cost 35 minutes. ``overwrite=False`` (the default) carries forward
        any subject whose cached volumes and study description are both present.
        """
        path = self.config.paths.manifest_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("cache_version", "").split(".")[0] != CACHE_VERSION.split(".")[0]:
                return {}
            return {r["subject_id"]: SubjectRecord.from_dict(r)
                    for r in data.get("subjects", [])}
        except Exception:
            log.warning("the existing manifest could not be read; re-ingesting all",
                        extra={"context": {"path": str(path)}})
            return {}

    def _reuse(self, subject: BratsSubject, previous: dict[str, SubjectRecord],
               subject_index: int
               ) -> tuple[SubjectRecord, list[list[float]], ChannelVerification] | None:
        """Carry an already-cached subject forward, rebuilding its slice rows.

        The rows are recomputed from the cached label rather than stored in the
        manifest, so a resumed ingest and a fresh one produce byte-identical indices.
        ``subject_index`` is re-derived because it is a *position* in the new manifest,
        which a partial re-ingest can change.
        """
        record = previous.get(subject.subject_id)
        if record is None:
            return None
        volumes = self.config.paths.cache_dir / _VOLUMES_SUBDIR
        image_path = volumes / f"{subject.subject_id}.img.npy"
        label_path = volumes / f"{subject.subject_id}.seg.npy"
        study_path = self.config.paths.studies_dir / f"{subject.subject_id}.json"
        if not (image_path.exists() and label_path.exists() and study_path.exists()):
            return None

        try:
            images = np.load(image_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            kept = json.loads(study_path.read_text(encoding="utf-8")
                              )["cache"]["slices_kept"]
        except Exception:
            log.warning("a cached subject could not be reused; re-ingesting it",
                        extra={"context": {"subject": subject.subject_id}})
            return None
        if labels.shape[0] != len(kept) or images.shape[0] != labels.shape[0]:
            return None

        rows: list[list[float]] = []
        for position, source_slice in enumerate(kept):
            plane = np.asarray(labels[position])
            brain = int(np.count_nonzero(np.asarray(images[position]).max(axis=0) > 0))
            rows.append([
                float(subject_index), float(position), float(source_slice), float(brain),
                float((plane == TumorRegion.NECROTIC_CORE.value).sum()),
                float((plane == TumorRegion.EDEMA.value).sum()),
                float((plane == TumorRegion.ENHANCING.value).sum()),
                float(record.quality_score),
            ])
        check = dict(record.channel_check)
        verification = ChannelVerification(
            subject_id=subject.subject_id,
            evaluated=bool(check.get("evaluated", False)),
            agrees=bool(check.get("agrees", False)),
            channel_contrast=tuple(check.get("channel_contrast", ())),
            voxels_tested=int(check.get("voxels_tested", 0)))
        return record, rows, verification

    # ------------------------------------------------------------------ #
    def _ingest_subject(self, subject: BratsSubject, split: SplitName,
                        subject_index: int
                        ) -> tuple[SubjectRecord, list[list[float]], ChannelVerification]:
        """Read, standardise, crop, and cache one subject."""
        paths = self.config.paths
        image_path = paths.cache_dir / _VOLUMES_SUBDIR / f"{subject.subject_id}.img.npy"
        label_path = paths.cache_dir / _VOLUMES_SUBDIR / f"{subject.subject_id}.seg.npy"

        volumes = self.reader.read_subject(subject)
        study, label, history = self._standardize(volumes)

        images = np.stack([s.volume.array for s in study.series], axis=0)
        if label.shape != images.shape[1:]:
            raise CorpusIntegrityError(
                f"{subject.subject_id}: the label is {label.shape} but the "
                f"standardised image grid is {images.shape[1:]}",
                detail={"label_shape": list(label.shape),
                        "image_shape": list(images.shape)})

        brain = self._union_mask(study, images)
        box = _bounding_box(brain, margin=self.config.ingest.crop_margin_voxels,
                            shape=images.shape[1:])
        if self.config.ingest.crop_to_brain:
            selector = (slice(None),) + box
            images = images[selector]
            label = label[box]
            brain = brain[box]

        keep = self._select_slices(brain)
        if keep.size == 0:
            raise CorpusIntegrityError(
                f"{subject.subject_id}: no slice reaches the minimum brain fraction "
                f"of {self.config.ingest.min_brain_fraction}",
                detail={"min_brain_fraction": self.config.ingest.min_brain_fraction})

        # (C, X, Y, Z) -> (Z, C, X, Y): one slice is then one contiguous read.
        cached_images = np.ascontiguousarray(
            np.transpose(images[..., keep], (3, 0, 1, 2)))
        cached_label = np.ascontiguousarray(np.transpose(label[..., keep], (2, 0, 1)))
        dtype = np.float16 if self.config.ingest.store_float16 else np.float32
        _write_npy(image_path, cached_images.astype(dtype, copy=False))
        _write_npy(label_path, cached_label.astype(np.uint8, copy=False))

        brain_kept = brain[..., keep]
        rows: list[list[float]] = []
        for position, source_slice in enumerate(keep.tolist()):
            plane = cached_label[position]
            rows.append([
                float(subject_index),
                float(position),
                float(source_slice),
                float(brain_kept[..., position].sum()),
                float((plane == TumorRegion.NECROTIC_CORE.value).sum()),
                float((plane == TumorRegion.EDEMA.value).sum()),
                float((plane == TumorRegion.ENHANCING.value).sum()),
                float(study.primary.quality.quality_score if study.primary else 0.0),
            ])

        study_json = study.to_dict()
        study_json["ingest_history"] = history.to_dict()
        study_json["label"] = {
            "source": "BraTS2020 expert consensus segmentation",
            "classes": {str(r.value): r.label for r in TumorRegion},
            "tumor_voxels": int((cached_label > 0).sum()),
            "per_class_voxels": {r.name.lower(): int((cached_label == r.value).sum())
                                 for r in FOREGROUND_REGIONS},
        }
        study_json["cache"] = {
            "image_file": image_path.name, "label_file": label_path.name,
            "shape": list(cached_images.shape), "dtype": str(dtype(0).dtype),
            "crop_box": [[int(s.start), int(s.stop)] for s in box],
            "slices_kept": keep.tolist(),
        }
        (paths.studies_dir / f"{subject.subject_id}.json").write_text(
            json.dumps(study_json, indent=2, default=str), encoding="utf-8")

        record = SubjectRecord(
            subject_id=subject.subject_id,
            volume_id=subject.volume_id,
            grade=subject.grade,
            split=split,
            shape=tuple(int(n) for n in cached_images.shape),   # type: ignore[arg-type]
            crop_box=tuple((int(s.start), int(s.stop)) for s in box),
            source_shape=volumes.shape,
            slices_cached=int(keep.size),
            slices_available=int(volumes.images.shape[3]),
            tumor_voxels=int((cached_label > 0).sum()),
            quality_score=float(study.primary.quality.quality_score
                                if study.primary else 0.0),
            quality_verdict=study.verdict.value,
            sequences=tuple(s.sequence.sequence.value for s in study.series),
            channel_check=volumes.verification.to_dict(),
            age_years=subject.age_years,
            survival_days=subject.survival_days,
            warnings=tuple(volumes.warnings),
        )
        return record, rows, volumes.verification

    # ------------------------------------------------------------------ #
    def _standardize(self, volumes: SubjectVolumes
                     ) -> tuple[FoundationStudy, np.ndarray, ProcessingHistory]:
        """Run the foundation pipeline on each sequence and carry the label with it."""
        from backend.foundation.mri.io.base import RawSeries

        history = ProcessingHistory()
        timer = StepTimer()
        raws: list[RawSeries] = self.reader.to_raw_series(volumes)
        series: list[StandardizedSeries] = [self.pipeline.run_series(raw)
                                            for raw in raws]
        history.record(step(
            "foundation_standardisation", StepStatus.APPLIED, timer,
            message=f"{len(series)} sequence(s) standardised through the MRI "
                    "Foundation Layer",
            implementation=type(self.pipeline).__name__,
            parameters={"sequences": [s.sequence.sequence.value for s in series],
                        "quality": [round(s.quality.quality_score, 3)
                                    for s in series]}))

        label, note = _align_label(volumes.label, raws[0].geometry.affine,
                                   series[0].volume.shape)
        history.record(step("label_alignment", StepStatus.APPLIED,
                            message=note, implementation="to_canonical",
                            parameters={"shape": list(label.shape)}))
        history.record(step(
            "intensity_normalization", StepStatus.SKIPPED,
            message="not run at ingest by design: the corpus arrives per-slice "
                    "z-scored with background restored to exactly zero, and the "
                    "training dataset normalises per slice over brain voxels. A "
                    "volume z-score here would destroy the background-zero property "
                    "without changing what the network sees.",
            implementation="backend.vision.brain.ingest"))

        study = FoundationStudy(
            study_id=volumes.subject.subject_id,
            series=tuple(series),
            source_format=raws[0].source_format,
            warnings=volumes.warnings,
            history=history,
        )
        return study, label, history

    @staticmethod
    def _union_mask(study: FoundationStudy, images: np.ndarray) -> np.ndarray:
        """Foreground mask shared by every sequence.

        The union rather than an intersection: a voxel that is brain in any sequence is
        brain, and a sequence whose mask happens to threshold a dark region away must
        not be allowed to crop it out of the others. Falls back to a positive-intensity
        test when the foundation layer produced no mask — which is exact for this
        corpus, since the reader restores background to zero.
        """
        mask: np.ndarray | None = None
        for series in study.series:
            if series.brain_mask.mask is None:
                continue
            mask = (series.brain_mask.mask if mask is None
                    else mask | series.brain_mask.mask)
        if mask is None:
            mask = np.any(images > 0, axis=0)
        return mask

    def _select_slices(self, brain: np.ndarray) -> np.ndarray:
        """Indices of slices whose brain fraction reaches the configured floor."""
        per_slice = brain.reshape(-1, brain.shape[2]).mean(axis=0)
        return np.flatnonzero(per_slice >= self.config.ingest.min_brain_fraction)

    @staticmethod
    def _verification_summary(results: Iterable[ChannelVerification]) -> dict[str, Any]:
        results = list(results)
        evaluated = [r for r in results if r.evaluated]
        agreed = [r for r in evaluated if r.agrees]
        return {
            "subjects": len(results),
            "evaluated": len(evaluated),
            "agreed": len(agreed),
            "agreement": (len(agreed) / len(evaluated)) if evaluated else None,
            "test": ("the post-contrast channel must maximise (mean(enhancing) - "
                     "mean(necrotic)) / std(brain) across the four image channels — "
                     "the definition of gadolinium enhancement"),
            "not_evaluated_reason": ("subjects with fewer than 200 enhancing or 200 "
                                     "necrotic voxels carry no signal for this test"),
            "disagreed_subjects": [r.subject_id for r in evaluated if not r.agrees][:20],
        }

    def _write_manifest(self, manifest: CacheManifest,
                        slice_rows: Sequence[Sequence[float]]) -> None:
        paths = self.config.paths
        paths.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
        table = (np.asarray(slice_rows, dtype=np.float64) if slice_rows
                 else np.zeros((0, len(_SLICE_COLUMNS)), dtype=np.float64))
        np.savez_compressed(
            paths.cache_dir / _SLICE_INDEX_NAME,
            columns=np.asarray(_SLICE_COLUMNS),
            subject_index=table[:, 0].astype(np.int32),
            cache_z=table[:, 1].astype(np.int32),
            source_slice=table[:, 2].astype(np.int32),
            brain_voxels=table[:, 3].astype(np.int32),
            area_ncr_net=table[:, 4].astype(np.int32),
            area_edema=table[:, 5].astype(np.int32),
            area_enhancing=table[:, 6].astype(np.int32),
            quality_score=table[:, 7].astype(np.float32),
        )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def assign_splits(subjects: Sequence[BratsSubject], config) -> dict[str, SplitName]:
    """Partition subjects into train/val/test, stratified by grade.

    Deterministic given the seed, and *by subject*: every slice of a subject lands in
    the same split. The corpus is 293 high-grade to 76 low-grade, so an unstratified
    draw of a 15% test set can plausibly return 5 low-grade subjects and make the
    grade probe noise.
    """
    rng = np.random.default_rng(config.seed)
    groups: dict[str, list[BratsSubject]] = {}
    for subject in subjects:
        key = subject.grade.value if config.stratify_by_grade else "all"
        groups.setdefault(key, []).append(subject)

    assignment: dict[str, SplitName] = {}
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda s: s.volume_id)
        order = rng.permutation(len(members))
        n_train = int(round(len(members) * config.train_fraction))
        n_val = int(round(len(members) * config.val_fraction))
        # Guarantee a non-empty validation split whenever there is more than one
        # subject: a run with an empty val split reports no metrics and looks fine.
        if len(members) > 1:
            n_train = min(n_train, len(members) - 1)
            n_val = max(1, min(n_val, len(members) - n_train))
        for position, index in enumerate(order):
            if position < n_train:
                split = SplitName.TRAIN
            elif position < n_train + n_val:
                split = SplitName.VAL
            else:
                split = SplitName.TEST
            assignment[members[index].subject_id] = split
    return assignment


# --------------------------------------------------------------------------- #
# Cache access
# --------------------------------------------------------------------------- #
def load_manifest(config: BrainVisionConfig) -> CacheManifest:
    """Read the cache manifest, refusing a version this code cannot interpret."""
    path = config.paths.manifest_path
    if not path.exists():
        raise CacheUnavailable(
            f"no ingest cache was found at {path}",
            detail={"hint": "run `python -m backend.vision.brain.cli ingest` first"})
    manifest = CacheManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if manifest.cache_version.split(".")[0] != CACHE_VERSION.split(".")[0]:
        raise CacheUnavailable(
            f"the cache was written by layout version {manifest.cache_version}; this "
            f"code reads {CACHE_VERSION}",
            detail={"path": str(path)})
    return manifest


def load_slice_index(config: BrainVisionConfig) -> dict[str, np.ndarray]:
    """Read the per-slice index as a dict of parallel arrays."""
    path = config.paths.cache_dir / _SLICE_INDEX_NAME
    if not path.exists():
        raise CacheUnavailable(f"no slice index was found at {path}",
                               detail={"path": str(path)})
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in _SLICE_COLUMNS}


def composite_area(areas: dict[str, np.ndarray], region: CompositeRegion) -> np.ndarray:
    """Total pixel area of a BraTS composite region, from the per-class areas."""
    keys = {TumorRegion.NECROTIC_CORE: "area_ncr_net",
            TumorRegion.EDEMA: "area_edema",
            TumorRegion.ENHANCING: "area_enhancing"}
    total = np.zeros_like(areas["area_ncr_net"])
    for member in region.members:
        total = total + areas[keys[member]]
    return total


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
#: Corpus-level caveats. Written into the manifest, and from there into the model card.
#: They are properties of the data, not of the model, so they belong to the cache.
_CORPUS_CAVEATS: list[str] = [
    "BraTS2020 contains only glioma. A model trained on it has never seen a "
    "metastasis, a meningioma, an abscess, a demyelinating lesion, or a healthy brain, "
    "and its output on any of those is undefined rather than negative.",
    "Every subject is pre-skull-stripped, N4-corrected, and co-registered to the SRI24 "
    "atlas by the challenge organisers. A clinical study that has not been through the "
    "same preparation is out of distribution.",
    "The HDF5 redistribution carries no affine. Laterality is declared from the BraTS "
    "convention and is NOT verified; left/right must not be reported from this model.",
    "Intensities arrive per-slice z-scored. They are exact within a slice and accurate "
    "to roughly +-20% across the slices of one volume.",
    "Segmentation labels are the challenge's expert consensus. They were not "
    "re-adjudicated here.",
]


def _align_label(label: np.ndarray, affine: np.ndarray,
                 target_shape: Sequence[int]) -> tuple[np.ndarray, str]:
    """Put the label through the same reorientation the image took.

    Uses the foundation layer's own :func:`to_canonical` with the same source affine,
    so the permutation and flips are identical by construction rather than by a
    reimplementation that could drift. When the standardised image was additionally
    resampled, the label is resampled to match with nearest-neighbour — order 0, never
    interpolated: an interpolated label is a label with classes that were never
    annotated.
    """
    array, _, original, changed = to_canonical(np.asarray(label), np.asarray(affine))
    array = np.ascontiguousarray(array)
    note = (f"label reoriented {''.join(original)} -> RAS by the same permutation and "
            "flips applied to the image" if changed
            else "label was already in canonical orientation")

    target = tuple(int(n) for n in target_shape)
    if tuple(array.shape) != target:
        try:
            from scipy import ndimage
        except ImportError as exc:                       # pragma: no cover - env dep
            raise CorpusIntegrityError(
                "the standardised image was resampled but scipy is unavailable, so "
                "the label cannot be resampled to match",
                detail={"label_shape": list(array.shape),
                        "image_shape": list(target)}) from exc
        factors = [t / s for t, s in zip(target, array.shape)]
        array = ndimage.zoom(array, factors, order=0, mode="nearest")
        array = _fit_exact(array, target)
        note += (f"; resampled {factors} to the standardised grid with "
                 "nearest-neighbour (order 0)")
    return array.astype(np.uint8, copy=False), note


def _fit_exact(array: np.ndarray, target: Sequence[int]) -> np.ndarray:
    """Trim or zero-pad to exactly ``target`` after a zoom's rounding."""
    result = np.zeros(tuple(target), dtype=array.dtype)
    extent = tuple(slice(0, min(a, t)) for a, t in zip(array.shape, target))
    result[extent] = array[extent]
    return result


def _bounding_box(mask: np.ndarray, *, margin: int,
                  shape: Sequence[int]) -> tuple[slice, slice, slice]:
    """Tight box around ``mask``, grown by ``margin`` and clipped to the volume."""
    if not mask.any():
        return tuple(slice(0, int(n)) for n in shape)     # type: ignore[return-value]
    bounds: list[slice] = []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        found = np.flatnonzero(mask.any(axis=others))
        low = max(0, int(found[0]) - margin)
        high = min(int(shape[axis]), int(found[-1]) + 1 + margin)
        bounds.append(slice(low, high))
    return tuple(bounds)                                  # type: ignore[return-value]


def _write_npy(path: Path, array: np.ndarray) -> None:
    """Write a standard ``.npy`` so the result is memory-mappable by anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)


__all__ = [
    "BrainCorpusIngestor", "CacheManifest", "SubjectRecord", "assign_splits",
    "composite_area", "load_manifest", "load_slice_index",
]
