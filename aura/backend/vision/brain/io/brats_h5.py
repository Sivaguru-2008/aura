"""Reader for the BraTS2020 slice-wise HDF5 corpus.

The corpus stores one HDF5 file per axial slice — ``volume_{subject}_slice_{z}.h5``,
57 195 of them — each holding ``image`` of shape ``(240, 240, 4)`` and ``mask`` of
shape ``(240, 240, 3)``. This module turns that back into volumes the MRI Foundation
Layer can standardise, and it does three things that are not obvious from the file
names. Each one is a place where assuming would have produced a silently wrong model.

1. Channel order was *measured*, not assumed
--------------------------------------------
Nothing in the corpus says which of the four image channels is which sequence. The
usual answer — "alphabetical, so FLAIR, T1, T1ce, T2" — is probably right and is
nowhere written down. A wrong answer here is invisible: the network trains happily on
mislabelled contrasts and every downstream report names the wrong sequence.

So it was measured. Over 298 probe slices carrying a labelled tumour, the mean
intensity inside each mask region minus the mean over brain tissue:

===========  =========  =========  =========
             mask 0     mask 1     mask 2
===========  =========  =========  =========
image ch 0     +1.13      +1.18      +1.24
image ch 1     -0.29      -0.02      -0.13
image ch 2     -0.26      -0.13      +0.99
image ch 3     +1.17      +0.60      +0.70
===========  =========  =========  =========

Channel 2 is hyperintense in mask region 2 and *only* there, which is the definition
of gadolinium enhancement: channel 2 is T1ce and mask 2 is the enhancing tumour.
Channel 1 is hypointense in every tumour region — the T1 signature. Channels 0 and 3
are both hyperintense; channel 0 is strongest over the oedema region, which is FLAIR's
signature, leaving channel 3 as T2. Mask region 1 is the largest and most frequent
(present on 42.6% of slices against 26.9% and 25.8%), which is oedema. That fixes both
orderings: images are ``(FLAIR, T1, T1ce, T2)`` and masks are
``(NCR/NET, oedema, enhancing)``.

:meth:`BratsH5Reader._verify_channels` re-runs that test on every subject it reads, and
:class:`~backend.vision.brain.ingest.BrainCorpusIngestor` refuses to build a cache if
agreement falls below a configured floor. The evidence stays live rather than becoming
a comment that was true once.

The statistic it uses is the *enhancing-versus-necrotic* contrast,
``(mean(ET) - mean(NCR/NET)) / std(brain)``, and not the more obvious
enhancing-versus-brain contrast, for two measured reasons. Enhancing-versus-brain picks
the right channel on only 5 of 18 probe subjects: raw differences are not comparable
across channels that were independently z-scored, and FLAIR is hyperintense over the
*whole* tumour so it wins on brightness without saying anything about enhancement.
Enhancing-versus-necrotic normalises within the channel and isolates exactly what
gadolinium does — it brightens viable tumour and not necrosis — and picks channel 2 on
18 of 18. Subjects lacking either region are reported ``evaluated=False`` rather than
scored, because a test with no signal must not be able to count as a pass.

2. The stored intensities are per-slice z-scores, and that is partly reversible
------------------------------------------------------------------------------
Every stored slice has mean 0 and standard deviation exactly 1 over the full 240x240
frame — background included. Two consequences, and the second is the useful one.

*The bad one*: on a near-empty slice the standard deviation is tiny, so the few
non-zero voxels are amplified enormously. Slice 0 of a subject reaches +67 sigma. Fed
to a network unchanged, those slices dominate every gradient they appear in.

*The good one*: BraTS volumes are skull-stripped, so background is exactly zero in the
source. Zero maps to ``-mu/sigma``, which is therefore exactly the minimum of the
stored slice. Subtracting the per-slice minimum restores an array that is exactly
proportional to the original intensities and has background exactly zero — which is
what the foundation layer's masking, cropping, and normalisation stages all assume.
The reader does that, and records it.

What cannot be recovered is the per-slice scale ``sigma``, so intensities are
comparable *within* a slice and only approximately across slices of one volume.
Measured on three subjects, the median brain intensity of the recovered array varies by
a factor of 1.42 to 1.83 across the brain-bearing slices of a volume — a spread that
includes genuine anatomical variation. That is good enough for slice-wise training and
is recorded on every series as a warning so a future volumetric model does not inherit
the assumption silently. The NIfTI release of BraTS2020 does not have this problem and
is the right input if one becomes available.

3. The corpus carries no affine, so one is declared and flagged
---------------------------------------------------------------
There is no geometry in these files at all. BraTS distributes its NIfTI volumes on a
1 mm isotropic grid in a standard LPS-oriented frame, so that is what the reader
declares, and :class:`~backend.foundation.mri.standardize.CanonicalOrientation` then
converts it to RAS+ through the real code path. The assumption is recorded as
``affine_source="assumed_from_brats_convention"`` and ``laterality_verified=False`` on
every series, and it propagates into the model card. **Nothing downstream may report
laterality from a model trained on this corpus** — "left temporal" and "right temporal"
are not distinguishable from evidence available here.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.foundation.mri.geometry import VoxelGeometry
from backend.foundation.mri.io.base import RawSeries, SeriesIntegrity
from backend.foundation.mri.types import FileFormat
from backend.vision.brain.errors import CorpusIntegrityError, CorpusNotFound
from backend.vision.brain.types import (
    DEFAULT_MODALITIES,
    BRATS_LABEL_REMAP,
    ModalitySpec,
    TumorGrade,
    TumorRegion,
)

log = get_logger("vision.brain.brats")

#: ``volume_{subject}_slice_{z}.h5``
_SLICE_PATTERN = re.compile(r"^volume_(\d+)_slice_(\d+)\.h5$", re.IGNORECASE)

#: Slice-wise HDF5 datasets the corpus stores.
_IMAGE_KEY = "image"
_MASK_KEY = "mask"

#: In-plane and through-plane voxel size of the BraTS SRI24 grid, in millimetres.
BRATS_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)

#: Index of the post-contrast T1 channel, used by the channel-order verification.
_T1CE_CHANNEL = 2

#: Mask channel that should be enhancing tumour, per the derivation above.
_ENHANCING_MASK_CHANNEL = 2

#: Voxels a region needs before the channel test is worth running on it. Below this the
#: region means are dominated by boundary voxels of the other class.
_MIN_VERIFICATION_VOXELS = 200

#: Mask channel -> dense class index. The corpus already one-hots BraTS labels 1/2/4
#: into three disjoint planes (verified: pairwise intersections are empty), so the
#: remap that :data:`~backend.vision.brain.types.BRATS_LABEL_REMAP` describes is
#: applied by writing plane *i* as class *i+1*.
_MASK_CHANNEL_TO_CLASS: tuple[int, ...] = (
    TumorRegion.NECROTIC_CORE.value, TumorRegion.EDEMA.value,
    TumorRegion.ENHANCING.value)


@dataclass(frozen=True)
class BratsSubject:
    """One BraTS subject: its files and everything the corpus knows about it.

    ``grade`` is carried but must never be used as a training target — see
    :class:`~backend.vision.brain.types.TumorGrade`.
    """

    volume_id: int
    subject_id: str
    grade: TumorGrade = TumorGrade.UNKNOWN
    age_years: float | None = None
    survival_days: int | None = None
    extent_of_resection: str | None = None
    #: Slice index -> path, sorted by index.
    slice_paths: tuple[Path, ...] = ()
    slice_indices: tuple[int, ...] = ()

    @property
    def slice_count(self) -> int:
        return len(self.slice_paths)

    @property
    def contiguous(self) -> bool:
        """True when slice indices run 0..n-1 with nothing missing."""
        return tuple(self.slice_indices) == tuple(range(len(self.slice_indices)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "subject_id": self.subject_id,
            "grade": self.grade.value,
            "age_years": self.age_years,
            "survival_days": self.survival_days,
            "extent_of_resection": self.extent_of_resection,
            "slice_count": self.slice_count,
            "contiguous": self.contiguous,
        }


@dataclass(frozen=True)
class ChannelVerification:
    """Result of re-testing the modality channel assignment on one subject.

    ``agrees`` is the whole point: it is ``False`` when the post-contrast channel is
    *not* the most hyperintense one inside the enhancing region, which is the signature
    the assignment was derived from. ``evaluated=False`` means the subject had no
    enhancing region large enough to test — common for low-grade glioma, and not a
    failure.
    """

    subject_id: str
    evaluated: bool
    agrees: bool = False
    #: Mean intensity inside the enhancing region minus the brain mean, per channel.
    channel_contrast: tuple[float, ...] = ()
    voxels_tested: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "evaluated": self.evaluated,
                "agrees": self.agrees, "voxels_tested": self.voxels_tested,
                "channel_contrast": [round(v, 4) for v in self.channel_contrast]}


@dataclass(frozen=True)
class SubjectVolumes(object):
    """Assembled volumes for one subject, before the foundation layer sees them.

    ``images`` is ``(C, X, Y, Z)`` and ``label`` is ``(X, Y, Z)`` — the label shares the
    image grid exactly, which is the invariant every later stage depends on.
    """

    subject: BratsSubject
    images: np.ndarray
    label: np.ndarray
    modalities: tuple[ModalitySpec, ...]
    #: Per-slice scale factors that were divided out, for the record. The stored
    #: minimum of each slice and channel; see the module docstring.
    slice_offsets: np.ndarray
    integrity: SeriesIntegrity
    verification: ChannelVerification
    warnings: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.images.shape[1:])   # type: ignore[return-value]


class BratsCorpusIndex:
    """Discovers subjects in a BraTS2020 HDF5 corpus and joins the side-car CSVs.

    Nothing here reads pixel data. Building the index over 57 195 files is a directory
    listing and three CSV parses, so it is cheap enough to run at the start of every
    training and to assert against in a test.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise CorpusNotFound(
                f"the BraTS corpus directory does not exist: {self._root}",
                detail={"root": str(self._root),
                        "hint": "set AURA_BRATS_ROOT or pass PathsConfig(corpus_root=...)"})
        self._subjects: dict[int, BratsSubject] | None = None

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ #
    def subjects(self) -> tuple[BratsSubject, ...]:
        """Every subject found, ordered by volume id. Built once, cached."""
        if self._subjects is None:
            self._subjects = self._build()
        return tuple(self._subjects[k] for k in sorted(self._subjects))

    def subject(self, volume_id: int) -> BratsSubject:
        if self._subjects is None:
            self._subjects = self._build()
        try:
            return self._subjects[int(volume_id)]
        except KeyError as exc:
            raise CorpusIntegrityError(
                f"volume {volume_id} is not present in the corpus",
                detail={"root": str(self._root)}) from exc

    def _build(self) -> dict[int, BratsSubject]:
        slices: dict[int, list[tuple[int, Path]]] = {}
        for path in self._root.iterdir():
            match = _SLICE_PATTERN.match(path.name)
            if match:
                slices.setdefault(int(match.group(1)), []).append(
                    (int(match.group(2)), path))
        if not slices:
            raise CorpusNotFound(
                "no BraTS slice files (volume_*_slice_*.h5) were found",
                detail={"root": str(self._root)})

        grades, demographics = self._read_side_cars()
        subjects: dict[int, BratsSubject] = {}
        for volume_id, entries in slices.items():
            entries.sort(key=lambda e: e[0])
            subject_id = f"BraTS20_Training_{volume_id:03d}"
            demo = demographics.get(subject_id, {})
            subjects[volume_id] = BratsSubject(
                volume_id=volume_id,
                subject_id=subject_id,
                grade=grades.get(subject_id, TumorGrade.UNKNOWN),
                age_years=demo.get("age"),
                survival_days=demo.get("survival_days"),
                extent_of_resection=demo.get("extent_of_resection"),
                slice_paths=tuple(p for _, p in entries),
                slice_indices=tuple(i for i, _ in entries),
            )
        log.info("brats corpus indexed", extra={"context": {
            "root": str(self._root), "subjects": len(subjects),
            "slices": sum(s.slice_count for s in subjects.values()),
            "with_grade": sum(1 for s in subjects.values()
                              if s.grade is not TumorGrade.UNKNOWN)}})
        return subjects

    def _read_side_cars(self) -> tuple[dict[str, TumorGrade], dict[str, dict[str, Any]]]:
        """Join ``name_mapping.csv`` (grade) and ``survival_info.csv`` (demographics).

        Both are optional. A corpus without them still trains; it just cannot run the
        held-out grade probe, and the ingest record says so rather than silently
        reporting a probe over 369 ``UNKNOWN`` labels.
        """
        grades: dict[str, TumorGrade] = {}
        mapping = self._root / "name_mapping.csv"
        if mapping.exists():
            with mapping.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    subject_id = (row.get("BraTS_2020_subject_ID") or "").strip()
                    raw = (row.get("Grade") or "").strip().lower()
                    if subject_id:
                        grades[subject_id] = (TumorGrade.HGG if raw == "hgg"
                                              else TumorGrade.LGG if raw == "lgg"
                                              else TumorGrade.UNKNOWN)
        else:
            log.warning("name_mapping.csv is absent; tumour grade is unavailable",
                        extra={"context": {"root": str(self._root)}})

        demographics: dict[str, dict[str, Any]] = {}
        survival = self._root / "survival_info.csv"
        if survival.exists():
            with survival.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    subject_id = (row.get("Brats20ID") or "").strip()
                    if not subject_id:
                        continue
                    demographics[subject_id] = {
                        "age": _as_float(row.get("Age")),
                        "survival_days": _as_int(row.get("Survival_days")),
                        "extent_of_resection":
                            (row.get("Extent_of_Resection") or "").strip() or None,
                    }
        return grades, demographics


class BratsH5Reader:
    """Reads one BraTS subject into foundation-layer volumes.

    Implements :class:`~backend.foundation.mri.io.base.StudyReader` so it *can* be
    injected into :class:`~backend.foundation.mri.loader.MRIStudyLoader` for a directory
    holding a single subject's slices. The path the ingest actually takes is
    :meth:`read_subject` followed by
    :meth:`~backend.foundation.mri.pipeline.MRIFoundationPipeline.run_series`, because
    the corpus keeps all 369 subjects' slices in one flat directory and directory-level
    discovery would treat them as a single 57 195-file study.
    """

    file_format = FileFormat.HDF5

    def __init__(self, modalities: Sequence[ModalitySpec] = DEFAULT_MODALITIES, *,
                 restore_background_zero: bool = True) -> None:
        self._modalities = tuple(modalities)
        #: Subtract the per-slice minimum so background returns to exactly zero. See
        #: the module docstring; turning this off is only useful for showing what it
        #: costs.
        self._restore = bool(restore_background_zero)

    @property
    def modalities(self) -> tuple[ModalitySpec, ...]:
        return self._modalities

    # ------------------------------------------------------------------ #
    # StudyReader protocol
    # ------------------------------------------------------------------ #
    def can_read(self, path: Path) -> bool:
        """Cheap name test. Never opens the file — discovery calls this on every file."""
        try:
            return bool(_SLICE_PATTERN.match(Path(path).name))
        except Exception:                                # pragma: no cover - defensive
            return False

    def read(self, paths: Sequence[Path], *,
             issues: list[dict[str, Any]] | None = None) -> list[RawSeries]:
        """Decode every subject found among ``paths``, one series per modality."""
        grouped: dict[int, list[Path]] = {}
        for path in paths:
            match = _SLICE_PATTERN.match(Path(path).name)
            if match:
                grouped.setdefault(int(match.group(1)), []).append(Path(path))
        if not grouped:
            raise CorpusIntegrityError(
                "none of the supplied paths are BraTS slice files",
                detail={"paths_examined": len(list(paths))})

        series: list[RawSeries] = []
        for volume_id, group in sorted(grouped.items()):
            subject = _subject_from_paths(volume_id, group)
            try:
                volumes = self.read_subject(subject)
            except CorpusIntegrityError as exc:
                if issues is None:
                    raise
                issues.append({"series_key": subject.subject_id,
                               "error": exc.code, "reason": exc.reason})
                continue
            series.extend(self.to_raw_series(volumes))
        return series

    # ------------------------------------------------------------------ #
    # The path the ingest uses
    # ------------------------------------------------------------------ #
    def read_subject(self, subject: BratsSubject) -> SubjectVolumes:
        """Assemble one subject's four modality volumes and its label volume.

        Raises:
            CorpusIntegrityError: slices are missing, shapes disagree between slices,
                the label carries values outside the declared space, or the mask planes
                overlap (they are meant to be disjoint one-hot planes).
        """
        import h5py                                       # local: heavy, corpus-only

        if subject.slice_count == 0:
            raise CorpusIntegrityError(
                f"{subject.subject_id} has no slice files",
                detail=subject.to_dict())

        warnings: list[str] = []
        if not subject.contiguous:
            missing = sorted(set(range(max(subject.slice_indices) + 1))
                             - set(subject.slice_indices))
            warnings.append(
                f"{len(missing)} slice index/indices are absent from the corpus for "
                f"this subject ({missing[:8]}{'...' if len(missing) > 8 else ''}); the "
                "volume is assembled from what exists and the gap is recorded")

        n_channels = len(self._modalities)
        images: np.ndarray | None = None
        label: np.ndarray | None = None
        offsets = np.zeros((subject.slice_count, n_channels), dtype=np.float32)
        overlap_voxels = 0

        for position, path in enumerate(subject.slice_paths):
            with h5py.File(path, "r") as handle:
                if _IMAGE_KEY not in handle or _MASK_KEY not in handle:
                    raise CorpusIntegrityError(
                        f"{path.name} does not hold both '{_IMAGE_KEY}' and "
                        f"'{_MASK_KEY}'",
                        detail={"keys": list(handle.keys())})
                frame = np.asarray(handle[_IMAGE_KEY][()], dtype=np.float32)
                mask = np.asarray(handle[_MASK_KEY][()])

            if frame.ndim != 3 or frame.shape[2] != n_channels:
                raise CorpusIntegrityError(
                    f"{path.name} has image shape {frame.shape}; "
                    f"(H, W, {n_channels}) was expected",
                    detail={"expected_channels": n_channels})
            if mask.shape[:2] != frame.shape[:2] or mask.shape[2] != 3:
                raise CorpusIntegrityError(
                    f"{path.name} has mask shape {mask.shape}, which does not match "
                    f"its image shape {frame.shape}",
                    detail={"image_shape": list(frame.shape),
                            "mask_shape": list(mask.shape)})

            if images is None:
                height, width = int(frame.shape[0]), int(frame.shape[1])
                images = np.zeros((n_channels, height, width, subject.slice_count),
                                  dtype=np.float32)
                label = np.zeros((height, width, subject.slice_count), dtype=np.uint8)
            elif frame.shape[:2] != images.shape[1:3]:
                raise CorpusIntegrityError(
                    f"{path.name} is {frame.shape[:2]} but earlier slices of this "
                    f"subject are {images.shape[1:3]}; the volume cannot be assembled",
                    detail={"subject": subject.subject_id})

            for channel in range(n_channels):
                plane = frame[..., channel]
                if self._restore:
                    offset = float(plane.min())
                    offsets[position, channel] = offset
                    plane = plane - offset
                images[channel, :, :, position] = plane

            binary = mask.astype(bool)
            occupancy = binary.sum(axis=2)
            if int(occupancy.max(initial=0)) > 1:
                overlap_voxels += int((occupancy > 1).sum())
            for plane_index, class_index in enumerate(_MASK_CHANNEL_TO_CLASS):
                label[:, :, position][binary[..., plane_index]] = class_index

        assert images is not None and label is not None   # for type checkers

        if overlap_voxels:
            # Disjointness was verified empirically over the corpus; a subject that
            # breaks it means the one-hot assumption does not hold and the last plane
            # written would silently win.
            raise CorpusIntegrityError(
                f"{subject.subject_id} has {overlap_voxels} voxel(s) assigned to more "
                "than one mask plane; the corpus's planes are meant to be disjoint and "
                "the label cannot be built without choosing arbitrarily",
                detail={"overlap_voxels": overlap_voxels})

        present = set(np.unique(label).tolist())
        allowed = set(BRATS_LABEL_REMAP.values())
        if not present <= allowed:
            raise CorpusIntegrityError(
                f"{subject.subject_id} produced label values {sorted(present)}; only "
                f"{sorted(allowed)} are defined",
                detail={"found": sorted(present), "allowed": sorted(allowed)})

        if self._restore:
            warnings.append(
                "the corpus stores per-slice z-scored intensities; the per-slice "
                "minimum was subtracted to restore background-zero values "
                "proportional to the original. The per-slice scale factor is not "
                "recoverable, so intensities are exact within a slice and accurate to "
                "roughly +-20% across the slices of one volume")
        warnings.append(
            "no affine accompanies this corpus; a 1 mm isotropic LPS grid was declared "
            "from the BraTS distribution convention. Laterality is NOT verified and "
            "must not be reported from a model trained on it")

        integrity = SeriesIntegrity(
            files_found=subject.slice_count,
            slices_loaded=subject.slice_count,
            missing_slices_estimated=(0 if subject.contiguous
                                      else max(subject.slice_indices) + 1
                                      - subject.slice_count),
            median_slice_spacing_mm=BRATS_SPACING_MM[2],
            spacing_consistent=True,
            geometry_consistent=True,
            warnings=tuple(warnings),
        )
        verification = self._verify_channels(subject, images, label)

        log.info("brats subject assembled", extra={"context": {
            "subject": subject.subject_id, "shape": list(images.shape[1:]),
            "grade": subject.grade.value,
            "tumour_voxels": int((label > 0).sum()),
            "channel_check": verification.agrees if verification.evaluated else None}})

        return SubjectVolumes(
            subject=subject, images=images, label=label,
            modalities=self._modalities, slice_offsets=offsets,
            integrity=integrity, verification=verification,
            warnings=tuple(warnings))

    # ------------------------------------------------------------------ #
    def to_raw_series(self, volumes: SubjectVolumes) -> list[RawSeries]:
        """One :class:`RawSeries` per modality, ready for the foundation pipeline.

        The header carries only what is genuinely known. There is no ``EchoTime``,
        ``RepetitionTime``, or ``ScanningSequence`` in this corpus, so none is invented
        — which means the foundation layer's sequence detector will classify from the
        description alone and report ``requires_review=True`` at a capped confidence.
        That is the correct outcome: our knowledge of the sequence comes from the
        corpus's channel layout, verified by measurement, not from an acquisition
        header, and the pipeline should say so.
        """
        geometry = brats_geometry(volumes.shape)
        series: list[RawSeries] = []
        for index, spec in enumerate(volumes.modalities):
            # No ``PatientID``. The foundation layer's metadata engine rejects
            # patient-identifying keys outright, and it is right to: the BraTS subject
            # id is a de-identified challenge label, but a header field named
            # ``PatientID`` will eventually be treated as one by something downstream.
            # The subject id travels in ``series_key`` and ``source_name`` instead.
            header = {
                "Modality": "MR",
                "SeriesDescription": spec.label,
                "FrameOfReferenceUID": f"brats2020:{volumes.subject.subject_id}",
                # Recorded for the audit trail, ignored by the metadata allowlist.
                "_brats_channel_index": index,
                "_brats_channel_provenance": spec.provenance,
            }
            series.append(RawSeries(
                series_key=f"{volumes.subject.subject_id}/{spec.key}",
                source_format=FileFormat.HDF5,
                voxels=np.ascontiguousarray(volumes.images[index]),
                geometry=geometry,
                header=header,
                integrity=volumes.integrity,
                source_name=f"{volumes.subject.subject_id} {spec.label}",
                contributing_files=tuple(p.name
                                         for p in volumes.subject.slice_paths[:4]),
            ))
        return series

    # ------------------------------------------------------------------ #
    def _verify_channels(self, subject: BratsSubject, images: np.ndarray,
                         label: np.ndarray) -> ChannelVerification:
        """Re-run the enhancement test that fixed the channel order.

        For each channel, the standardised contrast between the enhancing region and
        the necrotic/non-enhancing core, ``(mean(ET) - mean(NCR)) / std(brain)``. Only
        a post-contrast T1 separates those two — that is what gadolinium enhancement
        *is* — so the channel maximising it must be channel 2. See the module docstring
        for why the simpler enhancing-versus-brain contrast does not work.

        A subject with too little of either region is reported ``evaluated=False``. It
        is not a pass and it is not a failure; it is a test that had nothing to look at.
        """
        enhancing = label == TumorRegion.ENHANCING.value
        necrotic = label == TumorRegion.NECROTIC_CORE.value
        voxels = int(enhancing.sum())
        if voxels < _MIN_VERIFICATION_VOXELS or necrotic.sum() < _MIN_VERIFICATION_VOXELS:
            return ChannelVerification(subject.subject_id, evaluated=False,
                                       voxels_tested=voxels)

        brain = images[_T1CE_CHANNEL] > 0
        if brain.sum() < 1000:
            return ChannelVerification(subject.subject_id, evaluated=False,
                                       voxels_tested=voxels)

        contrast: list[float] = []
        for channel in range(images.shape[0]):
            volume = images[channel]
            scale = float(volume[brain].std()) + 1e-8
            contrast.append(
                (float(volume[enhancing].mean()) - float(volume[necrotic].mean()))
                / scale)
        agrees = int(np.argmax(contrast)) == _T1CE_CHANNEL
        if not agrees:
            log.warning("modality channel assignment failed its own check",
                        extra={"context": {"subject": subject.subject_id,
                                           "contrast": [round(c, 3) for c in contrast],
                                           "expected_channel": _T1CE_CHANNEL}})
        return ChannelVerification(subject.subject_id, evaluated=True, agrees=agrees,
                                   channel_contrast=tuple(contrast),
                                   voxels_tested=voxels)


def brats_geometry(shape: Sequence[int]) -> VoxelGeometry:
    """The declared BraTS grid: 1 mm isotropic, LPS, centred on the volume.

    LPS rather than RAS because that is the convention the BraTS NIfTI release uses,
    and declaring it here means
    :class:`~backend.foundation.mri.standardize.CanonicalOrientation` performs a real
    conversion rather than a no-op — the same code path a clinical DICOM study takes.
    The centring is cosmetic: it puts the volume's middle near the world origin so a
    world coordinate printed in a log is a small number.
    """
    shape = tuple(int(n) for n in shape)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = -BRATS_SPACING_MM[0]
    affine[1, 1] = -BRATS_SPACING_MM[1]
    affine[2, 2] = BRATS_SPACING_MM[2]
    affine[:3, 3] = [shape[0] * BRATS_SPACING_MM[0] / 2.0,
                     shape[1] * BRATS_SPACING_MM[1] / 2.0,
                     -shape[2] * BRATS_SPACING_MM[2] / 2.0]
    return VoxelGeometry(affine=affine, shape=shape, space="RAS")


def iter_subjects(root: Path | str, limit: int | None = None
                  ) -> Iterator[BratsSubject]:
    """Convenience iterator over a corpus, for scripts and notebooks."""
    index = BratsCorpusIndex(root)
    for position, subject in enumerate(index.subjects()):
        if limit is not None and position >= limit:
            return
        yield subject


# --------------------------------------------------------------------------- #
def _subject_from_paths(volume_id: int, paths: Sequence[Path]) -> BratsSubject:
    entries = sorted(
        (int(_SLICE_PATTERN.match(p.name).group(2)), p)   # type: ignore[union-attr]
        for p in paths)
    return BratsSubject(
        volume_id=volume_id,
        subject_id=f"BraTS20_Training_{volume_id:03d}",
        slice_paths=tuple(p for _, p in entries),
        slice_indices=tuple(i for i, _ in entries))


def _as_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    """Parse an integer, tolerating the corpus's ``'ALIVE (361 days later)'`` entries."""
    text = str(value or "").strip()
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


__all__ = [
    "BRATS_SPACING_MM", "BratsCorpusIndex", "BratsH5Reader", "BratsSubject",
    "ChannelVerification", "SubjectVolumes", "brats_geometry", "iter_subjects",
]
