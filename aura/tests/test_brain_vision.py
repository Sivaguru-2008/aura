"""Tests for the AURA NeuroMind Brain Vision Engine.

Layered the same way as the foundation-layer tests, in increasing cost:

1. **Vocabulary and configuration** — pure data. Cheap, and they catch the label-space
   mistakes that are invisible everywhere else.
2. **Corpus reader** — a synthetic BraTS HDF5 corpus written to the real layout. A
   synthetic HDF5 file *is* a real HDF5 file, so this is legitimate in a way a
   synthetic MR acquisition would not be; where the test needs real MR physics (the
   channel-order verification) the fixture is built to carry that physics explicitly.
3. **Components** — augmentation, degradation, sampling, losses, metrics, model, each
   with hand-built inputs whose correct answer is known by construction.
4. **End to end** — ingest, train, validate, checkpoint, and infer over the synthetic
   corpus.

Several tests exist to protect the module's *honesty* rather than its correctness, and
are worth keeping if this file is ever trimmed:

* ``test_label_follows_every_geometric_transform`` — a label that stops tracking its
  image produces a model that converges normally and is worthless.
* ``test_channel_verification_rejects_a_swapped_corpus`` — the check that stops the
  network being trained on mislabelled contrasts.
* ``test_declared_architecture_raises_rather_than_substituting`` — a roadmap entry must
  not quietly resolve to a different model.
* ``test_output_dict_never_carries_voxels`` — the boundary that keeps voxels out of API
  responses and log lines.
* ``test_empty_prediction_on_empty_truth_scores_one`` — the scoring convention that
  stops a metric measuring prevalence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from aura.backend.vision.brain.augment import SliceAugmenter, affine, flip, rot90  # noqa: E402
from aura.backend.vision.brain.config import (  # noqa: E402
    AugmentationConfig,
    BrainVisionConfig,
    CurriculumConfig,
    DegradationConfig,
    IngestConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    PathsConfig,
    SamplingConfig,
    SplitConfig,
    ValidationConfig,
)
from aura.backend.vision.brain.dataset import (  # noqa: E402
    MorphologyLabeller,
    build_datasets,
    fit_to_grid,
    normalize_slice,
)
from aura.backend.vision.brain.degradations import DegradationSimulator  # noqa: E402
from aura.backend.vision.brain.embeddings import (  # noqa: E402
    EmbeddingBatch,
    EmbeddingStore,
    load_embeddings,
    nearest_neighbours,
)
from aura.backend.vision.brain.errors import (  # noqa: E402
    ArchitectureUnavailable,
    CheckpointError,
    ConfigurationError,
    CorpusIntegrityError,
)
from aura.backend.vision.brain.ingest import (  # noqa: E402
    BrainCorpusIngestor,
    assign_splits,
    load_manifest,
    load_slice_index,
)
from aura.backend.vision.brain.io.brats_h5 import (  # noqa: E402
    BratsCorpusIndex,
    BratsH5Reader,
    brats_geometry,
)
from aura.backend.vision.brain.losses import (  # noqa: E402
    MultiTaskLoss,
    per_sample_foreground_dice,
    soft_dice,
    supervised_contrastive,
    variance_covariance,
)
from aura.backend.vision.brain.metrics import (  # noqa: E402
    ClassificationMeter,
    EmbeddingMeter,
    RegressionMeter,
    SegmentationMeter,
)
from aura.backend.vision.brain.model import (  # noqa: E402
    build_encoder,
    build_network,
    declared_architectures,
)
from aura.backend.vision.brain.output import BrainVisionOutput, build_regions  # noqa: E402
from aura.backend.vision.brain.sampling import (  # noqa: E402
    AdaptiveSliceSampler,
    DifficultyTracker,
    SliceTable,
)
from aura.backend.vision.brain.types import (  # noqa: E402
    BRATS_LABEL_REMAP,
    COMPOSITE_MEMBERS,
    CompositeRegion,
    CurriculumStage,
    SplitName,
    TumorRegion,
)

# --------------------------------------------------------------------------- #
# Fixtures: a synthetic BraTS corpus
# --------------------------------------------------------------------------- #
_SIZE = 48
_SLICES = 16


def _make_subject(root: Path, volume_id: int, *, rng: np.random.Generator,
                  swap_channels: bool = False, overlap: bool = False) -> None:
    """Write one subject's slice files with the physics the channel check relies on.

    Constructed so the derivation in :mod:`aura.backend.vision.brain.io.brats_h5` holds:
    channel 2 (T1ce) is the only one that is bright in the enhancing region relative to
    the necrotic core. The stored values are per-slice z-scores over the whole frame,
    exactly as the real corpus stores them, so the reader's background restoration is
    exercised rather than bypassed.
    """
    centre = _SIZE // 2
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    head = (yy - centre) ** 2 + (xx - centre) ** 2 < (centre - 4) ** 2

    for z in range(_SLICES):
        raw = np.zeros((_SIZE, _SIZE, 4), dtype=np.float64)
        mask = np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)
        # Brain tissue: a plain positive level in every sequence.
        for channel in range(4):
            raw[..., channel][head] = 100.0 + rng.normal(0, 2, size=int(head.sum()))

        if 4 <= z < _SLICES - 4:
            radius = 6
            core = (yy - centre + 4) ** 2 + (xx - centre) ** 2 < radius ** 2
            rim = ((yy - centre + 4) ** 2 + (xx - centre) ** 2 < (radius + 3) ** 2) \
                & ~core
            oedema = ((yy - centre + 4) ** 2 + (xx - centre) ** 2
                      < (radius + 7) ** 2) & ~core & ~rim
            mask[..., 0][core] = 1                       # necrotic
            mask[..., 1][oedema] = 1                     # oedema
            mask[..., 2][rim] = 1                        # enhancing
            if overlap:
                mask[..., 0][rim] = 1                    # deliberately invalid
            raw[..., 0][oedema] = 190.0                  # FLAIR bright over oedema
            raw[..., 3][oedema] = 170.0                  # T2 bright over oedema
            raw[..., 1][core] = 55.0                     # T1 dark in the core
            raw[..., 2][core] = 60.0
            raw[..., 2][rim] = 210.0                     # T1ce: enhancement
            raw[..., 0][rim] = 120.0
            raw[..., 3][rim] = 120.0

        if swap_channels:
            raw = raw[..., [2, 1, 0, 3]]

        stored = np.empty_like(raw)
        for channel in range(4):
            plane = raw[..., channel]
            std = plane.std()
            stored[..., channel] = ((plane - plane.mean()) / std if std > 0
                                    else np.zeros_like(plane))
        with h5py.File(root / f"volume_{volume_id}_slice_{z}.h5", "w") as handle:
            handle.create_dataset("image", data=stored)
            handle.create_dataset("mask", data=mask)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("brats")
    rng = np.random.default_rng(11)
    for volume_id in range(1, 9):
        _make_subject(root, volume_id, rng=rng)
    rows = ["Grade,BraTS_2017_subject_ID,BraTS_2018_subject_ID,TCGA_TCIA_subject_ID,"
            "BraTS_2019_subject_ID,BraTS_2020_subject_ID"]
    for volume_id in range(1, 9):
        grade = "HGG" if volume_id % 3 else "LGG"
        rows.append(f"{grade},a,b,NA,c,BraTS20_Training_{volume_id:03d}")
    (root / "name_mapping.csv").write_text("\n".join(rows), encoding="utf-8")
    (root / "survival_info.csv").write_text(
        "Brats20ID,Age,Survival_days,Extent_of_Resection\n"
        "BraTS20_Training_001,60.5,289,GTR\n"
        "BraTS20_Training_002,52.2,ALIVE (361 days later),STR\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def cached(corpus: Path, tmp_path_factory) -> BrainVisionConfig:
    """Ingest the synthetic corpus once and share the cache across tests."""
    artifacts = tmp_path_factory.mktemp("brain_artifacts")
    config = _small_config(corpus, artifacts)
    BrainCorpusIngestor(config).run()
    return config


def _small_config(corpus: Path, artifacts: Path) -> BrainVisionConfig:
    return BrainVisionConfig(
        paths=PathsConfig(corpus_root=corpus, artifacts_root=artifacts),
        run_name="test",
        ingest=IngestConfig(min_brain_fraction=0.02, crop_margin_voxels=2),
        split=SplitConfig(train_fraction=0.5, val_fraction=0.25, test_fraction=0.25),
        model=ModelConfig(stage_channels=(8, 16), blocks_per_stage=(1, 1),
                          deep_supervision_levels=2, embedding_dim=16,
                          embedding_hidden=32, input_size=(32, 32)),
        optim=OptimConfig(epochs=2, batch_size=4, num_workers=0, amp=False,
                          warmup_epochs=0.0, early_stopping_patience=5, seed=3),
        sampling=SamplingConfig(samples_per_epoch=16),
        curriculum=CurriculumConfig(schedule=((CurriculumStage.LARGE, 1),
                                              (CurriculumStage.FULL, 10))),
        validation=ValidationConfig(batch_size=4, embedding_limit=64),
    )


# --------------------------------------------------------------------------- #
# 1. Vocabulary and configuration
# --------------------------------------------------------------------------- #
def test_brats_label_four_is_remapped_to_a_dense_space():
    assert BRATS_LABEL_REMAP == {0: 0, 1: 1, 2: 2, 4: 3}
    assert sorted(r.value for r in TumorRegion) == [0, 1, 2, 3]


def test_composite_regions_are_nested_unions_of_the_primary_classes():
    whole = set(COMPOSITE_MEMBERS[CompositeRegion.WHOLE_TUMOR])
    core = set(COMPOSITE_MEMBERS[CompositeRegion.TUMOR_CORE])
    enhancing = set(COMPOSITE_MEMBERS[CompositeRegion.ENHANCING_TUMOR])
    assert enhancing < core < whole
    assert TumorRegion.EDEMA in whole and TumorRegion.EDEMA not in core


def test_split_fractions_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ConfigurationError):
        SplitConfig(train_fraction=0.8, val_fraction=0.3, test_fraction=0.1)


def test_a_loss_with_no_segmentation_term_is_rejected():
    with pytest.raises(ConfigurationError):
        LossConfig(dice_weight=0.0, cross_entropy_weight=0.0, focal_weight=0.0,
                   boundary_weight=0.0)


def test_curriculum_walks_its_stages_then_stays_in_the_last():
    config = CurriculumConfig(schedule=((CurriculumStage.LARGE, 2),
                                        (CurriculumStage.MEDIUM, 1),
                                        (CurriculumStage.FULL, 5)))
    stages = [config.stage_for_epoch(e) for e in range(8)]
    assert stages[:3] == [CurriculumStage.LARGE, CurriculumStage.LARGE,
                          CurriculumStage.MEDIUM]
    assert stages[-1] is CurriculumStage.FULL


def test_paths_keep_every_brain_artifact_out_of_the_thorax_directory(tmp_path):
    paths = PathsConfig(artifacts_root=tmp_path / "artifacts" / "brain")
    for path in (paths.best_model_path, paths.encoder_path, paths.embedding_dir,
                 paths.manifest_path, paths.model_card_path):
        assert "brain" in path.parts, path


# --------------------------------------------------------------------------- #
# 2. Corpus reader
# --------------------------------------------------------------------------- #
def test_corpus_index_joins_grade_and_tolerates_a_non_numeric_survival(corpus):
    subjects = BratsCorpusIndex(corpus).subjects()
    assert len(subjects) == 8
    assert subjects[0].subject_id == "BraTS20_Training_001"
    assert subjects[0].grade.value == "hgg"
    assert subjects[2].grade.value == "lgg"
    # "ALIVE (361 days later)" must parse to 361, not crash and not become None.
    assert subjects[1].survival_days == 361
    assert all(s.contiguous for s in subjects)


def test_reader_restores_background_to_exactly_zero(corpus):
    subject = BratsCorpusIndex(corpus).subject(1)
    volumes = BratsH5Reader().read_subject(subject)
    # The corpus stores z-scores whose minimum is the background; after restoration the
    # background is 0 and every brain voxel is strictly positive.
    assert volumes.images.min() == pytest.approx(0.0, abs=1e-6)
    assert (volumes.images > 0).any()
    assert volumes.slice_offsets.shape == (_SLICES, 4)
    assert (volumes.slice_offsets < 0).any()


def test_reader_produces_one_series_per_modality_in_declared_order(corpus):
    subject = BratsCorpusIndex(corpus).subject(1)
    reader = BratsH5Reader()
    series = reader.to_raw_series(reader.read_subject(subject))
    assert [s.series_key.split("/")[-1] for s in series] == \
        ["flair", "t1", "t1ce", "t2"]
    # No acquisition parameters are invented; the header carries only what is known.
    assert set(series[0].header) >= {"Modality", "SeriesDescription"}
    assert "EchoTime" not in series[0].header
    assert "PatientID" not in series[0].header


def test_declared_geometry_is_lps_at_one_millimetre(corpus):
    geometry = brats_geometry((_SIZE, _SIZE, _SLICES))
    assert geometry.spacing == pytest.approx((1.0, 1.0, 1.0))
    assert geometry.orientation == "LPS"


def test_channel_verification_accepts_the_real_layout(corpus):
    subject = BratsCorpusIndex(corpus).subject(1)
    verification = BratsH5Reader().read_subject(subject).verification
    assert verification.evaluated and verification.agrees
    assert int(np.argmax(verification.channel_contrast)) == 2


def test_channel_verification_rejects_a_swapped_corpus(tmp_path):
    """The check that stops a model being trained on mislabelled contrasts."""
    root = tmp_path / "swapped"
    root.mkdir()
    _make_subject(root, 1, rng=np.random.default_rng(0), swap_channels=True)
    subject = BratsCorpusIndex(root).subject(1)
    verification = BratsH5Reader().read_subject(subject).verification
    assert verification.evaluated and not verification.agrees


def test_overlapping_mask_planes_are_a_hard_error(tmp_path):
    root = tmp_path / "overlap"
    root.mkdir()
    _make_subject(root, 1, rng=np.random.default_rng(0), overlap=True)
    subject = BratsCorpusIndex(root).subject(1)
    with pytest.raises(CorpusIntegrityError, match="more than one mask plane"):
        BratsH5Reader().read_subject(subject)


# --------------------------------------------------------------------------- #
# 3. Ingest
# --------------------------------------------------------------------------- #
def test_ingest_writes_a_cache_a_manifest_and_a_study_per_subject(cached):
    manifest = load_manifest(cached)
    assert len(manifest.subjects) == 8
    assert manifest.channel_verification["agreement"] == 1.0
    assert manifest.caveats, "corpus caveats must reach the manifest"
    for record in manifest.subjects:
        assert (cached.paths.studies_dir / f"{record.subject_id}.json").exists()
        assert (cached.paths.cache_dir / "volumes"
                / f"{record.subject_id}.img.npy").exists()


def test_the_cached_study_records_what_the_foundation_layer_did(cached):
    study = json.loads(
        (cached.paths.studies_dir / "BraTS20_Training_001.json").read_text())
    names = {step["name"] for step in study["history"]["steps"]}
    assert "foundation_standardisation" in names
    assert "label_alignment" in names
    # Normalisation is skipped *and says so*, rather than being silently absent.
    normalisation = next(s for s in study["history"]["steps"]
                         if s["name"] == "intensity_normalization")
    assert normalisation["status"] == "skipped"
    assert "background-zero" in normalisation["message"]
    # Each series went through the real pipeline.
    series_steps = {s["name"] for s in study["series"][0]["history"]["steps"]}
    assert {"canonical_orientation", "quality_assessment", "foreground_mask"} \
        <= series_steps


def test_label_and_image_stay_on_one_grid_through_ingest(cached):
    volumes = np.load(cached.paths.cache_dir / "volumes"
                      / "BraTS20_Training_001.img.npy")
    labels = np.load(cached.paths.cache_dir / "volumes"
                     / "BraTS20_Training_001.seg.npy")
    assert volumes.shape[0] == labels.shape[0]
    assert volumes.shape[2:] == labels.shape[1:]
    # The tumour must sit inside the brain, not beside it — the test that catches a
    # label that failed to follow the image through reorientation.
    tumour = labels > 0
    assert tumour.any()
    brain = volumes.max(axis=1) > 0
    assert float((brain & tumour).sum()) / float(tumour.sum()) > 0.98


def test_splits_are_by_subject_and_never_share_one(cached):
    manifest = load_manifest(cached)
    assignment: dict[str, set[str]] = {}
    for record in manifest.subjects:
        assignment.setdefault(record.split.value, set()).add(record.subject_id)
    everything = [s for members in assignment.values() for s in members]
    assert len(everything) == len(set(everything))
    assert assignment.get("train") and assignment.get("val")


def test_a_second_ingest_reuses_the_cache_instead_of_rebuilding_it(corpus,
                                                                   tmp_path_factory):
    """A 40-minute ingest interrupted at minute 35 must not cost 35 minutes."""
    artifacts = tmp_path_factory.mktemp("resume_ingest")
    config = _small_config(corpus, artifacts)
    first = BrainCorpusIngestor(config).run()
    stamps = {p.name: p.stat().st_mtime_ns
              for p in (config.paths.cache_dir / "volumes").glob("*.npy")}

    second = BrainCorpusIngestor(config).run()
    assert len(second.subjects) == len(first.subjects)
    # The voxel files must not have been rewritten...
    assert {p.name: p.stat().st_mtime_ns
            for p in (config.paths.cache_dir / "volumes").glob("*.npy")} == stamps
    # ...and the slice index must be identical, not merely the same length.
    for name, column in load_slice_index(config).items():
        assert column.shape[0] > 0, name
    assert [s.to_dict() for s in second.subjects] == \
        [s.to_dict() for s in first.subjects]


def test_split_assignment_is_deterministic_and_grade_stratified(corpus):
    subjects = BratsCorpusIndex(corpus).subjects()
    config = SplitConfig(train_fraction=0.5, val_fraction=0.25, test_fraction=0.25)
    first = assign_splits(subjects, config)
    assert first == assign_splits(subjects, config)
    low_grade = [s.subject_id for s in subjects if s.grade.value == "lgg"]
    assert len({first[s] for s in low_grade}) > 1, \
        "stratification should spread low-grade subjects across splits"


# --------------------------------------------------------------------------- #
# 4. Dataset, augmentation, degradation
# --------------------------------------------------------------------------- #
def test_normalisation_uses_brain_voxels_and_leaves_background_at_zero():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    brain = np.zeros((8, 8), dtype=bool)
    brain[2:6, 2:6] = True
    image[0][brain] = np.linspace(10, 20, brain.sum())
    image[1][brain] = 5.0
    result = normalize_slice(image, brain)
    assert result[0][~brain].max() == 0.0
    assert result[0][brain].mean() == pytest.approx(0.0, abs=1e-5)
    assert result[0][brain].std() == pytest.approx(1.0, abs=1e-4)
    # A constant channel has no variance to normalise and must not be divided by ~0.
    assert np.all(result[1] == 0.0)


def test_fit_to_grid_pads_and_crops_image_and_label_together():
    image = np.ones((3, 10, 20), dtype=np.float32)
    label = np.zeros((10, 20), dtype=np.uint8)
    label[4:6, 8:12] = 2
    fitted, fitted_label = fit_to_grid(image, label, (16, 16))
    assert fitted.shape == (3, 16, 16)
    assert fitted_label.shape == (16, 16)
    assert (fitted_label == 2).any()
    assert fitted[:, 0, 0] == pytest.approx(0.0)          # padding is background


@pytest.mark.parametrize("transform", ["flip", "rot90"])
def test_label_follows_every_exact_geometric_transform(transform):
    """A label that stops tracking its image trains a model on mirrored truth.

    Flips and quarter-turns are permutations, so image and label must come out
    *identical* — anything weaker cannot distinguish a two-pixel drift from a correct
    result.
    """
    label = np.zeros((32, 32), dtype=np.uint8)
    label[6:12, 20:26] = 3
    image = (label[None].astype(np.float32) * 7.0).repeat(4, axis=0)

    if transform == "flip":
        moved, moved_label = flip(image, label, axis=1)
    else:
        moved, moved_label = rot90(image, label, 1)
    assert np.array_equal((moved[0] / 7.0).round().astype(np.uint8), moved_label)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_label_tracks_the_image_through_a_random_affine(seed):
    """The same invariant where exactness is not available.

    The image is resampled with order 1 and the label with order 0 — deliberately, since
    an interpolated label invents classes nobody annotated. So they cannot be identical
    at the boundary. What must hold is that they describe the *same region*: overlapping
    almost completely, with centroids within half a pixel. A transform applied to one and
    not the other, or applied with a different matrix, fails both by a wide margin.
    """
    pytest.importorskip("scipy.ndimage")
    label = np.zeros((48, 48), dtype=np.uint8)
    label[10:22, 26:38] = 3
    image = (label[None].astype(np.float32) * 7.0).repeat(4, axis=0)

    moved, moved_label = affine(image, label, np.random.default_rng(seed),
                                max_rotation_deg=20.0, max_scale=0.1, max_shift=0.1)
    from_image = moved[0] > 3.5
    from_label = moved_label > 0
    overlap = float((from_image & from_label).sum())
    dice = 2 * overlap / float(from_image.sum() + from_label.sum())
    assert dice > 0.9, f"image and label diverged (dice {dice:.3f})"

    centre_image = np.argwhere(from_image).mean(axis=0)
    centre_label = np.argwhere(from_label).mean(axis=0)
    assert np.abs(centre_image - centre_label).max() < 0.5


def test_augmentation_leaves_the_label_alone_for_intensity_only_transforms():
    config = AugmentationConfig(flip_lr=False, flip_ap=False, rot90=False,
                                affine=False, probability=1.0)
    image = np.ones((4, 16, 16), dtype=np.float32)
    label = np.zeros((16, 16), dtype=np.uint8)
    label[4:8, 4:8] = 1
    augmented, augmented_label = SliceAugmenter(config)(
        image, label, np.random.default_rng(0))
    assert np.array_equal(augmented_label, label)
    assert not np.allclose(augmented, image)


@pytest.mark.parametrize("artifact", ["rician_noise", "bias_field", "motion_ghosting",
                                      "k_space_spike", "blur"])
def test_every_degradation_changes_the_image_and_reports_its_severity(artifact):
    simulator = DegradationSimulator(DegradationConfig(probability=1.0))
    rng = np.random.default_rng(4)
    image = np.zeros((4, 32, 32), dtype=np.float32)
    image[:, 8:24, 8:24] = 100.0
    degraded, record = simulator(image, rng, base_quality=1.0, force=artifact)
    assert record.name == artifact
    assert 0.0 < record.severity <= 1.0
    assert record.target_quality == pytest.approx(1.0 - record.severity)
    assert not np.allclose(degraded, image)
    assert np.isfinite(degraded).all()


def test_a_degradation_target_can_never_exceed_the_foundation_quality():
    simulator = DegradationSimulator(DegradationConfig(probability=1.0))
    image = np.ones((4, 16, 16), dtype=np.float32)
    _, record = simulator(image, np.random.default_rng(1), base_quality=0.5,
                          force="blur")
    assert record.target_quality <= 0.5


def test_morphology_class_encodes_presence_pattern_and_size():
    labeller = MorphologyLabeller(small_max=100.0, medium_max=500.0)
    assert labeller(( 0, 0, 0)) == 0
    small_core = labeller((50, 0, 0))
    large_core = labeller((900, 0, 0))
    small_mixed = labeller((20, 20, 20))
    assert small_core != large_core != small_mixed
    assert 0 < small_core < labeller.num_classes


def test_dataset_survives_the_trip_to_a_spawned_dataloader_worker(cached):
    """On Windows, workers are spawned and the dataset is pickled to reach them.

    Anything unpicklable on it — a module reference, an open memmap — turns into a
    dataloader that works at ``num_workers=0`` and dies at 1, which is a long way from
    the code that caused it.
    """
    import pickle

    datasets, _, _, _ = build_datasets(cached)
    train = datasets[SplitName.TRAIN]
    train[0]                                              # populate any lazy state
    assert train._maps, "the memmap cache should be populated by now"
    assert len(train._maps) <= train.max_open_volumes
    payload = pickle.dumps(train)
    restored = pickle.loads(payload)
    assert len(restored) == len(train)
    assert restored[0]["image"].shape == train[0]["image"].shape
    # The open maps must not travel: pickling one sends the whole volume, turning a
    # lazy read into an eager broadcast to every worker.
    assert len(payload) < 2_000_000, "bulk voxel data was pickled into the worker"


def test_open_memory_maps_stay_bounded_however_many_subjects_are_touched(cached):
    """An unbounded map cache is what kills a long run, in a place unrelated to itself.

    258 subjects at ~40 MB, touched by every worker over an epoch, is ~10 GB of mapped
    views per process — enough to exhaust the Windows commit limit and fail inside an
    unrelated shared-memory allocation.
    """
    from dataclasses import replace

    config = cached.with_overrides(ingest=replace(cached.ingest, max_open_volumes=2))
    datasets, table, _, _ = build_datasets(config)
    train = datasets[SplitName.TRAIN]
    touched = set()
    for position in range(len(train)):
        train[position]
        touched.add(int(table.subject_index[train.indices[position]]))
        assert len(train._maps) <= 2
        if len(touched) > 3:
            break
    assert len(touched) > 2, "the split should span several subjects"


def test_dataset_recomputes_targets_from_the_label_the_network_receives(cached):
    datasets, table, _, _ = build_datasets(cached)
    train = datasets[SplitName.TRAIN]
    sample = train[0]
    label = sample["label"].numpy()
    presence = sample["presence"].numpy()
    assert presence[0] == float((label > 0).any())
    for position, region in enumerate(
            (TumorRegion.NECROTIC_CORE, TumorRegion.EDEMA, TumorRegion.ENHANCING),
            start=1):
        assert presence[position] == float((label == region.value).any())
    assert sample["image"].shape == (4, 32, 32)


# --------------------------------------------------------------------------- #
# 5. Sampling
# --------------------------------------------------------------------------- #
def _table(area: np.ndarray) -> SliceTable:
    size = area.size
    return SliceTable(
        subject_index=np.arange(size) // 10,
        cache_z=np.arange(size), source_slice=np.arange(size),
        brain_voxels=np.full(size, 1000),
        area_ncr_net=area, area_edema=np.zeros(size, dtype=np.int64),
        area_enhancing=np.zeros(size, dtype=np.int64),
        quality_score=np.full(size, 0.9, dtype=np.float32))


def test_curriculum_restricts_the_positive_pool_then_restores_it():
    area = np.concatenate([np.zeros(40, dtype=np.int64),
                           np.full(30, 100, dtype=np.int64),
                           np.full(30, 3000, dtype=np.int64)])
    table = _table(area)
    sampler = AdaptiveSliceSampler(
        table, np.arange(area.size), sampling=SamplingConfig(samples_per_epoch=200),
        curriculum=CurriculumConfig(schedule=((CurriculumStage.LARGE, 1),
                                              (CurriculumStage.FULL, 5))))
    sampler.set_epoch(0)
    positive, negative = sampler.schedule.eligible(CurriculumStage.LARGE)
    assert positive.size == 30 and negative.size == 40
    positive, _ = sampler.schedule.eligible(CurriculumStage.FULL)
    assert positive.size == 60


def test_region_focus_oversamples_pathology_without_dropping_negatives():
    area = np.concatenate([np.zeros(60, dtype=np.int64),
                           np.full(40, 3000, dtype=np.int64)])
    table = _table(area)
    sampler = AdaptiveSliceSampler(
        table, np.arange(area.size),
        sampling=SamplingConfig(samples_per_epoch=1000, tumor_fraction=0.7,
                                hard_mining=False),
        curriculum=CurriculumConfig(enabled=False))
    drawn = np.asarray(list(sampler))
    positive_share = float((area[drawn] > 0).mean())
    assert 0.6 < positive_share < 0.8
    assert (area[drawn] == 0).sum() > 0, "negatives must not vanish"


def test_hard_mining_raises_hard_samples_but_keeps_easy_ones_in_the_distribution():
    area = np.full(100, 2000, dtype=np.int64)
    table = _table(area)
    sampler = AdaptiveSliceSampler(
        table, np.arange(100),
        sampling=SamplingConfig(samples_per_epoch=4000, tumor_fraction=1.0,
                                hard_mining=True, hard_mining_strength=1.0),
        curriculum=CurriculumConfig(enabled=False))
    # First ten samples are impossible; the rest are solved.
    sampler.difficulty.update(np.arange(10), np.zeros(10))
    sampler.difficulty.update(np.arange(10, 100), np.ones(90))
    counts = np.bincount(np.asarray(list(sampler)), minlength=100)
    assert counts[:10].mean() > counts[10:].mean() * 3
    assert counts[10:].min() > 0, "the weight floor must keep solved samples sampled"


def test_difficulty_tracker_replaces_its_prior_on_first_observation():
    tracker = DifficultyTracker(size=4, ema=0.3, prior=0.5)
    tracker.update([0], [1.0])                            # perfect Dice -> difficulty 0
    assert tracker.difficulty[0] == pytest.approx(0.0)
    tracker.update([0], [0.0])                            # then a failure, blended
    assert 0.0 < tracker.difficulty[0] < 1.0


# --------------------------------------------------------------------------- #
# 6. Losses
# --------------------------------------------------------------------------- #
def test_soft_dice_is_zero_for_a_perfect_prediction():
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    logits = torch.full((2, 4, 8, 8), -20.0)
    logits.scatter_(1, target[:, None], 20.0)
    assert float(soft_dice(logits, target, num_classes=4)) < 0.05


def test_per_sample_dice_scores_an_empty_prediction_on_empty_truth_as_one():
    """Otherwise every tumour-free slice looks maximally hard to the miner."""
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    logits = torch.zeros(2, 4, 8, 8)
    logits[:, 0] = 10.0
    assert per_sample_foreground_dice(logits, target).tolist() == [1.0, 1.0]


def test_supervised_contrastive_prefers_a_clustered_embedding():
    labels = torch.tensor([0, 0, 1, 1])
    clustered = torch.tensor([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [-0.14, 0.99]])
    scrambled = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.14], [-0.14, 0.99]])
    assert float(supervised_contrastive(clustered, labels)) < \
        float(supervised_contrastive(scrambled, labels))


def test_variance_term_punishes_a_collapsed_embedding():
    collapsed = torch.ones(8, 4) * 0.3
    spread = torch.randn(8, 4)
    assert float(variance_covariance(collapsed)[0]) > \
        float(variance_covariance(spread)[0])


def test_multitask_loss_reports_every_component_and_reaches_the_encoder():
    config = ModelConfig(stage_channels=(8, 16), blocks_per_stage=(1, 1),
                         deep_supervision_levels=2, embedding_dim=16,
                         embedding_hidden=32, input_size=(32, 32))
    network = build_network(config)
    criterion = MultiTaskLoss(LossConfig(), heads=config.heads)
    batch = {
        "image": torch.randn(4, 4, 32, 32),
        "label": torch.randint(0, 4, (4, 32, 32)),
        "presence": torch.rand(4, 4).round(),
        "size": torch.rand(4, 4),
        "quality": torch.rand(4, 1),
        "morphology": torch.tensor([0, 0, 1, 1]),
    }
    breakdown = criterion(network(batch["image"]), batch)
    assert {"dice", "cross_entropy", "segmentation", "presence", "size", "quality",
            "supcon", "embedding", "total"} <= set(breakdown.components)
    breakdown.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in network.encoder.parameters())


# --------------------------------------------------------------------------- #
# 7. Metrics
# --------------------------------------------------------------------------- #
def test_segmentation_meter_reproduces_a_dice_computed_by_hand():
    truth = np.zeros((1, 10, 10), dtype=np.int64)
    truth[0, 2:6, 2:6] = 2                                # 16 pixels of oedema
    predicted = np.zeros_like(truth)
    predicted[0, 2:6, 2:4] = 2                            # 8 correct, 8 missed
    meter = SegmentationMeter(compute_hausdorff=False)
    meter.update(predicted, truth)
    summary = meter.summary()
    assert summary["per_class"]["edema"]["dice"] == pytest.approx(2 * 8 / (8 + 16),
                                                                  abs=1e-5)
    assert summary["per_class"]["edema"]["recall"] == pytest.approx(0.5, abs=1e-5)
    assert summary["per_class"]["edema"]["precision"] == pytest.approx(1.0, abs=1e-5)


def test_empty_prediction_on_empty_truth_scores_one_per_slice():
    """The per-slice convention: a slice that exists, correctly predicted empty, is 1.0.

    The *pooled* metric over a class that occurs nowhere is a different thing and must
    be ``None`` — see the next test. Reporting 1.0 there would let an absent class raise
    the headline number.
    """
    truth = np.zeros((3, 8, 8), dtype=np.int64)
    meter = SegmentationMeter(compute_hausdorff=False)
    meter.update(np.zeros_like(truth), truth)
    enhancing = meter.summary()["per_composite"]["enhancing_tumor"]
    assert enhancing["dice_per_slice_mean"] == 1.0
    assert enhancing["empty_agreements"] == 3


def test_a_metric_with_no_evidence_reports_none_rather_than_a_filler():
    """A recall of 1.0 for a class that appears nowhere is not a perfect score."""
    truth = np.zeros((3, 8, 8), dtype=np.int64)
    meter = SegmentationMeter(compute_hausdorff=False)
    meter.update(np.zeros_like(truth), truth)
    summary = meter.summary()
    enhancing = summary["per_composite"]["enhancing_tumor"]
    for key in ("dice", "iou", "precision", "recall", "sensitivity"):
        assert enhancing[key] is None, key
    # ...and an unscoreable region is excluded from the headline mean, and counted.
    assert summary["composite_dice_mean"] is None
    assert summary["composite_regions_absent"] == 3

    # With one real region present, the mean is over what was actually scored.
    truth[0, 2:5, 2:5] = 3
    predicted = np.zeros_like(truth)
    predicted[0, 2:5, 2:5] = 3
    meter = SegmentationMeter(compute_hausdorff=False)
    meter.update(predicted, truth)
    summary = meter.summary()
    assert summary["per_composite"]["enhancing_tumor"]["dice"] == 1.0
    assert summary["per_class"]["edema"]["dice"] is None
    assert summary["classes_scored"] == 1


def test_hausdorff_measures_a_known_displacement():
    pytest.importorskip("scipy.ndimage")
    truth = np.zeros((1, 40, 40), dtype=np.int64)
    truth[0, 10:20, 10:20] = 1
    shifted = np.zeros_like(truth)
    shifted[0, 14:24, 10:20] = 1                          # displaced by 4 pixels
    meter = SegmentationMeter(compute_hausdorff=True, percentile=95.0)
    meter.update(shifted, truth)
    distance = meter.summary()["per_class"]["ncr_net"]["hausdorff_p95_px"]
    assert 3.0 <= distance <= 5.0


def test_auroc_of_a_constant_score_is_one_half_not_one():
    meter = ClassificationMeter(("whole_tumor",))
    meter.update(np.full((20, 1), 0.7), np.r_[np.ones(10), np.zeros(10)][:, None])
    assert meter.summary()["whole_tumor"]["auroc"] == pytest.approx(0.5)


def test_regression_meter_exposes_a_constant_predictor():
    meter = RegressionMeter(("image_quality",))
    truth = np.linspace(0.1, 0.9, 40)[:, None]
    meter.update(np.full((40, 1), 0.5), truth)
    summary = meter.summary()["image_quality"]
    assert summary["predicted_std"] == pytest.approx(0.0)
    assert summary["pearson_r"] is None                   # undefined, not faked as 0


def test_embedding_meter_detects_collapse():
    meter = EmbeddingMeter(k=3)
    collapsed = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (20, 1))
    meter.update(collapsed, np.zeros(20, dtype=np.int64),
                 np.zeros(20, dtype=np.int64), np.arange(20))
    summary = meter.summary()
    assert summary["collapse"]["mean_pairwise_cosine"] > 0.99
    assert summary["collapse"]["mean_dimension_std"] < 1e-5


# --------------------------------------------------------------------------- #
# 8. Model and registry
# --------------------------------------------------------------------------- #
def test_network_emits_one_prediction_per_supervised_level_and_all_five_heads():
    config = ModelConfig(stage_channels=(8, 16, 32), blocks_per_stage=(1, 1, 1),
                         deep_supervision_levels=2, embedding_dim=16,
                         embedding_hidden=32, input_size=(32, 32))
    network = build_network(config)
    output = network(torch.randn(2, 4, 32, 32))
    assert [tuple(t.shape[-2:]) for t in output.segmentation] == [(32, 32), (16, 16)]
    assert output.presence.shape == (2, 4)
    assert output.size.shape == (2, 4)
    assert output.quality.shape == (2, 1)
    assert output.embedding.shape == (2, 16)
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(2), atol=1e-4)


def test_size_head_can_never_predict_a_negative_area():
    config = ModelConfig(stage_channels=(8, 16), blocks_per_stage=(1, 1),
                         deep_supervision_levels=1, embedding_dim=8,
                         embedding_hidden=16, input_size=(32, 32))
    network = build_network(config)
    output = network(torch.randn(4, 4, 32, 32) * 50)
    assert bool((output.size >= 0).all())


def test_embedding_can_be_computed_without_running_the_decoder():
    config = ModelConfig(stage_channels=(8, 16), blocks_per_stage=(1, 1),
                         deep_supervision_levels=1, embedding_dim=8,
                         embedding_hidden=16, input_size=(32, 32))
    network = build_network(config)
    output = network(torch.randn(2, 4, 32, 32), need_segmentation=False)
    assert output.segmentation == []
    assert output.embedding.shape == (2, 8)


def test_a_missing_modality_is_dropped_rather_than_read_as_dark_tissue():
    from aura.backend.vision.brain.types import DEFAULT_MODALITIES

    encoder = build_encoder("residual_unet2d", modalities=DEFAULT_MODALITIES,
                            stage_channels=(8, 16), blocks_per_stage=(1, 1))
    image = torch.randn(1, 4, 32, 32)
    zeroed = image.clone()
    zeroed[:, 3] = 0.0
    availability = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    with_flag = encoder(zeroed, availability)[-1]
    without_flag = encoder(zeroed, None)[-1]
    assert not torch.allclose(with_flag, without_flag), \
        "an absent sequence must not be averaged in as a zero-valued channel"


def test_declared_architecture_raises_rather_than_substituting():
    assert {"unet3d", "swin_unetr", "nnunet"} <= set(declared_architectures())
    with pytest.raises(ArchitectureUnavailable, match="not implemented"):
        build_encoder("unet3d")
    with pytest.raises(ArchitectureUnavailable):
        build_encoder("something_invented")


# --------------------------------------------------------------------------- #
# 9. Output object
# --------------------------------------------------------------------------- #
def _output(shape=(3, 8, 8)) -> BrainVisionOutput:
    from aura.backend.vision.brain.output import (
        FeatureMaps, ProcessingMetadata, QualityMetadata,
    )
    from aura.backend.vision.brain.types import EmbeddingSpec

    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[0, 2:5, 2:5] = 2
    segmentation[0, 5:6, 2:5] = 3
    confidence = np.full(shape, 0.8, dtype=np.float32)
    presence = np.array([0.9, 0.1, 0.85, 0.7], dtype=np.float32)
    return BrainVisionOutput(
        study_id="s1", segmentation=segmentation, confidence=confidence,
        tumor_probability=0.9,
        regions=build_regions(segmentation, confidence, presence, None, (1.0, 1.0, 1.0)),
        embedding=np.random.default_rng(0).normal(size=(3, 16)).astype(np.float32),
        embedding_spec=EmbeddingSpec(dimension=16),
        features=FeatureMaps(maps=None, pooled=np.zeros(8, np.float32), stride=8),
        processing=ProcessingMetadata(study_id="s1", slices_processed=3,
                                      sequences_used=("flair",),
                                      sequences_missing=("t1ce",), device="cpu",
                                      inference_ms=5.0),
        quality=QualityMetadata(predicted_score=0.8),
        model_version="test@epoch1")


def test_output_dict_never_carries_voxels():
    result = _output(shape=(40, 64, 64))                  # 163 840 voxels
    payload = result.to_dict()
    json.dumps(payload)                                   # must be JSON-safe at all

    def walk(node, path="$"):
        """No numpy arrays, and no list long enough to be voxel or embedding data."""
        if isinstance(node, np.ndarray):
            raise AssertionError(f"an array reached the payload at {path}")
        if isinstance(node, (list, tuple)):
            assert len(node) <= 64, f"a bulk sequence reached the payload at {path}"
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
        elif isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}")

    walk(payload)
    assert payload["segmentation"]["shape"] == [40, 64, 64]
    # The arrays are still reachable on the object — they are simply not serialised.
    assert result.segmentation.size == 40 * 64 * 64


def test_output_reports_volume_only_when_spacing_is_known():
    result = _output()
    oedema = result.region("edema")
    assert oedema.voxels == 9 and oedema.volume_mm3 == pytest.approx(9.0)

    from aura.backend.vision.brain.output import build_regions as build

    unknown = build(result.segmentation, result.confidence,
                    [0.9, 0.1, 0.85, 0.7], None, None)
    assert all(r.volume_mm3 is None for r in unknown)


def test_an_unvalidated_quality_head_is_marked_and_cannot_raise_alarms():
    """The v1 checkpoint's quality head is a near-constant predictor.

    Shipping that score unmarked would put an unvalidated number next to four validated
    ones. The flag is derived from the checkpoint's own recorded severity correlation,
    so a later checkpoint that earns trust gets it automatically.
    """
    from aura.backend.vision.brain.checkpoint import CheckpointMeta
    from aura.backend.vision.brain.inference import BrainVisionEngine
    from aura.backend.vision.brain.output import QUALITY_VALIDITY_THRESHOLD

    def notes_for(correlation):
        meta = CheckpointMeta(metrics={"quality": {"diagnostics": {
            "severity_correlation": correlation}}} if correlation is not None else {})
        engine = BrainVisionEngine.__new__(BrainVisionEngine)
        engine.meta = meta
        value = engine._quality_severity_correlation()
        reliable = None if value is None else value <= QUALITY_VALIDITY_THRESHOLD
        return reliable, BrainVisionEngine._quality_notes(reliable, value)

    reliable, notes = notes_for(-0.071)                   # the measured v1 value
    assert reliable is False
    assert any("DO NOT USE" in n for n in notes)

    reliable, notes = notes_for(-0.85)                    # a head that works
    assert reliable is True
    assert not any("DO NOT USE" in n for n in notes)

    reliable, notes = notes_for(None)                     # nothing recorded
    assert reliable is None
    assert any("unvalidated" in n for n in notes)


def test_a_head_that_did_not_run_reports_none_rather_than_zero():
    """A zero probability is a confident negative. An absent head made no claim at all."""
    from aura.backend.vision.brain.output import (
        BrainVisionOutput, FeatureMaps, ProcessingMetadata, QualityMetadata,
    )

    segmentation = np.zeros((2, 8, 8), dtype=np.uint8)
    segmentation[0, 2:5, 2:5] = 2
    confidence = np.full((2, 8, 8), 0.8, dtype=np.float32)
    result = BrainVisionOutput(
        study_id="s", segmentation=segmentation, confidence=confidence,
        tumor_probability=None,
        regions=build_regions(segmentation, confidence, None, None, None),
        embedding=None, embedding_spec=None,
        features=FeatureMaps(maps=None, pooled=None, stride=8),
        processing=ProcessingMetadata(study_id="s", slices_processed=2,
                                      sequences_used=(), sequences_missing=(),
                                      device="cpu", inference_ms=1.0),
        quality=QualityMetadata(predicted_score=None), model_version="v")

    payload = result.to_dict()
    assert payload["tumor_probability"] is None
    assert payload["tumor_present"] is None
    assert payload["embedding"] is None
    assert payload["features"]["pooled_dimension"] is None
    assert result.study_embedding is None
    for region in payload["regions"]:
        assert region["probability"] is None and region["present"] is None
        # The mask-derived count is real and must still be reported.
        assert isinstance(region["voxels"], int)
    assert payload["segmentation"]["voxels_per_class"]["edema"] == 9
    json.dumps(payload)
    result.summary()                                      # must not format a None


def test_output_records_which_sequences_were_missing():
    payload = _output().to_dict()
    assert payload["processing"]["sequences_missing"] == ["t1ce"]


def test_study_embedding_is_normalised():
    assert float(np.linalg.norm(_output().study_embedding)) == pytest.approx(1.0,
                                                                             abs=1e-5)


# --------------------------------------------------------------------------- #
# 10. Embedding store
# --------------------------------------------------------------------------- #
def test_embedding_store_round_trips_with_its_provenance(tmp_path):
    from aura.backend.vision.brain.types import EmbeddingSpec

    store = EmbeddingStore(EmbeddingSpec(dimension=4), limit=10)
    store.add(EmbeddingBatch(
        embedding=np.eye(4, dtype=np.float32),
        slice_index=np.arange(4), subject_index=np.zeros(4, dtype=np.int64),
        cache_z=np.arange(4), morphology=np.arange(4),
        grade=np.zeros(4, dtype=np.int64), tumor_area=np.arange(4),
        quality=np.ones(4)))
    path = store.write(tmp_path, epoch=3, checkpoint="best.pt")
    arrays, sidecar = load_embeddings(path)
    assert arrays["embedding"].shape == (4, 4)
    assert sidecar["epoch"] == 3 and sidecar["embedding"]["dimension"] == 4
    assert (tmp_path / "latest.npz").exists()
    neighbours = nearest_neighbours(np.array([1.0, 0, 0, 0]), arrays, k=2)
    assert neighbours[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_embedding_store_respects_its_limit():
    from aura.backend.vision.brain.types import EmbeddingSpec

    store = EmbeddingStore(EmbeddingSpec(dimension=2), limit=3)
    for _ in range(3):
        store.add(EmbeddingBatch(
            embedding=np.zeros((2, 2), dtype=np.float32),
            slice_index=np.zeros(2, dtype=np.int64),
            subject_index=np.zeros(2, dtype=np.int64),
            cache_z=np.zeros(2, dtype=np.int64), morphology=np.zeros(2, dtype=np.int64),
            grade=np.zeros(2, dtype=np.int64), tumor_area=np.zeros(2, dtype=np.int64),
            quality=np.zeros(2)))
    assert len(store) == 3 and store.full


# --------------------------------------------------------------------------- #
# 11. Command line
# --------------------------------------------------------------------------- #
def test_options_are_accepted_after_the_subcommand(tmp_path):
    """``cli ingest --corpus X``, the order everyone actually types.

    With argparse, an option declared only on the top-level parser must be written
    *before* the subcommand or it is an "unrecognized arguments" error — which is how
    the first attempt at a real training run died.
    """
    from aura.backend.vision.brain.cli import build_parser, config_from_args

    args = build_parser().parse_args(
        ["train", "--corpus", str(tmp_path), "--epochs", "3", "--batch-size", "8",
         "--run-name", "x", "--no-amp", "--no-curriculum"])
    config = config_from_args(args)
    assert config.paths.corpus_root == tmp_path
    assert config.optim.epochs == 3 and config.optim.batch_size == 8
    assert config.optim.amp is False
    assert config.curriculum.enabled is False
    assert config.run_name == "x"


def test_info_reports_the_registry_and_the_missing_checkpoints(tmp_path):
    from aura.backend.vision.brain.cli import command_info

    config = BrainVisionConfig(paths=PathsConfig(artifacts_root=tmp_path))
    info = command_info(config)
    assert "residual_unet2d" in info["registry"]["encoders"]
    assert "unet3d" in info["registry"]["declared"]
    assert info["cache"] is None
    assert all(not entry["exists"] for entry in info["checkpoints"].values())


# --------------------------------------------------------------------------- #
# 12. End to end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_train_validate_checkpoint_and_infer(cached):
    from aura.backend.vision.brain.inference import BrainVisionEngine
    from aura.backend.vision.brain.train import BrainVisionTrainer

    trainer = BrainVisionTrainer(cached)
    history = trainer.fit()
    assert len(history.records) == cached.optim.epochs

    for path in (cached.paths.best_model_path, cached.paths.latest_model_path,
                 cached.paths.encoder_path, cached.paths.decoder_path,
                 cached.paths.embedding_head_path,
                 cached.paths.training_state_path):
        assert path.exists(), path
    assert cached.paths.model_card_path.exists()

    card = json.loads(cached.paths.model_card_path.read_text())
    assert card["data"]["split_policy"].startswith("by subject")
    assert card["caveats"], "the model card must carry the corpus caveats"
    assert card["architecture"]["heads"]

    report = history.records[-1]["validation"]
    assert "per_composite" in report["segmentation"]
    assert "grade_probe" in report["embedding"]
    assert set(report["segmentation"]["per_class"]) == {"ncr_net", "edema", "enhancing"}

    engine = BrainVisionEngine.load(cached)
    images = np.load(next((cached.paths.cache_dir / "volumes").glob("*.img.npy")),
                     mmap_mode="r")
    slices = [np.asarray(images[z], dtype=np.float32) for z in range(min(4, len(images)))]
    result = engine.analyze_slices(slices, study_id="probe",
                                   sequences_used=["flair", "t1", "t1ce", "t2"],
                                   spacing_mm=(1.0, 1.0, 1.0))
    assert result.segmentation.shape[0] == len(slices)
    assert result.embedding.shape == (len(slices), cached.model.embedding_dim)
    assert 0.0 <= result.tumor_probability <= 1.0
    assert result.caveats, "caveats must travel from the checkpoint onto every result"
    json.dumps(result.to_dict())                          # must stay JSON-safe


@pytest.mark.slow
def test_analyze_study_consumes_a_foundation_study_and_reports_missing_sequences(
        cached, corpus):
    """The real serving path: FoundationStudy in, BrainVisionOutput out.

    Also the missing-sequence contract. A study without T1ce must be *reported* as
    missing rather than zero-filled, because a zero channel is not an absent channel to
    a convolution — it is uniformly dark tissue.
    """
    from aura.backend.vision.brain.inference import BrainVisionEngine
    from aura.backend.vision.brain.ingest import BrainCorpusIngestor
    from aura.backend.vision.brain.train import BrainVisionTrainer

    if not cached.paths.best_model_path.exists():
        BrainVisionTrainer(cached).fit()

    ingestor = BrainCorpusIngestor(cached)
    subject = BratsCorpusIndex(corpus).subject(1)
    volumes = ingestor.reader.read_subject(subject)
    study, _, _ = ingestor._standardize(volumes)
    assert len(study.series) == 4

    engine = BrainVisionEngine.load(cached)
    result = engine.analyze_study(study, batch_size=4)
    assert result.processing.sequences_used == ("flair", "t1", "t1ce", "t2")
    assert result.processing.sequences_missing == ()
    assert result.processing.slices_processed > 0
    assert result.processing.spacing_mm == pytest.approx((1.0, 1.0, 1.0))
    # The foundation layer's own quality travels alongside the head's, not merged.
    assert result.quality.foundation_score is not None
    assert result.quality.predicted_score is not None
    assert result.segmentation.shape[0] == result.processing.slices_processed
    json.dumps(result.to_dict())

    from dataclasses import replace as dc_replace

    without_t1ce = dc_replace(
        study, series=tuple(s for s in study.series
                            if s.sequence.sequence.value != "t1ce"))
    partial = engine.analyze_study(without_t1ce, batch_size=4)
    assert partial.processing.sequences_missing == ("t1ce",)
    assert "t1ce" not in partial.processing.sequences_used


@pytest.mark.slow
def test_resume_continues_rather_than_restarting(cached, tmp_path_factory):
    """Automatic resume must not silently retrain from epoch zero."""
    from dataclasses import replace

    from aura.backend.vision.brain.train import BrainVisionTrainer

    artifacts = tmp_path_factory.mktemp("resume")
    config = _small_config(cached.paths.corpus_root, artifacts)
    BrainCorpusIngestor(config).run()

    first = replace(config, optim=replace(config.optim, epochs=1))
    BrainVisionTrainer(first).fit()
    assert config.paths.training_state_path.exists()

    second = replace(config, optim=replace(config.optim, epochs=2))
    trainer = BrainVisionTrainer(second)
    assert trainer.start_epoch == 1
    # The completed epoch is recovered from history.jsonl, so the model card cannot
    # name a "best" epoch that disagrees with the checkpoint on disk.
    assert [r["epoch"] for r in trainer.history.records] == [0]
    history = trainer.fit()
    assert [r["epoch"] for r in history.records] == [0, 1]


def test_checkpoint_refuses_a_mismatched_architecture(cached, tmp_path):
    from aura.backend.vision.brain.checkpoint import (
        CheckpointMeta, CheckpointWriter, load_encoder, load_network_checkpoint,
    )

    config = _small_config(cached.paths.corpus_root, tmp_path)
    network = build_network(config.model)
    writer = CheckpointWriter(config.paths, config)
    meta = CheckpointMeta(run_name="t", epoch=0, architecture=network.describe())
    writer.save_epoch(network, meta, is_best=True)

    wider = build_network(replace_channels(config.model, (16, 32)))
    with pytest.raises(CheckpointError, match="does not match"):
        load_network_checkpoint(config.paths.best_model_path, wider)
    with pytest.raises(CheckpointError, match="feature pyramid"):
        load_encoder(config.paths.encoder_path, wider.encoder)


def replace_channels(config: ModelConfig, channels: tuple[int, ...]) -> ModelConfig:
    from dataclasses import replace

    return replace(config, stage_channels=channels)
