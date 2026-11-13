"""Encoder embeddings for all 369 official BraTS 2020 subjects, with official grades.

Why this exists
---------------
The full-coverage kernel comparison (``qkl_full_coverage.py``) read ``epoch_001.npz``,
which was exported from ``datasets/brats2021_hdf5``. The 2026-09-13 provenance audit
found that folder's subject identifiers do not exist in BraTS 2020 and that its grades
agree with the official mapping in only 32 of 47 colliding names. The comparison was
withdrawn from the paper (v5 erratum) until it could be repeated on data whose labels
trace to an official source.

This script produces that data. It reads the BraTS 2020 HDF5 corpus the served brain
model was trained on (``best_brain_model.pt``; 258/55/56 subjects), takes grade from
the corpus's own ``name_mapping.csv`` (the challenge's release file, 293 HGG / 76 LGG),
and runs the served checkpoint's encoder over every retained slice of every subject.

Preprocessing parity with training is by construction, not by reimplementation: each
subject goes through ``BrainCorpusIngestor._ingest_subject`` (the ingest the model was
trained from) and each slice through ``normalize_slice`` and ``fit_to_grid`` with a
centred crop — exactly ``BrainSliceDataset.__getitem__`` with augmentation and synthetic
degradation off. Degradation is off deliberately: the validation pass that wrote the
earlier dumps applies synthetic artefacts to a fixed subset of slices, which is right for
testing the quality head and wrong for a grading feature.

Disk: a full ingest cache of 369 subjects is ~30 GB. This streams instead — one subject
is cached into a scratch directory, embedded, and deleted before the next.

Label-mapping check
-------------------
The HDF5 redistribution names subjects ``volume_N``; the reader maps that to
``BraTS20_Training_{N:03d}``. That mapping is what the grades hang on, so it is tested
against the images rather than assumed: low-grade gliomas frequently show no contrast
enhancement, high-grade ones rarely do. If volume numbers were assigned to subjects in
some other order, enhancing-tumour absence would be independent of the mapped grade.
Per-subject enhancing-voxel counts are written so the check can be run on the output
(``verify_brats2020_grade_mapping.py``).

Writes ``artifacts/brain/embeddings_official/brats2020_369.npz`` and a JSON sidecar.
Nothing under ``artifacts/brain/embeddings`` or ``artifacts/brain/cache`` is touched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from aura.backend.vision.brain.config import BrainVisionConfig, PathsConfig
from aura.backend.vision.brain.dataset import fit_to_grid, normalize_slice
from aura.backend.vision.brain.ingest import BrainCorpusIngestor, assign_splits
from aura.backend.vision.brain.io.brats_h5 import BratsCorpusIndex
from aura.backend.vision.brain.types import TumorGrade, TumorRegion
from aura.common.config import ARTIFACTS

CORPUS = Path(r"E:\AURA\datasets\brats2020\BraTS2020_training_data\content\data")
SERVED_CHECKPOINT = ARTIFACTS / "brain" / "checkpoints" / "best_brain_model.pt"
OUT_DIR = ARTIFACTS / "brain" / "embeddings_official"
SCRATCH = ARTIFACTS / "brain" / "_export_scratch"

#: Same coding as the training dataset and the existing dumps: 0 = HGG, 1 = LGG.
GRADE_CODE = {TumorGrade.HGG: 0, TumorGrade.LGG: 1}


def main(argv: list[str] | None = None) -> int:
    import torch

    from aura.backend.vision.brain.checkpoint import load_network_checkpoint
    from aura.backend.vision.brain.model.network import build_network

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--checkpoint", type=Path, default=SERVED_CHECKPOINT)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "brats2020_369.npz")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    started = time.time()
    paths = replace(PathsConfig(), corpus_root=args.corpus, artifacts_root=SCRATCH)
    config = BrainVisionConfig(paths=paths)
    paths.ensure()
    (paths.cache_dir / "volumes").mkdir(parents=True, exist_ok=True)

    index = BratsCorpusIndex(args.corpus)
    pool = list(index.subjects())
    ungraded = [s.subject_id for s in pool if s.grade not in GRADE_CODE]
    if ungraded:
        raise SystemExit(f"{len(ungraded)} subjects have no official grade: {ungraded[:5]}")
    # The split is computed over the whole corpus before any truncation, so a
    # --max-subjects smoke run reports the same split each subject had in training.
    splits = assign_splits(pool, config.split)
    if args.max_subjects:
        pool = pool[:args.max_subjects]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = build_network(config.model).to(device).eval()
    meta = load_network_checkpoint(args.checkpoint, network, device=str(device))
    print(f"[model] {args.checkpoint.name} epoch {meta.epoch} on {device}", flush=True)

    ingestor = BrainCorpusIngestor(config)
    size = config.model.input_size

    emb, subj_idx, cache_z, source_slice, grade, tumour_area = [], [], [], [], [], []
    subjects_out = []

    for position, subject in enumerate(pool):
        t0 = time.time()
        record, rows, _ = ingestor._ingest_subject(subject, splits[subject.subject_id],
                                                   position)
        vol_dir = paths.cache_dir / "volumes"
        img_path = vol_dir / f"{subject.subject_id}.img.npy"
        seg_path = vol_dir / f"{subject.subject_id}.seg.npy"
        images = np.load(img_path)               # (Z, C, H, W) float16
        labels = np.load(seg_path)               # (Z, H, W) uint8

        batch_x, batch_area = [], []
        vectors = []
        for z in range(images.shape[0]):
            image = np.asarray(images[z], dtype=np.float32)
            label = np.asarray(labels[z], dtype=np.uint8)
            brain = image.max(axis=0) > 0
            image = normalize_slice(image, brain)
            image, label = fit_to_grid(image, label, size, rng=None)
            batch_x.append(np.ascontiguousarray(image, dtype=np.float32))
            batch_area.append(int((label > 0).sum()))
        with torch.no_grad():
            for start in range(0, len(batch_x), args.batch_size):
                x = torch.from_numpy(np.stack(batch_x[start:start + args.batch_size]))
                x = x.to(device)
                with torch.autocast(device_type=device.type,
                                    enabled=device.type == "cuda"):
                    out = network(x, need_features=False, need_segmentation=False)
                vectors.append(out.embedding.float().cpu().numpy())
        vectors = np.concatenate(vectors)

        n = vectors.shape[0]
        emb.append(vectors)
        subj_idx.append(np.full(n, position, dtype=np.int64))
        cache_z.append(np.arange(n, dtype=np.int64))
        source_slice.append(np.asarray([int(r[2]) for r in rows], dtype=np.int64))
        grade.append(np.full(n, GRADE_CODE[subject.grade], dtype=np.int64))
        tumour_area.append(np.asarray(batch_area, dtype=np.int64))

        subjects_out.append({
            "subject_index": position,
            "subject_id": subject.subject_id,
            "volume_id": subject.volume_id,
            "grade": subject.grade.value,
            "split": splits[subject.subject_id].value,
            "slices": int(n),
            "voxels": {
                "ncr_net": int((labels == TumorRegion.NECROTIC_CORE.value).sum()),
                "edema": int((labels == TumorRegion.EDEMA.value).sum()),
                "enhancing": int((labels == TumorRegion.ENHANCING.value).sum()),
            },
        })

        img_path.unlink(missing_ok=True)
        seg_path.unlink(missing_ok=True)
        (paths.studies_dir / f"{subject.subject_id}.json").unlink(missing_ok=True)
        print(f"[{position + 1}/{len(pool)}] {subject.subject_id} {subject.grade.value} "
              f"{splits[subject.subject_id].value} slices={n} "
              f"[{time.time() - t0:.1f}s]", flush=True)

    shutil.rmtree(SCRATCH, ignore_errors=True)

    arrays = {
        "embedding": np.concatenate(emb).astype(np.float32),
        "subject_index": np.concatenate(subj_idx),
        "cache_z": np.concatenate(cache_z),
        "source_slice": np.concatenate(source_slice),
        "grade": np.concatenate(grade),
        "tumor_area": np.concatenate(tumour_area),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    split_counts = {}
    for s in subjects_out:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    sidecar = {
        "purpose": ("Encoder embeddings for every official BraTS 2020 subject, graded "
                    "from the release's name_mapping.csv. Replaces epoch_001.npz "
                    "(datasets/brats2021_hdf5, untraceable IDs and grades) as the input "
                    "to the full-coverage kernel comparison."),
        "corpus": str(args.corpus),
        "grade_source": str(args.corpus / "name_mapping.csv"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": meta.epoch,
        "preprocessing": ("ingest._ingest_subject -> normalize_slice -> centred "
                          "fit_to_grid; no augmentation, no synthetic degradation"),
        "grade_coding": {"HGG": 0, "LGG": 1},
        "count": int(arrays["embedding"].shape[0]),
        "n_subjects": len(subjects_out),
        "grades": {g: sum(1 for s in subjects_out if s["grade"] == g)
                   for g in ("HGG", "LGG")},
        "splits_as_trained": split_counts,
        "encoder_training_note": ("The encoder was trained (segmentation, presence, "
                                  "size, quality, morphology-contrastive) on the 258 "
                                  "'train' subjects. Grade was never a training target. "
                                  "Val and test subjects (111) never influenced weights."),
        "subjects": subjects_out,
        "runtime_seconds": round(time.time() - started, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2),
                                             encoding="utf-8")
    print(f"[done] {sidecar['count']} slices, {len(subjects_out)} subjects, "
          f"splits {split_counts} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
