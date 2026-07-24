"""The embedding store: latent brain representations, written once and reused forever.

The requirement this satisfies is stated plainly in the specification — *future
NeuroMind modules should never need to recompute these embeddings* — and it is a
requirement about cost and about reproducibility in equal measure. Recomputing means
loading the network, which means the answer a downstream module gets depends on which
checkpoint happened to be on disk that day. A stored embedding is a fact with a
provenance.

So every stored vector travels with the checkpoint that produced it, the encoder
configuration, the embedding specification, and the sample it describes. A consumer that
finds ``embeddings/epoch_007.npz`` can tell whether it is looking at the same
representation its own model was fitted against, and refuse if not.

Storage is a compressed ``.npz`` of parallel arrays rather than one file per sample:
20 000 samples at 128 float32 is 10 MB in one file and 20 000 filesystem entries in the
other, and every access pattern a downstream module has — "all embeddings for subject
X", "the whole validation set" — is a slice of an array rather than a directory walk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.vision.brain.errors import BrainVisionError
from backend.vision.brain.types import BRAIN_VISION_VERSION, EmbeddingSpec

log = get_logger("vision.brain.embeddings")

#: Arrays every store holds, alongside ``embedding``.
_COLUMNS: tuple[str, ...] = ("slice_index", "subject_index", "cache_z",
                             "morphology", "grade", "tumor_area", "quality")


@dataclass
class EmbeddingBatch:
    """One batch's worth of embeddings and the descriptors that identify them."""

    embedding: np.ndarray            # (N, D) float32
    slice_index: np.ndarray          # index into the cache's slice table
    subject_index: np.ndarray
    cache_z: np.ndarray
    morphology: np.ndarray
    grade: np.ndarray
    tumor_area: np.ndarray
    quality: np.ndarray


class EmbeddingStore:
    """Accumulates embeddings across a validation pass and writes them once."""

    def __init__(self, spec: EmbeddingSpec, *, limit: int = 20000) -> None:
        self.spec = spec
        self.limit = int(limit)
        self._embeddings: list[np.ndarray] = []
        self._columns: dict[str, list[np.ndarray]] = {name: [] for name in _COLUMNS}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @property
    def full(self) -> bool:
        return self._count >= self.limit

    def add(self, batch: EmbeddingBatch) -> None:
        """Append a batch, stopping at the configured limit."""
        if self.full:
            return
        room = self.limit - self._count
        take = min(room, int(batch.embedding.shape[0]))
        self._embeddings.append(np.asarray(batch.embedding[:take], dtype=np.float32))
        for name in _COLUMNS:
            values = np.asarray(getattr(batch, name)[:take])
            self._columns[name].append(values)
        self._count += take

    # ------------------------------------------------------------------ #
    def arrays(self) -> dict[str, np.ndarray]:
        if not self._embeddings:
            return {"embedding": np.zeros((0, self.spec.dimension), dtype=np.float32),
                    **{name: np.zeros(0, dtype=np.int64) for name in _COLUMNS}}
        return {"embedding": np.concatenate(self._embeddings),
                **{name: np.concatenate(values)
                   for name, values in self._columns.items()}}

    def write(self, directory: Path, *, epoch: int, checkpoint: str,
              architecture: dict[str, Any] | None = None,
              subject_ids: Sequence[str] | None = None) -> Path:
        """Write ``epoch_XXX.npz`` plus a sidecar JSON describing it.

        The sidecar is separate so a consumer can read the provenance without loading
        10 MB of vectors, and so a directory listing is browsable by a human.
        """
        directory.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        path = directory / f"epoch_{epoch:03d}.npz"
        np.savez_compressed(path, **arrays)

        sidecar = {
            "brain_vision_version": BRAIN_VISION_VERSION,
            "epoch": epoch,
            "checkpoint": checkpoint,
            "count": int(arrays["embedding"].shape[0]),
            "embedding": self.spec.to_dict(),
            "architecture": dict(architecture or {}),
            "columns": list(_COLUMNS),
            "subject_ids": list(subject_ids or []),
            "usage": ("np.load(path)['embedding'] is (N, D) float32, L2-normalised. "
                      "Rows align with every other array in the file; slice_index "
                      "indexes the ingest cache's slice table."),
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2),
                                             encoding="utf-8")
        # A stable name for "the current representation", so a downstream module can
        # depend on a path rather than on knowing which epoch won.
        latest = directory / "latest.npz"
        np.savez_compressed(latest, **arrays)
        latest.with_suffix(".json").write_text(json.dumps(sidecar, indent=2),
                                               encoding="utf-8")
        log.info("embeddings exported", extra={"context": {
            "path": str(path), "count": sidecar["count"],
            "dimension": self.spec.dimension}})
        return path


def load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read an embedding store back, with its provenance.

    The entry point a future NeuroMind module calls. Refuses a file whose major version
    differs from this code's rather than returning vectors from a different space.
    """
    path = Path(path)
    if not path.exists():
        raise BrainVisionError(f"no embedding store at {path}",
                               detail={"path": str(path)})
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    sidecar_path = path.with_suffix(".json")
    sidecar: dict[str, Any] = (json.loads(sidecar_path.read_text(encoding="utf-8"))
                               if sidecar_path.exists() else {})
    version = str(sidecar.get("brain_vision_version", BRAIN_VISION_VERSION))
    if version.split(".")[0] != BRAIN_VISION_VERSION.split(".")[0]:
        raise BrainVisionError(
            f"the embedding store was written by version {version}; this code reads "
            f"{BRAIN_VISION_VERSION}. The latent space is not guaranteed comparable "
            "across major versions.",
            detail={"path": str(path), "store_version": version})
    return arrays, sidecar


def nearest_neighbours(query: np.ndarray, arrays: dict[str, np.ndarray], k: int = 10
                       ) -> list[dict[str, Any]]:
    """Most similar stored samples to ``query``, by cosine similarity.

    The one retrieval primitive this module provides, because it is what every planned
    consumer starts with: "show me the studies whose tumour looks like this one".
    Anything more — an index, a metric learner — belongs to the module that needs it.
    """
    embeddings = arrays["embedding"]
    if embeddings.size == 0:
        return []
    normalised = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    scores = normalised @ vector
    order = np.argsort(-scores)[:k]
    return [{"rank": rank, "similarity": round(float(scores[i]), 5),
             **{name: int(arrays[name][i]) for name in _COLUMNS if name in arrays}}
            for rank, i in enumerate(order, start=1)]


def group_by_subject(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """Row indices per subject — the access pattern a longitudinal module needs."""
    subjects = arrays.get("subject_index")
    if subjects is None:
        return {}
    return {int(s): np.flatnonzero(subjects == s) for s in np.unique(subjects)}


def subject_embeddings(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """One vector per subject: the mean of its slice embeddings, re-normalised.

    A crude study-level representation, and deliberately labelled as such. It is the
    right starting point for a longitudinal comparison and the wrong thing to build a
    prognosis on — a mean over slices weights an empty slice the same as the one through
    the tumour centre. A study-level head trained as such is future work.
    """
    grouped = group_by_subject(arrays)
    embeddings = arrays["embedding"]
    result: dict[int, np.ndarray] = {}
    for subject, rows in grouped.items():
        mean = embeddings[rows].mean(axis=0)
        result[subject] = mean / (np.linalg.norm(mean) + 1e-8)
    return result


def stack_batches(batches: Iterable[EmbeddingBatch]) -> EmbeddingBatch:
    """Concatenate batches. Used by tests and by offline export scripts."""
    batches = list(batches)
    if not batches:
        raise ValueError("no batches to stack")
    return EmbeddingBatch(
        embedding=np.concatenate([b.embedding for b in batches]),
        **{name: np.concatenate([getattr(b, name) for b in batches])
           for name in _COLUMNS})


__all__ = ["EmbeddingBatch", "EmbeddingStore", "group_by_subject", "load_embeddings",
           "nearest_neighbours", "stack_batches", "subject_embeddings"]
