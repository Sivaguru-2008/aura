"""The Brain Vision Engine's public result object.

Everything the specification asks a trained model to return — mask, confidences, tumour
probability, size estimate, latent embedding, feature maps, processing metadata, quality
metadata, model version — in one frozen dataclass, and **no torch tensor anywhere**.

That last constraint is not stylistic. A tensor that escapes this package carries three
things with it: an autograd graph, a CUDA context, and a device. Something downstream —
a report renderer, a cache, an async task — will eventually hold one alive, and the
symptom is a process whose GPU memory grows across requests with no allocation in sight.
Converting once, at the boundary, in :meth:`BrainVisionEngine._to_output`, costs a copy
and removes the class of bug.

:meth:`BrainVisionOutput.to_dict` is separately careful: it emits *descriptions* of the
arrays — shape, dtype, statistics — and never the arrays. A segmentation over a whole
study is tens of megabytes, and an API response or a log line is not where that belongs.
The arrays are reached through attributes by a caller that has decided it wants them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from .types import (
    BRAIN_VISION_VERSION,
    COMPOSITE_MEMBERS,
    CompositeRegion,
    FOREGROUND_REGIONS,
    REGION_KEYS,
    REGION_LABELS,
    EmbeddingSpec,
    TumorRegion,
)


@dataclass(frozen=True)
class RegionFinding:
    """One tumour region's measurement.

    ``probability`` comes from the presence head, not from thresholding the mask. They
    answer different questions and they disagree usefully: a mask with eleven enhancing
    voxels and a presence probability of 0.2 is a model that segmented something it does
    not believe in, and a report should be able to say so.
    """

    region: str
    label: str
    #: ``None`` when the model has no presence head to answer with.
    present: bool | None
    probability: float | None
    #: Voxels assigned this class by the argmax mask.
    voxels: int
    #: Physical volume, when voxel spacing is known. ``None`` when it is not — never
    #: silently assumed to be 1 mm isotropic.
    volume_mm3: float | None
    #: Independent size estimate from the regression head, in voxels. Kept beside the
    #: mask-derived count rather than replacing it: agreement between two routes to the
    #: same quantity is evidence, and a large disagreement is a flag.
    estimated_voxels: float | None
    #: Mean softmax confidence over the voxels assigned this class.
    mean_confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region, "label": self.label, "present": self.present,
            "probability": (round(self.probability, 5)
                            if self.probability is not None else None),
            "voxels": self.voxels,
            "volume_mm3": (round(self.volume_mm3, 2)
                           if self.volume_mm3 is not None else None),
            "estimated_voxels": (round(self.estimated_voxels, 1)
                                 if self.estimated_voxels is not None else None),
            "mean_confidence": (round(self.mean_confidence, 5)
                                if self.mean_confidence is not None else None),
        }


@dataclass(frozen=True)
class ProcessingMetadata:
    """What ran, on what, and how long it took."""

    study_id: str
    slices_processed: int
    sequences_used: tuple[str, ...]
    sequences_missing: tuple[str, ...]
    device: str
    inference_ms: float
    preprocessing_ms: float = 0.0
    input_size: tuple[int, int] = (0, 0)
    spacing_mm: tuple[float, float, float] | None = None
    foundation_version: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "slices_processed": self.slices_processed,
            "sequences_used": list(self.sequences_used),
            "sequences_missing": list(self.sequences_missing),
            "device": self.device,
            "inference_ms": round(self.inference_ms, 3),
            "preprocessing_ms": round(self.preprocessing_ms, 3),
            "input_size": list(self.input_size),
            "spacing_mm": ([round(v, 4) for v in self.spacing_mm]
                           if self.spacing_mm else None),
            "foundation_version": self.foundation_version,
            "created_at": self.created_at,
        }


#: A quality head is considered to work when its predicted score tracks known
#: degradation severity at least this strongly (correlation is negative by
#: construction: more severity, less quality). Chosen as the point below which the
#: prediction explains under ~10% of the variance in severity and is not worth acting on.
QUALITY_VALIDITY_THRESHOLD = -0.30


@dataclass(frozen=True)
class QualityMetadata:
    """Image quality, from two independent sources that are kept apart.

    ``foundation_score`` is measured by the MRI Foundation Layer from the volume itself:
    resolution, field of view, slice completeness, intensity, noise, motion. ``predicted
    _score`` is this network's quality head. They are not the same quantity and merging
    them into one "quality" number would destroy the only useful thing about having
    both — that a disagreement is informative. A high foundation score with a low
    predicted score means the artefact is one the foundation checks do not cover.

    ``predicted_score_reliable`` is the gate, and it is derived rather than declared: it
    comes from the checkpoint's own recorded ``severity_correlation`` — how well this
    model's quality head tracked degradations of known severity on the held-out split —
    compared against :data:`QUALITY_VALIDITY_THRESHOLD`. It is ``False`` for the v1
    checkpoint, whose head measured -0.07 and is a near-constant predictor. A consumer
    that checks this flag will start trusting the score automatically once a checkpoint
    earns it, and no sooner.
    """

    predicted_score: float | None
    predicted_per_slice: tuple[float, ...] = ()
    foundation_score: float | None = None
    foundation_verdict: str | None = None
    foundation_warnings: tuple[str, ...] = ()
    #: True when either *usable* source falls below the usability floor. An unreliable
    #: predicted score never triggers it — a broken head must not be able to raise
    #: alarms any more than it may be believed.
    review_recommended: bool = False
    #: Whether this checkpoint's quality head passed its own validation.
    predicted_score_reliable: bool | None = None
    #: The measurement behind the flag, carried so the judgement can be re-checked.
    severity_correlation: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_score": (round(self.predicted_score, 5)
                                if self.predicted_score is not None else None),
            "predicted_score_min": (round(float(min(self.predicted_per_slice)), 5)
                                    if self.predicted_per_slice else None),
            "predicted_score_reliable": self.predicted_score_reliable,
            "severity_correlation": (round(self.severity_correlation, 5)
                                     if self.severity_correlation is not None else None),
            "foundation_score": (round(self.foundation_score, 5)
                                 if self.foundation_score is not None else None),
            "foundation_verdict": self.foundation_verdict,
            "foundation_warnings": list(self.foundation_warnings),
            "review_recommended": self.review_recommended,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FeatureMaps:
    """Spatial features, exposed at a size that is honest about being reusable.

    Full-resolution decoder activations for a 155-slice study are several gigabytes, so
    what travels is the *coarsest* decoder level — one eighth resolution by default —
    plus the pooled bottleneck vector. That is enough for the things feature maps are
    actually wanted for downstream: attention overlays, coarse localisation, a spatial
    prior for a registration or a planning module. Anything needing full resolution
    should re-run the network with ``need_features=True`` rather than have every caller
    pay to carry it.
    """

    #: ``(levels, C, H, W)`` at the exported resolution, or ``None`` when not requested.
    maps: np.ndarray | None
    #: Pooled bottleneck, ``(C,)``. ``None`` when no forward pass produced one —
    #: never a zero vector, which a consumer would read as a real all-zero
    #: representation rather than as an absence.
    pooled: np.ndarray | None
    stride: int
    level: str = "decoder_coarsest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level, "stride": self.stride,
            "maps_shape": list(self.maps.shape) if self.maps is not None else None,
            "pooled_dimension": (int(self.pooled.shape[-1])
                                 if self.pooled is not None else None),
            "note": "arrays are reachable on the object; they are not serialised here",
        }


@dataclass(frozen=True)
class BrainVisionOutput:
    """The Brain Vision Engine's structured result. Numpy in, numpy out, never tensors."""

    study_id: str
    #: Dense class labels, ``(Z, H, W)`` uint8 for a study or ``(H, W)`` for one slice.
    segmentation: np.ndarray
    #: Per-voxel confidence — the maximum softmax probability of the assigned class.
    confidence: np.ndarray
    #: Whole-tumour probability from the presence head. ``None`` when the model has
    #: no presence head — never 0.0, which is a confident negative it never made.
    tumor_probability: float | None
    regions: tuple[RegionFinding, ...]
    #: Latent brain representation. One vector per processed slice, plus the study-level
    #: mean in :attr:`study_embedding`.
    embedding: np.ndarray | None
    embedding_spec: EmbeddingSpec | None
    features: FeatureMaps
    processing: ProcessingMetadata
    quality: QualityMetadata
    model_version: str
    brain_vision_version: str = BRAIN_VISION_VERSION
    #: Corpus and model caveats, carried from the checkpoint. Present on every result
    #: because the alternative is a caller who has never read the model card.
    caveats: tuple[str, ...] = ()
    #: Per-slice class probabilities, only when a caller asked for them.
    class_probabilities: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    @property
    def study_embedding(self) -> np.ndarray | None:
        """One L2-normalised vector for the whole study — the mean over slices.

        Crude on purpose, and labelled as such in :mod:`aura.backend.vision.brain.embeddings`:
        a mean over slices weights an empty slice like the one through the tumour
        centre. It is a reasonable retrieval key and not a prognostic feature.
        """
        if self.embedding is None or self.embedding.size == 0:
            return None
        if self.embedding.ndim == 1:
            return self.embedding
        mean = self.embedding.mean(axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)

    @property
    def tumor_present(self) -> bool | None:
        """``None`` when there is no presence head to ask."""
        if self.tumor_probability is None:
            return None
        return self.tumor_probability >= 0.5

    @property
    def total_tumor_voxels(self) -> int:
        return int(np.count_nonzero(self.segmentation > 0))

    def region(self, name: str) -> RegionFinding | None:
        return next((r for r in self.regions if r.region == name), None)

    def composite_mask(self, region: CompositeRegion) -> np.ndarray:
        """Boolean mask of a BraTS composite region, derived from the class labels."""
        mask = np.zeros(self.segmentation.shape, dtype=bool)
        for member in COMPOSITE_MEMBERS[region]:
            mask |= self.segmentation == member.value
        return mask

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary. Voxels are described, never serialised."""
        return {
            "study_id": self.study_id,
            "model_version": self.model_version,
            "brain_vision_version": self.brain_vision_version,
            "tumor_present": self.tumor_present,
            "tumor_probability": (round(self.tumor_probability, 5)
                                  if self.tumor_probability is not None else None),
            "total_tumor_voxels": self.total_tumor_voxels,
            "segmentation": {
                "shape": list(self.segmentation.shape),
                "dtype": str(self.segmentation.dtype),
                "classes": {int(r.value): REGION_LABELS[r] for r in TumorRegion},
                "voxels_per_class": {
                    REGION_KEYS[r]: int(np.count_nonzero(self.segmentation == r.value))
                    for r in TumorRegion},
            },
            "confidence": {
                "mean": round(float(self.confidence.mean()), 5)
                        if self.confidence.size else None,
                "mean_over_tumor": (
                    round(float(self.confidence[self.segmentation > 0].mean()), 5)
                    if self.total_tumor_voxels else None),
                "p05": round(float(np.percentile(self.confidence, 5)), 5)
                       if self.confidence.size else None,
            },
            "composite_regions": {
                composite.value: {
                    "voxels": int(np.count_nonzero(self.composite_mask(composite))),
                    "members": [REGION_KEYS[m] for m in COMPOSITE_MEMBERS[composite]],
                }
                for composite in CompositeRegion},
            "regions": [r.to_dict() for r in self.regions],
            "embedding": ({**self.embedding_spec.to_dict(),
                           "vectors": int(self.embedding.shape[0])
                                      if self.embedding.ndim > 1 else 1}
                          if self.embedding is not None
                          and self.embedding_spec is not None else None),
            "features": self.features.to_dict(),
            "processing": self.processing.to_dict(),
            "quality": self.quality.to_dict(),
            "caveats": list(self.caveats),
        }

    def summary(self) -> str:
        """One line for a log or a report header."""
        found = [r.region for r in self.regions
                 if r.present and r.region != "whole_tumor"]
        probability = ("unavailable" if self.tumor_probability is None
                       else f"{self.tumor_probability:.2f}")
        line = (f"{self.study_id}: tumour p={probability}, "
                f"{self.total_tumor_voxels} voxel(s) over "
                f"{self.processing.slices_processed} slice(s), regions "
                f"[{', '.join(found) or 'none'}]")
        if self.quality.predicted_score is not None:
            line += f", quality {self.quality.predicted_score:.2f}"
        if self.quality.review_recommended:
            line += " [review recommended]"
        return line


def build_regions(segmentation: np.ndarray, confidence: np.ndarray,
                  presence: Sequence[float] | None,
                  estimated_voxels: Sequence[float] | None,
                  spacing_mm: Sequence[float] | None) -> tuple[RegionFinding, ...]:
    """Assemble the per-region findings from a mask, a presence vector, and a size vector.

    ``presence`` and ``estimated_voxels`` follow the head's own ordering: whole tumour
    first, then the three primary classes. Either may be ``None`` when the model has
    no such head, and the corresponding fields then read ``None`` rather than a
    substituted probability. Voxel counts come from the mask and are always real.
    """
    voxel_volume = (float(np.prod(spacing_mm)) if spacing_mm is not None else None)
    findings: list[RegionFinding] = []

    total_voxels = int(np.count_nonzero(segmentation > 0))
    whole_confidence = (float(confidence[segmentation > 0].mean())
                        if total_voxels else None)
    whole_probability = float(presence[0]) if presence is not None else None
    findings.append(RegionFinding(
        region="whole_tumor", label="Whole tumour",
        present=(whole_probability >= 0.5)
                if whole_probability is not None else None,
        probability=whole_probability,
        voxels=total_voxels,
        volume_mm3=(total_voxels * voxel_volume) if voxel_volume else None,
        estimated_voxels=(float(estimated_voxels[0]) if estimated_voxels is not None
                          else None),
        mean_confidence=whole_confidence))

    for position, region in enumerate(FOREGROUND_REGIONS, start=1):
        mask = segmentation == region.value
        voxels = int(np.count_nonzero(mask))
        probability = float(presence[position]) if presence is not None else None
        findings.append(RegionFinding(
            region=REGION_KEYS[region], label=REGION_LABELS[region],
            present=(probability >= 0.5) if probability is not None else None,
            probability=probability,
            voxels=voxels,
            volume_mm3=(voxels * voxel_volume) if voxel_volume else None,
            estimated_voxels=(float(estimated_voxels[position])
                              if estimated_voxels is not None else None),
            mean_confidence=(float(confidence[mask].mean()) if voxels else None)))
    return tuple(findings)


__all__ = ["BrainVisionOutput", "FeatureMaps", "ProcessingMetadata", "QualityMetadata",
           "RegionFinding", "build_regions"]
