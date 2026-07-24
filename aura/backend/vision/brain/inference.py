"""Serving: a :class:`FoundationStudy` in, a :class:`BrainVisionOutput` out.

This is the boundary. Everything on the inside is tensors and torch; everything that
crosses it is numpy and plain Python. :meth:`BrainVisionEngine._to_output` is where the
conversion happens, once, deliberately.

Two things this module refuses to do, and both refusals matter more than what it does.

**It will not run without trained weights.** A randomly initialised network produces a
segmentation that is visually obvious nonsense in a debug viewer and completely
plausible in a JSON payload — a mask, some confidences, a tumour probability of 0.5.
There is no downstream check that would catch it, so
:class:`~backend.vision.brain.errors.ModelNotTrained` is raised at load rather than a
warning being logged at inference.

**It will not silently substitute a missing sequence.** The network is trained on four
co-registered sequences. When a study lacks one, the corresponding channel is zeroed
*and* its availability flag is cleared, so the encoder's per-modality stem drops it from
the average instead of interpreting an all-zero channel as "uniformly dark tissue" —
which is what a zero-filled channel means to a convolution. Which sequences were used
and which were missing is recorded on every result.

Slice selection is by brain content, using the same criterion the ingest applies, so a
study is scored over the same slices it would have contributed to training rather than
over 155 slices of which a third are air.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from backend.core.shared.logging import get_logger
from backend.vision.brain.checkpoint import CheckpointMeta, load_network_checkpoint
from backend.vision.brain.config import BrainVisionConfig, ModelConfig
from backend.vision.brain.dataset import decode_size, fit_to_grid, normalize_slice
from backend.vision.brain.errors import ModelNotTrained
from backend.vision.brain.model.network import BrainVisionNetwork, build_network
from backend.vision.brain.output import (
    QUALITY_VALIDITY_THRESHOLD,
    BrainVisionOutput,
    FeatureMaps,
    ProcessingMetadata,
    QualityMetadata,
    build_regions,
)
from backend.vision.brain.types import BRAIN_VISION_VERSION, ModalitySpec

log = get_logger("vision.brain.inference")

#: Quality below which the result carries a review recommendation.
_REVIEW_QUALITY_FLOOR = 0.60


class BrainVisionEngine:
    """Loads a trained network and produces :class:`BrainVisionOutput` objects."""

    def __init__(self, network: BrainVisionNetwork, meta: CheckpointMeta, *,
                 config: BrainVisionConfig, device: torch.device) -> None:
        self.network = network.eval()
        self.meta = meta
        self.config = config
        self.device = device

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config: BrainVisionConfig | None = None, *,
             checkpoint: Path | None = None,
             device: str | None = None) -> "BrainVisionEngine":
        """Build the network from the checkpoint's own architecture record and load it.

        The architecture comes from the checkpoint, not from the caller's configuration.
        A checkpoint trained with 320-channel stages loaded into a network configured for
        256 is a wall of shape errors at best and, with ``strict=False`` somewhere in the
        chain, a partially initialised network at worst.
        """
        config = config or BrainVisionConfig()
        path = Path(checkpoint or config.paths.best_model_path)
        if not path.exists():
            raise ModelNotTrained(
                f"no trained Brain Vision checkpoint at {path}",
                detail={"path": str(path),
                        "hint": "run `python -m backend.vision.brain.cli train`"})

        resolved = torch.device(device or config.device
                                or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_config = _model_config_from_checkpoint(path, config.model,
                                                     str(resolved))
        network = build_network(model_config).to(resolved)
        meta = load_network_checkpoint(path, network, device=str(resolved))
        log.info("brain vision engine ready", extra={"context": {
            "checkpoint": str(path), "device": str(resolved),
            "epoch": meta.epoch, "monitor": meta.monitor,
            "value": meta.monitor_value}})
        return cls(network, meta, config=config.with_overrides(model=model_config),
                   device=resolved)

    # ------------------------------------------------------------------ #
    def analyze_study(self, study: Any, *, batch_size: int = 8,
                      return_probabilities: bool = False,
                      return_feature_maps: bool = True) -> BrainVisionOutput:
        """Analyse a :class:`~backend.foundation.mri.study.FoundationStudy`.

        The study's series are matched to the network's declared modalities by their
        detected :class:`~backend.foundation.mri.types.SequenceType`. A sequence the
        study does not have is reported as missing rather than substituted.
        """
        started = time.perf_counter()
        volumes, used, missing, spacing = self._assemble(study)
        preprocessing_ms = (time.perf_counter() - started) * 1000.0

        brain = np.any(volumes > 0, axis=0)
        per_slice = brain.reshape(-1, brain.shape[2]).mean(axis=0)
        keep = np.flatnonzero(per_slice >= self.config.ingest.min_brain_fraction)
        if keep.size == 0:
            keep = np.arange(volumes.shape[3])

        slices = [np.ascontiguousarray(volumes[..., int(z)]) for z in keep]
        result = self.analyze_slices(
            slices, study_id=getattr(study, "study_id", "study"),
            sequences_used=used, sequences_missing=missing, spacing_mm=spacing,
            batch_size=batch_size, return_probabilities=return_probabilities,
            return_feature_maps=return_feature_maps,
            foundation_quality=_foundation_quality(study),
            preprocessing_ms=preprocessing_ms)
        return result

    @torch.no_grad()
    def analyze_slices(self, slices: Sequence[np.ndarray], *, study_id: str = "study",
                       sequences_used: Sequence[str] = (),
                       sequences_missing: Sequence[str] = (),
                       spacing_mm: Sequence[float] | None = None,
                       batch_size: int = 8, return_probabilities: bool = False,
                       return_feature_maps: bool = True,
                       foundation_quality: dict[str, Any] | None = None,
                       preprocessing_ms: float = 0.0) -> BrainVisionOutput:
        """Analyse a sequence of ``(C, H, W)`` background-zero slices."""
        if not slices:
            raise ValueError("no slices were supplied")
        availability = torch.tensor(
            [[0.0 if spec.key in set(sequences_missing) else 1.0
              for spec in self.network.modalities]], device=self.device)

        masks: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        presences: list[np.ndarray] = []
        sizes: list[np.ndarray] = []
        qualities: list[float] = []
        pooled: list[np.ndarray] = []
        coarse_maps: list[np.ndarray] = []
        original_shapes: list[tuple[int, int]] = []

        started = time.perf_counter()
        for start in range(0, len(slices), batch_size):
            chunk = slices[start:start + batch_size]
            prepared = []
            for array in chunk:
                array = np.asarray(array, dtype=np.float32)
                original_shapes.append((array.shape[1], array.shape[2]))
                brain = array.max(axis=0) > 0
                normalised = normalize_slice(array, brain)
                fitted, _ = fit_to_grid(normalised,
                                        np.zeros(array.shape[1:], dtype=np.uint8),
                                        self.config.model.input_size)
                prepared.append(fitted)
            batch = torch.from_numpy(np.stack(prepared)).to(self.device)

            with torch.autocast(device_type=self.device.type,
                                enabled=self.device.type == "cuda"):
                output = self.network(
                    batch, availability=availability.expand(batch.shape[0], -1),
                    need_features=return_feature_maps)

            logits = output.logits.float()
            softmax = torch.softmax(logits, dim=1)
            masks.append(softmax.argmax(dim=1).cpu().numpy().astype(np.uint8))
            confidences.append(softmax.max(dim=1).values.cpu().numpy())
            if return_probabilities:
                probabilities.append(softmax.cpu().numpy().astype(np.float16))
            if output.embedding is not None:
                embeddings.append(output.embedding.float().cpu().numpy())
            if output.presence is not None:
                presences.append(torch.sigmoid(output.presence.float()).cpu().numpy())
            if output.size is not None:
                sizes.append(output.size.float().cpu().numpy())
            if output.quality is not None:
                qualities.extend(output.quality.float().cpu().numpy().reshape(-1)
                                 .tolist())
            if output.pooled is not None:
                pooled.append(output.pooled.float().cpu().numpy())
            if return_feature_maps and output.decoder_features:
                coarse_maps.append(
                    output.decoder_features[-1].float().cpu().numpy().astype(np.float16))
        inference_ms = (time.perf_counter() - started) * 1000.0

        return self._to_output(
            study_id=study_id, masks=masks, confidences=confidences,
            probabilities=probabilities, embeddings=embeddings, presences=presences,
            sizes=sizes, qualities=qualities, pooled=pooled, coarse_maps=coarse_maps,
            sequences_used=sequences_used, sequences_missing=sequences_missing,
            spacing_mm=spacing_mm, inference_ms=inference_ms,
            preprocessing_ms=preprocessing_ms,
            foundation_quality=foundation_quality or {})

    # ------------------------------------------------------------------ #
    def _to_output(self, *, study_id: str, masks: list[np.ndarray],
                   confidences: list[np.ndarray], probabilities: list[np.ndarray],
                   embeddings: list[np.ndarray], presences: list[np.ndarray],
                   sizes: list[np.ndarray], qualities: list[float],
                   pooled: list[np.ndarray], coarse_maps: list[np.ndarray],
                   sequences_used: Sequence[str], sequences_missing: Sequence[str],
                   spacing_mm: Sequence[float] | None, inference_ms: float,
                   preprocessing_ms: float,
                   foundation_quality: dict[str, Any]) -> BrainVisionOutput:
        """The tensor/numpy boundary. Nothing above this line leaves the package."""
        # Nothing here substitutes a value for a head that did not run. A zero vector
        # standing in for an absent presence head reads downstream as "probability 0.0"
        # — a confident negative the model never produced — and a zero feature vector
        # reads as a real embedding. Absent is ``None``, everywhere.
        segmentation = np.concatenate(masks) if masks else np.zeros((0, 0, 0), np.uint8)
        confidence = (np.concatenate(confidences) if confidences
                      else np.zeros((0, 0, 0), np.float32))
        embedding = np.concatenate(embeddings) if embeddings else None
        presence = np.concatenate(presences).mean(axis=0) if presences else None
        # Size is the *sum* over slices, not the mean: the head predicts a per-slice
        # area and a study's tumour is the union of its slices. Averaging would report
        # the mean cross-section as if it were the volume.
        size_estimate = None
        if sizes:
            pixels = float(np.prod(self.config.model.input_size))
            size_estimate = decode_size(np.concatenate(sizes), pixels).sum(axis=0)

        predicted_quality = float(np.mean(qualities)) if qualities else None
        foundation_score = foundation_quality.get("score")
        severity_correlation = self._quality_severity_correlation()
        quality_reliable = (None if severity_correlation is None
                            else severity_correlation <= QUALITY_VALIDITY_THRESHOLD)
        # An unreliable head neither raises alarms nor suppresses them. Only the
        # foundation layer's measured score, and a predicted score that earned it, may
        # recommend review.
        review = bool(
            (quality_reliable and predicted_quality is not None
             and predicted_quality < _REVIEW_QUALITY_FLOOR)
            or (foundation_score is not None
                and float(foundation_score) < _REVIEW_QUALITY_FLOOR))

        features = FeatureMaps(
            maps=np.concatenate(coarse_maps) if coarse_maps else None,
            pooled=np.concatenate(pooled).mean(axis=0) if pooled else None,
            stride=int(self.network.decoder.strides[-1])
                   if getattr(self.network.decoder, "strides", None) else 1)

        return BrainVisionOutput(
            study_id=study_id,
            segmentation=segmentation,
            confidence=confidence,
            tumor_probability=float(presence[0]) if presence is not None else None,
            regions=build_regions(segmentation, confidence, presence, size_estimate,
                                  spacing_mm),
            embedding=embedding,
            embedding_spec=(self.network.embedding_spec
                            if embedding is not None else None),
            features=features,
            processing=ProcessingMetadata(
                study_id=study_id,
                slices_processed=int(segmentation.shape[0]) if segmentation.size else 0,
                sequences_used=tuple(sequences_used),
                sequences_missing=tuple(sequences_missing),
                device=str(self.device), inference_ms=inference_ms,
                preprocessing_ms=preprocessing_ms,
                input_size=tuple(self.config.model.input_size),
                spacing_mm=tuple(float(v) for v in spacing_mm) if spacing_mm else None,
                foundation_version=foundation_quality.get("foundation_version")),
            quality=QualityMetadata(
                predicted_score=predicted_quality,
                predicted_per_slice=tuple(float(q) for q in qualities),
                foundation_score=(float(foundation_score)
                                  if foundation_score is not None else None),
                foundation_verdict=foundation_quality.get("verdict"),
                foundation_warnings=tuple(foundation_quality.get("warnings", ())),
                review_recommended=review,
                predicted_score_reliable=quality_reliable,
                severity_correlation=severity_correlation,
                notes=self._quality_notes(quality_reliable, severity_correlation)),
            model_version=f"{self.meta.run_name or 'brain_vision'}@epoch{self.meta.epoch}",
            brain_vision_version=BRAIN_VISION_VERSION,
            caveats=tuple(self.meta.caveats),
            class_probabilities=(np.concatenate(probabilities) if probabilities
                                 else None),
        )

    # ------------------------------------------------------------------ #
    def _quality_severity_correlation(self) -> float | None:
        """How well this checkpoint's quality head tracked known degradation severity.

        Read from the metrics the checkpoint recorded at its best epoch, not from a
        constant in this file, so the flag it drives updates itself when a checkpoint
        earns it.
        """
        diagnostics = ((self.meta.metrics.get("quality") or {})
                       .get("diagnostics") or {})
        value = diagnostics.get("severity_correlation")
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _quality_notes(reliable: bool | None,
                       correlation: float | None) -> tuple[str, ...]:
        notes = [
            "the predicted score comes from this network's quality head, trained "
            "against synthetic degradations; the foundation score is measured from the "
            "volume by the MRI Foundation Layer. They are different quantities and are "
            "not merged.",
        ]
        if reliable is False:
            notes.append(
                "DO NOT USE the predicted score from this checkpoint. Its quality head "
                f"tracks known degradation severity at r={correlation:+.3f}, against a "
                f"validity threshold of {QUALITY_VALIDITY_THRESHOLD:+.2f} — it is very "
                "nearly a constant predictor. Root cause: the dataset z-scores each "
                "slice over brain voxels *after* the artefact is applied, which removes "
                "most of the intensity-statistic evidence the head would need. Use "
                "foundation_score instead.")
        elif reliable is None:
            notes.append(
                "the quality head's reliability is unknown: this checkpoint carries no "
                "recorded severity correlation. Treat the predicted score as "
                "unvalidated.")
        return tuple(notes)

    def _assemble(self, study: Any) -> tuple[np.ndarray, list[str], list[str],
                                             tuple[float, float, float] | None]:
        """Stack a foundation study's series into the network's channel order."""
        reference = None
        used: list[str] = []
        missing: list[str] = []
        planes: list[np.ndarray | None] = []

        for spec in self.network.modalities:
            series = study.first(spec.sequence) if hasattr(study, "first") else None
            if series is None:
                missing.append(spec.key)
                planes.append(None)
                continue
            array = np.asarray(series.volume.array, dtype=np.float32)
            if reference is None:
                reference = array.shape
                spacing = series.spacing
            if array.shape != reference:
                # Series that are not on one grid cannot be stacked into channels. The
                # foundation layer standardises spacing but not extent, so this is a
                # real possibility and silently resampling here would be preprocessing
                # done in the wrong place.
                log.warning("sequence dropped: it is not on the study's common grid",
                            extra={"context": {"sequence": spec.key,
                                               "shape": list(array.shape),
                                               "expected": list(reference)}})
                missing.append(spec.key)
                planes.append(None)
                continue
            used.append(spec.key)
            planes.append(array)

        if reference is None:
            raise ValueError(
                "the study holds none of the sequences this model consumes: "
                f"{[s.key for s in self.network.modalities]}")
        volumes = np.stack([p if p is not None else np.zeros(reference, np.float32)
                            for p in planes], axis=0)
        return volumes, used, missing, tuple(float(v) for v in spacing)


def _model_config_from_checkpoint(path: Path, fallback: ModelConfig,
                                  device: str) -> ModelConfig:
    """Rebuild the model configuration the checkpoint was trained with."""
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = (payload.get("meta") or {}).get("architecture") or {}
    if not architecture:
        log.warning("the checkpoint carries no architecture record; using the "
                    "configured architecture", extra={"context": {"path": str(path)}})
        return fallback

    modalities = tuple(
        ModalitySpec(key=m["key"], sequence=_sequence(m["sequence"]),
                     label=m.get("label", m["key"]),
                     required=bool(m.get("required", True)),
                     provenance=m.get("provenance", ""))
        for m in architecture.get("modalities", []))
    stored = (payload.get("meta") or {}).get("config", {}).get("model", {})
    return ModelConfig(
        encoder=architecture.get("encoder", fallback.encoder),
        decoder=architecture.get("decoder", fallback.decoder),
        stage_channels=tuple(architecture.get("stage_channels",
                                              fallback.stage_channels)),
        blocks_per_stage=tuple(stored.get("blocks_per_stage",
                                          fallback.blocks_per_stage)),
        norm=stored.get("norm", fallback.norm),
        activation=stored.get("activation", fallback.activation),
        dropout=float(stored.get("dropout", fallback.dropout)),
        deep_supervision_levels=int(architecture.get("deep_supervision_levels",
                                                     fallback.deep_supervision_levels)),
        embedding_dim=int(architecture.get("embedding", {}).get(
            "dimension", fallback.embedding_dim)),
        embedding_hidden=int(stored.get("embedding_hidden", fallback.embedding_hidden)),
        heads=fallback.heads,
        modalities=modalities or fallback.modalities,
        input_size=tuple(architecture.get("input_size", fallback.input_size)),
    )


def _sequence(value: str):
    from backend.foundation.mri.types import SequenceType

    try:
        return SequenceType(value)
    except ValueError:                                   # pragma: no cover - defensive
        return SequenceType.UNKNOWN


def _foundation_quality(study: Any) -> dict[str, Any]:
    """Pull the foundation layer's own quality verdict off the study, if it has one."""
    primary = getattr(study, "primary", None)
    if primary is None:
        return {}
    return {
        "score": float(primary.quality.quality_score),
        "verdict": study.verdict.value if hasattr(study, "verdict") else None,
        "warnings": tuple(primary.quality.warnings),
        "foundation_version": getattr(study, "foundation_version", None),
    }


__all__ = ["BrainVisionEngine"]
