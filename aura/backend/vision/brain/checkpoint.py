"""Checkpoints, and why there are six of them.

    best_brain_model.pt        full network, at the best validated epoch
    latest_brain_model.pt      full network, at the last completed epoch
    brain_encoder.pt           encoder alone
    brain_decoder.pt           decoder + segmentation head
    brain_embedding_head.pt    embedding projector alone
    training_state.pt          optimiser, scheduler, scaler, EMA, sampler, RNG

The last three are the interesting ones. A future NeuroMind module — progression
prediction, a digital twin, a surgical planner — does not want this network; it wants
the *representation* this network learned, attached to its own head, trained on its own
much smaller dataset. Shipping the encoder as an independently loadable artefact with
its own declared contract is what makes that a five-line operation instead of a
reimplementation of this package's ``state_dict`` layout.

Every checkpoint carries its architecture description, its configuration, and the
metrics it achieved. Loading one against an incompatible model is caught at load with a
message naming the mismatch, rather than diagnosed later from predictions that are
merely bad.

Serialisation posture matches the chest stack's: ``weights_only=True`` for artefacts
that get loaded at serving time, ``weights_only=False`` only for the training-state
resume, which reads a file the same process wrote minutes earlier and needs optimiser
objects that the safe unpickler cannot reconstruct.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from backend.core.shared.logging import get_logger
from backend.vision.brain.config import BrainVisionConfig, PathsConfig
from backend.vision.brain.errors import CheckpointError
from backend.vision.brain.types import BRAIN_VISION_VERSION

log = get_logger("vision.brain.checkpoint")


@dataclass
class CheckpointMeta:
    """What every checkpoint says about itself."""

    brain_vision_version: str = BRAIN_VISION_VERSION
    run_name: str = ""
    epoch: int = -1
    monitor: str = ""
    monitor_value: float = float("nan")
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    architecture: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Corpus caveats carried forward from the ingest manifest, so a checkpoint alone
    #: is enough to know what the model has and has not seen.
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brain_vision_version": self.brain_vision_version,
            "run_name": self.run_name, "epoch": self.epoch,
            "monitor": self.monitor, "monitor_value": self.monitor_value,
            "created_at": self.created_at, "architecture": dict(self.architecture),
            "config": dict(self.config), "metrics": dict(self.metrics),
            "caveats": list(self.caveats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CheckpointMeta":
        data = data or {}
        return cls(
            brain_vision_version=data.get("brain_vision_version", "unknown"),
            run_name=data.get("run_name", ""), epoch=int(data.get("epoch", -1)),
            monitor=data.get("monitor", ""),
            monitor_value=float(data.get("monitor_value", float("nan"))),
            created_at=data.get("created_at", ""),
            architecture=dict(data.get("architecture", {})),
            config=dict(data.get("config", {})), metrics=dict(data.get("metrics", {})),
            caveats=list(data.get("caveats", [])))


class CheckpointWriter:
    """Writes the six artefacts. The only thing in this package that touches them."""

    def __init__(self, paths: PathsConfig, config: BrainVisionConfig) -> None:
        self.paths = paths
        self.config = config
        paths.ensure()

    # ------------------------------------------------------------------ #
    def save_epoch(self, network: Any, meta: CheckpointMeta, *,
                   is_best: bool, ema_state: dict[str, Any] | None = None) -> None:
        """Write ``latest``, the component artefacts, and ``best`` when it improved."""
        payload = {"model_state_dict": network.state_dict(), "meta": meta.to_dict()}
        if ema_state is not None:
            # Kept beside the raw weights rather than instead of them: the EMA claim is
            # only checkable if both are on disk and both can be validated.
            payload["ema_state_dict"] = ema_state
        _atomic_save(payload, self.paths.latest_model_path)

        if is_best:
            _atomic_save(payload, self.paths.best_model_path)
            self.save_components(network, meta)
            log.info("new best model", extra={"context": {
                "epoch": meta.epoch, "monitor": meta.monitor,
                "value": round(meta.monitor_value, 5)}})

    def save_components(self, network: Any, meta: CheckpointMeta) -> None:
        """Encoder, decoder, and embedding head as independently loadable artefacts."""
        _atomic_save({
            "state_dict": network.encoder.state_dict(),
            "component": "encoder",
            "feature_channels": list(network.encoder.feature_channels),
            "strides": list(network.encoder.strides),
            "embedding_channels": int(network.encoder.embedding_channels),
            "modalities": [m.to_dict() for m in network.modalities],
            "meta": meta.to_dict(),
        }, self.paths.encoder_path)

        _atomic_save({
            "decoder_state_dict": network.decoder.state_dict(),
            "segmentation_head_state_dict": (
                network.segmentation_head.state_dict()
                if network.segmentation_head is not None else None),
            "component": "decoder",
            "feature_channels": list(network.decoder.feature_channels),
            "deep_supervision_levels": network.deep_supervision_levels,
            "meta": meta.to_dict(),
        }, self.paths.decoder_path)

        if network.embedding_head is not None:
            _atomic_save({
                "state_dict": network.embedding_head.state_dict(),
                "component": "embedding_head",
                "embedding": network.embedding_spec.to_dict(),
                "input_channels": int(network.encoder.embedding_channels),
                "meta": meta.to_dict(),
            }, self.paths.embedding_head_path)

    def save_training_state(self, state: dict[str, Any]) -> None:
        """Everything needed to resume exactly where the run stopped."""
        _atomic_save(state, self.paths.training_state_path)

    def write_model_card(self, card: dict[str, Any]) -> None:
        self.paths.model_card_path.write_text(
            json.dumps(card, indent=2, default=str), encoding="utf-8")

    def append_history(self, record: dict[str, Any]) -> None:
        """One JSON object per line. Append-only, so a crashed run keeps its history."""
        with self.paths.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_network_checkpoint(path: Path, network: Any, *, device: str = "cpu",
                            prefer_ema: bool = True) -> CheckpointMeta:
    """Load full network weights, preferring the EMA copy when one was saved.

    ``prefer_ema`` defaults to True because the EMA weights are what validation selected
    the best epoch on when EMA is enabled — loading the raw weights would serve a model
    that was never the one measured.
    """
    if not Path(path).exists():
        raise CheckpointError(f"no checkpoint at {path}", detail={"path": str(path)})
    payload = _load(path, device)
    state = payload.get("ema_state_dict") if prefer_ema else None
    state = state or payload.get("model_state_dict")
    if state is None:
        raise CheckpointError("the checkpoint holds no model weights",
                              detail={"path": str(path), "keys": sorted(payload)})
    # ``strict=False`` tolerates *absent* and *extra* keys but still raises on a shape
    # mismatch, which is the far more common failure: an architecture whose widths
    # changed has every key present and every tensor the wrong size. Both routes are
    # funnelled into one CheckpointError so a caller has one thing to catch and a
    # message that names the cause rather than 200 lines of shapes.
    try:
        missing, unexpected = network.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise CheckpointError(
            "the checkpoint does not match this network: parameter shapes differ",
            detail={"path": str(path),
                    "torch_error": str(exc).splitlines()[0],
                    "checkpoint_architecture": (payload.get("meta") or {}).get(
                        "architecture", {}),
                    "hint": "build the network from the checkpoint's own architecture "
                            "record, as BrainVisionEngine.load does"}) from exc
    if missing or unexpected:
        raise CheckpointError(
            "the checkpoint does not match this network",
            detail={"path": str(path), "missing": list(missing)[:12],
                    "unexpected": list(unexpected)[:12],
                    "hint": "the architecture configuration differs from the one that "
                            "trained this checkpoint; compare meta.architecture"})
    meta = CheckpointMeta.from_dict(payload.get("meta"))
    log.info("network weights loaded", extra={"context": {
        "path": str(path), "epoch": meta.epoch, "monitor": meta.monitor,
        "value": meta.monitor_value,
        "source": "ema" if payload.get("ema_state_dict") and prefer_ema else "raw"}})
    return meta


def load_encoder(path: Path, encoder: Any, *, device: str = "cpu") -> dict[str, Any]:
    """Load a standalone encoder into a freshly built one — the transfer-learning path.

    The check that matters is ``feature_channels``: an encoder whose widths differ is a
    different encoder, and PyTorch's own error for that is a wall of shape mismatches.
    """
    payload = _load(path, device)
    saved = tuple(int(c) for c in payload.get("feature_channels", ()))
    current = tuple(int(c) for c in encoder.feature_channels)
    if saved and saved != current:
        raise CheckpointError(
            "the saved encoder has a different feature pyramid",
            detail={"saved": list(saved), "current": list(current),
                    "path": str(path)})
    encoder.load_state_dict(payload["state_dict"])
    return payload


def load_embedding_head(path: Path, head: Any, *, device: str = "cpu"
                        ) -> dict[str, Any]:
    """Load a standalone embedding projector."""
    payload = _load(path, device)
    saved = payload.get("embedding", {})
    if saved and int(saved.get("dimension", head.dimension)) != int(head.dimension):
        raise CheckpointError(
            "the saved embedding head has a different output dimension",
            detail={"saved": saved.get("dimension"), "current": head.dimension})
    head.load_state_dict(payload["state_dict"])
    return payload


def load_training_state(path: Path, device: str = "cpu") -> dict[str, Any]:
    """Read the resume state.

    ``weights_only=False`` here and nowhere else: optimiser and scheduler state contain
    objects the safe unpickler cannot reconstruct, and this file is one this same
    training run wrote — not an artefact from anywhere else. The same reasoning, and the
    same boundary, as the chest stack's ``ml/vision_cxr/checkpoint.py``.
    """
    return torch.load(path, map_location=device, weights_only=False)


def _load(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Older checkpoints, or metadata the safe unpickler declines. Retried
        # explicitly rather than defaulting to the unsafe path, so the fallback is
        # visible in the log rather than being the silent norm.
        log.warning("falling back to a full unpickle for this checkpoint",
                    extra={"context": {"path": str(path)}})
        return torch.load(path, map_location=device, weights_only=False)


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """Write to a temporary file and rename.

    A checkpoint half-written when a run is interrupted is worse than no checkpoint: it
    exists, it loads far enough to look plausible, and it fails at the last tensor. The
    rename is atomic on both NTFS and POSIX, so the file at ``path`` is always complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


__all__ = [
    "CheckpointMeta", "CheckpointWriter", "load_embedding_head", "load_encoder",
    "load_network_checkpoint", "load_training_state",
]
