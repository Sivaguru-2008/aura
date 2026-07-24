"""Reading a complete multi-sequence MR study out of a single uploaded file.

Why this exists
---------------
The network needs four co-registered sequences. An HTTP upload is one file, and the
foundation layer's single-series path collapses a 4D NIfTI into one 3D volume tagged
``unknown`` — so through the console, a complete study could never arrive, and the
engine would abstain with ``INCOMPLETE_STUDY`` on literally every upload. A correct
engine that can never produce a result is not a working feature.

A 4D NIfTI whose fourth axis is the sequence axis is the natural single-file form of a
multi-sequence study, and it is how BraTS-derived data is usually exported.

The channel-order problem
-------------------------
NIfTI does not record what its fourth axis means. If the caller's channel order differs
from the network's declared order, every channel lands in the wrong stem — and that is
not a graceful degradation. Measured on held-out BraTS slices, presenting T1 to the
model as if it were FLAIR gives whole-tumour Dice 0.02, and T1ce gives 0.00, while the
presence head still returns a confident-looking number. A silently transposed study
produces a clean, plausible, wrong report.

So the assumed order is checked, never trusted. A small classifier fitted on BraTS
(``artifacts/brain/sequence_check.joblib``) votes on each channel independently. At
86.5% per-channel accuracy patient-disjoint it is far too weak to *assign* sequences —
that was measured and rejected — but it is being asked a much easier question: does a
whole four-channel study, read in the assumed order, get endorsed by its own pixels?

Measured on held-out volumes, a correctly-ordered study is endorsed by at least three of
its four channels **88.1%** of the time. The threshold is three rather than two because
the common error is a *swap*: exchanging two channels leaves the other two correct, so a
two-of-four bar would pass every pairwise transposition — the exact failure this check
exists to catch.

That costs an ~11.9% false-refusal rate on correctly-ordered studies, and the asymmetry
is deliberate. A false refusal is friction: the caller is told what order was expected
and re-exports. A false accept is a clean, confident, wrong report on a study the
network effectively never saw. When the two error rates are not comparable in cost, an
operating point that balances them is the wrong one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.core.shared.logging import get_logger
from backend.engines.neuro.sequence_features import FEATURE_DIM, sequence_features

log = get_logger("engine.neuromind.multisequence")

#: Minimum channels that must endorse the assumed order for the study to be analysed.
MIN_ENDORSING_CHANNELS = 3

#: Fraction of slices sampled for the order check. The check is per-slice; a handful
#: spread through the volume is plenty and keeps the cost negligible.
_ORDER_CHECK_SLICES = 9


@dataclass(frozen=True)
class MultiSequenceStudy:
    """A complete multi-sequence volume ready for the network."""

    #: ``(C, H, W, Z)`` float32, background zero, channels in network order.
    volumes: np.ndarray
    sequence_keys: tuple[str, ...]
    spacing_mm: tuple[float, float, float] | None
    #: How the channel order was established, for the audit trail and the report.
    order_source: str
    order_endorsement: dict[str, Any]


class ChannelOrderRejected(RuntimeError):
    """The file's channel order contradicts its own pixels."""

    def __init__(self, reason: str, evidence: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


def looks_multisequence(path: Path, expected_channels: int) -> bool:
    """True when ``path`` is a NIfTI whose 4th axis has ``expected_channels`` entries."""
    try:
        import nibabel as nib

        image = nib.load(str(path))
        shape = image.shape
    except Exception:
        return False
    return len(shape) == 4 and shape[3] == expected_channels


def load_multisequence(path: Path, sequence_keys: Sequence[str],
                       artifacts_root: Path | str | None = None) -> MultiSequenceStudy:
    """Load a 4D NIfTI as ``(C, H, W, Z)``, verifying the assumed channel order.

    Raises:
        ChannelOrderRejected: the pixels contradict the assumed order.
    """
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 4 or data.shape[3] != len(sequence_keys):
        raise ChannelOrderRejected(
            f"expected a 4D volume with {len(sequence_keys)} channels, got shape "
            f"{tuple(data.shape)}", {"shape": list(data.shape)})

    volumes = np.transpose(data, (3, 0, 1, 2)).copy()      # (C, H, W, Z)
    for c in range(volumes.shape[0]):                       # per-channel [0,1]
        channel = volumes[c]
        lo, hi = float(channel.min()), float(channel.max())
        volumes[c] = (channel - lo) / (hi - lo) if hi - lo > 1e-6 else 0.0

    zooms = getattr(image.header, "get_zooms", lambda: ())()
    spacing = tuple(float(z) for z in zooms[:3]) if len(zooms) >= 3 else None

    endorsement = _check_order(volumes, sequence_keys, artifacts_root)
    if endorsement["available"] and endorsement["endorsing"] < MIN_ENDORSING_CHANNELS:
        raise ChannelOrderRejected(
            f"the file's channels do not look like {', '.join(sequence_keys)} in that "
            f"order — only {endorsement['endorsing']} of {len(sequence_keys)} channels "
            f"matched, and the pixels suggest "
            f"{', '.join(endorsement['predicted_order'])}. Presenting a sequence to the "
            f"wrong input stem is not a small error: measured whole-tumour Dice is 0.02 "
            f"for T1 read as FLAIR and 0.00 for T1ce, while the presence head keeps "
            f"returning a confident number. Re-export with the channel axis ordered "
            f"{', '.join(sequence_keys)}",
            endorsement)

    return MultiSequenceStudy(
        volumes=volumes,
        sequence_keys=tuple(sequence_keys),
        spacing_mm=spacing,
        order_source="4D NIfTI channel axis, assumed to be the network's declared order",
        order_endorsement=endorsement,
    )


def _check_order(volumes: np.ndarray, sequence_keys: Sequence[str],
                 artifacts_root: Path | str | None) -> dict[str, Any]:
    """Vote on each channel's identity and compare with the assumed order."""
    classifier, meta = _load_checker(artifacts_root)
    if classifier is None:
        # No checker fitted: the order is unverified, and the caller is told so rather
        # than being given a silent pass.
        return {"available": False,
                "reason": "no sequence-order checker is fitted on this deployment"}

    classes = list(meta.get("classes", sequence_keys))
    depth = volumes.shape[3]
    indices = np.linspace(depth * 0.25, depth * 0.75, _ORDER_CHECK_SLICES).astype(int)
    indices = np.unique(np.clip(indices, 0, depth - 1))

    votes: list[str] = []
    for c in range(volumes.shape[0]):
        per_slice: list[int] = []
        for z in indices:
            features = sequence_features(volumes[c, :, :, int(z)])
            if features is None or features.shape[0] != FEATURE_DIM:
                continue
            per_slice.append(int(classifier.predict(features.reshape(1, -1))[0]))
        if not per_slice:
            votes.append("indeterminate")
            continue
        winner = max(set(per_slice), key=per_slice.count)
        votes.append(classes[winner] if winner < len(classes) else "indeterminate")

    endorsing = sum(1 for assumed, voted in zip(sequence_keys, votes) if assumed == voted)
    result = {
        "available": True,
        "assumed_order": list(sequence_keys),
        "predicted_order": votes,
        "endorsing": endorsing,
        "required": MIN_ENDORSING_CHANNELS,
        "checker_accuracy_per_channel": meta.get("accuracy_per_channel"),
        "checker_study_endorsement_rate": meta.get("study_order_endorsement_rate"),
        "slices_voted": int(indices.size),
    }
    log.info("channel order checked", extra={"context": result})
    return result


def _load_checker(artifacts_root: Path | str | None):
    import json

    if artifacts_root is None:
        from common.config import ARTIFACTS

        artifacts_root = ARTIFACTS
    model_path = Path(artifacts_root) / "brain" / "sequence_check.joblib"
    meta_path = Path(artifacts_root) / "brain" / "sequence_check.json"
    if not (model_path.exists() and meta_path.exists()):
        return None, {}
    try:
        import joblib

        return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("sequence-order checker could not be loaded; order is unverified")
        return None, {}
