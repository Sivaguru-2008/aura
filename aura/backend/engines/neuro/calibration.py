"""Calibration for the brain presence head.

The head's raw sigmoid discriminates almost perfectly and is scaled badly. Measured on
2400 patient-disjoint BraTS slices with all four sequences present:

    AUROC 0.993        the ordering is right
    ECE   0.093        the numbers are not

The failure is concentrated in the middle of the range, where it matters most — the
0.7-0.8 raw bin fired at an actual rate of 0.227, and 0.6-0.7 at 0.176. The extremes
are fine (0.039 raw -> 0.000 actual; 0.993 -> 0.994), which is exactly the profile of a
head that is being asked to express uncertainty it was never trained to express.

Reporting the raw value would put "72% probability of tumour" on a console next to a
study whose real rate is 23%. A Platt scaler fitted on the logit fixes the scale without
touching the ordering, so AUROC is unchanged by construction.

The fitted coefficients live in ``artifacts/brain/presence_calibration.json``, written
by the fitting script and carrying its own provenance. When that file is absent the
engine does **not** fall back to the raw value: it abstains. A silently uncalibrated
probability is worse than no probability, because nothing downstream can tell the
difference.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aura.backend.core.shared.logging import get_logger

log = get_logger("engine.neuromind.calibration")

#: Where the fitting script writes its output, relative to the artifacts root.
CALIBRATION_FILENAME = "presence_calibration.json"


class CalibrationUnavailable(RuntimeError):
    """No fitted calibrator for this checkpoint. The caller must abstain, not guess."""


@dataclass(frozen=True)
class PresenceCalibrator:
    """Platt scaling on the presence head's logit."""

    a: float
    b: float
    metadata: dict[str, Any]

    def __call__(self, raw_probability: float) -> float:
        p = min(max(float(raw_probability), 1e-6), 1.0 - 1e-6)
        logit = math.log(p / (1.0 - p))
        return 1.0 / (1.0 + math.exp(-(self.a * logit + self.b)))

    @property
    def summary(self) -> str:
        held = self.metadata.get("held_out", {})
        return (f"Platt-calibrated presence head "
                f"(held-out AUROC {held.get('auroc_calibrated', float('nan')):.3f}, "
                f"ECE {held.get('ece_raw', float('nan')):.3f} -> "
                f"{held.get('ece_calibrated', float('nan')):.3f}, "
                f"n={self.metadata.get('n_eval_slices', '?')} patient-disjoint slices)")


def load_calibrator(artifacts_root: Path | str | None = None) -> PresenceCalibrator:
    """Load the fitted calibrator, or raise :class:`CalibrationUnavailable`."""
    if artifacts_root is None:
        from aura.common.config import ARTIFACTS

        artifacts_root = ARTIFACTS
    path = Path(artifacts_root) / "brain" / CALIBRATION_FILENAME
    if not path.exists():
        raise CalibrationUnavailable(
            f"no presence-head calibration at {path}; fit one before serving brain "
            f"studies — the raw head is overconfident by a measured ECE of 0.093")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibrator = PresenceCalibrator(
            a=float(payload["a"]), b=float(payload["b"]), metadata=payload)
    except Exception as exc:
        raise CalibrationUnavailable(
            f"presence calibration at {path} could not be read: {exc}") from exc
    log.info("presence calibrator loaded",
             extra={"context": {"path": str(path), "a": calibrator.a, "b": calibrator.b}})
    return calibrator
