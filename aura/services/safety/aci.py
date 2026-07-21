"""Adaptive Conformal Inference (ACI) — coverage that survives covariate shift.

Why this exists
---------------
``calibration.fit_conformal`` gives a *split-conformal* threshold: fit ``q̂`` once
on a calibration set, use it forever. Its 90% coverage guarantee rests on
**exchangeability** — calibration and test data drawn from one distribution. A
real clinic violates this constantly: a new scanner, a seasonal case-mix change, a
population shift. Under such covariate shift the fixed ``q̂`` silently under- or
over-covers, and a safety system that *thinks* it covers 90% while actually
covering 70% is worse than one that knows it is uncertain.

ACI (Gibbs & Candès, 2021) fixes this by treating the conformal cut-off as an
online control variable and steering it toward the target coverage from realised
outcomes. The canonical update is written on the miscoverage *level* ``α_t`` (the
set being the ``(1−α_t)`` quantile of the score window):

    α_{t+1} = α_t + γ · ( α − 1{ Y_t ∉ Ĉ_t } ).

We parameterise instead by the nonconformity *threshold* ``q̂`` directly, where a
set keeps class ``c`` iff ``s_c = 1 − p[c] ≤ q̂`` — so a **larger** ``q̂`` gives a
**wider** set. Raising the level lowers the threshold and vice-versa, so in
threshold space the equivalent update carries the opposite sign:

    q̂_{t+1} = q̂_t + γ · ( 1{ Y_t ∉ Ĉ_t } − α ).                       (★)

Read (★) directly: on a **miss** (``err = 1``) the bracket is ``1 − α > 0`` so
``q̂`` **rises** and future sets widen until coverage recovers; on a **cover**
(``err = 0``) the bracket is ``−α`` so ``q̂`` drifts down and sets tighten. That
is exactly "recent coverage below ``1−α`` ⟹ sets widen", with the sign made
consistent with a nonconformity-threshold parameterisation. ``γ`` is the step
size: larger = faster adaptation, noisier ``q̂``.

Guarantee (holds with **no** exchangeability assumption)
--------------------------------------------------------
Summing (★) from 1..T telescopes:

    q̂_{T+1} − q̂_1 = γ Σ_t ( err_t − α ),    err_t = 1{ Y_t ∉ Ĉ_t }

Because ``q̂`` self-corrects it stays bounded in a range ``[q_lo, q_hi]`` (whenever
``q̂`` is huge, sets contain everything, ``err_t = 0`` pulls it down; when tiny,
``err_t = 1`` pulls it up), so ``|q̂_{T+1} − q̂_1| ≤ (q_hi − q_lo)``. Therefore

    | (1/T) Σ_t err_t − α |  ≤  (q_hi − q_lo) / (γ T)  ⟶ 0.

The *long-run empirical miscoverage converges to α for any sequence*, adversarial
or shifting. That is the whole point: the safety guarantee no longer depends on
the data being i.i.d.

This module is pure-numpy state + update; ``gateway.storage`` persists it to
SQLite so the adaptation runs entirely offline on the edge device as confirmed
outcomes are written.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ACIState:
    """Serializable ACI state for one nonconformity stream.

    ``qhat``      : current threshold on nonconformity score s = 1 − p[true].
    ``alpha``     : target miss-rate (1 − coverage).
    ``gamma``     : step size.
    ``t``         : number of outcomes processed.
    ``recent``    : rolling miscoverage indicators (for *localized* coverage
                    reporting; the update itself is memoryless per Gibbs–Candès).
    ``window``    : length of the localized-coverage window.
    ``qhat_lo/hi``: clamp keeping q̂ a valid nonconformity threshold in [0, 1].
    """
    qhat: float = 0.90
    alpha: float = 0.10
    gamma: float = 0.02
    t: int = 0
    window: int = 200
    qhat_lo: float = 0.0
    qhat_hi: float = 1.0
    recent: deque = field(default_factory=lambda: deque(maxlen=200))

    def to_row(self) -> dict:
        return {
            "qhat": float(self.qhat),
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "t": int(self.t),
            "window": int(self.window),
            "recent": list(self.recent),
        }

    @classmethod
    def from_row(cls, row: dict) -> "ACIState":
        st = cls(
            qhat=float(row.get("qhat", 0.90)),
            alpha=float(row.get("alpha", 0.10)),
            gamma=float(row.get("gamma", 0.02)),
            t=int(row.get("t", 0)),
            window=int(row.get("window", 200)),
        )
        st.recent = deque(row.get("recent", []), maxlen=st.window)
        return st


class AdaptiveConformalInference:
    """Online ACI updater. One instance per nonconformity stream (e.g. global, or
    per class for a Mondrian-ACI hybrid)."""

    def __init__(self, state: ACIState | None = None, coverage: float = 0.90,
                 gamma: float = 0.02, window: int = 200):
        if state is None:
            state = ACIState(qhat=1.0 - (1.0 - coverage), alpha=1.0 - coverage,
                             gamma=gamma, window=window)
            state.recent = deque(maxlen=window)
        self.state = state

    # ---- prediction set --------------------------------------------------- #
    def predict_set(self, probs: np.ndarray) -> list[int]:
        """Conformal set at the *current* q̂: keep class c if (1 − p[c]) ≤ q̂.

        Never returns empty — falls back to the arg-max so a decision always exists.
        """
        probs = np.asarray(probs, dtype=float)
        keep = [c for c in range(len(probs)) if (1.0 - probs[c]) <= self.state.qhat]
        return keep or [int(probs.argmax())]

    @staticmethod
    def nonconformity(probs: np.ndarray, true_index: int) -> float:
        """Standard APS-style score for a confirmed outcome: s = 1 − p[true]."""
        return float(1.0 - np.asarray(probs, dtype=float)[int(true_index)])

    # ---- the ACI update --------------------------------------------------- #
    def update(self, probs: np.ndarray, true_index: int) -> dict:
        """Process one confirmed outcome; return the transition telemetry.

        ``probs`` is the calibrated posterior AURA emitted for the case; the set it
        emitted is ``predict_set(probs)`` at the pre-update q̂ (so the indicator is
        computed against the set the clinician actually saw).
        """
        s = self.state
        emitted_set = self.predict_set(probs)
        covered = int(true_index) in emitted_set
        err = 0.0 if covered else 1.0

        qhat_prev = s.qhat
        # Threshold-space ACI update (★): q̂_{t+1} = q̂_t + γ (err − α).
        # miss (err=1) -> q̂ rises -> wider sets; cover (err=0) -> q̂ drifts down.
        s.qhat = float(np.clip(s.qhat + s.gamma * (err - s.alpha), s.qhat_lo, s.qhat_hi))
        s.t += 1
        s.recent.append(err)

        return {
            "t": s.t,
            "qhat_prev": round(qhat_prev, 6),
            "qhat": round(s.qhat, 6),
            "covered": bool(covered),
            "target_alpha": s.alpha,
            "localized_miscoverage": self.localized_miscoverage(),
            "localized_coverage": round(1.0 - self.localized_miscoverage(), 4),
            "set_size": len(emitted_set),
        }

    # ---- diagnostics ------------------------------------------------------ #
    def localized_miscoverage(self) -> float:
        """Empirical miss-rate over the rolling window — should track α."""
        if not self.state.recent:
            return self.state.alpha
        return float(np.mean(self.state.recent))

    @property
    def qhat(self) -> float:
        return self.state.qhat
