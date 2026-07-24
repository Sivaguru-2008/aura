"""Quantum Measurement-Budgeted Abstention — spend shots until the decision resolves.

The idea
--------
On quantum hardware you do not get an expectation value; you get an *estimate* of
one, and its precision is bought with measurements:

    Var[<Z_i>] = (1 - <Z_i>^2) / n_shots

Every classical model treats inference cost as fixed. A quantum model does not: the
same circuit, run longer, returns a sharper posterior. That makes "how many
measurements is this patient worth?" a real question with a real answer, and this
module answers it sequentially.

Start with a small budget. Estimate the posterior and the shot-noise spread of the
**decision margin** ``p_top1 - p_top2``. If the margin is separated from zero at the
required confidence, commit. If not, double the budget and measure again. If the
ceiling is reached without separation, abstain.

Why this is worth having
------------------------
It separates two kinds of uncertainty that every other part of the stack conflates,
and it does so with quantum resources as the meter:

* **Measurement-limited** — the analytic margin is comfortably non-zero, and the run
  simply had not bought enough precision yet. More shots would fix it, and
  :attr:`BudgetDecision.predicted_shots` says how many.
* **Model-limited** — the lead is below the clinical-significance floor, or would
  need more shots than any device could deliver. The two diagnoses are not separable
  by a meaningful amount *under this model*, so abstention is the correct clinical
  action and more measurement is wasted effort.

That distinction is actionable in a way a single confidence number is not. "Run the
circuit longer" and "this case needs a human" are different instructions, and the
system can now tell them apart rather than emitting one number that means either.

What this is honestly not
-------------------------
Shot noise is **aleatoric readout noise**, not model error. As ``n_shots -> inf`` the
estimate converges to the analytic expectation the serving path already uses; it does
not converge to the truth. Spending more measurements makes a confident wrong answer
*more* confident and *still* wrong. So QMBA is a resolution budget, not a correctness
guarantee, and it composes with — never replaces — the conformal and OOD abstention
in :mod:`services.safety`. A study that QMBA commits can still be abstained on by the
safety engine, and that ordering is deliberate.

The analytic path stays the default for serving. QMBA is what makes the *cost* of a
decision visible, and what a real device would run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.mathx import softmax
from schemas.clinical import DIAGNOSES

#: Shot budget the first stage spends. Small on purpose: most studies separate here,
#: and the whole point is not to pay for precision nobody needed.
DEFAULT_INITIAL_SHOTS = 128

#: Hard ceiling for one study. 8192 is the per-circuit shot limit on most current
#: superconducting devices, so a run that exceeds it is not something today's
#: hardware would deliver in one job.
DEFAULT_MAX_SHOTS = 8192

#: Required separation of the margin from zero, in units of its own shot-noise
#: standard deviation. 2.0 is a ~95% one-sided normal bound; the margin is a
#: difference of Monte-Carlo-propagated softmax outputs, so this is a working
#: threshold rather than an exact test, and it is reported alongside the raw z.
DEFAULT_REQUIRED_Z = 2.0

#: Beyond this many shots the case is called model-limited rather than
#: measurement-limited.
#:
#: The number is set from device physics, not taste. Shallow circuits on current
#: superconducting hardware repeat at roughly 1-10 kHz, so 100k shots is ~10-100
#: seconds of pure device time for a single patient, before queueing and before the
#: other studies in the worklist. Past that, "run it longer" stops being an
#: instruction anyone can act on, and the honest description of the ambiguity is that
#: it lives in the model rather than in the readout.
#:
#: Because ``n_required`` scales as ``1/margin^2``, this ceiling is equivalent to a
#: floor on the analytic margin — it is one threshold expressed in the units a
#: clinician can act on (device time) instead of the units only a modeller reads.
PLAUSIBLE_SHOT_CEILING = 100_000

#: Monte-Carlo draws used to propagate shot noise through the softmax. 256 gives the
#: margin standard deviation to ~4% relative error, which is well inside the
#: precision the z threshold needs.
DEFAULT_MC_SAMPLES = 256

#: Smallest posterior lead that counts as a clinical decision, independent of how
#: precisely it can be measured.
#:
#: This is not a tuning knob, it is a correctness requirement, and it was added
#: because the schedule without it committed to a top-1 lead of 6e-7 — statistically
#: separated, because the shot noise happened to be smaller still, and clinically
#: meaningless. Statistical separation and clinical significance are different
#: questions and a diagnostic system has to ask both.
#:
#: The value is anchored to the served model's own calibration error rather than
#: chosen: measured ECE for these fusion backends is 0.06-0.08, so posterior
#: differences below ~0.05 are smaller than the resolution at which the model's
#: probabilities are known to be accurate. Committing on a lead finer than your own
#: calibration error is claiming precision you have not demonstrated. Production
#: deployments should set this from ``artifacts/backend_calibration.json``.
DEFAULT_MIN_MARGIN = 0.05

#: Margins below this are treated as zero when the measured spread is also zero.
#: Guards a degenerate case that is otherwise scored as infinite confidence: if the
#: readout head ignores the quantum state (a zero or near-zero ``W``), shot noise
#: cannot move the posterior at all, so ``margin_std`` is exactly 0 and ``margin /
#: margin_std`` is ``inf`` for *any* non-zero margin — including one of 1e-9. A
#: deterministic tie is a tie, not certainty.
DETERMINISTIC_MARGIN_FLOOR = 1e-9


@dataclass(frozen=True)
class BudgetStage:
    """One round of the sequential measurement schedule."""

    shots: int
    margin: float
    margin_std: float
    separation_z: float
    top: str
    runner_up: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shots": self.shots,
            "margin": round(self.margin, 6),
            "margin_std": round(self.margin_std, 6),
            "separation_z": round(self.separation_z, 4),
            "top": self.top,
            "runner_up": self.runner_up,
        }


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome of a measurement-budgeted decision."""

    committed: bool
    #: Highest-posterior diagnosis at the final budget. Populated even when the
    #: decision abstained — the caller needs to know *what* could not be confirmed.
    top: str
    runner_up: str
    posterior: dict[str, float]
    #: Shot-noise standard deviation of each posterior entry at the final budget.
    posterior_std: dict[str, float]

    shots_spent: int
    margin: float
    margin_std: float
    separation_z: float

    #: Margin at infinite shots — the analytic value the serving path uses. The
    #: reference that decides whether more measurement could ever have helped.
    analytic_margin: float
    #: Shots the first stage predicted would be needed to reach the required z.
    #: ``None`` when the first stage already separated.
    predicted_shots: int | None
    #: ``"measurement"`` (more shots would resolve it) | ``"model"`` (no achievable
    #: budget would) | ``None`` (committed). Two values, because they map to two
    #: different actions: run the circuit longer, or escalate to a human.
    limiting_factor: str | None
    #: True when the ``"model"`` verdict came from the clinical-significance floor
    #: rather than the shot ceiling. Both mean escalate; this says which test fired.
    floor_limited: bool = False
    reason: str = ""
    trajectory: tuple[BudgetStage, ...] = field(default_factory=tuple)

    @property
    def measurement_limited(self) -> bool:
        return self.limiting_factor == "measurement"

    @property
    def model_limited(self) -> bool:
        return self.limiting_factor == "model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed": self.committed,
            "top": self.top,
            "runner_up": self.runner_up,
            "posterior": {k: round(v, 6) for k, v in self.posterior.items()},
            "posterior_std": {k: round(v, 6) for k, v in self.posterior_std.items()},
            "shots_spent": self.shots_spent,
            "margin": round(self.margin, 6),
            "margin_std": round(self.margin_std, 6),
            "separation_z": round(self.separation_z, 4),
            "analytic_margin": round(self.analytic_margin, 6),
            "predicted_shots": self.predicted_shots,
            "limiting_factor": self.limiting_factor,
            "floor_limited": self.floor_limited,
            "reason": self.reason,
            "trajectory": [s.to_dict() for s in self.trajectory],
        }


class QuantumMeasurementBudget:
    """Sequential shot allocation for one fusion decision.

    Stateless between calls and deterministic given a seed, because a clinical
    decision that changes when you re-run it is not a decision. The seed enters the
    Monte-Carlo propagation only — the underlying expectations are analytic.
    """

    def __init__(
        self,
        model: Any,
        *,
        initial_shots: int = DEFAULT_INITIAL_SHOTS,
        max_shots: int = DEFAULT_MAX_SHOTS,
        required_z: float = DEFAULT_REQUIRED_Z,
        min_margin: float = DEFAULT_MIN_MARGIN,
        mc_samples: int = DEFAULT_MC_SAMPLES,
        plausible_ceiling: int = PLAUSIBLE_SHOT_CEILING,
        seed: int = 7,
    ) -> None:
        if initial_shots < 1 or max_shots < initial_shots:
            raise ValueError("require 1 <= initial_shots <= max_shots")
        self.model = model
        self.initial_shots = int(initial_shots)
        self.max_shots = int(max_shots)
        self.required_z = float(required_z)
        self.min_margin = float(min_margin)
        self.mc_samples = int(mc_samples)
        self.plausible_ceiling = int(plausible_ceiling)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    def decide(self, x: np.ndarray) -> BudgetDecision:
        """Run the sequential schedule for one evidence vector."""
        x = np.asarray(x, dtype=float).ravel()
        # One circuit evaluation. Every stage reuses these expectations and only
        # re-draws the shot noise, because the *state* does not change with the
        # budget — only how precisely we can read it.
        z_exp = self.model._expectations(x)
        analytic_posterior = softmax(self.model.W @ z_exp + self.model.b)
        order = np.argsort(analytic_posterior)[::-1]
        top_index, runner_index = int(order[0]), int(order[1])
        analytic_margin = float(analytic_posterior[top_index]
                                - analytic_posterior[runner_index])

        # The clinical floor is checked against the *analytic* margin, which is what
        # the lead converges to. If the model's own best estimate of the lead is below
        # the floor, no measurement budget raises it — spending shots would only
        # measure an insignificant difference more precisely.
        if analytic_margin < self.min_margin:
            return self._below_clinical_floor(
                x, z_exp, analytic_posterior, top_index, runner_index, analytic_margin)

        trajectory: list[BudgetStage] = []
        shots = self.initial_shots
        predicted: int | None = None
        stage: BudgetStage | None = None
        posterior = analytic_posterior
        posterior_std = np.zeros_like(analytic_posterior)

        while True:
            posterior, posterior_std, margin, margin_std = self._estimate(
                z_exp, shots, top_index, runner_index)
            separation = self._separation(margin, margin_std)
            stage = BudgetStage(
                shots=shots, margin=margin, margin_std=margin_std,
                separation_z=separation,
                top=self._label(top_index),
                runner_up=self._label(runner_index),
            )
            trajectory.append(stage)

            if separation >= self.required_z:
                break
            # Re-predicted every stage rather than kept from the first. ``z`` measured
            # at 128 shots is itself noisy, so the early extrapolation is coarse; the
            # last stage gives the caller the best available answer to the only
            # question they can act on — how many more measurements would settle this.
            predicted = self._predict_shots(shots, separation) or predicted
            if shots >= self.max_shots:
                break
            shots = min(shots * 2, self.max_shots)

        assert stage is not None
        committed = stage.separation_z >= self.required_z
        limiting = None if committed else self._limiting_factor(predicted)
        return BudgetDecision(
            committed=committed,
            top=stage.top,
            runner_up=stage.runner_up,
            posterior=self._as_dict(posterior),
            posterior_std=self._as_dict(posterior_std),
            shots_spent=sum(s.shots for s in trajectory),
            margin=stage.margin,
            margin_std=stage.margin_std,
            separation_z=stage.separation_z,
            analytic_margin=analytic_margin,
            predicted_shots=predicted,
            limiting_factor=limiting,
            reason=self._reason(committed, stage, limiting, predicted,
                                analytic_margin),
            trajectory=tuple(trajectory),
        )

    # ------------------------------------------------------------------ #
    def _below_clinical_floor(self, x: np.ndarray, z_exp: np.ndarray,
                              analytic_posterior: np.ndarray, top: int, runner: int,
                              analytic_margin: float) -> BudgetDecision:
        """Abstain without spending anything: the lead is too small to matter.

        Returned before the schedule starts, because the schedule buys *precision*
        and precision is not the problem here. Measuring an insignificant difference
        to more decimal places does not make it significant, and the shots would be
        wasted on a case that was always going to need a human.
        """
        posterior, posterior_std, margin, margin_std = self._estimate(
            z_exp, self.initial_shots, top, runner)
        stage = BudgetStage(
            shots=self.initial_shots, margin=margin, margin_std=margin_std,
            separation_z=self._separation(margin, margin_std),
            top=self._label(top), runner_up=self._label(runner))
        return BudgetDecision(
            committed=False,
            top=stage.top,
            runner_up=stage.runner_up,
            posterior=self._as_dict(posterior),
            posterior_std=self._as_dict(posterior_std),
            shots_spent=self.initial_shots,
            margin=margin,
            margin_std=margin_std,
            separation_z=stage.separation_z,
            analytic_margin=analytic_margin,
            predicted_shots=None,
            # "model", not a third category. The clinical floor and the shot ceiling
            # are two routes to the same conclusion — no achievable measurement
            # budget produces a decision — and they carry the same clinical action:
            # escalate. Splitting them in the label would imply the caller should do
            # something different, which they should not. The reason text says which
            # route it was.
            limiting_factor="model",
            floor_limited=True,
            reason=(
                f"abstained: {stage.top} leads {stage.runner_up} by "
                f"{analytic_margin:.4f} at infinite measurement precision, below the "
                f"{self.min_margin:.2f} clinical floor. The lead is smaller than the "
                f"model's own calibration error, so it is not a decision no matter "
                f"how precisely it is measured; no further shots were spent"),
            trajectory=(stage,),
        )

    @staticmethod
    def _label(index: int) -> str:
        d = DIAGNOSES[index]
        return d.value if hasattr(d, "value") else str(d)

    def _estimate(self, z_exp: np.ndarray, shots: int, top: int, runner: int
                  ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Propagate finite-shot variance through the readout to the decision margin.

        The margin's spread is taken from the *joint* Monte-Carlo draws rather than
        by adding the two entries' variances. The softmax couples every output, so
        ``p_top`` and ``p_runner`` are strongly anti-correlated; treating them as
        independent overstates the spread by roughly a factor of two and would make
        the schedule buy shots it does not need.

        ``top`` and ``runner`` are fixed from the *analytic* posterior, not re-chosen
        per draw. The question being asked is "is the analytic leader separated from
        its analytic rival", and re-ranking each draw would answer a different one.
        A consequence worth expecting: at very low budgets the sampled margin can come
        out negative, because shot noise flipped the pair. That is not a bug — it is
        the correct reading that this budget cannot even establish the ordering.
        """
        rng = np.random.default_rng(self.seed + shots)
        std_z = np.sqrt(np.clip((1.0 - z_exp ** 2) / max(shots, 1), 0.0, None))

        draws = z_exp[None, :] + rng.normal(0.0, 1.0, size=(self.mc_samples,
                                                            z_exp.size)) * std_z[None, :]
        logits = draws @ self.model.W.T + self.model.b
        probs = np.apply_along_axis(softmax, 1, logits)

        margins = probs[:, top] - probs[:, runner]
        return (probs.mean(axis=0), probs.std(axis=0),
                float(margins.mean()), float(margins.std()))

    @staticmethod
    def _separation(margin: float, margin_std: float) -> float:
        """Margin in units of its own shot noise, handling the zero-spread case.

        ``margin_std`` is exactly zero when shot noise cannot reach the posterior at
        all — a readout head that ignores the quantum state. Dividing by it and
        calling the result infinite would score a 1e-9 margin as total certainty,
        which is how a degenerate model would silently commit to everything. A
        deterministic tie is a tie.
        """
        if margin_std > 0:
            return float(margin / margin_std)
        if abs(margin) > DETERMINISTIC_MARGIN_FLOOR:
            return float("inf")          # genuinely deterministic and separated
        return 0.0                       # genuinely deterministic and tied

    def _predict_shots(self, shots: int, separation: float) -> int | None:
        """Shots needed to reach ``required_z``, extrapolated from one stage.

        ``margin_std`` scales as ``1/sqrt(n)``, so ``z`` scales as ``sqrt(n)`` and
        ``n_required = n * (z_required / z)^2``. One measured stage therefore predicts
        the whole schedule — which is the practically useful output even when the
        answer is "more than you can afford".
        """
        if separation <= 0 or not np.isfinite(separation):
            return None
        needed = shots * (self.required_z / separation) ** 2
        return int(np.ceil(needed)) if np.isfinite(needed) else None

    def _limiting_factor(self, predicted: int | None) -> str:
        """Decide whether more measurement could have helped."""
        if predicted is None or predicted > self.plausible_ceiling:
            return "model"
        return "measurement"

    @staticmethod
    def _as_dict(values: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, d in enumerate(DIAGNOSES):
            key = d.value if hasattr(d, "value") else str(d)
            out[key] = float(values[i])
        return out

    def _reason(self, committed: bool, stage: BudgetStage, limiting: str | None,
                predicted: int | None, analytic_margin: float) -> str:
        if committed:
            return (
                f"committed to {stage.top} after {stage.shots} shots: the margin over "
                f"{stage.runner_up} is {stage.margin:.3f} +/- {stage.margin_std:.3f}, "
                f"{stage.separation_z:.1f} standard deviations of shot noise from zero")
        if limiting == "model":
            return (
                f"abstained: {stage.top} and {stage.runner_up} are separated by only "
                f"{analytic_margin:.4f} at infinite measurement precision, so no "
                f"achievable shot budget resolves them. This is model ambiguity, not "
                f"readout noise — the case needs a human, not a longer run")
        return (
            f"abstained: {stage.top} leads {stage.runner_up} by {stage.margin:.3f} "
            f"+/- {stage.margin_std:.3f} after {self.max_shots} shots, which is only "
            f"{stage.separation_z:.1f} sigma. An estimated {predicted} shots would "
            f"reach {self.required_z:.0f} sigma — the decision is measurement-limited "
            f"and a longer run on the same circuit would resolve it")


def shot_sweep(model: Any, x: np.ndarray, shot_counts: tuple[int, ...] = (
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
        mc_samples: int = DEFAULT_MC_SAMPLES, seed: int = 7) -> list[dict[str, Any]]:
    """Measure the decision margin across a range of fixed shot budgets.

    The demonstration companion to :class:`QuantumMeasurementBudget`: it shows the
    posterior sharpening as measurement budget grows, which is the behaviour the
    adaptive schedule exploits. Returned as plain dicts so it can be written to an
    artifact and plotted without importing this module.
    """
    budget = QuantumMeasurementBudget(model, mc_samples=mc_samples, seed=seed)
    x = np.asarray(x, dtype=float).ravel()
    z_exp = model._expectations(x)
    analytic = softmax(model.W @ z_exp + model.b)
    order = np.argsort(analytic)[::-1]
    top, runner = int(order[0]), int(order[1])

    rows: list[dict[str, Any]] = []
    for shots in shot_counts:
        _, posterior_std, margin, margin_std = budget._estimate(z_exp, shots, top,
                                                                runner)
        rows.append({
            "shots": int(shots),
            "margin": round(margin, 6),
            "margin_std": round(margin_std, 6),
            "separation_z": round(float(margin / margin_std) if margin_std > 0
                                  else float("inf"), 4),
            "mean_posterior_std": round(float(posterior_std.mean()), 6),
        })
    return rows


__all__ = ["BudgetDecision", "BudgetStage", "QuantumMeasurementBudget", "shot_sweep",
           "DEFAULT_INITIAL_SHOTS", "DEFAULT_MAX_SHOTS", "DEFAULT_REQUIRED_Z",
           "DEFAULT_MIN_MARGIN", "PLAUSIBLE_SHOT_CEILING"]
