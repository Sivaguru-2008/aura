"""Which slice the network sees next, and why.

Three mechanisms, composed into one sampler. Each exists because uniform sampling over
cached slices wastes most of an epoch, and each is separately switchable so its
contribution can be ablated rather than assumed.

**Region focus.** 43% of cached slices carry a tumour label (measured over a 3 000-slice
sample). Uniform draws spend the majority of every epoch on slices whose correct answer
is "nothing here". ``tumor_fraction`` fixes the share of each epoch drawn from the
positive pool. It is deliberately not 1.0: the presence head needs negatives, and a
network that has only ever seen tumour-bearing anatomy will find a tumour in a normal
brain.

**Curriculum.** Early epochs draw only from slices whose tumour is large enough to be
unmissable, and the floor drops each stage until the full distribution is restored. The
thresholds are percentiles of the measured area distribution, not round numbers.

**Hard-example mining.** After every validation the sampler re-weights towards slices
the model is currently segmenting badly. The difficulty signal comes from *training*
observations, not from validation, and that is not a shortcut — it is the only thing
that can work. Validation samples are held-out subjects; they say nothing about which
*training* slice is hard, and computing a fresh difficulty pass over the training split
would cost a second forward pass over 45 000 slices per epoch. The trainer already
computes a per-sample Dice for every sample it touches, so the signal is free and
arrives continuously instead of once per cycle.

Two safeguards on that, both structural:

* Difficulty is an exponential moving average, so one unlucky batch does not brand a
  sample as hard for the rest of training.
* Weights are clamped to ``[min_weight, max_weight]``. The floor keeps solved samples
  in the distribution — a model that stops seeing them forgets them — and the ceiling
  stops a handful of mislabelled slices from owning the epoch, which is the classic way
  hard-example mining fails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from aura.backend.core.shared.logging import get_logger
from .config import CurriculumConfig, SamplingConfig
from .types import CurriculumStage

log = get_logger("vision.brain.sampling")


@dataclass
class SliceTable:
    """The per-slice index, as parallel arrays, with the queries training needs.

    Deliberately not a list of objects: the sampler recomputes a probability vector over
    every training slice at each epoch boundary, and doing that over 45 000 Python
    objects is measurably slower than three numpy operations.
    """

    subject_index: np.ndarray
    cache_z: np.ndarray
    source_slice: np.ndarray
    brain_voxels: np.ndarray
    area_ncr_net: np.ndarray
    area_edema: np.ndarray
    area_enhancing: np.ndarray
    quality_score: np.ndarray
    #: Split of each subject, aligned with ``subject_index``.
    subject_split: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U8"))
    #: Grade of each subject, aligned with ``subject_index``.
    subject_grade: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U8"))

    def __len__(self) -> int:
        return int(self.cache_z.size)

    # -- derived quantities --------------------------------------------------- #
    @property
    def area_total(self) -> np.ndarray:
        """Whole-tumour area in pixels — the union of the three foreground classes."""
        return self.area_ncr_net + self.area_edema + self.area_enhancing

    @property
    def area_core(self) -> np.ndarray:
        """Tumour-core area: necrotic/non-enhancing plus enhancing."""
        return self.area_ncr_net + self.area_enhancing

    @property
    def positive(self) -> np.ndarray:
        return self.area_total > 0

    def split_indices(self, split: str) -> np.ndarray:
        """Indices of every slice belonging to subjects in ``split``."""
        if self.subject_split.size == 0:
            return np.arange(len(self))
        return np.flatnonzero(self.subject_split[self.subject_index] == split)

    def describe(self) -> dict[str, Any]:
        area = self.area_total
        positive = area[area > 0]
        return {
            "slices": len(self),
            "subjects": int(np.unique(self.subject_index).size),
            "positive_slices": int(self.positive.sum()),
            "positive_fraction": round(float(self.positive.mean()), 4) if len(self) else 0.0,
            "area_percentiles_positive": (
                {f"p{q}": float(np.percentile(positive, q)) for q in (5, 25, 50, 75, 95)}
                if positive.size else {}),
        }


@dataclass
class DifficultyTracker:
    """Per-sample difficulty, maintained online from the trainer's own observations.

    ``difficulty`` is ``1 - dice`` in [0, 1], smoothed. ``seen`` counts observations so
    an unobserved sample can be told apart from an easy one — a distinction that
    matters at the start of training, when every difficulty is still its prior.
    """

    size: int
    ema: float = 0.30
    #: Prior difficulty for a sample nothing is known about. 0.5 rather than 1.0: a
    #: never-seen sample should be sampled a little more often than a solved one, not
    #: treated as the hardest thing in the corpus.
    prior: float = 0.5

    def __post_init__(self) -> None:
        self.difficulty = np.full(self.size, self.prior, dtype=np.float32)
        self.seen = np.zeros(self.size, dtype=np.int32)

    def update(self, indices: Sequence[int] | np.ndarray,
               dice: Sequence[float] | np.ndarray) -> None:
        """Fold a batch's observed per-sample Dice into the running difficulty."""
        indices = np.asarray(indices, dtype=np.int64)
        observed = np.clip(1.0 - np.asarray(dice, dtype=np.float32), 0.0, 1.0)
        if indices.size == 0:
            return
        current = self.difficulty[indices]
        fresh = self.seen[indices] == 0
        # First observation replaces the prior outright; later ones blend. Otherwise a
        # sample needs a dozen visits before its difficulty stops being mostly prior.
        blended = np.where(fresh, observed,
                           (1.0 - self.ema) * current + self.ema * observed)
        self.difficulty[indices] = blended
        self.seen[indices] += 1

    def snapshot(self) -> dict[str, Any]:
        observed = self.difficulty[self.seen > 0]
        return {
            "samples": int(self.size),
            "observed": int((self.seen > 0).sum()),
            "mean_difficulty": (round(float(observed.mean()), 4)
                                if observed.size else None),
            "p90_difficulty": (round(float(np.percentile(observed, 90)), 4)
                               if observed.size else None),
            "hardest_fraction_above_0.7": (round(float((observed > 0.7).mean()), 4)
                                           if observed.size else None),
        }


class CurriculumSchedule:
    """Maps an epoch to a stage, and a stage to the slices it admits."""

    def __init__(self, config: CurriculumConfig, table: SliceTable,
                 pool: np.ndarray) -> None:
        self.config = config
        self._table = table
        self._pool = np.asarray(pool, dtype=np.int64)

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        return self.config.stage_for_epoch(epoch)

    def eligible(self, stage: CurriculumStage) -> tuple[np.ndarray, np.ndarray]:
        """Positive and negative pools admitted at ``stage``.

        Returns ``(positive_indices, negative_indices)`` as indices into the full slice
        table. The negative pool is never empty before the final stage — see the
        ``stage_negative_fraction`` comment in the configuration.
        """
        area = self._table.area_total[self._pool]
        floor = int(self.config.stage_min_area.get(stage, 0))
        positive = self._pool[area >= max(floor, 1)]
        negative = self._pool[area == 0]
        if positive.size == 0:
            # A stage whose floor admits nothing would silently train on negatives
            # alone. Fall back to every positive slice and say so.
            log.warning("curriculum stage admits no positive slice; falling back",
                        extra={"context": {"stage": stage.value, "min_area": floor}})
            positive = self._pool[area > 0]
        return positive, negative

    def negative_fraction(self, stage: CurriculumStage) -> float:
        return float(self.config.stage_negative_fraction.get(stage, 0.3))


class AdaptiveSliceSampler:
    """The training sampler: curriculum, region focus, and hard-example mining.

    Implements the ``torch.utils.data.Sampler`` interface — ``__iter__`` and
    ``__len__`` — without importing torch, so the sampling policy can be tested, and its
    distribution asserted, with numpy alone.
    """

    def __init__(self, table: SliceTable, pool: np.ndarray, *,
                 sampling: SamplingConfig, curriculum: CurriculumConfig,
                 seed: int = 7) -> None:
        self.table = table
        self.pool = np.asarray(pool, dtype=np.int64)
        self.sampling = sampling
        self.schedule = CurriculumSchedule(curriculum, table, self.pool)
        self.difficulty = DifficultyTracker(len(table), ema=sampling.difficulty_ema)
        self._rng = np.random.default_rng(seed)
        self._epoch = 0
        self._stage = curriculum.stage_for_epoch(0)
        self._plan: dict[str, Any] = {}
        self.set_epoch(0)

    # ------------------------------------------------------------------ #
    @property
    def stage(self) -> CurriculumStage:
        return self._stage

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        return int(self.sampling.samples_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        """Recompute the epoch's pools. Called by the trainer at each epoch boundary."""
        self._epoch = int(epoch)
        self._stage = self.schedule.stage_for_epoch(self._epoch)
        self._positive, self._negative = self.schedule.eligible(self._stage)
        log.info("sampler epoch configured", extra={"context": {
            "epoch": self._epoch, "stage": self._stage.value,
            "positive_pool": int(self._positive.size),
            "negative_pool": int(self._negative.size),
            "hard_mining": self.sampling.hard_mining}})

    def refresh_difficulty(self) -> dict[str, Any]:
        """Re-derive sampling weights from observed difficulty. Called after validation.

        Returns the difficulty snapshot so the trainer can record it in the run history
        — hard-example mining that cannot be inspected is indistinguishable from hard-
        example mining that is not running.
        """
        snapshot = self.difficulty.snapshot()
        log.info("hard-example weights refreshed",
                 extra={"context": {"epoch": self._epoch, **snapshot}})
        return snapshot

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[int]:
        total = int(self.sampling.samples_per_epoch)
        if not self.sampling.enabled:
            yield from self._rng.choice(self.pool, size=total, replace=True).tolist()
            return

        negative_fraction = self.schedule.negative_fraction(self._stage)
        tumor_fraction = min(self.sampling.tumor_fraction, 1.0 - negative_fraction) \
            if self._negative.size else 1.0
        n_positive = int(round(total * tumor_fraction))
        n_negative = total - n_positive
        if self._negative.size == 0:
            n_positive, n_negative = total, 0
        if self._positive.size == 0:
            n_positive, n_negative = 0, total

        draws: list[np.ndarray] = []
        if n_positive:
            draws.append(self._draw(self._positive, n_positive))
        if n_negative:
            draws.append(self._draw(self._negative, n_negative))
        order = np.concatenate(draws) if draws else np.zeros(0, dtype=np.int64)
        self._rng.shuffle(order)
        self._plan = {"stage": self._stage.value, "positive": n_positive,
                      "negative": n_negative,
                      "tumor_fraction_effective": round(n_positive / max(total, 1), 4)}
        yield from order.tolist()

    def _draw(self, candidates: np.ndarray, count: int) -> np.ndarray:
        """Sample ``count`` indices from ``candidates`` with hard-mined weights."""
        if not self.sampling.hard_mining:
            return self._rng.choice(candidates, size=count, replace=True)
        weights = self._weights(candidates)
        return self._rng.choice(candidates, size=count, replace=True, p=weights)

    def _weights(self, candidates: np.ndarray) -> np.ndarray:
        """Normalised sampling probabilities over ``candidates``.

        A convex blend of a flat prior and the difficulty signal, so
        ``hard_mining_strength`` moves continuously between "ignore difficulty" and
        "difficulty decides", and the clamp is applied to the *ratio* to the mean rather
        than to the raw difficulty — which is what makes ``max_weight`` mean "at most
        6x the average sample" rather than an arbitrary scale.
        """
        difficulty = self.difficulty.difficulty[candidates].astype(np.float64)
        shaped = np.power(np.clip(difficulty, 1e-6, 1.0),
                          self.sampling.hard_mining_power)
        mean = float(shaped.mean()) or 1.0
        ratio = np.clip(shaped / mean, self.sampling.min_weight,
                        self.sampling.max_weight)
        strength = float(np.clip(self.sampling.hard_mining_strength, 0.0, 1.0))
        blended = (1.0 - strength) + strength * ratio
        total = float(blended.sum())
        if not np.isfinite(total) or total <= 0:         # pragma: no cover - defensive
            return np.full(candidates.size, 1.0 / candidates.size)
        return blended / total

    # ------------------------------------------------------------------ #
    def plan(self) -> dict[str, Any]:
        """What the last epoch's draw actually consisted of."""
        return dict(self._plan)

    def state_dict(self) -> dict[str, Any]:
        """Difficulty state, so a resumed run does not restart hard mining from zero."""
        return {"difficulty": self.difficulty.difficulty.copy(),
                "seen": self.difficulty.seen.copy(),
                "epoch": self._epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        difficulty = np.asarray(state.get("difficulty", []), dtype=np.float32)
        if difficulty.size == self.difficulty.size:
            self.difficulty.difficulty = difficulty
            self.difficulty.seen = np.asarray(state.get("seen", []), dtype=np.int32)
        else:
            log.warning("difficulty state does not match the current slice table; "
                        "hard-example mining restarts from its prior",
                        extra={"context": {"saved": int(difficulty.size),
                                           "expected": int(self.difficulty.size)}})
        self.set_epoch(int(state.get("epoch", 0)))


__all__ = ["AdaptiveSliceSampler", "CurriculumSchedule", "DifficultyTracker",
           "SliceTable"]
