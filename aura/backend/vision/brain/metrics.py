"""Validation metrics. Dice is one of them, not the report.

Everything here accumulates over batches and reports at the end, and everything is
reported *per class and per composite region*, because a single averaged number over
this label space is close to uninformative: background is 97% of every slice, oedema is
three times the size of the enhancing tumour, and a model that finds oedema and misses
enhancement scores well on any average that includes them together.

Three choices worth stating.

**Composite regions are the headline.** Whole tumour, tumour core, and enhancing tumour
are what the BraTS challenge scores and what every published number refers to. Reporting
only the primary classes would produce numbers that are not comparable with anything.
Both are reported; the monitored metric is the composite mean.

**Dice on an empty class is 1.0, not 0.0, when the prediction is also empty.** Roughly
57% of slices carry no enhancing tumour. Scoring those as zero would make the enhancing
Dice a measurement of prevalence, and a model that learned to predict "enhancing
everywhere" would beat one that correctly predicts nothing. Empty-empty is a correct
answer and is scored as one; empty-truth with a non-empty prediction is scored zero.
The counts of each case are reported alongside, so the aggregate can always be
decomposed.

**Hausdorff is the 95th percentile of the surface distance, not the maximum.** The
maximum is decided by one outlier voxel and moves by tens of millimetres between epochs
for reasons that have nothing to do with the model. HD95 is the standard for the same
reason.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from backend.vision.brain.types import (
    COMPOSITE_MEMBERS,
    CompositeRegion,
    FOREGROUND_REGIONS,
    REGION_KEYS,
)


@dataclass
class BinaryCounts:
    """Voxel confusion counts for one binary region, accumulated over a split."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    #: Per-slice Dice values, kept so the report can give a distribution and not only a
    #: pooled number. Pooled and per-slice-averaged Dice differ substantially when
    #: region sizes vary by two orders of magnitude, and only saying which one you mean
    #: makes the number comparable.
    slice_dice: list[float] = field(default_factory=list)
    slice_iou: list[float] = field(default_factory=list)
    surface_distances: list[float] = field(default_factory=list)
    #: Slices where both prediction and truth were empty — a correct negative.
    empty_agreements: int = 0
    #: Slices where truth was empty and the prediction was not.
    false_alarms: int = 0
    slices: int = 0

    def update(self, predicted: np.ndarray, truth: np.ndarray) -> None:
        """Fold one batch of ``(B, H, W)`` boolean masks in."""
        for index in range(predicted.shape[0]):
            p, t = predicted[index], truth[index]
            tp = int(np.count_nonzero(p & t))
            fp = int(np.count_nonzero(p & ~t))
            fn = int(np.count_nonzero(~p & t))
            tn = int(p.size - tp - fp - fn)
            self.true_positive += tp
            self.false_positive += fp
            self.false_negative += fn
            self.true_negative += tn
            self.slices += 1

            denominator = 2 * tp + fp + fn
            if denominator == 0:
                self.slice_dice.append(1.0)
                self.slice_iou.append(1.0)
                self.empty_agreements += 1
            else:
                self.slice_dice.append(2.0 * tp / denominator)
                union = tp + fp + fn
                self.slice_iou.append(tp / union if union else 1.0)
                if not t.any():
                    self.false_alarms += 1

    # -- pooled metrics ------------------------------------------------------- #
    # Every one of these returns ``None`` when its denominator is zero, rather than a
    # filler. A recall of "1.0" for a class that appears nowhere in the split is not a
    # perfect score, it is an absence of evidence, and a reader scanning a report cannot
    # tell the two apart. The per-slice empty-empty convention in ``update`` is
    # different and stays: there, both prediction and truth are empty on a slice that
    # exists, which is a correct answer to a real question.
    @property
    def dice(self) -> float | None:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return (2.0 * self.true_positive / denominator) if denominator else None

    @property
    def iou(self) -> float | None:
        union = self.true_positive + self.false_positive + self.false_negative
        return (self.true_positive / union) if union else None

    @property
    def precision(self) -> float | None:
        """Undefined when the model predicted this class nowhere."""
        denominator = self.true_positive + self.false_positive
        return (self.true_positive / denominator) if denominator else None

    @property
    def recall(self) -> float | None:
        """Also the sensitivity — reported under both names because both were asked for
        and because a reader should not have to know they are the same quantity.
        Undefined when the class appears nowhere in the ground truth."""
        denominator = self.true_positive + self.false_negative
        return (self.true_positive / denominator) if denominator else None

    @property
    def specificity(self) -> float | None:
        denominator = self.true_negative + self.false_positive
        return (self.true_negative / denominator) if denominator else None

    def to_dict(self, percentile: float = 95.0) -> dict[str, Any]:
        distances = np.asarray(self.surface_distances, dtype=np.float64)
        rounded = lambda v: round(v, 5) if v is not None else None  # noqa: E731
        return {
            "dice": rounded(self.dice),
            "dice_per_slice_mean": round(float(np.mean(self.slice_dice)), 5)
                                   if self.slice_dice else None,
            "dice_per_slice_median": round(float(np.median(self.slice_dice)), 5)
                                     if self.slice_dice else None,
            "iou": rounded(self.iou),
            "iou_per_slice_mean": round(float(np.mean(self.slice_iou)), 5)
                                  if self.slice_iou else None,
            "precision": rounded(self.precision),
            "recall": rounded(self.recall),
            "sensitivity": rounded(self.recall),
            "specificity": rounded(self.specificity),
            f"hausdorff_p{int(percentile)}_px": (
                round(float(np.mean(distances)), 4) if distances.size else None),
            "hausdorff_slices_measured": int(distances.size),
            "voxels": {"tp": self.true_positive, "fp": self.false_positive,
                       "fn": self.false_negative, "tn": self.true_negative},
            "slices": self.slices,
            "empty_agreements": self.empty_agreements,
            "false_alarms": self.false_alarms,
        }


class SegmentationMeter:
    """Accumulates every segmentation metric over a validation pass."""

    def __init__(self, *, compute_hausdorff: bool = True,
                 percentile: float = 95.0) -> None:
        self.compute_hausdorff = bool(compute_hausdorff)
        self.percentile = float(percentile)
        self.classes: dict[str, BinaryCounts] = {
            REGION_KEYS[region]: BinaryCounts() for region in FOREGROUND_REGIONS}
        self.composites: dict[str, BinaryCounts] = {
            region.value: BinaryCounts() for region in CompositeRegion}
        self._distance_backend = _try_distance_backend()

    def update(self, predicted: np.ndarray, truth: np.ndarray) -> None:
        """``predicted`` and ``truth`` are ``(B, H, W)`` dense class labels."""
        for region in FOREGROUND_REGIONS:
            key = REGION_KEYS[region]
            p = predicted == region.value
            t = truth == region.value
            self.classes[key].update(p, t)
            self._maybe_hausdorff(self.classes[key], p, t)

        for composite in CompositeRegion:
            members = COMPOSITE_MEMBERS[composite]
            p = np.isin(predicted, [m.value for m in members])
            t = np.isin(truth, [m.value for m in members])
            self.composites[composite.value].update(p, t)
            self._maybe_hausdorff(self.composites[composite.value], p, t)

    def _maybe_hausdorff(self, counts: BinaryCounts, predicted: np.ndarray,
                         truth: np.ndarray) -> None:
        if not self.compute_hausdorff or self._distance_backend is None:
            return
        for index in range(predicted.shape[0]):
            # Only slices where both masks exist. A distance to an empty set is
            # undefined, and substituting the image diagonal — as some implementations
            # do — makes the metric a function of how often the class is absent.
            if predicted[index].any() and truth[index].any():
                counts.surface_distances.append(
                    surface_distance_percentile(predicted[index], truth[index],
                                                self.percentile,
                                                self._distance_backend))

    def summary(self) -> dict[str, Any]:
        per_class = {name: counts.to_dict(self.percentile)
                     for name, counts in self.classes.items()}
        per_composite = {name: counts.to_dict(self.percentile)
                         for name, counts in self.composites.items()}
        composite_dice = [per_composite[c.value]["dice"] for c in CompositeRegion]
        class_dice = [per_class[REGION_KEYS[r]]["dice"] for r in FOREGROUND_REGIONS]
        # A region that never occurs contributes no Dice, so it is excluded from the
        # mean and the exclusion is counted. Substituting a value would let an absent
        # class raise or lower the headline number.
        usable_composite = [v for v in composite_dice if v is not None]
        usable_class = [v for v in class_dice if v is not None]
        return {
            "per_class": per_class,
            "per_composite": per_composite,
            "composite_dice_mean": (round(float(np.mean(usable_composite)), 5)
                                    if usable_composite else None),
            "composite_regions_scored": len(usable_composite),
            "composite_regions_absent": len(composite_dice) - len(usable_composite),
            "class_dice_mean": (round(float(np.mean(usable_class)), 5)
                                if usable_class else None),
            "classes_scored": len(usable_class),
            "hausdorff_percentile": self.percentile,
            "hausdorff_available": self._distance_backend is not None,
            "note": ("dice is pooled over all voxels of the split; "
                     "dice_per_slice_mean averages the per-slice value, which weighs a "
                     "3-pixel focus the same as a 3000-pixel tumour"),
        }


def surface_distance_percentile(predicted: np.ndarray, truth: np.ndarray,
                                percentile: float, ndimage: Any) -> float:
    """Symmetric ``percentile``-th surface distance, in pixels.

    Symmetric because the one-sided version is trivially gamed: a prediction that is a
    single pixel inside the true region has a perfect prediction-to-truth distance. Both
    directions are pooled and the percentile is taken over the union, which is the
    formulation the BraTS evaluation uses.
    """
    predicted_surface = _surface(predicted, ndimage)
    truth_surface = _surface(truth, ndimage)
    if not predicted_surface.any() or not truth_surface.any():
        return float("nan")
    distance_to_truth = ndimage.distance_transform_edt(~truth_surface)
    distance_to_prediction = ndimage.distance_transform_edt(~predicted_surface)
    pooled = np.concatenate([distance_to_truth[predicted_surface],
                             distance_to_prediction[truth_surface]])
    return float(np.percentile(pooled, percentile))


def _surface(mask: np.ndarray, ndimage: Any) -> np.ndarray:
    """Boundary voxels of a binary mask: the mask minus its erosion."""
    if not mask.any():
        return mask
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


# --------------------------------------------------------------------------- #
# The other heads
# --------------------------------------------------------------------------- #
class ClassificationMeter:
    """Presence-head metrics: accuracy, sensitivity, specificity, and AUROC per region."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        self._scores: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, scores: np.ndarray, targets: np.ndarray) -> None:
        self._scores.append(np.asarray(scores, dtype=np.float64))
        self._targets.append(np.asarray(targets, dtype=np.float64))

    def summary(self, threshold: float = 0.5) -> dict[str, Any]:
        if not self._scores:
            return {}
        scores = np.concatenate(self._scores)
        targets = np.concatenate(self._targets)
        result: dict[str, Any] = {}
        for index, name in enumerate(self.names):
            score = scores[:, index]
            target = targets[:, index] > 0.5
            predicted = score >= threshold
            tp = int(np.count_nonzero(predicted & target))
            fp = int(np.count_nonzero(predicted & ~target))
            fn = int(np.count_nonzero(~predicted & target))
            tn = int(np.count_nonzero(~predicted & ~target))
            # ``max(denominator, 1)`` would turn "this class never occurs" into a
            # sensitivity of 0.0 and "the model never predicted it" into a precision of
            # 0.0 — two fillers that read as measured failures. Undefined is None.
            def rate(numerator: int, denominator: int) -> float | None:
                return round(numerator / denominator, 5) if denominator else None

            result[name] = {
                "prevalence": round(float(target.mean()), 5),
                "positives": int(tp + fn),
                "accuracy": rate(tp + tn, len(target)),
                "sensitivity": rate(tp, tp + fn),
                "specificity": rate(tn, tn + fp),
                "precision": rate(tp, tp + fp),
                "auroc": _auroc(score, target),
            }
        return result


class RegressionMeter:
    """Size- and quality-head metrics: error, and correlation with the truth.

    Correlation is reported because it is the statistic that catches a head predicting
    the target's mean. A constant predictor can have a respectable mean absolute error
    on a low-variance target and has a correlation of exactly zero — which is the failure
    mode the quality head was designed against, so it has to be measured, not assumed
    away.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        self._predicted: list[np.ndarray] = []
        self._truth: list[np.ndarray] = []

    def update(self, predicted: np.ndarray, truth: np.ndarray) -> None:
        self._predicted.append(np.atleast_2d(np.asarray(predicted, dtype=np.float64)))
        self._truth.append(np.atleast_2d(np.asarray(truth, dtype=np.float64)))

    def summary(self) -> dict[str, Any]:
        if not self._predicted:
            return {}
        predicted = np.concatenate(self._predicted)
        truth = np.concatenate(self._truth)
        result: dict[str, Any] = {}
        for index, name in enumerate(self.names):
            p, t = predicted[:, index], truth[:, index]
            result[name] = {
                "mae": round(float(np.mean(np.abs(p - t))), 5),
                "rmse": round(float(np.sqrt(np.mean((p - t) ** 2))), 5),
                "pearson_r": _pearson(p, t),
                "predicted_std": round(float(p.std()), 5),
                "target_std": round(float(t.std()), 5),
            }
        return result


class EmbeddingMeter:
    """Measures whether the exported latent space is worth exporting.

    Three questions, in increasing order of how much they prove.

    1. Is it collapsed? Mean pairwise cosine similarity and the mean per-dimension
       standard deviation. A collapsed space has similarity near 1 and standard
       deviation near 0, and every other number here becomes meaningless.
    2. Does it cluster what it was trained to cluster? k-NN purity over the morphology
       class. Necessary but weak — it is measuring the training objective.
    3. Does it carry something it was never told? k-NN accuracy over the **tumour
       grade**, which no loss in this package ever sees. This is the honest probe: a
       representation that separates high- from low-grade glioma without having been
       shown a grade has learned something about tumours rather than about the label
       function.
    """

    def __init__(self, k: int = 15, max_samples: int = 6000,
                 probe_grade: bool = True) -> None:
        self.k = int(k)
        #: Run the held-out grade probe. Off for a corpus with no grade labels, where
        #: the probe would report a k-NN accuracy over 6 000 ``UNKNOWN`` values.
        self.probe_grade = bool(probe_grade)
        #: Cap on samples entering the probe. The similarity matrix is N x N — 7 400
        #: validation slices is 219 MB and a top-k search over every row — so the meter
        #: takes a prefix of the pass. Validation runs in a fixed random permutation,
        #: so a prefix is a random subset rather than the first few subjects.
        self.max_samples = int(max_samples)
        self._embeddings: list[np.ndarray] = []
        self._morphology: list[np.ndarray] = []
        self._grade: list[np.ndarray] = []
        self._subject: list[np.ndarray] = []

    @property
    def _collected(self) -> int:
        return sum(int(e.shape[0]) for e in self._embeddings)

    def update(self, embeddings: np.ndarray, morphology: np.ndarray,
               grade: np.ndarray, subject: np.ndarray) -> None:
        room = self.max_samples - self._collected
        if room <= 0:
            return
        take = min(room, int(np.asarray(embeddings).shape[0]))
        self._embeddings.append(np.asarray(embeddings[:take], dtype=np.float32))
        self._morphology.append(np.asarray(morphology[:take], dtype=np.int64))
        self._grade.append(np.asarray(grade[:take], dtype=np.int64))
        self._subject.append(np.asarray(subject[:take], dtype=np.int64))

    def summary(self) -> dict[str, Any]:
        if not self._embeddings:
            return {}
        embeddings = np.concatenate(self._embeddings)
        morphology = np.concatenate(self._morphology)
        grade = np.concatenate(self._grade)
        subject = np.concatenate(self._subject)
        if embeddings.shape[0] < self.k + 2:
            return {"samples": int(embeddings.shape[0]),
                    "note": "too few samples for a neighbourhood probe"}

        normalised = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True)
                                   + 1e-8)
        similarity = normalised @ normalised.T
        np.fill_diagonal(similarity, -np.inf)

        collapse = {
            "mean_pairwise_cosine": round(
                float(np.mean(similarity[np.isfinite(similarity)])), 5),
            "mean_dimension_std": round(float(embeddings.std(axis=0).mean()), 5),
            "effective_rank": _effective_rank(embeddings),
        }
        neighbours = _top_k(similarity, self.k)

        known = grade < 2
        grade_probe: dict[str, Any]
        if not self.probe_grade:
            grade_probe = {"note": "disabled by configuration"}
        elif known.sum() >= self.k + 2 and len(np.unique(grade[known])) > 1:
            # Neighbours are restricted to *other subjects*: slice 71 and slice 72 of
            # one subject are near-duplicates, so an unrestricted k-NN would report the
            # grade of the sample's own neighbouring slices and score ~1.0 while proving
            # nothing at all.
            grade_probe = {
                "knn_accuracy_cross_subject": _knn_score(
                    similarity, grade, subject, self.k, exclude_same_subject=True),
                "majority_baseline": round(float(np.mean(
                    grade[known] == np.bincount(grade[known]).argmax())), 5),
                "samples_with_grade": int(known.sum()),
                "note": ("tumour grade is never a training target; neighbours from the "
                         "same subject are excluded so adjacent slices cannot answer "
                         "for each other"),
            }
        else:
            grade_probe = {"note": "not enough graded samples for the probe"}

        return {
            "samples": int(embeddings.shape[0]),
            "dimension": int(embeddings.shape[1]),
            "collapse": collapse,
            "morphology_knn_purity": _knn_purity(neighbours, morphology),
            "morphology_classes_present": int(np.unique(morphology).size),
            "grade_probe": grade_probe,
        }


class PerformanceMeter:
    """Inference latency and peak GPU memory — the two numbers a deployment needs."""

    def __init__(self) -> None:
        self._batches: list[tuple[float, int]] = []
        self.peak_memory_bytes: int = 0

    def record(self, seconds: float, samples: int) -> None:
        self._batches.append((float(seconds), int(samples)))

    def note_memory(self, peak_bytes: int) -> None:
        self.peak_memory_bytes = max(self.peak_memory_bytes, int(peak_bytes))

    def summary(self) -> dict[str, Any]:
        if not self._batches:
            return {}
        total_time = sum(t for t, _ in self._batches)
        total_samples = sum(n for _, n in self._batches)
        per_slice = np.asarray([t / max(n, 1) for t, n in self._batches])
        return {
            "slices": total_samples,
            "total_seconds": round(total_time, 4),
            "ms_per_slice_mean": round(float(per_slice.mean() * 1000), 4),
            "ms_per_slice_p95": round(float(np.percentile(per_slice, 95) * 1000), 4),
            "slices_per_second": round(total_samples / total_time, 2)
                                 if total_time > 0 else None,
            "peak_gpu_memory_mb": (round(self.peak_memory_bytes / 1024 ** 2, 2)
                                   if self.peak_memory_bytes else None),
        }


class LossMeter:
    """Running mean of every loss component."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, values: Mapping[str, float], weight: int = 1) -> None:
        for name, value in values.items():
            if np.isfinite(value):
                self._sums[name] += float(value) * weight
                self._counts[name] += weight

    def summary(self) -> dict[str, float]:
        return {name: round(total / self._counts[name], 6)
                for name, total in self._sums.items() if self._counts[name]}

    @property
    def total(self) -> float:
        return self.summary().get("total", float("nan"))


# --------------------------------------------------------------------------- #
def _try_distance_backend() -> Any:
    try:
        from scipy import ndimage

        return ndimage
    except ImportError:                                  # pragma: no cover - env dep
        return None


def _auroc(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Rank-based AUROC. ``None`` when one class is absent, never 0.5 by default."""
    positives = int(np.count_nonzero(targets))
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Average ranks within ties, otherwise a head that outputs a constant scores 1.0.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sums = np.zeros(unique.size)
    np.add.at(tie_sums, inverse, ranks)
    ranks = (tie_sums / counts)[inverse]
    rank_sum = float(ranks[targets > 0].sum())
    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives),
                 5)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 5)


def _effective_rank(embeddings: np.ndarray) -> float:
    """Entropy-based effective rank of the embedding covariance.

    A 128-dimensional space that really uses 4 directions has an effective rank near 4.
    More informative than the raw dimension, which is a configuration constant.
    """
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    p = singular / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    return round(float(np.exp(entropy)), 3)


def _top_k(similarity: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` most similar entries per row.

    ``argpartition`` rather than ``argsort``: only the *membership* of the neighbourhood
    matters to purity and to a majority vote, not the order within it, and a full sort
    of a 6 000 x 6 000 matrix costs tens of seconds on every validation cycle.
    """
    k = min(int(k), similarity.shape[1] - 1)
    partitioned = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    return partitioned


def _knn_purity(neighbours: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of each sample's k neighbours that share its label."""
    same = labels[neighbours] == labels[:, None]
    return round(float(same.mean()), 5)


def _knn_score(similarity: np.ndarray, labels: np.ndarray, groups: np.ndarray,
               k: int, *, exclude_same_subject: bool) -> float | None:
    """Leave-one-out k-NN accuracy, optionally excluding same-group neighbours."""
    known = labels < 2
    if known.sum() < k + 2:
        return None
    scores = similarity.copy()
    if exclude_same_subject:
        scores[groups[:, None] == groups[None, :]] = -np.inf
    scores[:, ~known] = -np.inf

    correct = 0
    total = 0
    for index in np.flatnonzero(known):
        row = scores[index]
        candidates = np.flatnonzero(np.isfinite(row))
        if candidates.size < k:
            continue
        values = row[candidates]
        top = candidates[np.argpartition(-values, kth=k - 1)[:k]]
        vote = np.bincount(labels[top], minlength=3)[:2]
        correct += int(vote.argmax() == labels[index])
        total += 1
    return round(correct / total, 5) if total else None


class Timer:
    """Context manager for one timed section."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self._start


__all__ = [
    "BinaryCounts", "ClassificationMeter", "EmbeddingMeter", "LossMeter",
    "PerformanceMeter", "RegressionMeter", "SegmentationMeter", "Timer",
    "surface_distance_percentile",
]
