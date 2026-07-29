"""Geometric and intensity augmentation for brain MRI slices.

Numpy and ``scipy.ndimage`` only — no torch, no albumentations. Two reasons, and the
second is the one that matters. Augmentation runs inside dataloader worker processes
where a torch import per worker is pure overhead; and, more importantly, every
geometric operation here has to be applied to the *label* as well as the image, with
nearest-neighbour interpolation, in the same call. A library whose transform signature
takes one array invites the bug where the label quietly does not follow — and a model
trained against a label that is rotated three degrees away from its image converges
perfectly happily to something that has learned to hedge every boundary.

So every function in this module takes ``(image, label)`` and returns ``(image,
label)``. There is no way to transform one without the other.

Left-right flipping is available and off by default. Laterality is a reportable
finding, and this corpus's laterality is unverified (see
:mod:`aura.backend.vision.brain.io.brats_h5`); mirroring on top of that would build a model
that is explicitly invariant to a distinction a radiologist has to make.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import AugmentationConfig


class SliceAugmenter:
    """Applies the configured augmentations to one ``(image, label)`` pair.

    Stateless apart from configuration: the random generator is passed in, so a
    dataloader worker owns its own stream and a validation pass can reproduce an exact
    sequence by handing in a seeded generator.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    @property
    def affine_available(self) -> bool:
        """Whether rotation/scale/shift can run. Without scipy the rest still works."""
        return _ndimage() is not None

    def __call__(self, image: np.ndarray, label: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Augment ``image`` ``(C, H, W)`` and ``label`` ``(H, W)`` together."""
        if not self.config.enabled:
            return image, label

        config = self.config
        p = config.probability

        if config.flip_lr and rng.random() < p:
            image, label = flip(image, label, axis=1)
        if config.flip_ap and rng.random() < p:
            image, label = flip(image, label, axis=0)
        if config.rot90 and rng.random() < p:
            image, label = rot90(image, label, int(rng.integers(1, 4)))
        if config.affine and _ndimage() is not None and rng.random() < p:
            image, label = affine(
                image, label, rng,
                max_rotation_deg=config.max_rotation_deg,
                max_scale=config.max_scale, max_shift=config.max_shift)
        if config.intensity_scale > 0 and rng.random() < p:
            image = intensity_jitter(image, rng, scale=config.intensity_scale,
                                     shift=config.intensity_shift)
        if config.gamma > 0 and rng.random() < p:
            image = gamma_jitter(image, rng, strength=config.gamma)
        return image, label

    def describe(self) -> dict[str, Any]:
        return {"enabled": self.config.enabled,
                "affine_available": self.affine_available,
                "flip_lr": self.config.flip_lr, "flip_ap": self.config.flip_ap,
                "rot90": self.config.rot90, "probability": self.config.probability}


# --------------------------------------------------------------------------- #
# Geometric — image and label always move together
# --------------------------------------------------------------------------- #
def flip(image: np.ndarray, label: np.ndarray, axis: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """Mirror both arrays about a spatial axis. ``axis`` is 0 or 1 in label space."""
    return np.flip(image, axis=axis + 1), np.flip(label, axis=axis)


def rot90(image: np.ndarray, label: np.ndarray, turns: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate both arrays by ``turns`` quarter-turns in plane.

    Exact — a transpose and a flip, no interpolation — so it can be applied to the
    label without any of the questions that a real rotation raises.
    """
    return (np.rot90(image, k=turns, axes=(1, 2)),
            np.rot90(label, k=turns, axes=(0, 1)))


def affine(image: np.ndarray, label: np.ndarray, rng: np.random.Generator, *,
           max_rotation_deg: float, max_scale: float,
           max_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Random in-plane rotation, isotropic scale, and translation.

    One matrix is drawn and applied to every image channel *and* to the label. The
    image is sampled with ``order=1``; the label with ``order=0``, because an
    interpolated label produces class values that were never annotated — a voxel
    halfway between oedema (2) and enhancing (3) becomes 2.5, and rounding it invents
    a boundary the annotator never drew.
    """
    ndimage = _ndimage()
    if ndimage is None:                                  # pragma: no cover - env dep
        return image, label
    height, width = label.shape
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    shift = np.array([rng.uniform(-max_shift, max_shift) * height,
                      rng.uniform(-max_shift, max_shift) * width])

    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    matrix = np.array([[cos, -sin], [sin, cos]])
    centre = np.array([height, width]) / 2.0
    offset = centre - matrix @ centre + shift

    warped = np.empty_like(image)
    for channel in range(image.shape[0]):
        ndimage.affine_transform(image[channel], matrix, offset=offset,
                                 output=warped[channel], order=1, mode="constant",
                                 cval=0.0, prefilter=False)
    warped_label = ndimage.affine_transform(
        label.astype(np.float32), matrix, offset=offset, order=0, mode="constant",
        cval=0.0, prefilter=False).astype(label.dtype)
    return warped, warped_label


# --------------------------------------------------------------------------- #
# Intensity — image only, by definition
# --------------------------------------------------------------------------- #
def intensity_jitter(image: np.ndarray, rng: np.random.Generator, *,
                     scale: float, shift: float) -> np.ndarray:
    """Per-channel multiplicative and additive jitter.

    Per channel rather than per image: MR sequences are acquired separately and drift
    independently, so a jitter that moves all four together simulates something that
    does not happen while leaving the thing that does — one sequence brighter than the
    others — untested.
    """
    channels = image.shape[0]
    gains = 1.0 + rng.uniform(-scale, scale, size=(channels, 1, 1))
    offsets = rng.uniform(-shift, shift, size=(channels, 1, 1))
    return (image * gains + offsets).astype(image.dtype, copy=False)


def gamma_jitter(image: np.ndarray, rng: np.random.Generator, *,
                 strength: float) -> np.ndarray:
    """Non-linear contrast change, applied on the image's own min-max range.

    Gamma is only defined on non-negative values, and normalised MR slices are not, so
    the channel is mapped to [0, 1], corrected, and mapped back. A channel with no
    dynamic range is returned untouched rather than producing a division by zero.
    """
    result = image.copy()
    for channel in range(image.shape[0]):
        plane = result[channel]
        low, high = float(plane.min()), float(plane.max())
        if high - low < 1e-6:
            continue
        gamma = float(np.exp(rng.uniform(-strength, strength)))
        normalised = (plane - low) / (high - low)
        result[channel] = np.power(normalised, gamma) * (high - low) + low
    return result


# --------------------------------------------------------------------------- #
#: Resolved once per process and held at module level rather than on the augmenter.
#: A :class:`SliceAugmenter` is an attribute of the dataset, and the dataset is pickled
#: to every spawn-based dataloader worker on Windows — a module object on it makes that
#: pickle fail with ``TypeError: cannot pickle 'module' object``, which surfaces as a
#: dataloader that works at ``num_workers=0`` and dies at 1.
_NDIMAGE: Any = None
_NDIMAGE_RESOLVED = False


def _ndimage() -> Any:
    global _NDIMAGE, _NDIMAGE_RESOLVED
    if not _NDIMAGE_RESOLVED:
        try:
            from scipy import ndimage

            _NDIMAGE = ndimage
        except ImportError:                              # pragma: no cover - env dep
            _NDIMAGE = None
        _NDIMAGE_RESOLVED = True
    return _NDIMAGE


__all__ = ["SliceAugmenter", "affine", "flip", "gamma_jitter", "intensity_jitter",
           "rot90"]
