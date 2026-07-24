"""Synthetic MR artefacts, and why the quality head needs them.

The MRI Foundation Layer measures image quality and does it honestly. On this corpus
that measurement is *nearly constant*: every BraTS subject was skull-stripped,
N4-corrected, and resampled by the challenge organisers with the same tooling, so the
foundation quality score varies over a range of a few hundredths. A regression head
trained on that target learns the mean and reports it with total confidence. It would
pass every test in this package and it would be worthless — a "quality prediction" that
is really a constant, dressed up as a model output.

So quality supervision is manufactured, and the label is exact because we chose it. A
configurable share of every batch is degraded by a known amount with a known artefact,
and the head predicts the severity it can see. Undegraded samples carry the foundation
layer's own per-slice quality estimate. The head therefore learns "how bad is this
image", not "what number does this corpus usually have", and the validation report
measures it against severities it has never seen — which is a claim that can be
checked.

Each artefact is simulated in the domain where it actually occurs. That is not
decoration:

* **Rician noise** is what magnitude MR noise really is. Gaussian noise added to a
  magnitude image produces negative values in the background that no scanner ever
  produces, and a network can learn to detect the simulation instead of the artefact.
* **Motion ghosting** replicates the object along the *phase-encode* direction only,
  because that is the axis k-space is traversed slowly along. Simulating it as a blur
  or a translation would teach the head to look for the wrong thing — and the
  foundation layer's own motion check measures the phase-encode/frequency-encode energy
  ratio, so the two would disagree.
* **A k-space spike** — a single corrupted sample — produces a full-field sinusoidal
  stripe pattern, which looks nothing like anything one would draw in image space.
* **Bias field** is multiplicative and smooth, which is what an inhomogeneous receive
  coil does. Additive shading is a different artefact with a different signature.

All of these run on the cached, background-zero intensities *before* normalisation,
because that is the order in which a scanner produces them: the artefact is in the
acquisition, and normalisation happens afterwards to whatever the acquisition gave.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from backend.vision.brain.config import DegradationConfig


#: Canonical artefact ordering. Fixed here rather than derived from configuration so
#: the artefact-type head has a stable output space: a checkpoint trained with four
#: artefacts enabled must still mean the same thing by "class 2" as one trained with
#: five. ``CLEAN_INDEX`` is the last class.
ARTIFACT_ORDER: tuple[str, ...] = ("rician_noise", "bias_field", "motion_ghosting",
                                   "k_space_spike", "blur")
CLEAN_INDEX: int = len(ARTIFACT_ORDER)
ARTIFACT_CLASSES: int = len(ARTIFACT_ORDER) + 1


@dataclass(frozen=True)
class Degradation:
    """One applied artefact and everything needed to score the head against it."""

    name: str
    #: Severity in [0, 1]. ``0.0`` means nothing was applied.
    severity: float
    #: The quality target the head is trained against: ``1 - severity``.
    target_quality: float
    parameters: dict[str, Any]

    @property
    def index(self) -> int:
        """Canonical class index — the artefact-type head's target."""
        try:
            return ARTIFACT_ORDER.index(self.name)
        except ValueError:
            return CLEAN_INDEX

    @classmethod
    def none(cls, quality: float = 1.0) -> "Degradation":
        return cls("none", 0.0, float(quality), {})

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "index": self.index,
                "severity": round(self.severity, 4),
                "target_quality": round(self.target_quality, 4),
                "parameters": dict(self.parameters)}


class DegradationSimulator:
    """Applies a randomly chosen artefact at a randomly chosen severity."""

    def __init__(self, config: DegradationConfig) -> None:
        self.config = config
        self._registry: dict[str, Callable[..., tuple[np.ndarray, dict[str, Any]]]] = {
            "rician_noise": rician_noise,
            "bias_field": bias_field,
            "motion_ghosting": motion_ghosting,
            "k_space_spike": k_space_spike,
            "blur": blur,
        }
        unknown = set(config.artifacts) - set(self._registry)
        if unknown:
            raise ValueError(f"unknown degradation(s): {sorted(unknown)}; "
                             f"available: {sorted(self._registry)}")

    @property
    def artifacts(self) -> tuple[str, ...]:
        return tuple(self.config.artifacts)

    def __call__(self, image: np.ndarray, rng: np.random.Generator, *,
                 base_quality: float = 1.0,
                 force: str | None = None) -> tuple[np.ndarray, Degradation]:
        """Maybe degrade ``image`` ``(C, H, W)``.

        Args:
            image: cached, background-zero intensities, before normalisation.
            rng: the caller's generator, so a worker owns its own stream.
            base_quality: quality target for an *undegraded* sample — the foundation
                layer's own estimate for this slice.
            force: apply this artefact regardless of ``probability``. Used by the
                validation pass, which needs known severities rather than a coin flip.
        """
        if not self.config.enabled or not self.config.artifacts:
            return image, Degradation.none(base_quality)
        if force is None and rng.random() >= self.config.probability:
            return image, Degradation.none(base_quality)

        name = force or str(rng.choice(self.config.artifacts))
        low, high = self.config.severity_range
        severity = float(rng.uniform(low, high))
        degraded, parameters = self._registry[name](image, severity, rng)
        # The target is the *observable* quality: a lightly degraded image is still a
        # good image. Scaling by the base quality means a slice the foundation layer
        # already flagged cannot be reported as pristine just because we added nothing.
        target = float(np.clip(base_quality * (1.0 - severity), 0.0, 1.0))
        return degraded, Degradation(name, severity, target, parameters)


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
def rician_noise(image: np.ndarray, severity: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    """Magnitude noise, the way magnitude MR actually has it.

    Real and imaginary Gaussian noise is added to the complex signal and the magnitude
    is taken, which is Rician. In the background — where the true signal is zero — that
    collapses to a Rayleigh distribution with a non-zero mean, exactly the grey speckle
    a real scanner puts in air, rather than the symmetric noise a naive additive model
    produces.
    """
    scale = severity * 0.25 * _reference_level(image)
    real = image + rng.normal(0.0, scale, size=image.shape)
    imaginary = rng.normal(0.0, scale, size=image.shape)
    noisy = np.sqrt(real * real + imaginary * imaginary)
    return noisy.astype(image.dtype, copy=False), {"sigma": round(float(scale), 5)}


def bias_field(image: np.ndarray, severity: float, rng: np.random.Generator
               ) -> tuple[np.ndarray, dict[str, Any]]:
    """Smooth multiplicative shading, as an inhomogeneous receive coil produces.

    Built as a low-order 2D polynomial rather than as smoothed noise: a real bias field
    has no high-frequency content at all, and smoothed noise retains a little, which is
    enough for a network to key on the simulator rather than the artefact.
    """
    height, width = image.shape[1], image.shape[2]
    grid_y, grid_x = np.meshgrid(np.linspace(-1.0, 1.0, height),
                                 np.linspace(-1.0, 1.0, width), indexing="ij")
    order = 3
    field = np.zeros((height, width), dtype=np.float32)
    for power_y in range(order + 1):
        for power_x in range(order + 1 - power_y):
            coefficient = rng.normal(0.0, 1.0)
            field += coefficient * (grid_y ** power_y) * (grid_x ** power_x)
    spread = float(np.abs(field).max()) + 1e-8
    # exp() keeps the field strictly positive: a bias field attenuates and amplifies,
    # it never inverts the sign of the signal.
    multiplier = np.exp((field / spread) * severity * 0.8).astype(np.float32)
    return (image * multiplier[None]).astype(image.dtype, copy=False), {
        "order": order, "max_gain": round(float(multiplier.max()), 4),
        "min_gain": round(float(multiplier.min()), 4)}


def motion_ghosting(image: np.ndarray, severity: float, rng: np.random.Generator
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Periodic k-space modulation along the phase-encode axis -> discrete ghosts.

    Every ``spacing``-th phase-encode line is scaled, which in image space replicates
    the object at regular offsets along that axis. That periodicity is the ghosting
    signature — and it is the same one the foundation layer's motion check looks for,
    so an artefact simulated here is detectable by the quality machinery already in the
    codebase rather than only by this head.
    """
    axis = int(rng.integers(1, 3))                 # a spatial axis of (C, H, W)
    spacing = int(rng.integers(2, 6))
    attenuation = 1.0 - 0.85 * severity
    spectrum = np.fft.fft(image, axis=axis)
    modulation = np.ones(image.shape[axis], dtype=np.float32)
    modulation[::spacing] = attenuation
    shape = [1, 1, 1]
    shape[axis] = image.shape[axis]
    ghosted = np.fft.ifft(spectrum * modulation.reshape(shape), axis=axis)
    return np.abs(ghosted).astype(image.dtype, copy=False), {
        "axis": axis, "line_spacing": spacing,
        "attenuation": round(float(attenuation), 4)}


def k_space_spike(image: np.ndarray, severity: float, rng: np.random.Generator
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    """A single corrupted k-space sample -> full-field sinusoidal stripes.

    A real and unmistakable artefact (RF interference, a bad gradient amplifier). Worth
    including precisely because it is invisible to a heuristic that looks for blur or
    noise: the image stays sharp and its histogram barely moves, and only the stripe
    pattern gives it away.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(image, axes=(1, 2)), axes=(1, 2))
    magnitude = float(np.abs(spectrum).max())
    height, width = image.shape[1], image.shape[2]
    # Away from the DC centre — a spike at the origin is just a brightness change.
    row = int(rng.integers(height // 8, height - height // 8))
    column = int(rng.integers(width // 8, width - width // 8))
    spectrum[:, row, column] += magnitude * severity * 0.35
    striped = np.fft.ifft2(np.fft.ifftshift(spectrum, axes=(1, 2)), axes=(1, 2))
    return np.abs(striped).astype(image.dtype, copy=False), {
        "row": row, "column": column,
        "relative_amplitude": round(severity * 0.35, 4)}


def blur(image: np.ndarray, severity: float, rng: np.random.Generator
         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Loss of high spatial frequency — partial volume, low resolution, gross motion.

    Applied in k-space with a Gaussian window when scipy is unavailable, so the
    artefact does not silently disappear in a deployment without it.
    """
    sigma = 0.2 + severity * 2.5
    try:
        from scipy import ndimage

        blurred = np.empty_like(image)
        for channel in range(image.shape[0]):
            ndimage.gaussian_filter(image[channel], sigma=sigma,
                                    output=blurred[channel], mode="nearest")
        return blurred, {"sigma": round(float(sigma), 4), "implementation": "scipy"}
    except ImportError:                                  # pragma: no cover - env dep
        height, width = image.shape[1], image.shape[2]
        freq_y = np.fft.fftfreq(height)[:, None]
        freq_x = np.fft.fftfreq(width)[None, :]
        window = np.exp(-2.0 * (np.pi * sigma) ** 2 * (freq_y ** 2 + freq_x ** 2))
        spectrum = np.fft.fft2(image, axes=(1, 2)) * window[None]
        blurred = np.real(np.fft.ifft2(spectrum, axes=(1, 2)))
        return blurred.astype(image.dtype, copy=False), {
            "sigma": round(float(sigma), 4), "implementation": "fft"}


# --------------------------------------------------------------------------- #
def _reference_level(image: np.ndarray) -> float:
    """A robust signal level for scaling noise: the 95th percentile of brain voxels.

    The maximum would be a single hot voxel and the mean would be dragged towards zero
    by the background. The 95th percentile of the non-zero voxels tracks tissue signal,
    which is what a scanner's SNR is quoted against.
    """
    positive = image[image > 0]
    if positive.size == 0:
        return 1.0
    return float(np.percentile(positive, 95)) or 1.0


__all__ = ["ARTIFACT_CLASSES", "ARTIFACT_ORDER", "CLEAN_INDEX", "Degradation",
           "DegradationSimulator", "bias_field", "blur", "k_space_spike",
           "motion_ghosting", "rician_noise"]
