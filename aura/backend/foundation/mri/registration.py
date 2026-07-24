"""Registration *preparation* — everything a registration step needs, and no transform.

Registration itself is explicitly out of scope. What is in scope is making the study
ready for it, which is a real and separable piece of work: agreeing the target space,
recording the moving image's geometry in that space, identifying which series share a
frame of reference and are therefore already aligned, and computing the centroid
offset that a rigid initialisation starts from.

:attr:`RegistrationPlan.transform` is ``None`` and stays ``None`` until something
actually computes one. It is typed as a placeholder rather than pre-filled with an
identity matrix on purpose: an identity transform is a *claim* that the volumes are
aligned, and a downstream module that applied it would silently assume a registration
that never happened.

The intra-study alignment finding is the piece with immediate practical value. Series
sharing a ``FrameOfReferenceUID`` were acquired in one physical frame without the
patient being repositioned, so they are already co-registered to within patient motion
— and re-registering them can be worse than leaving them alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from backend.foundation.mri.masking import BrainMaskSlot
from backend.foundation.mri.volume import MRIVolume


@dataclass(frozen=True)
class RegistrationPlan:
    """What a future registration step needs, computed but not applied."""

    #: World space the volume currently lives in.
    source_space: str = "RAS"
    #: Space registration should target. ``None`` means no template was requested;
    #: the study is prepared for registration but not committed to a destination.
    target_template: str | None = None
    target_spacing_mm: tuple[float, float, float] | None = None
    target_shape: tuple[int, int, int] | None = None

    #: Geometry of the volume as it stands — the moving image's description.
    source_spacing_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_shape: tuple[int, int, int] = (0, 0, 0)
    source_orientation: str = ""
    #: World coordinates of the volume's geometric centre, mm.
    geometric_centre_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: World coordinates of the intensity centroid over the mask, when one exists.
    #: This is what a rigid initialisation uses; the geometric centre is a poorer
    #: starting point when the head sits off-centre in the field of view.
    mask_centroid_mm: tuple[float, float, float] | None = None
    #: Offset from geometric centre to mask centroid. Large values mean the head is
    #: off-centre, which is the case where centre-of-mass initialisation earns its keep.
    centre_offset_mm: tuple[float, float, float] | None = None

    #: Frame of reference shared with other series, when the source recorded one.
    frame_of_reference_uid: str | None = None
    #: Series keys in the same study already in this frame of reference.
    aligned_series: tuple[str, ...] = ()

    #: The computed transform. ``None`` until a registration step fills it — never
    #: pre-filled with identity, which would assert an alignment nobody verified.
    transform: np.ndarray | None = None
    status: str = "prepared"
    notes: tuple[str, ...] = ()

    @property
    def is_registered(self) -> bool:
        return self.transform is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_space": self.source_space,
            "target_template": self.target_template,
            "target_spacing_mm": (list(self.target_spacing_mm)
                                  if self.target_spacing_mm else None),
            "target_shape": list(self.target_shape) if self.target_shape else None,
            "source_spacing_mm": [round(v, 5) for v in self.source_spacing_mm],
            "source_shape": list(self.source_shape),
            "source_orientation": self.source_orientation,
            "geometric_centre_mm": [round(v, 3) for v in self.geometric_centre_mm],
            "mask_centroid_mm": ([round(v, 3) for v in self.mask_centroid_mm]
                                 if self.mask_centroid_mm else None),
            "centre_offset_mm": ([round(v, 3) for v in self.centre_offset_mm]
                                 if self.centre_offset_mm else None),
            "frame_of_reference_uid": self.frame_of_reference_uid,
            "aligned_series": list(self.aligned_series),
            "is_registered": self.is_registered,
            "transform": None if self.transform is None else
                         [[round(float(v), 8) for v in row] for row in self.transform],
            "notes": list(self.notes),
        }


class RegistrationPreparer:
    """Computes a :class:`RegistrationPlan`. Applies nothing."""

    def prepare(self, volume: MRIVolume, *, mask: BrainMaskSlot | None = None,
                frame_of_reference_uid: str | None = None,
                aligned_series: Sequence[str] = (),
                target_template: str | None = None,
                target_spacing_mm: tuple[float, float, float] | None = None,
                ) -> RegistrationPlan:
        geometry = volume.geometry
        notes: list[str] = []

        centroid = self._mask_centroid(volume, mask)
        centre = geometry.centre_world_mm
        offset = (tuple(float(c - g) for c, g in zip(centroid, centre))
                  if centroid is not None else None)

        if offset is not None and max(abs(v) for v in offset) > 30.0:
            notes.append(
                f"the mask centroid sits {max(abs(v) for v in offset):.0f} mm from the "
                "field-of-view centre; a rigid registration should initialise from the "
                "centroid rather than the image centre")
        if mask is not None and mask.present and not mask.is_brain_mask:
            notes.append(
                "the centroid was computed over a head/foreground mask, not a brain "
                "mask; it includes skull and scalp and will shift inferiorly relative "
                "to a true brain centroid")
        if aligned_series:
            notes.append(
                f"{len(aligned_series)} other series share this frame of reference and "
                "are already in the same physical space; registering them to each "
                "other can introduce error rather than remove it")
        if target_template is None:
            notes.append(
                "no target template was requested, so the plan describes the moving "
                "image only; no transform has been computed")

        return RegistrationPlan(
            source_space=geometry.space,
            target_template=target_template,
            target_spacing_mm=target_spacing_mm,
            target_shape=None,
            source_spacing_mm=geometry.spacing,
            source_shape=geometry.shape,
            source_orientation=geometry.orientation,
            geometric_centre_mm=centre,
            mask_centroid_mm=centroid,
            centre_offset_mm=offset,
            frame_of_reference_uid=frame_of_reference_uid,
            aligned_series=tuple(aligned_series),
            transform=None,
            status="prepared",
            notes=tuple(notes),
        )

    @staticmethod
    def _mask_centroid(volume: MRIVolume,
                       mask: BrainMaskSlot | None) -> tuple[float, float, float] | None:
        """Centre of mass of the mask, in world millimetres."""
        if mask is None or mask.mask is None or not mask.mask.any():
            return None
        indices = np.argwhere(mask.mask)
        centre_voxel = indices.mean(axis=0)
        world = volume.geometry.affine @ np.append(centre_voxel, 1.0)
        return tuple(float(v) for v in world[:3])          # type: ignore[return-value]


__all__ = ["RegistrationPlan", "RegistrationPreparer"]
