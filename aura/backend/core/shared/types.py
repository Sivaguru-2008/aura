"""Primitive types shared by intake, routing, and engines.

The one type that deserves explanation is :class:`ImagingModality`. AURA already has
``schemas.clinical.Modality`` (``CXR`` / ``CT`` / ``MR``), which records the
*acquisition* modality on a study. That is deliberately coarse and it cannot express
what the router has to decide: "chest radiograph" and "abdominal radiograph" are both
``CXR``-class acquisitions, and "brain MRI" and "knee MRI" are both ``MR``. Routing is
a question about *acquisition x body region*, because that pair is what determines
which trained model may look at the image.

So :class:`ImagingModality` is the router's vocabulary — one member per clinically
meaningful (modality, region) route — and :func:`to_clinical_modality` maps it back
onto the existing enum wherever a ``StudyInput`` has to be built. The existing enum is
untouched.

Members exist for modalities AURA cannot analyse yet (CT, mammography, retina,
ultrasound). That is intentional: naming a study correctly and reporting "no engine
supports this" is a far better answer than ``UNKNOWN``, and it means adding the engine
later requires no change to this vocabulary.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ImagingModality(str, Enum):
    """Routing vocabulary: an acquisition modality paired with a body region."""

    # --- currently routable ------------------------------------------------- #
    CHEST_XRAY = "chest_xray"
    BRAIN_MRI = "brain_mri"

    # --- recognised, not yet served (future engines) ------------------------ #
    HEAD_CT = "head_ct"
    CHEST_CT = "chest_ct"
    MAMMOGRAPHY = "mammography"
    RETINAL_FUNDUS = "retinal_fundus"
    ULTRASOUND = "ultrasound"

    # --- fallbacks ---------------------------------------------------------- #
    #: A medical image whose region could not be pinned down (e.g. an MR series with
    #: no body-part tag that the pixel signature cannot place).
    UNKNOWN_MEDICAL = "unknown_medical"
    #: Nothing matched — likely not a medical image at all.
    UNKNOWN = "unknown"


MODALITY_LABELS: dict[ImagingModality, str] = {
    ImagingModality.CHEST_XRAY: "Chest radiograph",
    ImagingModality.BRAIN_MRI: "Brain MRI",
    ImagingModality.HEAD_CT: "Head CT",
    ImagingModality.CHEST_CT: "Chest CT",
    ImagingModality.MAMMOGRAPHY: "Mammogram",
    ImagingModality.RETINAL_FUNDUS: "Retinal fundus photograph",
    ImagingModality.ULTRASOUND: "Ultrasound",
    ImagingModality.UNKNOWN_MEDICAL: "Unidentified medical image",
    ImagingModality.UNKNOWN: "Unrecognised image",
}


def to_clinical_modality(modality: ImagingModality):
    """Map a routing modality onto the existing ``schemas.clinical.Modality``.

    Returns ``None`` when the routing modality has no acquisition-level equivalent
    (the ``UNKNOWN*`` members). Imported lazily so ``core.shared`` stays importable
    without the clinical schema package loaded.
    """
    from aura.schemas.clinical import Modality

    return {
        ImagingModality.CHEST_XRAY: Modality.CXR,
        ImagingModality.BRAIN_MRI: Modality.MR,
        ImagingModality.HEAD_CT: Modality.CT,
        ImagingModality.CHEST_CT: Modality.CT,
        ImagingModality.MAMMOGRAPHY: Modality.CXR,      # radiographic acquisition
        ImagingModality.ULTRASOUND: None,
        ImagingModality.RETINAL_FUNDUS: None,
        ImagingModality.UNKNOWN_MEDICAL: None,
        ImagingModality.UNKNOWN: None,
    }.get(modality)


class EngineStatus(str, Enum):
    """Lifecycle state an engine advertises about itself.

    The registry and API surface this so a client can tell the difference between
    "AURA cannot route this" and "AURA routed it correctly to something still being
    built" — the distinction the NeuroMind placeholder exists to demonstrate.
    """

    AVAILABLE = "available"                 # loaded and serving
    PLACEHOLDER = "placeholder"             # registered, routes resolve, analysis pending
    UNAVAILABLE = "unavailable"             # declared but not constructible here


@dataclass(frozen=True)
class ImageAsset:
    """One staged upload on local disk, plus the metadata intake established.

    Created only by ``backend.core.upload``; engines and the router receive it
    read-only. ``path`` points at a temp file whose lifetime the intake context
    manager owns — an engine must not assume it survives the request.
    """

    path: Path
    original_filename: str
    content_type: str | None
    size_bytes: int
    #: Content hash — the stable identifier for this upload in audit records. It lets
    #: a routing decision be tied to exact bytes without retaining the image.
    sha256: str
    #: Populated by intake when the file parses as DICOM. Empty for plain images.
    dicom_tags: dict[str, str] = field(default_factory=dict)

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def is_dicom(self) -> bool:
        return bool(self.dicom_tags)

    @staticmethod
    def digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()
