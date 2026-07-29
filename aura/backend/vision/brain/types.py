"""Vocabulary of the Brain Vision Engine.

Enums and small frozen value types only, separated from the code that uses them
because they are this module's public contract. A future NeuroMind module —
progression prediction, a digital twin, a surgical planner — imports
:class:`TumorRegion` or :class:`EmbeddingSpec` to declare what it consumes, and must
not have to import a training loop to do it.

Two conventions are fixed here and everything else follows from them.

**Label space.** BraTS ships labels ``0/1/2/4``; the gap at 3 is a historical artefact
of a removed class. Carrying it forward would mean every ``nn.Module`` in this package
allocating five channels to hold four classes, so label 4 is remapped to 3 exactly
once, at ingest, and :data:`BRATS_LABEL_REMAP` is the only place that knows. Downstream
code sees a dense ``0..3``.

**Two region vocabularies, deliberately.** :class:`TumorRegion` is what the network
predicts — four mutually exclusive classes, which is what a softmax can represent.
:class:`CompositeRegion` is what the field scores: whole tumour, tumour core, enhancing
tumour, which are *nested unions* of the primary classes and are not mutually
exclusive. Reporting only the primary classes would make our numbers incomparable with
every published BraTS result; predicting the composites directly would mean three
overlapping sigmoid heads whose outputs can contradict each other (an enhancing voxel
outside the whole tumour). Predicting one and reporting both is the resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from aura.backend.foundation.mri.types import SequenceType

#: Output-format version for everything this module writes — checkpoints, embedding
#: stores, cached studies, :class:`~aura.backend.vision.brain.output.BrainVisionOutput`.
#: Bumped when the shape of any of those changes in a way a consumer could notice, so
#: an artefact found on disk can always be matched against the code that produced it.
BRAIN_VISION_VERSION = "1.0.0"

#: Version of the on-disk ingest cache layout. Separate from the model version because
#: the two move independently: retraining does not invalidate a cache, and a cache
#: format change does not invalidate a trained network's weights.
CACHE_VERSION = "1.0.0"


class TumorRegion(int, Enum):
    """The four mutually exclusive classes the segmentation head predicts.

    Integer-valued because these *are* the channel indices of the network's output and
    the values stored in a cached label volume; an ``IntEnum`` lets both be written
    without a lookup table and read back without a cast.
    """

    BACKGROUND = 0
    #: Necrotic and non-enhancing tumour core (BraTS label 1).
    NECROTIC_CORE = 1
    #: Peritumoural oedema / invaded tissue (BraTS label 2).
    EDEMA = 2
    #: Gd-enhancing tumour (BraTS label 4, remapped — see the module docstring).
    ENHANCING = 3

    @property
    def label(self) -> str:
        return REGION_LABELS[self]


#: Human-readable names for reports and API responses.
REGION_LABELS: dict[TumorRegion, str] = {
    TumorRegion.BACKGROUND: "Background",
    TumorRegion.NECROTIC_CORE: "Necrotic / non-enhancing core",
    TumorRegion.EDEMA: "Peritumoural oedema",
    TumorRegion.ENHANCING: "Enhancing tumour",
}

#: Short keys used in metric dictionaries and JSON. Stable — downstream dashboards
#: key off these.
REGION_KEYS: dict[TumorRegion, str] = {
    TumorRegion.BACKGROUND: "background",
    TumorRegion.NECROTIC_CORE: "ncr_net",
    TumorRegion.EDEMA: "edema",
    TumorRegion.ENHANCING: "enhancing",
}

#: BraTS on-disk label -> dense class index. Applied once, at ingest.
BRATS_LABEL_REMAP: dict[int, int] = {0: 0, 1: 1, 2: 2, 4: 3}

#: Classes a loss or a metric should actually be computed over. Background is excluded
#: from foreground metrics because it is 97% of every slice and averaging it in turns
#: a Dice of 0.0 on the enhancing tumour into a headline number of 0.49.
FOREGROUND_REGIONS: tuple[TumorRegion, ...] = (
    TumorRegion.NECROTIC_CORE, TumorRegion.EDEMA, TumorRegion.ENHANCING)


class CompositeRegion(str, Enum):
    """The nested unions the BraTS challenge scores. Derived, never predicted directly."""

    WHOLE_TUMOR = "whole_tumor"          # NCR/NET + oedema + enhancing
    TUMOR_CORE = "tumor_core"            # NCR/NET + enhancing
    ENHANCING_TUMOR = "enhancing_tumor"  # enhancing alone

    @property
    def members(self) -> tuple[TumorRegion, ...]:
        return COMPOSITE_MEMBERS[self]

    @property
    def label(self) -> str:
        return COMPOSITE_LABELS[self]


COMPOSITE_MEMBERS: dict[CompositeRegion, tuple[TumorRegion, ...]] = {
    CompositeRegion.WHOLE_TUMOR: (TumorRegion.NECROTIC_CORE, TumorRegion.EDEMA,
                                  TumorRegion.ENHANCING),
    CompositeRegion.TUMOR_CORE: (TumorRegion.NECROTIC_CORE, TumorRegion.ENHANCING),
    CompositeRegion.ENHANCING_TUMOR: (TumorRegion.ENHANCING,),
}

COMPOSITE_LABELS: dict[CompositeRegion, str] = {
    CompositeRegion.WHOLE_TUMOR: "Whole tumour",
    CompositeRegion.TUMOR_CORE: "Tumour core",
    CompositeRegion.ENHANCING_TUMOR: "Enhancing tumour",
}


class HeadName(str, Enum):
    """The prediction heads of the multi-task network.

    Names are stable identifiers: they key the loss weights in configuration, the
    per-head metrics in a validation report, and the head state dicts in a checkpoint.
    Adding a head means adding a member here and nothing else structural — the trainer
    iterates over whatever the network declares.
    """

    SEGMENTATION = "segmentation"
    PRESENCE = "presence"
    SIZE = "size"
    QUALITY = "quality"
    EMBEDDING = "embedding"


class CurriculumStage(str, Enum):
    """Difficulty tiers the sampler walks through during training.

    ``FULL`` is not "the hard stage" — it is the whole distribution including
    tumour-free slices, which the earlier stages deliberately under-represent. A model
    that never returns to the full distribution has been trained on a prevalence it
    will never see at inference.
    """

    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"
    FULL = "full"


class TumorGrade(str, Enum):
    """WHO grade group, from the BraTS name mapping.

    Never used as a training target in this module. It is held out on purpose so that
    "does the embedding separate high- from low-grade glioma?" stays an honest probe of
    representation quality rather than a measurement of what we optimised.
    """

    HGG = "hgg"
    LGG = "lgg"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModalitySpec:
    """One input channel of the network.

    The seam that makes the "add PET, CT, histopathology" requirement a configuration
    change rather than a rewrite. A network is built from a tuple of these; the input
    stem allocates one filter bank per spec, the dataset stacks channels in this order,
    and a checkpoint records the tuple it was trained with so a mismatched input is
    caught at load rather than diagnosed from bad predictions.

    ``required=False`` marks a channel a study may legitimately lack. The dataset
    zero-fills it and sets the corresponding availability flag, so a model trained on
    four sequences can still run on a study that only has three — with the absence
    visible rather than imputed.
    """

    key: str
    sequence: SequenceType
    label: str
    required: bool = True
    #: Free-text note about where this channel comes from, carried into the model card.
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "sequence": self.sequence.value, "label": self.label,
                "required": self.required, "provenance": self.provenance}


#: The four structural sequences BraTS provides, in the channel order the corpus
#: stores them. That order was verified by measurement, not assumed — see
#: :mod:`aura.backend.vision.brain.io.brats_h5`.
DEFAULT_MODALITIES: tuple[ModalitySpec, ...] = (
    ModalitySpec("flair", SequenceType.FLAIR, "FLAIR", True,
                 "BraTS2020 HDF5 image channel 0"),
    ModalitySpec("t1", SequenceType.T1, "T1-weighted", True,
                 "BraTS2020 HDF5 image channel 1"),
    ModalitySpec("t1ce", SequenceType.T1CE, "T1-weighted post-contrast", True,
                 "BraTS2020 HDF5 image channel 2"),
    ModalitySpec("t2", SequenceType.T2, "T2-weighted", True,
                 "BraTS2020 HDF5 image channel 3"),
)


@dataclass(frozen=True)
class EmbeddingSpec:
    """Description of the latent representation this module exports.

    Travels with every stored embedding and inside every checkpoint. A consumer that
    finds an embedding on disk can tell what it is, what produced it, and whether its
    own assumptions still hold, without reading this package's source.
    """

    dimension: int
    #: Where in the network the embedding is taken from. ``encoder_bottleneck`` is the
    #: only implemented source, and the constraint is intentional: an embedding taken
    #: from decoder features could not be computed without running segmentation, which
    #: would defeat the purpose of exporting it.
    source: str = "encoder_bottleneck"
    normalized: bool = True
    #: Objective the embedding space was shaped by, for the record.
    objective: str = "supervised_contrastive+variance_covariance"
    version: str = BRAIN_VISION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "source": self.source,
                "normalized": self.normalized, "objective": self.objective,
                "version": self.version}


class SplitName(str, Enum):
    """Dataset partitions. Split by *subject*, never by slice — see the dataset module."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


__all__ = [
    "BRAIN_VISION_VERSION", "BRATS_LABEL_REMAP", "CACHE_VERSION",
    "COMPOSITE_LABELS", "COMPOSITE_MEMBERS", "CompositeRegion", "CurriculumStage",
    "DEFAULT_MODALITIES", "EmbeddingSpec", "FOREGROUND_REGIONS", "HeadName",
    "ModalitySpec", "REGION_KEYS", "REGION_LABELS", "SplitName", "TumorGrade",
    "TumorRegion",
]
