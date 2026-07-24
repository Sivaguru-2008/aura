"""AURA NeuroMind — Brain Vision Engine.

The first perception module of NeuroMind. It is not a segmentation tool that happens to
expose some features; it is a representation learner whose segmentation output is one
of five things it produces, and the other four exist because the modules that come
after it — digital twin, progression prediction, longitudinal comparison, treatment and
surgical planning, the neuro-oncology copilot — need a description of a brain, not a
mask.

    from backend.vision.brain import BrainVisionEngine

    engine = BrainVisionEngine.load()
    result = engine.analyze_study(foundation_study)   # -> BrainVisionOutput
    result.embedding            # 128-d latent, ready for a downstream module
    result.regions              # per-region volume, confidence, presence
    result.to_dict()            # JSON-safe; never a tensor

The pipeline::

    BraTS corpus
        -> MRI Foundation Layer          reorientation, QC, masking, provenance
        -> FoundationStudy               per subject, cached as JSON + voxels
        -> BrainSliceDataset             curriculum + region-focused + hard-mined
        -> multi-task training           5 heads over 1 shared encoder
        -> BrainVisionNetwork
        -> BrainVisionOutput             mask, confidence, size, embedding, metadata

Four properties this module holds to, each because the alternative fails invisibly:

* **The label goes wherever the image goes.** Every geometric transform — orientation,
  crop, augmentation — is applied to both, in the same call, with nearest-neighbour
  interpolation for the label. A model trained on a mirrored or shifted ground truth
  converges normally and is worthless.
* **No raw tensor leaves this package.** :class:`BrainVisionOutput` holds numpy arrays
  and plain Python, so nothing downstream can accidentally hold a CUDA reference, and
  ``to_dict()`` cannot serialise a hundred megabytes of voxels into a log line.
* **What was measured and what was assumed are kept apart.** The corpus's modality
  channel order was verified by a test that reruns on every ingest; its laterality was
  not, and the model card says so. A number nobody validated must not look like one
  somebody did.
* **Nothing here touches the Thorax stack.** Every artefact this module writes lives
  under ``artifacts/brain/``; every import points at ``backend.foundation`` or at
  itself.
"""
from typing import TYPE_CHECKING, Any

from backend.vision.brain.config import (
    AugmentationConfig,
    BrainVisionConfig,
    CurriculumConfig,
    DegradationConfig,
    IngestConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    PathsConfig,
    SamplingConfig,
    SplitConfig,
    ValidationConfig,
    smoke_config,
)
from backend.vision.brain.errors import (
    ArchitectureUnavailable,
    BrainVisionError,
    CacheUnavailable,
    CheckpointError,
    ConfigurationError,
    CorpusIntegrityError,
    CorpusNotFound,
    ModelNotTrained,
)
from backend.vision.brain.types import (
    BRAIN_VISION_VERSION,
    CACHE_VERSION,
    CompositeRegion,
    CurriculumStage,
    EmbeddingSpec,
    HeadName,
    ModalitySpec,
    SplitName,
    TumorGrade,
    TumorRegion,
)

if TYPE_CHECKING:                                        # pragma: no cover
    from backend.vision.brain.dataset import BrainSliceDataset
    from backend.vision.brain.ingest import BrainCorpusIngestor, CacheManifest
    from backend.vision.brain.inference import BrainVisionEngine
    from backend.vision.brain.model import BrainVisionNetwork
    from backend.vision.brain.output import BrainVisionOutput
    from backend.vision.brain.train import BrainVisionTrainer

#: Names that pull in torch. Resolved on first attribute access rather than at import,
#: so ``import backend.vision.brain`` stays cheap and works in a deployment that has the
#: foundation layer but no deep-learning stack — the registry imports engine metadata at
#: startup and must not pay for CUDA initialisation to do it.
_LAZY: dict[str, str] = {
    "BrainCorpusIngestor": "backend.vision.brain.ingest",
    "CacheManifest": "backend.vision.brain.ingest",
    "load_manifest": "backend.vision.brain.ingest",
    "BrainSliceDataset": "backend.vision.brain.dataset",
    "build_datasets": "backend.vision.brain.dataset",
    "BrainVisionNetwork": "backend.vision.brain.model",
    "build_network": "backend.vision.brain.model",
    "BrainVisionTrainer": "backend.vision.brain.train",
    "BrainVisionEngine": "backend.vision.brain.inference",
    "BrainVisionOutput": "backend.vision.brain.output",
    "RegionFinding": "backend.vision.brain.output",
    "EmbeddingStore": "backend.vision.brain.embeddings",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    # engine surface (lazy)
    "BrainVisionEngine", "BrainVisionOutput", "RegionFinding", "BrainVisionNetwork",
    "build_network", "BrainVisionTrainer", "BrainSliceDataset", "build_datasets",
    "BrainCorpusIngestor", "CacheManifest", "load_manifest", "EmbeddingStore",
    # configuration
    "BrainVisionConfig", "PathsConfig", "IngestConfig", "SplitConfig", "SamplingConfig",
    "CurriculumConfig", "AugmentationConfig", "DegradationConfig", "ModelConfig",
    "LossConfig", "OptimConfig", "ValidationConfig", "smoke_config",
    # vocabulary
    "TumorRegion", "CompositeRegion", "HeadName", "CurriculumStage", "TumorGrade",
    "SplitName", "ModalitySpec", "EmbeddingSpec", "BRAIN_VISION_VERSION",
    "CACHE_VERSION",
    # errors
    "BrainVisionError", "CorpusNotFound", "CorpusIntegrityError", "CacheUnavailable",
    "ConfigurationError", "ArchitectureUnavailable", "CheckpointError",
    "ModelNotTrained",
]
