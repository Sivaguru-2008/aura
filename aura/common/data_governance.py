"""Which trained artifacts AURA may be sold with, and the gate that enforces it.

Every model AURA serves today was trained on data released for research only; the
dataset-by-dataset review is docs/DATA_LICENSES.md. ``AURA_COMMERCIAL_MODE=1`` declares a
commercial deployment. In that mode the gateway and preflight refuse to start while any
trained or fitted file under ``artifacts/`` lacks a recorded commercial licence. The gate
fails closed: a file this module has no record of blocks too, so a new checkpoint cannot
slip through by being unknown. Research and pilot use (``AURA_COMMERCIAL_MODE=0``, the
default) are unaffected.

An engineering control, not legal advice. A record here states what the team believes the
terms are; mark an artifact COMMERCIAL_AUTHORIZED only after counsel has confirmed it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aura.common.config import ARTIFACTS, get_settings

#: Trained or fitted artifacts: files that carry information learned from data.
WEIGHT_SUFFIXES = (".pt", ".pth", ".npz", ".npy", ".joblib", ".onnx", ".pkl", ".safetensors")

#: Violations named in the error message; the rest are counted.
_MAX_LISTED = 20


class LicenseType(str, Enum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    COMMERCIAL_AUTHORIZED = "COMMERCIAL_AUTHORIZED"
    UNVERIFIED = "UNVERIFIED"


class CommercialLicenseViolationError(RuntimeError):
    """A commercial deployment would serve artifacts not licensed for commercial use."""


@dataclass(frozen=True)
class ModelProvenance:
    relative_path: str          # under artifacts/, with forward slashes
    description: str
    source_dataset: str
    governing_terms: str
    license_type: LicenseType
    notes: str = ""


_MIMIC = ("MIMIC-CXR, from a third-party Kaggle re-upload rather than PhysioNet; no longer "
          "on disk (docs/DATA_LICENSES.md, audit of 2026-09-13)")
_MIMIC_TERMS = "PhysioNet Credentialed Health Data License 1.5.0: scientific research only"
_BRATS = "BraTS 2020 training set, 369 subjects (UPenn CBICA)"
_BRATS_TERMS = "BraTS 2020 data terms: use in your own research; no commercial terms"
_CHEXPERT = "CheXpert-v1.0-small (Stanford AIMI)"
_CHEXPERT_TERMS = "Stanford research use agreement: non-commercial"
_RESEARCH = LicenseType.RESEARCH_ONLY

KNOWN_PROVENANCE: dict[str, ModelProvenance] = {p.relative_path: p for p in (
    ModelProvenance("best_model.pt", "Served chest DenseNet-121 (retrain_v2, epoch 7)",
                    _MIMIC, _MIMIC_TERMS, _RESEARCH,
                    "Initialised from ImageNet weights; ImageNet's terms of access are "
                    "non-commercial too and belong in the same legal review"),
    ModelProvenance("vision_serving_calibration.json",
                    "Served per-finding Platt scaling and thresholds",
                    _MIMIC + "; 2,099 validation images", _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("fusion_classical.npz", "Served diagnosis fusion head (product of experts)",
                    _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("fusion_quantum.npz", "Selectable VQC fusion head",
                    _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("fusion_learnable.npz", "Research fusion head", _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("fusion_ensemble.npz", "Research fusion head", _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("safety_classical.npz", "Served temperature and conformal calibration",
                    _MIMIC + "; 446 studies", _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("safety.npz", "Calibration used by the research harnesses",
                    _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("safety_quantum.npz", "Per-backend calibration", _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("safety_learnable.npz", "Per-backend calibration", _MIMIC, _MIMIC_TERMS, _RESEARCH),
    ModelProvenance("brain/checkpoints/best_brain_model.pt", "Served brain tumour segmenter",
                    _BRATS, _BRATS_TERMS, _RESEARCH),
    ModelProvenance("brain/presence_calibration.json", "Served tumour-presence Platt scaling",
                    _BRATS, _BRATS_TERMS, _RESEARCH),
    ModelProvenance("brain/sequence_check.joblib", "MRI channel-order check",
                    _BRATS, _BRATS_TERMS, _RESEARCH),
    ModelProvenance("brain/embeddings/latest.npz", "Brain case-similarity embeddings",
                    _BRATS, _BRATS_TERMS, _RESEARCH),
    ModelProvenance("brain/qkl_classifier.npz", "Glioma-grade head (55 validation subjects)",
                    _BRATS, _BRATS_TERMS, _RESEARCH),
    ModelProvenance("vision_cnn_chexpert.pt", "Unserved CheXpert-trained CNN",
                    _CHEXPERT, _CHEXPERT_TERMS, _RESEARCH,
                    "Excluded from Docker images by .dockerignore"),
    ModelProvenance("vision_cnn_chexpert_last.pt", "Unserved CheXpert-trained CNN (last epoch)",
                    _CHEXPERT, _CHEXPERT_TERMS, _RESEARCH,
                    "Excluded from Docker images by .dockerignore"),
    ModelProvenance("chexpert_evidence.npz", "External-validation evidence arrays",
                    _CHEXPERT, _CHEXPERT_TERMS, _RESEARCH,
                    "Excluded from Docker images by .dockerignore"),
)}


@dataclass
class GovernanceAuditReport:
    commercial_mode: bool
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def cleared_for_commercial_use(self) -> bool:
        return not self.violations and all(
            a["license_type"] == LicenseType.COMMERCIAL_AUTHORIZED.value for a in self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {"commercial_mode": self.commercial_mode,
                "cleared_for_commercial_use": self.cleared_for_commercial_use,
                "artifacts": self.artifacts,
                "violations": self.violations}


class DataGovernanceManager:
    """Audits an artifacts directory against ``KNOWN_PROVENANCE``."""

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS

    def _artifact_files(self) -> list[str]:
        found = {rel for rel in KNOWN_PROVENANCE if (self.artifacts_dir / rel).is_file()}
        if self.artifacts_dir.is_dir():
            for path in self.artifacts_dir.rglob("*"):
                if path.suffix.lower() in WEIGHT_SUFFIXES and path.is_file():
                    found.add(path.relative_to(self.artifacts_dir).as_posix())
        return sorted(found)

    def audit_artifacts(self, commercial_mode: bool | None = None) -> GovernanceAuditReport:
        """Every trained or fitted file found with its recorded licence and, in commercial
        mode, each one that may not be sold."""
        if commercial_mode is None:
            commercial_mode = get_settings().commercial_mode
        report = GovernanceAuditReport(commercial_mode=commercial_mode)
        for rel in self._artifact_files():
            prov = KNOWN_PROVENANCE.get(rel)
            license_type = prov.license_type if prov else LicenseType.UNVERIFIED
            report.artifacts.append({
                "path": rel,
                "license_type": license_type.value,
                "source_dataset": prov.source_dataset if prov else "",
                "governing_terms": prov.governing_terms if prov else "",
            })
            if commercial_mode and license_type is not LicenseType.COMMERCIAL_AUTHORIZED:
                reason = prov.governing_terms if prov else "no provenance record"
                report.violations.append(f"{rel}: {license_type.value} ({reason})")
        return report

    def enforce_commercial_gate(self, commercial_mode: bool | None = None) -> GovernanceAuditReport | None:
        """Raise if a commercial deployment would serve unlicensed artifacts.

        Returns None without scanning when commercial mode is off: the scan walks every
        file under artifacts/, and research and pilot deployments are not what it guards.
        """
        if commercial_mode is None:
            commercial_mode = get_settings().commercial_mode
        if not commercial_mode:
            return None
        report = self.audit_artifacts(commercial_mode=True)
        if report.violations:
            listed = report.violations[:_MAX_LISTED]
            more = len(report.violations) - len(listed)
            raise CommercialLicenseViolationError(
                f"AURA_COMMERCIAL_MODE=1, but {len(report.violations)} artifact(s) under "
                f"{self.artifacts_dir} are not licensed for commercial use:\n  - "
                + "\n  - ".join(listed)
                + (f"\n  ... and {more} more" if more else "")
                + "\nRetrain on data licensed for commercial use, or record a licence in "
                  "aura/common/data_governance.py once counsel has confirmed it "
                  "(docs/DATA_PROVENANCE_REGISTRY.md).")
        return report


_default_manager: DataGovernanceManager | None = None


def get_governance_manager() -> DataGovernanceManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = DataGovernanceManager()
    return _default_manager
