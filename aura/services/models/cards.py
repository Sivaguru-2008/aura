"""Registry cards for the models this build serves, derived from the artifacts on disk.

A card says what a model is, what it was trained and measured on, and the SHA-256 of the
file being served. Every figure is read from an artifact; a missing artifact leaves the
field empty rather than filled in. Approval is never granted here: cards are written
``PENDING_REVIEW`` and a person changes that through the audited status endpoint.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from aura.common.config import ARTIFACTS

SERVED_MODEL_IDS: tuple[str, ...] = (
    "vision-cxr-densenet121-production",
    "fusion-poe-v1",
    "brain-resunet-v1",
)

#: ``validation_dataset`` values written by the 2026-09-12 seed, which marked all three
#: cards APPROVED and cited a superseded 602-image chest evaluation. A card still carrying
#: one of these is rewritten at startup.
SUPERSEDED_SEED_DATASETS = frozenset({
    "MIMIC-CXR test split (n=602)",
    "CheXpert external validation (n=4800)",
    "BraTS 2020 validation (n=7531 slices)",
})


@lru_cache(maxsize=16)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(rel: str) -> str:
    """SHA-256 of an artifact, or "" when absent. Cached on size and mtime."""
    p = ARTIFACTS / rel
    if not p.is_file():
        return ""
    st = p.stat()
    return _sha256_cached(str(p), st.st_size, st.st_mtime_ns)


def _json(rel: str) -> dict:
    p = ARTIFACTS / rel
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


def _version(sha: str) -> str:
    # The served file's own hash, so the version cannot drift from what is deployed.
    return f"sha256:{sha[:12]}" if sha else "unknown (artifact absent)"


def _chest_card() -> dict:
    ev = _json("evaluation_retrain_v2/metrics.json")
    macro = ev.get("macro") or {}
    n = ev.get("n_images")
    metrics: dict = {}
    if macro.get("auroc") is not None:
        metrics = {
            "macro_auroc": macro["auroc"],
            "macro_auprc": macro.get("auprc"),
            "macro_precision": macro.get("precision"),
            "macro_ece": macro.get("ece"),
            "n_images": n,
            "source_artifact": "evaluation_retrain_v2/metrics.json",
        }
    sha = _sha256("best_model.pt")
    return {
        "model_id": "vision-cxr-densenet121-production",
        "org_id": "system",
        "name": "DenseNet-121 chest radiograph findings model",
        "version": _version(sha),
        "modality": "CXR",
        "intended_use": ("Research decision support: seven findings on adult frontal chest "
                         "radiographs, every output reviewed by a clinician. Not a cleared "
                         "device."),
        "source": ("DenseNet-121 initialised from ImageNet weights and fine-tuned in this "
                   "repository (ml/vision_cxr)."),
        "training_data_provenance": (
            "MIMIC-CXR, from a Kaggle re-upload of the PhysioNet release (research-use "
            "licence), about 259,000 frontal images. Labels come from a rule-based parser "
            "over the radiology reports. See docs/DATA_LICENSES.md."),
        "validation_dataset": (
            f"MIMIC-CXR validation split ({n} images, patient-disjoint). The same images "
            "selected the training epoch, so this is not an untouched test set."
            if n else "not recorded (evaluation_retrain_v2/metrics.json absent)"),
        "validation_metrics": metrics,
        "checksum_sha256": sha,
        "approval_status": "PENDING_REVIEW",
        "deployment_status": "ACTIVE",
    }


def _fusion_card() -> dict:
    ext = ((_json("chexpert_external.json").get("primary_uncertain_as_negative") or {})
           .get("backends") or {})
    poe = ext.get("classical") or {}
    metrics: dict = {}
    if poe:
        metrics = {k: poe.get(k) for k in ("accuracy", "nll", "ece", "macro_auroc", "n")}
        metrics["source_artifact"] = "chexpert_external.json"
    sha = _sha256("fusion_classical.npz")
    return {
        "model_id": "fusion-poe-v1",
        "org_id": "system",
        "name": "Bayesian product-of-experts evidence fusion (served backend)",
        "version": _version(sha),
        "modality": "MULTIMODAL_CXR",
        "intended_use": ("Combines the chest model's seven finding probabilities with "
                         "structured priors into a posterior over six diagnoses."),
        "source": "services/fusion/classical.py, trained by ml/training/train_fusion.py",
        "training_data_provenance": ("Evidence vectors produced by the chest model on a "
                                     "class-capped sample of MIMIC-CXR studies."),
        "validation_dataset": ("CheXpert, preregistered external validation, test studies, "
                               "models frozen" if poe
                               else "not recorded (chexpert_external.json absent)"),
        "validation_metrics": metrics,
        "checksum_sha256": sha,
        "approval_status": "PENDING_REVIEW",
        "deployment_status": "ACTIVE",
    }


def _brain_card() -> dict:
    rep = _json("brain/reports/test_report.json")
    seg = rep.get("segmentation") or {}
    comp = seg.get("per_composite") or {}
    metrics: dict = {}
    if comp:
        metrics = {
            "dice_whole_tumor": (comp.get("whole_tumor") or {}).get("dice"),
            "dice_tumor_core": (comp.get("tumor_core") or {}).get("dice"),
            "dice_enhancing_tumor": (comp.get("enhancing_tumor") or {}).get("dice"),
            "composite_dice": seg.get("composite_dice_mean"),
            "presence_auroc": ((rep.get("presence") or {}).get("whole_tumor") or {}).get("auroc"),
            "slices": rep.get("samples"),
            "scoring": "Dice pooled over 2-D slices; not comparable to per-case 3-D BraTS scores",
            "source_artifact": "brain/reports/test_report.json",
        }
    sha = _sha256("brain/checkpoints/best_brain_model.pt")
    return {
        "model_id": "brain-resunet-v1",
        "org_id": "system",
        "name": "Multi-task residual U-Net, brain tumour segmentation",
        "version": _version(sha),
        "modality": "BRAIN_MRI",
        "intended_use": ("Research decision support: glioma segmentation on complete "
                         "four-sequence MRI (FLAIR, T1, T1ce, T2). Not validated for other "
                         "tumour types or post-operative scans."),
        "source": "Trained from scratch in this repository (backend/vision/brain).",
        "training_data_provenance": (
            "BraTS 2020 training release (research use); subject-level train/validation/test "
            "split recorded in brain/reports/model_card.json. See docs/DATA_LICENSES.md."),
        "validation_dataset": (f"{rep.get('samples')} slices from held-out BraTS 2020 subjects "
                               "(test split)" if rep
                               else "not recorded (brain/reports/test_report.json absent)"),
        "validation_metrics": metrics,
        "checksum_sha256": sha,
        "approval_status": "PENDING_REVIEW",
        "deployment_status": "ACTIVE",
    }


def served_model_cards() -> dict[str, dict]:
    """``{model_id: card}`` for every model in :data:`SERVED_MODEL_IDS`."""
    return {c["model_id"]: c for c in (_chest_card(), _fusion_card(), _brain_card())}
