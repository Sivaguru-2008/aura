"""Model Registry — cryptographic verification of model checkpoints.

Extends the basic version registry to log SHA-256 checksums of model
checkpoints (best_model.pt, fusion_*.npz, safety.npz, etc.).  Hashes are
committed to the provenance ledger, preventing silent model updates or
corruption.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aura.common.config import ARTIFACTS


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


class ModelRegistry:
    """Model registry with cryptographic checkpoint verification.

    Tracks model versions, their artifact paths, and SHA-256 hashes.
    Any mismatch between stored and recomputed hashes indicates corruption.
    """

    def __init__(self, filename: str = "registry.json"):
        self.path = ARTIFACTS / filename

    def list_versions(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def write(self, versions: list[dict]) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(versions, indent=2))

    def active(self) -> list[dict]:
        return [v for v in self.list_versions() if v.get("status") == "active"]

    def register_checkpoint(self, name: str, artifact_path: Path | str,
                            version: str = "", metadata: dict | None = None) -> dict:
        """Register a model checkpoint with its SHA-256 hash.

        Returns the checkpoint record.
        """
        artifact_path = Path(artifact_path)
        record = {
            "name": name,
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path) if artifact_path.exists() else "",
            "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        # Append or update in versions list
        versions = self.list_versions()
        existing_idx = next(
            (i for i, v in enumerate(versions) if v.get("name") == name), None
        )
        if existing_idx is not None:
            record["previous_hash"] = versions[existing_idx].get("sha256", "")
            versions[existing_idx] = record
        else:
            record["previous_hash"] = ""
            versions.append(record)

        self.write(versions)
        return record

    def verify_checkpoint(self, name: str) -> dict:
        """Verify that a registered checkpoint's hash matches the current file.

        Returns {"valid": bool, "recorded_hash": str, "current_hash": str, ...}
        """
        versions = self.list_versions()
        record = next((v for v in versions if v.get("name") == name), None)
        if record is None:
            return {"valid": False, "error": "not_registered", "name": name}

        path = Path(record["path"])
        if not path.exists():
            return {
                "valid": False, "error": "file_missing",
                "name": name, "path": record["path"],
                "recorded_hash": record.get("sha256", ""),
            }

        current_hash = sha256_file(path)
        valid = current_hash == record.get("sha256", "")

        return {
            "valid": valid,
            "name": name,
            "path": str(path),
            "recorded_hash": record.get("sha256", ""),
            "current_hash": current_hash,
            "version": record.get("version", ""),
            "size_bytes": record.get("size_bytes", 0),
        }

    def verify_all(self) -> list[dict]:
        """Verify all registered checkpoints."""
        versions = self.list_versions()
        results = []
        for v in versions:
            name = v.get("name", "")
            if name:
                results.append(self.verify_checkpoint(name))
        return results

    def list_artifacts(self) -> list[dict]:
        """List all known artifact files and their verification status."""
        known_artifacts = [
            "safety.npz", "safety_quantum.npz", "safety_classical.npz",
            "safety_learnable.npz", "fusion_quantum.npz", "fusion_classical.npz",
            "fusion_ensemble.npz", "fusion_learnable.npz",
            "vision.npz", "conformal_mondrian.npy",
            "vision_serving_calibration.json",
        ]
        results = []
        for name in known_artifacts:
            path = ARTIFACTS / name
            exists = path.exists()
            hash_val = sha256_file(path) if exists else ""
            results.append({
                "name": name,
                "exists": exists,
                "sha256": hash_val,
                "size_bytes": path.stat().st_size if exists else 0,
            })
        return results

    def provenance_entry(self, name: str) -> dict | None:
        """Generate a provenance ledger entry for a checkpoint."""
        verification = self.verify_checkpoint(name)
        if verification.get("error"):
            return None
        return {
            "type": "model_checkpoint",
            "name": name,
            "sha256": verification.get("current_hash", ""),
            "version": verification.get("version", ""),
            "verified": verification.get("valid", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
