"""ModelRegistry — reads artifacts/registry.json written by training."""
from __future__ import annotations

import json
from pathlib import Path

from aura.common.config import ARTIFACTS


class ModelRegistry:
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
