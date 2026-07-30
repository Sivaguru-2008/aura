from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aura.schemas.clinical import Diagnosis, Modality


@dataclass
class AgentVerdict:
    agent_name: str
    agent_version: str = "1.0"
    findings: dict[Diagnosis, float] = field(default_factory=dict)
    confidence: float = 0.0
    arguments: list[str] = field(default_factory=list)
    guideline_references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ClinicalAgent(ABC):
    agent_name: str = "base"

    @abstractmethod
    async def evaluate(self, evidence: dict, priors: dict) -> AgentVerdict:
        ...

    def supported(self, modality: Modality) -> bool:
        return True

    def get_base_weight(self) -> float:
        return 0.2
