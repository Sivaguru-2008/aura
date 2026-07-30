from __future__ import annotations

from typing import Any

from aura.services.agent.base import ClinicalAgent


class AgentRegistry:
    _agents: dict[str, ClinicalAgent] = {}

    @classmethod
    def register(cls, agent: ClinicalAgent) -> None:
        cls._agents[agent.agent_name] = agent

    @classmethod
    def get(cls, name: str) -> ClinicalAgent | None:
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> list[str]:
        return list(cls._agents.keys())

    @classmethod
    def all(cls) -> list[ClinicalAgent]:
        return list(cls._agents.values())

    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
