"""MemoryEngine — associative retrieval over evidence embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MemoryRecord:
    case_id: str
    embedding: np.ndarray
    diagnosis: str


class MemoryEngine:
    def __init__(self):
        self._store: list[MemoryRecord] = []
        self.model_version = "memory-v1"

    def index(self, case_id: str, embedding: list[float] | np.ndarray, diagnosis: str) -> None:
        self._store.append(
            MemoryRecord(case_id=case_id, embedding=np.asarray(embedding, dtype=float),
                         diagnosis=diagnosis)
        )

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def similar(self, embedding: list[float] | np.ndarray, k: int = 3,
                exclude: str | None = None) -> list[dict]:
        q = np.asarray(embedding, dtype=float)
        scored = [
            {"case_id": r.case_id, "diagnosis": r.diagnosis,
             "similarity": round(self._cosine(q, r.embedding), 4)}
            for r in self._store if r.case_id != exclude
        ]
        scored.sort(key=lambda d: -d["similarity"])
        return scored[:k]

    def prior_delta(self, current: list[float], prior: list[float]) -> dict:
        """Element-wise evidence delta vs a prior study (registration assumed done)."""
        c, p = np.asarray(current, dtype=float), np.asarray(prior, dtype=float)
        delta = c - p
        return {
            "l2": round(float(np.linalg.norm(delta)), 4),
            "max_increase": round(float(delta.max()), 4),
            "max_decrease": round(float(delta.min()), 4),
        }
