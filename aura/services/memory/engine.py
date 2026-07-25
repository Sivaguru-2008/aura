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
    def __init__(self, store=None):
        self.store = store
        self._store: list[MemoryRecord] = []
        self.model_version = "memory-v1"

    def index(self, case_id: str, embedding: list[float] | np.ndarray, diagnosis: str) -> None:
        emb_arr = np.asarray(embedding, dtype=float)
        self._store.append(
            MemoryRecord(case_id=case_id, embedding=emb_arr, diagnosis=diagnosis)
        )
        if self.store is not None:
            try:
                self.store.add_memory_record(case_id, emb_arr, diagnosis)
            except Exception as e:
                print(f"[MemoryEngine] failed to write persistent memory record: {e}")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def similar(self, embedding: list[float] | np.ndarray, k: int = 3,
                 exclude: str | None = None) -> list[dict]:
        # Sync from persistent store if local cache is empty
        if not self._store and self.store is not None:
            try:
                records = self.store.list_memory_records()
                for r in records:
                    self._store.append(
                        MemoryRecord(case_id=r["case_id"], embedding=np.asarray(r["embedding"], dtype=float),
                                     diagnosis=r["diagnosis"])
                    )
            except Exception as e:
                print(f"[MemoryEngine] failed to load persistent memory records: {e}")

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
