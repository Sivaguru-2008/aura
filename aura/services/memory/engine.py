"""MemoryEngine — associative retrieval over evidence embeddings."""
from __future__ import annotations

import numpy as np

from aura.services.memory.vectorstore import get_vector_store, VectorRecord


class MemoryEngine:
    def __init__(self, store=None):
        # We allow wrapping a custom store object if provided, otherwise we resolve
        # to the process-wide persistent vector store.
        self.store = store if (store is not None and hasattr(store, "upsert")) else get_vector_store()
        self.model_version = "memory-v2"

    def index(self, case_id: str, embedding: list[float] | np.ndarray, diagnosis: str,
              patient_id: str = "", modality: str = "", study_id: str = "") -> None:
        emb_arr = np.asarray(embedding, dtype=np.float32)
        # Create a VectorRecord and upsert it into the vector store
        record = VectorRecord(
            case_id=case_id,
            embedding=emb_arr,
            diagnosis=diagnosis,
            patient_id=patient_id,
            modality=modality,
            study_id=study_id,
        )
        self.store.upsert(record)

    def similar(self, embedding: list[float] | np.ndarray, k: int = 3,
                 exclude: str | None = None, patient_id: str | None = None,
                 modality: str | None = None, diagnosis: str | None = None,
                 exclude_patient: str | None = None) -> list[dict]:
        emb_arr = np.asarray(embedding, dtype=np.float32)
        hits = self.store.search(
            emb_arr,
            k=k,
            exclude=exclude,
            patient_id=patient_id,
            modality=modality,
            diagnosis=diagnosis,
            exclude_patient=exclude_patient
        )
        return [
            {
                "case_id": hit.case_id,
                "diagnosis": hit.diagnosis,
                "similarity": round(float(hit.similarity), 4),
            }
            for hit in hits
        ]

    def prior_delta(self, current: list[float], prior: list[float]) -> dict:
        """Element-wise evidence delta vs a prior study (registration assumed done)."""
        c, p = np.asarray(current, dtype=float), np.asarray(prior, dtype=float)
        delta = c - p
        return {
            "l2": round(float(np.linalg.norm(delta)), 4),
            "max_increase": round(float(delta.max()), 4),
            "max_decrease": round(float(delta.min()), 4),
        }

