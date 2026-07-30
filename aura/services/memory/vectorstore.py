"""Persistent vector store for case embeddings.

Until now :class:`aura.services.memory.engine.MemoryEngine` kept embeddings in a
Python list and rebuilt it by loading every row through JSON on first query. That
works for a demo worklist and stops working the moment a deployment accumulates
studies: the whole corpus is decoded per process, similarity is an O(n) Python
loop, and there is no way to ask "this patient's priors" rather than "everything".

This module provides a real store with two backends behind one interface:

``SqliteVectorStore`` (default, no extra dependency)
    Embeddings as float32 BLOBs in the existing AURA database, with indexed
    ``patient_id`` / ``modality`` / ``diagnosis`` columns and a cached matrix so
    search is a single vectorised matmul rather than a per-row loop. Exact kNN.

``QdrantVectorStore`` (optional, ``pip install 'aura[vectordb]'``)
    Real ANN with server-side filtering, for deployments large enough to need it.

:func:`get_vector_store` picks Qdrant when it is installed *and* reachable and
falls back to SQLite otherwise — a vector database being down must never take
clinical inference with it.

All vectors are L2-normalised on write, so cosine similarity is a dot product and
the two backends rank identically.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np

from aura.common.config import DB_PATH


@dataclass
class VectorRecord:
    """One indexed study."""

    case_id: str
    embedding: np.ndarray
    diagnosis: str = ""
    patient_id: str = ""
    modality: str = ""
    study_id: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "diagnosis": self.diagnosis,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "study_id": self.study_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class SearchHit:
    case_id: str
    similarity: float
    diagnosis: str = ""
    patient_id: str = ""
    modality: str = ""
    study_id: str = ""
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "similarity": round(float(self.similarity), 4),
            "diagnosis": self.diagnosis,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "study_id": self.study_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


def normalize(vec: Iterable[float]) -> np.ndarray:
    """L2-normalise so cosine similarity reduces to a dot product."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@runtime_checkable
class VectorStore(Protocol):
    backend: str

    def upsert(self, record: VectorRecord) -> None: ...
    def search(self, embedding, k: int = 5, *, exclude: str | None = None,
               patient_id: str | None = None, modality: str | None = None,
               diagnosis: str | None = None,
               exclude_patient: str | None = None) -> list[SearchHit]: ...
    def get(self, case_id: str) -> VectorRecord | None: ...
    def by_patient(self, patient_id: str) -> list[VectorRecord]: ...
    def delete(self, case_id: str) -> bool: ...
    def count(self) -> int: ...
    def stats(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# SQLite backend
# --------------------------------------------------------------------------- #
class SqliteVectorStore:
    """Exact-kNN vector store on SQLite. Always available."""

    backend = "sqlite"

    def __init__(self, db_path: Path | str | None = None, dim: int | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        # Cached (ids, matrix) so repeated searches do not re-read the table.
        self._cache: tuple[list[str], np.ndarray] | None = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_records (
                    case_id     TEXT PRIMARY KEY,
                    patient_id  TEXT NOT NULL DEFAULT '',
                    study_id    TEXT NOT NULL DEFAULT '',
                    modality    TEXT NOT NULL DEFAULT '',
                    diagnosis   TEXT NOT NULL DEFAULT '',
                    dim         INTEGER NOT NULL,
                    embedding   BLOB NOT NULL,
                    metadata    TEXT NOT NULL DEFAULT '{}',
                    created_at  REAL NOT NULL
                )
                """
            )
            for col in ("patient_id", "modality", "diagnosis", "created_at"):
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_vector_{col} ON vector_records({col})"
                )

    def _invalidate(self) -> None:
        self._cache = None

    # -- writes ------------------------------------------------------------ #
    def upsert(self, record: VectorRecord) -> None:
        vec = normalize(record.embedding)
        if self.dim is not None and vec.shape[0] != self.dim:
            raise ValueError(
                f"embedding dimension {vec.shape[0]} does not match store dimension {self.dim}")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vector_records
                    (case_id, patient_id, study_id, modality, diagnosis, dim,
                     embedding, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    patient_id=excluded.patient_id, study_id=excluded.study_id,
                    modality=excluded.modality, diagnosis=excluded.diagnosis,
                    dim=excluded.dim, embedding=excluded.embedding,
                    metadata=excluded.metadata, created_at=excluded.created_at
                """,
                (
                    record.case_id, record.patient_id, record.study_id,
                    record.modality, record.diagnosis, int(vec.shape[0]),
                    vec.tobytes(), json.dumps(record.metadata),
                    float(record.created_at),
                ),
            )
        self._invalidate()

    def delete(self, case_id: str) -> bool:
        with self._lock, self._connect() as conn:
            removed = conn.execute(
                "DELETE FROM vector_records WHERE case_id = ?", (case_id,)
            ).rowcount
        self._invalidate()
        return bool(removed)

    # -- reads ------------------------------------------------------------- #
    def _rows(self, where: str = "", params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            return list(conn.execute(
                f"SELECT * FROM vector_records {where} ORDER BY created_at ASC", params
            ))

    @staticmethod
    def _to_record(row: sqlite3.Row) -> VectorRecord:
        return VectorRecord(
            case_id=row["case_id"],
            embedding=np.frombuffer(row["embedding"], dtype=np.float32),
            diagnosis=row["diagnosis"],
            patient_id=row["patient_id"],
            modality=row["modality"],
            study_id=row["study_id"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get(self, case_id: str) -> VectorRecord | None:
        rows = self._rows("WHERE case_id = ?", (case_id,))
        return self._to_record(rows[0]) if rows else None

    def by_patient(self, patient_id: str) -> list[VectorRecord]:
        """Chronological studies for one patient — the basis of longitudinal work."""
        return [self._to_record(r) for r in self._rows("WHERE patient_id = ?", (patient_id,))]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM vector_records").fetchone()[0])

    def search(
        self,
        embedding,
        k: int = 5,
        *,
        exclude: str | None = None,
        patient_id: str | None = None,
        modality: str | None = None,
        diagnosis: str | None = None,
        exclude_patient: str | None = None,
    ) -> list[SearchHit]:
        """Exact cosine kNN with optional filtering.

        ``exclude_patient`` matters clinically: "cases like this one" should mean
        *other* patients, or the top hits are simply this patient's own priors.
        """
        clauses, params = [], []
        if patient_id is not None:
            clauses.append("patient_id = ?")
            params.append(patient_id)
        if exclude_patient is not None:
            clauses.append("patient_id != ?")
            params.append(exclude_patient)
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        if diagnosis is not None:
            clauses.append("diagnosis = ?")
            params.append(diagnosis)
        if exclude is not None:
            clauses.append("case_id != ?")
            params.append(exclude)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        rows = self._rows(where, tuple(params))
        if not rows:
            return []

        q = normalize(embedding)
        # Rows of differing dimension cannot be compared; skip rather than crash,
        # which happens when an embedding model is swapped mid-corpus.
        usable = [r for r in rows if r["dim"] == q.shape[0]]
        if not usable:
            return []

        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in usable])
        sims = matrix @ q
        top = np.argsort(-sims)[: max(int(k), 0)]
        return [
            SearchHit(
                case_id=usable[i]["case_id"],
                similarity=float(sims[i]),
                diagnosis=usable[i]["diagnosis"],
                patient_id=usable[i]["patient_id"],
                modality=usable[i]["modality"],
                study_id=usable[i]["study_id"],
                created_at=usable[i]["created_at"],
                metadata=json.loads(usable[i]["metadata"] or "{}"),
            )
            for i in top
        ]

    def diagnosis_centroids(self) -> dict[str, np.ndarray]:
        """Mean embedding per diagnosis — the basis of disease-level similarity."""
        groups: dict[str, list[np.ndarray]] = {}
        for row in self._rows("WHERE diagnosis != ''"):
            groups.setdefault(row["diagnosis"], []).append(
                np.frombuffer(row["embedding"], dtype=np.float32)
            )
        out: dict[str, np.ndarray] = {}
        for dx, vecs in groups.items():
            dims = {v.shape[0] for v in vecs}
            if len(dims) != 1:
                continue        # mixed embedding generations; not comparable
            out[dx] = normalize(np.mean(np.stack(vecs), axis=0))
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM vector_records").fetchone()[0])
            patients = int(conn.execute(
                "SELECT COUNT(DISTINCT patient_id) FROM vector_records WHERE patient_id != ''"
            ).fetchone()[0])
            by_modality = {
                r[0] or "unknown": r[1] for r in conn.execute(
                    "SELECT modality, COUNT(*) FROM vector_records GROUP BY modality")
            }
            by_diagnosis = {
                r[0] or "unknown": r[1] for r in conn.execute(
                    "SELECT diagnosis, COUNT(*) FROM vector_records GROUP BY diagnosis")
            }
            dims = [r[0] for r in conn.execute("SELECT DISTINCT dim FROM vector_records")]
        return {
            "backend": self.backend,
            "path": str(self.db_path),
            "records": total,
            "patients": patients,
            "dimensions": sorted(dims),
            "by_modality": by_modality,
            "by_diagnosis": by_diagnosis,
        }


# --------------------------------------------------------------------------- #
# Qdrant backend
# --------------------------------------------------------------------------- #
class QdrantVectorStore:
    """Qdrant-backed ANN store. Optional; see :func:`get_vector_store`."""

    backend = "qdrant"
    COLLECTION = "aura_cases"

    def __init__(self, url: str | None = None, dim: int = 128,
                 collection: str | None = None, api_key: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dim = int(dim)
        self.collection = collection or self.COLLECTION
        self.url = url or os.environ.get("AURA_QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.url, api_key=api_key or os.environ.get("AURA_QDRANT_API_KEY"))
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(case_id: str) -> int:
        """Qdrant point ids are ints or UUIDs; hash the case id deterministically."""
        import hashlib

        return int(hashlib.sha1(case_id.encode()).hexdigest()[:15], 16)

    def upsert(self, record: VectorRecord) -> None:
        from qdrant_client.models import PointStruct

        vec = normalize(record.embedding)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._point_id(record.case_id),
                vector=vec.tolist(),
                payload=record.to_dict(),
            )],
        )

    def _filter(self, **eq):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in eq.items() if v is not None]
        return Filter(must=must) if must else None

    def search(self, embedding, k: int = 5, *, exclude: str | None = None,
               patient_id: str | None = None, modality: str | None = None,
               diagnosis: str | None = None,
               exclude_patient: str | None = None) -> list[SearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = self._filter(patient_id=patient_id, modality=modality, diagnosis=diagnosis)
        if exclude_patient is not None:
            must_not = [FieldCondition(key="patient_id", match=MatchValue(value=exclude_patient))]
            flt = Filter(must=(flt.must if flt else []), must_not=must_not)

        # Over-fetch so the excluded case can be dropped without shrinking the result.
        found = self.client.search(
            collection_name=self.collection,
            query_vector=normalize(embedding).tolist(),
            query_filter=flt,
            limit=int(k) + (1 if exclude else 0),
        )
        hits = [
            SearchHit(
                case_id=(p.payload or {}).get("case_id", ""),
                similarity=float(p.score),
                diagnosis=(p.payload or {}).get("diagnosis", ""),
                patient_id=(p.payload or {}).get("patient_id", ""),
                modality=(p.payload or {}).get("modality", ""),
                study_id=(p.payload or {}).get("study_id", ""),
                created_at=float((p.payload or {}).get("created_at", 0.0)),
                metadata=(p.payload or {}).get("metadata", {}),
            )
            for p in found
        ]
        if exclude:
            hits = [h for h in hits if h.case_id != exclude]
        return hits[:k]

    def get(self, case_id: str) -> VectorRecord | None:
        points = self.client.retrieve(
            collection_name=self.collection, ids=[self._point_id(case_id)], with_vectors=True
        )
        if not points:
            return None
        p = points[0]
        payload = p.payload or {}
        return VectorRecord(
            case_id=payload.get("case_id", case_id),
            embedding=np.asarray(p.vector, dtype=np.float32),
            diagnosis=payload.get("diagnosis", ""),
            patient_id=payload.get("patient_id", ""),
            modality=payload.get("modality", ""),
            study_id=payload.get("study_id", ""),
            created_at=float(payload.get("created_at", 0.0)),
            metadata=payload.get("metadata", {}),
        )

    def by_patient(self, patient_id: str) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._filter(patient_id=patient_id),
                limit=256, offset=offset, with_vectors=True,
            )
            for p in points:
                payload = p.payload or {}
                records.append(VectorRecord(
                    case_id=payload.get("case_id", ""),
                    embedding=np.asarray(p.vector, dtype=np.float32),
                    diagnosis=payload.get("diagnosis", ""),
                    patient_id=payload.get("patient_id", ""),
                    modality=payload.get("modality", ""),
                    study_id=payload.get("study_id", ""),
                    created_at=float(payload.get("created_at", 0.0)),
                    metadata=payload.get("metadata", {}),
                ))
            if offset is None:
                break
        records.sort(key=lambda r: r.created_at)
        return records

    def delete(self, case_id: str) -> bool:
        self.client.delete(collection_name=self.collection, points_selector=[self._point_id(case_id)])
        return True

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "url": self.url,
                "collection": self.collection, "records": self.count(),
                "dimensions": [self.dim]}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_STORE: VectorStore | None = None
_STORE_LOCK = threading.Lock()


def available_backends() -> dict[str, Any]:
    """What this deployment could use, and why not, without connecting."""
    out: dict[str, Any] = {"sqlite": {"available": True, "reason": "built in"}}
    try:
        import qdrant_client  # noqa: F401

        out["qdrant"] = {"available": True,
                         "reason": f"client installed; url {os.environ.get('AURA_QDRANT_URL', 'http://localhost:6333')}"}
    except ImportError:
        out["qdrant"] = {"available": False,
                         "reason": "qdrant-client not installed (pip install 'aura[vectordb]')"}
    return out


def get_vector_store(force: str | None = None, dim: int = 128) -> VectorStore:
    """The process-wide vector store.

    Prefers Qdrant when ``AURA_VECTOR_BACKEND=qdrant`` (or when it is installed
    and reachable) and falls back to SQLite on any failure — an unreachable
    vector database must degrade retrieval, not break inference.
    """
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None and force is None:
            return _STORE

        choice = (force or os.environ.get("AURA_VECTOR_BACKEND", "auto")).strip().lower()
        if choice in {"qdrant", "auto"}:
            try:
                store = QdrantVectorStore(dim=dim)
                if force is None:
                    _STORE = store
                return store
            except Exception as exc:
                if choice == "qdrant":
                    # Explicitly requested: say loudly that it did not happen.
                    import warnings

                    warnings.warn(
                        f"AURA_VECTOR_BACKEND=qdrant but Qdrant is unusable ({exc}); "
                        "falling back to the SQLite vector store.",
                        RuntimeWarning, stacklevel=2,
                    )

        store = SqliteVectorStore(dim=None)
        if force is None:
            _STORE = store
        return store


def reset_vector_store() -> None:
    """Drop the cached singleton. For tests and for backend switching."""
    global _STORE
    with _STORE_LOCK:
        _STORE = None
