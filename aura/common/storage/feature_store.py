"""SQLite Feature Store — cache intermediate embeddings for downstream analytics.

Stores per-study embeddings (vision embeddings, fused embeddings, uncertainty
vectors) indexed by study hash.  Enables:
  * Batch analytics without re-running inference
  * Semantic similarity search over historical cases
  * DRP/consensus evaluation across patient cohorts
  * Ablation studies and reproducibility benchmarks
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from aura.common.config import ARTIFACTS


class _Base(DeclarativeBase):
    pass


class FeatureRow(_Base):
    """One cached embedding record."""
    __tablename__ = "features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_hash: Mapped[str] = mapped_column(String, index=True, unique=True)
    study_id: Mapped[str] = mapped_column(String, index=True)
    case_id: Mapped[str] = mapped_column(String, default="")
    # Embedding data
    vision_embedding: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    fused_embedding: Mapped[str] = mapped_column(Text, default="[]")   # JSON array
    uncertainty_vector: Mapped[str] = mapped_column(Text, default="[]")
    evidence_vector: Mapped[str] = mapped_column(Text, default="[]")
    # Metadata
    modality: Mapped[str] = mapped_column(String, default="")
    diagnosis: Mapped[str] = mapped_column(String, default="")
    top_probability: Mapped[float] = mapped_column(Float, default=0.0)
    # Timing
    vision_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    fusion_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    # Hash for deduplication
    embedding_hash: Mapped[str] = mapped_column(String, default="")
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class FeatureStore:
    """SQLite-backed feature store for caching intermediate embeddings.

    Deduplicates by (study_hash, embedding_hash) so the same image never
    gets cached twice.
    """

    def __init__(self, filename: str = "features.db"):
        self.path = ARTIFACTS / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.path}", future=True)
        _Base.metadata.create_all(self.engine)

    @staticmethod
    def compute_study_hash(image_bytes: bytes, study_id: str = "") -> str:
        """Compute a deterministic hash for a study from its raw bytes."""
        h = hashlib.sha256(image_bytes)
        if study_id:
            h.update(study_id.encode())
        return h.hexdigest()

    def _embeddings_hash(self, vision_emb: list, fused_emb: list,
                         uncertainty: list) -> str:
        """Hash the embedding vectors for deduplication."""
        parts = json.dumps(vision_emb) + json.dumps(fused_emb) + json.dumps(uncertainty)
        return hashlib.sha256(parts.encode()).hexdigest()

    def store(self, study_hash: str, *,
              study_id: str = "", case_id: str = "",
              vision_embedding: np.ndarray | list | None = None,
              fused_embedding: np.ndarray | list | None = None,
              uncertainty_vector: np.ndarray | list | None = None,
              evidence_vector: np.ndarray | list | None = None,
              modality: str = "", diagnosis: str = "",
              top_probability: float = 0.0,
              vision_latency_ms: float = 0.0,
              fusion_latency_ms: float = 0.0,
              total_latency_ms: float = 0.0,
              extra: dict | None = None) -> dict:
        """Cache embeddings for a study.  Returns the stored record."""
        def to_list(x):
            if x is None:
                return []
            if isinstance(x, np.ndarray):
                return x.flatten().tolist()
            return list(x)

        vision_list = to_list(vision_embedding)
        fused_list = to_list(fused_embedding)
        uncertainty_list = to_list(uncertainty_vector)
        evidence_list = to_list(evidence_vector)
        emb_hash = self._embeddings_hash(vision_list, fused_list, uncertainty_list)

        with Session(self.engine) as ses:
            row = ses.execute(
                select(FeatureRow).where(FeatureRow.study_hash == study_hash)
            ).scalars().first()

            if row is None:
                row = FeatureRow(study_hash=study_hash)
                ses.add(row)

            row.study_id = study_id
            row.case_id = case_id
            row.vision_embedding = json.dumps(vision_list)
            row.fused_embedding = json.dumps(fused_list)
            row.uncertainty_vector = json.dumps(uncertainty_list)
            row.evidence_vector = json.dumps(evidence_list)
            row.modality = modality
            row.diagnosis = diagnosis
            row.top_probability = top_probability
            row.vision_latency_ms = vision_latency_ms
            row.fusion_latency_ms = fusion_latency_ms
            row.total_latency_ms = total_latency_ms
            row.embedding_hash = emb_hash
            row.extra = extra or {}
            row.created_at = datetime.now(timezone.utc)
            ses.commit()

        return {
            "study_hash": study_hash,
            "study_id": study_id,
            "case_id": case_id,
            "embedding_hash": emb_hash,
            "stored": True,
        }

    def get(self, study_hash: str) -> dict | None:
        """Retrieve cached features for a study hash."""
        with Session(self.engine) as ses:
            row = ses.execute(
                select(FeatureRow).where(FeatureRow.study_hash == study_hash)
            ).scalars().first()
            if row is None:
                return None
            return self._row_to_dict(row)

    def get_by_case(self, case_id: str) -> dict | None:
        """Retrieve cached features by case_id."""
        with Session(self.engine) as ses:
            row = ses.execute(
                select(FeatureRow).where(FeatureRow.case_id == case_id)
            ).scalars().first()
            if row is None:
                return None
            return self._row_to_dict(row)

    def list_all(self, limit: int = 500) -> list[dict]:
        """List all cached feature records (metadata only, embeddings omitted)."""
        with Session(self.engine) as ses:
            rows = ses.execute(
                select(FeatureRow).order_by(FeatureRow.created_at.desc()).limit(limit)
            ).scalars().all()
            return [{
                "study_hash": r.study_hash,
                "study_id": r.study_id,
                "case_id": r.case_id,
                "modality": r.modality,
                "diagnosis": r.diagnosis,
                "top_probability": r.top_probability,
                "vision_latency_ms": r.vision_latency_ms,
                "fusion_latency_ms": r.fusion_latency_ms,
                "total_latency_ms": r.total_latency_ms,
                "created_at": r.created_at.isoformat(),
            } for r in rows]

    def count(self) -> int:
        with Session(self.engine) as ses:
            return ses.query(FeatureRow).count()

    def search_by_diagnosis(self, diagnosis: str, limit: int = 100) -> list[dict]:
        """Find all cached embeddings for a specific diagnosis."""
        with Session(self.engine) as ses:
            rows = ses.execute(
                select(FeatureRow)
                .where(FeatureRow.diagnosis == diagnosis)
                .order_by(FeatureRow.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def similarity_search(self, query_embedding: list[float],
                          modality: str | None = None,
                          top_k: int = 10) -> list[dict]:
        """Find the most similar cached embeddings using cosine similarity.

        For large-scale use, replace with FAISS or Annoy; this brute-force
        approach is adequate for P0 (< 10k cached cases).
        """
        with Session(self.engine) as ses:
            stmt = select(FeatureRow)
            if modality:
                stmt = stmt.where(FeatureRow.modality == modality)
            rows = ses.execute(stmt).scalars().all()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-12:
            return []

        scored: list[tuple[float, dict]] = []
        for row in rows:
            fused = np.array(json.loads(row.fused_embedding), dtype=float)
            if len(fused) != len(query_vec):
                continue
            fuse_norm = np.linalg.norm(fused)
            if fuse_norm < 1e-12:
                continue
            cosine = float(np.dot(query_vec, fused) / (query_norm * fuse_norm))
            scored.append((cosine, self._row_to_dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, rec in scored[:top_k]:
            rec["similarity"] = score
            results.append(rec)
        return results

    def delete(self, study_hash: str) -> bool:
        with Session(self.engine) as ses:
            row = ses.execute(
                select(FeatureRow).where(FeatureRow.study_hash == study_hash)
            ).scalars().first()
            if row is None:
                return False
            ses.delete(row)
            ses.commit()
            return True

    def _row_to_dict(self, row: FeatureRow) -> dict:
        return {
            "study_hash": row.study_hash,
            "study_id": row.study_id,
            "case_id": row.case_id,
            "vision_embedding": json.loads(row.vision_embedding),
            "fused_embedding": json.loads(row.fused_embedding),
            "uncertainty_vector": json.loads(row.uncertainty_vector),
            "evidence_vector": json.loads(row.evidence_vector),
            "modality": row.modality,
            "diagnosis": row.diagnosis,
            "top_probability": row.top_probability,
            "vision_latency_ms": row.vision_latency_ms,
            "fusion_latency_ms": row.fusion_latency_ms,
            "total_latency_ms": row.total_latency_ms,
            "embedding_hash": row.embedding_hash,
            "extra": row.extra,
            "created_at": row.created_at.isoformat(),
        }
