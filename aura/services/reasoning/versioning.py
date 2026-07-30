"""Versioned Evidence Graph — incremental history over destructive updates.

When new findings, lab results, or symptom updates arrive, the reasoner
generates a new version (v1 → v2 → v3) instead of mutating the existing graph.
This enables:
  * Historical trace for the DRP to evaluate S_consistency over time
  * Audit-grade provenance of how evidence evolved
  * Safe re-computation without losing prior state
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from aura.schemas.contracts import EvidenceEdge, EvidenceGraph, EvidenceNode


@dataclass
class GraphVersion:
    """One snapshot of the evidence graph at a point in time."""
    version: int
    graph: EvidenceGraph
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    added_nodes: list[str] = field(default_factory=list)
    added_edges: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    source: str = ""  # what triggered this version (e.g. "lab_results", "symptom_update")


class VersionedEvidenceGraph:
    """Maintains an incremental version history of the evidence graph.

    Each ``update`` call creates a new version rather than mutating the
    existing graph.  The full history is available for DRP consistency
    evaluation and audit provenance.
    """

    def __init__(self, initial: EvidenceGraph | None = None):
        self._versions: list[GraphVersion] = []
        self._current: EvidenceGraph = initial or EvidenceGraph()
        if initial is not None:
            self._versions.append(GraphVersion(
                version=1,
                graph=initial.model_copy(deep=True),
                source="initial",
            ))

    @property
    def current(self) -> EvidenceGraph:
        return self._current

    @property
    def version_number(self) -> int:
        return len(self._versions)

    @property
    def versions(self) -> list[GraphVersion]:
        return list(self._versions)

    def update(self, *, added_nodes: list[EvidenceNode] | None = None,
               added_edges: list[EvidenceEdge] | None = None,
               removed_node_ids: list[str] | None = None,
               source: str = "update") -> GraphVersion:
        """Apply changes and create a new version snapshot.

        Returns the new ``GraphVersion``.
        """
        v_num = len(self._versions) + 1
        added_n = added_nodes or []
        added_e = added_edges or []
        removed = removed_node_ids or []

        # Apply additions to the current graph
        for node in added_n:
            self._current.add_node(node)
        for edge in added_e:
            self._current.add_edge(edge)
        for nid in removed:
            self._current.nodes.pop(nid, None)
            self._current.edges = [
                e for e in self._current.edges
                if e.source_id != nid and e.target_id != nid
            ]

        # Snapshot the new state
        snapshot = self._current.model_copy(deep=True)
        version = GraphVersion(
            version=v_num,
            graph=snapshot,
            added_nodes=[n.id for n in added_n],
            added_edges=[f"{e.source_id}->{e.target_id}" for e in added_e],
            removed_nodes=removed,
            source=source,
        )
        self._versions.append(version)
        return version

    def consistency_trend(self) -> list[tuple[int, float]]:
        """Compute the support/refute ratio for each version.

        Returns a list of ``(version_number, support_ratio)`` tuples.
        A ratio closer to 1.0 means more supporting than refuting edges,
        which indicates improving consistency over time.
        """
        trend: list[tuple[int, float]] = []
        for v in self._versions:
            g = v.graph
            supporting = sum(1 for e in g.edges if e.relation.value in ("supports", "mediates"))
            refuting = sum(1 for e in g.edges if e.relation.value in ("refutes", "contradicts"))
            total = supporting + refuting
            ratio = supporting / total if total > 0 else 0.5
            trend.append((v.version, ratio))
        return trend

    def diff(self, v1: int, v2: int) -> dict:
        """Compare two versions and return the differences.

        Version numbers are 1-indexed.
        """
        if v1 < 1 or v2 < 1 or v1 > len(self._versions) or v2 > len(self._versions):
            raise ValueError(f"Version must be between 1 and {len(self._versions)}")
        g1 = self._versions[v1 - 1].graph
        g2 = self._versions[v2 - 1].graph
        nodes_added = set(g2.nodes.keys()) - set(g1.nodes.keys())
        nodes_removed = set(g1.nodes.keys()) - set(g2.nodes.keys())
        return {
            "v1": v1, "v2": v2,
            "nodes_added": sorted(nodes_added),
            "nodes_removed": sorted(nodes_removed),
            "v1_node_count": len(g1.nodes),
            "v2_node_count": len(g2.nodes),
            "v1_edge_count": len(g1.edges),
            "v2_edge_count": len(g2.edges),
        }

    def get_version(self, version: int) -> GraphVersion | None:
        """Retrieve a specific version by number (1-indexed)."""
        if 1 <= version <= len(self._versions):
            return self._versions[version - 1]
        return None
