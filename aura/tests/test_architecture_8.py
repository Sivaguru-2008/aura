"""Tests for the 8 new architectural improvements.

Covers:
  1. Pluggable Modality Architecture (plugins, registry, ThoraxPlugin, NeuroPlugin)
  2. Decision State Machine (PipelineFSM, transitions, constraints)
  3. Versioned Evidence Graph (versioning, diff, consistency trend)
  4. Explainability API (endpoint contracts, lazy-loading, method listing)
  5. Standardized Benchmarking Framework (MetricsCard, BenchmarkRunner)
  6. Model Registry with SHA-256 checksums (register, verify, artifact listing)
  7. Async Event Bus pub-sub (typed events, wildcards, chaining, history)
  8. SQLite Feature Store (put, get, similarity search, deduplication)
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _run(coro):
    """Run an async coroutine in a fresh event loop (for tests without pytest-asyncio)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Pluggable Modality Architecture
# ═══════════════════════════════════════════════════════════════════════

class TestPluggableModalityArchitecture:
    def test_base_plugin_is_abstract(self):
        from aura.gateway.adapters.base_plugin import BaseModalityPlugin
        with pytest.raises(TypeError):
            BaseModalityPlugin("test", "test")

    def test_base_plugin_required_methods(self):
        from aura.gateway.adapters.base_plugin import BaseModalityPlugin
        assert hasattr(BaseModalityPlugin, "create_inspector")
        assert hasattr(BaseModalityPlugin, "create_standardizer")
        assert hasattr(BaseModalityPlugin, "create_engine")
        assert hasattr(BaseModalityPlugin, "validate_signature")
        assert hasattr(BaseModalityPlugin, "pipeline_hooks")

    def test_pixel_signature_dataclass(self):
        from aura.gateway.adapters.base_plugin import PixelSignature
        sig = PixelSignature(
            mime_types=("image/", "application/dicom"),
            extensions=(".png", ".jpg", ".jpeg", ".dcm"),
            min_dims=2,
            max_dims=3,
        )
        assert sig.mime_types == ("image/", "application/dicom")
        assert ".dcm" in sig.extensions
        assert sig.min_dims == 2

    def test_pixel_signature_defaults(self):
        from aura.gateway.adapters.base_plugin import PixelSignature
        sig = PixelSignature()
        assert sig.mime_types == ()
        assert sig.extensions == ()
        assert sig.magic_bytes == b""
        assert sig.min_dims == 2
        assert sig.max_dims == 3

    def test_plugin_registry_functions_exist(self):
        from aura.gateway.adapters.plugin_registry import (
            register_plugin, get_plugin, registered_plugins,
            plugin_modalities, resolve_plugin,
        )
        assert callable(register_plugin)
        assert callable(get_plugin)
        assert callable(registered_plugins)
        assert callable(plugin_modalities)
        assert callable(resolve_plugin)

    def test_thorax_plugin_implements_interface(self):
        from aura.gateway.adapters.thorax_plugin import ThoraxPlugin
        plugin = ThoraxPlugin()
        assert plugin.modality == "chest_xray"
        assert plugin.display_name == "Chest X-ray"
        assert ".dcm" in plugin.pixel_signature.extensions
        assert plugin.pixel_signature.min_dims == 2
        assert "intake_gate" in plugin.pipeline_hooks()

    def test_neuro_plugin_implements_interface(self):
        from aura.gateway.adapters.neuro_plugin import NeuroPlugin
        plugin = NeuroPlugin()
        assert plugin.modality == "brain_mri"
        assert plugin.display_name == "Brain MRI"
        assert ".nii" in plugin.pixel_signature.extensions
        assert plugin.pixel_signature.max_dims == 4
        assert "mri_gate" in plugin.pipeline_hooks()

    def test_plugin_registry_register_and_retrieve(self):
        from aura.gateway.adapters.plugin_registry import (
            register_plugin, get_plugin, plugin_modalities,
            _plugin_registry,
        )
        from aura.gateway.adapters.base_plugin import BaseModalityPlugin, PixelSignature

        class DummyPlugin(BaseModalityPlugin):
            modality = "dummy"
            display_name = "Dummy"
            def create_inspector(self): return None
            def create_standardizer(self): return None
            def create_engine(self): return None

        register_plugin(DummyPlugin())
        assert get_plugin("dummy") is not None
        assert "dummy" in plugin_modalities()
        # Cleanup
        _plugin_registry.pop("dummy", None)

    def test_plugin_validate_signature(self):
        from aura.gateway.adapters.thorax_plugin import ThoraxPlugin
        plugin = ThoraxPlugin()
        assert plugin.validate_signature(filename="chest.png") is True
        assert plugin.validate_signature(filename="chest.dcm") is True
        assert plugin.validate_signature(filename="brain.nii") is False

    def test_plugin_validate_signature_content_type(self):
        from aura.gateway.adapters.thorax_plugin import ThoraxPlugin
        plugin = ThoraxPlugin()
        assert plugin.validate_signature(content_type="image/png") is True
        assert plugin.validate_signature(content_type="application/nifti") is False

    def test_auto_discover_plugins(self):
        from aura.gateway.adapters.plugin_registry import _auto_discover_plugins, _plugin_registry
        _auto_discover_plugins()
        assert "chest_xray" in _plugin_registry
        assert "brain_mri" in _plugin_registry

    def test_resolve_plugin(self):
        from aura.gateway.adapters.plugin_registry import resolve_plugin, _auto_discover_plugins
        _auto_discover_plugins()
        plugin = resolve_plugin(filename="scan.dcm")
        assert plugin is not None
        assert plugin.modality in ("chest_xray", "brain_mri")

    def test_resolve_plugin_no_match(self):
        from aura.gateway.adapters.plugin_registry import resolve_plugin
        plugin = resolve_plugin(filename="data.txt", content_type="text/plain")
        assert plugin is None

    def test_registered_plugins_returns_copy(self):
        from aura.gateway.adapters.plugin_registry import registered_plugins, _plugin_registry
        _plugin_registry.clear()
        r1 = registered_plugins()
        r1["fake"] = MagicMock()
        assert "fake" not in _plugin_registry


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Decision State Machine
# ═══════════════════════════════════════════════════════════════════════

class TestDecisionStateMachine:
    def _fsm(self):
        from aura.common.state.fsm import PipelineFSM
        return PipelineFSM()

    def test_fsm_starts_at_input(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        assert fsm.state == PipelineState.INPUT

    def test_valid_transition_input_to_safety_check(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        assert fsm.state == PipelineState.SAFETY_CHECK

    def test_invalid_transition_raises(self):
        from aura.common.state.fsm import PipelineState, TransitionViolation
        fsm = self._fsm()
        with pytest.raises(TransitionViolation):
            fsm.transition(PipelineState.REPORT)  # skipping steps

    def test_full_happy_path(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        fsm.transition(PipelineState.READY)
        fsm.transition(PipelineState.REPORT)
        assert fsm.state == PipelineState.REPORT
        assert fsm.is_terminal

    def test_safety_veto_blocks_forward_progress(self):
        from aura.common.state.fsm import PipelineState, SafetyVerdict, TransitionViolation
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.set_safety_verdict(SafetyVerdict.UNSAFE)
        with pytest.raises(TransitionViolation):
            fsm.transition(PipelineState.EVIDENCE_COLLECTION)

    def test_safety_unsafe_can_still_abstain(self):
        from aura.common.state.fsm import PipelineState, SafetyVerdict
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.set_safety_verdict(SafetyVerdict.UNSAFE)
        fsm.transition(PipelineState.ABSTAINED)
        assert fsm.state == PipelineState.ABSTAINED
        assert fsm.is_terminal

    def test_safety_unsafe_can_fail(self):
        from aura.common.state.fsm import PipelineState, SafetyVerdict
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.set_safety_verdict(SafetyVerdict.UNSAFE)
        fsm.transition(PipelineState.FAILED)
        assert fsm.state == PipelineState.FAILED

    def test_drp_needs_evidence_blocks_ready(self):
        from aura.common.state.fsm import PipelineState, ReadinessVerdict, TransitionViolation
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        fsm.set_readiness_verdict(ReadinessVerdict.NEEDS_ADDITIONAL_EVIDENCE)
        with pytest.raises(TransitionViolation):
            fsm.transition(PipelineState.READY)

    def test_not_ready_does_not_block_ready(self):
        from aura.common.state.fsm import PipelineState, ReadinessVerdict
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        # NOT_READY does not block — only NEEDS_ADDITIONAL_EVIDENCE blocks
        fsm.set_readiness_verdict(ReadinessVerdict.NOT_READY)
        fsm.transition(PipelineState.READY)
        assert fsm.state == PipelineState.READY

    def test_clinician_override_bypasses_drp_block(self):
        from aura.common.state.fsm import PipelineState, ReadinessVerdict
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        fsm.set_readiness_verdict(ReadinessVerdict.NEEDS_ADDITIONAL_EVIDENCE)
        fsm.set_clinician_bypass(True)
        fsm.transition(PipelineState.READY)
        assert fsm.state == PipelineState.READY

    def test_ready_can_transition_to_report(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        fsm.transition(PipelineState.READY)
        fsm.transition(PipelineState.REPORT)
        assert fsm.state == PipelineState.REPORT

    def test_ready_can_abstain(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        fsm.transition(PipelineState.READY)
        fsm.transition(PipelineState.ABSTAINED)
        assert fsm.state == PipelineState.ABSTAINED

    def test_terminal_states_cannot_transition(self):
        from aura.common.state.fsm import PipelineState, TransitionViolation
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.ABSTAINED)
        with pytest.raises(TransitionViolation):
            fsm.transition(PipelineState.SAFETY_CHECK)

    def test_can_transition_check(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        assert fsm.can_transition(PipelineState.SAFETY_CHECK) is True
        assert fsm.can_transition(PipelineState.REPORT) is False

    def test_snapshot(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        snap = fsm.snapshot()
        assert snap["state"] == "evidence_collection"
        assert len(snap["transitions"]) == 2
        assert snap["transitions"][0]["from"] == "input"
        assert snap["transitions"][0]["to"] == "safety_check"

    def test_transition_history_tracked(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        assert len(fsm.history) == 2

    def test_cannot_skip_to_ready(self):
        from aura.common.state.fsm import PipelineState, TransitionViolation
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        with pytest.raises(TransitionViolation):
            fsm.transition(PipelineState.READY)

    def test_any_state_can_fail(self):
        from aura.common.state.fsm import PipelineState
        # INPUT → FAILED is always valid
        fsm = self._fsm()
        fsm.transition(PipelineState.FAILED)
        assert fsm.state == PipelineState.FAILED

    def test_constraint_violations_recorded(self):
        from aura.common.state.fsm import PipelineState, SafetyVerdict
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.set_safety_verdict(SafetyVerdict.UNSAFE)
        try:
            fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        except Exception:
            pass
        assert len(fsm.constraint_violations) > 0

    def test_reset(self):
        from aura.common.state.fsm import PipelineState
        fsm = self._fsm()
        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.reset()
        assert fsm.state == PipelineState.INPUT
        assert len(fsm.history) == 0


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Versioned Evidence Graph
# ═══════════════════════════════════════════════════════════════════════

class TestVersionedEvidenceGraph:
    def test_initial_version_is_0_without_graph(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        veg = VersionedEvidenceGraph()
        assert veg.version_number == 0
        assert veg.current is not None

    def test_initial_version_with_graph(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        from aura.schemas.contracts import EvidenceGraph
        g = EvidenceGraph()
        veg = VersionedEvidenceGraph(initial=g)
        assert veg.version_number == 1

    def test_update_creates_new_version(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        veg = VersionedEvidenceGraph()
        v1 = veg.update(source="lab_results")
        assert v1.version == 1
        v2 = veg.update(source="symptom_update")
        assert v2.version == 2
        assert veg.version_number == 2

    def test_update_with_nodes(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        from aura.schemas.contracts import EvidenceNode, EvidenceKind
        veg = VersionedEvidenceGraph()
        node = EvidenceNode(
            id="n1", kind=EvidenceKind.STRUCTURED_PRIOR,
            label="WBC", value=0.8, modality="labs",
        )
        v1 = veg.update(added_nodes=[node], source="labs")
        assert v1.version == 1
        assert "n1" in veg.current.nodes

    def test_update_with_edges(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        from aura.schemas.contracts import EvidenceNode, EvidenceEdge, EvidenceKind, RelationType
        veg = VersionedEvidenceGraph()
        n1 = EvidenceNode(id="n1", kind=EvidenceKind.STRUCTURED_PRIOR, label="WBC", modality="labs")
        n2 = EvidenceNode(id="n2", kind=EvidenceKind.CLINICIAN_INPUT, label="Pneumonia", modality="hypothesis")
        edge = EvidenceEdge(source_id="n1", target_id="n2", relation=RelationType.SUPPORTS)
        v1 = veg.update(added_nodes=[n1, n2], added_edges=[edge])
        assert v1.version == 1
        assert len(veg.current.edges) == 1

    def test_diff_between_versions(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        from aura.schemas.contracts import EvidenceNode, EvidenceKind
        veg = VersionedEvidenceGraph()
        node1 = EvidenceNode(id="n1", kind=EvidenceKind.STRUCTURED_PRIOR, modality="labs")
        veg.update(added_nodes=[node1], source="labs")
        node2 = EvidenceNode(id="n2", kind=EvidenceKind.CLINICIAN_INPUT, modality="symptoms")
        veg.update(added_nodes=[node2], source="update")
        diff = veg.diff(1, 2)
        assert diff["v1"] == 1
        assert diff["v2"] == 2
        assert "n2" in diff["nodes_added"]
        assert diff["v1_node_count"] == 1
        assert diff["v2_node_count"] == 2

    def test_diff_invalid_version_raises(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        veg = VersionedEvidenceGraph()
        with pytest.raises(ValueError):
            veg.diff(0, 1)

    def test_consistency_trend(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        from aura.schemas.contracts import EvidenceNode, EvidenceEdge, EvidenceKind, RelationType
        veg = VersionedEvidenceGraph()
        n1 = EvidenceNode(id="n1", kind=EvidenceKind.STRUCTURED_PRIOR, modality="labs")
        n2 = EvidenceNode(id="n2", kind=EvidenceKind.CLINICIAN_INPUT, modality="hypothesis")
        edge = EvidenceEdge(source_id="n1", target_id="n2", relation=RelationType.SUPPORTS)
        veg.update(added_nodes=[n1, n2], added_edges=[edge], source="v1")
        # Add a contradicting edge in v2
        n3 = EvidenceNode(id="n3", kind=EvidenceKind.ABSENT_EVIDENCE, modality="labs")
        edge2 = EvidenceEdge(source_id="n3", target_id="n2", relation=RelationType.CONTRADICTS)
        veg.update(added_nodes=[n3], added_edges=[edge2], source="v2")
        trend = veg.consistency_trend()
        assert len(trend) == 2
        assert trend[0][0] == 1
        assert trend[0][1] == 1.0  # v1: all supports → ratio = 1.0
        assert trend[1][0] == 2
        # v2: 1 support + 1 contradicts → 1/(1+1) = 0.5
        assert trend[1][1] == pytest.approx(0.5)

    def test_get_version(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        veg = VersionedEvidenceGraph()
        veg.update(source="test")
        v = veg.get_version(1)
        assert v is not None
        assert v.version == 1
        assert veg.get_version(99) is None

    def test_versions_list(self):
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        veg = VersionedEvidenceGraph()
        veg.update(source="v1")
        veg.update(source="v2")
        vs = veg.versions
        assert len(vs) == 2
        assert vs[0].source == "v1"
        assert vs[1].source == "v2"


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Explainability API (unit-level tests)
# ═══════════════════════════════════════════════════════════════════════

class TestExplainabilityAPI:
    def test_explain_module_importable(self):
        from aura.gateway.api.explain import router
        assert router is not None
        assert router.prefix == "/v1"
        assert router.tags == ["explain"]

    def test_init_explain_api_sets_references(self):
        from aura.gateway.api import explain
        explain.init_explain_api(store="s", vision_engine="v",
                                 fusion_engine="f", explain_engine="e")
        assert explain._store == "s"
        assert explain._vision_engine == "v"
        explain.init_explain_api()

    def test_explain_endpoint_returns_503_when_not_initialized(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from aura.gateway.api.explain import router
        from aura.gateway.api import explain
        explain.init_explain_api()
        app = FastAPI()
        app.include_router(router)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/explain/test-case-1")
            assert resp.status_code == 503


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Standardized Benchmarking Framework
# ═══════════════════════════════════════════════════════════════════════

class TestBenchmarkingFramework:
    def test_metrics_card_defaults(self):
        from bench.runner import MetricsCard
        card = MetricsCard(backend="test")
        assert card.backend == "test"
        assert card.accuracy == 0.0
        assert card.f1_macro == 0.0
        assert card.ece == 0.0

    def test_metrics_card_to_dict(self):
        from bench.runner import MetricsCard
        card = MetricsCard(backend="test", accuracy=0.95, f1_macro=0.93)
        d = card.to_dict()
        assert d["backend"] == "test"
        assert d["accuracy"] == 0.95

    def test_benchmark_runner_evaluate_backend(self):
        from bench.runner import BenchmarkRunner
        runner = BenchmarkRunner()
        card = runner.evaluate_backend("dummy")
        assert card.backend == "dummy"
        assert card.n_eval == 0

    def test_benchmark_runner_with_data(self):
        from bench.runner import BenchmarkRunner
        runner = BenchmarkRunner()
        np.random.seed(42)
        X = np.random.randn(50, 8)
        y = np.random.randint(0, 6, size=50)
        logits = np.random.randn(50, 6)
        card = runner.evaluate_backend("test", X=X, y=y, logits=logits)
        assert card.accuracy > 0.0
        assert card.f1_macro >= 0.0
        assert card.ece >= 0.0
        assert card.brier > 0.0
        assert card.nll > 0.0
        assert card.per_class_f1 is not None

    def test_ece_calculation_perfect(self):
        from bench.runner import BenchmarkRunner
        runner = BenchmarkRunner()
        probs = np.eye(5)
        y = np.arange(5)
        assert runner._ece(probs, y) == pytest.approx(0.0)

    def test_compare_identifies_best(self):
        from bench.runner import BenchmarkRunner, MetricsCard
        runner = BenchmarkRunner()
        c1 = MetricsCard(backend="a", accuracy=0.90, f1_macro=0.88)
        c2 = MetricsCard(backend="b", accuracy=0.95, f1_macro=0.92)
        result = runner.compare([c1, c2])
        assert result.best_backend == "b"

    def test_comparison_pairwise(self):
        from bench.runner import BenchmarkRunner, MetricsCard
        runner = BenchmarkRunner()
        c1 = MetricsCard(backend="a", accuracy=0.90, f1_macro=0.88, ece=0.05, nll=0.3)
        c2 = MetricsCard(backend="b", accuracy=0.95, f1_macro=0.92, ece=0.02, nll=0.1)
        result = runner.compare([c1, c2])
        key = "a_vs_b"
        assert key in result.comparison
        assert result.comparison[key]["accuracy_diff"] == pytest.approx(-0.05, abs=0.01)

    def test_benchmark_result_to_dict(self):
        from bench.runner import BenchmarkResult, MetricsCard
        result = BenchmarkResult(
            cards=[MetricsCard(backend="a", accuracy=0.9)],
            best_backend="a",
        )
        d = result.to_dict()
        assert len(d["cards"]) == 1
        assert d["best_backend"] == "a"


# ═══════════════════════════════════════════════════════════════════════
# Step 6 — Model Registry with SHA-256 checksums
# ═══════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_registry_list_empty(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            assert reg.list_versions() == []

    def test_register_checkpoint(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        checkpoint = tmp_path / "model.npz"
        checkpoint.write_bytes(b"fake model weights")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            record = reg.register_checkpoint("test_model", checkpoint, version="1.0")
            assert record["name"] == "test_model"
            assert record["version"] == "1.0"
            assert len(record["sha256"]) == 64

    def test_verify_checkpoint_valid(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        checkpoint = tmp_path / "model.npz"
        checkpoint.write_bytes(b"valid checkpoint data")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            reg.register_checkpoint("test", checkpoint)
            result = reg.verify_checkpoint("test")
            assert result["valid"] is True
            assert len(result["current_hash"]) == 64

    def test_verify_checkpoint_corrupted(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        checkpoint = tmp_path / "model.npz"
        checkpoint.write_bytes(b"original data")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            reg.register_checkpoint("test", checkpoint)
            checkpoint.write_bytes(b"corrupted data")
            result = reg.verify_checkpoint("test")
            assert result["valid"] is False

    def test_verify_checkpoint_not_registered(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            result = reg.verify_checkpoint("nonexistent")
            assert result["valid"] is False
            assert result["error"] == "not_registered"

    def test_verify_checkpoint_file_missing(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        checkpoint = tmp_path / "model.npz"
        checkpoint.write_bytes(b"data")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            reg.register_checkpoint("test", checkpoint)
            checkpoint.unlink()
            result = reg.verify_checkpoint("test")
            assert result["valid"] is False
            assert result["error"] == "file_missing"

    def test_verify_all(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        f1 = tmp_path / "a.npz"
        f1.write_bytes(b"aaa")
        f2 = tmp_path / "b.npz"
        f2.write_bytes(b"bbb")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            reg.register_checkpoint("a", f1)
            reg.register_checkpoint("b", f2)
            results = reg.verify_all()
            assert len(results) == 2
            assert all(r["valid"] for r in results)

    def test_sha256_bytes(self):
        from aura.services.models.registry import sha256_bytes
        h = sha256_bytes(b"hello world")
        assert len(h) == 64
        assert isinstance(h, str)

    def test_register_overwrites_existing(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        f = tmp_path / "model.npz"
        f.write_bytes(b"v1")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            reg.register_checkpoint("m", f, version="1.0")
            f.write_bytes(b"v2")
            record = reg.register_checkpoint("m", f, version="2.0")
            assert record["version"] == "2.0"
            assert record["previous_hash"] != ""

    def test_provenance_entry(self, tmp_path):
        from unittest.mock import patch
        from aura.services.models.registry import ModelRegistry
        f = tmp_path / "model.npz"
        f.write_bytes(b"data")
        with patch("aura.services.models.registry.ARTIFACTS", tmp_path):
            reg = ModelRegistry()
            reg.register_checkpoint("m", f)
            entry = reg.provenance_entry("m")
            assert entry is not None
            assert entry["type"] == "model_checkpoint"
            assert entry["verified"] is True


# ═══════════════════════════════════════════════════════════════════════
# Step 7 — Async Event Bus pub-sub
# ═══════════════════════════════════════════════════════════════════════

class TestEventBus:
    def test_basic_subscribe_publish(self):
        from aura.common.eventbus import EventBus, Event
        bus = EventBus()
        received = []
        async def handler(e: Event):
            received.append(e)
        bus.subscribe("test.topic", handler)
        _run(bus.publish("test.topic", foo="bar"))
        assert len(received) == 1
        assert received[0].topic == "test.topic"
        assert received[0].payload["foo"] == "bar"

    def test_typed_study_received_event(self):
        from aura.common.eventbus import EventBus, StudyReceivedEvent, STUDY_RECEIVED
        bus = EventBus()
        received = []
        async def handler(e):
            received.append(e)
        bus.subscribe(STUDY_RECEIVED, handler)
        event = StudyReceivedEvent(study_id="s1", case_id="c1")
        _run(bus.publish(STUDY_RECEIVED, event=event))
        assert received[0].study_id == "s1"

    def test_wildcard_single_segment(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        received = []
        async def handler(e):
            received.append(e.topic)
        bus.subscribe("study.*", handler)
        _run(bus.publish("study.received"))
        _run(bus.publish("study.processed"))
        _run(bus.publish("fusion.completed"))
        assert "study.received" in received
        assert "study.processed" in received
        assert "fusion.completed" not in received

    def test_wildcard_double_star(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        received = []
        async def handler(e):
            received.append(e.topic)
        bus.subscribe("**", handler)
        _run(bus.publish("any.topic.here"))
        assert "any.topic.here" in received

    def test_priority_ordering(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        order = []
        async def low_priority(e):
            order.append("low")
        async def high_priority(e):
            order.append("high")
        bus.subscribe("test", low_priority, priority=1)
        bus.subscribe("test", high_priority, priority=10)
        _run(bus.publish("test"))
        assert order == ["high", "low"]

    def test_once_subscription(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        count = [0]
        async def counter(e):
            count[0] += 1
        bus.subscribe("test", counter, once=True)
        _run(bus.publish("test"))
        _run(bus.publish("test"))
        assert count[0] == 1

    def test_event_history(self):
        from aura.common.eventbus import EventBus
        bus = EventBus(enable_history=True)
        async def noop(e): pass
        bus.subscribe("test", noop)
        _run(bus.publish("test"))
        _run(bus.publish("test"))
        h = bus.history()
        assert len(h) == 2
        assert h[0].event.topic == "test"

    def test_unsubscribe(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        async def noop(e): pass
        bus.subscribe("test", noop, tag="my_sub")
        assert bus.subscriber_count("test") == 1
        bus.unsubscribe("test", tag="my_sub")
        assert bus.subscriber_count("test") == 0

    def test_unsubscribe_all(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        async def noop(e): pass
        bus.subscribe("test", noop, tag="a")
        bus.subscribe("test", noop, tag="b")
        bus.unsubscribe("test")
        assert bus.subscriber_count("test") == 0

    def test_subscriber_count(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        async def noop(e): pass
        bus.subscribe("a", noop)
        bus.subscribe("b", noop)
        assert bus.subscriber_count() == 2
        assert bus.subscriber_count("a") == 1

    def test_topics(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        async def noop(e): pass
        bus.subscribe("x.y", noop)
        bus.subscribe("z", noop)
        assert bus.topics() == ["x.y", "z"]

    def test_handler_exception_does_not_break_others(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        ok = [False]
        async def bad_handler(e):
            raise RuntimeError("fail")
        async def good_handler(e):
            ok[0] = True
        bus.subscribe("test", bad_handler, priority=10)
        bus.subscribe("test", good_handler, priority=1)
        _run(bus.publish("test"))
        assert ok[0] is True
        h = bus.history()
        assert len(h) == 1
        assert len(h[0].errors) == 1

    def test_chain_depth_limit(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        call_count = [0]
        async def self_publisher(e):
            call_count[0] += 1
            await bus.publish("test.loop")
        bus.subscribe("test.loop", self_publisher)
        _run(bus.publish("test.loop"))
        assert call_count[0] <= 10  # depth limit

    def test_chain_cycle_detection(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        count = [0]
        async def handler_a(e):
            count[0] += 1
            if count[0] < 3:
                await bus.publish("topic_a")
        bus.subscribe("topic_a", handler_a)
        _run(bus.publish("topic_a"))
        assert count[0] <= 10

    def test_clear_history(self):
        from aura.common.eventbus import EventBus
        bus = EventBus()
        async def noop(e): pass
        bus.subscribe("test", noop)
        _run(bus.publish("test"))
        assert len(bus.history()) == 1
        bus.clear_history()
        assert len(bus.history()) == 0

    def test_all_canonical_topics(self):
        from aura.common.eventbus import (
            STUDY_RECEIVED, VISION_COMPLETED, FUSION_COMPLETED,
            SAFETY_CHECKED, REASONING_COMPLETED, CASE_READY,
            FEEDBACK_RECORDED, DRP_COMPUTED,
        )
        topics = [
            STUDY_RECEIVED, VISION_COMPLETED, FUSION_COMPLETED,
            SAFETY_CHECKED, REASONING_COMPLETED, CASE_READY,
            FEEDBACK_RECORDED, DRP_COMPUTED,
        ]
        assert len(topics) == 8
        # Verify they are unique dot-separated strings
        assert len(set(topics)) == 8
        for t in topics:
            assert "." in t


# ═══════════════════════════════════════════════════════════════════════
# Step 8 — SQLite Feature Store
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureStore:
    def _store(self, tmp_path):
        from aura.common.storage.feature_store import FeatureStore
        return FeatureStore(filename=str(tmp_path / "features.db"))

    def test_store_and_get(self, tmp_path):
        store = self._store(tmp_path)
        emb = np.array([1.0, 2.0, 3.0])
        result = store.store("hash1", study_id="s1", case_id="c1",
                             vision_embedding=emb, fused_embedding=emb,
                             modality="chest_xray", diagnosis="pneumonia",
                             top_probability=0.9)
        assert result["stored"] is True
        assert result["study_hash"] == "hash1"
        rec = store.get("hash1")
        assert rec is not None
        assert rec["study_id"] == "s1"
        assert rec["modality"] == "chest_xray"
        assert rec["diagnosis"] == "pneumonia"
        np.testing.assert_array_almost_equal(rec["vision_embedding"], [1.0, 2.0, 3.0])

    def test_get_missing_returns_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store.get("nonexistent") is None

    def test_get_by_case(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", case_id="case_1", study_id="s1")
        rec = store.get_by_case("case_1")
        assert rec is not None
        assert rec["case_id"] == "case_1"
        assert store.get_by_case("nonexistent") is None

    def test_count(self, tmp_path):
        store = self._store(tmp_path)
        assert store.count() == 0
        store.store("h1", study_id="s1")
        store.store("h2", study_id="s2")
        assert store.count() == 2

    def test_list_all(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1", modality="chest_xray")
        store.store("h2", study_id="s2", modality="brain_mri")
        items = store.list_all()
        assert len(items) == 2

    def test_delete(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1")
        assert store.delete("h1") is True
        assert store.get("h1") is None
        assert store.delete("h1") is False

    def test_store_upserts(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1", diagnosis="pneumonia")
        store.store("h1", study_id="s1_updated", diagnosis="cardiomegaly")
        rec = store.get("h1")
        assert rec["study_id"] == "s1_updated"
        assert rec["diagnosis"] == "cardiomegaly"
        assert store.count() == 1

    def test_search_by_diagnosis(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1", diagnosis="pneumonia",
                     fused_embedding=[1.0, 0.0])
        store.store("h2", study_id="s2", diagnosis="cardiomegaly",
                     fused_embedding=[0.0, 1.0])
        results = store.search_by_diagnosis("pneumonia")
        assert len(results) == 1
        assert results[0]["diagnosis"] == "pneumonia"

    def test_similarity_search(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1", fused_embedding=[1.0, 0.0, 0.0])
        store.store("h2", study_id="s2", fused_embedding=[0.9, 0.1, 0.0])
        store.store("h3", study_id="s3", fused_embedding=[0.0, 0.0, 1.0])
        results = store.similarity_search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["similarity"] >= results[1]["similarity"]

    def test_similarity_search_empty(self, tmp_path):
        store = self._store(tmp_path)
        results = store.similarity_search([1.0, 0.0])
        assert results == []

    def test_similarity_search_by_modality(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1", modality="chest_xray",
                     fused_embedding=[1.0, 0.0])
        store.store("h2", study_id="s2", modality="brain_mri",
                     fused_embedding=[0.9, 0.1])
        results = store.similarity_search([1.0, 0.0], modality="chest_xray")
        assert len(results) == 1
        assert results[0]["modality"] == "chest_xray"

    def test_compute_study_hash(self):
        from aura.common.storage.feature_store import FeatureStore
        h1 = FeatureStore.compute_study_hash(b"image_data")
        h2 = FeatureStore.compute_study_hash(b"image_data")
        h3 = FeatureStore.compute_study_hash(b"different_data")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64

    def test_compute_study_hash_with_id(self):
        from aura.common.storage.feature_store import FeatureStore
        h1 = FeatureStore.compute_study_hash(b"data", study_id="s1")
        h2 = FeatureStore.compute_study_hash(b"data", study_id="s2")
        assert h1 != h2

    def test_store_with_lists(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", vision_embedding=[1.0, 2.0],
                     fused_embedding=[3.0, 4.0])
        rec = store.get("h1")
        assert rec["vision_embedding"] == [1.0, 2.0]
        assert rec["fused_embedding"] == [3.0, 4.0]

    def test_store_latency_fields(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", study_id="s1",
                     vision_latency_ms=15.5, fusion_latency_ms=8.2,
                     total_latency_ms=23.7)
        rec = store.get("h1")
        assert rec["vision_latency_ms"] == 15.5
        assert rec["fusion_latency_ms"] == 8.2
        assert rec["total_latency_ms"] == 23.7

    def test_extra_metadata(self, tmp_path):
        store = self._store(tmp_path)
        store.store("h1", extra={"version": "1.0", "gpu": "A100"})
        rec = store.get("h1")
        assert rec["extra"]["version"] == "1.0"
        assert rec["extra"]["gpu"] == "A100"


# ═══════════════════════════════════════════════════════════════════════
# Integration checks
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_all_new_modules_importable(self):
        """Verify every new module is importable without side effects."""
        from aura.gateway.adapters.base_plugin import BaseModalityPlugin, PixelSignature
        from aura.gateway.adapters.plugin_registry import (
            register_plugin, get_plugin, registered_plugins,
            plugin_modalities, resolve_plugin,
        )
        from aura.gateway.adapters.thorax_plugin import ThoraxPlugin
        from aura.gateway.adapters.neuro_plugin import NeuroPlugin
        from aura.common.state import PipelineFSM, PipelineState, SafetyVerdict, ReadinessVerdict
        from aura.services.reasoning.versioning import VersionedEvidenceGraph, GraphVersion
        from aura.gateway.api.explain import router as explain_router
        from bench.runner import BenchmarkRunner, BenchmarkResult, MetricsCard
        from aura.services.models.registry import ModelRegistry, sha256_bytes
        from aura.common.eventbus import EventBus, Event, StudyReceivedEvent, VisionCompletedEvent
        from aura.common.storage.feature_store import FeatureStore
        assert True

    def test_fsm_with_safety_controller_integration(self):
        """FSM state transitions align with safety controller flow."""
        from aura.common.state.fsm import PipelineFSM, PipelineState, SafetyVerdict
        from aura.services.safety.controller import ClinicalSafetyController

        controller = ClinicalSafetyController()
        fsm = PipelineFSM()

        fsm.transition(PipelineState.SAFETY_CHECK)
        fsm.set_safety_verdict(SafetyVerdict.SAFE)
        fsm.transition(PipelineState.EVIDENCE_COLLECTION)
        fsm.transition(PipelineState.REASONING)
        assert fsm.state == PipelineState.REASONING

    def test_versioned_graph_with_evidence_nodes(self):
        """VersionedEvidenceGraph works with actual EvidenceNode schema."""
        from aura.services.reasoning.versioning import VersionedEvidenceGraph
        from aura.schemas.contracts import EvidenceNode, EvidenceGraph, EvidenceKind
        graph = EvidenceGraph()
        veg = VersionedEvidenceGraph(initial=graph)
        node = EvidenceNode(
            id="fever", kind=EvidenceKind.CLINICIAN_INPUT,
            label="Fever", value=0.9, modality="symptoms",
        )
        v = veg.update(added_nodes=[node], source="clinical_assessment")
        assert v.version == 2
        assert "fever" in veg.current.nodes
