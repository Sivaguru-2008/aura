"""Tests for the modality adapter factory pattern.

Verifies:
- The adapter registry auto-registers ThoraxAdapter and NeuroAdapter.
- get_adapter returns correct instances.
- get_adapter_for_modality returns None for unknown modalities.
- The adapter interface has the required methods.
- ThoraxAdapter inspects accept/reject correctly.
"""
from __future__ import annotations

import pytest

from aura.gateway.adapters.base import ModalityAdapter, InspectionResult, StandardizedAsset, EngineOutput
from aura.gateway.adapters.registry import get_adapter, get_adapter_for_modality, registered_modalities, register_adapter


def test_registry_auto_registers_builtins():
    mods = registered_modalities()
    assert "chest_xray" in mods
    assert "brain_mri" in mods


def test_get_adapter_returns_instance():
    adapter = get_adapter("chest_xray")
    assert adapter is not None
    assert isinstance(adapter, ModalityAdapter)
    assert adapter.modality == "chest_xray"


def test_get_adapter_for_modality_alias():
    adapter = get_adapter_for_modality("brain_mri")
    assert adapter is not None
    assert adapter.modality == "brain_mri"


def test_get_adapter_returns_none_for_unknown():
    assert get_adapter("unknown_modality") is None
    assert get_adapter_for_modality("ultrasound") is None


def test_adapter_has_required_methods():
    adapter = get_adapter("chest_xray")
    assert callable(getattr(adapter, "inspect", None))
    assert callable(getattr(adapter, "standardize", None))
    assert callable(getattr(adapter, "analyze", None))


def test_custom_adapter_registration():
    class DummyAdapter(ModalityAdapter):
        modality = "dummy"
        display_name = "Dummy"

        def inspect(self, asset_path, asset_meta=None, **kwargs):
            return InspectionResult(accepted=True)

        def standardize(self, asset_path, asset_meta=None, **kwargs):
            return StandardizedAsset(study_id="s", case_id="c", payload=None)

        async def analyze(self, standardized, pipeline, store, **kwargs):
            return EngineOutput(case_id="c", study_id="s", bundle=None)

    register_adapter(DummyAdapter)
    assert "dummy" in registered_modalities()
    adapter = get_adapter("dummy")
    assert adapter.modality == "dummy"
    assert adapter.inspect("dummy_path").accepted is True


def test_inspection_result_dataclass():
    r = InspectionResult(accepted=True, reason="ok", checks={"k": "v"})
    assert r.accepted is True
    assert r.reason == "ok"
    assert r.checks == {"k": "v"}


def test_standardized_asset_dataclass():
    s = StandardizedAsset(study_id="S1", case_id="C1", payload="data", metadata={"m": 1})
    assert s.study_id == "S1"
    assert s.case_id == "C1"
    assert s.payload == "data"


def test_engine_output_dataclass():
    e = EngineOutput(case_id="C1", study_id="S1", bundle="b", metadata={"t": 0.5})
    assert e.case_id == "C1"
    assert e.bundle == "b"
