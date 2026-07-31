"""A field that does not exist on the model is dropped in silence. Catch that.

Pydantic v2 defaults to ``extra='ignore'``, so ``Model(known=1, typo=2)`` constructs
happily and ``typo`` vanishes. No exception, no warning, no failing test — the value
simply never arrives, and the only symptom is a feature that quietly does nothing.

That is not hypothetical here. Three real cases were live at once:

* ``FusionResult(quantum_entanglement=...)`` — ``measure_entanglement`` ran on every
  quantum study at ~31 ms each and the result was discarded. The console's
  entanglement telemetry panel is gated on that key, so it could never render.
* ``Explanation(geometry=...)`` — ``geometry_for`` ran at ~74 ms per study and was
  discarded. The console reads ``explanation.geometry`` to draw region polygons.
* ``FusionResult(qae_enabled=..., qbn_enabled=...)`` — flags describing *settings*
  rather than what executed, feeding console rows that announce quantum components.

~105 ms of wasted work per study and two dead panels, invisible to a green suite.

This test AST-scans every construction of a contract model in the serving tree and
fails on any keyword the model does not declare. It needs no runtime and no fixtures,
so it costs nothing and fires on the first regression.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel

import aura.schemas.clinical as clinical
import aura.schemas.contracts as contracts

PKG_ROOT = Path(__file__).resolve().parents[1]      # .../aura-main/aura
REPO_ROOT = PKG_ROOT.parent

# Directories whose constructions are not serving code. Tests are excluded because a
# test may legitimately construct a stub; the point of this guard is the live path.
SKIP_PARTS = {"__pycache__", ".pytest_cache", "pytest-tmp", "tests", "change",
              ".venv", "venv", "artifacts", "demo_data", "data"}


def _contract_models() -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for module in (contracts, clinical):
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                models[name] = obj
    return models


def _source_files() -> list[Path]:
    return [p for p in PKG_ROOT.rglob("*.py")
            if not any(part in SKIP_PARTS for part in p.parts)]


def _dropped_kwargs() -> list[str]:
    models = _contract_models()
    offenders: list[str] = []

    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:                   # pragma: no cover
            pytest.fail(f"{path} does not parse: {exc}")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            else:
                continue
            model = models.get(name)
            if model is None:
                continue
            # Aliases count: a field declared with alias="x" is settable as "x".
            allowed = set(model.model_fields)
            for fname, finfo in model.model_fields.items():
                if getattr(finfo, "alias", None):
                    allowed.add(finfo.alias)

            for kw in node.keywords:
                if kw.arg is None:                   # **kwargs — cannot check statically
                    continue
                if kw.arg not in allowed:
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"  {rel}:{node.lineno}  {name}(..., {kw.arg}=...)")
    return offenders


def test_no_model_construction_passes_an_undeclared_field():
    offenders = _dropped_kwargs()
    assert not offenders, (
        "These keywords are silently discarded by pydantic (extra='ignore'). The "
        "work that produced each value is being thrown away, and any consumer "
        "reading the key gets undefined:\n"
        + "\n".join(sorted(offenders))
        + "\n\nEither declare the field on the model or stop computing the value."
    )


def test_the_two_recovered_fields_are_declared():
    """Regression pins for the specific fields this guard was written after."""
    assert "quantum_entanglement" in contracts.FusionResult.model_fields
    assert "qae_applied" in contracts.FusionResult.model_fields
    assert "geometry" in contracts.Explanation.model_fields


def test_qbn_flag_is_gone():
    """QuantumBayesianNetwork is not on the serving path, so no flag may claim it is.

    Reinstate `qbn_enabled` only alongside a QBN the pipeline actually constructs.
    """
    assert "qbn_enabled" not in contracts.FusionResult.model_fields
    engine_src = (PKG_ROOT / "services" / "fusion" / "engine.py").read_text(encoding="utf-8")
    assert "qbn_enabled=" not in engine_src, (
        "services/fusion/engine.py still passes qbn_enabled. The QBN is not wired "
        "into the serving path; a flag that reports a setting as a running component "
        "is the failure mode AURA_ALLOW_FALLBACK_VISION exists to prevent."
    )


def test_qae_flag_describes_execution_not_configuration():
    """`qae_applied` must be derived from the branch taken, not from the setting."""
    engine_src = (PKG_ROOT / "services" / "fusion" / "engine.py").read_text(encoding="utf-8")
    assert "qae_enabled=bool(getattr(s" not in engine_src, (
        "qae_applied is being set from the qae_enabled setting again. The compression "
        "path also requires a vision embedding and loaded QAE weights "
        "(QuantumAutoencoder.load() returns None when they are absent, which is the "
        "shipped state), so the setting alone does not mean the autoencoder ran."
    )
