"""`aura.common.config` must keep resolving to the module, and say so when it can't.

Two failure modes this pins:

1. **Name collision.** `aura/common/config.py` used to sit next to a directory
   `aura/common/config/` holding safety_policy.yaml. That resolves to the module only
   because a regular module outranks a *namespace* package — the moment anything puts
   an `__init__.py` in the directory it becomes a regular package, wins the lookup,
   and every `from aura.common.config import get_settings` in the tree breaks at once.
   The YAML now lives in `common/policy/`; this test stops the collision returning.

2. **Silent policy fallback.** The thresholds in that file decide when AURA abstains
   from a clinical claim. A missing or unparseable file falls back to built-in
   defaults, which is the right behaviour — but it has to be *loud*, or a path
   regression looks exactly like a successful load.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import aura.common.config as config

PKG_ROOT = Path(config.__file__).resolve().parents[1]     # .../aura-main/aura


def test_config_resolves_to_the_module_not_a_package():
    assert not hasattr(config, "__path__"), (
        "aura.common.config resolved to a PACKAGE. Something added an __init__.py to "
        "a directory named 'config' beside config.py; every import of get_settings "
        "in the codebase now reads the wrong thing."
    )
    assert Path(config.__file__).name == "config.py"


def test_no_directory_shadows_the_config_module():
    shadow = PKG_ROOT / "common" / "config"
    assert not shadow.exists(), (
        f"{shadow} exists and competes with config.py for 'aura.common.config'. "
        f"Put configuration data under common/policy/ instead."
    )


def test_the_policy_file_is_where_the_loader_looks():
    assert config._SAFETY_POLICY_PATH.exists(), (
        f"safety_policy.yaml not found at {config._SAFETY_POLICY_PATH}. The served "
        f"abstention thresholds are silently falling back to built-in defaults."
    )


def test_both_loaders_share_one_path():
    """They had independent copies; a move fixed one and left the other stale."""
    src = Path(config.__file__).read_text(encoding="utf-8")
    assert src.count('"safety_policy.yaml"') <= 1, (
        "safety_policy.yaml is named more than once in config.py — the two loaders "
        "have drifted apart again. Both must use _SAFETY_POLICY_PATH."
    )


def test_missing_policy_file_warns_and_falls_back(monkeypatch, tmp_path, capsys):
    """Fall back safely, but never silently."""
    monkeypatch.setattr(config, "_SAFETY_POLICY_PATH", tmp_path / "absent.yaml")
    config.get_safety_policy.cache_clear()
    try:
        policy = config.get_safety_policy()
    finally:
        config.get_safety_policy.cache_clear()

    assert policy == config.SafetyPolicyThresholds()
    err = capsys.readouterr().err
    assert "safety policy" in err and "not found" in err, (
        "a missing safety-policy file fell back to defaults without warning"
    )


def test_unparseable_policy_file_warns_and_falls_back(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "safety_policy.yaml"
    bad.write_text("policies: [this is not: a mapping\n", encoding="utf-8")
    monkeypatch.setattr(config, "_SAFETY_POLICY_PATH", bad)
    config.get_safety_policy.cache_clear()
    try:
        policy = config.get_safety_policy()
    finally:
        config.get_safety_policy.cache_clear()

    assert policy == config.SafetyPolicyThresholds()
    assert "safety policy" in capsys.readouterr().err


def test_served_policy_matches_the_yaml():
    """The active profile is the one the file says is default, with its values."""
    import yaml

    data = yaml.safe_load(config._SAFETY_POLICY_PATH.read_text(encoding="utf-8"))
    expected = data["policies"][data["default_policy"]]
    policy = config.get_safety_policy()

    assert policy.ood_threshold == pytest.approx(expected["ood_threshold"])
    assert policy.epistemic_threshold == pytest.approx(expected["epistemic_threshold"])
    assert policy.min_coverage == pytest.approx(expected["min_coverage"])


def test_blank_env_var_is_treated_as_unset(monkeypatch):
    """`KEY=` in a .env must not be parsed as a value.

    AURA_RATE_LIMIT_RPM= reached int("") and killed the process at startup;
    AURA_FUSION_BACKEND= was read as a real backend name and broke fusion.
    """
    monkeypatch.setenv("AURA_RATE_LIMIT_RPM", "")
    monkeypatch.setenv("AURA_FUSION_BACKEND", "")
    config.get_settings.cache_clear()
    try:
        s = config.get_settings()
        assert s.rate_limit_rpm == 0
        assert s.fusion_backend in {"quantum", "classical", "learnable", "ensemble"}
    finally:
        config.get_settings.cache_clear()
