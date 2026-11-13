"""Pillar 1 (data strategy and model foundation), as it actually stands.

- A commercial deployment refuses research-licensed artifacts, and fails closed on weight
  files it has no record of.
- An operating mode is honoured only when the serving calibration records thresholds
  fitted for it; otherwise the validated per-finding thresholds apply. None is recorded
  today: the MIMIC-CXR validation images they would be fitted on are not on this machine.
- An incomplete MRI study is refused rather than filled in from another sequence: the
  network's measured single-sequence Dice (T1 0.02, T1ce 0.00) makes a copied channel a
  fabricated input, not a degraded one.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from aura.common import config
from aura.common import data_governance as dg
from aura.ml.evaluation.vision_calibration import (
    _specificity_guard_threshold,
    select_multi_mode_thresholds,
)
from aura.schemas.clinical import ClinicalOperatingMode

IMG = np.full((64, 64), 0.5, dtype=np.float32)


# ---- data governance ---------------------------------------------------------------- #
def test_every_served_model_is_recorded_as_research_only():
    for rel in ("best_model.pt", "fusion_classical.npz", "safety_classical.npz",
                "vision_serving_calibration.json", "brain/checkpoints/best_brain_model.pt"):
        assert dg.KNOWN_PROVENANCE[rel].license_type is dg.LicenseType.RESEARCH_ONLY


def test_commercial_mode_refuses_research_licensed_weights(tmp_path):
    (tmp_path / "best_model.pt").write_bytes(b"\0")
    with pytest.raises(dg.CommercialLicenseViolationError, match=r"best_model\.pt: RESEARCH_ONLY"):
        dg.DataGovernanceManager(tmp_path).enforce_commercial_gate(commercial_mode=True)


def test_commercial_mode_fails_closed_on_unrecorded_weights(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "mystery_model.pt").write_bytes(b"\0")
    with pytest.raises(dg.CommercialLicenseViolationError, match=r"sub/mystery_model\.pt: UNVERIFIED"):
        dg.DataGovernanceManager(tmp_path).enforce_commercial_gate(commercial_mode=True)


def test_a_recorded_commercial_licence_passes(tmp_path, monkeypatch):
    (tmp_path / "licensed.pt").write_bytes(b"\0")
    monkeypatch.setitem(dg.KNOWN_PROVENANCE, "licensed.pt", dg.ModelProvenance(
        "licensed.pt", "test model", "partner hospital", "signed agreement",
        dg.LicenseType.COMMERCIAL_AUTHORIZED))
    report = dg.DataGovernanceManager(tmp_path).enforce_commercial_gate(commercial_mode=True)
    assert report.violations == []
    assert report.cleared_for_commercial_use


def test_research_mode_does_not_block_or_scan():
    assert dg.DataGovernanceManager().enforce_commercial_gate(commercial_mode=False) is None


# ---- operating modes ---------------------------------------------------------------- #
def test_operating_modes_are_named():
    assert [m.value for m in ClinicalOperatingMode] == ["triage", "balanced", "specificity_guard"]


def test_an_unmeasured_mode_falls_back_to_the_validated_threshold(monkeypatch):
    monkeypatch.setattr(config, "_CALIBRATION_CACHE",
                        {"per_finding_threshold": {"pneumothorax": 0.0334}})
    assert config.measured_operating_modes() == ()
    for mode in (None, "triage", "balanced", "specificity_guard"):
        assert config.finding_present_threshold("pneumothorax", mode=mode) == 0.0334


def test_a_measured_mode_is_used_only_when_asked_for(monkeypatch):
    monkeypatch.setattr(config, "_CALIBRATION_CACHE", {
        "per_finding_threshold": {"pneumothorax": 0.0334},
        "modes": {"specificity_guard": {"thresholds": {"pneumothorax": 0.4}}}})
    assert config.measured_operating_modes() == ("specificity_guard",)
    assert config.finding_present_threshold("pneumothorax", mode="specificity_guard") == 0.4
    assert config.finding_present_threshold("pneumothorax") == 0.0334


def test_any_recorded_mode_carries_what_it_measured():
    """A mode may be served only with the precision and recall it achieved and the data
    it was fitted on, so a threshold cannot be recorded under a name it does not earn."""
    served = json.loads((config.ARTIFACTS / "vision_serving_calibration.json").read_text())
    for name, mode in served.get("modes", {}).items():
        assert {"thresholds", "precision", "recall", "fitted_on"} <= set(mode), name


@pytest.fixture(scope="module")
def vision_engine():
    from aura.services.vision.engine import VisionEngine
    return VisionEngine.load()


def test_vision_engine_ignores_an_unmeasured_mode(vision_engine):
    res = vision_engine.analyze("TEST-01", IMG, operating_mode="specificity_guard")
    assert res.operating_mode is None
    for f in res.findings:
        assert f.operating_mode is None
        assert f.threshold == round(config.finding_present_threshold(f.finding.value), 4)
        assert isinstance(f.present, bool)


def test_vision_engine_applies_a_measured_mode(vision_engine, monkeypatch):
    cal = dict(config._serving_calibration())
    cal["modes"] = {"specificity_guard": {"thresholds": {"pneumothorax": 0.4}}}
    monkeypatch.setattr(config, "_CALIBRATION_CACHE", cal)
    res = vision_engine.analyze("TEST-02", IMG, operating_mode="specificity_guard")
    assert res.operating_mode == "specificity_guard"
    by_finding = {f.finding.value: f for f in res.findings}
    assert by_finding["pneumothorax"].threshold == 0.4
    assert by_finding["opacity"].threshold == round(cal["per_finding_threshold"]["opacity"], 4)


def test_specificity_guard_threshold_meets_its_target():
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1, 1])
    t = _specificity_guard_threshold(p, y, min_precision=0.75)
    assert t is not None
    assert (y[p >= t] == 1).mean() >= 0.75


def test_specificity_guard_threshold_is_none_when_unreachable():
    # The best achievable precision here is 1/3, so no threshold earns the name.
    p = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([0, 0, 1, 0])
    assert _specificity_guard_threshold(p, y, min_precision=0.75) is None


def test_multi_mode_selection_reports_what_each_mode_achieves():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, 400), 0, 1)
    modes = select_multi_mode_thresholds(p, y, "opacity")
    for name in ("triage", "balanced"):
        assert {"threshold", "precision", "recall"} <= set(modes[name])
    guard = modes["specificity_guard"]
    assert guard is None or guard["precision"] >= 0.75


# ---- brain MRI ---------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(8, 8, 4), (8, 8, 4, 3)])
def test_an_incomplete_mri_study_is_refused_not_filled_in(tmp_path, shape):
    import nibabel as nib

    from aura.backend.engines.neuro.multisequence import ChannelOrderRejected, load_multisequence

    path = tmp_path / "study.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)), str(path))
    with pytest.raises(ChannelOrderRejected, match="expected a 4D volume with 4 channels"):
        load_multisequence(path, ("flair", "t1", "t1ce", "t2"))
