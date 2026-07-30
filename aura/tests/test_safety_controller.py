"""Tests for Step 1: Guidelines templates and safety policy configuration."""
from __future__ import annotations

import pytest

from aura.common.config import SafetyPolicyThresholds, get_safety_policy
from aura.knowledge.guidelines.templates import (
    GUIDELINE_TEMPLATES,
    GuidelineTemplate,
    coverage_ratio,
    get_template,
)


# --------------------------------------------------------------------------- #
# Guideline templates
# --------------------------------------------------------------------------- #
class TestGuidelineTemplates:
    def test_all_diagnoses_covered(self):
        expected = {"pneumonia", "heart_failure", "copd", "malignancy", "pneumothorax", "normal"}
        assert set(GUIDELINE_TEMPLATES.keys()) == expected

    def test_template_structure(self):
        for key, tmpl in GUIDELINE_TEMPLATES.items():
            assert isinstance(tmpl, GuidelineTemplate)
            assert tmpl.diagnosis == key
            assert isinstance(tmpl.imaging, list)
            assert isinstance(tmpl.labs, list)
            assert isinstance(tmpl.symptoms, list)
            assert len(tmpl.guideline_source) > 0

    def test_get_template(self):
        t = get_template("pneumonia")
        assert t is not None
        assert t.diagnosis == "pneumonia"
        assert "consolidation" in t.imaging
        assert "wbc" in t.labs

    def test_get_template_missing(self):
        assert get_template("nonexistent") is None

    def test_pneumonia_has_expected_indicators(self):
        t = GUIDELINE_TEMPLATES["pneumonia"]
        assert "consolidation" in t.imaging
        assert "wbc" in t.labs
        assert "procalcitonin" in t.labs
        assert "fever" in t.symptoms

    def test_heart_failure_has_expected_indicators(self):
        t = GUIDELINE_TEMPLATES["heart_failure"]
        assert "cardiomegaly" in t.imaging
        assert "bnp" in t.labs
        assert "orthopnea" in t.symptoms


# --------------------------------------------------------------------------- #
# Coverage ratio
# --------------------------------------------------------------------------- #
class TestCoverageRatio:
    def test_no_data_returns_zero(self):
        assert coverage_ratio("pneumonia") == 0.0

    def test_full_coverage(self):
        t = GUIDELINE_TEMPLATES["pneumonia"]
        all_evidence = set(t.imaging + t.labs + t.symptoms)
        assert coverage_ratio(
            "pneumonia",
            available_imaging=t.imaging,
            available_labs=t.labs,
            available_symptoms=t.symptoms,
        ) == 1.0

    def test_partial_coverage(self):
        ratio = coverage_ratio(
            "pneumonia",
            available_imaging=["consolidation"],
            available_labs=["wbc", "procalcitonin"],
        )
        # 3 out of 7 indicators present
        assert 0.2 < ratio < 0.7

    def test_missing_diagnosis_returns_zero(self):
        assert coverage_ratio("nonexistent") == 0.0

    def test_normal_template_no_indicators(self):
        """Normal has empty imaging + symptoms, only spo2 lab."""
        t = GUIDELINE_TEMPLATES["normal"]
        assert coverage_ratio("normal", available_labs=["spo2"]) == 1.0


# --------------------------------------------------------------------------- #
# Safety policy
# --------------------------------------------------------------------------- #
class TestSafetyPolicy:
    def test_default_loads(self):
        p = get_safety_policy()
        assert isinstance(p, SafetyPolicyThresholds)
        assert p.ood_threshold > 0
        assert 0 < p.epistemic_threshold < 1
        assert 0 < p.min_coverage < 1
        assert 0 < p.low_confidence_threshold < 1

    def test_default_is_community_conservative(self):
        p = get_safety_policy()
        assert p.ood_threshold == 2.5
        assert p.epistemic_threshold == 0.15
        assert p.min_coverage == 0.70
