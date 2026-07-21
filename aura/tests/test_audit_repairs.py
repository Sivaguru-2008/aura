"""Regression tests for the audit repairs (F2, F5, F6, F7, F9, F10 + hardening).

These exercise the *wiring* and *invariants* the audit found broken, using the
synthetic fusion artifacts (no CNN / MIMIC needed), so they run in CI. Each test
names the finding it guards.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from schemas.clinical import DIAGNOSES, Diagnosis, Finding
from schemas.contracts import (
    AbstentionReason,
    DifferentialItem,
    ReasoningTrace,
    Recommendation,
    SafetyAssessment,
    Prediction,
    VisionResult,
    FindingScore,
)


# --------------------------------------------------------------------------- #
# F2 — the conflict-guard-resolved posterior reaches the safety assessment.
# --------------------------------------------------------------------------- #
def test_f2_resolved_logits_drive_safety_top():
    """safety.top must reflect the logits actually passed in (the resolved backend),
    not a recomputation from the default model."""
    from services.safety import SafetyEngine
    from services.fusion import FusionEngine

    fe = FusionEngine()
    se = SafetyEngine()
    x = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Force a resolved posterior that puts all mass on COPD via crafted logits.
    copd_i = DIAGNOSES.index(Diagnosis.COPD)
    resolved = np.full(len(DIAGNOSES), -5.0)
    resolved[copd_i] = 10.0
    s = se.assess("t", x, fe.model, resolved_logits=resolved)
    assert s.top == Diagnosis.COPD
    # And the default (no resolved logits) path can differ — proving the wiring.
    s_default = se.assess("t", x, fe.model)
    assert s_default.top == DIAGNOSES[int(np.argmax(se.calibrated_posterior(fe.model.logits(x))))]


def test_f2_guard_fallback_removes_contradiction():
    """When the Wasserstein guard fires, the pipeline feeds the classical logits to
    safety, so safety.top equals the guard-resolved top (no contradiction)."""
    from services.safety import SafetyEngine
    from services.fusion import FusionEngine

    fe = FusionEngine()
    if not fe._guard_enabled:
        pytest.skip("conflict guard not enabled (non-quantum backend)")
    se = SafetyEngine()
    rng = np.random.default_rng(3)
    fired = False
    for _ in range(500):
        x = rng.random(8)
        res = fe.fuse_vector(x)
        if res.fallback_triggered:
            fired = True
            resolved = fe.resolved_logits(x, res)
            s = se.assess("t", x, fe.model, resolved_logits=resolved)
            fusion_top = max(res.posterior, key=res.posterior.get)
            assert s.top == fusion_top          # report == guard decision
    assert fired, "expected at least one guard fallback across 500 random vectors"


# --------------------------------------------------------------------------- #
# F7 — Mondrian conformal thresholds are no longer degenerate.
# --------------------------------------------------------------------------- #
def test_f7_min_calibration_count():
    from services.safety.uncertainty import min_calibration_count

    assert min_calibration_count(0.90) == 19     # need n>=19 for a non-max quantile
    assert min_calibration_count(0.95) == 39


def test_f7_mondrian_no_saturation():
    """Small per-class calibration sets fall back to the pooled marginal instead of
    saturating to the class maximum (which put malignancy in ~78% of sets)."""
    from services.safety.uncertainty import mondrian_qhats

    rng = np.random.default_rng(0)
    C = len(DIAGNOSES)
    labels = rng.choice(C, size=140, p=[0.34, 0.18, 0.15, 0.13, 0.12, 0.08])
    probs = np.full((140, C), 0.05)
    for i, l in enumerate(labels):
        probs[i, l] = 0.7 if l != 4 else 0.2
        probs[i] /= probs[i].sum()
    q = mondrian_qhats(probs, labels, 0.90, C)
    assert not np.any(q > 0.95), f"a class saturated: {q}"
    assert np.all(q >= 0.0) and np.all(q <= 1.0)


def test_f7_quantile_hi_returns_none_when_degenerate():
    from services.safety.uncertainty import _quantile_hi

    assert _quantile_hi(np.array([0.3] * 5), 0.90) is None       # too few points
    assert _quantile_hi(np.array([]), 0.90) is None
    assert _quantile_hi(np.linspace(0, 1, 50), 0.90) is not None  # enough points


# --------------------------------------------------------------------------- #
# F9 — the online ACI threshold reaches the conformal set.
# --------------------------------------------------------------------------- #
def test_f9_aci_qhat_tags_and_widens_conformal():
    from services.safety import SafetyEngine
    from services.fusion import FusionEngine

    fe = FusionEngine()
    se = SafetyEngine()
    x = np.array([0.6, 0.2, 0.1, 0.7, 0.1, 0.05, 0.05, 0.1])

    base = se.assess("c", x, fe.model)
    wide = se.assess("c", x, fe.model, aci_qhat=0.99)      # very wide threshold
    assert wide.conformal_method.endswith("aci")
    assert "aci" not in base.conformal_method
    assert len(wide.conformal_set) >= len(base.conformal_set)


def test_f9_pipeline_reads_persisted_aci(tmp_path):
    """Pipeline with a store loads the persisted ACI q̂ and applies it."""
    from gateway.pipeline import Pipeline
    from gateway.storage import Store

    store = Store(tmp_path / "aci.db")
    probs = [0.55, 0.15, 0.12, 0.08, 0.06, 0.04]
    for i in range(20):                              # misses -> q̂ rises
        store.record_outcome(f"c{i}", probs, true_index=5)
    pipe = Pipeline(store=store)
    q = pipe._aci_qhat()
    assert q is not None and q > 0.5


# --------------------------------------------------------------------------- #
# F10 — clinical reasoning drives the impression (no impression/differential clash).
# --------------------------------------------------------------------------- #
def _mk_safety(top: Diagnosis, prob: float) -> SafetyAssessment:
    preds = [Prediction(diagnosis=d, probability=(prob if d == top else 0.05),
                        ci_low=0.0, ci_high=1.0) for d in DIAGNOSES]
    return SafetyAssessment(
        study_id="s", predictions=preds, top=top, top_probability=prob,
        conformal_set=[top], conformal_coverage=0.9, conformal_method="mondrian",
        abstained=False, abstention_reason=AbstentionReason.NONE, model_version="t")


def test_f10_reasoning_adjusted_posterior_sets_impression():
    from services.report import ReportEngine

    # Imaging says HEART_FAILURE, reasoning revises to PNEUMONIA.
    safety = _mk_safety(Diagnosis.HEART_FAILURE, 0.6)
    adj = {d: 0.02 for d in DIAGNOSES}
    adj[Diagnosis.PNEUMONIA] = 0.75
    reasoning = ReasoningTrace(
        study_id="s", adjusted_posterior=adj,
        steps=[__import__("schemas.contracts", fromlist=["ReasoningStep"]).ReasoningStep(
            statement="labs", evidence=["labs.wbc"], effect={Diagnosis.PNEUMONIA: 1.0})],
        differential=[DifferentialItem(diagnosis=Diagnosis.PNEUMONIA, probability=0.75)],
    )
    vision = VisionResult(study_id="s", findings=[
        FindingScore(finding=f, probability=0.1) for f in Finding], embedding=[0.0],
        model_version="v")
    draft = ReportEngine().compose(vision, safety, [], reasoning)
    assert draft.grounding["impression"][0] == Diagnosis.PNEUMONIA.value
    assert "pneumonia" in draft.impression_text.lower()
    assert "clinical correlation" in draft.impression_text.lower()


def test_f10_no_multimodal_leaves_impression_from_imaging():
    from services.report import ReportEngine

    safety = _mk_safety(Diagnosis.COPD, 0.7)
    reasoning = ReasoningTrace(study_id="s", adjusted_posterior={}, steps=[])  # inert
    vision = VisionResult(study_id="s", findings=[
        FindingScore(finding=f, probability=0.1) for f in Finding], embedding=[0.0],
        model_version="v")
    draft = ReportEngine().compose(vision, safety, [], reasoning)
    assert draft.grounding["impression"][0] == Diagnosis.COPD.value
    assert "clinical correlation" not in draft.impression_text.lower()


# --------------------------------------------------------------------------- #
# Thread safety — RecommendEngine keeps no per-request state on self.
# --------------------------------------------------------------------------- #
def test_recommend_is_reentrant_no_instance_state():
    from services.recommend import RecommendEngine
    from services.fusion import FusionEngine

    fe = FusionEngine()
    eng = RecommendEngine()
    x = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.2, 0.15, 0.1])
    r1 = eng.recommend(fe.model, x)
    r2 = eng.recommend(fe.model, x)
    assert not hasattr(eng, "_panel_evoi"), "per-request state leaked onto self"
    assert [r.action for r in r1] == [r.action for r in r2]   # deterministic / stable


# --------------------------------------------------------------------------- #
# Security hardening.
# --------------------------------------------------------------------------- #
def test_upload_size_cap_rejects_oversized():
    from gateway.security import read_capped
    from fastapi import HTTPException

    class _FakeUpload:
        def __init__(self, data): self._d = data; self._i = 0
        async def read(self, n=-1):
            chunk = self._d[self._i:self._i + (n if n and n > 0 else len(self._d))]
            self._i += len(chunk)
            return chunk

    small = asyncio.run(read_capped(_FakeUpload(b"x" * 100), 1000))
    assert small == b"x" * 100
    with pytest.raises(HTTPException) as exc:
        asyncio.run(read_capped(_FakeUpload(b"x" * 5000), 1000))
    assert exc.value.status_code == 413


def test_upload_type_allowlist():
    from gateway.security import validate_upload_name
    from fastapi import HTTPException

    validate_upload_name("chest.png", "image/png")     # ok
    validate_upload_name("scan.dcm", "application/dicom")
    validate_upload_name("noext", "application/octet-stream")
    with pytest.raises(HTTPException) as e1:
        validate_upload_name("evil.exe", "application/octet-stream")
    assert e1.value.status_code == 415
    with pytest.raises(HTTPException):
        validate_upload_name("x.png", "text/html")


def test_rate_limiter_blocks_over_budget():
    from gateway.security import RateLimiter

    rl = RateLimiter(rpm=3)
    assert all(rl.allow("user") for _ in range(3))
    assert not rl.allow("user")                       # 4th within the window
    assert rl.allow("other")                          # independent key
    assert RateLimiter(rpm=0).allow("anyone")         # disabled = always allow


# --------------------------------------------------------------------------- #
# F5 — full-fidelity intake grid.
# --------------------------------------------------------------------------- #
def test_f5_default_grid_is_full_fidelity():
    from services.vision.io import DEFAULT_GRID, _resize_grid

    assert DEFAULT_GRID == 224
    big = np.random.default_rng(0).random((1000, 1200)).astype(np.float32)
    small = _resize_grid(big, 224)
    assert small.shape == (224, 224)
    # area-averaging preserves the mean better than point sampling
    assert abs(float(small.mean()) - float(big.mean())) < 0.02
