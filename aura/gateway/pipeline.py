"""The analysis pipeline — orchestrates the engines for one study.

This is the operational form of docs/ARCHITECTURE.md section 7. It wires the
engines together and emits events at each stage; in production each stage is a
separate service consuming the prior stage's event. Here they run in-process,
but the boundaries and the event contract are identical, so extraction to
independent services is a deployment change, not a rewrite.
"""
from __future__ import annotations

import numpy as np

from common import eventbus as ev
from common.config import get_settings
from common.eventbus import EventBus
from common.mathx import entropy, softmax
from schemas.clinical import DIAGNOSES, Diagnosis
from schemas.contracts import (
    CaseBundle,
    CaseState,
    StructuredPriors,
    StudyInput,
)
from services.explain import ExplainEngine
from services.fusion import FusionEngine
from services.fusion.evidence import encode, to_evidence_items
from services.memory import MemoryEngine
from services.reasoning import ClinicalReasoner
from services.recommend import RecommendEngine
from services.report import ReportEngine
from services.safety import SafetyEngine
from services.vision import VisionEngine
from schemas.clinical import Finding


class Pipeline:
    """Holds one instance of each engine (loaded models) and runs cases through them."""

    def __init__(self, bus: EventBus | None = None, memory: MemoryEngine | None = None,
                 store=None):
        self.bus = bus or EventBus()
        self.vision = VisionEngine.load()
        self.fusion = FusionEngine()
        self.safety = SafetyEngine()
        self.explain = ExplainEngine()
        self.recommend = RecommendEngine()
        self.reasoner = ClinicalReasoner()
        self.report = ReportEngine()
        self.memory = memory or MemoryEngine()
        # Optional persistence handle — lets serving read the online Adaptive
        # Conformal Inference threshold (Module 8) the feedback endpoint updates.
        # None for standalone/test construction, so those paths stay unchanged.
        self.store = store

    def _aci_qhat(self) -> float | None:
        """Current online ACI threshold, or None when ACI is off / no store / unset."""
        s = get_settings()
        if self.store is None or not getattr(s, "aci_enabled", False):
            return None
        try:
            row = self.store.load_aci_state()
            if not row:
                return None
            return float(row.get("qhat"))
        except Exception:
            return None

    def _priority(self, top: Diagnosis, safety) -> float:
        """Worklist priority: urgent + confident floats up; abstained flagged high too."""
        urgency = {
            Diagnosis.PNEUMOTHORAX: 1.0, Diagnosis.MALIGNANCY: 0.85,
            Diagnosis.HEART_FAILURE: 0.7, Diagnosis.PNEUMONIA: 0.6,
            Diagnosis.COPD: 0.4, Diagnosis.NORMAL: 0.1,
        }.get(top, 0.5)
        if safety.abstained:
            return round(0.75 + 0.25 * safety.epistemic_uncertainty, 4)
        return round(urgency * safety.top_probability, 4)

    async def run(self, study: StudyInput, case_id: str) -> CaseBundle:
        img = np.array(study.image, dtype=float).reshape(study.image_shape)
        await self.bus.publish(ev.STUDY_RECEIVED, study_id=study.study_id, case_id=case_id)

        # 1) Vision
        vision = self.vision.analyze(study.study_id, img)
        await self.bus.publish(ev.VISION_COMPLETED, study_id=study.study_id)

        # 2) Evidence + fusion
        x = encode(vision, study.priors)
        fusion = self.fusion.fuse_vector(x, study_id=study.study_id)
        evidence = to_evidence_items(x, study.priors)
        await self.bus.publish(ev.FUSION_COMPLETED, study_id=study.study_id)

        # 3) Clinical reasoning — fuse the calibrated imaging posterior with
        # labs/symptoms/history + guideline likelihood ratios, BEFORE safety, so the
        # posterior that safety validates is the *final* one shown to the clinician
        # (audit F10). With no multimodal evidence the reasoner is inert and the
        # adjusted posterior equals the imaging posterior — imaging behaviour is
        # unchanged. The conflict-guard-resolved logits feed this (audit F2).
        resolved_logits = self.fusion.resolved_logits(x, fusion)
        imaging_probs = self.safety.calibrated_posterior(resolved_logits)
        imaging_prior = {d: float(imaging_probs[i]) for i, d in enumerate(DIAGNOSES)}
        findings_map = {fs.finding: fs.probability for fs in vision.findings}
        reasoning = self.reasoner.reason(
            study.study_id, findings_map, imaging_prior, study.priors, study.multimodal
        )
        final_posterior = None
        if reasoning.steps:                      # reasoner actually changed the call
            final_posterior = np.array(
                [reasoning.adjusted_posterior.get(d, 0.0) for d in DIAGNOSES], dtype=float
            )

        # 4) Safety (calibration, conformal, OOD, abstention) on the FINAL posterior.
        # Also folds in the online ACI threshold (audit F9).
        safety = self.safety.assess(
            study.study_id, x, self.fusion.model,
            resolved_logits=resolved_logits, aci_qhat=self._aci_qhat(),
            final_posterior=final_posterior,
        )

        # 5) Explainability
        explanation = self.explain.explain(
            study.study_id, self.vision, img, self.fusion.model, x, safety.top
        )

        # 6) Missing-evidence recommendations
        recommendations = self.recommend.recommend(self.fusion.model, x)

        # 8) Report (grounded in findings, safety, recommendations, and reasoning)
        report = self.report.compose(vision, safety, recommendations, reasoning)

        # 9) Memory index (for similarity/priors)
        self.memory.index(case_id, vision.embedding, safety.top.value)

        state = CaseState.ABSTAINED if safety.abstained else CaseState.READY
        bundle = CaseBundle(
            case_id=case_id,
            study_id=study.study_id,
            state=state,
            priority_score=self._priority(safety.top, safety),
            priors=study.priors,
            image=[round(float(v), 4) for v in img.flatten()],
            image_shape=study.image_shape,
            vision=vision,
            evidence=evidence,
            fusion=fusion,
            safety=safety,
            explanation=explanation,
            reasoning=reasoning,
            recommendations=recommendations,
            report=report,
            multimodal=study.multimodal,
            ground_truth=study.ground_truth,
        )
        await self.bus.publish(ev.CASE_READY, case_id=case_id, study_id=study.study_id,
                               state=state.value)
        return bundle
