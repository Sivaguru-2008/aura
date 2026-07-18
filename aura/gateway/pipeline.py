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
from common.eventbus import EventBus
from common.mathx import entropy, softmax
from schemas.clinical import Diagnosis
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

    def __init__(self, bus: EventBus | None = None, memory: MemoryEngine | None = None):
        self.bus = bus or EventBus()
        self.vision = VisionEngine.load()
        self.fusion = FusionEngine()
        self.safety = SafetyEngine()
        self.explain = ExplainEngine()
        self.recommend = RecommendEngine()
        self.reasoner = ClinicalReasoner()
        self.report = ReportEngine()
        self.memory = memory or MemoryEngine()

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

        # 3) Safety (calibration, conformal, OOD, abstention)
        safety = self.safety.assess(study.study_id, x, self.fusion.model)

        # 4) Explainability
        explanation = self.explain.explain(
            study.study_id, self.vision, img, self.fusion.model, x, safety.top
        )

        # 5) Missing-evidence recommendations
        recommendations = self.recommend.recommend(self.fusion.model, x)

        # 6) Clinical reasoning — fuse imaging with labs/symptoms/history + guidelines.
        findings_map = {fs.finding: fs.probability for fs in vision.findings}
        imaging_prior = {p.diagnosis: p.probability for p in safety.predictions}
        reasoning = self.reasoner.reason(
            study.study_id, findings_map, imaging_prior, study.priors, study.multimodal
        )

        # 7) Report (grounded in findings, safety, recommendations, and reasoning)
        report = self.report.compose(vision, safety, recommendations, reasoning)

        # 7) Memory index (for similarity/priors)
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
