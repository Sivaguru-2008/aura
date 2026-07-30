"""The analysis pipeline — orchestrates the engines for one study.

This is the operational form of docs/ARCHITECTURE.md section 7. It wires the
engines together and emits events at each stage; in production each stage is a
separate service consuming the prior stage's event. Here they run in-process,
but the boundaries and the event contract are identical, so extraction to
independent services is a deployment change, not a rewrite.
"""
from __future__ import annotations

import logging

import numpy as np

from aura.common import eventbus as ev
from aura.common.config import get_settings
from aura.common.eventbus import EventBus
from aura.common.mathx import entropy, softmax
from aura.schemas.clinical import DIAGNOSES, Diagnosis
from aura.schemas.contracts import (
    CaseBundle,
    CaseState,
    MeasurementBudget,
    StructuredPriors,
    StudyInput,
)
from aura.services.explain import ExplainEngine
from aura.services.fusion import FusionEngine
from aura.services.fusion.evidence import encode, to_evidence_items
from aura.services.fusion.qmba import QuantumMeasurementBudget
from aura.services.memory import MemoryEngine
from aura.services.reasoning import ClinicalReasoner
from aura.services.recommend import RecommendEngine
from aura.services.report import ReportEngine
from aura.services.safety import SafetyEngine, ClinicalSafetyController, ClinicalDecisionReadinessEngine
from aura.services.vision import VisionEngine
from aura.schemas.clinical import Finding
from aura.services.agent.consensus import ConsensusEngine


class SafetyVeto(Exception):
    """Raised when the ClinicalSafetyController blocks pipeline continuation."""
    def __init__(self, controller_output):
        self.controller_output = controller_output
        super().__init__(f"Safety controller {controller_output.state}: {controller_output.recommendation}")


log = logging.getLogger("aura.pipeline")


class Pipeline:
    """Holds one instance of each engine (loaded models) and runs cases through them."""

    def __init__(self, bus: EventBus | None = None, memory: MemoryEngine | None = None,
                 store=None):
        self.bus = bus or EventBus()
        self.vision = VisionEngine.load()
        self.fusion = FusionEngine()
        # Safety calibration is a property of the fusion backend's logits, so it must
        # follow the backend the fusion engine *actually resolved to* — not the raw
        # setting. FusionEngine falls back to classical when quantum artifacts are
        # absent; binding safety to `settings.fusion_backend` instead would then scale
        # classical logits with the quantum temperature (0.46 vs 0.99), silently
        # distorting every served probability, conformal set and abstention decision.
        self.safety = SafetyEngine(backend=self.fusion.backend)
        # Measurement-budgeted decisions. Only meaningful on a quantum backend: it
        # sequences shot budgets against the shot-noise spread of the decision margin,
        # and a product-of-experts has no shot noise to sequence. Bound to the
        # *resolved* backend for the same reason safety is (see above) — if the
        # quantum artifacts were absent and fusion fell back to classical, running
        # QMBA anyway would report a measurement budget for a model that has none.
        self.measurement_budget = (
            QuantumMeasurementBudget(self.fusion.model)
            if self.fusion.backend == "quantum" else None
        )
        # Both read their own config (settings / safety policy) at construction.
        self.safety_controller = ClinicalSafetyController()
        self.cdre = ClinicalDecisionReadinessEngine()
        self.explain = ExplainEngine()
        self.recommend = RecommendEngine()
        self.reasoner = ClinicalReasoner()
        self.report = ReportEngine()
        self.consensus = ConsensusEngine()
        self.memory = memory or MemoryEngine(store=store)
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

    def _measure_budget(self, x) -> MeasurementBudget | None:
        """Run the sequential shot schedule for one evidence vector.

        Returns None on a classical backend (nothing to budget) and, deliberately,
        also on any failure: measurement economics are *reporting*, not a gate. The
        served posterior is the analytic one either way, so a QMBA problem must never
        be able to fail a study — it can only fail to annotate one.
        """
        if self.measurement_budget is None:
            return None
        try:
            d = self.measurement_budget.decide(x)
            return MeasurementBudget(
                committed=d.committed,
                top=Diagnosis(d.top),
                runner_up=Diagnosis(d.runner_up),
                shots_spent=d.shots_spent,
                margin=float(d.margin),
                margin_std=float(d.margin_std),
                separation_z=float(d.separation_z),
                analytic_margin=float(d.analytic_margin),
                predicted_shots=d.predicted_shots,
                limiting_factor=d.limiting_factor,
                floor_limited=bool(d.floor_limited),
                reason=d.reason,
                trajectory=[s.to_dict() for s in d.trajectory],
            )
        except Exception as exc:                     # pragma: no cover - defensive
            log.warning("measurement budget skipped: %s: %s", type(exc).__name__, exc)
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

        # 2.25) Measurement economics (quantum backends only).
        # Costs one extra circuit evaluation — every budget stage reuses the same
        # expectations and only re-draws shot noise, because the *state* does not
        # change with the budget, only how precisely it can be read. The analytic
        # posterior above remains what is served; this answers the separate question
        # of whether an unresolved case is unresolved for want of measurement or for
        # want of a better model, which are different instructions to the clinician.
        measurement = self._measure_budget(x)
        if measurement is not None:
            await self.bus.publish(
                "fusion.measurement_budgeted",
                study_id=study.study_id,
                shots=measurement.shots_spent,
                limiting_factor=measurement.limiting_factor or "committed",
            )

        # 2.5) Clinical Safety Controller (Layer 1) — veto gate BEFORE reasoning
        resolved_logits = self.fusion.resolved_logits(x, fusion)
        ood_logits = self.fusion.model.logits(x)
        epistemic = self.safety._epistemic_ensemble(x)
        controller_output = self.safety_controller.check(
            evidence_vector=x,
            logits=ood_logits,
            temperature=self.safety.cal.temperature,
            ood_mean=self.safety.cal.ood_mean,
            ood_std=self.safety.cal.ood_std,
            epistemic_std=float(epistemic["epistemic_std"]),
            epistemic_mi=float(epistemic["epistemic_mi"]),
        )
        await self.bus.publish("safety.controller_completed",
                               study_id=study.study_id, state=controller_output.state)

        # If the controller vetoes, abort the pipeline immediately
        if controller_output.state == "FAILED":
            # Still produce a minimal bundle for audit trail
            safety = self.safety.assess(
                study.study_id, x, self.fusion.model,
                resolved_logits=resolved_logits, aci_qhat=self._aci_qhat(),
            )
            bundle = CaseBundle(
                case_id=case_id,
                study_id=study.study_id,
                state=CaseState.ABSTAINED,
                priority_score=1.0,
                priors=study.priors,
                image=[round(float(v), 4) for v in img.flatten()],
                image_shape=study.image_shape,
                vision=vision,
                evidence=evidence,
                fusion=fusion,
                measurement=measurement,
                safety=safety,
                safety_controller=controller_output,
                multimodal=study.multimodal,
                ground_truth=study.ground_truth,
            )
            await self.bus.publish(ev.CASE_READY, case_id=case_id,
                                   study_id=study.study_id, state="abstained")
            return bundle

        # 3) Clinical reasoning — fuse the calibrated imaging posterior with
        # labs/symptoms/history + guideline likelihood ratios, BEFORE safety, so the
        # posterior that safety validates is the *final* one shown to the clinician
        # (audit F10). With no multimodal evidence the reasoner is inert and the
        # adjusted posterior equals the imaging posterior — imaging behaviour is
        # unchanged. The conflict-guard-resolved logits feed this (audit F2).
        imaging_probs = self.safety.calibrated_posterior(resolved_logits)
        imaging_prior = {d: float(imaging_probs[i]) for i, d in enumerate(DIAGNOSES)}
        findings_map = {fs.finding: fs.probability for fs in vision.findings}
        reasoning, evidence_graph = self.reasoner.reason(
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

        # 7) Clinical Decision Readiness Engine (CDRE) — Layer 2
        # Extract classical/quantum logits for consensus agreement
        classical_logits = None
        quantum_logits = None
        if hasattr(self.fusion, "model"):
            classical_logits = self.fusion.model.logits(x)
        if hasattr(self.fusion, "_quantum_model") and self.fusion._quantum_model is not None:
            quantum_logits = self.fusion._quantum_model.logits(x)

        # 7.5) Multi-Agent Clinical Consensus Panel
        mm = study.multimodal
        agent_evidence = {
            "findings": findings_map,
            "embedding": vision.embedding.tolist() if hasattr(vision.embedding, "tolist") else [],
            "labs": mm.labs.model_dump() if mm and mm.labs else {},
            "symptoms": mm.symptoms.model_dump() if mm and mm.symptoms else {},
        }
        priors_for_agents = study.priors.model_dump() if hasattr(study.priors, "model_dump") else {}
        consensus_result = await self.consensus.evaluate(
            evidence=agent_evidence,
            priors=priors_for_agents,
            modality=study.modality,
        )

        decision_readiness = self.cdre.evaluate(
            reasoning=reasoning,
            evidence_graph=evidence_graph,
            recommendations=recommendations,
            vision_quality=None,
            fusion_model=self.fusion.model,
            evidence_vector=x,
            classical_logits=classical_logits,
            quantum_logits=quantum_logits,
            consensus_result=consensus_result,
        )

        # 8) Report (grounded in findings, safety, recommendations, reasoning, and panel discussion)
        report = self.report.compose(vision, safety, recommendations, reasoning, consensus_result)

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
            evidence_graph=evidence_graph,
            fusion=fusion,
            measurement=measurement,
            safety=safety,
            explanation=explanation,
            reasoning=reasoning,
            recommendations=recommendations,
            report=report,
            safety_controller=controller_output,
            decision_readiness=decision_readiness,
            drp=decision_readiness,
            consensus_result=consensus_result.to_dict() if consensus_result else None,
            multimodal=study.multimodal,
            ground_truth=study.ground_truth,
        )

        # Log decision provenance
        if self.store is not None:
            try:
                self.store.log_decision_provenance(case_id, {
                    "study_id": study.study_id,
                    "controller_state": controller_output.state,
                    "controller_confidence": controller_output.safety_confidence,
                    "readiness_state": decision_readiness.state.value if hasattr(decision_readiness.state, 'value') else str(decision_readiness.state),
                    "readiness_limiting": decision_readiness.limiting_factor,
                    "safety_abstained": safety.abstained,
                    "top_diagnosis": safety.top.value,
                })
            except Exception:
                pass

        await self.bus.publish(ev.CASE_READY, case_id=case_id, study_id=study.study_id,
                               state=state.value)
        return bundle
