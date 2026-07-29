from __future__ import annotations

import time
import numpy as np
import cv2
from pydantic import BaseModel
from typing import Dict, List, Optional

from aura.common.config import get_settings
from aura.common.mathx import softmax
from aura.schemas.clinical import DIAGNOSES, Diagnosis, Finding
from aura.knowledge.guidelines.templates import GUIDELINE_TEMPLATES
from ..recommend.engine import RecommendEngine, CATALOG, _COST_W, _RISK_W

class DecisionReadinessProfile(BaseModel):
    status: str  # "READY" or "NOT_READY"
    coverage: float
    quality: float
    consistency: float
    robustness: float
    stability: float
    consensus: float
    limiting_factor: str
    limiting_dimension: str
    edv: float
    evidence_dependency_profile: Dict[str, Dict[str, float]]

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()
    q = np.clip(q, 1e-9, 1.0)
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)

class ClinicalDecisionReadinessEngine:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.min_coverage = self.settings.min_coverage
        self.recommend_engine = RecommendEngine()

    def evaluate_coverage(self, primary_dx: Diagnosis, findings_map: dict, study) -> float:
        template = GUIDELINE_TEMPLATES.get(primary_dx)
        if not template:
            return 1.0
        
        total = len(template.imaging) + len(template.symptoms) + len(template.labs)
        if total == 0:
            return 1.0

        known = 0
        # 1. Imaging: vision findings are always known
        for f in template.imaging:
            if f in findings_map:
                known += 1
                
        # 2. Multimodal context
        if study.multimodal is not None:
            # Symptoms are always boolean in study.multimodal.symptoms
            for s in template.symptoms:
                known += 1
            # Labs are known if value is not None
            for l in template.labs:
                if getattr(study.multimodal.labs, l, None) is not None:
                    known += 1

        return float(known / total)

    def evaluate_quality(self, primary_dx: Diagnosis, study, img: np.ndarray) -> float:
        current_time = time.time()
        freshness_scores = []
        
        # 1. Calculate decay (freshness) of labs
        template = GUIDELINE_TEMPLATES.get(primary_dx)
        if template and study.multimodal is not None and study.multimodal.labs is not None:
            timestamps = getattr(study.multimodal.labs, "timestamps", {}) or {}
            for l in template.labs:
                val = getattr(study.multimodal.labs, l, None)
                if val is not None:
                    t_lab = timestamps.get(l)
                    if t_lab is not None:
                        dt = max(0.0, current_time - t_lab)
                        freshness = float(np.exp(-np.log(2.0) * dt / 86400.0))  # 24 hours half-life
                        freshness_scores.append(freshness)
                    else:
                        freshness_scores.append(1.0)
                        
        freshness_factor = np.mean(freshness_scores) if freshness_scores else 1.0

        # 2. Check image quality
        modality = getattr(study, "modality", None)
        is_cxr = True
        if modality is not None:
            is_cxr = (modality.value == "CXR" if hasattr(modality, "value") else str(modality) == "CXR")
            
        if is_cxr:
            from ..vision.xray_gate import _structural_score
            score, _ = _structural_score(img)
            img_quality = float(score / 3.0)
        else:
            # MR quality from payload
            payload = getattr(study, "payload", None) or study
            img_quality = 1.0
            if hasattr(payload, "series") and payload.series:
                scores = []
                for s in payload.series:
                    if hasattr(s, "quality") and hasattr(s.quality, "quality_score"):
                        scores.append(s.quality.quality_score)
                if scores:
                    img_quality = float(np.mean(scores))

        return float(min(freshness_factor, img_quality))

    def evaluate_consistency(self, primary_dx: Diagnosis, reasoning) -> float:
        if not reasoning or not reasoning.steps:
            return 1.0
            
        supporting_sum = 0.0
        refuting_sum = 0.0
        
        for step in reasoning.steps:
            weight = step.effect.get(primary_dx, 0.0)
            if weight > 0.05:
                supporting_sum += weight
            elif weight < -0.05:
                refuting_sum += abs(weight)
                
        if supporting_sum + refuting_sum > 0:
            return float(supporting_sum / (supporting_sum + refuting_sum))
        return 1.0

    def evaluate_edv(self, fusion_model, x: np.ndarray) -> float:
        # Evaluate remaining tests from recommend catalog
        best_dv = 0.0
        for item in CATALOG:
            if self.recommend_engine._resolvable(x, item["channels"]):
                evoi, _ = self.recommend_engine._evoi_and_eig(fusion_model, x, item["channels"])
                cost_penalty = _COST_W[item["cost"]] * _RISK_W[item["risk"]]
                dv = evoi - cost_penalty
                if dv > best_dv:
                    best_dv = dv
        return float(best_dv)

    def evaluate_robustness(self, primary_dx: Diagnosis, reasoning) -> float:
        if not reasoning or not reasoning.steps:
            return 1.0
            
        w_clinical = np.array([step.effect.get(primary_dx, 0.0) for step in reasoning.steps], dtype=float)
        
        # Calculate observed influence using LOO loop on reasoning posterior
        p0 = np.array([max(1e-9, reasoning.prior_posterior.get(d, 0.0)) for d in DIAGNOSES], dtype=float)
        p0 = p0 / p0.sum()
        base_logit = np.log(p0)
        
        full_logit = base_logit.copy()
        for step in reasoning.steps:
            for i, d in enumerate(DIAGNOSES):
                full_logit[i] += step.effect.get(d, 0.0)
                
        full_prob = softmax(full_logit)
        p_full = full_prob[DIAGNOSES.index(primary_dx)]
        
        w_observed = []
        for step in reasoning.steps:
            loo_logit = full_logit.copy()
            for i, d in enumerate(DIAGNOSES):
                loo_logit[i] -= step.effect.get(d, 0.0)
            loo_prob = softmax(loo_logit)
            p_loo = loo_prob[DIAGNOSES.index(primary_dx)]
            w_observed.append(p_full - p_loo)
            
        w_observed = np.array(w_observed, dtype=float)
        
        # Normalize weights
        sum_abs_clinical = np.sum(np.abs(w_clinical))
        sum_abs_observed = np.sum(np.abs(w_observed))
        
        w_clinical_norm = w_clinical / sum_abs_clinical if sum_abs_clinical > 0 else w_clinical
        w_observed_norm = w_observed / sum_abs_observed if sum_abs_observed > 0 else w_observed
        
        return float(1.0 - 0.5 * np.sum(np.abs(w_clinical_norm - w_observed_norm)))

    def evaluate_stability(self, primary_dx: Diagnosis, findings_map: dict, reasoning, study, img: np.ndarray, fusion_engine, x: np.ndarray, vision_engine) -> float:
        # 1. Clinical value perturbations
        clinical_same = 0
        rng = np.random.default_rng(self.settings.seed)
        for _ in range(5):
            xp = np.clip(x + rng.normal(0.0, 0.05, size=x.shape), 0.0, 1.0)
            p_perturbed, _ = fusion_engine.model.fuse(xp)
            top_perturbed = max(p_perturbed, key=p_perturbed.get)
            if top_perturbed == safety_top_dx(p_perturbed): # Helper to align
                pass
            # Just verify if top diagnosis is same as primary_dx
            p_array = np.array([p_perturbed.get(d, 0.0) for d in DIAGNOSES])
            if DIAGNOSES[np.argmax(p_array)] == primary_dx:
                clinical_same += 1
        clinical_stability = clinical_same / 5.0

        # 2. Image perturbations (noise, rotation)
        image_stability = 1.0
        if vision_engine is not None and img.ndim == 2:
            try:
                # Rotate image slightly
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), 3.0, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # Add Gaussian noise
                noise = np.random.normal(0, 0.02, img.shape)
                perturbed_img = np.clip(rotated + noise, 0.0, 1.0)
                
                # Run vision and fusion
                vision_perturbed = vision_engine.analyze(study.study_id, perturbed_img)
                from ..fusion.evidence import encode
                xp = encode(vision_perturbed, study.priors)
                p_perturbed, _ = fusion_engine.model.fuse(xp)
                p_array = np.array([p_perturbed.get(d, 0.0) for d in DIAGNOSES])
                if DIAGNOSES[np.argmax(p_array)] != primary_dx:
                    image_stability = 0.0
            except Exception:
                image_stability = 1.0

        # 3. Rule deletion stability
        node_stability = 1.0
        if reasoning and reasoning.steps:
            node_same = 0
            # Full base logit
            p0 = np.array([max(1e-9, reasoning.prior_posterior.get(d, 0.0)) for d in DIAGNOSES], dtype=float)
            p0 = p0 / p0.sum()
            base_logit = np.log(p0)
            
            for step_to_delete in reasoning.steps:
                logit = base_logit.copy()
                for step in reasoning.steps:
                    if step is step_to_delete:
                        continue
                    for i, d in enumerate(DIAGNOSES):
                        logit[i] += step.effect.get(d, 0.0)
                prob = softmax(logit)
                if DIAGNOSES[np.argmax(prob)] == primary_dx:
                    node_same += 1
            node_stability = node_same / len(reasoning.steps)

        return float((clinical_stability + image_stability + node_stability) / 3.0)

    def evaluate_consensus(self, fusion_engine, x: np.ndarray) -> float:
        if fusion_engine.quantum is not None and fusion_engine.classical is not None:
            try:
                p_poe = softmax(fusion_engine.classical.logits(x))
                p_vqc = softmax(fusion_engine.quantum.logits(x))
                cai_js = jensen_shannon_divergence(p_poe, p_vqc)
                return float(1.0 - cai_js)
            except Exception:
                return 1.0
        return 1.0

    def assess(self, study, img: np.ndarray, x: np.ndarray, fusion_engine, reasoning, findings_map: dict, safety_assessment, vision_engine=None) -> DecisionReadinessProfile:
        primary_dx = safety_assessment.top
        
        S_coverage = self.evaluate_coverage(primary_dx, findings_map, study)
        S_quality = self.evaluate_quality(primary_dx, study, img)
        S_consistency = self.evaluate_consistency(primary_dx, reasoning)
        S_robustness = self.evaluate_robustness(primary_dx, reasoning)
        S_stability = self.evaluate_stability(primary_dx, findings_map, reasoning, study, img, fusion_engine, x, vision_engine)
        S_consensus = self.evaluate_consensus(fusion_engine, x)
        
        # Calculate Expected Decision Value
        edv_score = self.evaluate_edv(fusion_engine.model, x)

        metrics = {
            "coverage": S_coverage,
            "quality": S_quality,
            "consistency": S_consistency,
            "robustness": S_robustness,
            "stability": S_stability,
            "consensus": S_consensus
        }
        
        limiting_dimension = min(metrics, key=metrics.get)
        lowest_score = metrics[limiting_dimension]
        
        # Determine readiness status (using configured min_coverage as threshold)
        status = "READY" if lowest_score >= self.min_coverage else "NOT_READY"
        
        limiting_factor_map = {
            "coverage": "Missing clinical evidence/indicators",
            "quality": "Low evidence quality or stale lab data",
            "consistency": "Conflicting clinical evidence",
            "robustness": "Unstable clinical reasoning posterior",
            "stability": "Sensitivity to input perturbations",
            "consensus": "Disagreement between quantum and classical fusion heads"
        }
        limiting_factor = limiting_factor_map[limiting_dimension]
        
        # Compile Evidence Dependency Profile (EDP)
        # Indicate relative observed influence vs clinical expectation for each template indicator that is present in reasoning steps
        edp = {}
        if reasoning and reasoning.steps:
            # We map finding names to observed and clinical weights
            for step in reasoning.steps:
                for f in step.evidence:
                    # check if finding represents an active finding
                    clinical = step.effect.get(primary_dx, 0.0)
                    # For observed influence, look at the LOO diff
                    p0 = np.array([max(1e-9, reasoning.prior_posterior.get(d, 0.0)) for d in DIAGNOSES], dtype=float)
                    p0 = p0 / p0.sum()
                    base_logit = np.log(p0)
                    
                    full_logit = base_logit.copy()
                    for s in reasoning.steps:
                        for i, d in enumerate(DIAGNOSES):
                            full_logit[i] += s.effect.get(d, 0.0)
                    p_full = softmax(full_logit)[DIAGNOSES.index(primary_dx)]
                    
                    loo_logit = full_logit.copy()
                    for i, d in enumerate(DIAGNOSES):
                        loo_logit[i] -= step.effect.get(d, 0.0)
                    p_loo = softmax(loo_logit)[DIAGNOSES.index(primary_dx)]
                    observed = p_full - p_loo
                    
                    edp[f] = {
                        "observed": float(observed),
                        "clinical": float(clinical)
                    }

        return DecisionReadinessProfile(
            status=status,
            coverage=S_coverage,
            quality=S_quality,
            consistency=S_consistency,
            robustness=S_robustness,
            stability=S_stability,
            consensus=S_consensus,
            limiting_factor=limiting_factor,
            limiting_dimension=limiting_dimension,
            edv=edv_score,
            evidence_dependency_profile=edp
        )

def safety_top_dx(p_perturbed: dict) -> Diagnosis:
    # helper for type alignment
    p_array = np.array([p_perturbed.get(d, 0.0) for d in DIAGNOSES])
    return DIAGNOSES[np.argmax(p_array)]
