"""Seed the worklist with synthetic de-identified studies, incl. an OOD case.

Runs the full pipeline for each so the dashboard opens with a realistic worklist.
"""
from __future__ import annotations

import numpy as np

from ml.data import make_dataset, make_multimodal, make_ood_sample, IMG
from schemas.contracts import StudyInput, StructuredPriors
from gateway.pipeline import Pipeline
from gateway.storage import Store


async def seed(store: Store, pipeline: Pipeline, n: int = 12, seed: int = 202) -> int:
    if store.count() > 0:
        return store.count()

    samples = make_dataset(n, seed=seed)
    rng = np.random.default_rng(seed)
    created = 0
    for i, s in enumerate(samples):
        study = StudyInput(
            study_id=f"STU-{1000 + i}",
            image=[float(v) for v in s.image.flatten()],
            image_shape=(IMG, IMG),
            priors=s.priors,
            multimodal=make_multimodal(s.diagnosis, rng),
            ground_truth=s.diagnosis,
        )
        bundle = await pipeline.run(study, case_id=f"CASE-{1000 + i}")
        store.save_case(bundle)
        store.audit("case.analyzed", "case", bundle.case_id,
                    detail={"top": bundle.safety.top.value,
                            "abstained": bundle.safety.abstained})
        created += 1

    # One out-of-distribution study so the safety/abstention path is visible.
    ood = make_ood_sample()
    study = StudyInput(
        study_id="STU-OOD-1", image=[float(v) for v in ood.flatten()],
        image_shape=(IMG, IMG), priors=StructuredPriors(age_band="40-65"),
    )
    bundle = await pipeline.run(study, case_id="CASE-OOD-1")
    store.save_case(bundle)
    store.audit("case.analyzed", "case", bundle.case_id,
                detail={"abstained": bundle.safety.abstained,
                        "reason": bundle.safety.abstention_reason.value})
    created += 1
    return created
