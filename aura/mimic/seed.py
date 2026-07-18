"""Step 13 — Seed the worklist with real MIMIC-CXR patients.

Drop-in replacement for ``gateway.seed.seed`` that runs the *same* pipeline over
*real* patients from :func:`mimic.patient.iter_patients` instead of the synthetic
``ml.data.make_dataset`` world. Identical side effects (``store.save_case`` +
``store.audit``) and an identical async signature, so the gateway calls it the
same way — see :func:`gateway.app` (switched via ``AURA_DATA_SOURCE``).

No API, schema, or service is modified; this only changes *where patients come
from*. Real studies currently trip the vision OOD guard and abstain (the vision
backbone is still calibrated on synthetic images — recalibration is gated by the
"don't touch vision yet" rule), which is the correct, safe behaviour, not a bug.
"""
from __future__ import annotations

import logging

from gateway.pipeline import Pipeline
from gateway.storage import Store
from mimic.config import get_mimic_paths
from mimic.patient import iter_patients

log = logging.getLogger("mimic.seed")


async def seed_mimic(
    store: Store,
    pipeline: Pipeline,
    n: int = 12,
    split: str = "validate",
    scan_limit: int = 400,
) -> int:
    """Populate the worklist with up to ``n`` real MIMIC-CXR cases.

    Idempotent: if the store already has cases, returns the current count without
    re-seeding (matches ``gateway.seed.seed``).
    """
    if store.count() > 0:
        return store.count()

    paths = get_mimic_paths()
    if not paths.exists_report().get("validate_csv"):
        log.warning("MIMIC-CXR not found at %s; worklist left empty", paths.root)
        return 0

    created = 0
    for patient in iter_patients(split, paths=paths, limit=scan_limit):
        if patient.n_studies == 0:
            continue
        try:
            study = patient.to_study_input(study_index=-1)
        except (ValueError, OSError) as e:
            log.warning("skip subject %s: %s", patient.subject_id, e)
            continue

        case_id = f"CASE-MIMIC-{patient.subject_id}"
        bundle = await pipeline.run(study, case_id=case_id)
        store.save_case(bundle)
        store.audit(
            "case.analyzed", "case", bundle.case_id,
            detail={
                "source": "mimic-cxr",
                "subject_id": patient.subject_id,
                "report_diagnosis": patient.diagnosis.value,
                "top": bundle.safety.top.value if bundle.safety else None,
                "abstained": bool(bundle.safety.abstained) if bundle.safety else None,
            },
        )
        created += 1
        if created >= n:
            break

    log.info("seeded %d real MIMIC-CXR cases (split=%s)", created, split)
    return created
