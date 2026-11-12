"""Nightly Recalibration Service — updates calibration parameters using clinician feedback."""
from __future__ import annotations

import os
import shutil
from datetime import datetime

import numpy as np

from aura.common.config import get_settings
from aura.schemas.clinical import DIAGNOSES, Diagnosis
from aura.services.safety.calibration import backend_calibration_path

# Reuse the backend loader and the canonical fit from the recalibrate_backend
# script. Re-implementing the fit here would give calibration two definitions
# that could drift apart silently.
from aura.ml.training.recalibrate_backend import (
    BACKENDS,
    _evidence_splits,
    _load_backend,
    calibrate,
)

#: Below this many clinician-labelled cases the run is a no-op. Recalibration
#: overwrites the served safety envelope, so it must be driven by new evidence.
MIN_FEEDBACK_SAMPLES = max(1, int(os.environ.get("AURA_MIN_RECALIBRATION_FEEDBACK", "10")))


def _backup_calibration(backend: str) -> str | None:
    """Copy safety_<backend>.npz aside so a bad fit can be rolled back."""
    src = backend_calibration_path(backend)
    if not src.exists():
        return None
    dst = src.with_suffix(f".{datetime.now():%Y%m%d-%H%M%S}.bak.npz")
    shutil.copy2(src, dst)
    return dst.name

def recalibrate_from_feedback(store) -> dict:
    """Run recalibration by loading base splits and appending clinician feedback."""
    settings = get_settings()

    # 1. Load baseline calibration/test splits
    try:
        Xcal, ycal, Xte, yte, source = _evidence_splits(n=900, seed=settings.seed)
        # Convert to list so we can append easily
        Xcal = list(Xcal)
        ycal = list(ycal)
    except Exception:
        # Fallback to generating synthetic splits if the cached splits fail to load
        from aura.ml.training.dataset import make_splits, build_evidence_dataset
        tr, cal, te = make_splits(n_samples=700, seed=settings.seed)
        _, _ = build_evidence_dataset(tr)
        Xcal_arr, ycal_arr = build_evidence_dataset(cal)
        Xte, yte = build_evidence_dataset(te)
        Xcal = list(Xcal_arr)
        ycal = list(ycal_arr)
        source = "synthetic-fallback"

    # 2. Read clinician feedback through the store. This used to open a raw sqlite3
    #    handle on store.db_path, which stops being the case store the moment
    #    AURA_DATABASE_URL points at PostgreSQL: the file at db_path is then only the
    #    local EHR fixture, and the query would find no feedback table at all.
    feedback_samples = []
    for row in store.list_feedback():
        case_id = row["case_id"]
        verdict = str(row["verdict"]).upper()
        correction = row["correction"]

        # Fetch the parsed case bundle
        case = store.get_case(case_id)
        if case is None or not hasattr(case, "fusion") or case.fusion is None:
            continue

        # Retrieve the evidence vector (inputs to the fusion model)
        x = getattr(case.fusion, "evidence_vector", None)
        if x is None or len(x) == 0:
            continue

        # Determine corrected label index
        y = None
        if verdict in ("INCORRECT", "REJECT") and correction:
            try:
                corrected_dx = Diagnosis(correction.lower())
                y = DIAGNOSES.index(corrected_dx)
            except (ValueError, TypeError):
                pass
        elif verdict in ("CORRECT", "ACCEPT", "APPROVE"):
            try:
                top_dx = Diagnosis(case.safety.top.value.lower())
                y = DIAGNOSES.index(top_dx)
            except (ValueError, TypeError):
                pass

        if y is not None:
            feedback_samples.append((np.asarray(x, dtype=float), int(y)))

    # 3. Append feedback samples to calibration set
    new_added = len(feedback_samples)
    if new_added < MIN_FEEDBACK_SAMPLES:
        # Refitting on the base splits alone would rewrite safety_<backend>.npz
        # with no new information, moving the temperature and conformal threshold
        # the abstention gate depends on for nothing. Do not touch the artifacts.
        return {
            "status": "skipped",
            "reason": (
                f"{new_added} usable feedback sample(s); "
                f"{MIN_FEEDBACK_SAMPLES} required before recalibrating."
            ),
            "feedback_samples_added": new_added,
            "data_source": source,
            "results": {},
        }

    for x, y in feedback_samples:
        Xcal.append(x)
        ycal.append(y)

    Xcal = np.array(Xcal)
    ycal = np.array(ycal)
    Xte = np.array(Xte)
    yte = np.array(yte)

    # 4. Fit calibration per available backend, using the canonical fit. Each
    #    artifact is copied aside first so a bad fit is recoverable.
    results = {}
    backups = {}
    for backend in BACKENDS:
        if _load_backend(backend) is None:
            continue
        backups[backend] = _backup_calibration(backend)
        summary = calibrate(backend, Xcal, ycal, Xte, yte, settings.conformal_coverage)
        if summary is not None:
            results[backend] = summary

    # Write audit event
    store.audit(
        "calibration.recalibrated",
        entity_type="system",
        detail={
            "feedback_samples_added": new_added,
            "total_calibration_samples": len(ycal),
            "backends_updated": list(results.keys()),
            "backups": backups,
        }
    )

    return {
        "status": "success",
        "feedback_samples_added": new_added,
        "total_calibration_samples": len(ycal),
        "data_source": source,
        "backups": backups,
        "results": results,
    }
