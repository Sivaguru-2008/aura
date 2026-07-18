"""Step 5 — Unified Patient Object: the single interface AURA consumes.

This is the honest, real-data replacement for a synthetic ``Sample``. One
``Patient`` bundles everything known about a real MIMIC-CXR subject and knows how
to emit the pipeline's *existing* ``StudyInput`` contract — so every downstream
engine (vision, fusion, safety, reasoning, report) runs unchanged.

Fields follow the plan's schema. Where MIMIC-CXR simply has no data for a field
(``hadm_id``, ``stay_id``, ``age``, ``race``, ``labs``, ``vitals``, ``meds``,
``procedures``), it is present but ``None``/empty and documented — never faked.
Radiology reports serve as both ``radiology`` and ``notes``.

The image → ``StudyInput`` conversion delegates to
``services.vision.io.study_from_cxr`` (the built-in PACS→pipeline seam), so the
Vision/CNN pipeline is not touched.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional

from mimic.cleaning import CleanedPatient, clean_record
from mimic.config import MimicPaths, get_mimic_paths
from mimic.labeling import ReportLabel
from mimic.loaders import MimicCxrLoader, PatientRecord
from mimic.timeline import PatientTimeline, StudyEvent, build_timeline
from schemas.clinical import Diagnosis, Finding
from schemas.contracts import MultimodalContext, StructuredPriors, StudyInput

log = logging.getLogger("mimic.patient")

# Preference order when choosing one representative image for a study.
_VIEW_PREFERENCE = ("PA", "AP", "Lateral", "Unknown")


@dataclass
class Patient:
    """One real MIMIC-CXR patient — AURA's single unit of input."""

    # --- identity ---
    subject_id: int
    patient_id: str = ""                       # string alias used across services
    # --- MIMIC-IV linkage fields (absent in MIMIC-CXR; kept for schema parity) ---
    hadm_id: Optional[int] = None
    stay_id: Optional[int] = None
    # --- demographics (not in the aug CSV; require patients.csv, not present) ---
    age: Optional[int] = None
    age_band: str = "unknown"
    sex: str = "unknown"
    race: str = "unknown"
    # --- clinical content actually present ---
    images: list[str] = field(default_factory=list)          # on-disk, deduped
    images_by_view: dict[str, list[str]] = field(default_factory=dict)
    radiology: list[str] = field(default_factory=list)        # radiology reports
    notes: list[str] = field(default_factory=list)            # == radiology here
    diagnosis: Diagnosis = Diagnosis.NORMAL                   # patient-level label
    findings: list[Finding] = field(default_factory=list)
    label: ReportLabel = field(default_factory=ReportLabel)
    timeline: PatientTimeline = field(default_factory=lambda: PatientTimeline(subject_id=-1))
    # --- MIMIC-IV tables that do not exist for this corpus (schema parity) ---
    admissions: list = field(default_factory=list)
    diagnoses: list = field(default_factory=list)
    labs: list = field(default_factory=list)
    vitals: list = field(default_factory=list)
    medications: list = field(default_factory=list)
    procedures: list = field(default_factory=list)
    # --- provenance ---
    quality_flags: list[str] = field(default_factory=list)
    _paths: Optional[MimicPaths] = None

    # ------------------------------------------------------------------ #
    @property
    def n_studies(self) -> int:
        return self.timeline.n_studies

    def priors(self) -> StructuredPriors:
        """De-identified priors AURA's fusion accepts. Demographics unknown here."""
        return StructuredPriors(age_band=self.age_band, sex=self.sex)

    # ------------------------------------------------------------------ #
    # Image selection
    # ------------------------------------------------------------------ #
    def _representative_image(self, event: StudyEvent) -> Optional[str]:
        """Pick the best single view for a study: frontal (PA/AP) over lateral."""
        for view in _VIEW_PREFERENCE:
            if event.views.get(view):
                return event.views[view][0]
        # fall back to any image the event has
        return event.images[0] if event.images else None

    # ------------------------------------------------------------------ #
    # StudyInput adapter — the seam into the untouched pipeline
    # ------------------------------------------------------------------ #
    def to_study_input(self, study_index: int = -1, grid: int = 64) -> StudyInput:
        """Emit a pipeline-ready :class:`StudyInput` for one study on the timeline.

        Loads the real radiograph via the existing ``services.vision.io`` seam,
        attaches this patient's priors and the report-derived ground-truth label.
        ``study_index=-1`` selects the most recent study.
        """
        from services.vision.io import study_from_cxr  # lazy: pulls numpy/PIL

        if self.n_studies == 0:
            raise ValueError(f"patient {self.subject_id} has no studies")
        event = self.timeline.events[study_index]
        rel = self._representative_image(event)
        if rel is None:
            raise ValueError(f"study {event.study_id} has no usable image")
        paths = self._paths or get_mimic_paths()
        abs_path = paths.resolve_image(rel)

        si = study_from_cxr(
            abs_path,
            study_id=event.study_id,
            priors=self.priors(),
            grid=grid,
        )
        # Attach the real ground-truth label + (empty) multimodal context.
        si.ground_truth = event.label.diagnosis
        si.multimodal = self.multimodal_context()
        return si

    def iter_study_inputs(self, grid: int = 64) -> Iterator[StudyInput]:
        """Yield a :class:`StudyInput` for every study, in chronological order."""
        for i in range(self.n_studies):
            try:
                yield self.to_study_input(study_index=i, grid=grid)
            except (ValueError, OSError) as e:  # unreadable image -> skip, don't crash
                log.warning("subject %s study %d skipped: %s", self.subject_id, i, e)

    def multimodal_context(self) -> Optional[MultimodalContext]:
        """No labs/symptoms/history exist in MIMIC-CXR, so this is ``None``.

        Kept as a method (not a hardcoded ``None``) so a future MIMIC-IV linkage
        can populate real labs/symptoms without changing any caller.
        """
        return None


def build_patient(
    record: PatientRecord, paths: Optional[MimicPaths] = None
) -> Patient:
    """Assemble a :class:`Patient` from a raw loaded record (clean + label + timeline)."""
    cleaned: CleanedPatient
    cleaned, _ = clean_record(record)
    timeline = build_timeline(record)           # raw record: preserves alignment
    return Patient(
        subject_id=record.subject_id,
        patient_id=f"MIMIC-{record.subject_id}",
        images=cleaned.images,
        images_by_view=cleaned.images_by_view,
        radiology=cleaned.reports,
        notes=cleaned.reports,
        diagnosis=cleaned.diagnosis,
        findings=cleaned.positive_findings,
        label=cleaned.label,
        timeline=timeline,
        quality_flags=cleaned.quality_flags,
        _paths=paths or get_mimic_paths(),
    )


def iter_patients(
    split: str = "train",
    paths: Optional[MimicPaths] = None,
    limit: Optional[int] = None,
    **loader_kwargs,
) -> Iterator[Patient]:
    """Stream :class:`Patient` objects for a split — the top-level entry point.

    This is what replaces ``ml.data.make_dataset`` as AURA's source of patients.
    """
    paths = paths or get_mimic_paths()
    loader = MimicCxrLoader(split, paths=paths, **loader_kwargs)
    for rec in loader.iter_records(limit=limit):
        yield build_patient(rec, paths=paths)
