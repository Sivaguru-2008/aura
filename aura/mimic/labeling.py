"""Step 3 — Structured labels from free-text radiology reports.

The synthetic world handed every sample a ground-truth :class:`Diagnosis`. Real
MIMIC-CXR patients only have free-text reports, so we recover structured labels
from the text with a negation-aware, forward-scoping matcher in the spirit of
CheXpert / NegBio — kept transparent and dependency-free (pure ``re``).

Output is expressed entirely in AURA's *existing* vocabulary
(``schemas.clinical.Finding`` / ``Diagnosis``), so nothing downstream changes:
the label simply comes from a real report instead of a fabricated one.

Labels use the CheXpert convention per concept:
    ``1`` present · ``-1`` uncertain · ``0`` absent (negated) · missing = not mentioned.

This is deliberately rule-based and swappable — in production it maps to the
official CheXpert labeler or a fine-tuned report classifier; here it is auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from schemas.clinical import Diagnosis, Finding

# --------------------------------------------------------------------------- #
# Concept lexicon — regexes matched against lowercased sentences.
# Concepts marked (*) have no Finding-enum equivalent but drive the Diagnosis map.
# --------------------------------------------------------------------------- #
CONCEPT_PATTERNS: dict[str, str] = {
    "effusion": r"\beffusion|pleural fluid\b",
    "consolidation": r"\bconsolidat",
    "pneumonia": r"\bpneumonia|pneumonic\b",
    "opacity": r"\bopacit|opacification|infiltrat|airspace disease\b",
    "cardiomegaly": r"\bcardiomegaly\b|enlarge\w*\s+(?:the\s+)?(?:cardiac|heart|cardiomediastin)"
    r"|(?:cardiac|heart)\s+\w*\s*enlarge",
    "edema": r"\bedema|oedema|vascular congestion|pulmonary congestion|fluid overload\b",
    "pneumothorax": r"\bpneumothora",
    "nodule": r"\bnodul|\bmass(?:es)?\b|neoplasm|malignan|metasta",
    "atelectasis": r"\batelecta|volume loss\b",
    "hyperinflation": r"\bhyperinflat|hyperexpan|emphysema|hyperlucen|flattened diaphragm",
}

# Concepts that correspond to an AURA Finding (imaging observation).
CONCEPT_TO_FINDING: dict[str, Finding] = {
    "opacity": Finding.OPACITY,
    "consolidation": Finding.CONSOLIDATION,
    "effusion": Finding.EFFUSION,
    "cardiomegaly": Finding.CARDIOMEGALY,
    "nodule": Finding.NODULE,
    "pneumothorax": Finding.PNEUMOTHORAX,
    "hyperinflation": Finding.HYPERINFLATION,
}

# Forward-scoping cue lexicons.
_NEG = re.compile(
    r"\b(no|not|without|never|free of|negative for|clear of|absence of|absent|"
    r"ruled out|rule out|resolved|no evidence|no sign|no definite|no new|unremarkable)\b"
)
_UNC = re.compile(
    r"\b(may|maybe|possible|possibly|could|cannot exclude|cannot be excluded|"
    r"questionable|question of|suggest\w*|suspicious|concerning|likely|probable|"
    r"probably|worrisome|differential|versus|\bvs\b|borderline)\b"
)
_BREAK = re.compile(r"\b(but|however|except|although|though|otherwise|aside from|other than)\b|;")

_NORMAL_CUE = re.compile(
    r"no acute cardiopulmonary|no acute (?:cardiopulmonary )?(?:process|abnormalit\w*|"
    r"disease|finding\w*)|\blungs? (?:are|is) clear\b|\bclear lungs?\b|within normal limits|"
    r"\bunremarkable\b|normal chest"
)

# Clinical-acuity priority for choosing a single diagnosis when several fire.
_DX_PRIORITY: list[Diagnosis] = [
    Diagnosis.PNEUMOTHORAX,
    Diagnosis.MALIGNANCY,
    Diagnosis.HEART_FAILURE,
    Diagnosis.PNEUMONIA,
    Diagnosis.COPD,
    Diagnosis.NORMAL,
]

_SENT_SPLIT = re.compile(r"[.!?\n]+")
_PLACEHOLDER = re.compile(r"_{2,}")  # de-identification blanks "___"


@dataclass
class ReportLabel:
    """Structured labels extracted from one radiology report."""

    concepts: dict[str, int] = field(default_factory=dict)      # CheXpert-style per concept
    findings: dict[Finding, int] = field(default_factory=dict)  # subset mapped to Finding enum
    diagnosis: Diagnosis = Diagnosis.NORMAL
    diagnosis_scores: dict[Diagnosis, float] = field(default_factory=dict)
    normal_cue: bool = False
    n_sentences: int = 0

    @property
    def positive_findings(self) -> list[Finding]:
        return [f for f, v in self.findings.items() if v == 1]


def _label_in_scope(sentence: str, match_start: int) -> int:
    """Return 1/-1/0 for a concept match at ``match_start`` given forward-scoping cues.

    A cue scopes forward until the next scope-breaker; the nearest breaker before
    the match bounds which cues still apply. Negation dominates uncertainty.
    """
    breaks = [m.end() for m in _BREAK.finditer(sentence) if m.end() <= match_start]
    window_start = max(breaks) if breaks else 0
    window = sentence[window_start:match_start]
    if _NEG.search(window):
        return 0
    if _UNC.search(window):
        return -1
    return 1


def _combine(existing: Optional[int], new: int) -> int:
    """CheXpert aggregation across mentions: positive > uncertain > negative."""
    if existing is None:
        return new
    order = {1: 3, -1: 2, 0: 1}
    return existing if order[existing] >= order[new] else new


def label_report(text: str) -> ReportLabel:
    """Extract concepts, findings, and a single diagnosis from one report."""
    out = ReportLabel()
    if not text or not text.strip():
        out.diagnosis = Diagnosis.NORMAL
        return out
    clean = _PLACEHOLDER.sub(" ", text.lower())
    # Drop the "findings:"/"impression:" section headers so they don't skew cues.
    clean = clean.replace("findings:", " ").replace("impression:", " ")

    sentences = [s.strip() for s in _SENT_SPLIT.split(clean) if s.strip()]
    out.n_sentences = len(sentences)
    concepts: dict[str, int] = {}
    for sent in sentences:
        if _NORMAL_CUE.search(sent):
            out.normal_cue = True
        for concept, pat in CONCEPT_PATTERNS.items():
            m = re.search(pat, sent)
            if m:
                lbl = _label_in_scope(sent, m.start())
                concepts[concept] = _combine(concepts.get(concept), lbl)

    out.concepts = concepts
    out.findings = {
        CONCEPT_TO_FINDING[c]: v for c, v in concepts.items() if c in CONCEPT_TO_FINDING
    }
    out.diagnosis, out.diagnosis_scores = _map_diagnosis(concepts, out.normal_cue)
    return out


def _map_diagnosis(concepts: dict[str, int], normal_cue: bool) -> tuple[Diagnosis, dict[Diagnosis, float]]:
    """Map positive concepts to a diagnosis posterior + a single top label.

    Heuristic and transparent — fusion/reasoning refine this later; here it only
    needs to give each real patient an honest ground-truth label.
    """
    def pos(c: str) -> bool:
        return concepts.get(c) == 1

    scores: dict[Diagnosis, float] = {d: 0.0 for d in Diagnosis}
    if pos("pneumothorax"):
        scores[Diagnosis.PNEUMOTHORAX] += 1.0
    if pos("nodule"):
        scores[Diagnosis.MALIGNANCY] += 1.0
    if pos("edema"):
        scores[Diagnosis.HEART_FAILURE] += 0.7
    if pos("cardiomegaly"):
        scores[Diagnosis.HEART_FAILURE] += 0.5
    if pos("effusion"):
        scores[Diagnosis.HEART_FAILURE] += 0.2
    if pos("pneumonia"):
        scores[Diagnosis.PNEUMONIA] += 0.9
    if pos("consolidation"):
        scores[Diagnosis.PNEUMONIA] += 0.6
    if pos("opacity"):
        scores[Diagnosis.PNEUMONIA] += 0.3
    if pos("hyperinflation"):
        scores[Diagnosis.COPD] += 0.9

    if max(scores.values()) <= 0.0:
        # nothing pathological asserted -> normal (strongly so if a normal cue fired)
        scores[Diagnosis.NORMAL] = 1.0 if normal_cue else 0.5
        return Diagnosis.NORMAL, scores

    top = max(scores.values())
    winners = [d for d in _DX_PRIORITY if scores[d] == top]
    return winners[0], scores


def label_patient_reports(reports: list[str]) -> tuple[ReportLabel, list[ReportLabel]]:
    """Label every report for a patient and produce a patient-level summary.

    The patient-level label aggregates concepts across all studies (CheXpert
    combine) and takes the highest-acuity diagnosis observed — a conservative
    choice for a triage worklist. Returns (patient_summary, per_report_labels).
    """
    per: list[ReportLabel] = [label_report(r) for r in reports]
    if not per:
        return ReportLabel(), per

    agg_concepts: dict[str, int] = {}
    normal_cue = False
    for rl in per:
        normal_cue = normal_cue or rl.normal_cue
        for c, v in rl.concepts.items():
            agg_concepts[c] = _combine(agg_concepts.get(c), v)

    summary = ReportLabel(
        concepts=agg_concepts,
        findings={CONCEPT_TO_FINDING[c]: v for c, v in agg_concepts.items() if c in CONCEPT_TO_FINDING},
        normal_cue=normal_cue,
        n_sentences=sum(rl.n_sentences for rl in per),
    )
    summary.diagnosis, summary.diagnosis_scores = _map_diagnosis(agg_concepts, normal_cue)
    return summary, per
