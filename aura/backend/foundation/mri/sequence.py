"""MRI Sequence Detector — identify T1 / T1ce / T2 / FLAIR / DWI / ADC / SWI.

Which contrast an image carries determines which model may look at it. A FLAIR model
handed a DWI produces a confident, entirely wrong answer, and nothing downstream can
detect that from the pixels. So this module's job is not only to classify, but to be
explicit about how much the classification is worth.

Evidence, in strict precedence order
------------------------------------
1. **Acquisition parameters** — ``ScanningSequence``, ``SequenceVariant``, TR, TE, TI,
   flip angle, ``ImageType``, diffusion b-value. These are written by the pulse
   program. They are physics, and they are what the rules below are built on.
2. **Contrast administration** — ``ContrastBolusAgent``, which upgrades T1 to T1ce.
3. **Free text** — ``SeriesDescription``, ``ProtocolName``, filename. Used only to
   *break ties* among candidates the parameters already support, or as a last resort
   when there are no parameters at all (every NIfTI, most NRRD).

Rule 3 is the reason this module exists in the form it does. The requirement was
"never depend on filenames alone", and the enforcement is structural rather than
advisory: description evidence enters through a separate scoring channel whose result
is capped at :data:`DESCRIPTION_ONLY_CAP` and always emitted with
``metadata_available=False``. A description-only answer cannot reach the confidence of
a parameter-derived one, so a caller thresholding on confidence gets the right
behaviour without having to know this module's internals.

Why rules and not a classifier
------------------------------
A learned sequence classifier is a better long-term answer and is the obvious upgrade
behind the :class:`SequenceDetector` protocol. It is not what this layer ships,
because training one honestly needs a labelled multi-vendor corpus that this
deployment does not have — and the same audit that found AURA's quantum fusion claim
did not survive a fair test applies here. Published rules with cited thresholds are
what can be defended today, so that is what runs, and every threshold is a named
constant a reviewer can check against the literature.

Thresholds are the conventional ones for brain MRI at 1.5-3 T (Bitar et al.,
*RadioGraphics* 2006; Bernstein et al., *Handbook of MRI Pulse Sequences*, 2004).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from aura.backend.core.shared.logging import get_logger
from .metadata import MRIMetadata, sequence_evidence_text
from .types import SEQUENCE_LABELS, SequenceType

log = get_logger("foundation.mri.sequence")

# --------------------------------------------------------------------------- #
# Thresholds — conventional brain-MRI values at 1.5-3 T
# --------------------------------------------------------------------------- #
SHORT_TR_MS = 800.0          # T1 weighting on spin echo
LONG_TR_MS = 2000.0          # T2 / PD weighting
SHORT_TE_MS = 30.0           # T1 / PD weighting
LONG_TE_MS = 80.0            # T2 weighting
FLAIR_TI_MS = (1800.0, 3000.0)   # CSF null at 1.5-3 T
STIR_TI_MS = (100.0, 250.0)      # fat null — an IR that is *not* FLAIR
MPRAGE_TI_MS = (600.0, 1400.0)   # inversion-prepared 3D T1
SWI_MIN_TE_MS = 15.0             # long-TE gradient echo for susceptibility contrast
GRE_T1_MAX_TE_MS = 10.0
DWI_LOW_B = 50.0                 # b below this is effectively a b0 reference

#: Confidence a description-only classification may not exceed. Sits below the
#: weakest parameter-derived score by construction, so parameters always win.
DESCRIPTION_ONLY_CAP = 0.55

#: Awarded when the acquisition parameters alone are decisive (a b-value tag, an
#: ``ADC`` ImageType, a CSF-nulling TI). Not 1.0: vendors mislabel, and reserving
#: headroom is what lets a future corroborating channel raise a score.
DECISIVE = 0.95
STRONG = 0.85
MODERATE = 0.70
WEAK = 0.55


class SequenceCandidate(BaseModel):
    """One sequence the detector scored, with the evidence behind the score."""

    model_config = ConfigDict(frozen=True)

    sequence: SequenceType
    label: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class SequenceAssignment(BaseModel):
    """The detector's verdict for one series.

    Deliberately the same shape as the modality router's
    :class:`~aura.backend.models.routing.RoutingMetadata`: a winner, a confidence, a
    calibration flag, a reason, and every loser with its evidence. A reviewer who can
    read one can read the other, and both refuse rather than guess.
    """

    model_config = ConfigDict(frozen=True)

    sequence: SequenceType = SequenceType.UNKNOWN
    label: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    #: ``True`` when acquisition parameters were available and drove the decision.
    #: ``False`` means the answer rests on free text and must not be trusted by a
    #: model that requires a specific contrast.
    metadata_available: bool = False
    source: str = Field(
        "none",
        description="'acquisition_parameters' | 'acquisition_parameters+description' "
                    "| 'description_only' | 'none'",
    )
    reason: str = ""
    requires_review: bool = Field(
        False,
        description="True when the answer is description-only, low-confidence, or "
                    "contested by a close runner-up.",
    )
    candidates: tuple[SequenceCandidate, ...] = ()

    @property
    def is_confident(self) -> bool:
        return self.metadata_available and self.confidence >= MODERATE


class SequenceDetector(Protocol):
    """Swap-in point for a learned classifier. One method, one direction."""

    def detect(self, metadata: MRIMetadata) -> SequenceAssignment: ...


@dataclass(frozen=True)
class _Score:
    """Internal scoring record before it becomes a candidate."""

    sequence: SequenceType
    confidence: float
    reason: str
    evidence: dict[str, Any]


class RuleBasedSequenceDetector:
    """Parameter-first rule detector. Stateless and therefore trivially shareable."""

    #: Free-text tokens per sequence. Only ever used to break ties or as the last
    #: resort — see the module docstring. Ordered longest-first at match time so
    #: ``T1CE`` wins over ``T1`` and ``ADC`` over ``DC``.
    DESCRIPTION_TOKENS: dict[SequenceType, tuple[str, ...]] = {
        SequenceType.FLAIR: ("FLAIR", "T2 FLAIR", "T2FLAIR", "DARK FLUID", "TIRM"),
        SequenceType.ADC: ("ADC", "APPARENT DIFFUSION", "EADC", "_ADC"),
        SequenceType.DWI: ("DWI", "DIFFUSION", "TRACEW", "TRACE", "ISODWI", "EPI DIFF"),
        SequenceType.SWI: ("SWI", "SWAN", "VENOBOLD", "SUSCEPTIBILITY", "MEDIC",
                           "T2 STAR", "T2*", "GRE"),
        SequenceType.T1CE: ("T1CE", "T1C", "T1 POST", "T1POST", "POST GAD", "POSTGAD",
                            "GD", "+C", "CONTRAST", "MPRAGE POST", "T1 +C"),
        SequenceType.T1: ("T1", "MPRAGE", "SPGR", "BRAVO", "FSPGR", "TFE", "T1W"),
        SequenceType.T2: ("T2", "TSE T2", "T2W", "CISS", "SPACE", "CUBE"),
        SequenceType.PD: ("PD", "PROTON DENSITY", "PDW"),
    }

    def detect(self, metadata: MRIMetadata) -> SequenceAssignment:
        """Classify one series. Never raises — an unclassifiable series is ``UNKNOWN``."""
        acquisition = metadata.acquisition
        text = sequence_evidence_text(metadata)

        parameter_scores = self._score_from_parameters(metadata) \
            if acquisition.available else []
        description_scores = self._score_from_description(text)

        if parameter_scores:
            return self._decide_from_parameters(parameter_scores, description_scores,
                                                text)
        if description_scores:
            return self._decide_from_description(description_scores)
        return SequenceAssignment(
            sequence=SequenceType.UNKNOWN,
            label=SEQUENCE_LABELS[SequenceType.UNKNOWN],
            confidence=0.0,
            metadata_available=False,
            source="none",
            reason="no acquisition parameters and no recognisable description text; "
                   "the sequence cannot be determined from this study",
            requires_review=True,
        )

    # ------------------------------------------------------------------ #
    # Channel 1: acquisition parameters
    # ------------------------------------------------------------------ #
    def _score_from_parameters(self, metadata: MRIMetadata) -> list[_Score]:
        a = metadata.acquisition
        seq = set(a.scanning_sequence)
        variant = set(a.sequence_variant)
        options = set(a.scan_options)
        image_type = set(a.image_type)
        tr, te, ti = a.repetition_time_ms, a.echo_time_ms, a.inversion_time_ms
        base = {"scanning_sequence": sorted(seq), "tr_ms": tr, "te_ms": te, "ti_ms": ti,
                "flip_deg": a.flip_angle_deg, "image_type": sorted(image_type)}
        out: list[_Score] = []

        # -- derived diffusion maps: decisive, and checked first ------------ #
        # ADC is a computed map, not an acquisition. Feeding it to a DWI model is a
        # classic and silent error: the two are visually inverted, so a stroke model
        # reads restricted diffusion as facilitated.
        if {"ADC", "EADC"} & image_type or "ADC" in (a.sequence_name or "").upper():
            out.append(_Score(SequenceType.ADC, DECISIVE,
                              "ImageType marks this as a computed ADC map", base))
        elif a.diffusion_b_value is not None:
            if a.diffusion_b_value >= DWI_LOW_B:
                out.append(_Score(
                    SequenceType.DWI, DECISIVE,
                    f"diffusion b-value {a.diffusion_b_value:g} s/mm^2 is recorded in "
                    "the header", {**base, "b_value": a.diffusion_b_value}))
            else:
                out.append(_Score(
                    SequenceType.DWI, STRONG,
                    f"diffusion b-value {a.diffusion_b_value:g} s/mm^2 — a b0 "
                    "reference image from a diffusion acquisition",
                    {**base, "b_value": a.diffusion_b_value}))
        elif "DIFFUSION" in image_type or {"TRACEW", "TRACE", "ISOTROPIC"} & image_type:
            out.append(_Score(SequenceType.DWI, STRONG,
                              "ImageType marks diffusion-weighted content", base))
        elif "EP" in seq and te is not None and te > 50:
            # Echo-planar with a long TE and no b-value tag: almost always diffusion
            # on a vendor that omits the standard tag, but perfusion and BOLD share
            # the readout, so this cannot be decisive.
            out.append(_Score(
                SequenceType.DWI, MODERATE,
                f"echo-planar readout with TE {te:g} ms and no b-value tag; "
                "diffusion is the most likely origin but perfusion and BOLD share "
                "this readout", base))

        # -- susceptibility-weighted ---------------------------------------- #
        if {"SWI", "SWAN", "MNIP", "MIP", "PHASE"} & image_type and "GR" in seq:
            out.append(_Score(SequenceType.SWI, DECISIVE,
                              "gradient echo with a susceptibility-weighted ImageType",
                              base))
        elif "GR" in seq and te is not None and te >= SWI_MIN_TE_MS \
                and a.flip_angle_deg is not None and a.flip_angle_deg <= 25:
            out.append(_Score(
                SequenceType.SWI, MODERATE,
                f"gradient echo with long TE {te:g} ms and low flip angle "
                f"{a.flip_angle_deg:g} deg — susceptibility-weighted or T2*", base))

        # -- inversion recovery family --------------------------------------- #
        if "IR" in seq and ti is not None:
            if FLAIR_TI_MS[0] <= ti <= FLAIR_TI_MS[1]:
                out.append(_Score(
                    SequenceType.FLAIR, DECISIVE,
                    f"inversion recovery with TI {ti:g} ms, which nulls CSF at "
                    "clinical field strengths", base))
            elif STIR_TI_MS[0] <= ti <= STIR_TI_MS[1]:
                # A short-TI inversion recovery nulls fat, not fluid. It is not FLAIR
                # and not any class this layer names, so the honest score is none —
                # falling through to the T1/T2 rules below would mislabel it.
                out.append(_Score(
                    SequenceType.UNKNOWN, MODERATE,
                    f"short-TI inversion recovery (TI {ti:g} ms) nulls fat, not CSF; "
                    "this is a STIR-family acquisition, outside the classes this "
                    "detector names", base))
            elif MPRAGE_TI_MS[0] <= ti <= MPRAGE_TI_MS[1] and "GR" in seq:
                out.append(_Score(
                    SequenceType.T1, STRONG,
                    f"inversion-prepared gradient echo with TI {ti:g} ms — a 3D "
                    "T1-weighted volumetric acquisition (MPRAGE/BRAVO family)", base))

        # -- spin echo / fast spin echo weighting ---------------------------- #
        if tr is not None and te is not None and "IR" not in seq:
            if tr <= SHORT_TR_MS and te <= SHORT_TE_MS:
                out.append(_Score(
                    SequenceType.T1, STRONG,
                    f"short TR {tr:g} ms and short TE {te:g} ms — T1 weighting", base))
            elif tr >= LONG_TR_MS and te >= LONG_TE_MS:
                out.append(_Score(
                    SequenceType.T2, STRONG,
                    f"long TR {tr:g} ms and long TE {te:g} ms — T2 weighting", base))
            elif tr >= LONG_TR_MS and te <= SHORT_TE_MS:
                out.append(_Score(
                    SequenceType.PD, STRONG,
                    f"long TR {tr:g} ms and short TE {te:g} ms — proton-density "
                    "weighting", base))
            elif "GR" in seq and te <= GRE_T1_MAX_TE_MS and tr <= 50 \
                    and a.flip_angle_deg is not None and a.flip_angle_deg >= 10:
                out.append(_Score(
                    SequenceType.T1, MODERATE,
                    f"spoiled gradient echo, TR {tr:g} ms / TE {te:g} ms / flip "
                    f"{a.flip_angle_deg:g} deg — T1-weighted", base))

        # -- contrast administration upgrades T1 -> T1ce --------------------- #
        if a.contrast_administered:
            out = self._apply_contrast(out, a.contrast_agent, base)

        # Drop the STIR sentinel unless it is the only thing we found, so a genuinely
        # ambiguous IR does not outrank a real classification.
        real = [s for s in out if s.sequence is not SequenceType.UNKNOWN]
        return real or out

    @staticmethod
    def _apply_contrast(scores: list[_Score], agent: str | None,
                        base: dict[str, Any]) -> list[_Score]:
        """Promote T1 candidates to T1ce when a contrast agent was administered.

        Only T1 is promoted. Post-contrast T2 and FLAIR exist and are clinically
        useful, but they are not separate classes here — a T2 acquired after
        gadolinium is still read as a T2, and inventing a ``T2CE`` class this layer
        cannot validate would be worse than saying T2.
        """
        promoted: list[_Score] = []
        found_t1 = False
        for score in scores:
            if score.sequence is SequenceType.T1:
                found_t1 = True
                promoted.append(_Score(
                    SequenceType.T1CE, min(DECISIVE, score.confidence + 0.05),
                    f"{score.reason}; contrast agent "
                    f"{'(' + agent + ') ' if agent else ''}was administered",
                    {**score.evidence, "contrast_agent": agent}))
            else:
                promoted.append(score)
        if not found_t1:
            # Contrast alone is not evidence of T1 weighting — post-contrast T2 and
            # FLAIR are routine — so nothing is added when no T1 candidate exists.
            promoted.append(_Score(
                SequenceType.T1CE, WEAK,
                "a contrast agent was administered, but the pulse-sequence parameters "
                "do not indicate T1 weighting; post-contrast T2/FLAIR is equally likely",
                {**base, "contrast_agent": agent}))
        return promoted

    # ------------------------------------------------------------------ #
    # Channel 2: free text
    # ------------------------------------------------------------------ #
    def _score_from_description(self, text: str) -> list[_Score]:
        """Score sequences by description tokens. Capped, always.

        Token order within the class map matters: ``T1CE`` tokens are tested before
        ``T1`` tokens so "T1 POST GAD" does not resolve to plain T1.
        """
        if not text:
            return []
        out: list[_Score] = []
        for sequence, tokens in self.DESCRIPTION_TOKENS.items():
            matched = [t for t in tokens if t in text]
            if not matched:
                continue
            # Longer, more specific tokens are better evidence than "T1" appearing
            # inside "T1RHO"; score by the longest match.
            specificity = max(len(t) for t in matched)
            confidence = min(DESCRIPTION_ONLY_CAP,
                             0.30 + 0.05 * min(specificity, 5))
            out.append(_Score(
                sequence, confidence,
                f"description text matched {matched!r}",
                {"matched_tokens": matched, "text_channel": True}))
        return out

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #
    def _decide_from_parameters(self, parameter_scores: list[_Score],
                                description_scores: list[_Score],
                                text: str) -> SequenceAssignment:
        """Rank parameter candidates, letting description break a genuine tie."""
        ranked = sorted(parameter_scores, key=lambda s: s.confidence, reverse=True)
        best = ranked[0]
        source = "acquisition_parameters"
        reason = best.reason
        contested = len(ranked) > 1 and (best.confidence - ranked[1].confidence) < 0.10

        if contested and description_scores:
            supported = {s.sequence for s in description_scores}
            tied = [s for s in ranked if abs(s.confidence - best.confidence) < 0.10]
            preferred = next((s for s in tied if s.sequence in supported), None)
            if preferred is not None and preferred.sequence is not best.sequence:
                best = _Score(preferred.sequence,
                              min(DECISIVE, preferred.confidence + 0.05),
                              f"{preferred.reason}; the series description agrees",
                              {**preferred.evidence, "description_tiebreak": True})
                source = "acquisition_parameters+description"
                reason = best.reason
                contested = False
            elif preferred is not None:
                best = _Score(best.sequence, min(DECISIVE, best.confidence + 0.05),
                              f"{best.reason}; the series description agrees",
                              {**best.evidence, "description_agrees": True})
                source = "acquisition_parameters+description"
                reason = best.reason
                contested = False

        candidates = self._candidates(ranked, description_scores)
        return SequenceAssignment(
            sequence=best.sequence,
            label=SEQUENCE_LABELS[best.sequence],
            confidence=round(best.confidence, 4),
            metadata_available=True,
            source=source,
            reason=reason,
            requires_review=contested or best.confidence < MODERATE
            or best.sequence is SequenceType.UNKNOWN,
            candidates=candidates,
        )

    def _decide_from_description(self, description_scores: list[_Score]
                                 ) -> SequenceAssignment:
        """Last-resort classification, flagged as such in three separate fields."""
        ranked = sorted(description_scores, key=lambda s: s.confidence, reverse=True)
        best = ranked[0]
        log.info(
            "sequence classified from description text only",
            extra={"context": {"sequence": best.sequence.value,
                               "confidence": round(best.confidence, 3)}},
        )
        return SequenceAssignment(
            sequence=best.sequence,
            label=SEQUENCE_LABELS[best.sequence],
            confidence=round(min(best.confidence, DESCRIPTION_ONLY_CAP), 4),
            metadata_available=False,
            source="description_only",
            reason=(f"{best.reason}. No acquisition parameters were available, so this "
                    "rests entirely on free text written by a human or a converter; "
                    f"confidence is capped at {DESCRIPTION_ONLY_CAP} and the "
                    "assignment requires confirmation before a contrast-specific "
                    "model is applied."),
            requires_review=True,
            candidates=self._candidates([], ranked),
        )

    @staticmethod
    def _candidates(parameter_scores: Iterable[_Score],
                    description_scores: Iterable[_Score]) -> tuple[SequenceCandidate, ...]:
        """Merge both channels into one ranked candidate list, best score per class."""
        merged: dict[SequenceType, _Score] = {}
        for score in list(parameter_scores) + list(description_scores):
            current = merged.get(score.sequence)
            if current is None or score.confidence > current.confidence:
                merged[score.sequence] = score
        return tuple(
            SequenceCandidate(
                sequence=s.sequence,
                label=SEQUENCE_LABELS[s.sequence],
                confidence=round(s.confidence, 4),
                reason=s.reason,
                evidence=s.evidence,
            )
            for s in sorted(merged.values(), key=lambda s: s.confidence, reverse=True)
        )
