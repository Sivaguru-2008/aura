"""Per-modality signatures — the pluggable scoring units the detector runs.

A signature answers one question: *how strongly does this fingerprint look like my
modality?* Adding a modality to AURA's routing vocabulary means adding a signature
here (and an engine, if you intend to analyse it); no detector or router code changes.

On the meaning of "confidence"
------------------------------
The number a signature returns is an **ordinal decision score in [0,1], not a
calibrated posterior probability.** Calling it a probability would require a labelled
multi-modality corpus and a fitted classifier, and AURA has neither yet — it has one
very large chest-radiograph corpus and a handful of other-modality DICOMs. Every
score carries a ``calibrated`` flag saying which of those two situations it came
from, and the router refuses to hide the difference:

* ``calibrated=True``  — the threshold was measured against real images of that class
  (see the per-signature docstrings for n and the measurement).
* ``calibrated=False`` — the threshold is derived from acquisition physics and
  reasoning rather than a measured corpus. Score is capped at
  :data:`UNCALIBRATED_CONFIDENCE_CAP` and the decision is flagged for review.

Evidence precedence
-------------------
DICOM header beats pixels, always. ``Modality`` and ``BodyPartExamined`` are written
by the acquisition device; a pixel heuristic that disagrees with them is wrong. This
matters most for MR, where — as measured — an axial brain MRI and an axial head CT
are pixel-wise indistinguishable, and only the header separates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.core.router.features import ImageFingerprint
from backend.core.shared.logging import get_logger
from backend.core.shared.types import MODALITY_LABELS, ImagingModality

log = get_logger("router.signatures")

#: Ceiling on any score whose thresholds have not been measured against a real
#: corpus of that modality. Kept below the router's high-confidence bar so an
#: uncalibrated route always lands in "supported, but have a human confirm it".
UNCALIBRATED_CONFIDENCE_CAP = 0.70

#: Score assigned when a signature positively rules its own modality out.
REJECT = 0.02

# --- header keyword vocabularies ------------------------------------------- #
_RADIOGRAPHIC_MODALITIES = {"CR", "DX", "RG"}
_CHEST_TERMS = ("CHEST", "THORAX", "THORACIC", "LUNG")
_HEAD_TERMS = ("HEAD", "BRAIN", "SKULL", "CRANI", "CEREBR", "NEURO", "PITUITARY")


def _mentions(blob: str, terms: tuple[str, ...]) -> str | None:
    """First term from ``terms`` present in ``blob``, else ``None``."""
    for t in terms:
        if t in blob:
            return t
    return None


@dataclass
class SignatureScore:
    """One signature's verdict on one fingerprint."""

    modality: ImagingModality
    confidence: float
    calibrated: bool
    source: str                 # "dicom_metadata" | "pixel_signature" | both
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Force human confirmation even when confidence and margin are both high. Set by
    #: a signature that is *sure of its measurement* but knows the measurement cannot
    #: settle the question — pixel-only head geometry, which separates head from chest
    #: with 100% sensitivity but cannot separate MR from CT at all.
    review_required: bool = False

    @property
    def label(self) -> str:
        return MODALITY_LABELS.get(self.modality, self.modality.value)


@runtime_checkable
class ModalitySignature(Protocol):
    """Contract for a modality scorer.

    Implementations must be stateless and cheap: the detector runs every registered
    signature on every upload.
    """

    modality: ImagingModality

    def score(self, fingerprint: ImageFingerprint) -> SignatureScore:
        """Score ``fingerprint`` against this signature's modality. Never raises."""
        ...


# --------------------------------------------------------------------------- #
# Chest radiograph
# --------------------------------------------------------------------------- #
class ChestRadiographSignature:
    """Detects chest radiographs — the modality AURA Thorax serves.

    The pixel channel delegates to :func:`services.vision.xray_gate.validate_cxr`,
    the production intake gate that already guards ``POST /v1/studies/upload``. It is
    reused rather than reimplemented for three reasons: it is the component that has
    been validated against the 261k-image MIMIC-CXR corpus, a second implementation
    would inevitably drift from it, and reusing it guarantees the router and the
    legacy upload path agree about what a chest film is.

    Calibration: the gate's thresholds were fitted on real MIMIC-CXR exports, and its
    accept rate on this deployment's corpus is re-measured in
    ``tests/test_backend_router.py`` (see ``docs`` in ``backend/README.md`` for the
    measured sensitivity and the false-accept behaviour on non-chest DICOMs).
    """

    modality = ImagingModality.CHEST_XRAY

    #: Score for a header that names a chest radiograph outright.
    _HEADER_CONFIDENCE = 0.97
    #: Score for a radiographic header with no body-region tag, backed by pixels.
    _HEADER_PLUS_PIXEL_CONFIDENCE = 0.93
    #: Score for a pixel-only accept (PNG/JPEG export, no header available).
    _PIXEL_CONFIDENCE = 0.88

    def score(self, fingerprint: ImageFingerprint) -> SignatureScore:
        dicom = fingerprint.dicom
        evidence: dict[str, Any] = {}

        # ---- header channel (decisive when present) ------------------------ #
        if dicom.present and dicom.modality:
            evidence["dicom_modality"] = dicom.modality
            blob = dicom.text_blob
            if dicom.modality not in _RADIOGRAPHIC_MODALITIES:
                return self._reject(
                    f"DICOM modality is {dicom.modality!r}, not a radiograph "
                    f"({'/'.join(sorted(_RADIOGRAPHIC_MODALITIES))})", evidence)

            chest_term = _mentions(blob, _CHEST_TERMS)
            head_term = _mentions(blob, _HEAD_TERMS)
            if chest_term:
                evidence["matched_term"] = chest_term
                return SignatureScore(
                    self.modality, self._HEADER_CONFIDENCE, True, "dicom_metadata",
                    f"DICOM header declares a {dicom.modality} radiograph of the "
                    f"chest (matched {chest_term!r})", evidence)
            if head_term or (dicom.body_part and not chest_term):
                # A radiograph of a *named* non-chest region: skull film, extremity,
                # abdomen. The thorax model must not see these.
                evidence["body_part"] = dicom.body_part or "(from description)"
                return self._reject(
                    f"radiograph is of {evidence['body_part']!r}, not the chest",
                    evidence)

            # Radiograph with no region tag at all — fall through to pixels, which
            # is exactly the case the CXR gate was built for.
            gate = self._run_gate(fingerprint, evidence)
            if gate is not None and gate.ok:
                return SignatureScore(
                    self.modality, self._HEADER_PLUS_PIXEL_CONFIDENCE, True,
                    "pixel_signature+dicom_metadata",
                    f"DICOM header declares a {dicom.modality} radiograph with no "
                    f"body-region tag; chest anatomy confirmed from pixels", evidence)
            return self._reject(
                "radiograph header present but chest anatomy was not found in the "
                f"image ({gate.reason if gate else 'pixels unreadable'})", evidence)

        # ---- pixel channel (PNG/JPEG export) ------------------------------- #
        if not fingerprint.readable:
            return self._reject("image could not be decoded", evidence)

        gate = self._run_gate(fingerprint, evidence)
        if gate is None:
            return self._reject("chest-radiograph gate unavailable", evidence)
        if not gate.ok:
            return self._reject(gate.reason, evidence)
        return SignatureScore(
            self.modality, self._PIXEL_CONFIDENCE, True, "pixel_signature",
            "chest anatomy confirmed by the validated chest-radiograph intake gate",
            evidence)

    def _run_gate(self, fingerprint: ImageFingerprint, evidence: dict[str, Any]):
        """Invoke the production CXR gate, recording its checks as evidence."""
        try:
            from services.vision.xray_gate import validate_cxr

            result = validate_cxr(fingerprint.path)
        except Exception:
            log.exception("chest-radiograph gate raised; treating as unavailable")
            return None
        evidence["gate_ok"] = bool(result.ok)
        evidence["gate_checks"] = result.checks
        if not result.ok:
            evidence["gate_reason"] = result.reason
        return result

    def _reject(self, reason: str, evidence: dict[str, Any]) -> SignatureScore:
        return SignatureScore(self.modality, REJECT, True,
                              "dicom_metadata" if evidence.get("dicom_modality")
                              else "pixel_signature", reason, evidence)


# --------------------------------------------------------------------------- #
# Cross-sectional head studies (shared pixel geometry)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _HeadGeometry:
    """Result of testing the cross-sectional-head pixel signature.

    An axial slice through a head — MR or CT — puts a compact subject in a field of
    signal-free air. The scoring lives in
    :func:`services.vision.xray_gate.head_geometry_score`, which is shared with the
    chest intake gate so the router and the gate cannot disagree about what a head
    slice looks like; that module's docstring carries the measurement table.

    Summary of that measurement — 1500 random MIMIC-CXR films vs 600 BraTS2020 axial
    slices, half of them re-encoded through the console's own non-aspect-preserving
    224-grid resize::

        chest head score  med 0.000  p99 0.518  max 0.801
        brain head score  min 0.922  p1  0.993  med 1.000
        threshold 0.85 -> chest false-reject 0.00%, brain detected 100.0%

    This replaced a six-way AND of ``all_edges_dark``, ``compact_subject``,
    ``air_background``, ``convex_subject``, ``centred_subject`` and ``grayscale``.
    Measured against the same corpora that conjunction detected only **81.7%** of real
    brain slices, because ``centred_subject`` and ``convex_subject`` do not separate
    the classes at all (centroid offset: chest median 0.061, brain 0.052) yet could
    each veto the three features that do. An additive score cannot be vetoed by a term
    that carries no information.

    This signature still cannot distinguish MR from CT. That is a property of the
    physics, not a shortcut: both put the same head cross-section in the same black
    air. The DICOM ``Modality`` tag is the only reliable separator, which is why the
    pixel-only path — however strong its geometry — is always flagged for review.
    """

    matches: bool
    reason: str
    evidence: dict[str, Any]
    score: float = 0.0


def _test_head_geometry(fingerprint: ImageFingerprint) -> _HeadGeometry:
    from services.vision.xray_gate import HEAD_COMMIT_SCORE, head_geometry_from_path

    px = fingerprint.pixels
    if not px.decodable:
        return _HeadGeometry(False, "image could not be decoded", {})
    if not px.is_grayscale:
        return _HeadGeometry(
            False, "image carries colour; cross-sectional MR and CT are grayscale",
            {"mean_saturation": round(px.mean_saturation, 3)})

    # Measured by the gate, not re-derived here: the router and the intake gate must
    # not be able to disagree about whether an image is a head slice.
    score, evidence = head_geometry_from_path(fingerprint.path)
    evidence = dict(evidence)
    # Recorded but unused by the score — see the class docstring for why.
    evidence["foreground_occupancy"] = round(px.foreground_occupancy, 3)
    evidence["centroid_offset"] = round(px.centroid_offset, 3)

    if score < HEAD_COMMIT_SCORE:
        return _HeadGeometry(
            False, f"cross-sectional head geometry scores {score:.2f}, below the "
                   f"{HEAD_COMMIT_SCORE:.2f} threshold (air around subject "
                   f"{px.background_fraction:.2f}, {px.dark_edge_count}/4 frame edges "
                   f"dark, subject spans {px.foreground_bbox_fraction:.2f} of frame)",
            evidence, score)
    route = evidence.get("route", "framing")
    detail = ("a compact mass surrounded by signal-free air"
              if route == "framing"
              else "an elliptical subject with air in every frame corner, cropped to "
                   "its own outline")
    return _HeadGeometry(
        True, f"subject is {detail} (head score {score:.2f}, via {route}) — the "
              f"geometry of an axial slice through a head",
        evidence, score)


class BrainMRISignature:
    """Detects brain MRI — the modality AURA NeuroMind will serve.

    Two channels with very different standing:

    **Header (calibrated).** ``Modality == "MR"`` plus a head/brain term in
    ``BodyPartExamined`` or any description field is definitional, not heuristic —
    the scanner wrote it. Verified against the real MR DICOMs available locally
    (``emri_small*.dcm``, tagged ``MR``/``HEAD``), including one whose pixel geometry
    would *fail* the head test because the acquisition is tightly cropped. That case
    is precisely why the header outranks the pixels.

    **Pixel-only (calibrated for head-vs-chest, undetermined for MR-vs-CT).** For a
    PNG/JPEG brain MRI export there is no header. The head-geometry discriminant is
    now fitted and validated against real corpora — 600 BraTS2020 axial slices against
    1500 MIMIC-CXR films, 100% detection at 0.00% chest false-reject — so *this is a
    head slice* is a measured claim and the score is marked ``calibrated=True``.

    What remains uncalibrated is *which* head study it is: geometry cannot separate MR
    from CT, and no head-CT image corpus is available here to fit a separator. So the
    score carries ``review_required=True`` unconditionally on this path and says so in
    its reason. The route is taken — a brain MRI reaching NeuroMind flagged for
    confirmation is strictly better than one silently reaching the chest model — but it
    is never presented as settled.

    Confidence sits at :data:`_GEOMETRY_CONFIDENCE`, deliberately **above** the chest
    signature's pixel-only accept (0.88). Previously this path was capped at
    :data:`UNCALIBRATED_CONFIDENCE_CAP` (0.70), which made a brain MRI PNG structurally
    unroutable: whenever the chest gate false-accepted a brain — which, before the
    gate's head veto, was every time — chest won by construction. The cap remains the
    right ceiling for a genuinely unmeasured signature; it is no longer the right
    ceiling for this one.
    """

    modality = ImagingModality.BRAIN_MRI

    _HEADER_CONFIDENCE = 0.97
    _HEADER_PLUS_PIXEL_CONFIDENCE = 0.90
    #: Pixel-only head geometry. Above the chest pixel accept (0.88) because the
    #: discriminant is measured against both corpora and the chest gate now vetoes
    #: this geometry itself; below the header paths, which are definitional.
    _GEOMETRY_CONFIDENCE = 0.91
    #: MR header with no region information and no head geometry. Deliberately below
    #: the router's commit threshold: "an MRI of something" is not a route.
    _HEADER_ONLY_CONFIDENCE = 0.45

    def score(self, fingerprint: ImageFingerprint) -> SignatureScore:
        dicom = fingerprint.dicom
        evidence: dict[str, Any] = {}

        if dicom.present and dicom.modality:
            evidence["dicom_modality"] = dicom.modality
            if dicom.modality != "MR":
                return SignatureScore(
                    self.modality, REJECT, True, "dicom_metadata",
                    f"DICOM modality is {dicom.modality!r}, not MR", evidence)

            blob = dicom.text_blob
            head_term = _mentions(blob, _HEAD_TERMS)
            if head_term:
                evidence["matched_term"] = head_term
                evidence["body_part"] = dicom.body_part or "(from description)"
                return SignatureScore(
                    self.modality, self._HEADER_CONFIDENCE, True, "dicom_metadata",
                    f"DICOM header declares an MR study of the head/brain "
                    f"(matched {head_term!r})", evidence)

            if dicom.body_part:
                # MR of a named non-head region — knee, abdomen, spine.
                evidence["body_part"] = dicom.body_part
                return SignatureScore(
                    self.modality, REJECT, True, "dicom_metadata",
                    f"MR study is of {dicom.body_part!r}, not the head", evidence)

            # MR with no region tag: let the geometry decide.
            geometry = _test_head_geometry(fingerprint)
            evidence["geometry"] = geometry.evidence
            if geometry.matches:
                return SignatureScore(
                    self.modality, self._HEADER_PLUS_PIXEL_CONFIDENCE, True,
                    "pixel_signature+dicom_metadata",
                    "DICOM header declares an MR study with no body-region tag; "
                    f"{geometry.reason}", evidence)
            return SignatureScore(
                self.modality, self._HEADER_ONLY_CONFIDENCE, True, "dicom_metadata",
                "DICOM header declares an MR study but names no body region, and the "
                f"image does not show head geometry ({geometry.reason})", evidence)

        # ---- pixel-only: uncalibrated, capped, and honest about why --------- #
        if not fingerprint.readable:
            return SignatureScore(self.modality, REJECT, True, "pixel_signature",
                                  "image could not be decoded", evidence)

        geometry = _test_head_geometry(fingerprint)
        evidence["geometry"] = geometry.evidence
        if not geometry.matches:
            return SignatureScore(self.modality, REJECT, True, "pixel_signature",
                                  geometry.reason, evidence)
        return SignatureScore(
            self.modality, self._GEOMETRY_CONFIDENCE, True, "pixel_signature",
            f"{geometry.reason}. No DICOM header is present, so this is routed to the "
            "brain-MR engine as the only head modality AURA serves — but pixel "
            "geometry cannot distinguish MR from CT, and a head CT would land here "
            "too. Confirm the modality before relying on the result",
            evidence, review_required=True)


class HeadCTSignature:
    """Recognises head CT so it is *named* rather than misrouted.

    No engine serves head CT yet. Without this signature an axial head CT export
    would match the brain-MRI geometry test and be routed to NeuroMind. With it, the
    header path identifies the study correctly and the router answers
    ``supported=false`` — which is the right answer, and becomes a working route the
    day a CT engine registers for this modality. Nothing else changes.
    """

    modality = ImagingModality.HEAD_CT

    def score(self, fingerprint: ImageFingerprint) -> SignatureScore:
        dicom = fingerprint.dicom
        evidence: dict[str, Any] = {}
        if not (dicom.present and dicom.modality == "CT"):
            # Pixel-only head CT is indistinguishable from brain MRI; that ambiguity
            # is reported by BrainMRISignature rather than guessed at twice.
            return SignatureScore(self.modality, REJECT, True, "dicom_metadata",
                                  "not a CT acquisition", evidence)
        evidence["dicom_modality"] = "CT"
        head_term = _mentions(dicom.text_blob, _HEAD_TERMS)
        if head_term:
            evidence["matched_term"] = head_term
            return SignatureScore(
                self.modality, 0.96, True, "dicom_metadata",
                f"DICOM header declares a CT study of the head (matched {head_term!r})",
                evidence)

        geometry = _test_head_geometry(fingerprint)
        evidence["geometry"] = geometry.evidence
        if geometry.matches:
            return SignatureScore(
                self.modality, 0.85, True, "pixel_signature+dicom_metadata",
                f"DICOM header declares a CT study; {geometry.reason}", evidence)
        return SignatureScore(self.modality, REJECT, True, "dicom_metadata",
                              "CT study, but not of the head", evidence)


# --------------------------------------------------------------------------- #
# Catch-all: name what the header says, even with no engine behind it
# --------------------------------------------------------------------------- #
class DicomModalitySignature:
    """Maps remaining DICOM acquisition modalities onto the routing vocabulary.

    Purpose is honesty, not capability: an ultrasound or a mammogram gets identified
    by name and reported as unsupported, instead of falling into ``UNKNOWN`` or —
    worse — matching some other modality's geometry. Each of these becomes a live
    route the moment an engine claims it.
    """

    #: Placeholder; this signature emits whichever modality it resolves.
    modality = ImagingModality.UNKNOWN_MEDICAL

    _MAP: dict[str, ImagingModality] = {
        "US": ImagingModality.ULTRASOUND,
        "MG": ImagingModality.MAMMOGRAPHY,
        "OP": ImagingModality.RETINAL_FUNDUS,     # ophthalmic photography
    }

    def score(self, fingerprint: ImageFingerprint) -> SignatureScore:
        dicom = fingerprint.dicom
        if not (dicom.present and dicom.modality):
            return SignatureScore(ImagingModality.UNKNOWN_MEDICAL, REJECT, True,
                                  "dicom_metadata", "no DICOM acquisition modality", {})
        evidence = {"dicom_modality": dicom.modality}

        mapped = self._MAP.get(dicom.modality)
        if mapped is not None:
            return SignatureScore(
                mapped, 0.95, True, "dicom_metadata",
                f"DICOM header declares a {dicom.modality} study "
                f"({MODALITY_LABELS[mapped]})", evidence)

        if dicom.modality == "CT" and _mentions(dicom.text_blob, _CHEST_TERMS):
            return SignatureScore(
                ImagingModality.CHEST_CT, 0.95, True, "dicom_metadata",
                "DICOM header declares a CT study of the chest", evidence)

        # A recognised DICOM acquisition AURA has no vocabulary entry for. Naming it
        # "an unidentified medical image" is still better than "unknown".
        return SignatureScore(
            ImagingModality.UNKNOWN_MEDICAL, 0.60, True, "dicom_metadata",
            f"valid DICOM study of modality {dicom.modality!r}, which AURA does not "
            "yet classify to a body region", evidence)


def default_signatures() -> list[ModalitySignature]:
    """The signature set the platform ships with, in no significant order.

    Precedence between signatures is the detector's job, not this list's.
    """
    return [
        ChestRadiographSignature(),
        BrainMRISignature(),
        HeadCTSignature(),
        DicomModalitySignature(),
    ]
