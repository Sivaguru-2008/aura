"""Tests for the Intelligent Modality Router.

Three layers, in increasing cost:

1. **Contract + registry** — pure unit tests, no images, no models.
2. **Routing on real images** — real MIMIC-CXR films and the real MR/CR/CT/US DICOMs
   bundled with pydicom. Synthetic phantoms would prove nothing about a detector
   whose whole job is telling real acquisitions apart, so tests that need real data
   skip when it is absent rather than fabricating it.
3. **API + dispatch** — the endpoints mounted on a bare app with a stub registry, so
   no model weights are loaded.

The batch test at the end is the one that matters most: it re-measures the
chest-radiograph accept rate and the brain-MRI misroute rate on real films, so a
threshold change that quietly degrades routing fails here instead of in production.
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest

from aura.backend.core.router.detector import SignatureModalityDetector
from aura.backend.core.router.router import ModalityRouter
from aura.backend.core.shared.types import EngineStatus, ImagingModality
from aura.backend.core.upload.intake import stage_bytes
from aura.backend.engines.base.contract import (
    AnalysisEngine,
    AnalysisResult,
    EngineDescriptor,
    EngineReport,
    PreparedStudy,
    ValidationOutcome,
)
from aura.backend.engines.base.registry import EngineRegistry
from aura.backend.models.routing import ResultStatus
from aura.backend.services.dispatch import DispatchService
from aura.schemas.clinical import Diagnosis
from aura.services.vision.xray_gate import validate_cxr

MIMIC_ROOT = Path(
    r"E:\AURA\datasets\simhadrisadaram\mimic-cxr-dataset\versions\2"
    r"\official_data_iccv_final\files"
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _mimic_films(n: int, seed: int = 3) -> list[Path]:
    """Sample real chest films. Empty list when the corpus is not on this machine."""
    if not MIMIC_ROOT.exists():
        return []
    rng = random.Random(seed)
    films: list[Path] = []
    for group in sorted(MIMIC_ROOT.iterdir())[:6]:
        if not group.is_dir():
            continue
        patients = [d for d in group.iterdir() if d.is_dir()]
        rng.shuffle(patients)
        for patient in patients[:40]:
            for study in patient.iterdir():
                if study.is_dir():
                    films.extend(study.glob("*.jpg"))
        if len(films) > n * 8:
            break
    rng.shuffle(films)
    return films[:n]


def _testdata(name: str) -> Path | None:
    """Path to a pydicom bundled test file, or None if pydicom is unavailable."""
    try:
        from pydicom.data import get_testdata_file
    except ImportError:
        return None
    try:
        path = get_testdata_file(name)
    except Exception:
        return None
    return Path(path) if path else None


def _route(path: Path, router: ModalityRouter):
    with stage_bytes(path.read_bytes(), path.name) as asset:
        return router.route(asset)


@pytest.fixture(scope="module")
def registry() -> EngineRegistry:
    """Registry with the real NeuroMind placeholder and a stub Thorax.

    The stub stands in for the real Thorax engine so routing can be tested without
    loading a DenseNet — the router only ever asks the registry which engine *claims*
    a modality, never constructs it.
    """
    reg = EngineRegistry()

    class _StubThorax(AnalysisEngine):
        descriptor = EngineDescriptor(
            engine_id="thorax", display_name="AURA Thorax (stub)", version="test",
            modalities=(ImagingModality.CHEST_XRAY,), status=EngineStatus.AVAILABLE,
        )

        def validate_input(self, asset):
            return ValidationOutcome(True)

        def preprocess(self, asset):
            return PreparedStudy("STU-TEST", ImagingModality.CHEST_XRAY, None)

        async def analyze(self, prepared):
            return AnalysisResult("STU-TEST", payload=None, case_id="CASE-TEST")

        def generate_report(self, result):
            return EngineReport(summary="stub report")

    from aura.backend.engines.neuro.engine import NeuroMindEngine

    reg.register(_StubThorax.descriptor, _StubThorax)
    reg.register(NeuroMindEngine.descriptor, NeuroMindEngine)
    return reg


@pytest.fixture(scope="module")
def router(registry: EngineRegistry) -> ModalityRouter:
    return ModalityRouter(SignatureModalityDetector(), registry)


# --------------------------------------------------------------------------- #
# 1. Contract + registry
# --------------------------------------------------------------------------- #
def test_engine_contract_requires_all_four_stages():
    """An engine missing any contract method cannot be instantiated."""

    class Incomplete(AnalysisEngine):
        descriptor = EngineDescriptor("bad", "Bad", "0", (ImagingModality.UNKNOWN,))

        def validate_input(self, asset):
            return ValidationOutcome(True)

    with pytest.raises(TypeError):
        Incomplete()


def test_registry_resolves_lazily_and_caches(registry: EngineRegistry):
    built: list[int] = []

    class _Counting(AnalysisEngine):
        descriptor = EngineDescriptor("counting", "Counting", "1",
                                      (ImagingModality.MAMMOGRAPHY,))

        def __init__(self):
            built.append(1)
            super().__init__()

        def validate_input(self, asset):
            return ValidationOutcome(True)

        def preprocess(self, asset):
            return PreparedStudy("s", ImagingModality.MAMMOGRAPHY, None)

        async def analyze(self, prepared):
            return AnalysisResult("s", None)

        def generate_report(self, result):
            return EngineReport()

    reg = EngineRegistry()
    reg.register(_Counting.descriptor, _Counting)
    assert built == []                                     # registration builds nothing
    assert reg.engine_id_for_modality(ImagingModality.MAMMOGRAPHY) == "counting"
    assert built == []                                     # lookup builds nothing either
    assert reg.resolve("counting") is reg.resolve("counting")
    assert built == [1]                                    # constructed exactly once


def test_registry_contains_a_failing_engine():
    """A broken engine is marked unavailable; other routes keep working."""
    from aura.backend.core.shared.errors import EngineNotAvailable

    def _explode():
        raise RuntimeError("missing model weights")

    reg = EngineRegistry()
    reg.register(
        EngineDescriptor("broken", "Broken", "1", (ImagingModality.ULTRASOUND,)),
        _explode,
    )
    with pytest.raises(EngineNotAvailable):
        reg.resolve("broken")
    assert reg.descriptors()[0].status is EngineStatus.UNAVAILABLE


# --------------------------------------------------------------------------- #
# 2. Routing on real images
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MIMIC_ROOT.exists(), reason="MIMIC-CXR corpus not present")
def test_real_chest_film_routes_to_thorax(router: ModalityRouter):
    films = _mimic_films(1)
    assert films, "expected at least one film from the MIMIC corpus"
    decision = _route(films[0], router)
    assert decision.modality == ImagingModality.CHEST_XRAY.value
    assert decision.selected_engine == "thorax"
    assert decision.supported is True
    assert decision.confidence >= 0.85
    assert decision.calibrated is True


def test_real_head_mr_dicom_routes_to_neuromind(router: ModalityRouter):
    """A real MR/HEAD DICOM routes to NeuroMind on the metadata channel.

    This file's pixel geometry does *not* satisfy the head test (the acquisition is
    tightly cropped, so the subject reaches every frame edge) — which is exactly the
    case that proves header evidence must outrank pixel geometry.
    """
    path = _testdata("emri_small.dcm")
    if path is None:
        pytest.skip("pydicom test data unavailable")
    decision = _route(path, router)
    assert decision.modality == ImagingModality.BRAIN_MRI.value
    assert decision.selected_engine == "neuromind"
    assert decision.supported is True
    assert decision.calibrated is True
    assert "dicom_metadata" in decision.candidates[0].source


def test_non_chest_radiograph_is_not_routed_to_thorax(router: ModalityRouter):
    """A CR of an extremity must not reach the chest model."""
    path = _testdata("RG3_UNCR.dcm")            # Modality=CR, BodyPartExamined=EXTREMITY
    if path is None:
        pytest.skip("pydicom test data unavailable")
    decision = _route(path, router)
    assert decision.selected_engine != "thorax"
    assert decision.modality != ImagingModality.CHEST_XRAY.value


def test_chest_radiograph_dicom_routes_to_thorax(router: ModalityRouter):
    path = _testdata("RG1_UNCR.dcm")            # Modality=CR, BodyPartExamined=CHEST
    if path is None:
        pytest.skip("pydicom test data unavailable")
    decision = _route(path, router)
    assert decision.modality == ImagingModality.CHEST_XRAY.value
    assert decision.selected_engine == "thorax"
    assert decision.confidence >= 0.95           # header evidence


def test_head_ct_is_named_and_reported_unsupported(router: ModalityRouter):
    """Head CT must be identified, not silently absorbed by the brain-MRI route."""
    path = _testdata("J2K_pixelrep_mismatch.dcm")    # Modality=CT, BodyPart=HEAD
    if path is None:
        pytest.skip("pydicom test data unavailable")
    decision = _route(path, router)
    assert decision.modality == ImagingModality.HEAD_CT.value
    assert decision.supported is False
    assert decision.selected_engine is None
    assert "no analysis engine is registered" in decision.reason


def test_ultrasound_is_named_and_reported_unsupported(router: ModalityRouter):
    path = _testdata("US1_UNCR.dcm")
    if path is None:
        pytest.skip("pydicom test data unavailable")
    decision = _route(path, router)
    assert decision.modality == ImagingModality.ULTRASOUND.value
    assert decision.supported is False


def test_undecodable_upload_is_declined_not_guessed(router: ModalityRouter):
    with stage_bytes(b"this is not an image at all", "notes.png") as asset:
        decision = router.route(asset)
    assert decision.modality == ImagingModality.UNKNOWN.value
    assert decision.supported is False
    assert decision.selected_engine is None
    assert decision.requires_review is True


def test_routing_metadata_has_the_required_contract_keys(router: ModalityRouter):
    """Requirement 6: the five contract fields are always present."""
    with stage_bytes(b"junk", "x.png") as asset:
        payload = router.route(asset).model_dump()
    for key in ("modality", "selected_engine", "confidence", "supported", "reason"):
        assert key in payload, f"missing required routing key {key!r}"


# --------------------------------------------------------------------------- #
# 3. Dispatch + API
# --------------------------------------------------------------------------- #
def test_neuromind_never_claims_a_tumour_subtype(registry):
    """An MR study routes to NeuroMind and comes back without an unearned subtype.

    NeuroMind now runs a real trained network, so this no longer asserts that nothing
    happens. It asserts the constraint that outlives the placeholder: the model is a
    BraTS segmentation network trained on a glioma-only corpus, so whatever it reports,
    it may never name a tumour subtype. A previous build derived one from
    ``int(sha256[:12], 16) % 5``.
    """
    path = _testdata("emri_small.dcm")
    if path is None:
        pytest.skip("pydicom test data unavailable")

    dispatch = DispatchService(ModalityRouter(SignatureModalityDetector(), registry),
                               registry)
    with stage_bytes(path.read_bytes(), path.name) as asset:
        # asyncio.run rather than a pytest-asyncio marker — the repo's test suite has
        # no async plugin and the routing layer should not add a test dependency.
        envelope = asyncio.run(dispatch.dispatch(asset))

    assert envelope.routing.selected_engine == "neuromind"
    assert envelope.result is not None
    assert envelope.result.status is not ResultStatus.UNSUPPORTED

    payload = envelope.result.payload or {}
    if envelope.result.status is not ResultStatus.COMPLETED or not payload:
        return                      # refused or unavailable: nothing was claimed at all

    forbidden = {Diagnosis.GLIOMA.value, Diagnosis.MENINGIOMA.value,
                 Diagnosis.STROKE.value, Diagnosis.HEMORRHAGE_DX.value}
    top = payload.get("top_diagnosis")
    assert top not in forbidden, (
        f"NeuroMind reported {top!r}; the trained model has no subtype head and its "
        f"corpus contains only glioma")
    # The probability that reaches a reader must be the calibrated one. The raw head
    # measures ECE 0.095 and is overconfident through the middle of its range.
    assert "presence_probability_calibrated" in payload
    assert payload["top_probability"] is not None


def _client(registry: EngineRegistry):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aura.backend.api.routes import build_router

    app = FastAPI()
    app.include_router(
        build_router(DispatchService(ModalityRouter(SignatureModalityDetector(),
                                                    registry), registry))
    )
    return TestClient(app)


def test_engines_endpoint_reports_placeholder_status(registry):
    body = _client(registry).get("/v1/engines").json()
    by_id = {e["engine_id"]: e for e in body["engines"]}
    assert by_id["neuromind"]["status"] == "available"
    assert by_id["thorax"]["status"] == "available"
    assert "brain_mri" in body["supported_modalities"]
    assert "chest_xray" in body["supported_modalities"]


def test_route_endpoint_returns_candidates(registry):
    path = _testdata("emri_small.dcm")
    if path is None:
        pytest.skip("pydicom test data unavailable")
    response = _client(registry).post(
        "/v1/studies/route",
        files={"file": (path.name, path.read_bytes(), "application/dicom")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "brain_mri"
    assert body["selected_engine"] == "neuromind"
    assert len(body["candidates"]) >= 2            # losers are reported too


def test_analyze_endpoint_routes_an_mr_study_to_neuromind(registry):
    path = _testdata("emri_small.dcm")
    if path is None:
        pytest.skip("pydicom test data unavailable")
    response = _client(registry).post(
        "/v1/studies/analyze",
        files={"file": (path.name, path.read_bytes(), "application/dicom")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["supported"] is True
    assert body["routing"]["selected_engine"] == "neuromind"
    # Whatever the engine did with it, the route is reported and the outcome is named.
    assert body["result"]["status"] in {"completed", "failed", "not_implemented"}


def test_declared_modality_cannot_override_a_confident_detection(registry):
    """A chest film submitted as a brain MRI is refused, not analysed.

    The mirror of the CASE-UPLOAD-27 failure: there, detection was ignored and a brain
    MRI reached the chest model. Here the caller *asserts* the wrong modality, and the
    declaration must lose to a calibrated detection rather than pick the engine.
    """
    films = _mimic_films(12, seed=5)
    film = next((f for f in films if validate_cxr(str(f)).ok), None)
    if film is None:
        pytest.skip("no gate-passing MIMIC film available")

    response = _client(registry).post(
        "/v1/studies/analyze?declared_modality=BRAIN_MRI",
        files={"file": (film.name, film.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "modality_conflict"
    assert body["detail"]["detected"] == "chest_xray"
    assert body["detail"]["declared"] == "brain_mri"


def test_disallowed_upload_type_is_rejected_before_decoding(registry):
    response = _client(registry).post(
        "/v1/studies/route",
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert response.json()["error"] == "upload_rejected"


# --------------------------------------------------------------------------- #
# 4. Batch validation on real films — the regression guard
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not MIMIC_ROOT.exists(), reason="MIMIC-CXR corpus not present")
def test_batch_real_films_route_to_thorax_and_never_to_neuromind(router: ModalityRouter):
    """Measured routing behaviour on real chest films.

    Two independent claims:

    * **sensitivity** — real chest films must reach Thorax at the rate the chest gate
      was validated at. The floor is set below the measured value so ordinary corpus
      variation does not fail the build, but a real regression does.
    * **safety** — chest films must essentially never be routed to NeuroMind, and the
      rate is bounded rather than asserted to zero.

    On why the safety bound is a rate and not ``== 0``. Two rules feed the head-geometry
    score. The framing rule separates the classes almost perfectly — over 4000 real MIMIC
    films the chest distribution reaches p99.95 = 0.816 against a brain minimum of 0.922
    — but it keys on a subject floating in air, which a crop destroys. The second rule
    tests the subject's *shape* so a head cropped to its own outline is still caught
    (this is what CASE-UPLOAD-27's upload actually was), and that one costs a measured
    2.0-2.5% of real chest films.

    That cost is accepted rather than tuned away, because the two directions are not
    symmetric:

    * a chest film reaching NeuroMind is **refused**: the engine requires a volumetric
      multi-sequence MR study and rejects a 2D radiograph at ``validate_input``, so the
      user gets a named refusal and no case is created;
    * a brain study reaching Thorax is **analysed**, and comes back as a chest
      diagnosis. That is CASE-UPLOAD-27, where an axial brain MRI was reported as
      pneumonia at p=0.25.

    So the thresholds protect the direction that produces a wrong report, and this bound
    is set above the measured divert rate with headroom — loose enough not to fail on
    corpus variation, tight enough that a genuine threshold regression (which would push
    this into double digits) still fails the build.
    """
    films = _mimic_films(200, seed=17)
    if len(films) < 50:
        pytest.skip("not enough films sampled")

    engines: dict[str, int] = {}
    for film in films:
        decision = _route(film, router)
        engines[decision.selected_engine or "none"] = (
            engines.get(decision.selected_engine or "none", 0) + 1)

    thorax_rate = engines.get("thorax", 0) / len(films)
    neuro_rate = engines.get("neuromind", 0) / len(films)
    print(f"\nrouted {len(films)} real chest films: {engines} "
          f"(thorax rate {thorax_rate:.3%}, neuromind rate {neuro_rate:.3%})")

    assert neuro_rate <= 0.05, (
        f"{neuro_rate:.3%} of chest films were routed to NeuroMind, above the 5% bound "
        f"(measured operating point is 2.0-2.5%) — a head-geometry threshold has "
        f"regressed")
    assert thorax_rate >= 0.90, f"chest routing rate fell to {thorax_rate:.3%}"
