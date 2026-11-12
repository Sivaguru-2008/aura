"""Routing-layer HTTP endpoints.

    POST /v1/studies/route     detect modality + select engine, no analysis
    POST /v1/studies/analyze   detect, select, and run the engine
    GET  /v1/engines           registered engines, modalities, detector configuration

``POST /v1/studies/upload`` — the legacy chest-X-ray endpoint — is untouched and
keeps its exact behaviour. Existing clients need no change; new clients use
``/analyze`` and get modality independence.

Why ``/route`` exists as its own endpoint: a client should be able to ask "what is
this and would you analyse it?" before committing to an inference run. It is cheap
(no model load, no engine construction) and it is the endpoint to reach for when
diagnosing a rejected or misrouted upload, since it returns every candidate the
detector scored, with the evidence behind each one.
"""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ..core.shared.errors import AuraBackendError
from ..core.shared.logging import get_logger, new_correlation_id, use_correlation_id
from ..models.routing import AnalysisEnvelope, RoutingMetadata
from ..services.dispatch import DispatchService

log = get_logger("api")


def build_router(dispatch: DispatchService) -> APIRouter:
    """Build the API router bound to a dispatch service.

    Takes the service as an argument rather than importing a singleton so tests can
    mount the routes against a stub, and so a deployment can run more than one
    configuration in one process.
    """
    api = APIRouter(tags=["modality-router"])

    def _error(exc: AuraBackendError) -> JSONResponse:
        """Uniform error rendering. Internal detail stays in the log."""
        log.info(
            "request rejected",
            extra={"context": {"code": exc.code, "reason": exc.reason}},
        )
        return JSONResponse(exc.to_payload(), status_code=exc.http_status)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/route", response_model=RoutingMetadata,
              summary="Identify a study's modality and the engine that would analyse it")
    async def route_study(file: UploadFile | None = File(None),
                          files: list[UploadFile] | None = File(None)):
        """Inspect an upload without analysing it.

        Returns the full :class:`RoutingMetadata`, including every scored candidate.
        Unroutable and unsupported uploads still return **200** here — "I could not
        identify this" is a successful inspection, and the body explains why. Only a
        transport-level rejection (disallowed type, oversized) is an error status.
        """
        from ..core.upload import UploadIntake

        request_id = new_correlation_id()
        with use_correlation_id(request_id):
            try:
                if files:
                    async with UploadIntake().receive_multiple(files) as asset:
                        return dispatch.inspect(asset)
                elif file:
                    async with UploadIntake().receive(file) as asset:
                        return dispatch.inspect(asset)
                else:
                    from ..core.shared.errors import UploadRejected
                    raise UploadRejected("No file or files uploaded.", http_status=400)
            except AuraBackendError as exc:
                return _error(exc)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/analyze", response_model=AnalysisEnvelope,
              summary="Route a study to its engine and analyse it")
    async def analyze_study(file: UploadFile | None = File(None),
                            files: list[UploadFile] | None = File(None),
                            declared_modality: str | None = None,
                            force_modality: str | None = None,
                            age: int | None = None,
                            sex: str | None = None,
                            symptoms: str | None = None,
                            previous_diagnosis: str | None = None,
                            previous_surgery: str | None = None,
                            radiotherapy: bool | None = None,
                            chemotherapy: bool | None = None,
                            clinical_notes: str | None = None):
        """Route and analyse in one call — the modality-agnostic upload endpoint."""
        from ..core.upload import UploadIntake

        request_id = new_correlation_id()
        with use_correlation_id(request_id):
            try:
                clinical_context = {
                    "age": age,
                    "sex": sex,
                    "symptoms": symptoms,
                    "previous_diagnosis": previous_diagnosis,
                    "previous_surgery": previous_surgery,
                    "radiotherapy": radiotherapy,
                    "chemotherapy": chemotherapy,
                    "clinical_notes": clinical_notes,
                }
                # Filter out None values
                clinical_context = {k: v for k, v in clinical_context.items() if v is not None}

                if files:
                    async with UploadIntake().receive_multiple(files) as asset:
                        return await dispatch.dispatch(
                            asset, request_id=request_id,
                            declared_modality=declared_modality or force_modality,
                            clinical_context=clinical_context)
                elif file:
                    async with UploadIntake().receive(file) as asset:
                        return await dispatch.dispatch(
                            asset, request_id=request_id,
                            declared_modality=declared_modality or force_modality,
                            clinical_context=clinical_context)
                else:
                    from ..core.shared.errors import UploadRejected
                    raise UploadRejected("No file or files uploaded.", http_status=400)
            except AuraBackendError as exc:
                return _error(exc)

    # ------------------------------------------------------------------ #
    @api.post("/v1/studies/preview",
              summary="Parse and extract metadata and preview thumbnails of an upload without running inference")
    async def preview_study(file: UploadFile | None = File(None),
                            files: list[UploadFile] | None = File(None)):
        """Parse and validate the upload, generating preview thumbnails and metadata without running AI inference."""
        from ..core.upload import UploadIntake
        from ..foundation.mri.intake_manager import MRIIntakeManager
        from PIL import Image
        import io
        import base64
        import numpy as np

        async def _do_preview(asset) -> dict[str, Any]:
            # Process using MRIIntakeManager
            manager = MRIIntakeManager()
            study = manager.process(asset.path)

            # Extract shape and spacing
            H, W, Z = study.volumes.shape[1:4]
            spacing = list(study.spacing_mm)

            # Generate 2D thumbnails for each sequence
            thumbnails = {}
            order = ["flair", "t1", "t1ce", "t2"]
            for c, seq_key in enumerate(order):
                vol = study.volumes[c] # Shape (H, W, Z)
                slice_2d = vol[:, :, Z // 2] # Get middle slice
                # Normalize to 0-255 uint8
                lo, hi = float(slice_2d.min()), float(slice_2d.max())
                if hi - lo > 1e-6:
                    slice_uint8 = np.clip((slice_2d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                else:
                    slice_uint8 = np.zeros(slice_2d.shape, dtype=np.uint8)

                # Flip vertically so it's not upside down (matches drawing logic)
                slice_uint8 = np.flipud(slice_uint8)

                # Encode to base64 PNG
                img = Image.fromarray(slice_uint8)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
                thumbnails[seq_key] = f"data:image/png;base64,{b64_str}"

            affine_repr = f"Diagonal spacing affine matrix [{spacing[0]}mm, {spacing[1]}mm, {spacing[2]}mm]"

            # Everything reported here is read off the parsed study. Fields the
            # intake layer does not carry (``MultiSequenceStudy`` holds volumes,
            # sequence keys, spacing and channel-order provenance — no DICOM
            # demographics, no scanner tags, no affine orientation) are returned as
            # null so the console renders "—". A preview that invents a patient id,
            # an orientation or a scanner make is indistinguishable from one that
            # read them off the file, which is exactly the confusion a clinical
            # intake screen must not create.
            seq_keys = list(getattr(study, "sequence_keys", ()) or [])
            n_seq = len(seq_keys) or study.volumes.shape[0]

            return {
                "status": "success",
                "patient_id": None,               # not carried by the intake layer
                "study_id": f"STU-MR-{asset.sha256[:12]}",
                "detected_modalities": [s.upper() for s in seq_keys],
                "voxel_spacing": spacing,
                "orientation": None,              # no affine is retained upstream
                "original_dimensions": [H, W, Z],
                "number_of_slices": Z,
                "affine_matrix_summary": affine_repr,
                "sequence_type": f"Multi-sequence 3D MRI study ({n_seq} sequences)",
                "channel_order_source": getattr(study, "order_source", None),
                "scanner_metadata": None,         # no DICOM tags retained upstream
                "thumbnails": thumbnails
            }

        request_id = new_correlation_id()
        with use_correlation_id(request_id):
            try:
                if files:
                    async with UploadIntake().receive_multiple(files) as asset:
                        return await _do_preview(asset)
                elif file:
                    async with UploadIntake().receive(file) as asset:
                        return await _do_preview(asset)
                else:
                    from ..core.shared.errors import UploadRejected
                    raise UploadRejected("No file or files uploaded.", http_status=400)
            except AuraBackendError as exc:
                return _error(exc)
            except Exception as exc:
                from ..foundation.mri.errors import StudyValidationError
                return _error(StudyValidationError(f"Preview failed: {exc}"))

    # ------------------------------------------------------------------ #
    @api.post("/v1/cases/progression",
              summary="Compare a previous MRI with a current MRI to track progression")
    def compare_cases(
        previous_case_id: str,
        current_case_id: str,
        growth_threshold: float | None = None,
        regression_threshold: float | None = None,
    ):
        from aura.gateway.app import store
        from ..services.reasoning.progression import LongitudinalAnalyzer
        from fastapi import HTTPException

        prev = store().get_case(previous_case_id)
        curr = store().get_case(current_case_id)
        if prev is None:
            raise HTTPException(status_code=404, detail=f"Previous case {previous_case_id} not found.")
        if curr is None:
            raise HTTPException(status_code=404, detail=f"Current case {current_case_id} not found.")

        report = LongitudinalAnalyzer.compare(
            prev, curr,
            growth_threshold=growth_threshold,
            regression_threshold=regression_threshold
        )
        return report

    # ------------------------------------------------------------------ #
    @api.get("/v1/cases/{case_id}/tracking",
             summary="Get historical tumor tracking timeline for a case")
    def track_case(case_id: str):
        from aura.gateway.app import store
        from ..services.reasoning.tracking import TumorTracker
        from fastapi import HTTPException

        case = store().get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

        timeline = TumorTracker.get_timeline(store(), case_id)
        return timeline

    # ------------------------------------------------------------------ #
    @api.get("/v1/engines",
             summary="Registered analysis engines, served modalities, and detector config")
    def list_engines():
        """Platform capability listing.

        `status` distinguishes `available` from `placeholder` (routes resolve,
        analysis pending) and `unavailable` (registered but failed to construct on
        this deployment), so a client can tell what will happen before uploading.
        """
        return dispatch.describe()

    # ------------------------------------------------------------------ #
    @api.get("/v1/quantum/providers",
             summary="Quantum execution providers, availability, and fallback chain")
    def quantum_providers():
        """Which execution surfaces this deployment can reach.

        Cheap and non-blocking — reports SDK/credential presence without
        enumerating remote devices. Use `/v1/quantum/backends` for discovery.
        """
        from aura.services.quantum import describe

        return describe()

    # ------------------------------------------------------------------ #
    @api.get("/v1/quantum/backends",
             summary="Discover quantum devices (queue depth, qubits, error rates)")
    def quantum_backends(provider: str | None = None, min_qubits: int = 1):
        """Live device discovery. Read-only and free — consumes no QPU quota.

        Unavailable providers are returned as entries carrying a `reason`
        rather than being omitted, so a caller can tell "no devices" from
        "not configured".
        """
        from aura.services.quantum import list_backends

        return {"backends": list_backends(provider=provider, min_qubits=min_qubits)}

    # ------------------------------------------------------------------ #
    @api.get("/v1/quantum/verify",
             summary="Verify SDK circuit translations against the PennyLane reference")
    def quantum_verify():
        """Prove the Qiskit/Braket rebuilds are the circuit the model was trained on.

        Runs on statevector simulators only. A mismatch means hardware would
        execute a different unitary and produce confidently wrong values, so
        this gates any hardware run.
        """
        import numpy as np

        from aura.services.quantum.base import CircuitSpec
        from aura.services.quantum.benchmark import served_vqc_spec, verify_translation

        out: dict[str, Any] = {}
        try:
            spec, _ = served_vqc_spec()
            out["served_fusion_vqc"] = verify_translation(spec)
        except Exception as exc:
            out["served_fusion_vqc"] = {"error": str(exc)}
        rng = np.random.default_rng(0)
        out["qkl_fidelity_kernel"] = verify_translation(
            CircuitSpec(kind="iqp_kernel", n_qubits=6, x=rng.random(6), x2=rng.random(6))
        )
        checked = [v.get("n_checked", 0) for v in out.values() if isinstance(v, dict)]
        out["any_sdk_available"] = sum(checked) > 0
        return out

    # ------------------------------------------------------------------ #
    @api.get("/v1/quantum/qkl",
             summary="Quantum-kernel brain classifier: weights, task, and held-out metrics")
    def qkl_status():
        """Serving status of the QKL head.

        Exposes what the classifier is *actually* trained on so a client can
        never present its output as a claim it is not entitled to make. Returns
        ``trained: false`` (rather than erroring) when weights are absent.
        """
        import json

        from aura.backend.engines.neuro.qkl import DEFAULT_WEIGHTS, QKLClassifier
        from aura.common.config import get_settings
        from aura.ml.training.train_qkl import REPORT_PATH

        clf = QKLClassifier.load()
        payload: dict[str, Any] = {
            "enabled": bool(get_settings().neuro_qkl_enabled),
            "trained": clf.is_trained,
            "weights_path": str(DEFAULT_WEIGHTS),
            "weights_present": DEFAULT_WEIGHTS.exists(),
            "task": clf.task,
            "classes": list(clf.classes),
            "n_qubits": clf.n_qubits,
            "hilbert_dim": 2 ** clf.n_qubits,
            "feature_map": "IQP (Hadamard + RZ + ring ZZ)",
            "kernel": "fidelity |<phi(x)|phi(x')>|^2",
            "support_vectors": int(len(clf.support_vectors)),
            "provenance": clf.provenance,
        }
        if REPORT_PATH.exists():
            try:
                report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
                payload["evaluation"] = {
                    "split": report.get("split"),
                    "test_quantum": report.get("test_quantum"),
                    "test_classical_rbf": report.get("test_classical_rbf"),
                    "bootstrap_ci_quantum": report.get("bootstrap_ci_quantum"),
                    "quantum_minus_classical_auroc": report.get("quantum_minus_classical_auroc"),
                    "label_axis_note": report.get("label_axis_note"),
                }
            except Exception:
                payload["evaluation"] = None
        return payload

    # ------------------------------------------------------------------ #
    @api.get("/v1/quantum/evidence",
             summary="Every measured quantum claim, read from its artifact")
    def quantum_evidence():
        """Consolidated evidence for the quantum layer, for the /quantum view.

        Each block is read straight from the artifact the generating script wrote,
        so nothing here can drift from what was actually measured — the same rule
        docs/BENCHMARKS.md follows. A missing artifact yields ``null`` for that
        block and an entry in ``missing`` rather than a placeholder number, because
        a plausible-looking default is exactly what this project refuses to serve.

        Deliberately includes the results that do **not** favour the quantum layer
        (the entanglement ablation, the design sweep, the QKL comparison). A client
        rendering only the flattering half would be misrepresenting the system, and
        the negative results are the ones a reviewer should see first.
        """
        import json

        from aura.common.config import ARTIFACTS, get_settings

        def _read(name: str):
            path = ARTIFACTS / name
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

        def _qae_served() -> bool:
            """True only if the autoencoder would actually compress an embedding."""
            try:
                from aura.services.fusion.qae import QuantumAutoencoder

                return bool(get_settings().qae_enabled) and QuantumAutoencoder.load() is not None
            except Exception:
                return False

        def _qbn_served() -> bool:
            """True only if a *trained* QBN would refine the posterior.

            load_trained() returns None when no fitted artifact is present; the
            reasoner then returns the rule-adjusted posterior untouched rather than
            serving the constructor's six unfitted constants as a quantum inference.
            """
            try:
                from aura.services.reasoning.qbn import QuantumBayesianNetwork

                return QuantumBayesianNetwork.load_trained() is not None
            except Exception:
                return False

        s = get_settings()
        sources = {
            "hardware": "ibm_hardware_run.json",
            "noise": "noise_rung.json",
            "transpile": "transpile_study.json",
            "design_sweep": "design_sweep.json",
            "study": "quantum_study.json",
            "benchmark": "benchmark.json",
        }
        blocks = {key: _read(name) for key, name in sources.items()}
        missing = [sources[k] for k, v in blocks.items() if v is None]

        payload: dict[str, Any] = {
            "served": {
                "fusion_backend": s.fusion_backend,
                "n_qubits": s.n_qubits,
                "n_layers": s.n_layers,
                "n_shots": s.n_shots,
                "entangler": "ring",
                "encoding": "RY(pi * x_i), one evidence channel per qubit",
                "readout": "<Z_i> -> linear head -> softmax",
                "neuro_qkl_enabled": bool(s.neuro_qkl_enabled),
                # Derived, never hardcoded. Both modules are wired into their engine
                # but load a trained artifact that does not ship, so today both are
                # False -- and if either is ever trained this flips on its own rather
                # than leaving the page asserting a stale "not served".
                "qae_served": _qae_served(),
                "qbn_served": _qbn_served(),
            },
            "missing": missing,
        }

        hw = (blocks["hardware"] or {}).get("hardware") or {}
        if hw:
            payload["hardware"] = {
                "backend": hw.get("backend"),
                "backend_qubits": hw.get("backend_qubits"),
                "job_id": hw.get("job_id"),
                "mean_abs_z_error_vs_analytic": hw.get("mean_abs_z_error_vs_analytic"),
                "top1_agrees_with_analytic": hw.get("top1_agrees_with_analytic"),
                "generated": (blocks["hardware"] or {}).get("generated"),
            }

        if blocks["noise"]:
            n = blocks["noise"]
            payload["noise"] = {
                "attribution": n.get("attribution"),
                "rungs": [
                    {"rung": r.get("rung"), "shots": r.get("shots"),
                     "mean_abs_z_error_vs_analytic": r.get("mean_abs_z_error_vs_analytic"),
                     "top1_agrees_with_analytic": r.get("top1_agrees_with_analytic")}
                    for r in n.get("rungs", [])
                ],
                "hardware_reference": n.get("hardware_reference"),
                "circuit": n.get("circuit"),
            }

        if blocks["transpile"]:
            t = blocks["transpile"]
            payload["transpile"] = {
                "backend": t.get("backend"),
                "logical": t.get("logical"),
                "levels": t.get("levels"),
                "served_level": t.get("served_level"),
                "improvement_vs_level_1": t.get("improvement_vs_level_1"),
            }

        if blocks["design_sweep"]:
            d = blocks["design_sweep"]
            cells = d.get("cells", [])
            payload["design_sweep"] = {
                "generated": d.get("generated"),
                "data": d.get("data"),
                "protocol": d.get("protocol"),
                "served": d.get("served"),
                "cells": cells,
                "n_cells": len(cells),
            }

        study = blocks["study"] or {}
        if study.get("q1_q2_ablation"):
            payload["entanglement_ablation"] = study["q1_q2_ablation"].get("entanglement_effect")
        if study.get("q3_measurement_budget"):
            payload["measurement_budget"] = study["q3_measurement_budget"]
        if study.get("q4_evidence_coupling"):
            c = study["q4_evidence_coupling"]
            payload["evidence_coupling"] = {
                "channels": c.get("channels"),
                "matrix": c.get("mean_abs_differential_matrix"),
            }

        if blocks["benchmark"]:
            payload["fusion_backends"] = (blocks["benchmark"] or {}).get("metrics_full")

        return payload

    return api
