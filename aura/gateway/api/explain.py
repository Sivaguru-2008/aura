"""Dedicated Explainability API — lazy-loading endpoint for visual explanations.

Decouples explainability computations (Grad-CAM++, LOO, Integrated Gradients)
from the primary inference path.  The dashboard loads the diagnostic report
instantly and lazy-loads expensive saliency maps in the background via this
endpoint.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/v1", tags=["explain"])

# Module-level references set at startup by app.py
_store = None
_vision_engine = None
_fusion_engine = None
_explain_engine = None


def init_explain_api(store=None, vision_engine=None, fusion_engine=None,
                     explain_engine=None):
    """Wire the engine references at application startup."""
    global _store, _vision_engine, _fusion_engine, _explain_engine
    _store = store
    _vision_engine = vision_engine
    _fusion_engine = fusion_engine
    _explain_engine = explain_engine


@router.get("/explain/{case_id}")
async def get_explanation(case_id: str):
    """Lazy-load the explanation for a completed case.

    Returns cached explanation if available, otherwise computes and caches it.
    This endpoint is designed to be called asynchronously by the dashboard after
    the primary report loads.
    """
    if _store is None:
        raise HTTPException(503, detail={"error": "store_not_initialized"})

    bundle = _store.get_case(case_id)
    if bundle is None:
        raise HTTPException(404, detail={"error": "case_not_found", "case_id": case_id})

    # Return cached explanation if already present and populated
    if (bundle.explanation and bundle.explanation.saliency
            and len(bundle.explanation.saliency) > 0):
        return {
            "case_id": case_id,
            "cached": True,
            "explanation": bundle.explanation.model_dump(),
        }

    # Compute explanation on demand
    if _vision_engine is None or _fusion_engine is None or _explain_engine is None:
        raise HTTPException(503, detail={"error": "engines_not_initialized"})

    try:
        img = np.array(bundle.image, dtype=float).reshape(bundle.image_shape)
        x = np.zeros(8, dtype=float)  # placeholder evidence vector
        if bundle.evidence:
            from aura.services.fusion.evidence import EVIDENCE_CHANNELS
            for item in bundle.evidence:
                if item.name in EVIDENCE_CHANNELS:
                    idx = EVIDENCE_CHANNELS.index(item.name)
                    x[idx] = item.value

        top = bundle.safety.top if bundle.safety else None
        if top is None:
            raise HTTPException(400, detail={"error": "no_top_diagnosis"})

        explanation = _explain_engine.explain(
            case_id, _vision_engine, img, _fusion_engine.model, x, top
        )

        # Update the bundle and persist
        bundle.explanation = explanation
        _store.save_case(bundle)

        return {
            "case_id": case_id,
            "cached": False,
            "explanation": explanation.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail={"error": "explanation_failed", "message": str(e)})


@router.get("/explain/{case_id}/methods")
async def get_explanation_methods(case_id: str):
    """Return available explanation methods for a case."""
    if _store is None:
        raise HTTPException(503, detail={"error": "store_not_initialized"})

    bundle = _store.get_case(case_id)
    if bundle is None:
        raise HTTPException(404, detail={"error": "case_not_found"})

    methods = []
    if bundle.explanation:
        methods = list(bundle.explanation.saliency_methods.keys())

    has_cnn = False
    if _vision_engine and hasattr(_vision_engine, "backbone") and _vision_engine.backbone is not None:
        has_cnn = True

    return {
        "case_id": case_id,
        "available_methods": methods or ["occlusion"],
        "has_cnn_backbone": has_cnn,
        "primary_method": bundle.explanation.method if bundle.explanation else "none",
    }
