"""Tumor tracking service to monitor volumetric changes across multiple scans."""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

class TrackingPoint(BaseModel):
    case_id: str
    date: datetime
    whole_tumor_volume: float
    tumor_core_volume: float
    enhancing_volume: float
    edema_volume: float

class TrackingTimeline(BaseModel):
    timeline: List[TrackingPoint]
    growth_rate_pct: float
    message: str

class TumorTracker:
    """Track brain tumor measurements across all historic cases in the store."""

    @staticmethod
    def get_timeline(store: Any, current_case_id: str) -> TrackingTimeline:
        # Retrieve all case summaries from the database
        summaries = store.list_cases(limit=1000)
        
        # Load full case bundles for brain cases
        brain_cases = []
        for summary in summaries:
            case_id = summary["case_id"]
            if case_id.startswith("CASE-MR-"):
                case = store.get_case(case_id)
                if case:
                    brain_cases.append(case)
        
        # Sort by creation date
        brain_cases.sort(key=lambda c: c.created_at)

        timeline = []
        for case in brain_cases:
            wt = 0.0
            core = 0.0
            enh = 0.0
            edema = 0.0
            if case.volumes:
                wt = case.volumes.get("whole_tumor", {}).get("volume_mm3", 0.0)
                core = case.volumes.get("tumor_core", {}).get("volume_mm3", 0.0)
                enh = case.volumes.get("enhancing_tumor", {}).get("volume_mm3", 0.0)
                edema = case.volumes.get("edema", {}).get("volume_mm3", 0.0)
            
            timeline.append(TrackingPoint(
                case_id=case.case_id,
                date=case.created_at,
                whole_tumor_volume=wt,
                tumor_core_volume=core,
                enhancing_volume=enh,
                edema_volume=edema,
            ))

        if len(timeline) >= 2:
            first_vol = timeline[0].whole_tumor_volume
            last_vol = timeline[-1].whole_tumor_volume
            if first_vol > 0:
                growth_rate = ((last_vol - first_vol) / first_vol) * 100.0
            else:
                growth_rate = 100.0 if last_vol > 0 else 0.0
            message = f"Tracked {len(timeline)} scans from {timeline[0].date.strftime('%Y-%m-%d')} to {timeline[-1].date.strftime('%Y-%m-%d')}."
        else:
            growth_rate = 0.0
            message = "Single scan baseline. Additional historical scans are required to establish progression trends."

        return TrackingTimeline(
            timeline=timeline,
            growth_rate_pct=round(growth_rate, 2),
            message=message
        )
