"""Longitudinal comparison engine for Brain MRI studies."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel

from aura.common.config import get_settings
from aura.schemas.contracts import CaseBundle

class ProgressionMetrics(BaseModel):
    whole_tumor_prev_mm3: Optional[float] = None
    whole_tumor_curr_mm3: Optional[float] = None
    whole_tumor_change_mm3: Optional[float] = None
    whole_tumor_change_pct: Optional[float] = None

    tumor_core_prev_mm3: Optional[float] = None
    tumor_core_curr_mm3: Optional[float] = None
    tumor_core_change_mm3: Optional[float] = None
    tumor_core_change_pct: Optional[float] = None

    enhancing_prev_mm3: Optional[float] = None
    enhancing_curr_mm3: Optional[float] = None
    enhancing_change_mm3: Optional[float] = None
    enhancing_change_pct: Optional[float] = None

    edema_prev_mm3: Optional[float] = None
    edema_curr_mm3: Optional[float] = None
    edema_change_mm3: Optional[float] = None
    edema_change_pct: Optional[float] = None

class ProgressionReport(BaseModel):
    previous_case_id: str
    current_case_id: str
    status: str  # "Stable" | "Progressive Disease" | "Regression" | "Cannot Determine"
    reason: str
    metrics: ProgressionMetrics
    growth_threshold: float
    regression_threshold: float
    report_markdown: str

class LongitudinalAnalyzer:
    """Analyze and compare two MR studies to track tumor progression."""

    @staticmethod
    def compare(
        prev: CaseBundle, 
        curr: CaseBundle,
        growth_threshold: Optional[float] = None,
        regression_threshold: Optional[float] = None
    ) -> ProgressionReport:
        settings = get_settings()
        g_thresh = growth_threshold if growth_threshold is not None else settings.progression_growth_threshold
        r_thresh = regression_threshold if regression_threshold is not None else settings.progression_regression_threshold

        # Helper to extract volume from CaseBundle
        def get_volume(bundle: CaseBundle, region_name: str) -> Optional[float]:
            # Extract from bundle.volumes if present
            if hasattr(bundle, "volumes") and bundle.volumes:
                reg_data = bundle.volumes.get(region_name)
                if reg_data and "volume_mm3" in reg_data:
                    return float(reg_data["volume_mm3"])

            # Fallback to parsing evidence list
            for item in bundle.evidence:
                if item.name == f"segmented_{region_name}":
                    # value is voxels. If we have spacing_mm in bundle metadata or image description,
                    # we might be able to compute volume, but we rely primarily on bundle.volumes
                    pass
            return None

        prev_wt = get_volume(prev, "whole_tumor")
        curr_wt = get_volume(curr, "whole_tumor")
        prev_core = get_volume(prev, "tumor_core")
        curr_core = get_volume(curr, "tumor_core")
        prev_enh = get_volume(prev, "enhancing_tumor")
        curr_enh = get_volume(curr, "enhancing_tumor")
        prev_edema = get_volume(prev, "edema")
        curr_edema = get_volume(curr, "edema")

        # If any required volume is missing or the studies are invalid:
        if prev_wt is None or curr_wt is None:
            # We check if they were empty segmentations (0 voxels)
            # If a study was successfully processed but has no segmented regions in evidence/volumes,
            # it means tumor volume is 0.0.
            if prev.vision and not prev.vision.findings:
                prev_wt = prev_wt or 0.0
                prev_core = prev_core or 0.0
                prev_enh = prev_enh or 0.0
                prev_edema = prev_edema or 0.0
            if curr.vision and not curr.vision.findings:
                curr_wt = curr_wt or 0.0
                curr_core = curr_core or 0.0
                curr_enh = curr_enh or 0.0
                curr_edema = curr_edema or 0.0

        if prev_wt is None or curr_wt is None:
            empty_metrics = ProgressionMetrics()
            return ProgressionReport(
                previous_case_id=prev.case_id,
                current_case_id=curr.case_id,
                status="Cannot Determine",
                reason="One or both scans are missing structured volume measurements.",
                metrics=empty_metrics,
                growth_threshold=g_thresh,
                regression_threshold=r_thresh,
                report_markdown="### PROGRESSION REPORT\n**Status**: Cannot Determine\n**Reason**: Missing structured volumetric measurements on one or both scans.",
            )

        # Helper to compute changes
        def compute_change(prev_v: float, curr_v: float) -> tuple[float, Optional[float]]:
            diff = curr_v - prev_v
            if prev_v > 0:
                pct = diff / prev_v
            elif curr_v > 0:
                pct = 1.0  # 100% growth from nothing
            else:
                pct = 0.0
            return round(diff, 2), round(pct, 4)

        wt_diff, wt_pct = compute_change(prev_wt, curr_wt)
        core_diff, core_pct = compute_change(prev_core or 0.0, curr_core or 0.0)
        enh_diff, enh_pct = compute_change(prev_enh or 0.0, curr_enh or 0.0)
        edema_diff, edema_pct = compute_change(prev_edema or 0.0, curr_edema or 0.0)

        metrics = ProgressionMetrics(
            whole_tumor_prev_mm3=prev_wt,
            whole_tumor_curr_mm3=curr_wt,
            whole_tumor_change_mm3=wt_diff,
            whole_tumor_change_pct=wt_pct,

            tumor_core_prev_mm3=prev_core,
            tumor_core_curr_mm3=curr_core,
            tumor_core_change_mm3=core_diff,
            tumor_core_change_pct=core_pct,

            enhancing_prev_mm3=prev_enh,
            enhancing_curr_mm3=curr_enh,
            enhancing_change_mm3=enh_diff,
            enhancing_change_pct=enh_pct,

            edema_prev_mm3=prev_edema,
            edema_curr_mm3=curr_edema,
            edema_change_mm3=edema_diff,
            edema_change_pct=edema_pct,
        )

        # Determine status
        # If there is no tumor in both: stable
        if prev_wt == 0.0 and curr_wt == 0.0:
            status = "Stable"
            reason = "No tumor burden detected in either the baseline or follow-up scans."
        elif prev_wt == 0.0 and curr_wt > 0.0:
            status = "Progressive Disease"
            reason = f"New tumor burden of {curr_wt:.1f} mm3 detected on the follow-up scan where baseline was clear."
        else:
            # We base the classification on the whole tumor growth percentage
            if wt_pct >= g_thresh:
                status = "Progressive Disease"
                reason = f"Whole tumor volume increased by {wt_pct * 100:.1f}%, exceeding the progressive disease threshold of {g_thresh * 100:.1f}%."
            elif wt_pct <= r_thresh:
                status = "Regression"
                reason = f"Whole tumor volume decreased by {abs(wt_pct) * 100:.1f}%, exceeding the regression threshold of {abs(r_thresh) * 100:.1f}%."
            else:
                status = "Stable"
                reason = f"Whole tumor volume change ({wt_pct * 100:+.1f}%) lies within the stable range ({r_thresh * 100:.1f}% to {g_thresh * 100:.1f}%)."

        # Generate report markdown
        report_md = f"""### LONGITUDINAL PROGRESSION REPORT
- **Baseline Case**: {prev.case_id}
- **Follow-up Case**: {curr.case_id}
- **Progression Status**: **{status}**
- **Clinical Assessment**: {reason}

#### Volumetric Progression Summary
| Region | Baseline Vol (mm³) | Follow-up Vol (mm³) | Absolute Change (mm³) | Growth (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Whole Tumor** | {prev_wt:.1f} | {curr_wt:.1f} | {wt_diff:+.1f} | {wt_pct * 100:+.1f}% |
| **Tumor Core** | {prev_core or 0.0:.1f} | {curr_core or 0.0:.1f} | {core_diff:+.1f} | {core_pct * 100:+.1f}% |
| **Enhancing Component** | {prev_enh or 0.0:.1f} | {curr_enh or 0.0:.1f} | {enh_diff:+.1f} | {enh_pct * 100:+.1f}% |
| **Peritumoral Edema** | {prev_edema or 0.0:.1f} | {curr_edema or 0.0:.1f} | {edema_diff:+.1f} | {edema_pct * 100:+.1f}% |

*Note: Progression analysis performed using configured thresholds (PD: ≥{g_thresh * 100:.0f}%, Regression: ≤{r_thresh * 100:.0f}%).*
"""

        return ProgressionReport(
            previous_case_id=prev.case_id,
            current_case_id=curr.case_id,
            status=status,
            reason=reason,
            metrics=metrics,
            growth_threshold=g_thresh,
            regression_threshold=r_thresh,
            report_markdown=report_md,
        )
