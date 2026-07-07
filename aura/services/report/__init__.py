"""Clinical Report Generator.

Two-stage and grounded: a deterministic composer builds findings/impression/
recommendation text where *every sentence traces to evidence nodes*, then an
optional LLM adapter may polish phrasing behind a hallucination gate. The P0
default is the deterministic composer (offline, no PHI leaves the box).
"""
from services.report.engine import ReportEngine

__all__ = ["ReportEngine"]
