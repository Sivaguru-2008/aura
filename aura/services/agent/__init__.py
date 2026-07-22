"""Active Diagnostic Agent — sequential, information-gain-driven diagnosis.

Additive layer over the existing fusion + recommend (EVOI/EIG) + safety engines.
Turns AURA from a one-shot classifier into an agent that quantifies what it does
NOT know and closes that uncertainty by acquiring the single most informative next
test — measured in bits — abstaining until confident.
"""
from services.agent.active_diagnosis import ActiveDiagnosisAgent, DiagnosisTrajectory

__all__ = ["ActiveDiagnosisAgent", "DiagnosisTrajectory"]
