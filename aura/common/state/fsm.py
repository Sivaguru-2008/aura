"""Decision State Machine — formalizes the diagnostic pipeline as an FSM.

Tracks pipeline stages and enforces transition constraints so the system
can never violate safety controls (e.g., blocking READY if safety is UNSAFE,
or REPORT if DRP needs additional evidence without clinician bypass).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class PipelineState(str, Enum):
    """All possible states of the diagnostic pipeline FSM."""
    INPUT = "input"
    SAFETY_CHECK = "safety_check"
    EVIDENCE_COLLECTION = "evidence_collection"
    REASONING = "reasoning"
    READY = "ready"
    REPORT = "report"
    ABSTAINED = "abstained"
    FAILED = "failed"


class SafetyVerdict(str, Enum):
    """Safety controller verdicts that gate transitions."""
    SAFE = "safe"
    UNSAFE = "unsafe"
    CONDITIONAL = "conditional"


class ReadinessVerdict(str, Enum):
    """CDRE readiness verdicts that gate transitions."""
    READY = "ready"
    NEEDS_ADDITIONAL_EVIDENCE = "needs_additional_evidence"
    NOT_READY = "not_ready"


# Valid transitions: state -> set of states it can move to
_VALID_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.INPUT: {PipelineState.SAFETY_CHECK, PipelineState.FAILED},
    PipelineState.SAFETY_CHECK: {
        PipelineState.EVIDENCE_COLLECTION,
        PipelineState.ABSTAINED,
        PipelineState.FAILED,
    },
    PipelineState.EVIDENCE_COLLECTION: {
        PipelineState.REASONING,
        PipelineState.FAILED,
    },
    PipelineState.REASONING: {
        PipelineState.READY,
        PipelineState.ABSTAINED,
        PipelineState.FAILED,
    },
    PipelineState.READY: {PipelineState.REPORT, PipelineState.ABSTAINED},
    PipelineState.REPORT: set(),  # terminal
    PipelineState.ABSTAINED: set(),  # terminal
    PipelineState.FAILED: set(),  # terminal
}


class TransitionViolation(Exception):
    """Raised when a forbidden state transition is attempted."""
    def __init__(self, from_state: PipelineState, to_state: PipelineState,
                 reason: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        msg = f"Cannot transition from {from_state.value} to {to_state.value}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PipelineFSM:
    """Finite State Machine governing the diagnostic pipeline.

    Enforces:
      * SAFETY_CHECK → READY blocked when safety == UNSAFE
      * REASONING → READY blocked when DRP == NEEDS_ADDITIONAL_EVIDENCE
        without explicit clinician bypass
      * Only valid state transitions are allowed
    """

    def __init__(self):
        self._state = PipelineState.INPUT
        self._history: list[tuple[PipelineState, PipelineState, str]] = []
        self._safety_verdict: SafetyVerdict = SafetyVerdict.SAFE
        self._readiness_verdict: ReadinessVerdict = ReadinessVerdict.NOT_READY
        self._clinician_bypass: bool = False
        self._constraint_violations: list[str] = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def history(self) -> list[tuple[PipelineState, PipelineState, str]]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._state in (PipelineState.REPORT, PipelineState.ABSTAINED, PipelineState.FAILED)

    @property
    def constraint_violations(self) -> list[str]:
        return list(self._constraint_violations)

    def set_safety_verdict(self, verdict: SafetyVerdict) -> None:
        """Record the safety controller's verdict before transitioning."""
        self._safety_verdict = verdict

    def set_readiness_verdict(self, verdict: ReadinessVerdict) -> None:
        """Record the CDRE's readiness verdict before transitioning."""
        self._readiness_verdict = verdict

    def set_clinician_bypass(self, bypass: bool) -> None:
        """Enable clinician override for DRP-gated transitions."""
        self._clinician_bypass = bypass

    def can_transition(self, to_state: PipelineState) -> bool:
        """Check whether a transition is valid without actually performing it."""
        try:
            self._validate_transition(to_state)
            return True
        except TransitionViolation:
            return False

    def transition(self, to_state: PipelineState, reason: str = "") -> None:
        """Attempt a state transition. Raises ``TransitionViolation`` on failure."""
        self._validate_transition(to_state)
        old = self._state
        self._state = to_state
        self._history.append((old, to_state, reason))

    def _validate_transition(self, to_state: PipelineState) -> None:
        """Validate that a transition is allowed and satisfies constraints."""
        # 1. Terminal states cannot transition
        if self._state in (PipelineState.REPORT, PipelineState.ABSTAINED, PipelineState.FAILED):
            raise TransitionViolation(
                self._state, to_state,
                f"Already in terminal state {self._state.value}"
            )

        # 2. Must be a valid transition in the graph
        valid_targets = _VALID_TRANSITIONS.get(self._state, set())
        if to_state not in valid_targets:
            raise TransitionViolation(
                self._state, to_state,
                f"No edge from {self._state.value} to {to_state.value}"
            )

        # 3. SAFETY_CHECK → READY/REASONING: safety must not be UNSAFE
        if self._state == PipelineState.SAFETY_CHECK and self._safety_verdict == SafetyVerdict.UNSAFE:
            if to_state in (PipelineState.EVIDENCE_COLLECTION, PipelineState.REASONING, PipelineState.READY):
                self._constraint_violations.append(
                    f"Blocked {self._state.value} → {to_state.value}: safety is UNSAFE"
                )
                raise TransitionViolation(
                    self._state, to_state,
                    "Safety controller verdict is UNSAFE — must abstain or fail"
                )

        # 4. REASONING → READY: DRP must not require additional evidence without bypass
        if self._state == PipelineState.REASONING and to_state == PipelineState.READY:
            if (self._readiness_verdict == ReadinessVerdict.NEEDS_ADDITIONAL_EVIDENCE
                    and not self._clinician_bypass):
                self._constraint_violations.append(
                    "Blocked reasoning → ready: DRP requires additional evidence "
                    "(no clinician bypass)"
                )
                raise TransitionViolation(
                    self._state, to_state,
                    "DRP states NEEDS_ADDITIONAL_EVIDENCE without clinician bypass"
                )

    def reset(self) -> None:
        """Reset the FSM to the initial state."""
        self._state = PipelineState.INPUT
        self._history.clear()
        self._safety_verdict = SafetyVerdict.SAFE
        self._readiness_verdict = ReadinessVerdict.NOT_READY
        self._clinician_bypass = False
        self._constraint_violations.clear()

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of the FSM state."""
        return {
            "state": self._state.value,
            "is_terminal": self.is_terminal,
            "safety_verdict": self._safety_verdict.value,
            "readiness_verdict": self._readiness_verdict.value,
            "clinician_bypass": self._clinician_bypass,
            "transitions": [
                {"from": f.value, "to": t.value, "reason": r}
                for f, t, r in self._history
            ],
            "constraint_violations": self._constraint_violations,
        }
