"""In-process async event bus — pub-sub with typed events, wildcards, and chaining.

The production design uses Redis Streams / NATS; this is the same publish/subscribe
contract behind an in-memory implementation so the P0 stack runs from one process.
Swapping to Redis means replacing this file only — subscribers are unchanged.

Enhancements over the basic version:
  * Typed event classes with dataclass payloads
  * Wildcard topic subscriptions (e.g. ``study.*`` matches ``study.received``)
  * Event chaining: one event can publish another, with cycle detection
  * Priority ordering for handler execution
  * Optional event history for debugging
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# ─── Event types ────────────────────────────────────────────────────────


@dataclass
class Event:
    """Base event — all bus events inherit from this."""
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


@dataclass
class StudyReceivedEvent(Event):
    """Emitted when a new study arrives via the gateway."""
    topic: str = "study.received"
    study_id: str = ""
    case_id: str = ""


@dataclass
class VisionCompletedEvent(Event):
    """Emitted when the vision engine finishes inference."""
    topic: str = "vision.completed"
    case_id: str = ""
    findings: list[str] = field(default_factory=list)


@dataclass
class FusionCompletedEvent(Event):
    """Emitted when the fusion engine produces a diagnosis."""
    topic: str = "fusion.completed"
    case_id: str = ""
    top_diagnosis: str = ""
    top_probability: float = 0.0


@dataclass
class SafetyCheckedEvent(Event):
    """Emitted when the safety controller clears or vetoes a case."""
    topic: str = "safety.checked"
    case_id: str = ""
    cleared: bool = False
    veto_reason: str = ""


@dataclass
class ReasoningCompletedEvent(Event):
    """Emitted when the reasoner finishes evidence assembly."""
    topic: str = "reasoning.completed"
    case_id: str = ""


@dataclass
class CaseReadyEvent(Event):
    """Emitted when a case is ready for clinician review."""
    topic: str = "case.ready"
    case_id: str = ""
    priority_score: float = 0.0


@dataclass
class FeedbackRecordedEvent(Event):
    """Emitted when clinician feedback is saved."""
    topic: str = "feedback.recorded"
    case_id: str = ""


@dataclass
class DRPComputedEvent(Event):
    """Emitted when the CDRE produces a readiness profile."""
    topic: str = "drp.computed"
    case_id: str = ""
    limiting_factor: str = ""
    overall_score: float = 0.0


# ─── Canonical topics ──────────────────────────────────────────────────

STUDY_RECEIVED = "study.received"
VISION_COMPLETED = "vision.completed"
FUSION_COMPLETED = "fusion.completed"
SAFETY_CHECKED = "safety.checked"
REASONING_COMPLETED = "reasoning.completed"
CASE_READY = "case.ready"
FEEDBACK_RECORDED = "feedback.recorded"
DRP_COMPUTED = "drp.computed"

ALL_TOPICS = [
    STUDY_RECEIVED, VISION_COMPLETED, FUSION_COMPLETED,
    SAFETY_CHECKED, REASONING_COMPLETED, CASE_READY,
    FEEDBACK_RECORDED, DRP_COMPUTED,
]

# ─── Handler types ──────────────────────────────────────────────────────

Handler = Callable[[Event], Awaitable[None]]


@dataclass
class Subscription:
    """A subscription record with priority and metadata."""
    handler: Handler
    priority: int = 0
    topic: str = ""
    once: bool = False
    tag: str = ""


# ─── Event history entry ───────────────────────────────────────────────

@dataclass
class HistoryEntry:
    event: Event
    handler_count: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


# ─── EventBus ──────────────────────────────────────────────────────────

class EventBus:
    """Pub-sub event bus with wildcards, priorities, and event chaining.

    Topics use dot-separated segments.  Wildcards:
      * ``*`` matches one segment  (e.g. ``study.*`` matches ``study.received``)
      * ``**`` matches any number of segments (e.g. ``**`` matches everything)
    """

    def __init__(self, *, max_history: int = 200, enable_history: bool = True) -> None:
        self._subs: dict[str, list[Subscription]] = defaultdict(list)
        self._history: list[HistoryEntry] = []
        self._max_history = max_history
        self._enable_history = enable_history
        self._publish_depth = 0
        self._max_depth = 10
        self._chain_seen: set[str] = set()

    def subscribe(self, topic: str, handler: Handler, *,
                  priority: int = 0, once: bool = False, tag: str = "") -> None:
        """Subscribe to a topic.  Wildcards ``*`` and ``**`` are supported."""
        sub = Subscription(handler=handler, priority=priority, topic=topic,
                           once=once, tag=tag)
        self._subs[topic].append(sub)
        self._subs[topic].sort(key=lambda s: s.priority, reverse=True)

    def unsubscribe(self, topic: str, tag: str = "") -> None:
        """Remove all (or specific tagged) subscriptions for a topic."""
        if tag:
            self._subs[topic] = [
                s for s in self._subs[topic] if s.tag != tag
            ]
        else:
            self._subs.pop(topic, None)

    def _match_topic(self, subscription_topic: str, event_topic: str) -> bool:
        """Check if a subscription topic pattern matches an event topic."""
        if subscription_topic == "**":
            return True
        sub_parts = subscription_topic.split(".")
        evt_parts = event_topic.split(".")
        return self._match_parts(sub_parts, evt_parts)

    def _match_parts(self, sub: list[str], evt: list[str]) -> bool:
        if not sub:
            return not evt
        if sub[0] == "**":
            for i in range(len(evt) + 1):
                if self._match_parts(sub[1:], evt[i:]):
                    return True
            return False
        if sub[0] == "*":
            if evt:
                return self._match_parts(sub[1:], evt[1:])
            return False
        if evt and sub[0] == evt[0]:
            return self._match_parts(sub[1:], evt[1:])
        return False

    def _get_matching_subs(self, topic: str) -> list[Subscription]:
        """Collect all subscriptions whose pattern matches the topic."""
        result: list[Subscription] = []
        for pattern, subs in self._subs.items():
            if self._match_topic(pattern, topic):
                result.extend(subs)
        result.sort(key=lambda s: s.priority, reverse=True)
        return result

    async def publish(self, topic: str, event: Event | None = None, **payload: Any) -> None:
        """Publish an event to all matching subscribers.

        Handles event chaining with cycle detection and depth limiting.
        """
        if event is None:
            event = Event(topic=topic, payload=payload)
        else:
            event.topic = topic

        self._publish_depth += 1
        event_key = f"{topic}:{id(event)}"

        if event_key in self._chain_seen:
            log.warning("Event chain cycle detected for %s — skipping", topic)
            self._publish_depth -= 1
            return

        if self._publish_depth > self._max_depth:
            log.warning("Event chain depth exceeded (%d) for topic %s",
                        self._publish_depth, topic)
            self._publish_depth -= 1
            return

        self._chain_seen.add(event_key)
        subs = self._get_matching_subs(topic)

        t0 = time.perf_counter()
        errors: list[str] = []

        for sub in subs:
            try:
                await sub.handler(event)
            except Exception as exc:
                errors.append(f"{sub.topic}:{sub.tag or 'anon'}:{exc}")
                log.exception("Handler %s failed for %s", sub.topic, topic)

        # Clean up one-shot handlers
        for sub in subs:
            if sub.once and sub in self._subs.get(sub.topic, []):
                self._subs[sub.topic].remove(sub)

        duration_ms = (time.perf_counter() - t0) * 1000

        if self._enable_history:
            self._history.append(HistoryEntry(
                event=event, handler_count=len(subs),
                duration_ms=round(duration_ms, 2), errors=errors,
            ))
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        self._chain_seen.discard(event_key)
        self._publish_depth -= 1

    def history(self) -> list[HistoryEntry]:
        """Return the recent event history (most recent last)."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def subscriber_count(self, topic: str | None = None) -> int:
        """Count active subscriptions, optionally for a specific topic."""
        if topic:
            return len(self._subs.get(topic, []))
        return sum(len(v) for v in self._subs.values())

    def topics(self) -> list[str]:
        """Return all topics with active subscriptions."""
        return sorted(self._subs.keys())
