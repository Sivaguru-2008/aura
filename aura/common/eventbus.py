"""In-process async event bus.

The production design uses Redis Streams / NATS; this is the same publish/subscribe
contract behind an in-memory implementation so the P0 stack runs from one process.
Swapping to Redis means replacing this file only — subscribers are unchanged.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Any


@dataclass
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    async def publish(self, topic: str, **payload: Any) -> None:
        event = Event(topic=topic, payload=payload)
        # Fan out; run handlers concurrently, surface the first failure.
        await asyncio.gather(*(h(event) for h in self._subs.get(topic, [])))


# Canonical topic names — mirror docs/ARCHITECTURE.md section 13.
STUDY_RECEIVED = "study.received"
VISION_COMPLETED = "vision.completed"
FUSION_COMPLETED = "fusion.completed"
CASE_READY = "case.ready"
FEEDBACK_RECORDED = "feedback.recorded"
