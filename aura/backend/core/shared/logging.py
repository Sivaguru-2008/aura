"""Structured logging with request correlation.

A routing decision is only auditable if you can follow one upload across intake,
detection, dispatch, and the engine that ran it. Every log line this module emits
carries the same correlation id for a given request, so ``grep <id>`` reconstructs
the whole path.

Configuration is environment-only (the deployment is offline; there is no logging
service to configure against):

* ``AURA_LOG_LEVEL``  — ``DEBUG`` | ``INFO`` (default) | ``WARNING`` | ``ERROR``
* ``AURA_LOG_JSON=1`` — emit one JSON object per line instead of the human format

Nothing here is global-state hostile: handlers are attached once to the
``aura.backend`` logger, and ``propagate`` is left off so we never double-print
through a host application's root handler.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Iterator

_ROOT_LOGGER_NAME = "aura.backend"

#: Correlation id for the in-flight request. Empty outside a request.
_correlation_id: ContextVar[str] = ContextVar("aura_correlation_id", default="")

_configured = False


def new_correlation_id() -> str:
    """Mint a short, collision-safe id for one request."""
    return uuid.uuid4().hex[:12]


def correlation_id() -> str:
    """Correlation id of the in-flight request, or ``""`` outside one."""
    return _correlation_id.get()


@contextlib.contextmanager
def use_correlation_id(value: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of the block.

    Context-var based, so it is safe under asyncio concurrency: each task gets its
    own copy of the context and cannot observe another request's id.
    """
    cid = value or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class _CorrelationFilter(logging.Filter):
    """Attach the current correlation id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id() or "-"
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line — for shipping into a log store."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload["context"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    """Human format that matches the gateway's existing ``[component] message`` style."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "-")
        base = f"[{record.name}][{cid}] {record.getMessage()}"
        extra = getattr(record, "context", None)
        if isinstance(extra, dict) and extra:
            base += " " + " ".join(f"{k}={v!r}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(os.environ.get("AURA_LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter() if os.environ.get("AURA_LOG_JSON") == "1" else _TextFormatter()
    )
    handler.addFilter(_CorrelationFilter())
    logger.addHandler(handler)
    # Don't hand records to the root logger as well — uvicorn installs its own
    # handlers there and every line would print twice.
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger for a backend component.

    ``name`` is the component's short name (``"router"``, ``"engine.thorax"``); it
    is namespaced under ``aura.backend`` so a deployment can raise or lower the
    level for the whole routing layer with one setting.
    """
    _configure_once()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def log_context(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Log ``message`` with structured key/value context.

    Convenience over ``logger.info(msg, extra={"context": {...}})`` — the ``extra``
    key must be exactly ``context`` for both formatters to pick it up, so wrapping
    it keeps that coupling in one place.
    """
    logger.log(level, message, extra={"context": context})
