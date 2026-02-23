"""Structured logging utilities for the orchestration pipeline."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_event_and_context(record: logging.LogRecord) -> tuple[str, dict[str, Any]]:
    """Extract event/context from a log record while preserving extra fields."""
    message = record.getMessage()
    event = message
    context: dict[str, Any] = {}

    if " | " in message:
        candidate_event, candidate_context = message.split(" | ", 1)
        event = candidate_event
        try:
            parsed = json.loads(candidate_context)
            if isinstance(parsed, dict):
                context = parsed
            else:
                context = {"context": parsed}
        except (TypeError, ValueError):
            context = {"context": candidate_context}

    standard_fields = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }
    for key, value in record.__dict__.items():
        if key not in standard_fields:
            context[key] = value

    return event, context


class HumanReadableFormatter(logging.Formatter):
    """Console formatter with explicit event and context fields."""

    def format(self, record: logging.LogRecord) -> str:
        event, context = _extract_event_and_context(record)
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        if context:
            context_str = json.dumps(context, sort_keys=True, default=str)
        else:
            context_str = "{}"
        return f"{timestamp} | {record.levelname} | {event} | {context_str}"


class JsonLineFormatter(logging.Formatter):
    """JSONL formatter (one JSON object per line)."""

    def format(self, record: logging.LogRecord) -> str:
        event, context = _extract_event_and_context(record)
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "event": event,
            **context,
        }
        return json.dumps(payload, sort_keys=True, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure stdout + rotating JSONL file handlers via dictConfig."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "human": {"()": "parrotlm._logging.HumanReadableFormatter"},
                "jsonl": {"()": "parrotlm._logging.JsonLineFormatter"},
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "human",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": level,
                    "formatter": "jsonl",
                    "filename": "logs/parrotlm.log",
                    "maxBytes": 5 * 1024 * 1024,
                    "backupCount": 3,
                    "encoding": "utf-8",
                },
            },
            "root": {"level": level, "handlers": ["stdout", "file"]},
        }
    )


def log_structured(level: int, event: str, **context: Any) -> None:
    """Log one event with machine-readable context for easier debugging."""
    try:
        context_json = json.dumps(context, sort_keys=True, default=str)
    except (TypeError, ValueError):
        context_json = str(context)
    logger.log(level, "%s | %s", event, context_json)


def is_retryable_exception(exception: BaseException) -> bool:
    """Retry transient failures, but not local validation errors.

    Type/Value errors usually indicate bad caller input and will not succeed
    on retry, so we do not attempt them again.
    """
    return not isinstance(exception, (TypeError, ValueError))
