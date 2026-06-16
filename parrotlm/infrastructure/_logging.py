"""Structured logging utilities for the orchestration pipeline."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, Dict

logger = logging.getLogger(__name__)


def parse_message_event(message: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a log message string into an event name and a context dictionary.

    Args:
        message: The raw log message string, potentially containing JSON context separated by ' | '.

    Returns:
        A tuple containing the event name string and the parsed context dictionary.
    """
    if " | " not in message:
        return message, {}

    parts = message.split(" | ", 1)
    event = parts[0]
    try:
        context = json.loads(parts[1]) if len(parts) > 1 else {}
        return event, context
    except (TypeError, ValueError):
        # We avoid logging an error here because this function is called by formatters.
        # Logging back to the same system would cause infinite recursion.
        return event, {"message": parts[1]} if len(parts) > 1 else {}


def extract_record_attributes(
    record: logging.LogRecord, context: Dict[str, Any]
) -> None:
    """Extract custom attributes from a log record into the context dictionary.

    Args:
        record: The standard library log record.
        context: The dictionary to populate with custom attributes.
    """
    ignored_keys = {
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
        if key not in ignored_keys:
            context[key] = value


def _extract_event_and_context(record: logging.LogRecord) -> Tuple[str, Dict[str, Any]]:
    """Extract event and context attributes from a log record.

    Args:
        record: The log record to process.

    Returns:
        A tuple containing the event string and the context dictionary.
    """
    message = record.getMessage()
    event, context = parse_message_event(message)
    extract_record_attributes(record, context)
    return event, context


class HumanReadableFormatter(logging.Formatter):
    """Console formatter with explicit event and context fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record into a human-readable string.

        Args:
            record: The log record to format.

        Returns:
            The formatted string.
        """
        event, context = _extract_event_and_context(record)
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        if context:
            context_string = json.dumps(context, sort_keys=True, default=str)
        else:
            context_string = "{}"
        return f"{timestamp} | {record.levelname} | {event} | {context_string}"


class JsonLineFormatter(logging.Formatter):
    """JSONL formatter (one JSON object per line)."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record into a JSON string.

        Args:
            record: The log record to format.

        Returns:
            The JSON formatted string.
        """
        event, context = _extract_event_and_context(record)
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "event": event,
            **context,
        }
        return json.dumps(payload, sort_keys=True, default=str)


def ensure_log_directory_exists() -> None:
    """Create the logs directory if it does not already exist.

    This ensures that file-based log handlers have a valid destination to write to.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)


def apply_logging_configuration(level: int) -> None:
    """Apply the standard logging configuration using dictionary config.

    Args:
        level: The logging level threshold (e.g., logging.INFO).
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "human": {
                    "()": "parrotlm.infrastructure._logging.HumanReadableFormatter"
                },
                "jsonl": {"()": "parrotlm.infrastructure._logging.JsonLineFormatter"},
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


def setup_logging(level: int = logging.INFO) -> None:
    """Configure stdout and rotating JSONL file handlers.

    Args:
        level: The logging severity level to capture. Defaults to logging.INFO.
    """
    ensure_log_directory_exists()
    apply_logging_configuration(level)


def log_structured(level: int, event: str, **context: Any) -> None:
    """Log one event with machine-readable context for easier debugging.

    Args:
        level: The logging severity level (e.g., logging.INFO).
        event: The name of the event being logged.
        context: Additional key-value pairs representing the context of the event.
    """
    try:
        context_json = json.dumps(context, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # If serialization fails, we fall back to a string representation.
        # We avoid logging an error here to prevent potential recursion or
        # cluttering the logs with serialization warnings.
        context_json = str(context)
    logger.log(level, "%s | %s", event, context_json)


def is_retryable_exception(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry attempt.

    Args:
        exception: The exception that was raised.

    Returns:
        True if the exception is retryable, False otherwise.
    """
    # Type/Value errors usually indicate bad caller input and will not succeed
    # on retry, so we filter them out and do not attempt them again.
    return not isinstance(exception, (TypeError, ValueError))
