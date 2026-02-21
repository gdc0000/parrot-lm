"""Structured logging utilities for the orchestration pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


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
