"""Upload simulation data and application logs to Supabase tables."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from parrotlm.infrastructure._logging import (
    extract_record_attributes,
    parse_message_event,
)
from parrotlm.infrastructure.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

SESSION_LOGS_TABLE_NAME = "session_logs"
APPLICATION_LOGS_TABLE_NAME = "application_logs"
TABLE_NAME = SESSION_LOGS_TABLE_NAME

_ALLOWED_COLUMNS = frozenset(
    {
        "experiment_id",
        "turn_id",
        "scenario",
        "speaker_model",
        "responder_model",
        "timestamp",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "content",
        "finish_reason",
        "is_refusal",
        "system_prompt_snapshot",
    }
)

_ALLOWED_APPLICATION_LOG_COLUMNS = frozenset(
    {
        "timestamp",
        "level",
        "logger_name",
        "module",
        "function_name",
        "line_number",
        "event",
        "message",
        "context",
        "exception",
        "process_id",
        "thread_name",
    }
)


def verify_client_availability() -> Tuple[bool, Any, str]:
    """Check if the Supabase client is properly configured and available.

    Returns:
        A tuple containing a boolean indicating availability, the client object
        (if available), and an error message string (if not available).
    """
    client = get_supabase_client()
    if client is None:
        error_message = "Supabase client unavailable (check .env)."
        logger.warning(
            "upload_skipped | reason=supabase_client_unavailable | hint=%s",
            error_message,
        )
        return False, None, error_message
    return True, client, ""


def sanitize_log_entries(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean session log entries to ensure they match the database schema.

    Args:
        logs: The raw log entries to clean.

    Returns:
        A list of cleaned log entries containing only allowed columns.
    """
    cleaned_log_entries = []
    # We must strip keys that do not exist in the Supabase table because PostgreSQL
    # enforces strict schema validation. Sending extra keys (like an auto-generated 'id'
    # or temporary 'input_preview' field) will cause the entire batch insert to fail.
    for entry in logs:
        cleaned_entry = {
            key: value for key, value in entry.items() if key in _ALLOWED_COLUMNS
        }
        cleaned_log_entries.append(cleaned_entry)
    return cleaned_log_entries


def sanitize_application_log_entries(
    logs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Clean application log entries to ensure they match the database schema."""
    return [
        {
            key: value
            for key, value in entry.items()
            if key in _ALLOWED_APPLICATION_LOG_COLUMNS
        }
        for entry in logs
    ]


def execute_table_insert(
    client: Any, table_name: str, rows: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Execute a batch insert operation against the Supabase database.

    Args:
        client: The Supabase client to use for the insert.
        table_name: The table to insert into.
        rows: The sanitized list of row dictionaries.

    Returns:
        A tuple containing a boolean success flag and a status message.
    """
    try:
        response = client.table(table_name).insert(rows).execute()
        inserted_count = len(response.data) if response.data else 0
        logger.info(
            "upload_success | table=%s rows_inserted=%s",
            table_name,
            inserted_count,
        )
        return True, f"Successfully inserted {inserted_count} rows."

    except Exception as exception:
        error_message = str(exception)
        logger.exception(
            "upload_failed | table=%s rows_attempted=%s | "
            "error_type=%s | error_message=%s",
            table_name,
            len(rows),
            type(exception).__name__,
            error_message,
        )
        return False, error_message


def execute_batch_insert(
    client: Any, cleaned_log_entries: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Execute the session log batch insert operation against Supabase."""
    return execute_table_insert(client, SESSION_LOGS_TABLE_NAME, cleaned_log_entries)


def upload_session_logs(logs: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Insert a batch of simulation logs into the Supabase database.

    Args:
        logs: A list of dictionaries representing the conversation logs.

    Returns:
        A tuple containing a boolean success flag and a status or error message.
    """
    if not logs:
        logger.info("upload_skipped | reason=empty_logs")
        return True, "No logs to upload."

    is_available, client, error_message = verify_client_availability()
    if not is_available:
        return False, error_message

    cleaned_log_entries = sanitize_log_entries(logs)
    return execute_batch_insert(client, cleaned_log_entries)


def make_json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation for Supabase JSONB columns."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def format_log_record_for_supabase(record: logging.LogRecord) -> Dict[str, Any]:
    """Convert a standard library log record into an application log row."""
    event, context = parse_message_event(record.getMessage())
    extract_record_attributes(record, context)
    safe_context = make_json_safe(context)

    exception_text = None
    if record.exc_info:
        exception_text = logging.Formatter().formatException(record.exc_info)

    return {
        "timestamp": datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).isoformat(),
        "level": record.levelname,
        "logger_name": record.name,
        "module": record.module,
        "function_name": record.funcName,
        "line_number": record.lineno,
        "event": event,
        "message": record.getMessage(),
        "context": safe_context,
        "exception": exception_text,
        "process_id": record.process,
        "thread_name": record.threadName,
    }


class SupabaseLogHandler(logging.Handler):
    """Logging handler that buffers application log records into Supabase."""

    def __init__(
        self,
        batch_size: int = 10,
        client: Optional[Any] = None,
        table_name: str = APPLICATION_LOGS_TABLE_NAME,
    ) -> None:
        """Initialize a Supabase-backed logging handler."""
        super().__init__()
        self.batch_size = max(1, batch_size)
        self.client = client if client is not None else get_supabase_client()
        self.table_name = table_name
        self.buffer: List[Dict[str, Any]] = []
        self._is_flushing = False

    @property
    def is_available(self) -> bool:
        """Return whether the handler has a usable Supabase client."""
        return self.client is not None

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer one log record and upload when the batch size is reached."""
        if not self.is_available or self._is_flushing:
            return
        if record.name.startswith("parrotlm.infrastructure.supabase"):
            return

        try:
            self.buffer.append(format_log_record_for_supabase(record))
            if len(self.buffer) >= self.batch_size:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Upload buffered application logs to Supabase."""
        if not self.buffer or not self.is_available or self._is_flushing:
            return

        self._is_flushing = True
        try:
            rows = sanitize_application_log_entries(self.buffer)
            execute_table_insert(self.client, self.table_name, rows)
            self.buffer = []
        finally:
            self._is_flushing = False


def install_supabase_log_handler(
    batch_size: int = 10,
    client: Optional[Any] = None,
    level: int = logging.INFO,
) -> Optional[SupabaseLogHandler]:
    """Attach one Supabase application-log handler to the root logger."""
    effective_client = client if client is not None else get_supabase_client()
    if effective_client is None:
        logger.warning(
            "supabase_application_logging_skipped | reason=client_unavailable"
        )
        return None

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, SupabaseLogHandler):
            return handler

    handler = SupabaseLogHandler(batch_size=batch_size, client=effective_client)
    handler.setLevel(level)
    root_logger.addHandler(handler)
    logger.info("supabase_application_logging_enabled | table=%s", handler.table_name)
    return handler


def flush_supabase_log_handlers() -> None:
    """Flush all installed Supabase application-log handlers."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, SupabaseLogHandler):
            handler.flush()


class SupabaseBufferedLogger:
    """A memory-efficient logger that uploads generated session rows in batches.

    This class provides a simple way to stream generated conversation data to
    Supabase without keeping all of it in memory. It's designed for long-running
    simulations where memory safety is a priority.
    """

    def __init__(self, batch_size: int = 10) -> None:
        """Initialize the buffered logger.

        Args:
            batch_size: Number of logs to accumulate before uploading.
        """
        self.batch_size = max(1, batch_size)
        self.buffer: List[Dict[str, Any]] = []
        self.is_available, self.client, self.error_message = (
            verify_client_availability()
        )

    def push(self, log_entry: Dict[str, Any]) -> None:
        """Add a generated row to the buffer and upload at batch size."""
        if not self.is_available:
            return

        self.buffer.append(log_entry)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Upload all currently buffered generated rows to Supabase."""
        if not self.buffer or not self.is_available:
            return

        cleaned_entries = sanitize_log_entries(self.buffer)
        execute_batch_insert(self.client, cleaned_entries)
        self.buffer = []
