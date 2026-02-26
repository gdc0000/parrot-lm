"""Upload simulation session logs to a Supabase ``session_logs`` table."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from parrotlm.infrastructure.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

TABLE_NAME = "session_logs"

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
            f"upload_skipped | reason=supabase_client_unavailable | hint={error_message}"
        )
        return False, None, error_message
    return True, client, ""


def sanitize_log_entries(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean log entries to ensure they match the database schema.

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


def execute_batch_insert(
    client: Any, cleaned_log_entries: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Execute the batch insert operation against the Supabase database.

    Args:
        client: The Supabase client to use for the insert.
        cleaned_log_entries: The sanitized list of log dictionaries.

    Returns:
        A tuple containing a boolean success flag and a status message.
    """
    try:
        response = client.table(TABLE_NAME).insert(cleaned_log_entries).execute()
        inserted_count = len(response.data) if response.data else 0
        logger.info(
            "upload_success | table=%s rows_inserted=%s",
            TABLE_NAME,
            inserted_count,
        )
        return True, f"Successfully inserted {inserted_count} rows."

    except Exception as exception:
        error_message = str(exception)
        logger.exception(
            "upload_failed | table=%s rows_attempted=%s | error_type=%s | error_message=%s",
            TABLE_NAME,
            len(cleaned_log_entries),
            type(exception).__name__,
            error_message,
        )
        return False, error_message


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


class SupabaseBufferedLogger:
    """A memory-efficient logger that uploads logs in batches.

    This class provides a simple way to stream logs to Supabase without
    keeping all of them in memory. It's designed for long-running
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
        """Add a log entry to the buffer and upload if the batch size is reached.

        Args:
            log_entry: The structured log dictionary to record.
        """
        if not self.is_available:
            return

        self.buffer.append(log_entry)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Upload all currently buffered logs to Supabase."""
        if not self.buffer or not self.is_available:
            return

        cleaned_entries = sanitize_log_entries(self.buffer)
        execute_batch_insert(self.client, cleaned_entries)
        self.buffer = []

