"""Upload simulation session logs to a Supabase ``session_logs`` table."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from parrotlm.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

TABLE_NAME = "session_logs"


def upload_session_logs(logs: List[Dict[str, Any]]) -> bool:
    """Insert *logs* into the Supabase ``session_logs`` table.

    Returns ``True`` on success and ``False`` on any failure (missing client,
    network error, API error).  Failures are logged but never raised so the
    caller's workflow is not interrupted.
    """
    if not logs:
        logger.info("upload_skipped | reason=empty_logs")
        return True

    client = get_supabase_client()
    if client is None:
        logger.warning(
            "upload_skipped | reason=supabase_client_unavailable "
            "| hint=check SUPABASE_URL and SUPABASE_ANON_KEY in .env"
        )
        return False

    # Strip keys that Supabase would reject (e.g. the auto-generated `id`).
    cleaned = [_clean_log_entry(entry) for entry in logs]

    try:
        response = client.table(TABLE_NAME).insert(cleaned).execute()
        inserted_count = len(response.data) if response.data else 0
        logger.info(
            "upload_success | table=%s rows_inserted=%s",
            TABLE_NAME,
            inserted_count,
        )
        return True
    except Exception:
        logger.exception(
            "upload_failed | table=%s rows_attempted=%s",
            TABLE_NAME,
            len(cleaned),
        )
        return False


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


def _clean_log_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only columns that exist in the ``session_logs`` table."""
    return {k: v for k, v in entry.items() if k in _ALLOWED_COLUMNS}
