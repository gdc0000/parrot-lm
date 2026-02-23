"""Session-state and persistence helpers for the Streamlit UI."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

LOCAL_STORAGE_LOG_KEY = "parrot_lm_logs"
logger = logging.getLogger(__name__)


def _empty_logs_dataframe() -> pd.DataFrame:
    """Return the canonical empty logs dataframe."""
    return pd.DataFrame()


def _load_persisted_logs(local_storage: Any) -> pd.DataFrame:
    """Load saved logs from local storage, returning an empty dataframe on failures."""
    try:
        saved_logs = local_storage.getItem(LOCAL_STORAGE_LOG_KEY)
    except (AttributeError, TypeError, ValueError):
        logger.exception(
            "local_storage_read_failed | key=%s",
            LOCAL_STORAGE_LOG_KEY,
        )
        return _empty_logs_dataframe()

    if not saved_logs:
        return _empty_logs_dataframe()

    try:
        return pd.DataFrame(saved_logs)
    except (TypeError, ValueError):
        logger.warning(
            "malformed_local_storage_logs_ignored | key=%s value_type=%s",
            LOCAL_STORAGE_LOG_KEY,
            type(saved_logs).__name__,
        )
        return _empty_logs_dataframe()


def initialize_session_state(local_storage: Any) -> None:
    """Initialize all required Streamlit session-state keys."""
    if "last_generated_config" not in st.session_state:
        st.session_state["last_generated_config"] = {}

    if "all_logs" in st.session_state:
        return

    st.session_state["all_logs"] = _load_persisted_logs(local_storage)


def _delete_local_storage_logs(local_storage: Any) -> None:
    """Delete logs from browser storage across supported local-storage adapters."""
    # Aggressive JS clear as the primary reliable method
    components.html(
        f"""
        <script>
            localStorage.removeItem("{LOCAL_STORAGE_LOG_KEY}");
        </script>
        """,
        height=0,
    )

    if hasattr(local_storage, "eraseItem"):
        # Some streamlit-local-storage versions expose eraseItem instead of deleteItem.
        try:
            local_storage.eraseItem(LOCAL_STORAGE_LOG_KEY, default=None)
        except (AttributeError, TypeError, ValueError):
            logger.exception(
                "local_storage_erase_failed | key=%s",
                LOCAL_STORAGE_LOG_KEY,
            )
        return

    try:
        local_storage.deleteItem(LOCAL_STORAGE_LOG_KEY)
    except KeyError:
        logger.info(
            "local_storage_key_missing_on_delete | key=%s",
            LOCAL_STORAGE_LOG_KEY,
        )
    except (AttributeError, TypeError, ValueError):
        logger.exception(
            "local_storage_delete_failed | key=%s",
            LOCAL_STORAGE_LOG_KEY,
        )


def clear_local_data(local_storage: Any) -> None:
    """Remove persisted logs from browser storage and reset in-memory dataframe."""
    # Overwrite with empty list first to ensure data is gone even if deleteItem fails.
    try:
        local_storage.setItem(LOCAL_STORAGE_LOG_KEY, [])
    except (AttributeError, TypeError, ValueError):
        logger.warning("local_storage_overwrite_failed_on_clear | key=%s", LOCAL_STORAGE_LOG_KEY)

    _delete_local_storage_logs(local_storage)
    st.session_state["all_logs"] = _empty_logs_dataframe()


def _merge_logs(current_df: pd.DataFrame, new_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and new log rows into one dataframe."""
    if current_df.empty:
        return new_logs_df
    # Reindex to keep a clean contiguous table for plotting and CSV export.
    return pd.concat([current_df, new_logs_df], ignore_index=True)


def _persist_logs_to_local_storage(local_storage: Any, logs_df: pd.DataFrame) -> bool:
    """Write the current log dataframe to local storage."""
    serializable_df = logs_df.astype(object).where(pd.notna(logs_df), None)
    try:
        local_storage.setItem(LOCAL_STORAGE_LOG_KEY, serializable_df.to_dict("records"))
    except (AttributeError, TypeError, ValueError):
        logger.exception(
            "local_storage_write_failed | key=%s row_count=%s",
            LOCAL_STORAGE_LOG_KEY,
            len(logs_df),
        )
        return False
    return True


def append_and_persist_logs(local_storage: Any, new_logs_df: pd.DataFrame) -> None:
    """Append new logs to session state and sync them to local storage."""
    current_df = st.session_state["all_logs"]
    updated_df = _merge_logs(current_df, new_logs_df)
    st.session_state["all_logs"] = updated_df
    saved = _persist_logs_to_local_storage(local_storage, updated_df)
    if not saved:
        logger.warning(
            "local_storage_sync_skipped_after_failure | key=%s row_count=%s",
            LOCAL_STORAGE_LOG_KEY,
            len(updated_df),
        )
