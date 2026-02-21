"""Session-state and persistence helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

LOCAL_STORAGE_LOG_KEY = "parrot_lm_logs"


def _empty_logs_dataframe() -> pd.DataFrame:
    """Return the canonical empty logs dataframe."""
    return pd.DataFrame()


def _load_persisted_logs(local_storage: Any) -> pd.DataFrame:
    """Load saved logs from local storage, returning an empty dataframe on failures."""
    saved_logs = local_storage.getItem(LOCAL_STORAGE_LOG_KEY)
    if not saved_logs:
        return _empty_logs_dataframe()

    try:
        return pd.DataFrame(saved_logs)
    except (TypeError, ValueError):
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
    if hasattr(local_storage, "eraseItem"):
        # Some streamlit-local-storage versions expose eraseItem instead of deleteItem.
        local_storage.eraseItem(LOCAL_STORAGE_LOG_KEY, default=None)
        return

    try:
        local_storage.deleteItem(LOCAL_STORAGE_LOG_KEY)
    except KeyError:
        pass


def clear_local_data(local_storage: Any) -> None:
    """Remove persisted logs from browser storage and reset in-memory dataframe."""
    _delete_local_storage_logs(local_storage)
    st.session_state["all_logs"] = _empty_logs_dataframe()


def _merge_logs(current_df: pd.DataFrame, new_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and new log rows into one dataframe."""
    if current_df.empty:
        return new_logs_df
    # Reindex to keep a clean contiguous table for plotting and CSV export.
    return pd.concat([current_df, new_logs_df], ignore_index=True)


def _persist_logs_to_local_storage(local_storage: Any, logs_df: pd.DataFrame) -> None:
    """Write the current log dataframe to local storage."""
    local_storage.setItem(LOCAL_STORAGE_LOG_KEY, logs_df.to_dict("records"))


def append_and_persist_logs(local_storage: Any, new_logs_df: pd.DataFrame) -> None:
    """Append new logs to session state and sync them to local storage."""
    current_df = st.session_state["all_logs"]
    updated_df = _merge_logs(current_df, new_logs_df)
    st.session_state["all_logs"] = updated_df
    _persist_logs_to_local_storage(local_storage, updated_df)
