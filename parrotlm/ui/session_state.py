"""Session-state and persistence helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

LOCAL_STORAGE_LOG_KEY = "parrot_lm_logs"


def initialize_session_state(local_storage: Any) -> None:
    """Initialize all required Streamlit session-state keys."""
    if "last_generated_config" not in st.session_state:
        st.session_state["last_generated_config"] = {}

    if "all_logs" in st.session_state:
        return

    saved_logs = local_storage.getItem(LOCAL_STORAGE_LOG_KEY)
    if not saved_logs:
        st.session_state["all_logs"] = pd.DataFrame()
        return

    try:
        st.session_state["all_logs"] = pd.DataFrame(saved_logs)
    except (TypeError, ValueError):
        st.session_state["all_logs"] = pd.DataFrame()


def clear_local_data(local_storage: Any) -> None:
    """Remove persisted logs from browser storage and reset in-memory dataframe."""
    if hasattr(local_storage, "eraseItem"):
        local_storage.eraseItem(LOCAL_STORAGE_LOG_KEY, default=None)
    else:
        try:
            local_storage.deleteItem(LOCAL_STORAGE_LOG_KEY)
        except KeyError:
            pass

    st.session_state["all_logs"] = pd.DataFrame()


def append_and_persist_logs(local_storage: Any, new_logs_df: pd.DataFrame) -> None:
    """Append new logs to session state and sync them to local storage."""
    current_df = st.session_state["all_logs"]
    if current_df.empty:
        updated_df = new_logs_df
    else:
        updated_df = pd.concat([current_df, new_logs_df], ignore_index=True)

    st.session_state["all_logs"] = updated_df
    local_storage.setItem(LOCAL_STORAGE_LOG_KEY, updated_df.to_dict("records"))

