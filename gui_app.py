"""Streamlit entrypoint for ParrotLM."""

from __future__ import annotations

import streamlit as st
from streamlit_local_storage import LocalStorage

from parrotlm.simulation_config import NUM_TURNS
from parrotlm.ui.analysis_tabs import render_basic_analysis_tab, render_stylometric_analysis_tab
from parrotlm.ui.chat_setup_tab import render_chatbot_setup_tab
from parrotlm.ui.session_state import clear_local_data, initialize_session_state
from parrotlm.ui.sidebar import render_sidebar


def main() -> None:
    """Render the full Streamlit app."""
    local_storage = LocalStorage()

    st.set_page_config(page_title="ParrotLM", layout="wide")
    st.title("ParrotLM")
    st.markdown(
        "A customizable Python framework for simulating conversations "
        "between two LLM chatbots with customizable personas, interaction settings, "
        "and analysis capabilities."
    )

    initialize_session_state(local_storage)

    settings, clear_requested = render_sidebar(NUM_TURNS)
    if clear_requested:
        clear_local_data(local_storage)
        st.success("Local data cleared.")
        st.rerun()

    st.markdown("---")
    setup_tab, basic_analysis_tab, stylometric_tab = st.tabs(
        ["Chatbot Setup", "Basic Analysis", "Stylometric Analysis"]
    )

    with setup_tab:
        render_chatbot_setup_tab(settings, local_storage)
    with basic_analysis_tab:
        render_basic_analysis_tab()
    with stylometric_tab:
        render_stylometric_analysis_tab()


if __name__ == "__main__":
    main()
