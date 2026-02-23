"""Sidebar rendering for technical simulation settings."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class TechnicalSettings:
    """Runtime settings used by simulation and analysis tabs."""

    num_turns: int
    temp_a: float
    temp_b: float
    max_tokens: int
    context_window: int


def _apply_api_key_if_present(api_key: str) -> None:
    """Persist API key to session state only when user provides one."""
    if api_key:
        st.session_state["openrouter_api_key"] = api_key

def render_sidebar(default_turns: int) -> tuple[TechnicalSettings, bool]:
    """Render sidebar inputs and return user-selected settings."""
    st.sidebar.header("Technical Settings")

    api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
    _apply_api_key_if_present(api_key)

    clear_requested = st.sidebar.button(
        "Clear Local Data",
        help="Wipes all conversation history from your browser storage.",
    )

    num_turns = st.sidebar.slider(
        "Turns per Chatbot",
        1,
        100,
        default_turns,
        help="The number of times each chatbot will speak. Total messages = Turns * 2.",
    )

    st.sidebar.markdown("### Model Parameters")
    temp_a = st.sidebar.slider(
        "Chatbot A Temperature",
        0.0,
        2.0,
        1.0,
        0.1,
        help=(
            "Controls creativity. Higher values make replies more unpredictable, "
            "lower values make them more focused."
        ),
    )
    temp_b = st.sidebar.slider("Chatbot B Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        100,
        4000,
        1000,
        help="Maximum length of a message. Increase this if responses are cut off.",
    )
    context_window = st.sidebar.slider(
        "Context Window (Turns)",
        1,
        50,
        20,
        help=(
            "Controls how many previous turns each chatbot remembers. "
            "Lower values reduce context, higher values keep more history."
        ),
    )

    settings = TechnicalSettings(
        num_turns=num_turns,
        temp_a=temp_a,
        temp_b=temp_b,
        max_tokens=max_tokens,
        context_window=context_window,
    )
    return settings, clear_requested
