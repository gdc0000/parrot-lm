"""Chatbot setup tab and simulation execution flow."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from parrotlm.orchestrator import Orchestrator
from parrotlm.prompt_utils import construct_system_prompt

from .session_state import append_and_persist_logs
from .sidebar import TechnicalSettings

DEFAULT_MODEL_SLUG = "google/gemini-2.5-flash-lite"
DEFAULT_INITIAL_MESSAGE = (
    "I'd like to align on the objectives and understand your current priorities."
)
MAX_SPEAKER_LABEL_LENGTH = 50


def _build_chatbot_config(
    model_slug: str,
    system_prompt: str,
    persona: str,
    context_window: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build orchestrator agent configuration from UI input fields."""
    return {
        "model": model_slug,
        "system_prompt": system_prompt,
        "user_persona_snapshot": persona,
        "max_history_turns": context_window,
        "params": {"temperature": temperature, "max_tokens": max_tokens},
    }


def _build_scenario_name(persona_a: str, persona_b: str) -> str:
    """Build a short scenario label from both personas."""
    return f"{persona_a[:15]} vs {persona_b[:15]}"


def render_chatbot_setup_tab(settings: TechnicalSettings, local_storage: Any) -> None:
    """Render chatbot setup controls and execute the conversation simulation."""
    st.markdown("### Configuration")

    chatbot_a_column, chatbot_b_column = st.columns(2)
    with chatbot_a_column:
        st.markdown("#### Chatbot A")
        model_a_slug = st.text_input("Model A Slug", DEFAULT_MODEL_SLUG, key="model_a")
        persona_a = st.text_area("Persona", "Chief Technology Officer", height=100)

    with chatbot_b_column:
        st.markdown("#### Chatbot B")
        model_b_slug = st.text_input("Model B Slug", DEFAULT_MODEL_SLUG, key="model_b")
        persona_b = st.text_area("Persona", "Financial Analyst", height=100)

    st.markdown("---")
    initial_message = st.text_input("The conversation starts with:", DEFAULT_INITIAL_MESSAGE)

    if not st.button("Start Conversation", type="primary", width="stretch"):
        return

    st.write("### Live Conversation")
    chat_container = st.container()

    simulation_logs = _run_and_render_simulation(
        settings=settings,
        model_a_slug=model_a_slug,
        model_b_slug=model_b_slug,
        persona_a=persona_a,
        persona_b=persona_b,
        initial_message=initial_message,
        chat_container=chat_container,
    )
    if simulation_logs is None:
        return

    append_and_persist_logs(local_storage, pd.DataFrame(simulation_logs))
    st.success("Simulation finished and persisted locally.")


def _create_orchestrator(
    settings: TechnicalSettings,
    model_a_slug: str,
    model_b_slug: str,
    persona_a: str,
    persona_b: str,
) -> Orchestrator:
    """Build the orchestrator from current UI settings."""
    system_prompt_a = construct_system_prompt(persona_a)
    system_prompt_b = construct_system_prompt(persona_b)

    chatbot_a_config = _build_chatbot_config(
        model_slug=model_a_slug,
        system_prompt=system_prompt_a,
        persona=persona_a,
        context_window=settings.context_window,
        temperature=settings.temp_a,
        max_tokens=settings.max_tokens,
    )
    chatbot_b_config = _build_chatbot_config(
        model_slug=model_b_slug,
        system_prompt=system_prompt_b,
        persona=persona_b,
        context_window=settings.context_window,
        temperature=settings.temp_b,
        max_tokens=settings.max_tokens,
    )

    return Orchestrator(
        agent_a_config=chatbot_a_config,
        agent_b_config=chatbot_b_config,
        scenario_name=_build_scenario_name(persona_a, persona_b),
    )


def _run_and_render_simulation(
    settings: TechnicalSettings,
    model_a_slug: str,
    model_b_slug: str,
    persona_a: str,
    persona_b: str,
    initial_message: str,
    chat_container: Any,
) -> List[Dict[str, Any]] | None:
    """Run the simulation, stream messages to the UI, and return logs."""
    total_output_tokens = 0

    try:
        orchestrator = _create_orchestrator(
            settings=settings,
            model_a_slug=model_a_slug,
            model_b_slug=model_b_slug,
            persona_a=persona_a,
            persona_b=persona_b,
        )

        with st.spinner("Agents are conversing..."):
            for log_entry in orchestrator.run_simulation(
                settings.num_turns,
                initial_message=initial_message,
            ):
                total_output_tokens += log_entry["output_tokens"]
                _render_chat_message(log_entry, model_a_slug, persona_a, persona_b, chat_container)
                time.sleep(0.1)
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        st.error(f"Simulation error: {error}")
        st.info("Tip: try increasing 'Max Tokens' if the API fails with low values.")
        return None

    st.success(f"Simulation complete. Total tokens: {total_output_tokens}")
    return orchestrator.logs


def _render_chat_message(
    log_entry: Dict[str, Any],
    model_a_slug: str,
    persona_a: str,
    persona_b: str,
    chat_container: Any,
) -> None:
    """Render one message bubble and its metadata row."""
    is_agent_a = log_entry["speaker_model"] == model_a_slug
    speaker_label = persona_a if is_agent_a else persona_b

    if len(speaker_label) > MAX_SPEAKER_LABEL_LENGTH:
        speaker_label = f"{speaker_label[:47]}..."

    with chat_container:
        with st.chat_message(name=speaker_label):
            st.write(log_entry["content"])

        st.markdown(
            f"<div style='text-align: right; margin-top: -15px; margin-bottom: 10px;'>"
            f"<span style='color: gray; font-size: 0.8rem;'>"
            f"Latency {log_entry['latency_ms']:.0f}ms | Tokens {log_entry['output_tokens']} | "
            f"Model {log_entry['speaker_model']}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
