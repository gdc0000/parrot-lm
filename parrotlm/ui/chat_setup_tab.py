"""Chatbot setup tab and simulation execution flow."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatSetupInputs:
    """User-entered chatbot setup values from the tab."""

    model_a_slug: str
    persona_a: str
    model_b_slug: str
    persona_b: str
    initial_message: str


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
    # Keep scenario names compact so log views remain readable across many runs.
    return f"{persona_a[:15]} vs {persona_b[:15]}"


def _render_chatbot_inputs() -> ChatSetupInputs:
    """Render setup inputs and return user-selected values."""
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

    return ChatSetupInputs(
        model_a_slug=model_a_slug,
        persona_a=persona_a,
        model_b_slug=model_b_slug,
        persona_b=persona_b,
        initial_message=initial_message,
    )


def render_chatbot_setup_tab(settings: TechnicalSettings, local_storage: Any) -> None:
    """Render chatbot setup controls and execute the conversation simulation."""
    chat_inputs = _render_chatbot_inputs()

    if not st.button("Start Conversation", type="primary", width="stretch"):
        return

    st.write("### Live Conversation")
    chat_container = st.container()

    simulation_logs = _execute_simulation(
        settings=settings,
        chat_inputs=chat_inputs,
        chat_container=chat_container,
    )
    if simulation_logs is None:
        return

    _persist_simulation_logs(local_storage, simulation_logs)


def _create_orchestrator(
    settings: TechnicalSettings,
    chat_inputs: ChatSetupInputs,
) -> Orchestrator:
    """Build the orchestrator from current UI settings."""
    system_prompt_a = construct_system_prompt(chat_inputs.persona_a)
    system_prompt_b = construct_system_prompt(chat_inputs.persona_b)

    chatbot_a_config = _build_chatbot_config(
        model_slug=chat_inputs.model_a_slug,
        system_prompt=system_prompt_a,
        persona=chat_inputs.persona_a,
        context_window=settings.context_window,
        temperature=settings.temp_a,
        max_tokens=settings.max_tokens,
    )
    chatbot_b_config = _build_chatbot_config(
        model_slug=chat_inputs.model_b_slug,
        system_prompt=system_prompt_b,
        persona=chat_inputs.persona_b,
        context_window=settings.context_window,
        temperature=settings.temp_b,
        max_tokens=settings.max_tokens,
    )

    return Orchestrator(
        agent_a_config=chatbot_a_config,
        agent_b_config=chatbot_b_config,
        scenario_name=_build_scenario_name(chat_inputs.persona_a, chat_inputs.persona_b),
    )


def _execute_simulation(
    settings: TechnicalSettings,
    chat_inputs: ChatSetupInputs,
    chat_container: Any,
) -> List[Dict[str, Any]] | None:
    """Create orchestrator, run the simulation, and surface user-facing status."""
    try:
        orchestrator = _create_orchestrator(
            settings=settings,
            chat_inputs=chat_inputs,
        )
        total_output_tokens = _stream_simulation_messages(
            orchestrator=orchestrator,
            num_turns=settings.num_turns,
            initial_message=chat_inputs.initial_message,
            model_a_slug=chat_inputs.model_a_slug,
            persona_a=chat_inputs.persona_a,
            persona_b=chat_inputs.persona_b,
            chat_container=chat_container,
        )
    except (KeyError, ValueError, TypeError, RuntimeError, OSError) as error:
        logger.exception(
            "simulation_execution_failed | model_a=%s model_b=%s turns=%s",
            chat_inputs.model_a_slug,
            chat_inputs.model_b_slug,
            settings.num_turns,
        )
        st.error(f"Simulation error: {error}")
        st.info("Tip: try increasing 'Max Tokens' if the API fails with low values.")
        return None

    st.success(f"Simulation complete. Total tokens: {total_output_tokens}")
    return orchestrator.logs


def _stream_simulation_messages(
    orchestrator: Orchestrator,
    num_turns: int,
    initial_message: str,
    model_a_slug: str,
    persona_a: str,
    persona_b: str,
    chat_container: Any,
) -> int:
    """Render each generated message and return the total output-token count."""
    total_output_tokens = 0

    with st.spinner("Agents are conversing..."):
        for log_entry in orchestrator.run_simulation(
            num_turns,
            initial_message=initial_message,
        ):
            output_tokens_raw = log_entry.get("output_tokens", 0)
            try:
                total_output_tokens += int(output_tokens_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "invalid_output_tokens_in_log_entry | value=%s",
                    output_tokens_raw,
                )

            try:
                _render_chat_message(log_entry, model_a_slug, persona_a, persona_b, chat_container)
            except (KeyError, TypeError, ValueError):
                logger.exception(
                    "chat_message_render_failed | speaker_model=%s turn_id=%s",
                    log_entry.get("speaker_model", "unknown"),
                    log_entry.get("turn_id", "unknown"),
                )

            # Slight pacing keeps streamed messages readable and avoids UI burst updates.
            time.sleep(0.1)

    return total_output_tokens


def _persist_simulation_logs(local_storage: Any, simulation_logs: List[Dict[str, Any]]) -> None:
    """Persist generated simulation logs to session state, local storage, and Supabase."""
    append_and_persist_logs(local_storage, pd.DataFrame(simulation_logs))
    st.success("Simulation finished and persisted locally.")

    # Cloud export (best-effort — never blocks the UI flow).
    from parrotlm.supabase_client import get_supabase_client
    from parrotlm.supabase_logger import upload_session_logs

    if not get_supabase_client():
        st.warning("⚠️ Cloud export skipped — check SUPABASE_URL and SUPABASE_ANON_KEY in .env.")
    elif upload_session_logs(simulation_logs):
        st.success("✅ Session logs exported to Supabase.")
    else:
        st.error("❌ Cloud export failed — ensure the 'session_logs' table exists in Supabase.")


def _render_chat_message(
    log_entry: Dict[str, Any],
    model_a_slug: str,
    persona_a: str,
    persona_b: str,
    chat_container: Any,
) -> None:
    """Render one message bubble and its metadata row."""
    speaker_model = str(log_entry.get("speaker_model", "unknown"))
    is_agent_a = speaker_model == model_a_slug
    speaker_label = persona_a if is_agent_a else persona_b
    message_content = str(log_entry.get("content", ""))

    latency_raw = log_entry.get("latency_ms", 0)
    tokens_raw = log_entry.get("output_tokens", 0)
    try:
        latency_display = f"{float(latency_raw):.0f}ms"
    except (TypeError, ValueError):
        latency_display = "n/a"
    try:
        tokens_display = str(int(tokens_raw))
    except (TypeError, ValueError):
        tokens_display = "n/a"

    # Clamp very long personas so chat headers do not dominate message content.
    if len(speaker_label) > MAX_SPEAKER_LABEL_LENGTH:
        speaker_label = f"{speaker_label[:47]}..."

    with chat_container:
        with st.chat_message(name=speaker_label):
            st.write(message_content)

        st.markdown(
            f"<div style='text-align: right; margin-top: -15px; margin-bottom: 10px;'>"
            f"<span style='color: gray; font-size: 0.8rem;'>"
            f"Latency {latency_display} | Tokens {tokens_display} | "
            f"Model {speaker_model}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
