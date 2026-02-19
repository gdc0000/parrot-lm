"""Chatbot setup tab and simulation execution flow."""

from __future__ import annotations

import time
from typing import Any, Dict

import pandas as pd
import streamlit as st

from parrotlm.orchestrator import Orchestrator
from parrotlm.prompt_utils import construct_system_prompt

from .session_state import append_and_persist_logs
from .sidebar import TechnicalSettings


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


def render_chatbot_setup_tab(settings: TechnicalSettings, local_storage: Any) -> None:
    """Render chatbot setup controls and execute the conversation simulation."""
    st.markdown("### 🎭 Configure the Encounter")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Chatbot A")
        model_a_slug = st.text_input(
            "Model A Slug",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            key="model_a",
        )
        persona_a = st.text_area("Persona", "A mysterious stranger at a jazz club", height=100)

    with col2:
        st.markdown("#### Chatbot B")
        model_b_slug = st.text_input(
            "Model B Slug",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            key="model_b",
        )
        persona_b = st.text_area("Persona", "A sharp-witted bartender", height=100)

    st.markdown("---")
    initial_message = st.text_input("The conversation starts with:", "Is this seat taken?")

    if not st.button("🚀 Start Conversation", type="primary", width="stretch"):
        return

    st.write("### 🟢 Live Conversation")
    chat_container = st.container()
    total_tokens = 0
    orchestrator: Orchestrator | None = None

    try:
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

        orchestrator = Orchestrator(
            agent_a_config=chatbot_a_config,
            agent_b_config=chatbot_b_config,
            scenario_name=f"{persona_a[:15]} vs {persona_b[:15]}",
        )

        with st.spinner("Agents are conversing..."):
            for log_entry in orchestrator.run_simulation(
                settings.num_turns,
                initial_message=initial_message,
            ):
                total_tokens += log_entry["output_tokens"]
                _render_chat_message(log_entry, model_a_slug, persona_a, persona_b, chat_container)
                time.sleep(0.1)

        st.success(f"Simulation Complete. Total Tokens: {total_tokens}")
    except Exception as error:
        st.error(f"❌ Simulation Error: {error}")
        st.info("💡 Tip: Try increasing 'Max Tokens' if the API is failing with low values.")

    if not orchestrator:
        return

    append_and_persist_logs(local_storage, pd.DataFrame(orchestrator.logs))
    st.success("Simulation Finished & Persisted Locally!")


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
    avatar = "🎭" if is_agent_a else "🍸"

    if len(speaker_label) > 50:
        speaker_label = f"{speaker_label[:47]}..."

    with chat_container:
        with st.chat_message(name=speaker_label, avatar=avatar):
            st.write(log_entry["content"])

        st.markdown(
            f"<div style='text-align: right; margin-top: -15px; margin-bottom: 10px;'>"
            f"<span style='color: gray; font-size: 0.8rem;'>"
            f"⏱️ {log_entry['latency_ms']:.0f}ms | 🔢 {log_entry['output_tokens']} tokens | "
            f"🤖 {log_entry['speaker_model']}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

