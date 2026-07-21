from __future__ import annotations

import streamlit as st
from parrotlm.orchestration.orchestrator import AgentConfig, Orchestrator
from parrotlm.validation.prompt_utils import construct_system_prompt

AGENT_COLORS = {"A": "#4A90D9", "B": "#D94A7B"}

st.set_page_config(page_title="ParrotLM", page_icon="🦜", layout="wide")
st.title("🦜 ParrotLM")
st.caption("Two-Agent Conversation Simulator")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")

    st.subheader("API Keys")
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Get one at https://openrouter.ai/keys",
        key="openrouter_api_key",
    )

    st.divider()

    st.subheader("Agent A")
    model_a = st.text_input("Model", value="google/gemma-3n-e4b-it", key="model_a")
    persona_a = st.text_area(
        "Persona", value="Chief Technology Officer", key="persona_a"
    )
    temp_a = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="temperature_a")

    st.divider()

    st.subheader("Agent B")
    model_b = st.text_input("Model", value="google/gemma-3n-e4b-it", key="model_b")
    persona_b = st.text_area("Persona", value="Financial Analyst", key="persona_b")
    temp_b = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="temperature_b")

    st.divider()

    st.subheader("Simulation")
    num_turns = st.number_input(
        "Number of turns (A-B round trips)",
        min_value=1,
        max_value=100,
        value=10,
        key="num_turns",
    )
    initial_message = st.text_area(
        "Initial message",
        value="What is your outlook on AI investment over the next 12 months?",
        key="initial_message",
    )
    max_tokens = st.number_input(
        "Max tokens per response",
        min_value=100,
        max_value=4096,
        value=1000,
        key="max_tokens",
    )
    context_window = st.number_input(
        "Context window (turns of history)",
        min_value=1,
        max_value=50,
        value=5,
        key="context_window",
    )

# --- Main Area ---
if not openrouter_key:
    st.info("Enter your OpenRouter API key in the sidebar to get started.")
    st.stop()

# --- Agent Identity Header ---
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {AGENT_COLORS['A']};
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            background: rgba(74, 144, 217, 0.08);
            margin-bottom: 8px;
        ">
            <div style="font-weight: 700; font-size: 1.05em; margin-bottom: 4px;">
                Agent A
            </div>
            <div style="font-size: 0.85em; opacity: 0.8;">
                {model_a}<br/>
                <em>{persona_a}</em> · temp {temp_a}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {AGENT_COLORS['B']};
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            background: rgba(217, 74, 123, 0.08);
            margin-bottom: 8px;
        ">
            <div style="font-weight: 700; font-size: 1.05em; margin-bottom: 4px;">
                Agent B
            </div>
            <div style="font-size: 0.85em; opacity: 0.8;">
                {model_b}<br/>
                <em>{persona_b}</em> · temp {temp_b}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --- Run Simulation ---
if st.button("▶ Run Simulation", type="primary", use_container_width=True):
    agent_a_config = AgentConfig(
        model=model_a,
        system_prompt=construct_system_prompt(persona_a),
        user_persona_snapshot=persona_a,
        max_history_turns=context_window,
        parameters={"max_tokens": max_tokens, "temperature": temp_a},
    )
    agent_b_config = AgentConfig(
        model=model_b,
        system_prompt=construct_system_prompt(persona_b),
        user_persona_snapshot=persona_b,
        max_history_turns=context_window,
        parameters={"max_tokens": max_tokens, "temperature": temp_b},
    )

    orchestrator = Orchestrator(
        agent_a_configuration=agent_a_config,
        agent_b_configuration=agent_b_config,
        scenario_name="streamlit",
        openrouter_api_key=openrouter_key,
    )

    # Show the initial message
    with st.chat_message("user"):
        st.markdown(initial_message)

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0.0
    turn_count = 0

    try:
        for log_entry in orchestrator.run_simulation(
            num_turns=num_turns,
            initial_message=initial_message,
        ):
            slot = log_entry["speaker_slot"]
            content = log_entry["content"]
            latency = log_entry["latency_ms"]
            in_tokens = log_entry["input_tokens"]
            out_tokens = log_entry["output_tokens"]
            turn = log_entry["turn_id"] + 1
            model = log_entry["speaker_model"]

            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            total_latency_ms += latency
            turn_count += 1

            with st.chat_message("assistant", avatar=slot):
                st.markdown(f"**Agent {slot}** · `{model}`")
                st.markdown(content)
                with st.status("Metrics", expanded=False):
                    st.markdown(
                        f"**Turn** {turn}/{num_turns}  \n"
                        f"**Latency** {latency:.0f}ms  \n"
                        f"**Tokens** {in_tokens} in → {out_tokens} out"
                    )

    except Exception as error:
        root_cause = error.__cause__ or error
        details = str(root_cause).replace(openrouter_key, "[redacted]")
        st.error(
            f"**Simulation stopped.**\n\n"
            f"OpenRouter returned: `{details}`\n\n"
            "Check that the key is valid, the selected model is available, "
            "and the account has sufficient credits."
        )
        st.stop()

    # --- Summary ---
    avg_latency = total_latency_ms / turn_count if turn_count else 0
    st.divider()
    st.success(
        f"**Simulation complete** — {turn_count} messages  \n"
        f"Total tokens: {total_input_tokens} in → {total_output_tokens} out  ·  "
        f"Avg latency: {avg_latency:.0f}ms"
    )
