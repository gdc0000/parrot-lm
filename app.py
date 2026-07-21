from __future__ import annotations

import logging

import streamlit as st
from parrotlm.infrastructure._logging import setup_logging
from parrotlm.infrastructure.supabase_client import get_supabase_client
from parrotlm.infrastructure.supabase_logger import (
    SupabaseBufferedLogger,
    flush_supabase_log_handlers,
    install_supabase_log_handler,
)
from parrotlm.orchestration.orchestrator import AgentConfig, Orchestrator
from parrotlm.validation.prompt_utils import construct_system_prompt

# Load .env so SUPABASE_URL / SUPABASE_ANON_KEY default-fill the sidebar.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import os  # noqa: E402  (imported after dotenv block for ordering clarity)

# Configure stdout + rotating JSONL file handlers once per process.
setup_logging()

AGENT_COLORS = {"A": "#4A90D9", "B": "#D94A7B"}
# Streamlit's `st.chat_message(avatar=...)` only accepts "user", "assistant",
# a single emoji, an image path, or a URL. A bare letter like "A" is treated as
# a file path and raises "Error opening 'A'" — so use emoji avatars instead.
AGENT_AVATARS = {"A": "🅰️", "B": "🅱️"}

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

    with st.expander("Supabase (optional cloud logging)", expanded=False):
        supabase_url = st.text_input(
            "Supabase URL",
            value=os.getenv("SUPABASE_URL", ""),
            help="Your Supabase project URL. Leave blank to disable cloud logging.",
            key="supabase_url",
        )
        supabase_key = st.text_input(
            "Supabase Anon Key",
            value=os.getenv("SUPABASE_ANON_KEY", ""),
            type="password",
            help="Your Supabase anon/publishable key.",
            key="supabase_anon_key",
        )
        if supabase_url.strip() and supabase_key.strip():
            st.caption("🟢 Cloud logging enabled — session + application logs will be uploaded.")
        else:
            st.caption("⚪ Cloud logging disabled (no URL/key). Simulation still runs locally.")

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
run_clicked = st.button(
    "▶ Run Simulation", type="primary", use_container_width=True
)

# Clear previous results whenever config changes or a new run starts.
if "last_run_signature" not in st.session_state:
    st.session_state["last_run_signature"] = None

signature = (
    model_a, model_b, persona_a, persona_b, temp_a, temp_b,
    num_turns, initial_message, max_tokens, context_window,
)
if st.session_state["last_run_signature"] != signature:
    st.session_state.pop("conversation_log", None)
    st.session_state["last_run_signature"] = signature

# Re-render any previously completed conversation so it survives reruns.
if "conversation_log" in st.session_state:
    for _entry in st.session_state["conversation_log"]:
        with st.chat_message("assistant", avatar=AGENT_AVATARS[_entry["slot"]]):
            st.markdown(f"**Agent {_entry['slot']}** · `{_entry['model']}`")
            st.markdown(_entry["content"])
            with st.status("Metrics", expanded=False):
                st.markdown(
                    f"**Turn** {_entry['turn']}/{_entry['num_turns']}  \n"
                    f"**Latency** {_entry['latency']:.0f}ms  \n"
                    f"**Tokens** {_entry['in_tokens']} in → {_entry['out_tokens']} out"
                )
    if "summary" in st.session_state:
        st.divider()
        st.success(st.session_state["summary"])

if run_clicked:
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

    # --- Supabase wiring (mirrors main.py) ---
    # The session client uploads each conversation turn to the `session_logs`
    # table; the application-log handler streams structured logs to
    # `application_logs`. Both are no-ops when Supabase credentials are absent.
    supabase_client = get_supabase_client(
        url=supabase_url.strip() or None,
        key=supabase_key.strip() or None,
    )
    install_supabase_log_handler(batch_size=10, client=supabase_client)
    session_logger = SupabaseBufferedLogger(batch_size=10)
    cloud_enabled = session_logger.is_available

    # Show the initial message
    with st.chat_message("user"):
        st.markdown(initial_message)

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0.0
    turn_count = 0
    conversation_log: list[dict] = []
    session_rows_pushed = 0

    status = st.status("Running simulation…", expanded=True)
    status.update(
        label=f"▶ Running simulation ({num_turns} turns, 2 calls each ≈ "
              f"{num_turns * 2 * 6}s)",
    )

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

            status.update(
                label=f"▶ Running… turn {turn}/{num_turns} "
                      f"({turn_count} of {num_turns * 2} messages)",
            )

            with st.chat_message("assistant", avatar=AGENT_AVATARS[slot]):
                st.markdown(f"**Agent {slot}** · `{model}`")
                st.markdown(content)
                with st.status("Metrics", expanded=False):
                    st.markdown(
                        f"**Turn** {turn}/{num_turns}  \n"
                        f"**Latency** {latency:.0f}ms  \n"
                        f"**Tokens** {in_tokens} in → {out_tokens} out"
                    )

            conversation_log.append({
                "slot": slot, "model": model, "content": content,
                "turn": turn, "num_turns": num_turns,
                "latency": latency, "in_tokens": in_tokens,
                "out_tokens": out_tokens,
            })

            # Stream this turn to Supabase's session_logs table.
            session_logger.push(log_entry)
            if session_logger.is_available:
                session_rows_pushed += 1

    except Exception as error:
        status.update(label="Simulation failed", state="error")
        root_cause = error.__cause__ or error
        details = str(root_cause).replace(openrouter_key, "[redacted]")
        st.error(
            f"**Simulation stopped.**\n\n"
            f"OpenRouter returned: `{details}`\n\n"
            "Check that the key is valid, the selected model is available, "
            "and the account has sufficient credits."
        )
        st.stop()
    finally:
        # Flush any buffered session rows + application logs even on early stop.
        session_logger.flush()
        flush_supabase_log_handlers()

    # --- Summary ---
    avg_latency = total_latency_ms / turn_count if turn_count else 0
    st.divider()
    summary = (
        f"**Simulation complete** — {turn_count} messages  \n"
        f"Total tokens: {total_input_tokens} in → {total_output_tokens} out  ·  "
        f"Avg latency: {avg_latency:.0f}ms"
    )
    if cloud_enabled:
        summary += (
            f"  \n☁️ Uploaded {session_rows_pushed} session rows + "
            f"application logs to Supabase."
        )
    else:
        summary += "  \n⚪ Cloud logging skipped (no Supabase credentials)."
    status.update(label="✅ Simulation complete", state="complete")
    st.success(summary)

    # Persist so the conversation survives Streamlit reruns triggered by
    # interacting with collapsed status widgets.
    st.session_state["conversation_log"] = conversation_log
    st.session_state["summary"] = summary
