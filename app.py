from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import streamlit as st
from parrotlm.infrastructure._logging import setup_logging
from parrotlm.infrastructure.supabase_client import (
    get_supabase_client,
    resolve_supabase_credentials,
)
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

logger = logging.getLogger(__name__)

AGENT_COLORS = {"A": "#4A90D9", "B": "#D94A7B"}
# Streamlit's `st.chat_message(avatar=...)` only accepts "user", "assistant",
# a single emoji, an image path, or a URL. A bare letter like "A" is treated as
# a file path and raises "Error opening 'A'" — so use emoji avatars instead.
AGENT_AVATARS = {"A": "🅰️", "B": "🅱️"}

# Minimum interval between in-place placeholder updates while streaming.
_STREAM_UPDATE_INTERVAL_S = 0.1


st.set_page_config(page_title="ParrotLM", page_icon="🦜", layout="wide")

st.markdown(
    """
    <style>
    /* ── Title gradient ── */
    .title-gradient {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7C5CFC 0%, #E040FB 50%, #FF6D00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: #8B949E;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        margin-top: -0.4rem;
    }

    /* ── Agent identity cards (glass-morphism) ── */
    .agent-card {
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 8px;
        transition: border-color 0.2s;
    }
    .agent-card:hover { border-color: rgba(255, 255, 255, 0.12); }
    .agent-card-agentA {
        background: linear-gradient(135deg, rgba(74, 144, 217, 0.15) 0%, rgba(124, 92, 252, 0.08) 100%);
        border-left: 3px solid #4A90D9;
    }
    .agent-card-agentB {
        background: linear-gradient(135deg, rgba(217, 74, 123, 0.15) 0%, rgba(224, 64, 251, 0.08) 100%);
        border-left: 3px solid #D94A7B;
    }
    .agent-card-name {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 4px;
    }
    .agent-card-meta {
        font-size: 0.82rem;
        color: #8B949E;
        line-height: 1.5;
    }

    /* ── Chat bubble refinements ── */
    .stChatMessage { border-radius: 12px !important; }

    /* ── Metrics pill ── */
    .metric-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.78rem;
        color: #8B949E;
        margin-right: 6px;
        margin-top: 4px;
    }
    .metric-pill strong { color: #E6EDF3; }

    /* ── Sidebar polish ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22 0%, #0E1117 100%);
    }
    [data-testid="stSidebar"] .stHeader { border-bottom: 1px solid rgba(255,255,255,0.06); }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C5CFC 0%, #E040FB 100%) !important;
        border: none !important;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.1);
    }

    /* ── Divider glow ── */
    .stDivider hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124, 92, 252, 0.3), transparent);
    }

    /* ── Success/info boxes ── */
    .stAlert { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title-gradient">🦜 ParrotLM</div>'
    '<div class="subtitle">Two-Agent Conversation Simulator</div>',
    unsafe_allow_html=True,
)

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")

    st.subheader("API Keys")
    _default_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if "openrouter_api_key" not in st.session_state or not st.session_state.openrouter_api_key:
        st.session_state.openrouter_api_key = _default_openrouter_key
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        value=_default_openrouter_key,
        type="password",
        help="Get one at https://openrouter.ai/keys. Leave as-is to use the key from your .env file.",
        key="openrouter_api_key",
    )

    _supabase_url, _supabase_key = resolve_supabase_credentials(None, None)
    if _supabase_url and _supabase_key:
        st.caption("☁️ Cloud logging active (Supabase)")
    else:
        st.caption("⚪ Cloud logging off")

    st.divider()

    st.subheader("Agent A")
    model_a = st.text_input("Model", value="openrouter/free", key="model_a")
    persona_a = st.text_area(
        "Persona", value="Chief Technology Officer", key="persona_a"
    )
    temp_a = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="temperature_a")

    st.divider()

    st.subheader("Agent B")
    model_b = st.text_input("Model", value="openrouter/free", key="model_b")
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
    st.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY in your .env or enter it in the sidebar.")
    st.stop()

# --- Agent Identity Header ---
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        f"""
        <div class="agent-card agent-card-agentA">
            <div class="agent-card-name">Agent A</div>
            <div class="agent-card-meta">
                {model_a}<br/>
                <em>{persona_a}</em> &middot; temp {temp_a}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="agent-card agent-card-agentB">
            <div class="agent-card-name">Agent B</div>
            <div class="agent-card-meta">
                {model_b}<br/>
                <em>{persona_b}</em> &middot; temp {temp_b}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# --- Run Simulation ---
# Top-down model: one rerun = one agent response.
# `st.session_state["run"]` holds the live run, if any:
#   {generator, stop_requested, num_turns, initial_message, cloud_enabled,
#    session_logger, session_rows_pushed, totals, experiment_id, models}
# `st.session_state["conversation_log"]` is the single source of truth for
# already-rendered messages (survives reruns).
# `st.session_state["summary"]` is shown once a run finishes.


def _render_entry(entry: Dict[str, Any]) -> None:
    """Render one completed conversation entry as a chat bubble."""
    with st.chat_message("assistant", avatar=AGENT_AVATARS[entry["slot"]]):
        st.markdown(f"**Agent {entry['slot']}** &middot; `{entry['model']}`")
        st.markdown(entry["content"])
        st.markdown(
            f'<span class="metric-pill"><strong>Turn</strong> {entry["turn"]}/{entry["num_turns"]}</span>'
            f'<span class="metric-pill"><strong>Latency</strong> {entry["latency"]:.0f}ms</span>'
            f'<span class="metric-pill"><strong>Tokens</strong> {entry["in_tokens"]} in → {entry["out_tokens"]} out</span>',
            unsafe_allow_html=True,
        )


def _finalise_run(
    run: Dict[str, Any], *, interrupted: bool = False, error: str | None = None
) -> None:
    """Flush logging, convert the live run into a summary, and clear the handle."""
    run["session_logger"].flush()
    flush_supabase_log_handlers()

    totals = run["totals"]
    turn_count = totals["count"]
    avg_latency = totals["latency_ms"] / turn_count if turn_count else 0

    summary = (
        f"**{'Interrupted' if interrupted else 'Simulation complete'}** "
        f"— {turn_count} messages  \n"
        f"Total tokens: {totals['input_tokens']} in → "
        f"{totals['output_tokens']} out  ·  "
        f"Avg latency: {avg_latency:.0f}ms"
    )
    if run["cloud_enabled"]:
        summary += (
            f"  \n☁️ Uploaded {run['session_rows_pushed']} session rows + "
            f"application logs to Supabase."
        )
    else:
        summary += "  \n⚪ Cloud logging skipped (no Supabase credentials)."
    if error:
        summary += f"  \n❌ {error}"
    summary += f"  \n🧪 experiment_id: `{run['experiment_id']}`"

    st.session_state["summary"] = summary
    # Remember the experiment_id so the Supabase panel can highlight its rows.
    st.session_state["last_experiment_id"] = run["experiment_id"]
    st.session_state.pop("run", None)


# --- Run / Stop buttons ---
live_run = st.session_state.get("run")
is_running = live_run is not None

col_run, col_stop = st.columns([1, 1])
with col_run:
    run_clicked = st.button(
        "▶ Run Simulation",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    )
with col_stop:
    stop_clicked = st.button(
        "⏹ Stop",
        type="secondary",
        use_container_width=True,
        disabled=not is_running,
    )

# Clear previous results whenever the *configuration* changes.
if "last_run_signature" not in st.session_state:
    st.session_state["last_run_signature"] = None

signature = (
    model_a,
    model_b,
    persona_a,
    persona_b,
    temp_a,
    temp_b,
    num_turns,
    initial_message,
    max_tokens,
    context_window,
)
if st.session_state["last_run_signature"] != signature and not is_running:
    st.session_state.pop("conversation_log", None)
    st.session_state.pop("summary", None)
    st.session_state["last_run_signature"] = signature

# --- Stop button handler ---
# Same granularity as before: the current response finishes, the run stops
# before the next one starts.
if stop_clicked and live_run:
    live_run["stop_requested"] = True
    st.toast("Stopping after the current turn…")

# --- Kick off a new run ---
if run_clicked and not is_running:
    # Fresh run: reset persisted state.
    st.session_state["conversation_log"] = []
    st.session_state.pop("summary", None)

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
    supabase_client = get_supabase_client()
    install_supabase_log_handler(batch_size=10, client=supabase_client)
    session_logger = SupabaseBufferedLogger(batch_size=10)

    run_state: Dict[str, Any] = {
        "stop_requested": False,
        "num_turns": num_turns,
        "initial_message": initial_message,
        "cloud_enabled": session_logger.is_available,
        "session_logger": session_logger,
        "session_rows_pushed": 0,
        "experiment_id": orchestrator.experiment_id,
        "models": {"A": model_a, "B": model_b},
        "totals": {
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
            "count": 0,
        },
    }
    # The token callback is bound per rerun (it writes into that rerun's
    # placeholder), so route it through a mutable holder on the run state.
    def _dispatch_token(piece: str) -> None:
        callback = run_state.get("token_callback")
        if callback is not None:
            callback(piece)

    run_state["generator"] = orchestrator.run_simulation(
        num_turns=num_turns,
        initial_message=initial_message,
        cancellation_requested=lambda: run_state["stop_requested"],
        on_token=_dispatch_token,
    )
    st.session_state["run"] = run_state
    st.rerun()

# --- Re-render any already-collected messages ---
conversation_log: List[Dict[str, Any]] = st.session_state.get("conversation_log", [])

# Show the initial user message if a run has happened or is in progress.
if conversation_log or is_running:
    initial_for_display = live_run["initial_message"] if live_run else initial_message
    with st.chat_message("user"):
        st.markdown(initial_for_display)

for entry in conversation_log:
    _render_entry(entry)

if "summary" in st.session_state and not is_running:
    st.divider()
    st.success(st.session_state["summary"])

# --- Advance the live run by exactly one agent response ---
if live_run:
    totals = live_run["totals"]
    # Responses strictly alternate A, B, A, B…
    slot = "A" if totals["count"] % 2 == 0 else "B"

    status_label = (
        f"▶ Running… {totals['count']} of {live_run['num_turns'] * 2 + 1} messages "
        f"(turn {(totals['count'] + 1) // 2}/{live_run['num_turns']})"
    )
    if live_run["stop_requested"]:
        status_label = (
            f"⛔ Stopping… {totals['count']} messages so far "
            f"(finishing current response)"
        )

    st.status(status_label, expanded=False)
    message_container = st.chat_message("assistant", avatar=AGENT_AVATARS[slot])
    with message_container:
        st.markdown(f"**Agent {slot}** &middot; `{live_run['models'][slot]}`")
        placeholder = st.empty()

        streamed_parts: List[str] = []
        last_render = [0.0]

        def on_token(piece: str) -> None:
            """Update the placeholder in real time (throttled, with cursor)."""
            streamed_parts.append(piece)
            now = time.time()
            if now - last_render[0] >= _STREAM_UPDATE_INTERVAL_S:
                last_render[0] = now
                placeholder.markdown("".join(streamed_parts) + "▌")

    live_run["token_callback"] = on_token
    error_message: str | None = None
    log_entry: Dict[str, Any] | None = None
    finished = False
    try:
        log_entry = next(live_run["generator"], None)
        if log_entry is None:
            finished = True
    except StopIteration:
        finished = True
    except Exception as exception:  # noqa: BLE001  (surfaced to UI)
        error_message = str(exception).replace(openrouter_key, "[redacted]")
        finished = True

    # Final in-place render of the full response (no cursor).
    final_content = (
        log_entry["content"] if log_entry else "".join(streamed_parts).strip()
    )
    placeholder.markdown(final_content or "…")

    if log_entry is not None:
        totals["input_tokens"] += log_entry["input_tokens"]
        totals["output_tokens"] += log_entry["output_tokens"]
        totals["latency_ms"] += log_entry["latency_ms"]
        totals["count"] += 1
        if live_run["cloud_enabled"]:
            live_run["session_logger"].push(log_entry)
            live_run["session_rows_pushed"] += 1

        rendered = {
            "slot": log_entry["speaker_slot"],
            "model": log_entry["speaker_model"],
            "content": log_entry["content"],
            "turn": log_entry["turn_id"] + 1,
            "num_turns": live_run["num_turns"],
            "latency": log_entry["latency_ms"],
            "in_tokens": log_entry["input_tokens"],
            "out_tokens": log_entry["output_tokens"],
        }
        conversation_log.append(rendered)
        st.session_state["conversation_log"] = conversation_log

        with message_container:
            st.markdown(
                f'<span class="metric-pill"><strong>Turn</strong> {rendered["turn"]}/{rendered["num_turns"]}</span>'
                f'<span class="metric-pill"><strong>Latency</strong> {rendered["latency"]:.0f}ms</span>'
                f'<span class="metric-pill"><strong>Tokens</strong> {rendered["in_tokens"]} in → {rendered["out_tokens"]} out</span>',
                unsafe_allow_html=True,
            )

    if finished or live_run["stop_requested"]:
        _finalise_run(
            live_run,
            interrupted=bool(live_run["stop_requested"]),
            error=error_message,
        )
    st.rerun()
