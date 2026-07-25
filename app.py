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

PRESETS = {
    "Chief Technology Officer": "Chief Technology Officer",
    "Financial Analyst": "Financial Analyst",
    "Startup Founder": "Ambitious startup founder pitching a new venture",
    "Skeptical Journalist": "Investigative journalist who challenges every claim",
    "Therapist": "Empathetic therapist who listens and asks probing questions",
    "Philosopher": "Reflective philosopher who questions assumptions",
}

_STREAM_UPDATE_INTERVAL_S = 0.1


def persona_editor(agent_key: str) -> str:
    choice = st.selectbox("Persona", list(PRESETS), key=f"{agent_key}_preset")
    return st.text_area(
        "Persona",
        value=PRESETS[choice],
        key=f"{agent_key}_persona",
        height=80,
        label_visibility="collapsed",
    )


def agent_panel(slot: str, default_model: str) -> None:
    st.markdown(f"#### Agent {slot}")
    st.text_input("Model", value=default_model, key=f"model_{slot.lower()}")
    persona_editor(f"agent_{slot.lower()}")
    st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key=f"temperature_{slot.lower()}")


def agent_state(slot: str, default_model: str, default_persona: str):
    key = slot.lower()
    return (
        st.session_state.get(f"model_{key}", default_model),
        st.session_state.get(f"agent_{key}_persona", default_persona),
        st.session_state.get(f"temperature_{key}", 1.0),
    )


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import os  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ParrotLM", page_icon="\U0001f99c", layout="wide")

# Bubble styling
st.markdown(
    """
    <style>
    .element-container:has(.mk-a) + .element-container div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: rgba(28, 131, 225, 0.12);
        border-radius: 16px 16px 16px 4px;
    }
    .element-container:has(.mk-b) + .element-container div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: rgba(120, 120, 120, 0.15);
        border-radius: 16px 16px 4px 16px;
    }
    .mk-a, .mk-b { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ParrotLM")

live_run = st.session_state.get("run")
is_running = live_run is not None

# --- Main-tab configuration ---
with st.expander("Configuration", expanded=not st.session_state.get("conversation_log")):
    cred_col, sim_col = st.columns(2)
    with cred_col:
        st.subheader("Credentials")
        _default_key = os.getenv("OPENROUTER_API_KEY", "")
        if "openrouter_api_key" not in st.session_state or not st.session_state.openrouter_api_key:
            st.session_state.openrouter_api_key = _default_key
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            value=_default_key,
            type="password",
            key="openrouter_api_key",
        )
        _supabase_url, _supabase_key = resolve_supabase_credentials(None, None)
        st.caption("Supabase: " + ("active" if _supabase_url and _supabase_key else "off"))
    with sim_col:
        st.subheader("Simulation")
        num_turns = st.number_input("Turns (A\u2013B round trips)", min_value=1, max_value=100, value=10)
        max_tokens = st.number_input("Max tokens / response", min_value=100, max_value=4096, value=1000)
        context_window = st.number_input("Context window (turns)", min_value=1, max_value=50, value=5)
    initial_message = st.text_area(
        "Opening message (Agent A)",
        value="What is your outlook on AI investment over the next 12 months?",
        height=68,
    )

if not openrouter_key:
    st.warning("No OpenRouter API key. Set OPENROUTER_API_KEY in .env or enter it above.")
    st.stop()

# --- Panel toggles + run controls ---
t_a, t_b, t_run, t_stop = st.columns([1, 1, 1, 1])
with t_a:
    show_a = st.toggle("Agent A panel", value=False, key="show_agent_a")
with t_b:
    show_b = st.toggle("Agent B panel", value=False, key="show_agent_b")
with t_run:
    run_clicked = st.button("Run", type="primary", use_container_width=True, disabled=is_running)
with t_stop:
    stop_clicked = st.button("Stop", type="secondary", use_container_width=True, disabled=not is_running)

# --- Layout: left panel | main | right panel ---
spec: List[float] = []
if show_a:
    spec.append(2)
spec.append(5)
if show_b:
    spec.append(2)
cols = st.columns(spec)

pos = 0
if show_a:
    with cols[pos]:
        with st.container(border=True):
            agent_panel("A", "openrouter/free")
    pos += 1
main_col = cols[pos]
pos += 1
if show_b:
    with cols[pos]:
        with st.container(border=True):
            agent_panel("B", "openrouter/free")

model_a, persona_a, temp_a = agent_state("A", "openrouter/free", "Chief Technology Officer")
model_b, persona_b, temp_b = agent_state("B", "openrouter/free", "Financial Analyst")

# --- State management ---
if "conversation_log" not in st.session_state:
    st.session_state["conversation_log"] = []
if "last_run_signature" not in st.session_state:
    st.session_state["last_run_signature"] = None

signature = (model_a, model_b, persona_a, persona_b, temp_a, temp_b, num_turns, initial_message, max_tokens, context_window)
if st.session_state["last_run_signature"] != signature and not is_running:
    st.session_state.pop("conversation_log", None)
    st.session_state.pop("summary", None)
    st.session_state["last_run_signature"] = signature

if stop_clicked and live_run:
    live_run["stop_requested"] = True
    st.toast("Stopping after the current turn\u2026")

# --- New run ---
if run_clicked and not is_running:
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
        "totals": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "count": 0},
    }

    def _dispatch_token(piece: str) -> None:
        cb = run_state.get("token_callback")
        if cb is not None:
            cb(piece)

    run_state["generator"] = orchestrator.run_simulation(
        num_turns=num_turns,
        initial_message=initial_message,
        cancellation_requested=lambda: run_state["stop_requested"],
        on_token=_dispatch_token,
    )
    st.session_state["run"] = run_state
    st.rerun()

# --- Chat area ---
with main_col:
    conversation_log: List[Dict[str, Any]] = st.session_state.get("conversation_log", [])

    # Completed messages first, so the conversation unfolds live during a run
    for i, entry in enumerate(conversation_log):
        slot = entry["slot"]
        header = (
            f"**Agent {slot}** \u00b7 T{entry['turn']}/{entry['num_turns']} "
            f"\u00b7 {entry['latency']:.0f}ms \u00b7 {entry['in_tokens']}\u2192{entry['out_tokens']}"
        )
        if slot == "A":
            body, _spacer = st.columns([5, 1])
        else:
            _spacer, body = st.columns([1, 5])
        with body:
            st.markdown(f'<span class="mk-{slot.lower()}"></span>', unsafe_allow_html=True)
            with st.container(border=True):
                st.caption(header)
                st.markdown(entry["content"])

    if not conversation_log and not is_running:
        st.caption("Start a simulation to see the conversation here.")

    if is_running:
        totals = live_run["totals"]
        expected = live_run["num_turns"] * 2 + 1
        if live_run["stop_requested"]:
            st.caption(f"Stopping\u2026 {totals['count']} messages so far")
        else:
            st.caption(f"Message {totals['count'] + 1} of {expected}")
        st.progress(min(totals["count"] / expected, 1.0))

        slot = "A" if totals["count"] % 2 == 0 else "B"
        if slot == "A":
            body, _spacer = st.columns([5, 1])
        else:
            _spacer, body = st.columns([1, 5])

        with body:
            st.markdown(f'<span class="mk-{slot.lower()}"></span>', unsafe_allow_html=True)
            with st.container(border=True):
                st.caption(f"**Agent {slot}** \u00b7 streaming\u2026")
                placeholder = st.empty()

                streamed_parts: List[str] = []
                last_render = [0.0]

                def on_token(piece: str) -> None:
                    streamed_parts.append(piece)
                    now = time.time()
                    if now - last_render[0] >= _STREAM_UPDATE_INTERVAL_S:
                        last_render[0] = now
                        placeholder.markdown("".join(streamed_parts) + "\u258c")

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
        except Exception as exc:
            error_message = str(exc).replace(openrouter_key, "[redacted]")
            finished = True

        final_content = log_entry["content"] if log_entry else "".join(streamed_parts).strip()
        placeholder.markdown(final_content or "\u2026")

        if log_entry is not None:
            totals["input_tokens"] += log_entry["input_tokens"]
            totals["output_tokens"] += log_entry["output_tokens"]
            totals["latency_ms"] += log_entry["latency_ms"]
            totals["count"] += 1
            if live_run["cloud_enabled"]:
                live_run["session_logger"].push(log_entry)
                live_run["session_rows_pushed"] += 1

            conversation_log.append({
                "slot": log_entry["speaker_slot"],
                "content": log_entry["content"],
                "turn": log_entry["turn_id"] + 1,
                "num_turns": live_run["num_turns"],
                "latency": log_entry["latency_ms"],
                "in_tokens": log_entry["input_tokens"],
                "out_tokens": log_entry["output_tokens"],
            })
            st.session_state["conversation_log"] = conversation_log

        if finished or live_run["stop_requested"]:
            run = live_run
            run["session_logger"].flush()
            flush_supabase_log_handlers()
            t = run["totals"]
            cnt = t["count"]
            avg = t["latency_ms"] / cnt if cnt else 0
            summary = (
                f"**{'Interrupted' if run['stop_requested'] else 'Done'}** \u2014 "
                f"{cnt} messages \u00b7 {t['input_tokens']}\u2192{t['output_tokens']} tokens "
                f"\u00b7 {avg:.0f}ms avg"
            )
            if run["cloud_enabled"]:
                summary += f" \u00b7 {run['session_rows_pushed']} rows \u2192 Supabase"
            if error_message:
                summary += f" \u00b7 {error_message}"
            summary += f" \u00b7 `{run['experiment_id']}`"
            st.session_state["summary"] = summary
            st.session_state["last_experiment_id"] = run["experiment_id"]
            st.session_state.pop("run", None)
        st.rerun()

    elif "summary" in st.session_state:
        st.success(st.session_state["summary"])
