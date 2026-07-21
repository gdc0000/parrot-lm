from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from typing import Any, Dict

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

logger = logging.getLogger(__name__)

AGENT_COLORS = {"A": "#4A90D9", "B": "#D94A7B"}
# Streamlit's `st.chat_message(avatar=...)` only accepts "user", "assistant",
# a single emoji, an image path, or a URL. A bare letter like "A" is treated as
# a file path and raises "Error opening 'A'" — so use emoji avatars instead.
AGENT_AVATARS = {"A": "🅰️", "B": "🅱️"}

# Sentinel pushed onto the run queue when the background worker finishes.
# It must be a plain string (not a module-level `object()`) because each
# Streamlit rerun re-executes this module — a fresh `object()` would get a new
# identity every rerun, so the draining loop's `is` check would never match the
# object the still-running worker captured.
_RUN_DONE = "__parrotlm_run_done__"


def _simulation_worker(
    orchestrator: Orchestrator,
    num_turns: int,
    initial_message: str,
    out_queue: "queue.Queue[Any]",
    stop_event: threading.Event,
    session_logger: SupabaseBufferedLogger,
) -> None:
    """Run the generator in a background thread, draining entries onto a queue.

    The orchestrator's `cancellation_requested` callback reads `stop_event`,
    so the main UI thread can interrupt the run before the next agent turn by
    setting the event. Each yielded log entry is also pushed to Supabase.
    """
    try:
        for log_entry in orchestrator.run_simulation(
            num_turns=num_turns,
            initial_message=initial_message,
            cancellation_requested=stop_event.is_set,
        ):
            out_queue.put(log_entry)
            # Stream this turn to Supabase's session_logs table.
            session_logger.push(log_entry)
    except Exception as exception:  # noqa: BLE001  (surfaced to UI via queue)
        out_queue.put(exception)
    finally:
        session_logger.flush()
        flush_supabase_log_handlers()
        out_queue.put(_RUN_DONE)

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
# `st.session_state["run"]` holds the live background run, if any:
#   {thread, queue, stop_event, num_turns, initial_message, cloud_enabled,
#    session_rows_pushed, totals, run_id}
# `st.session_state["conversation_log"]` is the single source of truth for
# already-rendered messages (survives reruns).
# `st.session_state["summary"]` is shown once a run finishes.

def _render_entry(entry: Dict[str, Any]) -> None:
    """Render one conversation entry as a chat bubble."""
    with st.chat_message("assistant", avatar=AGENT_AVATARS[entry["slot"]]):
        st.markdown(f"**Agent {entry['slot']}** · `{entry['model']}`")
        st.markdown(entry["content"])
        with st.status("Metrics", expanded=False):
            st.markdown(
                f"**Turn** {entry['turn']}/{entry['num_turns']}  \n"
                f"**Latency** {entry['latency']:.0f}ms  \n"
                f"**Tokens** {entry['in_tokens']} in → {entry['out_tokens']} out"
            )


def _finalise_run(run: Dict[str, Any], *, interrupted: bool = False,
                   error: str | None = None) -> None:
    """Convert the live run into a persisted summary and clear the live handle."""
    totals = run["totals"]
    turn_count = totals["count"]
    avg_latency = totals["latency_ms"] / turn_count if turn_count else 0
    label = "⛔ Simulation stopped" if interrupted else "✅ Simulation complete"
    state = "error" if interrupted else "complete"

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
    # Replace the status widget that was created during the run.
    if "status_widget" in st.session_state:
        st.session_state["status_widget"].update(label=label, state=state)
        st.session_state.pop("status_widget", None)


def _fetch_recent_supabase_rows(
    client: Any, experiment_id: str, *, limit: int = 20
) -> Dict[str, Any]:
    """Fetch the latest session + application rows for display.

    Uses ordered + limited queries so the Supabase REST API's 1000-row
    PostgREST pagesize cap never hides the newest rows.
    """
    result: Dict[str, Any] = {"session": [], "application": [], "error": None}
    try:
        session_resp = (
            client.table("session_logs")
            .select(
                "experiment_id,turn_id,scenario,speaker_model,responder_model,"
                "content,timestamp,latency_ms,input_tokens,output_tokens"
            )
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        result["session"] = session_resp.data or []
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"session_logs: {type(exc).__name__}: {exc}"
    try:
        app_resp = (
            client.table("application_logs")
            .select("timestamp,level,event,message")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        result["application"] = app_resp.data or []
    except Exception as exc:  # noqa: BLE001
        prev = result["error"]
        result["error"] = (prev + " | " if prev else "") + (
            f"application_logs: {type(exc).__name__}: {exc}"
        )
    return result


def _render_supabase_panel(
    supabase_url_value: str, supabase_key_value: str,
) -> None:
    """Render a collapsible panel showing the latest Supabase rows."""
    if not (supabase_url_value.strip() and supabase_key_value.strip()):
        return
    st.divider()
    with st.expander("☁️ Recent Supabase logs", expanded=False):
        st.caption(
            "Ordered by timestamp (newest first) with a `LIMIT`, so the "
            "Supabase REST API's 1000-row pagesize cap never hides new rows."
        )
        col_n, col_refresh = st.columns([1, 1])
        with col_n:
            show_n = st.number_input(
                "Rows to show", min_value=1, max_value=100, value=10, step=1,
                key="supabase_panel_n",
            )
        with col_refresh:
            st.write("")
            st.write("")
            refresh = st.button("🔄 Refresh", use_container_width=True)

        # Cache the fetched rows so they survive reruns until Refresh is hit.
        cache_key = "supabase_panel_cache"
        if refresh or cache_key not in st.session_state:
            client = get_supabase_client(
                url=supabase_url_value.strip() or None,
                key=supabase_key_value.strip() or None,
            )
            if client is None:
                st.session_state[cache_key] = {
                    "error": "Supabase client unavailable (check URL/key).",
                    "session": [], "application": [], "n": show_n,
                }
            else:
                exp_id = st.session_state.get("last_experiment_id")
                st.session_state[cache_key] = {
                    **_fetch_recent_supabase_rows(client, exp_id or "", limit=int(show_n)),
                    "n": int(show_n),
                }

        cache = st.session_state[cache_key]
        if cache.get("error"):
            st.error(cache["error"])

        session_rows = cache.get("session", [])
        app_rows = cache.get("application", [])
        st.markdown(f"**`session_logs`** — {len(session_rows)} rows")
        if session_rows:
            import pandas as pd  # type: ignore[import-not-typed]
            df = pd.DataFrame(session_rows)[
                ["timestamp", "turn_id", "speaker_model", "content",
                 "latency_ms", "input_tokens", "output_tokens"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No session rows found.")

        st.markdown(f"**`application_logs`** — {len(app_rows)} rows")
        if app_rows:
            import pandas as pd  # type: ignore[import-not-typed]
            df = pd.DataFrame(app_rows)[["timestamp", "level", "event", "message"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No application rows found.")


# --- Run / Stop buttons ---
live_run = st.session_state.get("run")
is_running = bool(live_run and live_run["thread"].is_alive())

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
    model_a, model_b, persona_a, persona_b, temp_a, temp_b,
    num_turns, initial_message, max_tokens, context_window,
)
if st.session_state["last_run_signature"] != signature and not is_running:
    st.session_state.pop("conversation_log", None)
    st.session_state.pop("summary", None)
    st.session_state["last_run_signature"] = signature

# --- Stop button handler ---
if stop_clicked and live_run:
    live_run["stop_event"].set()
    st.toast("Stopping after the current turn…")
    st.rerun()

# --- Re-render any already-collected messages ---
conversation_log: list[Dict[str, Any]] = st.session_state.get("conversation_log", [])

# Show the initial user message if a run has happened or is in progress.
if conversation_log or is_running or run_clicked:
    initial_for_display = (
        live_run["initial_message"] if live_run else initial_message
    )
    with st.chat_message("user"):
        st.markdown(initial_for_display)

for entry in conversation_log:
    _render_entry(entry)

if "summary" in st.session_state and not is_running:
    st.divider()
    st.success(st.session_state["summary"])

# --- Kick off a new run ---
if run_clicked and not is_running:
    # Fresh run: reset persisted state.
    st.session_state["conversation_log"] = []
    st.session_state.pop("summary", None)
    conversation_log = []

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
    supabase_client = get_supabase_client(
        url=supabase_url.strip() or None,
        key=supabase_key.strip() or None,
    )
    install_supabase_log_handler(batch_size=10, client=supabase_client)
    session_logger = SupabaseBufferedLogger(batch_size=10)
    cloud_enabled = session_logger.is_available

    out_queue: "queue.Queue[Any]" = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_simulation_worker,
        args=(
            orchestrator,
            num_turns,
            initial_message,
            out_queue,
            stop_event,
            session_logger,
        ),
        daemon=True,
    )
    st.session_state["run"] = {
        "thread": worker,
        "queue": out_queue,
        "stop_event": stop_event,
        "num_turns": num_turns,
        "initial_message": initial_message,
        "cloud_enabled": cloud_enabled,
        "session_rows_pushed": 0,
        "experiment_id": orchestrator.experiment_id,
        "totals": {
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
            "count": 0,
        },
        "run_id": uuid.uuid4().hex,
    }
    worker.start()
    st.rerun()

# --- Drain the live run queue and render new messages incrementally ---
if live_run:
    status = st.status("Running simulation…", expanded=True)
    st.session_state["status_widget"] = status
    totals = live_run["totals"]

    interrupted = False
    error_message: str | None = None
    finished = False

    # Drain everything currently in the queue without blocking the UI long.
    while True:
        try:
            item = live_run["queue"].get_nowait()
        except queue.Empty:
            break
        if item == _RUN_DONE:
            finished = True
            break
        if isinstance(item, Exception):
            error_message = str(item).replace(openrouter_key, "[redacted]")
            finished = True
            break

        entry: Dict[str, Any] = item
        totals["input_tokens"] += entry["input_tokens"]
        totals["output_tokens"] += entry["output_tokens"]
        totals["latency_ms"] += entry["latency_ms"]
        totals["count"] += 1
        if live_run["cloud_enabled"]:
            live_run["session_rows_pushed"] += 1

        rendered = {
            "slot": entry["speaker_slot"],
            "model": entry["speaker_model"],
            "content": entry["content"],
            "turn": entry["turn_id"] + 1,
            "num_turns": live_run["num_turns"],
            "latency": entry["latency_ms"],
            "in_tokens": entry["input_tokens"],
            "out_tokens": entry["output_tokens"],
        }
        conversation_log.append(rendered)
        _render_entry(rendered)
        st.session_state["conversation_log"] = conversation_log

    # Guard: if the worker thread died without pushing the sentinel (e.g.
    # an interpreter-level crash), finalise anyway so the UI doesn't loop forever.
    if not finished and not live_run["thread"].is_alive():
        # Drain any straggler items first.
        while True:
            try:
                item = live_run["queue"].get_nowait()
            except queue.Empty:
                break
            if item == _RUN_DONE:
                finished = True
                break
            if isinstance(item, Exception):
                error_message = str(item).replace(openrouter_key, "[redacted]")
                finished = True
                break
            # (real log entries are already handled above; skip stragglers safely)
        if not finished:
            finished = True
            if not error_message:
                error_message = "Worker thread exited unexpectedly."

    # Update the status label.
    if live_run["stop_event"].is_set() and not finished:
        status.update(
            label=f"⛔ Stopping… {totals['count']} messages so far "
                  f"(finishing current turn)",
        )
    elif not finished:
        status.update(
            label=f"▶ Running… {totals['count']} of "
                  f"{live_run['num_turns'] * 2} messages "
                  f"(turn {(totals['count'] + 1) // 2}/{live_run['num_turns']})",
        )

    if finished:
        interrupted = bool(live_run["stop_event"].is_set())
        _finalise_run(
            live_run,
            interrupted=interrupted,
            error=error_message,
        )
        st.rerun()
    else:
        # Keep the UI responsive: poll again shortly so new messages render
        # as soon as they arrive and the Stop button stays clickable.
        time.sleep(0.5)
        st.rerun()

# --- Recent Supabase logs panel ---
# Shown only when no run is in progress and Supabase credentials are present.
if not is_running:
    _render_supabase_panel(supabase_url, supabase_key)
