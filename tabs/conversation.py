import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
import uuid
import json
from simulation_config import DATA_DIR
from analysis_utils import process_logs, process_custom_lexicon
from prompt_utils import construct_system_prompt

# --- Conversation Tab Logic ---
def init_conversation_state():
    """Initialize conversation state with proper defaults"""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "conversation_status" not in st.session_state:
        st.session_state.conversation_status = "idle"
    if "conversation_metrics" not in st.session_state:
        st.session_state.conversation_metrics = {
            "total_tokens": 0,
            "total_messages": 0,
            "avg_latency": 0,
            "agent_a_messages": 0,
            "agent_b_messages": 0,
        }
    if "conversation_turns_completed" not in st.session_state:
        st.session_state.conversation_turns_completed = 0
    if "conversation_max_turns" not in st.session_state:
        st.session_state.conversation_max_turns = 0
    if "auto_save_enabled" not in st.session_state:
        st.session_state.auto_save_enabled = True

def render_conversation_tab(config):
    st.markdown("### 💬 Agent Conversation")
    init_conversation_state()

    if not config:
        st.warning("⚠️ Please configure agents and prompts in 'Interaction & Prompts' tab first.")
        st.stop()

    # --- Enhanced Conversation Control Panel ---
    st.markdown("#### 🎮 Conversation Control Panel")

    # Status display
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        status_color = {
            "idle": "🔴",
            "running": "🟢",
            "paused": "🟡",
            "finished": "🔵",
            "continued": "🟣",
        }.get(st.session_state.conversation_status, "⚪")

        st.markdown(
            f"### {status_color} {st.session_state.conversation_status.title()}"
        )

    with status_col2:
        current_messages = len(st.session_state.conversation_history)
        max_messages = (
            st.session_state.conversation_max_turns * 2
            if st.session_state.conversation_max_turns > 0
            else "∞"
        )
        st.markdown(f"### 💬 {current_messages}/{max_messages}")

    with status_col3:
        if st.session_state.conversation_max_turns > 0:
            progress = min(
                current_messages / (st.session_state.conversation_max_turns * 2), 1.0
            )
            st.markdown("### 📊 Progress")
            st.progress(progress)
            st.caption(f"{progress * 100:.1f}% Complete")
        else:
            st.markdown("### 📊 No Limit")

    # Control buttons
    st.markdown("---")
    control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 1])

    with control_col1:
        # Start button
        start_disabled = st.session_state.conversation_status in ["running"]
        if st.button(
            "▶️ Start",
            type="primary",
            disabled=start_disabled,
            help="Start a new conversation or resume from current state",
        ):
            if st.session_state.conversation_status in [
                "idle",
                "finished",
                "continued",
            ]:
                # Set max turns from config
                st.session_state.conversation_max_turns = config["num_turns"]
                st.session_state.conversation_status = "running"
                st.session_state.conversation_turns_completed = 0
                st.rerun()

    with control_col2:
        # Pause button
        pause_disabled = st.session_state.conversation_status != "running"
        if st.button(
            "⏸️ Pause",
            disabled=pause_disabled,
            help="Pause the conversation (can be resumed)",
        ):
            st.session_state.conversation_status = "paused"
            st.rerun()

    with control_col3:
        # Continue button
        continue_disabled = st.session_state.conversation_status not in [
            "paused",
            "finished",
        ]
        if st.button(
            "⏹️ Continue",
            disabled=continue_disabled,
            help="Continue conversation from where it left off",
        ):
            st.session_state.conversation_status = "running"
            st.rerun()

    with control_col4:
        # Reset button with confirmation
        if st.button(
            "🔄 Reset", type="secondary", help="Clear conversation and start fresh"
        ):
            st.session_state.conversation_status = "idle"
            st.session_state.conversation_history = []
            st.session_state.conversation_metrics = {
                "total_tokens": 0,
                "total_messages": 0,
                "avg_latency": 0,
                "agent_a_messages": 0,
                "agent_b_messages": 0,
            }
            st.session_state.conversation_turns_completed = 0
            st.session_state.conversation_max_turns = 0
            st.rerun()

    # --- Action Navigation Panel ---
    st.markdown("---")
    st.markdown("#### 🎯 Next Actions")

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        # Go to Analysis button
        analysis_enabled = len(st.session_state.conversation_history) > 0
        if st.button(
            "📊 Go to Analysis",
            disabled=not analysis_enabled,
            help="Navigate to Basic Analysis with current conversation data",
        ):
            # Save conversation data for analysis tab
            st.session_state.analysis_data = (
                st.session_state.conversation_history.copy()
            )
            st.info(
                "📊 Conversation data ready for analysis. Switch to 'Basic Analysis' tab."
            )

    with action_col2:
        # Extend conversation button
        extend_enabled = st.session_state.conversation_status in [
            "finished",
            "continued",
        ]
        if st.button(
            "▶️ Extend Conversation",
            disabled=not extend_enabled,
            help="Add more turns to current conversation",
        ):
            # Add 5 more turns
            st.session_state.conversation_max_turns += 5
            st.session_state.conversation_status = "continued"
            st.info(
                f"📈 Extended to {st.session_state.conversation_max_turns} turns. Click 'Continue' to proceed."
            )

    with action_col3:
        # Save progress button
        if st.button("💾 Save Progress", help="Save current conversation state"):
            # Create save data
            save_data = {
                "conversation_id": st.session_state.conversation_id,
                "timestamp": time.time(),
                "status": st.session_state.conversation_status,
                "history": st.session_state.conversation_history,
                "metrics": st.session_state.conversation_metrics,
                "config": config,
                "turns_completed": st.session_state.conversation_turns_completed,
                "max_turns": st.session_state.conversation_max_turns,
            }

            # Store in session state for download
            st.session_state.saved_conversation = save_data
            st.success(
                "✅ Conversation progress saved! Use export options below to download."
            )

    # --- Export options (always available when there's conversation data) ---
    if len(st.session_state.conversation_history) > 0:
        st.markdown("---")
        st.markdown("#### 📤 Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            if st.button("📄 Export as JSON"):
                json_data = json.dumps(st.session_state.conversation_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"conversation_{st.session_state.conversation_id[:8]}.json",
                    mime="application/json",
                )

        with export_col2:
            if st.button("📊 Export as CSV"):
                df = pd.DataFrame(st.session_state.conversation_history)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"conversation_{st.session_state.conversation_id[:8]}.csv",
                    mime="text/csv",
                )

        with export_col3:
            if st.button("📝 Export as Text"):
                text_data = ""
                for msg in st.session_state.conversation_history:
                    speaker = (
                        "Agent A"
                        if msg["speaker_model"] == config["model_a_slug"]
                        else "Agent B"
                    )
                    text_data += f"{speaker}: {msg['content']}\n\n"

                st.download_button(
                    label="Download Text",
                    data=text_data,
                    file_name=f"conversation_{st.session_state.conversation_id[:8]}.txt",
                    mime="text/plain",
                )

    # --- Real-time Statistics Sidebar ---
    with st.sidebar:
        st.markdown("### 📊 Live Statistics")

        metrics = st.session_state.conversation_metrics
        st.metric("Total Tokens", metrics["total_tokens"])
        st.metric("Avg Latency", f"{metrics['avg_latency']:.0f} ms")

        # Progress bar
        if st.session_state.conversation_max_turns > 0:
            current_messages = len(st.session_state.conversation_history)
            max_messages = st.session_state.conversation_max_turns * 2
            progress = min(current_messages / max_messages, 1.0)
            st.progress(progress)
            st.markdown(f"**Progress:** {current_messages}/{max_messages} messages")

    # --- Enhanced Conversation Display Area ---
    st.markdown("#### 💭 Conversation")

    # Auto-save indicator
    if (
        st.session_state.auto_save_enabled
        and len(st.session_state.conversation_history) > 0
    ):
        st.caption(
            "🔄 Auto-save enabled | Conversation ID: "
            + st.session_state.conversation_id[:8]
        )

    # Create container for messages with better state management
    conversation_container = st.container()

    # Display existing conversation history with enhanced formatting
    if st.session_state.conversation_history:
        with conversation_container:
            for i, message in enumerate(st.session_state.conversation_history):
                # Determine speaker with enhanced styling
                is_agent_a = message["speaker_model"] == config["model_a_slug"]
                speaker_name = "Agent A" if is_agent_a else "Agent B"
                speaker_color = "#1f77b4" if is_agent_a else "#ff7f0e"
                speaker_emoji = "🤖" if is_agent_a else "🎭"

                # Enhanced message display
                with st.chat_message(message["speaker_model"], avatar=None):
                    # Improved header with more info
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown(
                            f"<div style='display: flex; align-items: center; margin-bottom: 8px;'>"
                            f"<span style='font-size: 1.2em; margin-right: 8px;'>{speaker_emoji}</span>"
                            f"<span style='color: {speaker_color}; font-weight: bold; font-size: 1.1em;'>{speaker_name}</span>"
                            f"<span style='margin-left: 8px; font-size: 0.9em; color: #666;'>• Message {i + 1}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with col_header2:
                        st.markdown(
                            f"<div style='text-align: right; font-size: 0.8em; color: #666;'>"
                            f"{message['speaker_model']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Message content with better formatting
                    st.markdown("---")
                    st.write(message["content"])

                    # Enhanced statistics expander
                    with st.expander(f"📊 Message {i + 1} Statistics", expanded=False):
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("⏱️ Latency", f"{message['latency_ms']:.0f} ms")
                        with stat_col2:
                            st.metric("🔤 Output", message["output_tokens"])
                        with stat_col3:
                            st.metric("📥 Input", message["input_tokens"])
                        with stat_col4:
                            total_tokens = (
                                message["input_tokens"] + message["output_tokens"]
                            )
                            st.metric("📊 Total", total_tokens)

                        st.caption(f"🕐 Timestamp: {message['timestamp']}")
                        st.caption(
                            f"🎯 Finish Reason: {message.get('finish_reason', 'N/A')}"
                        )
    else:
        with conversation_container:
            st.info(
                "🚀 No conversation yet. Configure your agents in the previous tabs, then click 'Start' to begin."
            )

    # --- Enhanced conversation running logic with better state management ---
    if st.session_state.conversation_status == "running":
        # Initialize orchestrator only once
        if "orchestrator" not in st.session_state:
            from orchestrator import Orchestrator

            agent_a_config = {
                "model": config["model_a_slug"],
                "system_prompt": config["system_prompt_a"],
                "params": {
                    "temperature": config["temp_a"],
                    "max_tokens": config["max_tokens"],
                },
            }
            agent_b_config = {
                "model": config["model_b_slug"],
                "system_prompt": config["system_prompt_b"],
                "params": {
                    "temperature": config["temp_b"],
                    "max_tokens": config["max_tokens"],
                },
            }

            st.session_state.orchestrator = Orchestrator(
                agent_a_config=agent_a_config,
                agent_b_config=agent_b_config,
                scenario_name=f"Conversation {st.session_state.conversation_id[:8]}",
            )

            # Initialize conversation if starting fresh
            if st.session_state.conversation_turns_completed == 0:
                st.session_state.conversation_start_message = config["initial_message"]

        # Create placeholder for streaming updates
        message_placeholder = st.empty()

        # Check if we should continue conversation
        max_messages = st.session_state.conversation_max_turns * 2
        current_messages = len(st.session_state.conversation_history)

        if current_messages < max_messages:
            # Run next turn(s) without full rerun to preserve state
            try:
                # Get the last message as context for continuation
                last_message = st.session_state.conversation_start_message
                if st.session_state.conversation_history:
                    last_message = st.session_state.conversation_history[-1]["content"]

                # Generate next message(s)
                messages_generated = 0
                for log_entry in st.session_state.orchestrator.run_simulation(
                    1,  # Generate one turn at a time for better control
                    initial_message=last_message,
                ):
                    # Add to conversation history
                    st.session_state.conversation_history.append(log_entry)

                    # Update metrics
                    metrics = st.session_state.conversation_metrics
                    metrics["total_tokens"] += log_entry["output_tokens"]
                    metrics["total_messages"] += 1
                    metrics["avg_latency"] = (
                        metrics["avg_latency"] * (metrics["total_messages"] - 1)
                        + log_entry["latency_ms"]
                    ) / metrics["total_messages"]

                    if log_entry["speaker_model"] == config["model_a_slug"]:
                        metrics["agent_a_messages"] += 1
                    else:
                        metrics["agent_b_messages"] += 1

                    messages_generated += 1

                    # Auto-save every 2 messages
                    if (
                        st.session_state.auto_save_enabled
                        and messages_generated % 2 == 0
                    ):
                        st.session_state.last_auto_save = time.time()

                    # Break if we've reached the limit
                    if len(st.session_state.conversation_history) >= max_messages:
                        break

                # Update turn counter
                st.session_state.conversation_turns_completed = (
                    len(st.session_state.conversation_history) // 2
                )

                # Check if conversation is complete
                if len(st.session_state.conversation_history) >= max_messages:
                    st.session_state.conversation_status = "finished"

                    # Save logs
                    jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
                    st.session_state.orchestrator.save_logs(jsonl_path)

                    st.success("🎉 Conversation completed successfully!")
                    st.balloons()
                else:
                    # Continue running for next turn
                    st.session_state.conversation_status = "running"

                # Trigger UI update without losing state
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during conversation: {str(e)}")
                st.session_state.conversation_status = "idle"
                # Clean up orchestrator on error
                if "orchestrator" in st.session_state:
                    del st.session_state.orchestrator
                st.rerun()
        else:
            # Conversation reached limit
            st.session_state.conversation_status = "finished"
            st.rerun()

    # Clean up orchestrator when not running
    elif (
        st.session_state.conversation_status in ["idle", "paused", "finished"]
        and "orchestrator" in st.session_state
    ):
        del st.session_state.orchestrator

    # --- Export options when conversation is finished ---
    if (
        st.session_state.conversation_status == "finished"
        and st.session_state.conversation_history
    ):
        st.markdown("#### Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            if st.button("Export as JSON"):
                import json

                json_data = json.dumps(st.session_state.conversation_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="conversation.json",
                    mime="application/json",
                )

        with export_col2:
            if st.button("Export as CSV"):
                import pandas as pd

                df = pd.DataFrame(st.session_state.conversation_history)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="conversation.csv",
                    mime="text/csv",
                )

        with export_col3:
            if st.button("Export as Text"):
                text_data = ""
                for msg in st.session_state.conversation_history:
                    speaker = (
                        "Agent A"
                        if msg["speaker_model"] == config["model_a_slug"]
                        else "Agent B"
                    )
                    text_data += f"{speaker}: {msg['content']}\n\n"

                st.download_button(
                    label="Download Text",
                    data=text_data,
                    file_name="conversation.txt",
                    mime="text/plain",
                )