import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import time
from simulation_config import MODELS, SCENARIOS, NUM_TURNS, ITERATIONS, DATA_DIR
from run_experiment import run_experiments

st.set_page_config(page_title="Multi-Agent Simulation v2", layout="wide")

st.title("🤖 Multi-Agent Social Dynamics Simulator v2")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# API Key
api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Leave empty to use .env")
if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

# --- Advanced Settings Expander ---
with st.sidebar.expander("📝 Simulation Settings", expanded=True):
    num_turns = st.slider("Turns per Agent", 1, 30, NUM_TURNS)
    iterations = st.slider("Iterations", 1, 10, ITERATIONS)
    
    st.markdown("### Model Parameters")
    temp_a = st.slider("Agent A Temperature", 0.0, 2.0, 1.0, 0.1)
    temp_b = st.slider("Agent B Temperature", 0.0, 2.0, 1.0, 0.1)
    
    max_tokens = st.number_input("Max Tokens", 50, 4096, 1000)

# --- Agent Configuration ---
st.sidebar.subheader("Agent Setup")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    st.markdown("**Agent A**")
    model_a_key = st.selectbox("Model A", list(MODELS.keys()), index=0, key="model_a")
    
with col_b:
    st.markdown("**Agent B**")
    model_b_key = st.selectbox("Model B", list(MODELS.keys()), index=1, key="model_b")

# --- Scenario & System Prompts ---
st.sidebar.subheader("Scenario & Prompts")
scenario_mode = st.sidebar.radio("Scenario Mode", ["Preset", "Custom"], horizontal=True)

if scenario_mode == "Preset":
    selected_scenario = st.sidebar.selectbox("Choose Scenario", list(SCENARIOS.keys()))
    base_prompt = SCENARIOS[selected_scenario]
else:
    base_prompt = "You are an AI assistant."

# Allow granular editing of system prompts
with st.sidebar.expander("✏️ Edit System Prompts", expanded=False):
    prompt_a = st.text_area("Agent A System Prompt", base_prompt, height=150)
    prompt_b = st.text_area("Agent B System Prompt", base_prompt, height=150)

# --- Main Content ---
tab1, tab2 = st.tabs(["🚀 Live Simulation", "📊 Analysis"])

with tab1:
    if st.button("Start Simulation", type="primary"):
        # Prepare Configs
        models = {model_a_key: MODELS[model_a_key], model_b_key: MODELS[model_b_key]}
        # We construct a single-entry scenario dict for this run to keep run_experiment logic simple
        # or we pass specific prompts. run_experiment iterates scenarios.
        # Let's create a temporary scenario dict.
        
        current_scenarios = {"Current Run": "Custom"} # Placeholder, prompts are passed via config override? 
        # Actually run_experiment iterates scenarios and uses the value as prompt.
        # But we want different prompts for A and B.
        # Our updated run_experiment takes agent_a_params but not agent_a_prompt override directly 
        # unless we change how run_experiment works or we rely on the loop.
        
        # Hack: We'll pass a single scenario with a placeholder, but we need to inject the specific prompts.
        # The current run_experiment uses the scenario value as the prompt for BOTH.
        # To support different prompts, we should probably update run_experiment or just use the params to carry the prompt?
        # Wait, run_experiment sets `system_prompt` in `agent_config`.
        # If we want different prompts, we need to modify run_experiment to accept them or 
        # we can just run the Orchestrator directly here for maximum control?
        # NO, better to use run_experiment generator.
        
        # Let's use a trick: We pass a single scenario. 
        # But wait, run_experiment forces the SAME prompt for both agents (from the scenario dict).
        # If the user edited them to be different, we have a problem with the current run_experiment logic.
        
        # FIX: We will instantiate Orchestrator directly here for the "Live Run" to give full control.
        # run_experiment is good for batch, but GUI "Live Run" is usually a single specific setup.
        
        st.write("### 🟢 Live Conversation")
        
        # Containers for layout
        chat_container = st.container()
        metric_col1, metric_col2 = st.columns(2)
        
        # Initialize Orchestrator directly for custom control
        from orchestrator import Orchestrator
        
        agent_a_config = {
            "model": MODELS[model_a_key],
            "system_prompt": prompt_a,
            "params": {"temperature": temp_a, "max_tokens": max_tokens}
        }
        agent_b_config = {
            "model": MODELS[model_b_key],
            "system_prompt": prompt_b,
            "params": {"temperature": temp_b, "max_tokens": max_tokens}
        }
        
        orchestrator = Orchestrator(
            agent_a_config=agent_a_config,
            agent_b_config=agent_b_config,
            scenario_name=scenario_mode if scenario_mode == "Custom" else selected_scenario
        )
        
        # Run and Stream
        total_tokens = 0
        
        with st.spinner("Agents are conversing..."):
            for log_entry in orchestrator.run_simulation(num_turns):
                # Update Metrics
                total_tokens += log_entry["output_tokens"]
                metric_col1.metric("Last Latency", f"{log_entry['latency_ms']:.0f} ms")
                metric_col2.metric("Total Tokens", total_tokens)
                
                # Display Message
                with chat_container:
                    with st.chat_message(log_entry["speaker_model"], avatar="🤖" if "A" in log_entry["speaker_model"] else "👾"):
                        st.markdown(f"**{log_entry['speaker_model']}**")
                        st.write(log_entry["content"])
                
                # Small delay for visual pacing (optional)
                time.sleep(0.1)
                
        # Save logs manually since we bypassed run_experiment
        jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
        orchestrator.save_logs(jsonl_path)
        st.success("Simulation Finished & Saved!")

with tab2:
    st.header("Data Analysis")
    csv_path = os.path.join(DATA_DIR, "experiment_log.csv")
    
    if st.button("Refresh Data"):
        st.rerun()
        
    # We need to convert JSONL to CSV if it's not up to date, or just read JSONL directly
    jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
    if os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
        st.dataframe(df)
        
        # Metrics
        st.subheader("Metrics Overview")
        col1, col2 = st.columns(2)
        with col1:
            avg_latency = df.groupby("speaker_model")["latency_ms"].mean().reset_index()
            fig_latency = px.bar(avg_latency, x="speaker_model", y="latency_ms", title="Average Latency (ms)")
            st.plotly_chart(fig_latency, use_container_width=True)
            
        with col2:
            avg_tokens = df.groupby("speaker_model")["output_tokens"].mean().reset_index()
            fig_tokens = px.bar(avg_tokens, x="speaker_model", y="output_tokens", title="Average Output Tokens")
            st.plotly_chart(fig_tokens, use_container_width=True)
            
    else:
        st.info("No data found. Run an experiment first.")
