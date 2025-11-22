import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import time
from simulation_config import SCENARIOS, NUM_TURNS, ITERATIONS, DATA_DIR
from run_experiment import run_experiments
from analysis_utils import process_logs, process_custom_lexicon

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
    model_a_slug = st.text_input("Model A Slug", "x-ai/grok-beta", key="model_a_custom")
    
with col_b:
    st.markdown("**Agent B**")
    model_b_slug = st.text_input("Model B Slug", "meta-llama/llama-3-70b-instruct", key="model_b_custom")

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
tab1, tab2, tab3 = st.tabs(["🚀 Live Simulation", "📊 Basic Analysis", "🧠 Stylometric Analysis"])

with tab1:
    if st.button("Start Simulation", type="primary"):
        st.write("### 🟢 Live Conversation")
        
        # Containers for layout
        chat_container = st.container()
        metric_col1, metric_col2 = st.columns(2)
        
        # Initialize Orchestrator directly for custom control
        from orchestrator import Orchestrator
        
        agent_a_config = {
            "model": model_a_slug,
            "system_prompt": prompt_a,
            "params": {"temperature": temp_a, "max_tokens": max_tokens}
        }
        agent_b_config = {
            "model": model_b_slug,
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
    st.header("Basic Data Analysis")
    
    if st.button("Refresh Data", key="refresh_basic"):
        st.rerun()
        
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

with tab3:
    st.header("🧠 Stylometric Analysis (spaCy)")
    st.markdown("Analyze linguistic patterns: Part-of-Speech usage, sentence length, and vocabulary richness.")
    
    # --- Custom Lexicon Input ---
    st.subheader("Custom Word Frequency (LIWC-style)")
    st.markdown("Define categories and words to count. Format: `Category: word1, word2, word3`")
    
    default_lexicon = """Positive: love, great, happy, good
Negative: hate, bad, sad, terrible
Hesitation: um, uh, er, maybe, perhaps"""
    
    lexicon_input = st.text_area("Define Custom Lexicon", default_lexicon, height=100)
    
    # Parse Lexicon
    category_dict = {}
    if lexicon_input:
        for line in lexicon_input.split('\n'):
            if ':' in line:
                cat, words = line.split(':', 1)
                category_dict[cat.strip()] = [w.strip() for w in words.split(',')]
    
    if st.button("Run Analysis", type="primary"):
        jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
        if os.path.exists(jsonl_path):
            with st.spinner("Processing text..."):
                df = pd.read_json(jsonl_path, lines=True)
                
                # 1. Standard SpaCy Analysis
                analyzed_df = process_logs(df)
                
                # 2. Custom Lexicon Analysis
                if category_dict:
                    analyzed_df = process_custom_lexicon(analyzed_df, category_dict)
                
                st.success("Analysis Complete!")
                st.dataframe(analyzed_df)
                
                # --- Visualizations ---
                st.subheader("Linguistic Patterns")
                
                # POS Ratios
                pos_cols = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"]
                avg_pos = analyzed_df.groupby("speaker_model")[pos_cols].mean().reset_index()
                
                # Melt for grouped bar chart
                melted_pos = avg_pos.melt(id_vars="speaker_model", var_name="POS Type", value_name="Ratio")
                
                fig_pos = px.bar(melted_pos, x="speaker_model", y="Ratio", color="POS Type", barmode="group",
                                 title="Part-of-Speech Distribution by Model")
                st.plotly_chart(fig_pos, use_container_width=True)
                
                # Custom Lexicon Visualization
                if category_dict:
                    st.subheader("Custom Category Frequencies")
                    lex_cols = list(category_dict.keys())
                    # Sum counts per model (total occurrences) or mean (avg per message)
                    # Let's show Average occurrences per message
                    avg_lex = analyzed_df.groupby("speaker_model")[lex_cols].mean().reset_index()
                    
                    melted_lex = avg_lex.melt(id_vars="speaker_model", var_name="Category", value_name="Avg Count")
                    
                    fig_lex = px.bar(melted_lex, x="speaker_model", y="Avg Count", color="Category", barmode="group",
                                     title="Custom Word Category Usage (Avg per Message)")
                    st.plotly_chart(fig_lex, use_container_width=True)
                
                # Sentence Length
                col1, col2 = st.columns(2)
                with col1:
                    avg_len = analyzed_df.groupby("speaker_model")["avg_sentence_length"].mean().reset_index()
                    fig_len = px.bar(avg_len, x="speaker_model", y="avg_sentence_length", 
                                     title="Average Sentence Length (tokens)")
                    st.plotly_chart(fig_len, use_container_width=True)
                    
                with col2:
                    # Vocabulary Richness (Type-Token Ratio approximation or similar)
                    # Here we just show pronoun usage as a proxy for 'personal' style
                    avg_pron = analyzed_df.groupby("speaker_model")["pron_ratio"].mean().reset_index()
                    fig_pron = px.bar(avg_pron, x="speaker_model", y="pron_ratio", 
                                      title="Pronoun Usage Ratio (Personal Style)")
                    st.plotly_chart(fig_pron, use_container_width=True)
                    
        else:
            st.warning("No data found. Please run a simulation first.")
