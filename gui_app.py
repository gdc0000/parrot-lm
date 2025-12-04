import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import time
from simulation_config import NUM_TURNS, ITERATIONS, DATA_DIR
from run_experiment import run_experiments
from analysis_utils import process_logs, process_custom_lexicon
from prompt_utils import construct_system_prompt

st.set_page_config(page_title="Multi-Agent Simulation v3", layout="wide")

st.title("🤖 Multi-Agent Social Dynamics Simulator v3")

# --- Sidebar: Technical Configuration ---
st.sidebar.header("⚙️ Technical Settings")

# API Key
api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Leave empty to use .env")
if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

num_turns = st.sidebar.slider("Turns per Agent", 1, 30, NUM_TURNS)
# iterations = st.sidebar.slider("Iterations", 1, 10, ITERATIONS) # Hidden for single run focus

st.sidebar.markdown("### Model Parameters")
temp_a = st.sidebar.slider("Agent A Temperature", 0.0, 2.0, 1.0, 0.1)
temp_b = st.sidebar.slider("Agent B Temperature", 0.0, 2.0, 1.0, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens", 50, 4096, 1000)

# --- Tabs: Main Structure ---
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🎭 Agent & Interaction Setup", "📊 Basic Analysis", "🧠 Stylometric Analysis"])

# --- Tab 1: Setup & Simulation ---
with tab1:
    st.markdown("### Configure Agents & Context")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Agent A")
        model_a_slug = st.text_input("Model A Slug", "x-ai/grok-beta", key="model_a")
        persona_a = st.text_area("Persona / Character", "Julius Caesar", height=100, 
                                 help="Describe the character, style, or historical figure.")

    with col2:
        st.markdown("#### Agent B")
        model_b_slug = st.text_input("Model B Slug", "meta-llama/llama-3-70b-instruct", key="model_b")
        persona_b = st.text_area("Persona / Character", "A modern data scientist", height=100,
                                 help="Describe the character, style, or historical figure.")

    st.markdown("#### 🌍 Interaction Context")
    context_col1, context_col2 = st.columns(2)
    with context_col1:
        interaction_setting = st.selectbox("Setting / Tone", ["Professional", "Intimate", "Casual", "Debate"])

    with context_col2:
        starter_mode = st.radio("Starter Mode", ["Preset", "Custom"], horizontal=True)
        PRESET_STARTERS = [
            "Hello.",
            "Greetings.",
            "We need to talk.",
            "What is your opinion on the current state of affairs?",
            "I have a confession to make."
        ]
        if starter_mode == "Preset":
            initial_message = st.selectbox("Initial Message", PRESET_STARTERS, label_visibility="collapsed")
        else:
            initial_message = st.text_input("Custom Initial Message", "Hello.", label_visibility="collapsed")

    st.markdown("---")
    
    if st.button("Start Simulation", type="primary", use_container_width=True):
        st.write("### 🟢 Live Conversation")
        
        # Construct System Prompts Dynamically
        system_prompt_a = construct_system_prompt(interaction_setting, persona_a)
        system_prompt_b = construct_system_prompt(interaction_setting, persona_b)
        
        # Display the generated prompts for transparency
        with st.expander("View Generated System Prompts"):
            st.markdown(f"**Agent A Prompt:**\n{system_prompt_a}")
            st.markdown(f"**Agent B Prompt:**\n{system_prompt_b}")
        
        # Containers for layout
        chat_container = st.container()
        metric_col1, metric_col2 = st.columns(2)
        
        # Initialize Orchestrator
        from orchestrator import Orchestrator
        
        agent_a_config = {
            "model": model_a_slug,
            "system_prompt": system_prompt_a,
            "params": {"temperature": temp_a, "max_tokens": max_tokens}
        }
        agent_b_config = {
            "model": model_b_slug,
            "system_prompt": system_prompt_b,
            "params": {"temperature": temp_b, "max_tokens": max_tokens}
        }
        
        orchestrator = Orchestrator(
            agent_a_config=agent_a_config,
            agent_b_config=agent_b_config,
            scenario_name=f"{interaction_setting} - {persona_a[:20]} vs {persona_b[:20]}"
        )
        
        # Run and Stream
        total_tokens = 0
        
        with st.spinner("Agents are conversing..."):
            for log_entry in orchestrator.run_simulation(num_turns, initial_message=initial_message):
                # Update Metrics
                total_tokens += log_entry["output_tokens"]
                metric_col1.metric("Last Latency", f"{log_entry['latency_ms']:.0f} ms")
                metric_col2.metric("Total Tokens", total_tokens)
                
                # Display Message
                with chat_container:
                    avatar = None  # Set avatar to None to avoid the image loading error
                    print(f'DEBUG: avatar="{avatar}", speaker_model="{log_entry["speaker_model"]}", model_a_slug="{model_a_slug}"')
                    with st.chat_message(log_entry["speaker_model"], avatar=avatar):
                        speaker_label = persona_a if log_entry["speaker_model"] == model_a_slug else persona_b
                        if len(speaker_label) > 20: speaker_label = speaker_label[:20] + "..."
                        st.markdown(f"**{speaker_label}** ({log_entry['speaker_model']})")
                        st.write(log_entry["content"])
                
                time.sleep(0.1)
                
        # Save logs
        jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
        orchestrator.save_logs(jsonl_path)
        st.success("Simulation Finished & Saved!")

# --- Tab 2: Basic Analysis ---
with tab2:
    st.header("Basic Data Analysis")
    if st.button("Refresh Data", key="refresh_basic"):
        st.rerun()
        
    jsonl_path = os.path.join(DATA_DIR, "experiment_log.jsonl")
    if os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
        st.dataframe(df)
        
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
        st.info("No data found.")

# --- Tab 3: Stylometric Analysis ---
with tab3:
    st.header("🧠 Stylometric Analysis (spaCy)")
    
    # --- Custom Lexicon Input ---
    st.subheader("Custom Word Frequency (LIWC-style)")
    st.markdown("Define categories and words to count. Format: `Category: word1, word2`")
    
    default_lexicon = """Positive: love, great, happy, good
Negative: hate, bad, sad, terrible
Hesitation: um, uh, er, maybe, perhaps"""
    
    lexicon_input = st.text_area("Define Custom Lexicon", default_lexicon, height=100)
    
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
                analyzed_df = process_logs(df)
                if category_dict:
                    analyzed_df = process_custom_lexicon(analyzed_df, category_dict)
                
                st.success("Analysis Complete!")
                st.dataframe(analyzed_df)
                
                # Export Button
                csv = analyzed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analysis as CSV",
                    data=csv,
                    file_name='stylometric_analysis.csv',
                    mime='text/csv',
                )
                
                st.subheader("Linguistic Patterns")
                pos_cols = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio"]
                avg_pos = analyzed_df.groupby("speaker_model")[pos_cols].mean().reset_index()
                melted_pos = avg_pos.melt(id_vars="speaker_model", var_name="POS Type", value_name="Ratio")
                
                # Inverted: Group by POS Type, Color by Model
                fig_pos = px.bar(melted_pos, x="POS Type", y="Ratio", color="speaker_model", barmode="group", 
                                 title="POS Distribution (Grouped by Category)")
                st.plotly_chart(fig_pos, use_container_width=True)
                
                if category_dict:
                    st.subheader("Custom Category Frequencies")
                    lex_cols = list(category_dict.keys())
                    avg_lex = analyzed_df.groupby("speaker_model")[lex_cols].mean().reset_index()
                    melted_lex = avg_lex.melt(id_vars="speaker_model", var_name="Category", value_name="Avg Count")
                    
                    # Inverted: Group by Category, Color by Model
                    fig_lex = px.bar(melted_lex, x="Category", y="Avg Count", color="speaker_model", barmode="group", 
                                     title="Custom Word Usage (Grouped by Category)")
                    st.plotly_chart(fig_lex, use_container_width=True)
        else:
            st.warning("No data found.")
