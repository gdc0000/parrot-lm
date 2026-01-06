import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
from streamlit_local_storage import LocalStorage
from simulation_config import NUM_TURNS
from analysis_utils import process_logs, process_custom_lexicon
from prompt_utils import construct_system_prompt

# --- Local Storage Setup ---
local_storage = LocalStorage()

st.set_page_config(page_title="🦜ParrotLM", layout="wide")

st.title("🦜ParrotLM")
st.markdown("A customizable Python framework for simulating conversations " \
"between two LLM chatbots with customizable personas, interaction settings, and analysis capabilities. ")

# --- Session State Initialization ---
if "last_generated_config" not in st.session_state:
    st.session_state["last_generated_config"] = {}

# Initialize logs in session state
if "all_logs" not in st.session_state:
    # Try to load from local storage
    saved_logs = local_storage.getItem("parrot_lm_logs")
    if saved_logs:
        try:
            st.session_state["all_logs"] = pd.DataFrame(saved_logs)
        except:
            st.session_state["all_logs"] = pd.DataFrame()
    else:
        st.session_state["all_logs"] = pd.DataFrame()

# --- Sidebar: Technical Configuration ---
st.sidebar.header("⚙️ Technical Settings")

# API Key
api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Leave empty to use .env")
if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

if st.sidebar.button("🗑️ Clear My Local Data", help="Wipes all conversation history from your browser storage."):
    local_storage.deleteAllItems()
    st.session_state["all_logs"] = pd.DataFrame()
    st.success("Local data cleared!")
    st.rerun()

num_turns = st.sidebar.slider("Turns per Chatbot", 1, 100, NUM_TURNS)
# iterations = st.sidebar.slider("Iterations", 1, 10, ITERATIONS) # Hidden for single run focus

st.sidebar.markdown("### Model Parameters")
temp_a = st.sidebar.slider("Chatbot A Temperature", 0.0, 2.0, 1.0, 0.1)
temp_b = st.sidebar.slider("Chatbot B Temperature", 0.0, 2.0, 1.0, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 500, 1000, 1000)

# --- Tabs: Main Structure ---
# New tab structure:
# 1. Chatbot Setup (includes interaction and simulation button)
# 2. Basic Analysis
# 3. Stylometric Analysis
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🎭 Chatbot Setup", "📊 Basic Analysis", "🧠 Stylometric Analysis"])

# --- Tab 1: Agent & Simulation Setup ---
with tab1:
    st.markdown("### 🎭 Configure the Encounter")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Chatbot A")
        model_a_slug = st.text_input("Model A Slug", "cognitivecomputations/dolphin-mistral-24b-venice-edition:free", key="model_a")
        persona_a = st.text_area("Persona", "A mysterious stranger at a jazz club", height=100)

    with col2:
        st.markdown("#### Chatbot B")
        model_b_slug = st.text_input("Model B Slug", "cognitivecomputations/dolphin-mistral-24b-venice-edition:free", key="model_b")
        persona_b = st.text_area("Persona", "A sharp-witted bartender", height=100)

    st.markdown("---")
    initial_message = st.text_input("The conversation starts with:", "Is this seat taken?")
    
    if st.button("🚀 Start Conversation", type="primary", use_container_width=True):
        st.write("### 🟢 Live Conversation")
        
        # Hidden System Prompt Generation
        system_prompt_a = construct_system_prompt(persona_a)
        system_prompt_b = construct_system_prompt(persona_b)

        # Containers for layout
        chat_container = st.container()
        
        # Initialize Orchestrator
        from orchestrator import Orchestrator
        
        chatbot_a_config = {
            "model": model_a_slug,
            "system_prompt": system_prompt_a,
            "user_persona_snapshot": persona_a,
            "params": {"temperature": temp_a, "max_tokens": max_tokens}
        }
        chatbot_b_config = {
            "model": model_b_slug,
            "system_prompt": system_prompt_b,
            "user_persona_snapshot": persona_b,
            "params": {"temperature": temp_b, "max_tokens": max_tokens}
        }
        
        orchestrator = Orchestrator(
            agent_a_config=chatbot_a_config,
            agent_b_config=chatbot_b_config,
            scenario_name=f"{persona_a[:15]} vs {persona_b[:15]}" 
        )
        
        # Run and Stream
        total_tokens = 0
        
        try:
            with st.spinner("Agents are conversing..."):
                for log_entry in orchestrator.run_simulation(num_turns, initial_message=initial_message): 
                    # Accumulate tokens
                    total_tokens += log_entry["output_tokens"]
                    
                    # Determine label and avatar
                    is_agent_a = log_entry["speaker_model"] == model_a_slug
                    speaker_label = persona_a if is_agent_a else persona_b
                    avatar = "🎭" if is_agent_a else "🍸"
                    
                    if len(speaker_label) > 50: speaker_label = speaker_label[:47] + "..."
                    
                    # Display Message
                    with chat_container:
                        # Use the speaker persona as the name for the chat message
                        with st.chat_message(name=speaker_label, avatar=avatar):
                            st.write(log_entry["content"])
                        
                        # Metadata OUTSIDE and BELOW the bubble for a cleaner look
                        st.markdown(
                            f"<div style='text-align: right; margin-top: -15px; margin-bottom: 10px;'>"
                            f"<span style='color: gray; font-size: 0.8rem;'>"
                            f"⏱️ {log_entry['latency_ms']:.0f}ms | 🔢 {log_entry['output_tokens']} tokens | 🤖 {log_entry['speaker_model']}"
                            f"</span></div>", 
                            unsafe_allow_html=True
                        )
                    
                    time.sleep(0.1)
            
            st.success(f"Simulation Complete. Total Tokens: {total_tokens}")
            
        except Exception as e:
            st.error(f"❌ Simulation Error: {str(e)}")
            st.info("💡 Tip: Try increasing 'Max Tokens' if the API is failing with low values.")
                
        # Save logs to session state and LocalStorage
        new_logs_df = pd.DataFrame(orchestrator.logs)
        if st.session_state["all_logs"].empty:
            st.session_state["all_logs"] = new_logs_df
        else:
            st.session_state["all_logs"] = pd.concat([st.session_state["all_logs"], new_logs_df], ignore_index=True)
        
        # Sync with LocalStorage
        local_storage.setItem("parrot_lm_logs", st.session_state["all_logs"].to_dict('records'))
        st.success("Simulation Finished & Persisted Locally!")

# --- Tab 2: Basic Analysis ---
with tab2:
    st.header("Basic Data Analysis")
    if st.button("Refresh Data", key="refresh_basic"):
        st.rerun()
        
    if not st.session_state["all_logs"].empty:
        df = st.session_state["all_logs"]
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
    st.header("🧠 Stylometric Analysis (NLTK)")
    
    # --- Custom Lexicon Input ---
    st.subheader("🏷️ Custom Lexicon Configuration")
    st.markdown("Define specific word categories to track during the conversation.")
    
    # Initialize session state for lexicon if not exists
    if "custom_lexicon" not in st.session_state:
        st.session_state["custom_lexicon"] = [
            {"category": "Positive", "words": "love, great, happy, good"},
            {"category": "Negative", "words": "hate, bad, sad, terrible"},
            {"category": "Hesitation", "words": "um, uh, er, maybe, perhaps"}
        ]

    # UI for editing Lexicon
    for i, item in enumerate(st.session_state["custom_lexicon"]):
        lex_col1, lex_col2, lex_col3 = st.columns([1, 2, 0.2])
        with lex_col1:
            item["category"] = st.text_input(f"Category Name {i}", item["category"], key=f"lex_cat_{i}", placeholder="Category", label_visibility="collapsed")
        with lex_col2:
            item["words"] = st.text_input(f"Words {i}", item["words"], key=f"lex_words_{i}", placeholder="words, separated, by, commas", label_visibility="collapsed")
        with lex_col3:
            if st.button("🗑️", key=f"lex_del_{i}", help="Remove category"):
                st.session_state["custom_lexicon"].pop(i)
                st.rerun()
    
    if st.button("➕ Add New Category"):
        st.session_state["custom_lexicon"].append({"category": "", "words": ""})
        st.rerun()

    # Convert to the dictionary format expected by processing functions
    category_dict = {
        item["category"].strip(): [w.strip() for w in item["words"].split(',') if w.strip()]
        for item in st.session_state["custom_lexicon"]
        if item["category"].strip()
    }
    
    st.markdown("---")
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not st.session_state["all_logs"].empty:
            with st.spinner("Processing text..."):
                df = st.session_state["all_logs"]
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
