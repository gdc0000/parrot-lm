import os
import streamlit as st

def render_sidebar(default_turns):
    api_key = st.sidebar.text_input(
        "OpenRouter API Key", type="password", help="Leave empty to use .env"
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    num_turns = st.sidebar.slider("Turns per Agent", 1, 30, default_turns)
    st.sidebar.markdown("### Model Parameters")
    temp_a = st.sidebar.slider("Agent A Temperature", 0.0, 2.0, 1.0, 0.1)
    temp_b = st.sidebar.slider("Agent B Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.sidebar.number_input("Max Tokens", 50, 4096, 1000)

    return {
        "api_key": api_key,
        "num_turns": num_turns,
        "temp_a": temp_a,
        "temp_b": temp_b,
        "max_tokens": max_tokens,
    }
