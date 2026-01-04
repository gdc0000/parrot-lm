import streamlit as st

def render_agent_setup():
    st.markdown("### Configure Agents")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Agent A")
        model_a_slug = st.text_input("Model A Slug", "x-ai/grok-beta", key="model_a")
    with col2:
        st.markdown("#### Agent B")
        model_b_slug = st.text_input(
            "Model B Slug", "meta-llama/llama-3-70b-instruct", key="model_b"
        )
    return model_a_slug, model_b_slug
