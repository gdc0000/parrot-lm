import streamlit as st

def render_prompts(
    model_a_slug,
    model_b_slug,
    temp_a,
    temp_b,
    max_tokens,
    num_turns,
    construct_system_prompt,
):
    st.markdown("### 🌍 System Prompts & Start Message")

    with st.expander("⚡ Quick Settings - Generate Default Prompts"):
        interaction_setting = st.selectbox(
            "Setting / Tone", ["Professional", "Intimate", "Casual", "Debate"]
        )
        if st.button("Generate Default Prompts", help="Create prompts based on selected setting"):
            default_prompt_a = construct_system_prompt(interaction_setting, "Agent A")
            default_prompt_b = construct_system_prompt(interaction_setting, "Agent B")
            st.session_state.system_prompt_a = default_prompt_a
            st.session_state.system_prompt_b = default_prompt_b
            st.rerun()

    st.markdown("### ✏️ Edit System Prompts")
    prompt_col1, prompt_col2 = st.columns(2)

    with prompt_col1:
        if "system_prompt_a" not in st.session_state:
            default_prompt_a = construct_system_prompt("Professional", "Agent A")
        else:
            default_prompt_a = construct_system_prompt("Professional", "Agent A")
        current_prompt_a = st.session_state.get("system_prompt_a", default_prompt_a)
        system_prompt_a = st.text_area(
            "Agent A System Prompt",
            current_prompt_a,
            height=200,
            key="system_prompt_a_edit",
            help="Edit the system prompt that defines Agent A's behavior and persona",
        )
        if system_prompt_a != current_prompt_a:
            st.session_state.system_prompt_a = system_prompt_a
        if st.button("🔄 Reset to Default", key="reset_a", help="Reset Agent A prompt to default"):
            st.session_state.system_prompt_a = default_prompt_a
            st.rerun()

    with prompt_col2:
        if "system_prompt_b" not in st.session_state:
            default_prompt_b = construct_system_prompt("Professional", "Agent B")
        else:
            default_prompt_b = construct_system_prompt("Professional", "Agent B")
        current_prompt_b = st.session_state.get("system_prompt_b", default_prompt_b)
        system_prompt_b = st.text_area(
            "Agent B System Prompt",
            current_prompt_b,
            height=200,
            key="system_prompt_b_edit",
            help="Edit the system prompt that defines Agent B's behavior and persona",
        )
        if system_prompt_b != current_prompt_b:
            st.session_state.system_prompt_b = system_prompt_b
        if st.button("🔄 Reset to Default", key="reset_b", help="Reset Agent B prompt to default"):
            st.session_state.system_prompt_b = default_prompt_b
            st.rerun()

    st.markdown("### Start Message")
    starter_mode = st.radio("Message Type", ["Preset", "Custom"], horizontal=True)
    PRESET_STARTERS = [
        "Hello.",
        "Greetings.",
        "We need to talk.",
        "What is your opinion on current state of affairs?",
        "I have a confession to make.",
    ]
    if starter_mode == "Preset":
        initial_message = st.selectbox("Initial Message", PRESET_STARTERS, label_visibility="collapsed")
    else:
        initial_message = st.text_input("Custom Initial Message", "Hello.", label_visibility="collapsed")

    # Store configuration in session state
    current_prompt_a = st.session_state.get("system_prompt_a", construct_system_prompt("Professional", "Agent A"))
    current_prompt_b = st.session_state.get("system_prompt_b", construct_system_prompt("Professional", "Agent B"))

    st.session_state.conversation_config = {
        "model_a_slug": model_a_slug,
        "model_b_slug": model_b_slug,
        "system_prompt_a": current_prompt_a,
        "system_prompt_b": current_prompt_b,
        "persona_a": "Agent A",
        "persona_b": "Agent B",
        "temp_a": temp_a,
        "temp_b": temp_b,
        "max_tokens": max_tokens,
        "num_turns": num_turns,
        "initial_message": initial_message,
    }

    st.info("💡 Configuration ready! Navigate to 'Start Conversation' tab to begin.")
